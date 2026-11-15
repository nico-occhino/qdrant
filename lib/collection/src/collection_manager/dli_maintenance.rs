//! Insert-driven private build research pipeline. Outputs are never published.
use super::dli_schedule::{CompactionPlan, HierarchyConfig};
use common::flags::FeatureFlags;
use parking_lot::Mutex;
use segment::entry::{ReadSegmentEntry, StorageSegmentEntry};
use segment::index::lmi_index::LmiConfig;
use segment::segment::Segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::types::{HnswGlobalConfig, Indexes};
use shard::locked_segment::LockedSegment;
use shard::segment_holder::locked::LockedSegmentHolder;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use uuid::Uuid;

/// Native optimizer adapter. The native optimizer owns proxy installation,
/// propagation and publication; constructing this adapter does not claim that
/// a previously observed plan is still current.
pub struct DliOptimizationStrategy {
    pub staging: PathBuf,
    pub segments_path: PathBuf,
    pub segment_config: segment::types::SegmentConfig,
    pub model: LmiConfig,
    /// Physical generations expected by the already selected job.
    pub expected_sources: std::collections::HashSet<Uuid>,
    pub expected_topology: Option<(HierarchyConfig, HashMap<Uuid, usize>, CompactionPlan)>,
}

impl shard::optimize::OptimizationStrategy for DliOptimizationStrategy {
    fn validate_topology(
        &self,
        holder: &shard::segment_holder::SegmentHolder,
    ) -> segment::common::operation_error::OperationResult<()> {
        let Some((hierarchy, roles, expected)) = &self.expected_topology else {
            return Ok(());
        };
        let fail = || {
            segment::common::operation_error::OperationError::service_error(
                "stale DLI hierarchy topology or plan",
            )
        };
        let mut seen = std::collections::HashSet::new();
        let mut counts = Vec::new();
        let mut buffer = 0usize;
        for (_, member) in holder.iter() {
            let LockedSegment::Original(segment) = member else {
                return Err(fail());
            };
            let guard = segment.read();
            let Some(&level) = roles.get(&guard.uuid) else {
                return Err(fail());
            };
            if !seen.insert(guard.uuid) || guard.has_deferred_points() {
                return Err(fail());
            }
            let count = guard.available_point_count_without_deferred();
            if level == 0 {
                buffer = buffer.checked_add(count).ok_or_else(fail)?;
            } else {
                counts.resize(counts.len().max(level), 0usize);
                counts[level - 1] = counts[level - 1].checked_add(count).ok_or_else(fail)?;
            }
        }
        if seen.len() != roles.len() {
            return Err(fail());
        }
        let actual = hierarchy
            .plan(buffer, &counts)
            .map_err(|_| fail())?
            .ok_or_else(fail)?;
        if actual.target_level != expected.target_level
            || actual.source_levels != expected.source_levels
        {
            return Err(fail());
        }
        let selected: std::collections::HashSet<_> = roles
            .iter()
            .filter_map(|(uuid, level)| {
                (*level == 0
                    || *level == actual.target_level
                    || actual.source_levels.contains(level))
                .then_some(*uuid)
            })
            .collect();
        if selected != self.expected_sources {
            return Err(fail());
        }
        Ok(())
    }
    fn validate_sources(
        &self,
        inputs: &[LockedSegment],
    ) -> segment::common::operation_error::OperationResult<()> {
        let actual: std::collections::HashSet<_> = inputs
            .iter()
            .map(|input| input.get().read().segment_uuid())
            .collect();
        if actual.len() != inputs.len() || actual != self.expected_sources {
            return Err(
                segment::common::operation_error::OperationError::service_error(
                    "stale DLI physical source generations",
                ),
            );
        }
        Ok(())
    }
    fn create_segment_builder(
        &self,
        inputs: &[LockedSegment],
    ) -> segment::common::operation_error::OperationResult<SegmentBuilder> {
        let mut config = self.segment_config.clone();
        let actual: std::collections::HashSet<_> = inputs
            .iter()
            .map(|input| input.get().read().segment_uuid())
            .collect();
        if actual.len() != inputs.len() || actual != self.expected_sources {
            return Err(
                segment::common::operation_error::OperationError::service_error(
                    "stale DLI physical source generations",
                ),
            );
        }
        if config.vector_data.len() != 1 || !config.sparse_vector_data.is_empty() {
            return Err(
                segment::common::operation_error::OperationError::service_error(
                    "unsupported DLI vector configuration",
                ),
            );
        }
        let vector = config
            .vector_data
            .get_mut(segment::data_types::vectors::DEFAULT_VECTOR_NAME)
            .ok_or_else(|| {
                segment::common::operation_error::OperationError::service_error(
                    "missing DLI vector configuration",
                )
            })?;
        if vector.multivector_config.is_some() {
            return Err(
                segment::common::operation_error::OperationError::service_error(
                    "unsupported DLI vector configuration",
                ),
            );
        }
        for input in inputs {
            let guard = input.get().read();
            let source = guard.config();
            let Some(source_vector) = source
                .vector_data
                .get(segment::data_types::vectors::DEFAULT_VECTOR_NAME)
            else {
                return Err(
                    segment::common::operation_error::OperationError::service_error(
                        "missing source vector",
                    ),
                );
            };
            if source.vector_data.len() != 1
                || !source.sparse_vector_data.is_empty()
                || source_vector.size != vector.size
                || source_vector.distance != vector.distance
                || source_vector.datatype != vector.datatype
                || source_vector.multivector_config.is_some()
            {
                return Err(
                    segment::common::operation_error::OperationError::service_error(
                        "incompatible DLI source vector configuration",
                    ),
                );
            }
        }
        vector.index = Indexes::LmiTrained(self.model.clone());
        SegmentBuilder::new(
            &self.staging,
            &config,
            &HnswGlobalConfig::default(),
            FeatureFlags::default(),
        )
    }

    fn create_temp_segment(
        &self,
    ) -> segment::common::operation_error::OperationResult<(
        LockedSegment,
        shard::segment_manifest::NewSegmentToken,
    )> {
        let mut config = self.segment_config.clone();
        if config.vector_data.len() != 1 || !config.sparse_vector_data.is_empty() {
            return Err(
                segment::common::operation_error::OperationError::service_error(
                    "unsupported DLI COW vector configuration",
                ),
            );
        }
        for vector in config.vector_data.values_mut() {
            if vector.multivector_config.is_some() || vector.quantization_config.is_some() {
                return Err(
                    segment::common::operation_error::OperationError::service_error(
                        "unsupported DLI COW vector options",
                    ),
                );
            }
            vector.index = Indexes::Plain {};
            vector.storage_type = segment::types::VectorStorageType::InRamChunkedMmap;
        }
        let (segment, token) =
            segment::segment_constructor::build_segment(&self.segments_path, &config, None, false)?;
        if !segment.is_appendable() {
            return Err(
                segment::common::operation_error::OperationError::service_error(
                    "DLI COW segment is not appendable",
                ),
            );
        }
        Ok((LockedSegment::from(segment), token))
    }

    fn live_vector_names(&self) -> Option<std::collections::HashSet<String>> {
        Some(self.segment_config.vector_data.keys().cloned().collect())
    }
}

#[derive(Debug)]
pub struct PrivateBuild {
    pub plan: CompactionPlan,
    pub sources: Vec<Uuid>,
    pub output: Result<Segment, String>,
}
#[derive(Debug)]
pub struct DliMaintenance {
    pub hierarchy: HierarchyConfig,
    pub members: HashMap<Uuid, usize>,
    pub model: LmiConfig,
    pub path: PathBuf,
    running: AtomicBool,
    last: Mutex<Vec<(Uuid, u64, usize)>>,
    /// Experimental native publication is explicitly enabled by a test only.
    pub native_enabled: AtomicBool,
    pub published: AtomicUsize,
    pub published_uuids: Mutex<Vec<Uuid>>,
    pub optimizer_errors: Mutex<Vec<String>>,
    pub scheduled: AtomicUsize,
    pub builds: Mutex<Vec<PrivateBuild>>,
    pub completed: tokio::sync::Notify,
}
impl DliMaintenance {
    pub fn new(
        hierarchy: HierarchyConfig,
        members: HashMap<Uuid, usize>,
        model: LmiConfig,
        path: PathBuf,
    ) -> Arc<Self> {
        Arc::new(Self {
            hierarchy,
            members,
            model,
            path,
            running: AtomicBool::new(false),
            last: Mutex::new(Vec::new()),
            native_enabled: AtomicBool::new(false),
            published: AtomicUsize::new(0),
            published_uuids: Mutex::new(Vec::new()),
            optimizer_errors: Mutex::new(Vec::new()),
            scheduled: AtomicUsize::new(0),
            builds: Mutex::new(Vec::new()),
            completed: Default::default(),
        })
    }
    /// Called after native update locks have been released. One private job per
    /// observed source version set; no authoritative point state is stored here.
    pub fn observe(self: &Arc<Self>, holder: &LockedSegmentHolder) {
        if self
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        let captured: Vec<_> = holder
            .read()
            .non_appendable_then_appendable_segments()
            .collect();
        let mut counts = vec![];
        let mut buffer = 0usize;
        let mut fingerprints = vec![];
        let mut roles = vec![];
        for locked in &captured {
            let LockedSegment::Original(segment) = locked else {
                self.running.store(false, Ordering::Release);
                return;
            };
            let guard = segment.read();
            let Some(&level) = self.members.get(&guard.uuid) else {
                self.running.store(false, Ordering::Release);
                return;
            };
            if guard.has_deferred_points() {
                self.running.store(false, Ordering::Release);
                return;
            }
            let count = guard.available_point_count_without_deferred();
            if level == 0 {
                buffer += count;
            } else {
                counts.resize(counts.len().max(level), 0);
                counts[level - 1] += count;
            }
            fingerprints.push((guard.uuid, guard.version(), count));
            roles.push(level);
        }
        fingerprints.sort_unstable();
        let Ok(Some(plan)) = self.hierarchy.plan(buffer, &counts) else {
            self.running.store(false, Ordering::Release);
            return;
        };
        fingerprints.retain(|(uuid, _, _)| {
            let level = self.members[uuid];
            level == 0 || level == plan.target_level || plan.source_levels.contains(&level)
        });
        let expected_sources: std::collections::HashSet<_> =
            fingerprints.iter().map(|(uuid, _, _)| *uuid).collect();
        {
            let mut last = self.last.lock();
            if *last == fingerprints {
                self.running.store(false, Ordering::Release);
                return;
            }
            *last = fingerprints;
        }
        let sources: Vec<_> = captured
            .into_iter()
            .zip(roles)
            .filter_map(|(s, l)| {
                (l == 0 || l == plan.target_level || plan.source_levels.contains(&l)).then_some(s)
            })
            .collect();
        self.scheduled.fetch_add(1, Ordering::Relaxed);
        let state = self.clone();
        let job_holder = holder.clone();
        tokio::task::spawn_blocking(move || {
            let _scope = common::ambient::unmeasured_guard(common::reason::reason(
                "DLI private maintenance",
            ));
            if state.native_enabled.load(Ordering::Acquire) {
                let run = || -> Result<(), String> {
                    let source = sources.first().ok_or("no sources")?.get().read();
                    let segment_config = source.config().clone();
                    let segments_path = source
                        .data_path()
                        .parent()
                        .ok_or("missing segments path")?
                        .to_owned();
                    drop(source);
                    let staging = tempfile::tempdir_in(&state.path).map_err(|e| e.to_string())?;
                    let ids: Vec<_> = job_holder
                        .read()
                        .iter()
                        .filter_map(|(id, member)| {
                            let uuid = member.get().read().segment_uuid();
                            expected_sources.contains(&uuid).then_some(id)
                        })
                        .collect();
                    if ids.len() != expected_sources.len() {
                        return Err("scheduled source missing".into());
                    }
                    let adapter = DliOptimizationStrategy {
                        staging: staging.path().to_owned(),
                        segments_path: segments_path.clone(),
                        segment_config,
                        model: state.model.clone(),
                        expected_sources,
                        expected_topology: Some((
                            state.hierarchy,
                            state.members.clone(),
                            plan.clone(),
                        )),
                    };
                    let output = Uuid::new_v4();
                    let budget = common::budget::ResourceBudget::new(1, 1);
                    let result = shard::optimize::execute_optimization("dli_research", job_holder.clone(), ids, output,
                        None, &shard::optimize::OptimizationPaths { segments_path, temp_path: staging.path().to_owned() },
                        budget.try_acquire(0,1).ok_or("resource unavailable")?, budget, &AtomicBool::new(false),
                        common::progress_tracker::ProgressTracker::new_for_test(),
                        &segment::common::operation_time_statistics::OperationDurationsAggregator::new(), &adapter, Box::new(|| {}));
                    result.map_err(|e| e.to_string())?;
                    if !job_holder
                        .read()
                        .iter_original()
                        .any(|(_, segment)| segment.read().uuid == output)
                    {
                        return Err("optimizer returned without publishing expected output".into());
                    }
                    state.published_uuids.lock().push(output);
                    state.published.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                };
                if let Err(error) = run() {
                    state.optimizer_errors.lock().push(error);
                    state.last.lock().clear();
                }
                state.running.store(false, Ordering::Release);
                state.completed.notify_one();
                return;
            }
            let mut physical = vec![];
            let result = (|| -> Result<Segment, String> {
                std::fs::create_dir_all(&state.path).map_err(|e| e.to_string())?;
                let source_config = sources
                    .first()
                    .ok_or("no inputs")?
                    .get()
                    .read()
                    .config()
                    .clone();
                let staging = tempfile::tempdir_in(&state.path).map_err(|e| e.to_string())?;
                let adapter = DliOptimizationStrategy {
                    staging: staging.path().to_owned(),
                    segments_path: state.path.clone(),
                    segment_config: source_config,
                    model: state.model.clone(),
                    expected_sources,
                    expected_topology: Some((state.hierarchy, state.members.clone(), plan.clone())),
                };
                let mut builder = shard::optimize::OptimizationStrategy::create_segment_builder(
                    &adapter, &sources,
                )
                .map_err(|e| e.to_string())?;
                let mut guards = vec![];
                for source in &sources {
                    let LockedSegment::Original(segment) = source else {
                        return Err("proxy source".into());
                    };
                    guards.push(segment.try_read().ok_or("private source busy")?);
                }
                for guard in &guards {
                    physical.push(guard.uuid);
                }
                let refs: Vec<_> = guards.iter().map(|g| &**g).collect();
                builder
                    .update(&refs, &AtomicBool::new(false))
                    .map_err(|e| e.to_string())?;
                drop(refs);
                drop(guards);
                // Training occurs after releasing source read guards. Research
                // private output is outside the shard; it is never searchable.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    builder.build_for_test(&state.path)
                }))
                .map_err(|_| "private native builder panicked".to_owned())
            })();
            if result.is_err() {
                state.last.lock().clear();
            }
            state.running.store(false, Ordering::Release);
            state.builds.lock().push(PrivateBuild {
                plan,
                sources: physical,
                output: result,
            });
            state.completed.notify_one();
        });
    }
}
