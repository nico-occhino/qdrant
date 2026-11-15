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
}

impl shard::optimize::OptimizationStrategy for DliOptimizationStrategy {
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
        Err(
            segment::common::operation_error::OperationError::service_error(
                "DLI optimizer requires an existing appendable segment outside its inputs",
            ),
        )
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
    last: Mutex<Vec<(Uuid, u64)>>,
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
            fingerprints.push((guard.uuid, guard.version()));
            roles.push(level);
        }
        fingerprints.sort_unstable();
        let Ok(Some(plan)) = self.hierarchy.plan(buffer, &counts) else {
            self.running.store(false, Ordering::Release);
            return;
        };
        fingerprints.retain(|(uuid, _)| {
            let level = self.members[uuid];
            level == 0 || level == plan.target_level || plan.source_levels.contains(&level)
        });
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
        tokio::task::spawn_blocking(move || {
            let _scope = common::ambient::unmeasured_guard(common::reason::reason(
                "DLI private maintenance",
            ));
            let mut physical = vec![];
            let result = (|| -> Result<Segment, String> {
                std::fs::create_dir_all(&state.path).map_err(|e| e.to_string())?;
                let mut guards = vec![];
                for source in &sources {
                    let LockedSegment::Original(segment) = source else {
                        return Err("proxy source".into());
                    };
                    guards.push(segment.try_read().ok_or("private source busy")?);
                }
                let mut config = guards.first().ok_or("no inputs")?.config().clone();
                for guard in &guards {
                    physical.push(guard.uuid);
                }
                let vector = config
                    .vector_data
                    .get_mut(segment::data_types::vectors::DEFAULT_VECTOR_NAME)
                    .ok_or("missing vector")?;
                vector.index = Indexes::LmiTrained(state.model.clone());
                let staging = tempfile::tempdir_in(&state.path).map_err(|e| e.to_string())?;
                let mut builder = SegmentBuilder::new(
                    staging.path(),
                    &config,
                    &HnswGlobalConfig::default(),
                    FeatureFlags::default(),
                )
                .map_err(|e| e.to_string())?;
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
