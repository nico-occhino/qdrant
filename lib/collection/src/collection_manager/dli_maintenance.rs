//! Insert-driven experimental DLI maintenance. Native mode publishes through
//! Qdrant's optimizer; the private mode remains a construction control.
use super::dli_schedule::{CompactionPlan, HierarchyConfig};
use common::flags::FeatureFlags;
use parking_lot::{Mutex, RwLock};
use segment::entry::{ReadSegmentEntry, StorageSegmentEntry};
use segment::index::VectorIndexEnum;
use segment::index::lmi_index::LmiConfig;
use segment::segment::Segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::types::{HnswGlobalConfig, Indexes};
use serde::{Deserialize, Serialize};
use shard::locked_segment::LockedSegment;
use shard::segment_holder::locked::LockedSegmentHolder;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use uuid::Uuid;

const DLI_DESCRIPTOR: &str = "dli_hierarchy.json";

/// Derived metadata only. The native active holder, not this file, owns data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DliHierarchyDescriptor {
    schema: u32,
    pub generation: usize,
    buffer_capacity: usize,
    arity: usize,
    members: Vec<(Uuid, usize)>,
}

impl DliHierarchyDescriptor {
    fn validate(
        &self,
        holder: &LockedSegmentHolder,
        hierarchy: HierarchyConfig,
    ) -> Result<HashMap<Uuid, usize>, String> {
        if self.schema != 1
            || self.generation == 0
            || self.buffer_capacity != hierarchy.buffer_capacity
            || self.arity != hierarchy.arity
        {
            return Err("incompatible DLI descriptor version or policy".into());
        }
        let roles: HashMap<_, _> = self.members.iter().copied().collect();
        if roles.len() != self.members.len() {
            return Err("duplicate DLI descriptor member".into());
        }
        let guard = holder.read();
        if guard.iter().count() != roles.len() {
            return Err("DLI descriptor does not cover active native membership".into());
        }
        for (_, locked) in guard.iter() {
            let LockedSegment::Original(segment) = locked else {
                return Err("DLI descriptor cannot activate over a proxy".into());
            };
            let segment = segment.read();
            let level = *roles
                .get(&segment.uuid)
                .ok_or("stale DLI descriptor member")?;
            if segment.has_deferred_points() {
                return Err("deferred DLI member needs native fallback".into());
            }
            let index = segment
                .vector_data
                .get(segment::data_types::vectors::DEFAULT_VECTOR_NAME)
                .ok_or("missing DLI vector")?
                .vector_index
                .borrow();
            match (level, &*index) {
                (0, VectorIndexEnum::Plain(_)) if segment.is_appendable() => {}
                (1.., VectorIndexEnum::Lmi(lmi)) if lmi.routing_state().is_some() => {
                    if segment.available_point_count_without_deferred()
                        > hierarchy
                            .level_capacity(level)
                            .map_err(|e| format!("{e:?}"))?
                    {
                        return Err("DLI level exceeds geometric capacity".into());
                    }
                }
                _ => return Err("DLI member role or learned artifacts are invalid".into()),
            }
        }
        Ok(roles)
    }
}

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
    #[cfg(test)]
    pub validation_hook: Option<Arc<DliBeforeNativeHook>>,
}

impl shard::optimize::OptimizationStrategy for DliOptimizationStrategy {
    fn after_source_validation(&self) -> segment::common::operation_error::OperationResult<()> {
        #[cfg(test)]
        if let Some(hook) = &self.validation_hook {
            hook.entered.send(()).map_err(|e| {
                segment::common::operation_error::OperationError::service_error(e.to_string())
            })?;
            hook.resume
                .lock()
                .recv_timeout(std::time::Duration::from_secs(10))
                .map_err(|e| {
                    segment::common::operation_error::OperationError::service_error(e.to_string())
                })?;
        }
        Ok(())
    }
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
#[cfg(test)]
pub struct DliBeforeNativeHook {
    pub entered: std::sync::mpsc::Sender<()>,
    pub resume: Mutex<std::sync::mpsc::Receiver<()>>,
}
pub struct DliMaintenance {
    pub hierarchy: HierarchyConfig,
    pub members: RwLock<HashMap<Uuid, usize>>,
    pub generation: AtomicUsize,
    pub search: Mutex<Option<Arc<super::segments_searcher::DliResearchConfig>>>,
    pub model: LmiConfig,
    pub path: PathBuf,
    running: AtomicBool,
    stopped: AtomicBool,
    worker: Mutex<Option<tokio::task::JoinHandle<()>>>,
    last: Mutex<Vec<(Uuid, u64, usize)>>,
    /// Experimental native publication is explicitly enabled by a test only.
    pub native_enabled: AtomicBool,
    pub published: AtomicUsize,
    pub published_uuids: Mutex<Vec<Uuid>>,
    pub optimizer_errors: Mutex<Vec<String>>,
    pub scheduled: AtomicUsize,
    pub builds: Mutex<Vec<PrivateBuild>>,
    pub completed: tokio::sync::Notify,
    #[cfg(test)]
    pub before_native: Mutex<Option<Arc<DliBeforeNativeHook>>>,
    #[cfg(test)]
    pub after_validation: Mutex<Option<Arc<DliBeforeNativeHook>>>,
}
impl std::fmt::Debug for DliMaintenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DliMaintenance")
            .field("generation", &self.generation.load(Ordering::Relaxed))
            .field("members", &*self.members.read())
            .finish_non_exhaustive()
    }
}
impl DliMaintenance {
    pub fn load_validated_descriptor(
        &self,
        holder: &LockedSegmentHolder,
    ) -> Result<DliHierarchyDescriptor, String> {
        // Fail closed before parsing: a corrupt file must not leave a former
        // process-local learned view active. Empty membership selects exact
        // native fallback in the search dispatcher.
        if let Some(search) = self.search.lock().as_ref() {
            search.members.write().clear();
        }
        let path = self.path.join(DLI_DESCRIPTOR);
        let descriptor: DliHierarchyDescriptor =
            common::fs::read_json(&path).map_err(|e| e.to_string())?;
        let roles = descriptor.validate(holder, self.hierarchy)?;
        *self.members.write() = roles.clone();
        self.generation
            .store(descriptor.generation, Ordering::Release);
        if let Some(search) = self.search.lock().as_ref() {
            *search.members.write() = roles;
        }
        Ok(descriptor)
    }

    fn save_descriptor(
        &self,
        members: &HashMap<Uuid, usize>,
        generation: usize,
    ) -> Result<(), String> {
        let mut sorted: Vec<_> = members
            .iter()
            .map(|(uuid, level)| (*uuid, *level))
            .collect();
        sorted.sort_unstable_by_key(|(uuid, _)| *uuid);
        let descriptor = DliHierarchyDescriptor {
            schema: 1,
            generation,
            buffer_capacity: self.hierarchy.buffer_capacity,
            arity: self.hierarchy.arity,
            members: sorted,
        };
        common::fs::atomic_save_json(&self.path.join(DLI_DESCRIPTOR), &descriptor)
            .map_err(|e| e.to_string())
    }
    pub fn new(
        hierarchy: HierarchyConfig,
        members: HashMap<Uuid, usize>,
        model: LmiConfig,
        path: PathBuf,
    ) -> Arc<Self> {
        Arc::new(Self {
            hierarchy,
            members: RwLock::new(members),
            generation: AtomicUsize::new(0),
            search: Mutex::new(None),
            model,
            path,
            running: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            worker: Mutex::new(None),
            last: Mutex::new(Vec::new()),
            native_enabled: AtomicBool::new(false),
            published: AtomicUsize::new(0),
            published_uuids: Mutex::new(Vec::new()),
            optimizer_errors: Mutex::new(Vec::new()),
            scheduled: AtomicUsize::new(0),
            builds: Mutex::new(Vec::new()),
            completed: Default::default(),
            #[cfg(test)]
            before_native: Mutex::new(None),
            #[cfg(test)]
            after_validation: Mutex::new(None),
        })
    }
    pub fn attach_search(&self, search: Arc<super::segments_searcher::DliResearchConfig>) {
        *self.search.lock() = Some(search);
    }
    /// Cancel native optimization and wait for its rollback/terminal state.
    pub async fn cancel_and_wait(&self) {
        self.request_cancel();
        let worker = self.worker.lock().take();
        if let Some(worker) = worker {
            if worker.await.is_err() {
                self.running.store(false, Ordering::Release);
                self.completed.notify_waiters();
            }
        }
        while self.running.load(Ordering::Acquire) {
            let notified = self.completed.notified();
            if !self.running.load(Ordering::Acquire) {
                break;
            }
            notified.await;
        }
    }
    pub fn request_cancel(&self) {
        self.stopped.store(true, Ordering::Release);
    }
    pub fn resume_after_cancel(&self) {
        assert!(!self.running.load(Ordering::Acquire));
        self.stopped.store(false, Ordering::Release);
    }
    /// Called after native update locks have been released. One private job per
    /// observed source version set; no authoritative point state is stored here.
    pub fn observe(self: &Arc<Self>, holder: &LockedSegmentHolder) {
        if self.stopped.load(Ordering::Acquire) {
            return;
        }
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
        let roles_snapshot = self.members.read().clone();
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
            let Some(&level) = roles_snapshot.get(&guard.uuid) else {
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
            let level = roles_snapshot[uuid];
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
        let (worker_start, worker_ready) = std::sync::mpsc::channel();
        let worker = tokio::task::spawn_blocking(move || {
            if worker_ready.recv().is_err() {
                return;
            }
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
                            roles_snapshot.clone(),
                            plan.clone(),
                        )),
                        #[cfg(test)]
                        validation_hook: state.after_validation.lock().clone(),
                    };
                    let output = Uuid::new_v4();
                    #[cfg(test)]
                    if let Some(hook) = state.before_native.lock().as_ref() {
                        hook.entered.send(()).map_err(|e| e.to_string())?;
                        hook.resume
                            .lock()
                            .recv_timeout(std::time::Duration::from_secs(10))
                            .map_err(|e| e.to_string())?;
                    }
                    let budget = common::budget::ResourceBudget::new(1, 1);
                    let result = shard::optimize::execute_optimization("dli_research", job_holder.clone(), ids, output,
                        None, &shard::optimize::OptimizationPaths { segments_path, temp_path: staging.path().to_owned() },
                        budget.try_acquire(0,1).ok_or("resource unavailable")?, budget, &state.stopped,
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
                    // Native publication precedes derived hierarchy activation.
                    // While this map is stale, the searcher's complete-membership
                    // check forces exact fallback on its captured handles.
                    let new_roles = {
                        let guard = job_holder.read();
                        let mut next = roles_snapshot.clone();
                        for source in &adapter.expected_sources {
                            next.remove(source);
                        }
                        next.insert(output, plan.target_level);
                        let mut active = std::collections::HashSet::new();
                        for (_, member) in guard.iter_original() {
                            let segment = member.read();
                            let uuid = segment.uuid;
                            if !active.insert(uuid) {
                                return Err("duplicate active DLI physical identity".into());
                            }
                            if !next.contains_key(&uuid) {
                                if !segment.is_appendable() {
                                    return Err("new DLI member is not appendable".into());
                                }
                                next.insert(uuid, 0);
                            }
                        }
                        if active.len() != next.len()
                            || next.keys().any(|uuid| !active.contains(uuid))
                        {
                            return Err("published DLI membership is incomplete".into());
                        }
                        next
                    };
                    let generation = state.generation.load(Ordering::Acquire) + 1;
                    // A failed metadata write cannot undo native publication.
                    // Keep in-memory search safe and report degraded durability.
                    if let Err(error) = state.save_descriptor(&new_roles, generation) {
                        state
                            .optimizer_errors
                            .lock()
                            .push(format!("DLI descriptor: {error}"));
                    }
                    *state.members.write() = new_roles.clone();
                    state.generation.store(generation, Ordering::Release);
                    if let Some(search) = state.search.lock().as_ref() {
                        *search.members.write() = new_roles;
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
                    expected_topology: Some((
                        state.hierarchy,
                        roles_snapshot.clone(),
                        plan.clone(),
                    )),
                    #[cfg(test)]
                    validation_hook: None,
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
                    .update(&refs, &state.stopped)
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
        *self.worker.lock() = Some(worker);
        let _ = worker_start.send(());
    }
}
