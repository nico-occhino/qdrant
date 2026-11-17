//! Experimental native routing primitive, not a collection search entry point.
//!
//! Callers must keep every segment read-locked for the whole call. The returned
//! per-segment results are deliberately not merged: candidate-only version
//! aggregation cannot establish that a newer, unselected copy does not exist.
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use common::types::DeferredBehavior;
use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector};
use segment::entry::ReadSegmentEntry;
use segment::id_tracker::IdTrackerRead;
use segment::index::candidate_scoring::score_candidates;
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::types::{Distance, PointIdType, ScoredPoint, WithPayload, WithVector};
use shard::search_result_aggregator::BatchResultAggregator;
use uuid::Uuid;

use super::dli_bucket_plan::{
    BucketPrediction, RoutingPolicy, SelectedBuckets, select, select_density, select_reserved,
};

/// Linux task-local counters. CPU is in clock ticks (getconf CLK_TCK), not ns.
/// Faults are task faults, not bytes read or a cache-residency measurement.
fn task_access_counters() -> Option<(u64, u64, u64)> {
    let stat = std::fs::read_to_string("/proc/thread-self/stat").ok()?;
    let (_, fields) = stat.rsplit_once(')')?;
    let fields: Vec<_> = fields.split_whitespace().collect();
    let value = |i: usize| fields.get(i)?.parse::<u64>().ok();
    Some((value(7)?, value(9)?, value(11)?.checked_add(value(12)?)?))
}

/// Output is partitioned by physical segment. It must not be returned to clients
/// until complete native point-version/visibility resolution has been applied.
pub struct NativeRouteResult {
    pub plan: SelectedBuckets,
    pub routing_and_planning_time: Duration,
    pub scoring_time: Duration,
    pub native_scorer_time: Duration,
    pub hit_materialization_time: Duration,
    pub router_inference_time: Duration,
    pub bucket_planning_time: Duration,
    pub models_evaluated: usize,
    pub posting_candidates: usize,
    pub unique_posting_offsets: usize,
    /// Live point count, which can exceed the number of vectors in the buffer.
    pub buffer_point_estimate: usize,
    pub per_segment: Vec<(Uuid, Vec<ScoredPoint>)>,
}

/// Reasons this bounded check cannot establish native visibility. The caller
/// must run complete native exact search on the same captured holder view.
#[derive(Debug, PartialEq, Eq)]
pub enum ResolutionFallback {
    DeletedState,
    DeferredState,
    DuplicateMember,
    AmbiguousOccurrence,
    CandidateMismatch,
}

pub enum CandidateResolution {
    Resolved(Vec<ScoredPoint>),
    ExactFallback(ResolutionFallback),
}

/// Conservative version check on caller-pinned original segments. The caller
/// must independently establish complete holder coverage and keep all segment
/// read locks for routing and resolution; proxies and filters are unsupported.
/// Native deletion removes the external-ID mapping and native scoring excludes
/// deleted offsets. Unrelated deleted slots therefore do not invalidate a live
/// candidate. Complete captured membership is still checked for every selected
/// identity; duplicate visible occurrences and deferred state remain fail closed.
/// This follows native search visibility, not a separate tombstone authority.
pub fn resolve_pinned_candidates(
    members: &[(usize, &Segment)],
    routed: NativeRouteResult,
    limit: usize,
) -> OperationResult<CandidateResolution> {
    let fallback = |reason| Ok(CandidateResolution::ExactFallback(reason));
    let mut physical = HashSet::new();
    for &(_, segment) in members {
        if !physical.insert(segment.uuid) {
            return fallback(ResolutionFallback::DuplicateMember);
        }
        if segment.has_deferred_points() {
            return fallback(ResolutionFallback::DeferredState);
        }
    }
    let mut selected = HashMap::new();
    for (owner, hits) in &routed.per_segment {
        if !physical.contains(owner) {
            return fallback(ResolutionFallback::CandidateMismatch);
        }
        let segment = members
            .iter()
            .find(|(_, segment)| segment.uuid == *owner)
            .expect("validated physical member")
            .1;
        let tracker = segment.id_tracker.borrow();
        for hit in hits {
            let Some(location) = hit.location else {
                return fallback(ResolutionFallback::CandidateMismatch);
            };
            if location.segment != *owner
                || tracker.external_id(location.offset) != Some(hit.id)
                || tracker.internal_version(location.offset) != Some(hit.version)
            {
                return fallback(ResolutionFallback::CandidateMismatch);
            }
            if selected.insert(hit.id, (*owner, hit.version)).is_some() {
                return fallback(ResolutionFallback::AmbiguousOccurrence);
            }
        }
    }
    let ids: Vec<PointIdType> = selected.keys().copied().collect();
    let mut occurrences = HashMap::new();
    for &(_, segment) in members {
        let visible = segment.has_points(&ids, DeferredBehavior::VisibleOnly)?;
        let heads = segment.point_versions(&ids)?;
        for id in visible {
            let Some(version) = heads.get(&id) else {
                return fallback(ResolutionFallback::CandidateMismatch);
            };
            if occurrences.insert(id, (segment.uuid, *version)).is_some() {
                return fallback(ResolutionFallback::AmbiguousOccurrence);
            }
        }
    }
    if selected != occurrences {
        return fallback(ResolutionFallback::CandidateMismatch);
    }
    let mut aggregator = BatchResultAggregator::new(std::iter::once(limit));
    aggregator.update_point_versions(routed.per_segment.iter().flat_map(|(_, hits)| hits));
    aggregator.update_batch_results(0, routed.per_segment.into_iter().flat_map(|(_, hits)| hits));
    Ok(CandidateResolution::Resolved(
        aggregator.into_topk().pop().unwrap(),
    ))
}

/// Route and score one dense, unfiltered query on explicitly pinned native
/// segment references. Level zero denotes a Plain buffer, all other levels
/// denote independently trained LMI indexes. The caller supplies complete
/// membership; this function cannot validate an unseen holder topology.
pub fn route_pinned(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
    context: &VectorQueryContext,
) -> OperationResult<NativeRouteResult> {
    route_pinned_with_policy(
        members,
        query,
        arity,
        budget,
        context,
        RoutingPolicy::SlotWeighted,
        usize::MAX,
    )
}

pub fn route_pinned_with_policy(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
    context: &VectorQueryContext,
    policy: RoutingPolicy,
    materialize_limit: usize,
) -> OperationResult<NativeRouteResult> {
    let fail = |message: &str| OperationError::service_error(message.to_owned());
    let start = Instant::now();
    let mut predictions = Vec::new();
    let mut router_inference_time = Duration::ZERO;
    let mut models_evaluated = 0usize;
    let mut seen = HashSet::new();
    let mut distance = None;
    for &(level, segment) in members {
        if !seen.insert(segment.uuid) {
            return Err(fail("duplicate DLI physical member"));
        }
        let config = segment
            .config()
            .vector_data
            .get(DEFAULT_VECTOR_NAME)
            .ok_or_else(|| fail("missing default vector configuration"))?;
        if config.size != query.len() || distance.is_some_and(|d| d != config.distance) {
            return Err(fail("incompatible DLI vector configuration"));
        }
        distance = Some(config.distance);
        let data = segment
            .vector_data
            .get(DEFAULT_VECTOR_NAME)
            .ok_or_else(|| fail("missing default vector storage"))?;
        if config.multivector_config.is_some() || data.quantized_vectors.borrow().is_some() {
            return Err(fail("unsupported DLI vector storage"));
        }
        let index = data.vector_index.borrow();
        match (level, &*index) {
            (0, VectorIndexEnum::Plain(_)) => predictions.push(BucketPrediction {
                level,
                physical: segment.uuid,
                bucket: None,
                probability: 0.0,
                occupancy: segment.id_tracker.borrow().available_point_count(),
            }),
            (1.., VectorIndexEnum::Lmi(lmi)) => {
                let state = lmi
                    .routing_state()
                    .ok_or_else(|| fail("missing LMI routing state"))?;
                let routed = if config.distance == Distance::Cosine {
                    Distance::Cosine.preprocess_vector::<f32>(query.to_vec())
                } else {
                    query.to_vec()
                };
                let inference_started = Instant::now();
                let logits = state.router().forward(&routed)?;
                router_inference_time += inference_started.elapsed();
                models_evaluated += 1;
                if logits.len() != state.postings().len() || logits.is_empty() {
                    return Err(fail("LMI router/postings bucket mismatch"));
                }
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = logits.iter().map(|v| (v - max).exp()).sum();
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(fail("invalid LMI router probabilities"));
                }
                for (bucket, logit) in logits.iter().enumerate() {
                    predictions.push(BucketPrediction {
                        level,
                        physical: segment.uuid,
                        bucket: Some(bucket),
                        probability: (logit - max).exp() / denominator,
                        occupancy: state
                            .postings()
                            .get(bucket)
                            .expect("validated bucket")
                            .len(),
                    });
                }
            }
            _ => return Err(fail("DLI member role/index mismatch")),
        }
    }
    let planning_started = Instant::now();
    let plan = match policy {
        RoutingPolicy::SlotWeighted => select(predictions, arity, budget),
        RoutingPolicy::ReservedPerLevel => select_reserved(predictions, arity, budget),
        RoutingPolicy::PopulationDensity => select_density(predictions, arity, budget),
    }
    .map_err(|e| fail(&format!("invalid DLI candidate plan: {e:?}")))?;
    let bucket_planning_time = planning_started.elapsed();
    let routing_and_planning_time = start.elapsed();
    if let Some(path) = std::env::var_os("DLI_ROUTING_EVENTS") {
        use std::io::Write;
        let event = serde_json::json!({"policy":policy,"budget":budget,"buckets":plan.buckets,"members":members.iter().map(|(l,s)|(s.uuid,l,s.available_point_count())).collect::<Vec<_>>()});
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{event}");
        }
    }

    let mut native_scorer_time = Duration::ZERO;
    let mut hit_materialization_time = Duration::ZERO;
    let started_scoring = Instant::now();
    let mut posting_candidates = 0usize;
    let mut unique_posting_offsets = 0usize;
    let mut buffer_point_estimate = 0usize;
    let mut per_segment = Vec::new();
    let query_vector: QueryVector = query.to_vec().into();
    let access_trace = std::env::var_os("DLI_NATIVE_ACCESS_EVENTS");
    let mut grouped: HashMap<Uuid, Vec<usize>> = HashMap::new();
    for bucket in &plan.buckets {
        grouped
            .entry(bucket.physical)
            .or_default()
            .extend(bucket.bucket);
    }
    for &(level, segment) in members {
        let Some(buckets) = grouped.get(&segment.uuid) else {
            continue;
        };
        let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
        let index = data.vector_index.borrow();
        let access_before = access_trace.as_ref().and_then(|_| task_access_counters());
        let access_started = Instant::now();
        let offsets = if level == 0 {
            // Every selected vector is scored, but only potential global winners need materialization.
            let count = segment.id_tracker.borrow().available_point_count();
            buffer_point_estimate += count;
            let score_started = Instant::now();
            let offsets = index
                .search(
                    &[&query_vector],
                    None,
                    count.min(materialize_limit),
                    None,
                    context,
                )?
                .pop()
                .expect("one buffer query");
            native_scorer_time += score_started.elapsed();
            offsets
        } else {
            let VectorIndexEnum::Lmi(lmi) = &*index else {
                unreachable!()
            };
            let state = lmi.routing_state().expect("validated while pinned");
            let mut candidates = Vec::new();
            for &bucket in buckets {
                candidates.extend_from_slice(state.postings().get(bucket).expect("planned bucket"));
            }
            posting_candidates += candidates.len();
            candidates.sort_unstable();
            candidates.dedup();
            unique_posting_offsets += candidates.len();
            let tracker = segment.id_tracker.borrow();
            let storage = data.vector_storage.borrow();
            let score_started = Instant::now();
            let offsets = score_candidates(
                &query_vector,
                &candidates,
                candidates.len().min(materialize_limit),
                &tracker,
                &storage,
                context,
            )?;
            native_scorer_time += score_started.elapsed();
            offsets
        };
        if let Some(path) = &access_trace {
            let wall_nanos = access_started.elapsed().as_nanos();
            let delta = access_before
                .zip(task_access_counters())
                .and_then(|(a, b)| {
                    Some((
                        b.0.checked_sub(a.0)?,
                        b.1.checked_sub(a.1)?,
                        b.2.checked_sub(a.2)?,
                    ))
                });
            let event = serde_json::json!({"physical":segment.uuid,"level":level,
                "storage":segment.config().vector_data[DEFAULT_VECTOR_NAME].storage_type,
                "phase":"candidate_prepare_and_score", "wall_nanos":wall_nanos,
                "minor_faults":delta.map(|d|d.0),"major_faults":delta.map(|d|d.1),
                "thread_cpu_ticks":delta.map(|d|d.2),"returned_offsets":offsets.len()});
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(file, "{event}");
            }
        }
        let materialization_started = Instant::now();
        let hits = segment.with_view(|view| {
            view.process_search_result(
                offsets,
                &WithPayload::default(),
                &WithVector::default(),
                &context.is_stopped(),
            )
        })?;
        hit_materialization_time += materialization_started.elapsed();
        per_segment.push((segment.uuid, hits));
    }
    Ok(NativeRouteResult {
        plan,
        routing_and_planning_time,
        native_scorer_time,
        hit_materialization_time,
        router_inference_time,
        bucket_planning_time,
        models_evaluated,
        scoring_time: started_scoring.elapsed(),
        posting_candidates,
        unique_posting_offsets,
        buffer_point_estimate,
        per_segment,
    })
}

/// Route and validate before producing any mergeable results. The caller must
/// honor `ExactFallback`; this wrapper does not execute collection search.
pub fn route_resolved_pinned(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
    limit: usize,
    context: &VectorQueryContext,
) -> OperationResult<CandidateResolution> {
    let routed = route_pinned(members, query, arity, budget, context)?;
    resolve_pinned_candidates(members, routed, limit)
}

#[cfg(all(test, feature = "lmi-training", feature = "testing"))]
mod tests {
    use common::flags::FeatureFlags;
    use segment::data_types::vectors::only_default_vector;
    use segment::entry::entry_point::{
        NonAppendableSegmentEntry, SegmentEntry, StorageSegmentEntry,
    };
    use segment::index::lmi_index::LmiConfig;
    use segment::segment_constructor::load_segment;
    use segment::segment_constructor::segment_builder::SegmentBuilder;
    use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
    use segment::types::{HnswGlobalConfig, Indexes, PointIdType};
    use std::sync::atomic::AtomicBool;

    use super::*;

    #[test]
    fn optimizer_adapter_rejects_stale_physical_sources() {
        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let source = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        let config = source.config().clone();
        let adapter = crate::collection_manager::dli_maintenance::DliOptimizationStrategy {
            staging: root.path().to_owned(),
            segments_path: root.path().to_owned(),
            segment_config: config,
            destination_on_disk: true,
            model: LmiConfig::default(),
            expected_sources: HashSet::from([Uuid::new_v4()]),
            expected_topology: None,
            #[cfg(test)]
            validation_hook: None,
        };
        let inputs = vec![shard::locked_segment::LockedSegment::from(source)];
        let (cow, _token) =
            shard::optimize::OptimizationStrategy::create_temp_segment(&adapter).unwrap();
        assert!(cow.get().read().is_appendable());
        cow.get()
            .write()
            .upsert_point(1, 7_u64.into(), only_default_vector(&[1.0, 0.0]))
            .unwrap();
        assert!(
            cow.get()
                .read()
                .has_point(7_u64.into(), common::types::DeferredBehavior::VisibleOnly)
        );
        assert!(
            shard::optimize::OptimizationStrategy::validate_sources(&adapter, &inputs).is_err()
        );
        assert!(
            shard::optimize::OptimizationStrategy::create_segment_builder(&adapter, &inputs)
                .is_err()
        );
    }

    #[test]
    fn dli_destination_storage_is_independent_of_source_order_and_reopens() {
        use segment::types::VectorStorageType;
        use shard::optimize::OptimizationStrategy;
        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        for (cold_a, cold_b) in [(false, false), (true, true), (true, false)] {
            let mut inputs = Vec::new();
            for (which, cold) in [(0, cold_a), (1, cold_b)] {
                let base = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
                let mut config = base.config().clone();
                config
                    .vector_data
                    .get_mut(DEFAULT_VECTOR_NAME)
                    .unwrap()
                    .storage_type = if cold {
                    VectorStorageType::ChunkedMmap
                } else {
                    VectorStorageType::InRamChunkedMmap
                };
                let (mut segment, _) =
                    segment::segment_constructor::build_segment(root.path(), &config, None, true)
                        .unwrap();
                for i in 0..2 {
                    segment
                        .upsert_point(
                            1,
                            (1 + which * 2 + i as u64).into(),
                            only_default_vector(if i == 0 { &[1., 0.] } else { &[0., 1.] }),
                        )
                        .unwrap();
                }
                inputs.push(shard::locked_segment::LockedSegment::from(segment));
            }
            for reversed in [false, true] {
                if reversed {
                    inputs.reverse();
                }
                for on_disk in [false, true] {
                    let config = inputs[0].get().read().config().clone();
                    let adapter =
                        crate::collection_manager::dli_maintenance::DliOptimizationStrategy {
                            staging: root.path().to_owned(),
                            segments_path: root.path().to_owned(),
                            segment_config: config,
                            destination_on_disk: on_disk,
                            model: LmiConfig {
                                n_buckets: 2,
                                nprobe: 2,
                                sample_size: 4,
                                hidden_dim: 4,
                                epochs: 1,
                                batch_size: 2,
                                kmeans_iterations: 2,
                                ..Default::default()
                            },
                            expected_sources: inputs
                                .iter()
                                .map(|s| s.get().read().segment_uuid())
                                .collect(),
                            expected_topology: None,
                            validation_hook: None,
                        };
                    let mut builder = adapter.create_segment_builder(&inputs).unwrap();
                    {
                        let guards: Vec<_> = inputs
                            .iter()
                            .map(|s| match s {
                                shard::locked_segment::LockedSegment::Original(s) => s.read(),
                                _ => unreachable!(),
                            })
                            .collect();
                        builder
                            .update(
                                &guards.iter().map(|s| &**s).collect::<Vec<_>>(),
                                &AtomicBool::new(false),
                            )
                            .unwrap();
                    }
                    let built = builder.build_for_test(root.path());
                    let expected = if on_disk {
                        VectorStorageType::ChunkedMmap
                    } else {
                        VectorStorageType::InRamChunkedMmap
                    };
                    assert_eq!(
                        built.config().vector_data[DEFAULT_VECTOR_NAME].storage_type,
                        expected
                    );
                    assert_eq!(built.available_point_count(), 4);
                    built.flush(true).unwrap();
                    let path = built.segment_path.clone();
                    let uuid = built.uuid;
                    drop(built);
                    let reopened =
                        load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
                    assert_eq!(
                        reopened.config().vector_data[DEFAULT_VECTOR_NAME].storage_type,
                        expected
                    );
                    assert_eq!(reopened.available_point_count(), 4);
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dli_query_guard_and_native_upsert_complete_without_deadlock() {
        query_guard_native_mutation(false).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dli_query_guard_and_native_delete_complete_without_resurrection() {
        query_guard_native_mutation(true).await;
    }

    async fn query_guard_native_mutation(delete: bool) {
        use crate::collection_manager::segments_searcher::{
            DliGuardHook, DliResearchConfig, SegmentsSearcher,
        };
        use crate::common::adaptive_handle::AdaptiveSearchHandle;
        use crate::operations::types::CoreSearchRequest;
        use common::ambient::{AmbientContext, Handoff};
        use segment::data_types::query_context::QueryContext;
        use shard::operations::point_ops::{PointStructPersisted, VectorStructPersisted};
        use shard::optimizers::config::DEFAULT_INDEXING_THRESHOLD_KB;
        use shard::search::CoreSearchRequestBatch;
        use shard::segment_holder::{SegmentHolder, locked::LockedSegmentHolder};
        use shard::update::{delete_points, upsert_points};
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let level = trained(
            root.path(),
            &[
                (11, [1., 0.]),
                (12, [0., 1.]),
                (13, [-1., 0.]),
                (14, [0., -1.]),
            ],
            17,
        );
        let buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        let config = Arc::new(
            DliResearchConfig::new(vec![(level.uuid, 1), (buffer.uuid, 0)], 2, 100).unwrap(),
        );
        config
            .materialize_topk
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let mut native = SegmentHolder::default();
        native.add_new(level);
        native.add_new(buffer);
        let holder = LockedSegmentHolder::new(native);
        let (entered_tx, entered_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        *config.test_guard_hook.lock() = Some(DliGuardHook {
            entered: entered_tx,
            resume: parking_lot::Mutex::new(resume_rx),
        });
        let request = CoreSearchRequest {
            query: vec![1., 0.].into(),
            filter: None,
            params: None,
            limit: 10,
            offset: 0,
            with_payload: None,
            with_vector: None,
            score_threshold: None,
        };
        let query_holder = holder.clone();
        let query_config = config.clone();
        let query = tokio::spawn(async move {
            SegmentsSearcher::search_dli_research(
                query_holder,
                Arc::new(CoreSearchRequestBatch {
                    searches: vec![request],
                }),
                &AdaptiveSearchHandle::current_for_tests(),
                QueryContext::new(
                    DEFAULT_INDEXING_THRESHOLD_KB,
                    Handoff::Measured(AmbientContext::new()),
                ),
                Duration::from_secs(10),
                query_config,
            )
            .await
        });
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let update_holder = holder.clone();
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _scope = common::ambient::test_guard();
            let _updates = update_holder.acquire_updates_lock();
            let guard = update_holder.read();
            started_tx.send(()).unwrap();
            let points = [PointStructPersisted {
                id: 11_u64.into(),
                vector: VectorStructPersisted::Single(vec![0., 1.]),
                payload: None,
            }];
            let result = if delete {
                delete_points(&guard, 100, &[11_u64.into()]).map(|_| ())
            } else {
                upsert_points(&guard, 100, &points, None).map(|_| ())
            };
            done_tx.send(result.is_ok()).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(done_rx.try_recv().is_err());
        resume_tx.send(()).unwrap();
        let result = tokio::time::timeout(Duration::from_secs(10), query)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(
            result[0]
                .iter()
                .any(|point| point.id == 11_u64.into() && point.score > 0.99)
        );
        assert!(done_rx.recv_timeout(Duration::from_secs(10)).unwrap());
        worker.join().unwrap();
        assert_eq!(
            config
                .learned_queries
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        *config.test_guard_hook.lock() = None;
        let request = CoreSearchRequest {
            query: vec![1., 0.].into(),
            filter: None,
            params: None,
            limit: 10,
            offset: 0,
            with_payload: None,
            with_vector: None,
            score_threshold: None,
        };
        let after = SegmentsSearcher::search_dli_research(
            holder,
            Arc::new(CoreSearchRequestBatch {
                searches: vec![request],
            }),
            &AdaptiveSearchHandle::current_for_tests(),
            QueryContext::new(
                DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::new()),
            ),
            Duration::from_secs(10),
            config.clone(),
        )
        .await
        .unwrap();
        if delete {
            assert!(!after[0].iter().any(|point| point.id == 11_u64.into()));
            assert_eq!(after[0].len(), 3);
        } else {
            let updated = after[0]
                .iter()
                .find(|point| point.id == 11_u64.into())
                .unwrap();
            assert_eq!(updated.version, 100);
            assert!(updated.score.abs() < 1e-5);
        }
        assert_eq!(
            config
                .exact_fallbacks
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dli_dispatch_during_native_proxy_build_and_publication() {
        use crate::collection_manager::segments_searcher::{DliResearchConfig, SegmentsSearcher};
        use crate::common::adaptive_handle::AdaptiveSearchHandle;
        use crate::operations::types::CoreSearchRequest;
        use common::ambient::{AmbientContext, Handoff};
        use common::budget::ResourceBudget;
        use common::progress_tracker::ProgressTracker;
        use segment::data_types::query_context::QueryContext;
        use shard::locked_segment::LockedSegment;
        use shard::operations::point_ops::{PointStructPersisted, VectorStructPersisted};
        use shard::optimizers::config::DEFAULT_INDEXING_THRESHOLD_KB;
        use shard::search::CoreSearchRequestBatch;
        use shard::segment_holder::{SegmentHolder, locked::LockedSegmentHolder};
        use shard::update::{delete_points, upsert_points};
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        struct Paused {
            path: std::path::PathBuf,
            expected: Uuid,
            validated: Arc<AtomicBool>,
            entered: mpsc::Sender<()>,
            resume: parking_lot::Mutex<mpsc::Receiver<()>>,
        }
        impl shard::optimize::OptimizationStrategy for Paused {
            fn validate_sources(&self, inputs: &[LockedSegment]) -> OperationResult<()> {
                assert_eq!(inputs.len(), 1);
                assert_eq!(inputs[0].get().read().segment_uuid(), self.expected);
                self.validated
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            fn create_segment_builder(
                &self,
                inputs: &[LockedSegment],
            ) -> OperationResult<SegmentBuilder> {
                let config = inputs[0].get().read().config().clone();
                self.entered.send(()).unwrap();
                self.resume
                    .lock()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
                let adapter = crate::collection_manager::dli_maintenance::DliOptimizationStrategy {
                    staging: self.path.clone(),
                    segments_path: self.path.clone(),
                    segment_config: config,
                    destination_on_disk: true,
                    model: LmiConfig {
                        n_buckets: 2,
                        sample_size: 4,
                        hidden_dim: 4,
                        epochs: 1,
                        batch_size: 4,
                        routing_batch_size: 4,
                        kmeans_iterations: 2,
                        nprobe: 2,
                        seed: 17,
                    },
                    expected_sources: inputs
                        .iter()
                        .map(|input| input.get().read().segment_uuid())
                        .collect(),
                    expected_topology: None,
                    #[cfg(test)]
                    validation_hook: None,
                };
                shard::optimize::OptimizationStrategy::create_segment_builder(&adapter, inputs)
            }
            fn create_temp_segment(
                &self,
            ) -> OperationResult<(LockedSegment, shard::segment_manifest::NewSegmentToken)>
            {
                panic!("fixture has a Plain buffer")
            }
            fn live_vector_names(&self) -> Option<HashSet<String>> {
                None
            }
        }
        let _scope = common::ambient::test_guard();
        for cancel in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let staging = tempfile::tempdir().unwrap();
            let level = trained(
                root.path(),
                &[
                    (11, [1., 0.]),
                    (12, [0., 1.]),
                    (13, [-1., 0.]),
                    (14, [0., -1.]),
                ],
                17,
            );
            let buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
            let config = Arc::new(
                DliResearchConfig::new(vec![(level.uuid, 1), (buffer.uuid, 0)], 2, 100).unwrap(),
            );
            config
                .materialize_topk
                .store(true, std::sync::atomic::Ordering::Relaxed);
            let mut native = SegmentHolder::default();
            let expected = level.uuid;
            let validated = Arc::new(AtomicBool::new(false));
            let input = native.add_new(level);
            native.add_new(buffer);
            let holder = LockedSegmentHolder::new(native);
            let (entered_tx, entered_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel();
            let strategy = Paused {
                path: staging.path().to_owned(),
                expected,
                validated: validated.clone(),
                entered: entered_tx,
                resume: parking_lot::Mutex::new(resume_rx),
            };
            let paths = shard::optimize::OptimizationPaths {
                segments_path: root.path().to_owned(),
                temp_path: staging.path().to_owned(),
            };
            let stop = Arc::new(AtomicBool::new(false));
            let worker_holder = holder.clone();
            let worker_stop = stop.clone();
            let (done_tx, done_rx) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                let _scope = common::ambient::test_guard();
                let budget = ResourceBudget::new(1, 1);
                let result = shard::optimize::execute_optimization(
                    "dli_dispatch",
                    worker_holder,
                    vec![input],
                    Uuid::new_v4(),
                    None,
                    &paths,
                    budget.try_acquire(0, 1).unwrap(),
                    budget,
                    &worker_stop,
                    ProgressTracker::new_for_test(),
                    &segment::common::operation_time_statistics::OperationDurationsAggregator::new(
                    ),
                    &strategy,
                    Box::new(|| {}),
                );
                done_tx.send(result).unwrap();
            });
            entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            assert!(validated.load(std::sync::atomic::Ordering::Relaxed));
            assert!(matches!(
                holder.read().get(input),
                Some(LockedSegment::Proxy(_))
            ));
            let run = || {
                let holder = holder.clone();
                let config = config.clone();
                async move {
                    SegmentsSearcher::search_dli_research(
                        holder,
                        Arc::new(CoreSearchRequestBatch {
                            searches: vec![CoreSearchRequest {
                                query: vec![1., 0.].into(),
                                filter: None,
                                params: None,
                                limit: 10,
                                offset: 0,
                                with_payload: None,
                                with_vector: None,
                                score_threshold: None,
                            }],
                        }),
                        &AdaptiveSearchHandle::current_for_tests(),
                        QueryContext::new(
                            DEFAULT_INDEXING_THRESHOLD_KB,
                            Handoff::Measured(AmbientContext::new()),
                        ),
                        Duration::from_secs(10),
                        config,
                    )
                    .await
                    .unwrap()
                }
            };
            let before = run().await;
            assert_eq!(before[0].len(), 4);
            assert_eq!(config.fallback_reasons.lock().get("proxy"), Some(&1));
            {
                let _updates = holder.acquire_updates_lock();
                let guard = holder.read();
                let points = [
                    PointStructPersisted {
                        id: 11_u64.into(),
                        vector: VectorStructPersisted::Single(vec![0., 1.]),
                        payload: None,
                    },
                    PointStructPersisted {
                        id: 99_u64.into(),
                        vector: VectorStructPersisted::Single(vec![1., 0.]),
                        payload: None,
                    },
                ];
                upsert_points(&guard, 100, &points, None).unwrap();
                delete_points(&guard, 101, &[12_u64.into()]).unwrap();
            }
            if cancel {
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            resume_tx.send(()).unwrap();
            let completed = done_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            if cancel {
                assert!(matches!(completed, Err(OperationError::Cancelled { .. })));
            } else {
                completed.unwrap();
            }
            worker.join().unwrap();
            let after = run().await;
            let expected = HashMap::from([(11_u64, 0.0_f32), (13, -1.0), (14, 0.0), (99, 1.0)]);
            assert_eq!(after[0].len(), expected.len());
            let mut ids = HashSet::new();
            for hit in &after[0] {
                let PointIdType::NumId(id) = hit.id else {
                    panic!("numeric fixture")
                };
                assert!(ids.insert(id));
                assert!((hit.score - expected[&id]).abs() < 1e-5);
            }
        }
    }

    fn trained(root: &std::path::Path, rows: &[(u64, [f32; 2])], seed: u64) -> Segment {
        let staging = tempfile::tempdir().unwrap();
        let mut src = build_simple_segment(root, 2, Distance::Cosine).unwrap();
        for (id, vector) in rows {
            src.upsert_point(*id, (*id).into(), only_default_vector(vector))
                .unwrap();
        }
        let mut config = src.config().clone();
        config
            .vector_data
            .get_mut(DEFAULT_VECTOR_NAME)
            .unwrap()
            .index = Indexes::LmiTrained(LmiConfig {
            n_buckets: 2,
            sample_size: 4,
            hidden_dim: 4,
            epochs: 1,
            batch_size: 4,
            routing_batch_size: 4,
            kmeans_iterations: 2,
            nprobe: 2,
            seed,
        });
        let mut builder = SegmentBuilder::new(
            staging.path(),
            &config,
            &HnswGlobalConfig::default(),
            FeatureFlags::default(),
        )
        .unwrap();
        builder.update(&[&src], &AtomicBool::new(false)).unwrap();
        builder.build_for_test(root)
    }

    #[test]
    fn unselected_newer_occurrence_requires_fallback() {
        let _scope = common::ambient::unmeasured_guard(common::reason::reason("DLI version check"));
        let root = tempfile::tempdir().unwrap();
        let rows = [
            (41, [1., 0.]),
            (42, [0.8, 0.6]),
            (43, [0., 1.]),
            (44, [-0.6, 0.8]),
            (51, [-1., 0.]),
            (52, [-0.8, -0.6]),
            (53, [0., -1.]),
            (54, [0.6, -0.8]),
        ];
        let mut a = trained(root.path(), &rows[..4], 17);
        let b = trained(root.path(), &rows[4..], 29);
        let mut buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        buffer
            .upsert_point(100, 99_u64.into(), only_default_vector(&[0., 1.]))
            .unwrap();
        let context = VectorQueryContext::default();
        let initial = route_pinned(
            &[(1, &a), (2, &b), (0, &buffer)],
            &[1., 0.],
            100,
            1,
            &context,
        )
        .unwrap();
        assert!(initial.plan.buckets.iter().all(|bucket| bucket.level != 0));
        let old = initial
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits)
            .next()
            .unwrap()
            .clone();
        assert!(matches!(
            resolve_pinned_candidates(&[(1, &a), (2, &b), (0, &buffer)], initial, 10).unwrap(),
            CandidateResolution::Resolved(_)
        ));
        // Simulate retained physical preimage and a newer visible native copy.
        // This is a constructed holder state, not a claim about a completed upsert.
        buffer
            .upsert_point(1000, old.id, only_default_vector(&[0., 1.]))
            .unwrap();
        let members = [(1, &a), (2, &b), (0, &buffer)];
        let selected = route_pinned(&members, &[1., 0.], 100, 1, &context).unwrap();
        assert!(selected.plan.buckets.iter().all(|bucket| bucket.level != 0));
        assert!(
            selected
                .per_segment
                .iter()
                .flat_map(|(_, hits)| hits)
                .any(|h| h.id == old.id && h.version == old.version)
        );
        let mut naive = BatchResultAggregator::new([10]);
        naive.update_point_versions(selected.per_segment.iter().flat_map(|(_, hits)| hits));
        naive.update_batch_results(
            0,
            selected
                .per_segment
                .iter()
                .flat_map(|(_, hits)| hits.iter().cloned()),
        );
        assert!(
            naive.into_topk()[0]
                .iter()
                .any(|h| h.id == old.id && h.version == old.version)
        );
        assert!(matches!(
            resolve_pinned_candidates(&members, selected, 10).unwrap(),
            CandidateResolution::ExactFallback(ResolutionFallback::AmbiguousOccurrence)
        ));
        assert!(matches!(
            route_resolved_pinned(&members, &[1., 0.], 100, 1, 10, &context).unwrap(),
            CandidateResolution::ExactFallback(ResolutionFallback::AmbiguousOccurrence)
        ));
        let all = route_pinned(&members, &[1., 0.], 100, 1000, &context).unwrap();
        let newest = all
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits)
            .find(|hit| hit.id == old.id && hit.version == 1000)
            .unwrap();
        assert!(newest.score.abs() < 1e-5);
        a.delete_point(2000, 41_u64.into()).unwrap();
        let members = [(1, &a), (2, &b), (0, &buffer)];
        let selected = route_pinned(&members, &[1., 0.], 100, 1, &context).unwrap();
        assert!(matches!(
            resolve_pinned_candidates(&members, selected, 10).unwrap(),
            CandidateResolution::ExactFallback(ResolutionFallback::AmbiguousOccurrence)
        ));
    }

    #[test]
    fn dli_topk_adversarial_deleted_shadowed_and_deferred() {
        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let rows = [
            (1, [1., 0.]),
            (2, [0.99, 0.14106736]),
            (3, [0.95, 0.3122499]),
            (4, [0.8, 0.6]),
            (5, [0.6, 0.8]),
            (6, [0., 1.]),
            (7, [-0.6, 0.8]),
            (8, [-1., 0.]),
        ];
        let mut learned = trained(root.path(), &rows, 73);
        let mut buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        let ctx = VectorQueryContext::default();
        // More high scoring deletions than k; deletion filtering must precede top-k.
        for id in [1u64, 2, 3, 4] {
            learned.delete_point(100 + id, id.into()).unwrap();
        }
        for top in [usize::MAX, 3] {
            let members = [(1, &learned), (0, &buffer)];
            let routed = route_pinned_with_policy(
                &members,
                &[1., 0.],
                2,
                100,
                &ctx,
                RoutingPolicy::SlotWeighted,
                top,
            )
            .unwrap();
            let CandidateResolution::Resolved(hits) =
                resolve_pinned_candidates(&members, routed, 3).unwrap()
            else {
                panic!("deleted offsets are not visible candidates")
            };
            assert_eq!(
                hits.iter().map(|h| h.id).collect::<Vec<_>>(),
                vec![5u64.into(), 6u64.into(), 7u64.into()]
            );
            assert!((hits[0].score - 0.6).abs() < 1e-4);
        }
        // Upsert/delete-reinsert goes to Plain after old mapping is deleted.
        buffer
            .upsert_point(200, 1u64.into(), only_default_vector(&[0., -1.]))
            .unwrap();
        // Retained live preimage plus newer Plain occurrence: never trust selected-only versions.
        buffer
            .upsert_point(201, 5u64.into(), only_default_vector(&[-1., 0.]))
            .unwrap();
        for top in [usize::MAX, 3] {
            let members = [(1, &learned), (0, &buffer)];
            let routed = route_pinned_with_policy(
                &members,
                &[1., 0.],
                100,
                1,
                &ctx,
                RoutingPolicy::SlotWeighted,
                top,
            )
            .unwrap();
            assert!(matches!(
                resolve_pinned_candidates(&members, routed, 3).unwrap(),
                CandidateResolution::ExactFallback(ResolutionFallback::AmbiguousOccurrence)
            ));
        }
        let mut third = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        third
            .upsert_point(999, 5u64.into(), only_default_vector(&[0., 1.]))
            .unwrap();
        let members = [(1, &learned), (0, &buffer), (0, &third)];
        let routed = route_pinned_with_policy(
            &members,
            &[1., 0.],
            100,
            1,
            &ctx,
            RoutingPolicy::SlotWeighted,
            3,
        )
        .unwrap();
        assert!(matches!(
            resolve_pinned_candidates(&members, routed, 3).unwrap(),
            CandidateResolution::ExactFallback(ResolutionFallback::AmbiguousOccurrence)
        ));
        learned.delete_point(202, 5u64.into()).unwrap();
        for top in [usize::MAX, 3] {
            let members = [(1, &learned), (0, &buffer)];
            let routed = route_pinned_with_policy(
                &members,
                &[1., 0.],
                2,
                100,
                &ctx,
                RoutingPolicy::SlotWeighted,
                top,
            )
            .unwrap();
            let CandidateResolution::Resolved(hits) =
                resolve_pinned_candidates(&members, routed, 3).unwrap()
            else {
                panic!("reinsert resolved")
            };
            assert!(
                hits.iter()
                    .all(|h| ![2u64.into(), 3u64.into(), 4u64.into()].contains(&h.id))
            );
            for hit in hits {
                if hit.id == 5u64.into() {
                    assert_eq!(hit.version, 201);
                    assert!((hit.score + 1.).abs() < 1e-4);
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dli_topk_underfill_uses_complete_native_fallback() {
        use crate::collection_manager::segments_searcher::{DliResearchConfig, SegmentsSearcher};
        use crate::common::adaptive_handle::AdaptiveSearchHandle;
        use crate::operations::types::CoreSearchRequest;
        use common::ambient::{AmbientContext, Handoff};
        use segment::data_types::query_context::QueryContext;
        use shard::search::CoreSearchRequestBatch;
        use shard::segment_holder::{SegmentHolder, locked::LockedSegmentHolder};
        use std::sync::Arc;
        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let mut learned = trained(
            root.path(),
            &[
                (1, [1., 0.]),
                (2, [0.8, 0.6]),
                (3, [-0.8, 0.6]),
                (4, [-1., 0.]),
            ],
            73,
        );
        let mut plain = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        let initial = route_pinned(
            &[(1, &learned), (0, &plain)],
            &[1., 0.],
            100,
            1,
            &VectorQueryContext::default(),
        )
        .unwrap();
        for (_, hits) in initial.per_segment {
            for hit in hits {
                learned.delete_point(100, hit.id).unwrap();
            }
        }
        for (id, v) in [
            (99, [1., 0.]),
            (98, [0.98, 0.19899748]),
            (97, [0.9, 0.4358899]),
        ] {
            plain
                .upsert_point(101, id.into(), only_default_vector(&v))
                .unwrap();
        }
        let config = Arc::new(
            DliResearchConfig::new(vec![(learned.uuid, 1), (plain.uuid, 0)], 100, 1).unwrap(),
        );
        config
            .materialize_topk
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let mut holder = SegmentHolder::default();
        holder.add_new(learned);
        holder.add_new(plain);
        let result = SegmentsSearcher::search_dli_research(
            LockedSegmentHolder::new(holder),
            Arc::new(CoreSearchRequestBatch {
                searches: vec![CoreSearchRequest {
                    query: vec![1., 0.].into(),
                    filter: None,
                    params: None,
                    limit: 3,
                    offset: 0,
                    with_payload: None,
                    with_vector: None,
                    score_threshold: None,
                }],
            }),
            &AdaptiveSearchHandle::current_for_tests(),
            QueryContext::new(
                shard::optimizers::config::DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::new()),
            ),
            std::time::Duration::from_secs(10),
            config.clone(),
        )
        .await
        .unwrap();
        assert_eq!(
            result[0].iter().map(|h| h.id).collect::<Vec<_>>(),
            vec![99u64.into(), 98u64.into(), 97u64.into()]
        );
        assert_eq!(
            config
                .fallback_reasons
                .lock()
                .get("insufficient_valid_candidates"),
            Some(&1)
        );
    }

    #[test]
    fn dli_topk_deferred_member_remains_fail_closed() {
        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let base = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        let (mut deferred, _token) =
            segment::segment_constructor::build_segment(root.path(), base.config(), Some(1), true)
                .unwrap();
        deferred
            .upsert_point(1, 11u64.into(), only_default_vector(&[0., 1.]))
            .unwrap();
        deferred
            .upsert_point(2, 12u64.into(), only_default_vector(&[1., 0.]))
            .unwrap();
        assert!(deferred.has_deferred_points());
        for top in [usize::MAX, 1] {
            let members = [(0, &deferred)];
            let route = route_pinned_with_policy(
                &members,
                &[1., 0.],
                2,
                100,
                &VectorQueryContext::default(),
                RoutingPolicy::SlotWeighted,
                top,
            )
            .unwrap();
            assert!(matches!(
                resolve_pinned_candidates(&members, route, 1).unwrap(),
                CandidateResolution::ExactFallback(ResolutionFallback::DeferredState)
            ));
        }
    }

    #[test]
    fn dli_learned_resolution_survives_delete_and_reinsert() {
        let scorer_start = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
        let _scope =
            common::ambient::unmeasured_guard(common::reason::reason("DLI delete resolution"));
        use segment::entry::{NonAppendableSegmentEntry, SegmentEntry};
        let root = tempfile::tempdir().unwrap();
        let rows = [
            (41, [1., 0.]),
            (42, [0.8, 0.6]),
            (43, [0., 1.]),
            (44, [-0.6, 0.8]),
        ];
        let mut learned = trained(root.path(), &rows, 17);
        let mut buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        let context = VectorQueryContext::default();
        for deleted in [41_u64, 44] {
            learned
                .delete_point(2000 + deleted, deleted.into())
                .unwrap();
            let members = [(1, &learned), (0, &buffer)];
            let routed = route_pinned(&members, &[1., 0.], 2, 1000, &context).unwrap();
            let CandidateResolution::Resolved(hits) =
                resolve_pinned_candidates(&members, routed, 10).unwrap()
            else {
                panic!("unrelated deleted slots forced exact fallback")
            };
            for hit in &hits {
                assert!(hit.id != 41_u64.into());
                if deleted == 44 {
                    assert!(hit.id != 44_u64.into());
                }
                let expected = rows
                    .iter()
                    .find(|(id, _)| PointIdType::from(*id) == hit.id)
                    .unwrap()
                    .1[0];
                assert!((hit.score - expected).abs() < 1e-3);
            }
            assert_eq!(hits.len(), if deleted == 41 { 3 } else { 2 });
        }
        buffer
            .upsert_point(3000, 41_u64.into(), only_default_vector(&[0., 1.]))
            .unwrap();
        let members = [(1, &learned), (0, &buffer)];
        let routed = route_pinned(&members, &[1., 0.], 2, 1000, &context).unwrap();
        let CandidateResolution::Resolved(hits) =
            resolve_pinned_candidates(&members, routed, 10).unwrap()
        else {
            panic!("reinsert fallback")
        };
        assert_eq!(hits.len(), 3);
        let reinserted = hits.iter().find(|h| h.id == 41_u64.into()).unwrap();
        assert_eq!(reinserted.version, 3000);
        assert!(reinserted.score.abs() < 1e-5);
        if std::env::var_os("DLI_ASSERT_SCORING_FIXTURE").is_some() {
            let after = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
            assert_eq!(after[1] - scorer_start[1], 8);
            assert_eq!(after[2] - scorer_start[2], 8);
        }
    }

    #[tokio::test]
    async fn captured_native_exact_fallback_resolves_unselected_newer_copy() {
        use crate::collection_manager::fixtures::TEST_TIMEOUT;
        use crate::collection_manager::holders::segment_holder::{LockedSegment, SegmentHolder};
        use crate::collection_manager::segments_searcher::SegmentsSearcher;
        use crate::common::adaptive_handle::AdaptiveSearchHandle;
        use crate::operations::types::CoreSearchRequest;
        use common::ambient::{AmbientContext, Handoff};
        use segment::data_types::query_context::QueryContext;
        use shard::optimizers::config::DEFAULT_INDEXING_THRESHOLD_KB;
        use shard::search::CoreSearchRequestBatch;
        use shard::segment_holder::locked::LockedSegmentHolder;
        use std::sync::Arc;

        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let rows = [
            (41, [1., 0.]),
            (42, [0.8, 0.6]),
            (43, [0., 1.]),
            (44, [-0.6, 0.8]),
            (51, [-1., 0.]),
            (52, [-0.8, -0.6]),
            (53, [0., -1.]),
            (54, [0.6, -0.8]),
        ];
        let a = trained(root.path(), &rows[..4], 17);
        let b = trained(root.path(), &rows[4..], 29);
        let mut buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        buffer
            .upsert_point(100, 99_u64.into(), only_default_vector(&[0., 1.]))
            .unwrap();
        let context = VectorQueryContext::default();
        let route = route_pinned(
            &[(1, &a), (2, &b), (0, &buffer)],
            &[1., 0.],
            100,
            1,
            &context,
        )
        .unwrap();
        let old = route
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits)
            .next()
            .unwrap()
            .clone();
        buffer
            .upsert_point(1000, old.id, only_default_vector(&[0., 1.]))
            .unwrap();
        let uuids = (a.uuid, b.uuid, buffer.uuid);
        let mut holder = SegmentHolder::default();
        holder.add_new(a);
        holder.add_new(b);
        holder.add_new(buffer);
        let holder = LockedSegmentHolder::new(holder);
        let captured = SegmentsSearcher::capture_segments(&holder, TEST_TIMEOUT).unwrap();
        {
            let guards: Vec<_> = captured
                .iter()
                .map(|locked| match locked {
                    LockedSegment::Original(segment) => segment.read(),
                    LockedSegment::Proxy(_) => panic!("unexpected proxy"),
                })
                .collect();
            let members: Vec<_> = guards
                .iter()
                .map(|segment| {
                    let level = if segment.uuid == uuids.0 {
                        1
                    } else if segment.uuid == uuids.1 {
                        2
                    } else {
                        assert_eq!(segment.uuid, uuids.2);
                        0
                    };
                    (level, &**segment)
                })
                .collect();
            assert!(matches!(
                route_resolved_pinned(&members, &[1., 0.], 100, 1, 20, &context).unwrap(),
                CandidateResolution::ExactFallback(ResolutionFallback::AmbiguousOccurrence)
            ));
        }
        let request = CoreSearchRequest {
            query: vec![1., 0.].into(),
            with_payload: None,
            with_vector: None,
            filter: None,
            params: Some(segment::types::SearchParams {
                exact: true,
                ..Default::default()
            }),
            limit: 20,
            score_threshold: None,
            offset: 0,
        };
        // Production searcher research dispatch makes the same fallback
        // decision automatically on its captured holder membership.
        let config = Arc::new(
            crate::collection_manager::segments_searcher::DliResearchConfig::new(
                vec![(uuids.0, 1), (uuids.1, 2), (uuids.2, 0)],
                100,
                1,
            )
            .unwrap(),
        );
        let mut normal_request = request.clone();
        normal_request.params = None;
        let automatic = SegmentsSearcher::search_dli_research(
            holder.clone(),
            Arc::new(CoreSearchRequestBatch {
                searches: vec![normal_request],
            }),
            &AdaptiveSearchHandle::current_for_tests(),
            QueryContext::new(
                DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::new()),
            ),
            TEST_TIMEOUT,
            config.clone(),
        )
        .await
        .unwrap();
        assert_eq!(
            config
                .exact_fallbacks
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(config.fallback_reasons.lock().get("duplicate"), Some(&1));
        assert_eq!(
            config
                .learned_queries
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let result = SegmentsSearcher::search_captured(
            captured,
            Arc::new(CoreSearchRequestBatch {
                searches: vec![request],
            }),
            &AdaptiveSearchHandle::current_for_tests(),
            false,
            QueryContext::new(
                DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::new()),
            ),
            TEST_TIMEOUT,
        )
        .await
        .unwrap();
        let matches: Vec<_> = result[0]
            .iter()
            .filter(|point| point.id == old.id)
            .collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].version, 1000);
        assert!(matches[0].score.abs() < 1e-5);
        let mut seen = HashSet::new();
        assert_eq!(result[0].len(), 9);
        assert_eq!(automatic, result);
        for hit in &result[0] {
            assert!(seen.insert(hit.id));
            let expected = if hit.id == old.id || hit.id == 99_u64.into() {
                0.0
            } else {
                let PointIdType::NumId(id) = hit.id else {
                    panic!("numeric fixture")
                };
                rows.iter().find(|row| row.0 == id).unwrap().1[0]
            };
            assert!((hit.score - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn two_native_routers_global_plan_and_per_segment_scoring() {
        let _scope = common::ambient::unmeasured_guard(common::reason::reason("DLI native route"));
        let root = tempfile::tempdir().unwrap();
        let rows = [
            (11, [1., 0.]),
            (12, [0.8, 0.6]),
            (13, [0., 1.]),
            (14, [-0.6, 0.8]),
            (21, [-1., 0.]),
            (22, [-0.8, -0.6]),
            (23, [0., -1.]),
            (24, [0.6, -0.8]),
        ];
        let mut a = trained(root.path(), &rows[..4], 17);
        let mut b = trained(root.path(), &rows[4..], 29);
        let mut buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        a.delete_point(100, 11_u64.into()).unwrap();
        b.delete_point(101, 21_u64.into()).unwrap();
        buffer
            .upsert_point(102, 11_u64.into(), only_default_vector(&[0., 1.]))
            .unwrap();
        buffer
            .upsert_point(103, 31_u64.into(), only_default_vector(&[0.6, 0.8]))
            .unwrap();
        let members = [(1, &a), (2, &b), (0, &buffer)];
        let context = VectorQueryContext::default();
        let all = route_pinned(&members, &[1., 0.], 2, 100, &context).unwrap();
        assert_eq!(
            all.plan
                .buckets
                .iter()
                .map(|b| b.level)
                .collect::<HashSet<_>>(),
            HashSet::from([0, 1, 2])
        );
        let mut ids = HashSet::new();
        for (_, hits) in &all.per_segment {
            for hit in hits {
                let PointIdType::NumId(id) = hit.id else {
                    panic!("numeric fixture")
                };
                assert!(ids.insert(id));
                if id == 11 {
                    assert_eq!(hit.version, 102);
                }
                let expected = match id {
                    11 => 0.,
                    31 => 0.6,
                    21 => panic!("deleted"),
                    _ => rows.iter().find(|r| r.0 == id).unwrap().1[0],
                };
                assert!((hit.score - expected).abs() < 1e-5, "id={id}");
            }
        }
        assert_eq!(ids, HashSet::from([11, 12, 13, 14, 22, 23, 24, 31]));
        // Per-member top-k is sufficient only because every retained ID is
        // subsequently checked against the complete pinned membership. Any
        // ambiguous possible winner still invokes exact fallback.
        for limit in [1, 3, 100] {
            let complete = resolve_pinned_candidates(
                &members,
                route_pinned(&members, &[1., 0.], 2, 100, &context).unwrap(),
                limit,
            )
            .unwrap();
            let bounded = route_pinned_with_policy(
                &members,
                &[1., 0.],
                2,
                100,
                &context,
                RoutingPolicy::SlotWeighted,
                limit,
            )
            .unwrap();
            assert!(
                bounded
                    .per_segment
                    .iter()
                    .all(|(_, hits)| hits.len() <= limit)
            );
            let bounded = resolve_pinned_candidates(&members, bounded, limit).unwrap();
            match (complete, bounded) {
                (CandidateResolution::Resolved(a), CandidateResolution::Resolved(b)) => {
                    assert_eq!(a, b)
                }
                _ => panic!("unambiguous live oracle must resolve"),
            }
        }
        let limited = route_pinned(&members, &[1., 0.], 2, 2, &context).unwrap();
        assert!(limited.plan.estimated_candidates >= 2);
        assert!(
            limited.unique_posting_offsets + limited.buffer_point_estimate
                <= limited.plan.estimated_candidates
        );
        assert!(route_pinned(&[(1, &a), (2, &a)], &[1., 0.], 2, 2, &context).is_err());
        assert!(route_pinned(&[(1, &buffer)], &[1., 0.], 2, 2, &context).is_err());
        println!(
            "DLI_ROUTE budget=2 selected={} postings={} unique_offsets_plus_buffer_estimate={} returned={}",
            limited.plan.buckets.len(),
            limited.posting_candidates,
            limited.unique_posting_offsets + limited.buffer_point_estimate,
            limited
                .per_segment
                .iter()
                .map(|(_, h)| h.len())
                .sum::<usize>()
        );

        // Private native build from the three inputs; not optimizer publication.
        let staging = tempfile::tempdir().unwrap();
        let config = a.config().clone();
        let mut builder = SegmentBuilder::new(
            staging.path(),
            &config,
            &HnswGlobalConfig::default(),
            FeatureFlags::default(),
        )
        .unwrap();
        builder
            .update(&[&a, &b, &buffer], &AtomicBool::new(false))
            .unwrap();
        let replacement = builder.build_for_test(root.path());
        let rebuilt = route_pinned(&[(1, &replacement)], &[1., 0.], 2, 100, &context).unwrap();
        let rebuilt_ids: HashSet<_> = rebuilt
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits.iter().map(|h| h.id))
            .collect();
        assert_eq!(
            rebuilt_ids,
            ids.into_iter().map(PointIdType::from).collect()
        );
        let expected_scores: HashMap<_, _> = all
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits.iter().map(|h| (h.id, h.score)))
            .collect();
        for (_, hits) in &rebuilt.per_segment {
            for hit in hits {
                assert!((hit.score - expected_scores[&hit.id]).abs() < 1e-5);
            }
        }
        replacement.flush(true).unwrap();
        let path = replacement.segment_path.clone();
        let uuid = replacement.uuid;
        drop(replacement);
        let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
        let after = route_pinned(&[(1, &reopened)], &[1., 0.], 2, 100, &context).unwrap();
        assert_eq!(after.per_segment[0].1.len(), rebuilt_ids.len());
        for hit in &after.per_segment[0].1 {
            assert!((hit.score - expected_scores[&hit.id]).abs() < 1e-5);
        }

        // The count scheduler chooses level 2: include level 1 and the buffer,
        // but leave the separate level 3 untouched. This remains a private
        // build and does not model concurrent updates or publication.
        let plan = crate::collection_manager::dli_schedule::HierarchyConfig {
            policy: Default::default(),
            buffer_capacity: 2,
            arity: 2,
        }
        .plan(2, &[3, 0, 3])
        .unwrap()
        .unwrap();
        assert_eq!(
            (plan.target_level, &plan.source_levels, plan.output_points),
            (2, &vec![1], 5)
        );
        let staging = tempfile::tempdir().unwrap();
        let mut selective = SegmentBuilder::new(
            staging.path(),
            &config,
            &HnswGlobalConfig::default(),
            FeatureFlags::default(),
        )
        .unwrap();
        selective
            .update(&[&a, &buffer], &AtomicBool::new(false))
            .unwrap();
        let selective = selective.build_for_test(root.path());
        let subset = route_pinned(&[(2, &selective)], &[1., 0.], 2, 100, &context).unwrap();
        let subset_ids: HashSet<_> = subset
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits.iter().map(|h| h.id))
            .collect();
        assert_eq!(
            subset_ids,
            [11_u64, 12, 13, 14, 31]
                .into_iter()
                .map(PointIdType::from)
                .collect()
        );
        let untouched = route_pinned(&[(3, &b)], &[1., 0.], 2, 100, &context).unwrap();
        assert_eq!(untouched.per_segment[0].1.len(), 3);
    }
}
