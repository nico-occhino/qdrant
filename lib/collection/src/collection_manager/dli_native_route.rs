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
use segment::index::candidate_scoring::{score_candidates, score_candidates_with_quantization};
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::types::{Distance, PointIdType, ScoredPoint, WithPayload, WithVector};
use shard::search_result_aggregator::BatchResultAggregator;
use uuid::Uuid;

use super::dli_bucket_plan::{
    BucketPrediction, RoutingPolicy, SelectedBuckets, select, select_density,
    select_density_uncached, select_model_normalized, select_reserved,
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
    /// Unique selected learned offsets that are soft-deleted (point or vector) or out of
    /// range: they consume routing budget but are never scored.
    pub dead_posting_offsets: usize,
    /// Optional request-local native scorer measurements for learned members.
    pub learned_scoring: Option<LearnedScoringProfile>,
    /// Learned members scored with original vectors because quantized scoring
    /// was requested but their derived quantized artifact is absent.
    pub quantization_degraded_members: usize,
    /// Live point count, which can exceed the number of vectors in the buffer.
    pub buffer_point_estimate: usize,
    pub per_segment: Vec<(Uuid, Vec<ScoredPoint>)>,
}

#[derive(Default, Debug, serde::Serialize)]
pub struct LearnedScoringProfile {
    pub quantized_scored: usize,
    pub original_scored: usize,
    /// Successful full-dimensional scorer inputs from the native Plain buffer.
    pub plain_scored: usize,
    pub quantized_elapsed_ns: u128,
    pub original_elapsed_ns: u128,
    pub plain_elapsed_ns: u128,
    pub shortlist_offsets: usize,
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
    route_pinned_with_quantized_shortlist(members, query, arity, budget, context, policy, materialize_limit, None)
}

/// Explicit experimental two-stage mode. Missing quantization fails closed;
/// callers retain complete captured-set exact fallback and version resolution.
pub fn route_pinned_with_quantized_shortlist(
    members: &[(usize, &Segment)], query: &[f32], arity: usize, budget: usize,
    context: &VectorQueryContext, policy: RoutingPolicy, materialize_limit: usize,
    shortlist: Option<usize>,
) -> OperationResult<NativeRouteResult> {
    route_pinned_with_quantized_shortlist_profiled(
        members,
        query,
        arity,
        budget,
        context,
        policy,
        materialize_limit,
        shortlist,
        std::env::var_os("DLI_REQUEST_SCORING_PROFILE").is_some(),
    )
}

/// Caller-owned request-local measurements for one learned routing attempt.
///
/// The trace is filled incrementally while scoring runs, so work performed
/// before a routing error (for example, a Plain member scored before a learned
/// member reports unavailable quantization) is still attributed to the attempt.
/// The routing-plan event is captured in memory; persisting it is the caller's
/// responsibility, after segment read guards have been released.
#[derive(Default, Debug)]
pub struct RouteTrace {
    pub scoring: LearnedScoringProfile,
    pub routing_event: Option<serde_json::Value>,
}

pub fn route_pinned_with_quantized_shortlist_profiled(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
    context: &VectorQueryContext,
    policy: RoutingPolicy,
    materialize_limit: usize,
    shortlist: Option<usize>,
    profile_request: bool,
) -> OperationResult<NativeRouteResult> {
    if !profile_request {
        return route_pinned_traced(
            members,
            query,
            arity,
            budget,
            context,
            policy,
            materialize_limit,
            shortlist,
            None,
        );
    }
    // Direct (non-Collection) callers keep the historical behavior: the plan
    // event is appended to `DLI_ROUTING_EVENTS` and the scorer profile is
    // returned in the result.
    let mut trace = RouteTrace::default();
    let result = route_pinned_traced(
        members,
        query,
        arity,
        budget,
        context,
        policy,
        materialize_limit,
        shortlist,
        Some(&mut trace),
    );
    if let (Some(event), Some(path)) = (
        trace.routing_event.take(),
        std::env::var_os("DLI_ROUTING_EVENTS"),
    ) {
        append_jsonl_line(std::path::Path::new(&path), &event);
    }
    result.map(|mut routed| {
        routed.learned_scoring = Some(trace.scoring);
        routed
    })
}

/// Best-effort diagnostic append. Callers must not hold segment locks or be
/// inside a timed region when calling this.
///
/// The complete line is serialized first and emitted with one `write_all` on
/// an `O_APPEND` descriptor. `writeln!(file, "{value}")` formats through
/// `Display` and issues several `write(2)` calls, so concurrent appenders
/// interleaved fragments of different lines (observed as unparsable JSONL).
pub(crate) fn append_jsonl_line(path: &std::path::Path, value: &serde_json::Value) {
    use std::io::Write;
    let mut line = value.to_string();
    line.push('\n');
    if let Ok(mut file) = fs_err::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Routing and scoring with optional caller-owned tracing. When `trace` is
/// `Some`, scorer work is accumulated into it as it happens (so it survives an
/// `Err`), and `NativeRouteResult::learned_scoring` is left `None`.
#[allow(clippy::too_many_arguments)]
pub fn route_pinned_traced(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
    context: &VectorQueryContext,
    policy: RoutingPolicy,
    materialize_limit: usize,
    shortlist: Option<usize>,
    trace: Option<&mut RouteTrace>,
) -> OperationResult<NativeRouteResult> {
    route_pinned_selected(
        members, query, arity, budget, context, policy, None, materialize_limit, shortlist, trace,
    )
}

/// Calibrated level-aware allocation (Phase 3 Part B, E1L) on the same routing,
/// scoring and resolution path as every policy; only bucket selection differs.
#[allow(clippy::too_many_arguments)]
pub fn route_pinned_calibrated(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
    context: &VectorQueryContext,
    calibration: &super::dli_bucket_plan::LevelCalibration,
    materialize_limit: usize,
    trace: Option<&mut RouteTrace>,
) -> OperationResult<NativeRouteResult> {
    let selector = |predictions: Vec<BucketPrediction>| {
        super::dli_bucket_plan::select_calibrated(predictions, calibration, budget)
    };
    route_pinned_selected(
        members,
        query,
        arity,
        budget,
        context,
        RoutingPolicy::SlotWeighted,
        Some(&selector),
        materialize_limit,
        None,
        trace,
    )
}

type Selector<'a> =
    &'a dyn Fn(Vec<BucketPrediction>) -> Result<SelectedBuckets, super::dli_bucket_plan::PlanError>;

#[allow(clippy::too_many_arguments)]
fn route_pinned_selected(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
    context: &VectorQueryContext,
    policy: RoutingPolicy,
    selector: Option<Selector>,
    materialize_limit: usize,
    shortlist: Option<usize>,
    mut trace: Option<&mut RouteTrace>,
) -> OperationResult<NativeRouteResult> {
    let fail = |message: &str| OperationError::service_error(message.to_owned());
    let temperature = match policy {
        RoutingPolicy::TemperatureDensity { temperature_milli } => {
            if !(100..=10000).contains(&temperature_milli) {
                return Err(fail("invalid routing temperature"));
            }
            temperature_milli as f32 / 1000.
        }
        _ => 1.,
    };
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
        if config.multivector_config.is_some() {
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
                let denominator: f32 = logits.iter().map(|v| ((v - max) / temperature).exp()).sum();
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(fail("invalid LMI router probabilities"));
                }
                for (bucket, logit) in logits.iter().enumerate() {
                    predictions.push(BucketPrediction {
                        level,
                        physical: segment.uuid,
                        bucket: Some(bucket),
                        probability: ((logit - max) / temperature).exp() / denominator,
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
    let plan = match (selector, policy) {
        (Some(selector), _) => selector(predictions),
        (None, RoutingPolicy::SlotWeighted) => select(predictions, arity, budget),
        (None, RoutingPolicy::ReservedPerLevel) => select_reserved(predictions, arity, budget),
        (None, RoutingPolicy::PopulationDensity | RoutingPolicy::TemperatureDensity { .. }) => {
            select_density(predictions, arity, budget)
        }
        (None, RoutingPolicy::PopulationDensityUncached) => {
            select_density_uncached(predictions, arity, budget)
        }
        (None, RoutingPolicy::ModelDrivenNormalized) => {
            select_model_normalized(predictions, arity, budget, false)
        }
        (None, RoutingPolicy::ModelDrivenNormalizedCode) => {
            select_model_normalized(predictions, arity, budget, true)
        }
    }
    .map_err(|e| fail(&format!("invalid DLI candidate plan: {e:?}")))?;
    let bucket_planning_time = planning_started.elapsed();
    let routing_and_planning_time = start.elapsed();
    if let Some(trace) = trace.as_deref_mut() {
        // Captured only; never written while the caller's guards are held.
        trace.routing_event = Some(serde_json::json!({"policy":if selector.is_some() { serde_json::json!("calibrated") } else { serde_json::json!(policy) },"budget":budget,"buckets":plan.buckets,"members":members.iter().map(|(l,s)|(s.uuid,l,s.available_point_count())).collect::<Vec<_>>()}));
    }
    let mut learned_scoring = trace.map(|trace| &mut trace.scoring);

    let mut native_scorer_time = Duration::ZERO;
    let mut hit_materialization_time = Duration::ZERO;
    let started_scoring = Instant::now();
    let mut posting_candidates = 0usize;
    let mut unique_posting_offsets = 0usize;
    let mut dead_posting_offsets = 0usize;
    let mut buffer_point_estimate = 0usize;
    let mut quantization_degraded_members = 0usize;
    // Thread-local attribution is sound here because the whole route executes
    // synchronously on this one blocking task; the guard is dropped (and the
    // flag cleared) on every return path, including `?` and unwinding.
    let _task_scoring_profile = learned_scoring.is_some().then(|| {
        segment::index::hnsw_index::point_scorer::TaskScoringProfileGuard::begin()
    });
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
            let task_scored_before = learned_scoring
                .as_ref()
                .map(|_| segment::index::hnsw_index::point_scorer::scoring_task_success_snapshot());
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
            if let (Some(profile), Some(before)) =
                (learned_scoring.as_mut(), task_scored_before)
            {
                profile.plain_scored += usize::try_from(
                    segment::index::hnsw_index::point_scorer::scoring_task_success_snapshot()
                        .saturating_sub(before),
                )
                .unwrap_or(usize::MAX);
                profile.plain_elapsed_ns += score_started.elapsed().as_nanos();
            }
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
            {
                use segment::vector_storage::VectorStorageRead as _;
                let deleted = tracker.deleted_point_bitslice();
                let total = storage.total_vector_count();
                dead_posting_offsets += candidates
                    .iter()
                    .filter(|&&offset| {
                        let i = offset as usize;
                        i >= total
                            || deleted.get(i).map_or(true, |bit| *bit)
                            || storage.is_deleted_vector(offset)
                    })
                    .count();
            }
            let score_started = Instant::now();
            let rerank_candidates;
            let candidates = if let Some(shortlist) = shortlist {
                if shortlist == 0 || materialize_limit == usize::MAX {
                    return Err(fail("quantized mode requires bounded materialization and nonzero shortlist"));
                }
                let quantized_guard = data.quantized_vectors.borrow();
                match quantized_guard.as_ref() {
                // The derived compressed scorer is absent (never built for this
                // member, or dropped as missing/corrupt/incompatible at load).
                // The member's learned postings and original vectors were
                // validated independently, so original scoring of exactly the
                // same candidates is the A0 path: correct, and far cheaper than
                // complete exact fallback. Counted, never silent.
                None => {
                    quantization_degraded_members += 1;
                    &candidates
                }
                Some(quantized) => {
                let expected_path = segment::segment_constructor::get_vector_storage_path(&segment.segment_path, DEFAULT_VECTOR_NAME);
                // A present but foreign/incompatible object signals mis-wiring:
                // fail closed to complete exact fallback.
                if quantized.storage_path() != expected_path || quantized.config().vector_parameters.dim != query.len() {
                    return Err(fail("quantized physical generation mismatch"));
                }
                let approximate = if let Some(profile) = learned_scoring.as_mut() {
                    let tick = Instant::now();
                    let (hits, stats) = segment::index::candidate_scoring::score_candidates_quantized_profiled(
                        &query_vector, &candidates, shortlist.max(materialize_limit), &*tracker, &*storage, Some(quantized), context)?;
                    profile.quantized_elapsed_ns += tick.elapsed().as_nanos();
                    profile.quantized_scored += stats.scored_vectors;
                    profile.shortlist_offsets += hits.len();
                    hits
                } else { score_candidates_with_quantization(
                    &query_vector, &candidates, shortlist.max(materialize_limit),
                    &*tracker, &*storage, Some(quantized), context,
                )? };
                rerank_candidates = approximate.into_iter().map(|hit| hit.idx).collect::<Vec<_>>();
                &rerank_candidates
                }
                }
            } else { &candidates };
            let offsets = if let Some(profile) = learned_scoring.as_mut() {
                let tick = Instant::now();
                let (hits, stats) = segment::index::candidate_scoring::score_candidates_quantized_profiled(
                    &query_vector, candidates, candidates.len().min(materialize_limit), &*tracker, &*storage, None, context)?;
                profile.original_elapsed_ns += tick.elapsed().as_nanos();
                profile.original_scored += stats.scored_vectors;
                hits
            } else { score_candidates(
                &query_vector, candidates, candidates.len().min(materialize_limit), &tracker, &storage, context,
            )? };
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
        dead_posting_offsets,
        // Owned by the caller's trace when tracing; see `route_pinned_traced`.
        learned_scoring: None,
        quantization_degraded_members,
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
                        reuse_router: None,
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
            let level = trained_configured(
                root.path(),
                &[
                    (11, [1., 0.]),
                    (12, [0., 1.]),
                    (13, [-1., 0.]),
                    (14, [0., -1.]),
                ],
                17, true,
            );
            level.flush(true).unwrap();
            let path = level.segment_path.clone(); let uuid = level.uuid; drop(level);
            let level = segment::segment_constructor::load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
            let buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
            let config = Arc::new(
                DliResearchConfig::new(vec![(level.uuid, 1), (buffer.uuid, 0)], 2, 100).unwrap(),
            );
            config
                .materialize_topk
                .store(true, std::sync::atomic::Ordering::Relaxed);
            config.quantized_shortlist.store(64, std::sync::atomic::Ordering::Relaxed);
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
            let mut pinned_query = None;
            if !cancel {
                use crate::collection_manager::segments_searcher::DliGuardHook;
                let (capture_tx,capture_rx) = mpsc::channel();
                let (release_tx,release_rx) = mpsc::channel();
                *config.test_guard_hook.lock() = Some(DliGuardHook { entered: capture_tx, resume: parking_lot::Mutex::new(release_rx) });
                let qholder = holder.clone(); let qconfig = config.clone();
                let query = tokio::spawn(async move {
                    SegmentsSearcher::search_dli_research(qholder, Arc::new(CoreSearchRequestBatch {
                        searches: vec![CoreSearchRequest { query: vec![1.,0.].into(), filter: None, params: None, limit:4,offset:0,with_payload:None,with_vector:None,score_threshold:None }]
                    }), &AdaptiveSearchHandle::current_for_tests(), QueryContext::new(DEFAULT_INDEXING_THRESHOLD_KB,Handoff::Measured(AmbientContext::new())), Duration::from_secs(10), qconfig).await.unwrap()
                });
                capture_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                pinned_query = Some((release_tx,query));
            }
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
                                limit: 4,
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
            let mutate = || {
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
            };
            if cancel { mutate(); }
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
            if let Some((release,query)) = pinned_query {
                // Optimizer has completed its real holder swap while old handles stay pinned.
                release.send(()).unwrap();
                let old = query.await.unwrap();
                assert_eq!(old[0].len(),4);
                let winner = old[0].iter().find(|p|p.id==11_u64.into()).unwrap();
                assert_eq!(winner.score,1.0); assert_eq!(winner.version,11);
                *config.test_guard_hook.lock() = None;
                assert_eq!(config.learned_queries.load(std::sync::atomic::Ordering::Relaxed),1);
                mutate();
            }
            let after = run().await;
            if cancel {
                assert_eq!(config.learned_queries.load(std::sync::atomic::Ordering::Relaxed), 1, "cancelled optimizer must restore a usable quantized generation");
            }

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
        trained_configured(root, rows, seed, false)
    }

    fn trained_configured(root: &std::path::Path, rows: &[(u64, [f32; 2])], seed: u64, quantized: bool) -> Segment {
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
            reuse_router: None,
        });
        if quantized {
            config.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().quantization_config = Some(segment::types::ScalarQuantizationConfig {
                memory: None, r#type: Default::default(), quantile: None, always_ram: None,
            }.into());
        }
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
    fn quantized_learned_generation_reopens_and_degrades_missing_object() {
        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let a = trained_configured(root.path(), &[(11,[1.,0.]),(12,[0.8,0.6]),(13,[0.,1.]),(14,[-1.,0.])], 42, true);
        a.flush(true).unwrap();
        let path = a.segment_path.clone(); let uuid = a.uuid; drop(a);
        let mut a = segment::segment_constructor::load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
        assert!(a.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors.borrow().is_some());
        a.delete_point(100,11_u64.into()).unwrap();
        let mut buffer = build_simple_segment(root.path(),2,Distance::Cosine).unwrap();
        buffer.upsert_point(101,11_u64.into(),only_default_vector(&[0.,1.])).unwrap();
        let members = [(1,&a),(0,&buffer)]; let ctx = VectorQueryContext::default();
        let original = route_pinned_with_policy(&members,&[1.,0.],2,100,&ctx,RoutingPolicy::PopulationDensity,3).unwrap();
        let quant = route_pinned_with_quantized_shortlist(&members,&[1.,0.],2,100,&ctx,RoutingPolicy::PopulationDensity,3,Some(64)).unwrap();
        assert_eq!(original.per_segment,quant.per_segment);
        assert!(matches!(resolve_pinned_candidates(&members,quant,3).unwrap(),CandidateResolution::Resolved(_)));
        a.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors.borrow_mut().take();
        // Missing derived artifact: the member degrades to original scoring of
        // the same postings (A0), identical to the original route, and counted.
        let degraded = route_pinned_with_quantized_shortlist(&members,&[1.,0.],2,100,&ctx,RoutingPolicy::PopulationDensity,3,Some(64)).unwrap();
        assert_eq!(degraded.quantization_degraded_members, 1);
        assert_eq!(degraded.per_segment, original.per_segment);
        let quant_path = segment::segment_constructor::get_vector_storage_path(&a.segment_path, DEFAULT_VECTOR_NAME);
        let metadata = quant_path.join("quantized.config.json");
        let saved = std::fs::read(&metadata).unwrap();
        std::fs::write(&metadata, b"not-json").unwrap();
        let scalar: segment::types::QuantizationConfig = segment::types::ScalarQuantizationConfig {
            memory: None, r#type: Default::default(), quantile: None, always_ram: None,
        }.into();
        {
            let storage = a.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
            assert!(segment::vector_storage::quantized::quantized_vectors::QuantizedVectors::load(&scalar, &storage, &quant_path, &AtomicBool::new(false)).is_err());
        }
        std::fs::write(&metadata, saved).unwrap();
        a.flush(true).unwrap(); drop(a);
        std::fs::rename(&metadata, quant_path.join("quantized.config.saved")).unwrap();
        let reopened = segment::segment_constructor::load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
        assert!(reopened.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors.borrow().is_none());
        let refs = [(1,&reopened),(0,&buffer)];
        // Reopen without quantized metadata: the derived artifact is dropped at
        // load, and quantized mode degrades this member to original scoring
        // with results identical to the original route.
        let degraded = route_pinned_with_quantized_shortlist(&refs,&[1.,0.],2,100,&ctx,RoutingPolicy::PopulationDensity,3,Some(64)).unwrap();
        let original = route_pinned_with_policy(&refs,&[1.,0.],2,100,&ctx,RoutingPolicy::PopulationDensity,3).unwrap();
        assert_eq!(degraded.quantization_degraded_members, 1);
        assert_eq!(degraded.per_segment, original.per_segment);

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
        config
            .profile_scoring
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let profile_path = root.path().join("dli-exact-fallback-profile.jsonl");
        *config.profile_output_path.write() = Some(profile_path.clone());
        // Independent expectations: exact search scores every live vector of
        // every captured segment exactly once; deleted offsets are never scored.
        let learned_live = learned.available_point_count();
        let plain_live = plain.available_point_count();
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
        let events: Vec<serde_json::Value> = std::fs::read_to_string(profile_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        // Exactly one authoritative event per logical request.
        assert_eq!(events.len(), 1, "one final event per request: {events:?}");
        let event = &events[0];
        assert_eq!(event["event"], "dli_request");
        assert_eq!(event["final_outcome"], "exact_fallback");
        assert_eq!(event["learned_route_supplied_results"], false);
        assert_eq!(event["fallback_reason"], "insufficient_valid_candidates");
        // The learned attempt resolved candidates but could not fill the
        // result; it must not be labelled as a resolved (learned) result.
        assert_eq!(event["attempt"]["attempt_resolution"], "underfilled");
        // Partial learned work preceding fallback is retained and separated.
        // With budget 1 the slot-weighted plan (P * a^level, a = 100) selects a
        // learned bucket before the Plain buffer (priority 1.0) and stops, so the
        // buffer is never scored: that starvation is why this request underfills.
        let attempt = &event["learned_scoring"];
        assert_eq!(attempt["plain_scored"].as_u64().unwrap(), 0);
        // Every offset in the selected bucket was deleted above: postings still
        // consume routing budget, but deleted offsets are filtered before the
        // scorer, so successful original evaluations are exactly zero.
        assert!(event["attempt"]["unique_offsets"].as_u64().unwrap() >= 1);
        assert_eq!(
            event["attempt"]["dead_offsets"], event["attempt"]["unique_offsets"],
            "every selected posting is dead"
        );
        assert_eq!(attempt["original_scored"].as_u64().unwrap(), 0);
        assert_eq!(attempt["quantized_scored"].as_u64().unwrap(), 0);
        let exact = &event["exact_fallback"];
        assert_eq!(
            exact["scored_vectors"].as_u64().unwrap(),
            (learned_live + plain_live) as u64,
            "exact fallback must score each live captured vector exactly once"
        );
        assert!(exact["elapsed_ns"].as_u64().unwrap() > 0);
        assert!(
            event["dli_shard_search_elapsed_ns"].as_u64().unwrap()
                >= exact["elapsed_ns"].as_u64().unwrap()
        );
        assert_eq!(event["returned_results"], 3);
    }

    /// Shared fixture for request-event tests: one trained learned member and
    /// one Plain member, both with known live counts and no deletions.
    fn request_event_fixture(
        root: &std::path::Path,
    ) -> (Segment, Segment) {
        let learned = trained(
            root,
            &[
                (1, [1., 0.]),
                (2, [0.8, 0.6]),
                (3, [-0.8, 0.6]),
                (4, [-1., 0.]),
                (5, [0.6, -0.8]),
                (6, [0., 1.]),
            ],
            91,
        );
        let mut plain = build_simple_segment(root, 2, Distance::Cosine).unwrap();
        for (id, v) in [(90, [0.9, 0.4358899]), (91, [-0.2, 0.9797959])] {
            plain
                .upsert_point(101, id.into(), only_default_vector(&v))
                .unwrap();
        }
        (learned, plain)
    }

    async fn run_request_event_query(
        holder: shard::segment_holder::locked::LockedSegmentHolder,
        config: std::sync::Arc<crate::collection_manager::segments_searcher::DliResearchConfig>,
        filter: Option<segment::types::Filter>,
        limit: usize,
    ) -> crate::operations::types::CollectionResult<Vec<Vec<segment::types::ScoredPoint>>> {
        use crate::collection_manager::segments_searcher::SegmentsSearcher;
        use crate::common::adaptive_handle::AdaptiveSearchHandle;
        use crate::operations::types::CoreSearchRequest;
        use common::ambient::{AmbientContext, Handoff};
        use segment::data_types::query_context::QueryContext;
        use shard::search::CoreSearchRequestBatch;
        SegmentsSearcher::search_dli_research(
            holder,
            std::sync::Arc::new(CoreSearchRequestBatch {
                searches: vec![CoreSearchRequest {
                    query: vec![1., 0.].into(),
                    filter,
                    params: None,
                    limit,
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
            config,
        )
        .await
    }

    fn read_events(path: &std::path::Path) -> Vec<serde_json::Value> {
        fs_err::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    /// Successful learned requests, run concurrently: each request's single
    /// event must carry exactly its own Plain and learned scorer work. The
    /// expected counts are derived from the fixture, not from the counters.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dli_request_events_are_final_and_request_local_under_concurrency() {
        use crate::collection_manager::segments_searcher::DliResearchConfig;
        use shard::segment_holder::{SegmentHolder, locked::LockedSegmentHolder};
        use std::sync::Arc;
        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let (learned, plain) = request_event_fixture(root.path());
        let learned_live = learned.available_point_count();
        let plain_live = plain.available_point_count();
        // Budget covers every bucket, so every live learned vector is a candidate.
        let config = Arc::new(
            DliResearchConfig::new(vec![(learned.uuid, 1), (plain.uuid, 0)], 2, 1_000).unwrap(),
        );
        config.profile_scoring.store(true, std::sync::atomic::Ordering::Relaxed);
        let profile_path = root.path().join("events.jsonl");
        let routing_path = root.path().join("routing.jsonl");
        *config.profile_output_path.write() = Some(profile_path.clone());
        *config.routing_output_path.write() = Some(routing_path.clone());
        let mut holder = SegmentHolder::default();
        holder.add_new(learned);
        holder.add_new(plain);
        let holder = LockedSegmentHolder::new(holder);
        const REQUESTS: usize = 16;
        let mut tasks = Vec::new();
        for _ in 0..REQUESTS {
            let (holder, config) = (holder.clone(), config.clone());
            tasks.push(tokio::spawn(async move {
                run_request_event_query(holder, config, None, 3).await.unwrap()
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap()[0].len(), 3);
        }
        let events = read_events(&profile_path);
        assert_eq!(events.len(), REQUESTS, "exactly one event per request");
        let ids: std::collections::HashSet<_> =
            events.iter().map(|e| e["request_id"].to_string()).collect();
        assert_eq!(ids.len(), REQUESTS, "request ids are unique");
        assert_eq!(read_events(&routing_path).len(), REQUESTS);
        for event in &events {
            assert_eq!(event["final_outcome"], "learned");
            assert_eq!(event["learned_route_supplied_results"], true);
            assert!(event["fallback_reason"].is_null());
            assert!(event["exact_fallback"].is_null());
            assert_eq!(event["attempt"]["attempt_resolution"], "resolved");
            let scoring = &event["learned_scoring"];
            assert_eq!(scoring["plain_scored"].as_u64().unwrap(), plain_live as u64);
            assert_eq!(scoring["original_scored"].as_u64().unwrap(), learned_live as u64);
            assert_eq!(scoring["quantized_scored"].as_u64().unwrap(), 0);
            assert_eq!(
                event["attempt"]["unique_offsets"].as_u64().unwrap(),
                learned_live as u64
            );
            assert_eq!(event["returned_results"], 3);
        }
        assert_eq!(
            config.learned_queries.load(std::sync::atomic::Ordering::Relaxed),
            REQUESTS as u64
        );
    }

    /// Early rejection (unsupported filter) produces one exact-fallback event
    /// with the precise reason and exact work equal to the filtered population;
    /// unavailable derived quantization degrades one member to original scoring.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dli_request_events_cover_rejection_and_degraded_quantization() {
        use crate::collection_manager::segments_searcher::DliResearchConfig;
        use shard::segment_holder::{SegmentHolder, locked::LockedSegmentHolder};
        use std::sync::Arc;
        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let (learned, plain) = request_event_fixture(root.path());
        let learned_live = learned.available_point_count();
        let plain_live = plain.available_point_count();
        let config = Arc::new(
            DliResearchConfig::new(vec![(learned.uuid, 1), (plain.uuid, 0)], 2, 1_000).unwrap(),
        );
        config.profile_scoring.store(true, std::sync::atomic::Ordering::Relaxed);
        config.materialize_topk.store(true, std::sync::atomic::Ordering::Relaxed);
        let profile_path = root.path().join("events.jsonl");
        *config.profile_output_path.write() = Some(profile_path.clone());
        let mut holder = SegmentHolder::default();
        holder.add_new(learned);
        holder.add_new(plain);
        let holder = LockedSegmentHolder::new(holder);

        // 1. Early rejection before any learned work.
        let filter = segment::types::Filter::new_must(segment::types::Condition::HasId(
            segment::types::HasIdCondition::from(ahash::AHashSet::from([1_u64.into()])),
        ));
        let result = run_request_event_query(holder.clone(), config.clone(), Some(filter), 3)
            .await
            .unwrap();
        assert_eq!(result[0].len(), 1);
        // 2. Quantized scoring requested, but no quantized generation exists.
        config.quantized_shortlist.store(4, std::sync::atomic::Ordering::Relaxed);
        let result = run_request_event_query(holder.clone(), config.clone(), None, 3)
            .await
            .unwrap();
        assert_eq!(result[0].len(), 3);

        let events = read_events(&profile_path);
        assert_eq!(events.len(), 2, "{events:?}");
        let rejected = &events[0];
        assert_eq!(rejected["final_outcome"], "exact_fallback");
        assert_eq!(rejected["fallback_reason"], "unsupported_query");
        assert!(rejected["attempt"].is_null());
        assert!(rejected["learned_scoring"].is_null());
        // The HasId filter admits one point; exact search scores only it.
        assert_eq!(rejected["exact_fallback"]["scored_vectors"].as_u64().unwrap(), 1);

        // A missing derived quantized artifact no longer forces complete exact
        // fallback: the member is scored with original vectors (the A0 path)
        // over the same postings, and the degradation is counted explicitly.
        let degraded = &events[1];
        assert_eq!(degraded["final_outcome"], "learned");
        assert!(degraded["fallback_reason"].is_null());
        assert!(degraded["exact_fallback"].is_null());
        assert_eq!(degraded["attempt"]["quantization_degraded_members"], 1);
        assert_eq!(degraded["learned_scoring"]["quantized_scored"].as_u64().unwrap(), 0);
        assert_eq!(
            degraded["learned_scoring"]["original_scored"].as_u64().unwrap(),
            learned_live as u64
        );
        assert_eq!(
            degraded["learned_scoring"]["plain_scored"].as_u64().unwrap(),
            plain_live as u64
        );
        assert_ne!(rejected["request_id"], degraded["request_id"]);
        let reasons = config.fallback_reasons.lock();
        assert_eq!(reasons.get("unsupported_query"), Some(&1));
        assert_eq!(reasons.len(), 1, "{reasons:?}");
        assert_eq!(
            config
                .quantization_degraded_queries
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    /// With profiling disabled (the latency-measurement configuration), no
    /// diagnostic file is created and no scorer profile is collected.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dli_request_events_absent_when_profiling_disabled() {
        use crate::collection_manager::segments_searcher::DliResearchConfig;
        use shard::segment_holder::{SegmentHolder, locked::LockedSegmentHolder};
        use std::sync::Arc;
        let _scope = common::ambient::test_guard();
        let root = tempfile::tempdir().unwrap();
        let (learned, plain) = request_event_fixture(root.path());
        let config = Arc::new(
            DliResearchConfig::new(vec![(learned.uuid, 1), (plain.uuid, 0)], 2, 1_000).unwrap(),
        );
        config.profile_scoring.store(false, std::sync::atomic::Ordering::Relaxed);
        let profile_path = root.path().join("events.jsonl");
        let routing_path = root.path().join("routing.jsonl");
        *config.profile_output_path.write() = Some(profile_path.clone());
        *config.routing_output_path.write() = Some(routing_path.clone());
        let mut holder = SegmentHolder::default();
        holder.add_new(learned);
        holder.add_new(plain);
        let result = run_request_event_query(LockedSegmentHolder::new(holder), config.clone(), None, 3)
            .await
            .unwrap();
        assert_eq!(result[0].len(), 3);
        assert_eq!(config.learned_queries.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(!profile_path.exists());
        assert!(!routing_path.exists());
    }

    #[test]
    fn invalid_quantized_shortlist_environment_selects_original_scoring() {
        use crate::collection_manager::segments_searcher::parse_env_quantized_shortlist as parse;
        assert_eq!(parse(Err(std::env::VarError::NotPresent)), None);
        assert_eq!(parse(Ok("128".into())), Some(128));
        assert_eq!(parse(Ok(" 64 ".into())), Some(64));
        // Previously these became Some(0), which forced every query into
        // exact fallback with a misleading "quantization_configuration" reason.
        assert_eq!(parse(Ok("0".into())), None);
        assert_eq!(parse(Ok("abc".into())), None);
        assert_eq!(parse(Ok("-1".into())), None);
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
            // Budget 1000 selects every learned posting; soft-deleted ones still
            // consume budget and are counted as dead.
            assert_eq!(routed.unique_posting_offsets, rows.len());
            let dead = if deleted == 41 { 1 } else { 2 };
            assert_eq!(routed.dead_posting_offsets, dead);
            let CandidateResolution::Resolved(hits) =
                resolve_pinned_candidates(&members, routed, 10).unwrap()
            else {
                panic!("unrelated deleted slots forced exact fallback")
            };
            // Independent check: live selected postings are exactly the scored hits.
            assert_eq!(hits.len(), rows.len() - dead);
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
        for segment in [&a, &b] {
            use segment::vector_storage::quantized::quantized_vectors::{QuantizedVectors, QuantizedVectorsStorageType};
            let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
            let storage = data.vector_storage.borrow();
            let config: segment::types::QuantizationConfig = segment::types::ScalarQuantizationConfig {
                memory: None, r#type: Default::default(), quantile: None, always_ram: None,
            }.into();
            let path = segment::segment_constructor::get_vector_storage_path(&segment.segment_path, DEFAULT_VECTOR_NAME);
            std::fs::create_dir_all(&path).unwrap();
            let quant = QuantizedVectors::create(&*storage, &config, QuantizedVectorsStorageType::Immutable,
                &path, 1, &std::sync::atomic::AtomicBool::new(false)).unwrap();
            *data.quantized_vectors.borrow_mut() = Some(quant);
        }
        let members = [(1, &a), (2, &b), (0, &buffer)];
        let context = VectorQueryContext::default();
        let quantized = route_pinned_with_quantized_shortlist_profiled(
            &members, &[1.,0.], 2, 100, &context, RoutingPolicy::PopulationDensity, 30, Some(30), true,
        ).unwrap();
        let original = route_pinned_with_quantized_shortlist_profiled(
            &members, &[1.,0.], 2, 100, &context, RoutingPolicy::PopulationDensity, 30, None, true,
        ).unwrap();
        // Profiling was requested explicitly; a missing profile is a failure,
        // never a reason to skip the conservation assertions.
        let q = quantized.learned_scoring.as_ref().expect("profiled quantized route");
        let o = original.learned_scoring.as_ref().expect("profiled original route");
        {
            assert_eq!(q.quantized_scored, 6);
            assert_eq!(q.original_scored, 6);
            assert_eq!(q.shortlist_offsets, 6);
            assert_eq!(q.plain_scored, 2, "request-local profile includes successful Plain scorer inputs");
            assert!(q.plain_elapsed_ns > 0);
            assert_eq!(o.quantized_scored, 0);
            assert_eq!(o.original_scored, 6);
            assert_eq!(o.plain_scored, 2);
        }
        assert_eq!(quantized.per_segment, original.per_segment);
        assert!(matches!(resolve_pinned_candidates(&members, quantized, 30).unwrap(), CandidateResolution::Resolved(_)));
        // Same dimension/population is insufficient: another physical generation must fail closed.
        {
            let mut qa = a.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors.borrow_mut();
            let mut qb = b.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors.borrow_mut();
            std::mem::swap(&mut *qa, &mut *qb);
        }
        assert!(route_pinned_with_quantized_shortlist(&members, &[1.,0.], 2, 100,
            &context, RoutingPolicy::PopulationDensity, 30, Some(30)).is_err());
        {
            let mut qa = a.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors.borrow_mut();
            let mut qb = b.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors.borrow_mut();
            std::mem::swap(&mut *qa, &mut *qb);
        }
        let saved = a.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors.borrow_mut().take();
        // Absent (not foreign) artifact: only member `a` degrades to original
        // scoring; `b` keeps quantized scoring; results stay identical.
        let degraded = route_pinned_with_quantized_shortlist(&members, &[1.,0.], 2, 100,
            &context, RoutingPolicy::PopulationDensity, 30, Some(30)).unwrap();
        assert_eq!(degraded.quantization_degraded_members, 1);
        assert_eq!(degraded.per_segment, original.per_segment);
        *a.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors.borrow_mut() = saved;
        let density = route_pinned_with_policy(
            &members,
            &[1., 0.],
            2,
            100,
            &context,
            RoutingPolicy::PopulationDensity,
            30,
        )
        .unwrap();
        let uncached = route_pinned_with_policy(
            &members,
            &[1., 0.],
            2,
            100,
            &context,
            RoutingPolicy::PopulationDensityUncached,
            30,
        )
        .unwrap();
        let t1 = route_pinned_with_policy(
            &members,
            &[1., 0.],
            2,
            100,
            &context,
            RoutingPolicy::TemperatureDensity {
                temperature_milli: 1000,
            },
            30,
        )
        .unwrap();
        assert_eq!(density.plan, uncached.plan);
        assert_eq!(density.plan, t1.plan);
        assert!(
            route_pinned_with_policy(
                &members,
                &[1., 0.],
                2,
                100,
                &context,
                RoutingPolicy::TemperatureDensity {
                    temperature_milli: 0
                },
                30
            )
            .is_err()
        );
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
