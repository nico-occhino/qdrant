//! Experimental DLI research dispatch over captured segment handles (`dli-research`),
//! plus the request-local profile used by the native exact fallback.

#[cfg(feature = "dli-research")]
use std::sync::Arc;
#[cfg(feature = "dli-research")]
use std::time::{Duration, Instant};

#[cfg(feature = "dli-research")]
use segment::data_types::query_context::QueryContext;
#[cfg(feature = "dli-research")]
use segment::types::ScoredPoint;
#[cfg(feature = "dli-research")]
use shard::search::CoreSearchRequestBatch;
#[cfg(feature = "dli-research")]
use shard::segment_holder::locked::LockedSegmentHolder;
#[cfg(feature = "dli-research")]
use tokio_util::task::AbortOnDropHandle;

use super::SegmentsSearcher;
#[cfg(feature = "dli-research")]
use crate::collection_manager::holders::segment_holder::LockedSegment;
#[cfg(feature = "dli-research")]
use crate::common::adaptive_handle::AdaptiveSearchHandle;
#[cfg(feature = "dli-research")]
use crate::operations::types::{CollectionError, CollectionResult};

/// Request-local counters for the native exact search that services an unsafe
/// DLI query. Worker tasks add successful scorer inputs to this request's Arc,
/// avoiding process-global deltas when collection searches overlap.
#[derive(Default)]
pub(crate) struct NativeExactSearchProfile {
    pub scored_vectors: std::sync::atomic::AtomicU64,
    pub segment_search_nanos: std::sync::atomic::AtomicU64,
    pub total_elapsed_nanos: std::sync::atomic::AtomicU64,
}

/// Experimental topology hint. UUIDs identify physical segments, not DLI
/// levels or model generations. This is not a persisted DLI descriptor.
#[cfg(feature = "dli-research")]
#[derive(Clone, Debug)]
pub struct DliTopologyHint {
    pub physical_segments: Vec<uuid::Uuid>,
}

/// Explicit, process-local research configuration; never a durable hierarchy.
/// Level zero is a Plain member. Every captured segment must be listed.
#[cfg(feature = "dli-research")]
pub struct DliResearchConfig {
    pub members: parking_lot::RwLock<std::collections::HashMap<uuid::Uuid, usize>>,
    pub arity: usize,
    pub candidate_budget: std::sync::atomic::AtomicUsize,
    pub materialize_topk: std::sync::atomic::AtomicBool,
    /// Per-shard research override; zero retains legacy environment/default behavior.
    pub quantized_shortlist: std::sync::atomic::AtomicUsize,
    /// Opt-in request-local stage timings and scorer attribution.
    pub profile_scoring: std::sync::atomic::AtomicBool,
    /// Optional request-event destination, kept per DLI shard for concurrent tests/runs.
    pub profile_output_path: parking_lot::RwLock<Option<std::path::PathBuf>>,
    /// Optional routing-plan event destination (`DLI_ROUTING_EVENTS`), read once.
    pub routing_output_path: parking_lot::RwLock<Option<std::path::PathBuf>>,
    /// Validated legacy `DLI_QUANTIZED_SHORTLIST` value, parsed once at construction.
    /// Used only when `quantized_shortlist` is zero. Absent, zero or malformed
    /// values select original scoring (the default) instead of forcing every
    /// query into exact fallback.
    pub env_quantized_shortlist: Option<usize>,
    pub routing_policy:
        parking_lot::RwLock<crate::collection_manager::dli_bucket_plan::RoutingPolicy>,
    pub learned_queries: std::sync::atomic::AtomicU64,
    pub exact_fallbacks: std::sync::atomic::AtomicU64,
    /// Learned queries in which at least one member requested quantized
    /// scoring but had no derived quantized artifact (scored with originals).
    pub quantization_degraded_queries: std::sync::atomic::AtomicU64,
    pub selected_postings: std::sync::atomic::AtomicU64,
    pub unique_posting_offsets: std::sync::atomic::AtomicU64,
    pub routing_and_planning_nanos: std::sync::atomic::AtomicU64,
    pub scoring_nanos: std::sync::atomic::AtomicU64,
    pub resolution_nanos: std::sync::atomic::AtomicU64,
    pub native_scorer_nanos: std::sync::atomic::AtomicU64,
    pub hit_materialization_nanos: std::sync::atomic::AtomicU64,
    pub router_inference_nanos: std::sync::atomic::AtomicU64,
    pub bucket_planning_nanos: std::sync::atomic::AtomicU64,
    pub models_evaluated: std::sync::atomic::AtomicU64,
    pub selected_buckets: std::sync::atomic::AtomicU64,
    pub selected_members: std::sync::atomic::AtomicU64,
    pub materialized_hits: std::sync::atomic::AtomicU64,

    pub read_guard_nanos: std::sync::atomic::AtomicU64,
    pub fallback_reasons: parking_lot::Mutex<std::collections::HashMap<&'static str, u64>>,
    #[cfg(test)]
    pub test_guard_hook: parking_lot::Mutex<Option<DliGuardHook>>,
}

#[cfg(all(test, feature = "dli-research"))]
pub struct DliGuardHook {
    pub entered: std::sync::mpsc::Sender<()>,
    pub resume: parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
}

#[cfg(feature = "dli-research")]
impl DliResearchConfig {
    pub fn new(members: Vec<(uuid::Uuid, usize)>, arity: usize, budget: usize) -> Option<Self> {
        let len = members.len();
        let members: std::collections::HashMap<_, _> = members.into_iter().collect();
        if arity < 2 || budget == 0 || len != members.len() || len == 0 {
            return None;
        }
        Some(Self {
            members: parking_lot::RwLock::new(members),
            arity,
            candidate_budget: budget.into(),
            materialize_topk: Default::default(),
            quantized_shortlist: Default::default(),
            profile_scoring: std::env::var_os("DLI_REQUEST_SCORING_PROFILE")
                .is_some()
                .into(),
            profile_output_path: parking_lot::RwLock::new(
                std::env::var_os("DLI_REQUEST_SCORING_EVENTS").map(std::path::PathBuf::from),
            ),
            routing_output_path: parking_lot::RwLock::new(
                std::env::var_os("DLI_ROUTING_EVENTS").map(std::path::PathBuf::from),
            ),
            env_quantized_shortlist: parse_env_quantized_shortlist(std::env::var(
                "DLI_QUANTIZED_SHORTLIST",
            )),
            routing_policy: Default::default(),
            learned_queries: Default::default(),
            exact_fallbacks: Default::default(),
            quantization_degraded_queries: Default::default(),
            selected_postings: Default::default(),
            unique_posting_offsets: Default::default(),
            routing_and_planning_nanos: Default::default(),
            scoring_nanos: Default::default(),
            resolution_nanos: Default::default(),
            native_scorer_nanos: Default::default(),
            hit_materialization_nanos: Default::default(),
            router_inference_nanos: Default::default(),
            bucket_planning_nanos: Default::default(),
            models_evaluated: Default::default(),
            selected_buckets: Default::default(),
            selected_members: Default::default(),
            materialized_hits: Default::default(),

            read_guard_nanos: Default::default(),
            fallback_reasons: Default::default(),
            #[cfg(test)]
            test_guard_hook: Default::default(),
        })
    }
}

#[cfg(feature = "dli-research")]
pub(crate) fn parse_env_quantized_shortlist(
    value: Result<String, std::env::VarError>,
) -> Option<usize> {
    match value {
        Err(std::env::VarError::NotPresent) => None,
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(shortlist) if shortlist > 0 => Some(shortlist),
            _ => {
                log::warn!(
                    "Ignoring DLI_QUANTIZED_SHORTLIST={raw:?}: expected a positive integer; original scoring is used"
                );
                None
            }
        },
        Err(std::env::VarError::NotUnicode(raw)) => {
            log::warn!(
                "Ignoring non-UTF-8 DLI_QUANTIZED_SHORTLIST={raw:?}; original scoring is used"
            );
            None
        }
    }
}

/// Request-local record of one learned attempt, collected while segment guards
/// are held and persisted only after they are released.
#[cfg(feature = "dli-research")]
#[derive(Default)]
struct DliAttemptRecord {
    /// Attempt stage summary; `None` when routing did not complete.
    summary: Option<serde_json::Value>,
    /// Scorer work, including partial work before a routing error.
    scoring: Option<crate::collection_manager::dli_native_route::LearnedScoringProfile>,
    routing_event: Option<serde_json::Value>,
}

/// Persist one request's diagnostic events. Called only after the request's
/// outcome is final and all segment guards are released; the blocking file
/// I/O runs off the async executor threads.
#[cfg(feature = "dli-research")]
async fn persist_dli_events(
    profile_path: Option<std::path::PathBuf>,
    routing_path: Option<std::path::PathBuf>,
    events: Option<(serde_json::Value, Option<serde_json::Value>)>,
) {
    let Some((event, routing_event)) = events else {
        return;
    };
    let _ = tokio::task::spawn_blocking(move || {
        use crate::collection_manager::dli_native_route::append_jsonl_line;
        if let (Some(path), Some(routing_event)) = (routing_path, routing_event) {
            append_jsonl_line(&path, &routing_event);
        }
        if let Some(path) = profile_path {
            append_jsonl_line(&path, &event);
        }
    })
    .await;
}

/// Reason the experimental entry point ran a complete native exact search.
#[cfg(feature = "dli-research")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DliFallbackReason {
    MissingHint,
    TopologyMismatch,
    ProxyPresent,
    TopologyBusy,
    LearnedRoutingUnavailable,
}

#[cfg(feature = "dli-research")]
fn dli_fallback_reason(
    segments: &[LockedSegment],
    hint: Option<&DliTopologyHint>,
) -> DliFallbackReason {
    use std::collections::HashSet;
    let Some(hint) = hint else {
        return DliFallbackReason::MissingHint;
    };
    if segments
        .iter()
        .any(|s| matches!(s, LockedSegment::Proxy(_)))
    {
        return DliFallbackReason::ProxyPresent;
    }
    let expected: HashSet<_> = hint.physical_segments.iter().copied().collect();
    let mut actual = HashSet::with_capacity(segments.len());
    for segment in segments {
        let Some(read) = segment.get_read().try_read() else {
            return DliFallbackReason::TopologyBusy;
        };
        actual.insert(read.segment_uuid());
    }
    if expected.len() != hint.physical_segments.len()
        || actual.len() != segments.len()
        || expected != actual
    {
        return DliFallbackReason::TopologyMismatch;
    }
    // A matching topology is necessary but insufficient: level assignment,
    // artifact integrity and coordinated learned routing are not implemented.
    DliFallbackReason::LearnedRoutingUnavailable
}

impl SegmentsSearcher {
    /// Experimental process-local DLI dispatch. Unsupported or unsafe states
    /// execute exact native search over the *same* captured handles.
    #[cfg(feature = "dli-research")]
    pub async fn search_dli_research(
        holder: LockedSegmentHolder,
        batch_request: Arc<CoreSearchRequestBatch>,
        runtime_handle: &AdaptiveSearchHandle,
        query_context: QueryContext,
        timeout: Duration,
        config: Arc<DliResearchConfig>,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        use std::sync::atomic::Ordering;

        use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, VectorInternal};
        use segment::entry::ReadSegmentEntry;
        use shard::query::query_enum::QueryEnum;

        use crate::collection_manager::dli_native_route::{
            CandidateResolution, resolve_pinned_candidates,
        };

        let started = Instant::now();
        let request_id = uuid::Uuid::new_v4().to_string();
        let profile_enabled = config.profile_scoring.load(Ordering::Relaxed);
        let rejection_reason = std::sync::Arc::new(parking_lot::Mutex::new(None));
        // Request-local, uncontended: written by the attempt task, read after it joins.
        let attempt_record =
            std::sync::Arc::new(parking_lot::Mutex::new(DliAttemptRecord::default()));
        let captured = Self::capture_segments(&holder, timeout)?;
        let attempt_segments = captured.clone();
        let attempt_request = batch_request.clone();
        let attempt_holder = holder.clone();
        let attempt_context = query_context.clone();
        let cpu_utilization = attempt_context.handoff().cpu_utilization();
        let handoff = attempt_context.handoff().clone();
        let attempt_config = config.clone();
        let attempt_reason = rejection_reason.clone();
        let attempt_trace_record = attempt_record.clone();
        let attempt = runtime_handle.spawn_blocking(move || {
            let _scope = handoff.enter_guard();
            cpu_utilization.measure(|| {
                macro_rules! reject {
                    ($reason:expr) => {{
                        *attempt_reason.lock() = Some($reason);
                        *attempt_config
                            .fallback_reasons
                            .lock()
                            .entry($reason)
                            .or_default() += 1;
                        return None;
                    }};
                }
                let [request] = attempt_request.searches.as_slice() else {
                    reject!("unsupported_query");
                };
                let QueryEnum::Nearest(named) = &request.query else {
                    reject!("unsupported_query");
                };
                let VectorInternal::Dense(query) = &named.query else {
                    reject!("unsupported_query");
                };
                if named.using.as_deref().unwrap_or(DEFAULT_VECTOR_NAME) != DEFAULT_VECTOR_NAME
                    || request.filter.is_some()
                    || request
                        .with_payload
                        .as_ref()
                        .is_some_and(|payload| payload.is_required())
                    || request
                        .with_vector
                        .as_ref()
                        .is_some_and(|vector| vector.is_enabled())
                    || request
                        .params
                        .as_ref()
                        .is_some_and(|params| *params != Default::default())
                {
                    reject!("unsupported_query");
                }
                let Some(limit) = request.limit.checked_add(request.offset) else {
                    reject!("unsupported_query")
                };
                let mut guards = Vec::with_capacity(attempt_segments.len());
                for segment in &attempt_segments {
                    let LockedSegment::Original(segment) = segment else {
                        reject!("proxy");
                    };
                    // Never wait while holding other segment locks.
                    let Some(guard) = segment.try_read() else {
                        reject!("busy")
                    };
                    guards.push(guard);
                }
                struct GuardTimer<'a> {
                    counter: &'a std::sync::atomic::AtomicU64,
                    started: Instant,
                }
                impl Drop for GuardTimer<'_> {
                    fn drop(&mut self) {
                        self.counter.fetch_add(
                            u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                            Ordering::Relaxed,
                        );
                    }
                }
                // Declared after guards, so recorded before their release on
                // every return path. Includes deliberate pauses in test builds.
                let _guard_timer = GuardTimer {
                    counter: &attempt_config.read_guard_nanos,
                    started: Instant::now(),
                };
                // Reject replacement between capture and lock acquisition. A
                // nonblocking holder read avoids lock-order deadlocks.
                let Some(holder_guard) = attempt_holder.try_read() else {
                    reject!("busy")
                };
                let active: Vec<_> = holder_guard
                    .non_appendable_then_appendable_segments()
                    .collect();
                if active.len() != attempt_segments.len()
                    || !active
                        .iter()
                        .zip(&attempt_segments)
                        .all(|(a, b)| std::ptr::eq(a.get(), b.get()))
                {
                    reject!("topology");
                }
                drop(holder_guard);
                let expected_members = attempt_config.members.read().clone();
                if guards.len() != expected_members.len() {
                    reject!("topology");
                }
                let mut members = Vec::with_capacity(guards.len());
                for segment in &guards {
                    let Some(&level) = expected_members.get(&segment.uuid) else {
                        reject!("topology")
                    };
                    members.push((level, &**segment));
                }
                if !members.iter().any(|(level, _)| *level > 0) {
                    reject!("topology");
                }
                #[cfg(test)]
                if let Some(hook) = attempt_config.test_guard_hook.lock().as_ref() {
                    if hook.entered.send(()).is_err()
                        || hook
                            .resume
                            .lock()
                            .recv_timeout(Duration::from_secs(10))
                            .is_err()
                    {
                        reject!("test_hook_timeout");
                    }
                }
                let segment_context = attempt_context.get_segment_query_context();
                let vector_context = segment_context.get_vector_context(DEFAULT_VECTOR_NAME, None);
                let mut trace = profile_enabled
                    .then(crate::collection_manager::dli_native_route::RouteTrace::default);
                let routed = match crate::collection_manager::dli_native_route::route_pinned_traced(
                    &members,
                    query,
                    attempt_config.arity,
                    attempt_config.candidate_budget.load(Ordering::Relaxed),
                    &vector_context,
                    *attempt_config.routing_policy.read(),
                    if attempt_config.materialize_topk.load(Ordering::Relaxed) {
                        limit
                    } else {
                        usize::MAX
                    },
                    match attempt_config.quantized_shortlist.load(Ordering::Relaxed) {
                        0 => attempt_config.env_quantized_shortlist,
                        shortlist => Some(shortlist),
                    },
                    trace.as_mut(),
                ) {
                    Ok(routed) => routed,
                    Err(error) => {
                        log::debug!("dli_research: routing fallback: {error}");
                        if let Some(trace) = trace.take() {
                            // Keep partial scorer work performed before the error.
                            let mut record = attempt_trace_record.lock();
                            record.scoring = Some(trace.scoring);
                            record.routing_event = trace.routing_event;
                        }
                        let reason = match &error {
                            segment::common::operation_error::OperationError::ServiceError { description, .. } => match description.as_str() {
                                "quantized physical generation mismatch" => "quantization_generation_mismatch",
                                "quantized mode requires bounded materialization and nonzero shortlist" => "quantization_configuration",
                                _ => "routing_error",
                            },
                            _ => "routing_error",
                        };
                        reject!(reason);
                    }
                };
                if !routed.plan.buckets.iter().any(|bucket| bucket.level > 0) {
                    if let Some(trace) = trace.take() {
                        let mut record = attempt_trace_record.lock();
                        record.scoring = Some(trace.scoring);
                        record.routing_event = trace.routing_event;
                    }
                    reject!("no_learned_bucket");
                }
                if routed.quantization_degraded_members > 0 {
                    attempt_config
                        .quantization_degraded_queries
                        .fetch_add(1, Ordering::Relaxed);
                }
                attempt_config.selected_postings.fetch_add(
                    u64::try_from(routed.posting_candidates).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                attempt_config.unique_posting_offsets.fetch_add(
                    u64::try_from(routed.unique_posting_offsets).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                attempt_config.routing_and_planning_nanos.fetch_add(
                    u64::try_from(routed.routing_and_planning_time.as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                attempt_config.scoring_nanos.fetch_add(
                    u64::try_from(routed.scoring_time.as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                attempt_config.native_scorer_nanos.fetch_add(
                    u64::try_from(routed.native_scorer_time.as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                attempt_config.hit_materialization_nanos.fetch_add(
                    u64::try_from(routed.hit_materialization_time.as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                attempt_config.router_inference_nanos.fetch_add(
                    u64::try_from(routed.router_inference_time.as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                attempt_config.bucket_planning_nanos.fetch_add(
                    u64::try_from(routed.bucket_planning_time.as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                attempt_config
                    .models_evaluated
                    .fetch_add(routed.models_evaluated as u64, Ordering::Relaxed);
                attempt_config
                    .selected_buckets
                    .fetch_add(routed.plan.buckets.len() as u64, Ordering::Relaxed);
                attempt_config
                    .selected_members
                    .fetch_add(routed.per_segment.len() as u64, Ordering::Relaxed);
                attempt_config.materialized_hits.fetch_add(
                    routed.per_segment.iter().map(|(_, h)| h.len() as u64).sum(),
                    Ordering::Relaxed,
                );
                let attempt_summary = trace.as_ref().map(|_| serde_json::json!({
                    "posting_entries": routed.posting_candidates,
                    "unique_offsets": routed.unique_posting_offsets, "dead_offsets": routed.dead_posting_offsets,
                    "models": routed.models_evaluated,
                    "selected_buckets": routed.plan.buckets.len(),
                    "routing_ns": routed.routing_and_planning_time.as_nanos(),
                    "scoring_ns": routed.scoring_time.as_nanos(),
                    "materialization_ns": routed.hit_materialization_time.as_nanos(),
                    "quantization_degraded_members": routed.quantization_degraded_members,
                }));
                let resolution_start = Instant::now();
                let resolution = resolve_pinned_candidates(&members, routed, limit);
                // Stop the clock before any bookkeeping; no I/O happens in this closure.
                let resolution_elapsed = resolution_start.elapsed();
                attempt_config.resolution_nanos.fetch_add(
                    u64::try_from(resolution_elapsed.as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                // Deleted posting entries must not consume the ANN budget and
                // silently underfill a query when the captured native population
                // can supply more hits. Native fallback also handles missing vectors.
                let required =
                    limit.min(members.iter().map(|(_, s)| s.available_point_count()).sum());
                // The attempt label is decided only after the underfill check, so
                // an attempt that resolved candidates but could not supply the
                // final result is never reported as "resolved".
                let (outcome, attempt_resolution) = match resolution {
                    Ok(CandidateResolution::Resolved(hits)) if hits.len() >= required => {
                        (Ok(hits), "resolved")
                    }
                    Ok(CandidateResolution::Resolved(_)) => {
                        (Err("insufficient_valid_candidates"), "underfilled")
                    }
                    Ok(CandidateResolution::ExactFallback(reason)) => {
                        log::debug!("dli_research: candidate fallback: {reason:?}");
                        use crate::collection_manager::dli_native_route::ResolutionFallback;
                        let reason = match reason {
                            ResolutionFallback::AmbiguousOccurrence => "duplicate",
                            ResolutionFallback::DeletedState => "deleted",
                            ResolutionFallback::DeferredState => "deferred",
                            ResolutionFallback::DuplicateMember
                            | ResolutionFallback::CandidateMismatch => "identity",
                        };
                        (Err(reason), "fallback")
                    }
                    Err(error) => {
                        log::debug!("dli_research: routing fallback: {error}");
                        (Err("resolution_error"), "error")
                    }
                };
                if let (Some(trace), Some(mut summary)) = (trace.take(), attempt_summary) {
                    summary["resolution_ns"] = serde_json::json!(resolution_elapsed.as_nanos());
                    summary["attempt_resolution"] = serde_json::json!(attempt_resolution);
                    let mut record = attempt_trace_record.lock();
                    record.summary = Some(summary);
                    record.scoring = Some(trace.scoring);
                    record.routing_event = trace.routing_event;
                }
                match outcome {
                    Ok(hits) => Some(hits),
                    Err(reason) => reject!(reason),
                }
            })
        });
        // One authoritative final event per logical request, written only after
        // the outcome is known, every segment guard has been released (the
        // attempt closure has returned) and every timer has stopped.
        let emit = |final_outcome: &'static str,
                    exact: Option<&NativeExactSearchProfile>,
                    returned: Option<usize>,
                    error: Option<String>| {
            if !profile_enabled {
                return None;
            }
            let record = std::mem::take(&mut *attempt_record.lock());
            let fallback_reason = match final_outcome {
                "learned" => None,
                _ => Some(*rejection_reason.lock()),
            };
            let event = serde_json::json!({
                "event": "dli_request",
                "request_id": request_id,
                "final_outcome": final_outcome,
                "learned_route_supplied_results": final_outcome == "learned",
                "fallback_reason": fallback_reason.map(|r| r.unwrap_or("unspecified_native_exact_fallback")),
                "error": error,
                "attempt": record.summary,
                "learned_scoring": record.scoring,
                "exact_fallback": exact.map(|p| serde_json::json!({
                    "scored_vectors": p.scored_vectors.load(Ordering::Relaxed),
                    "segment_search_ns_sum": p.segment_search_nanos.load(Ordering::Relaxed),
                    "elapsed_ns": p.total_elapsed_nanos.load(Ordering::Relaxed),
                })),
                "returned_results": returned,
                "dli_shard_search_elapsed_ns": started.elapsed().as_nanos(),
            });
            Some((event, record.routing_event))
        };
        let profile_path = config.profile_output_path.read().clone();
        let routing_path = config.routing_output_path.read().clone();
        let persist =
            |events| persist_dli_events(profile_path.clone(), routing_path.clone(), events);

        let attempt_result = match AbortOnDropHandle::new(attempt).await {
            Ok(result) => result,
            Err(error) => {
                let error = CollectionError::from(error);
                persist(emit("error", None, None, Some(error.to_string()))).await;
                return Err(error);
            }
        };
        if let Some(hits) = attempt_result {
            config.learned_queries.fetch_add(1, Ordering::Relaxed);
            persist(emit("learned", None, Some(hits.len()), None)).await;
            return Ok(vec![hits]);
        }
        let mut exact_request = (*batch_request).clone();
        for request in &mut exact_request.searches {
            let params = request.params.get_or_insert_default();
            if params.indexed_only {
                let error = CollectionError::bad_request(
                    "experimental DLI fallback requires complete segment coverage",
                );
                persist(emit("error", None, None, Some(error.to_string()))).await;
                return Err(error);
            }
            params.exact = true;
        }
        config.exact_fallbacks.fetch_add(1, Ordering::Relaxed);
        let exact_request = Arc::new(exact_request);
        let fallback_started = Instant::now();
        let exact_profile = profile_enabled.then(|| Arc::new(NativeExactSearchProfile::default()));
        let results = if let Some(profile) = exact_profile.as_ref() {
            Self::search_captured_profiled(
                captured,
                exact_request,
                runtime_handle,
                false,
                query_context,
                timeout.saturating_sub(started.elapsed()),
                profile.clone(),
            )
            .await
        } else {
            Self::search_captured(
                captured,
                exact_request,
                runtime_handle,
                false,
                query_context,
                timeout.saturating_sub(started.elapsed()),
            )
            .await
        };
        if let Some(profile) = exact_profile.as_ref() {
            profile.total_elapsed_nanos.store(
                u64::try_from(fallback_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        }
        match results {
            Ok(results) => {
                let returned = results.iter().map(Vec::len).sum::<usize>();
                persist(emit(
                    "exact_fallback",
                    exact_profile.as_deref(),
                    Some(returned),
                    None,
                ))
                .await;
                Ok(results)
            }
            Err(error) => {
                persist(emit(
                    "error",
                    exact_profile.as_deref(),
                    None,
                    Some(error.to_string()),
                ))
                .await;
                Err(error)
            }
        }
    }

    /// Experimental correctness baseline for a future derived DLI view.
    /// Always executes native exact search: learned DLI routing is not yet
    /// integrated. The capture and execution share the same native handles.
    #[cfg(feature = "dli-research")]
    pub async fn search_dli_exact_fallback(
        holder: LockedSegmentHolder,
        hint: Option<&DliTopologyHint>,
        batch_request: Arc<CoreSearchRequestBatch>,
        runtime_handle: &AdaptiveSearchHandle,
        query_context: QueryContext,
        timeout: Duration,
    ) -> CollectionResult<(Vec<Vec<ScoredPoint>>, DliFallbackReason)> {
        let started = Instant::now();
        let segments = Self::capture_segments(&holder, timeout)?;
        let reason = dli_fallback_reason(&segments, hint);
        let mut exact_request = (*batch_request).clone();
        for request in &mut exact_request.searches {
            if !matches!(
                &request.query,
                shard::query::query_enum::QueryEnum::Nearest(named)
                    if matches!(&named.query, segment::data_types::vectors::VectorInternal::Dense(_))
            ) {
                return Err(CollectionError::bad_request(
                    "experimental DLI fallback supports dense nearest-neighbor queries only",
                ));
            }
            let params = request.params.get_or_insert_default();
            if params.indexed_only {
                return Err(CollectionError::bad_request(
                    "experimental DLI fallback requires complete segment coverage",
                ));
            }
            params.exact = true;
        }
        log::debug!("dli_research: full native exact fallback: {reason:?}");
        let result = Self::search_captured(
            segments,
            Arc::new(exact_request),
            runtime_handle,
            false,
            query_context,
            timeout.saturating_sub(started.elapsed()),
        )
        .await?;
        Ok((result, reason))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use ahash::AHashSet;
    use common::ambient;
    use common::ambient::{AmbientContext, Handoff};
    use segment::data_types::query_context::QueryContext;
    use segment::types::{Condition, Filter, HasIdCondition};
    use shard::optimizers::config::DEFAULT_INDEXING_THRESHOLD_KB;
    use shard::search::CoreSearchRequestBatch;
    use tempfile::Builder;

    use super::*;
    use crate::collection_manager::fixtures::{TEST_TIMEOUT, build_test_holder};
    use crate::collection_manager::segments_searcher::SegmentsSearcher;
    use crate::common::adaptive_handle::AdaptiveSearchHandle;
    use crate::operations::types::CoreSearchRequest;
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_exact_profiles_are_request_local_under_concurrency() {
        let _scope = ambient::test_guard();
        let first_dir = Builder::new().prefix("profiled_exact_a").tempdir().unwrap();
        let second_dir = Builder::new().prefix("profiled_exact_b").tempdir().unwrap();
        let first_holder = build_test_holder(first_dir.path());
        let second_holder = build_test_holder(second_dir.path());
        let first_segments =
            SegmentsSearcher::capture_segments(&first_holder, TEST_TIMEOUT).unwrap();
        let second_segments =
            SegmentsSearcher::capture_segments(&second_holder, TEST_TIMEOUT).unwrap();
        let query = vec![1.0, 1.0, 1.0, 1.0];
        let request = |filter| CoreSearchRequest {
            query: query.clone().into(),
            with_payload: None,
            with_vector: None,
            filter,
            params: Some(segment::types::SearchParams {
                exact: true,
                ..Default::default()
            }),
            limit: 5,
            score_threshold: None,
            offset: 0,
        };
        let first_profile = Arc::new(NativeExactSearchProfile::default());
        let second_profile = Arc::new(NativeExactSearchProfile::default());
        let runtime = AdaptiveSearchHandle::current_for_tests();
        let first = SegmentsSearcher::search_captured_profiled(
            first_segments,
            Arc::new(CoreSearchRequestBatch {
                searches: vec![request(None)],
            }),
            &runtime,
            false,
            QueryContext::new(
                DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::new()),
            ),
            TEST_TIMEOUT,
            first_profile.clone(),
        );
        let ids = AHashSet::from([3_u64.into(), 11_u64.into()]);
        let second = SegmentsSearcher::search_captured_profiled(
            second_segments,
            Arc::new(CoreSearchRequestBatch {
                searches: vec![request(Some(Filter::new_must(Condition::HasId(
                    HasIdCondition::from(ids),
                ))))],
            }),
            &runtime,
            false,
            QueryContext::new(
                DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::new()),
            ),
            TEST_TIMEOUT,
            second_profile.clone(),
        );
        let (first_result, second_result) = tokio::join!(first, second);
        assert_eq!(first_result.unwrap()[0].len(), 5);
        assert_eq!(second_result.unwrap()[0].len(), 2);
        let first_scored = first_profile.scored_vectors.load(Ordering::Relaxed);
        let second_scored = second_profile.scored_vectors.load(Ordering::Relaxed);
        // Independent expectations from the fixture, not from the counters:
        // an exact unfiltered search scores every live vector of every segment
        // once; the HasId-filtered search scores only matching live vectors.
        let (expected_all, expected_filtered) = {
            let guard = second_holder.read();
            let mut all = 0u64;
            let mut filtered = 0u64;
            for (_, segment) in guard.iter() {
                let segment = segment.get();
                let segment = segment.read();
                all += segment.available_point_count() as u64;
                filtered += [3_u64, 11]
                    .into_iter()
                    .filter(|id| {
                        segment
                            .has_point((*id).into(), common::types::DeferredBehavior::VisibleOnly)
                    })
                    .count() as u64;
            }
            (all, filtered)
        };
        assert_eq!(
            first_scored, expected_all,
            "unfiltered exact search scores each live vector once"
        );
        assert_eq!(
            second_scored, expected_filtered,
            "filtered request retains only its own scorer work"
        );
        assert!(first_profile.segment_search_nanos.load(Ordering::Relaxed) > 0);
        assert!(second_profile.segment_search_nanos.load(Ordering::Relaxed) > 0);
    }

    #[tokio::test]
    async fn test_captured_search_uses_only_pinned_holder_members() {
        use segment::data_types::vectors::only_default_vector;
        use segment::entry::SegmentEntry as _;
        use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
        use segment::types::Distance;

        let _scope = ambient::test_guard();
        let dir = Builder::new().prefix("captured-search").tempdir().unwrap();
        let holder = build_test_holder(dir.path());
        let captured = SegmentsSearcher::capture_segments(&holder, TEST_TIMEOUT).unwrap();
        let mut added = build_simple_segment(dir.path(), 4, Distance::Dot).unwrap();
        added
            .upsert_point(
                100,
                999_999_u64.into(),
                only_default_vector(&[100.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        holder.write().add_new(added);
        let request = Arc::new(CoreSearchRequestBatch {
            searches: vec![CoreSearchRequest {
                query: vec![1.0, 1.0, 1.0, 1.0].into(),
                with_payload: None,
                with_vector: None,
                filter: None,
                params: None,
                limit: 20,
                score_threshold: None,
                offset: 0,
            }],
        });
        let context = || {
            QueryContext::new(
                DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::new()),
            )
        };
        let old = SegmentsSearcher::search_captured(
            captured,
            request.clone(),
            &AdaptiveSearchHandle::current_for_tests(),
            false,
            context(),
            TEST_TIMEOUT,
        )
        .await
        .unwrap();
        assert!(!old[0].iter().any(|p| p.id == 999_999_u64.into()));
        let current = SegmentsSearcher::search(
            holder,
            request,
            &AdaptiveSearchHandle::current_for_tests(),
            false,
            context(),
            TEST_TIMEOUT,
        )
        .await
        .unwrap();
        assert_eq!(current[0][0].id, 999_999_u64.into());
    }

    #[cfg(feature = "dli-research")]
    #[tokio::test]
    async fn test_dli_exact_fallback_uses_native_search() {
        let dir = Builder::new().prefix("dli-fallback").tempdir().unwrap();
        let holder = build_test_holder(dir.path());
        let req = CoreSearchRequest {
            query: vec![1.0, 1.0, 1.0, 1.0].into(),
            with_payload: None,
            with_vector: None,
            filter: Some(Filter::new_must(Condition::HasId(HasIdCondition::from(
                [3_u64.into(), 11_u64.into()]
                    .into_iter()
                    .collect::<AHashSet<_>>(),
            )))),
            params: None,
            limit: 5,
            score_threshold: None,
            offset: 0,
        };
        let batch = Arc::new(CoreSearchRequestBatch {
            searches: vec![req.clone()],
        });
        let context = || {
            QueryContext::new(
                DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::new()),
            )
        };
        let captured = SegmentsSearcher::capture_segments(&holder, TEST_TIMEOUT).unwrap();
        let mut hint = DliTopologyHint {
            physical_segments: captured
                .iter()
                .map(|s| s.get_read().read().segment_uuid())
                .collect(),
        };
        assert_eq!(
            dli_fallback_reason(&captured, Some(&hint)),
            DliFallbackReason::LearnedRoutingUnavailable
        );
        let prior_hint = hint.clone();
        hint.physical_segments.pop();
        assert_eq!(
            dli_fallback_reason(&captured, Some(&hint)),
            DliFallbackReason::TopologyMismatch
        );
        hint.physical_segments.push(hint.physical_segments[0]);
        assert_eq!(
            dli_fallback_reason(&captured, Some(&hint)),
            DliFallbackReason::TopologyMismatch
        );
        let actual = SegmentsSearcher::search_dli_exact_fallback(
            holder.clone(),
            None,
            batch.clone(),
            &AdaptiveSearchHandle::current_for_tests(),
            context(),
            TEST_TIMEOUT,
        )
        .await
        .unwrap();
        holder
            .write()
            .add_new(crate::collection_manager::fixtures::empty_segment(
                dir.path(),
            ));
        let stale = SegmentsSearcher::search_dli_exact_fallback(
            holder.clone(),
            Some(&prior_hint),
            batch,
            &AdaptiveSearchHandle::current_for_tests(),
            context(),
            TEST_TIMEOUT,
        )
        .await
        .unwrap();
        assert_eq!(stale.1, DliFallbackReason::TopologyMismatch);
        assert_eq!(stale.0, actual.0);
        let mut incomplete = req.clone();
        incomplete.params = Some(segment::types::SearchParams {
            indexed_only: true,
            ..Default::default()
        });
        let rejected = SegmentsSearcher::search_dli_exact_fallback(
            holder.clone(),
            None,
            Arc::new(CoreSearchRequestBatch {
                searches: vec![incomplete],
            }),
            &AdaptiveSearchHandle::current_for_tests(),
            context(),
            TEST_TIMEOUT,
        )
        .await;
        assert!(
            rejected.is_err(),
            "indexed-only cannot promise full coverage"
        );
        let mut exact = req;
        exact.params = Some(segment::types::SearchParams {
            exact: true,
            ..Default::default()
        });
        let expected = SegmentsSearcher::search(
            holder,
            Arc::new(CoreSearchRequestBatch {
                searches: vec![exact],
            }),
            &AdaptiveSearchHandle::current_for_tests(),
            false,
            context(),
            TEST_TIMEOUT,
        )
        .await
        .unwrap();
        assert_eq!(actual.1, DliFallbackReason::MissingHint);
        assert!(!actual.0[0].is_empty());
        assert_eq!(actual.0, expected);
    }
}
