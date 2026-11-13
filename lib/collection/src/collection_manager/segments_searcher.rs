use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ahash::AHashMap;
use common::ambient;
use common::ambient::hw;
use common::types::{DeferredBehavior, ScoreType};
use futures::stream::FuturesUnordered;
use futures::{FutureExt, TryStreamExt};
use itertools::Itertools;
use ordered_float::Float;
use segment::common::operation_error::OperationError;
use segment::data_types::modifier::Modifier;
use segment::data_types::query_context::{FormulaContext, QueryContext, SegmentQueryContext};
use segment::data_types::segment_record::SegmentRecordRaw;
use segment::data_types::vectors::QueryVector;
use segment::index::field_index::full_text_index::Bm25Params;
use segment::json_path::JsonPath;
use segment::types::{
    Filter, Indexes, PointIdType, ScoredPoint, SegmentConfig, VectorName, WithPayload, WithVector,
};
use shard::common::stopping_guard::StoppingGuard;
use shard::optimizers::config::DEFAULT_INDEXING_THRESHOLD_KB;
use shard::query::query_context::{fill_segment_query_context, init_query_context};
use shard::query::text::init_text_query_context;
use shard::retrieve::record_internal::RecordInternal;
use shard::retrieve::retrieve_blocking::{retrieve_blocking, retrieve_raw_blocking};
use shard::search::{
    BatchSearchParams, CoreSearchRequestBatch, SearchBatchGroup, group_search_batches,
};
use shard::search_result_aggregator::BatchResultAggregator;
use shard::segment_holder::locked::LockedSegmentHolder;
use tokio_util::task::AbortOnDropHandle;

use crate::collection_manager::holders::segment_holder::LockedSegment;
use crate::collection_manager::probabilistic_search_sampling::find_search_sampling_over_point_distribution;
use crate::common::adaptive_handle::AdaptiveSearchHandle;
use crate::config::CollectionConfigInternal;
use crate::operations::types::{CollectionError, CollectionResult};

type BatchOffset = usize;
type SegmentOffset = usize;

// batch -> point for one segment
type SegmentBatchSearchResult = Vec<Vec<ScoredPoint>>;
// Segment -> batch -> point
type BatchSearchResult = Vec<SegmentBatchSearchResult>;

// Result of batch search in one segment
type SegmentSearchExecutedResult = CollectionResult<(SegmentBatchSearchResult, Vec<bool>)>;

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
    pub members: std::collections::HashMap<uuid::Uuid, usize>,
    pub arity: usize,
    pub candidate_budget: usize,
    pub learned_queries: std::sync::atomic::AtomicU64,
    pub exact_fallbacks: std::sync::atomic::AtomicU64,
    pub selected_postings: std::sync::atomic::AtomicU64,
    pub unique_posting_offsets: std::sync::atomic::AtomicU64,
    pub routing_and_planning_nanos: std::sync::atomic::AtomicU64,
    pub scoring_nanos: std::sync::atomic::AtomicU64,
    pub resolution_nanos: std::sync::atomic::AtomicU64,
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
            members,
            arity,
            candidate_budget: budget,
            learned_queries: Default::default(),
            exact_fallbacks: Default::default(),
            selected_postings: Default::default(),
            unique_posting_offsets: Default::default(),
            routing_and_planning_nanos: Default::default(),
            scoring_nanos: Default::default(),
            resolution_nanos: Default::default(),
            fallback_reasons: Default::default(),
            #[cfg(test)]
            test_guard_hook: Default::default(),
        })
    }
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

/// Simple implementation of segment manager
///  - rebuild segment for memory optimization purposes
#[derive(Default)]
pub struct SegmentsSearcher;

impl SegmentsSearcher {
    /// Execute searches in parallel and return results in the same order as the searches were provided
    async fn execute_searches(
        searches: Vec<AbortOnDropHandle<SegmentSearchExecutedResult>>,
    ) -> CollectionResult<(BatchSearchResult, Vec<Vec<bool>>)> {
        let results_len = searches.len();

        let mut search_results_per_segment_res = FuturesUnordered::new();
        for (idx, search) in searches.into_iter().enumerate() {
            // map the result to include the request index for later reordering
            let result_with_request_index = search.map(move |res| res.map(|s| (idx, s)));
            search_results_per_segment_res.push(result_with_request_index);
        }

        let mut search_results_per_segment = vec![Vec::new(); results_len];
        let mut further_searches_per_segment = vec![Vec::new(); results_len];
        // process results as they come in and store them in the correct order
        while let Some((idx, search_result)) = search_results_per_segment_res.try_next().await? {
            let (search_results, further_searches) = search_result?;
            debug_assert!(search_results.len() == further_searches.len());
            search_results_per_segment[idx] = search_results;
            further_searches_per_segment[idx] = further_searches;
        }
        Ok((search_results_per_segment, further_searches_per_segment))
    }

    /// Processes search result of `[segment_size x batch_size]`.
    ///
    /// # Arguments
    /// * `search_result` - `[segment_size x batch_size]`
    /// * `limits` - `[batch_size]` - how many results to return for each batched request
    /// * `further_results` - `[segment_size x batch_size]` - whether we can search further in the segment
    ///
    /// Returns batch results aggregated by `[batch_size]` and list of queries, grouped by segment to re-run
    pub(crate) fn process_search_result_step1(
        search_result: BatchSearchResult,
        limits: Vec<usize>,
        further_results: &[Vec<bool>],
    ) -> (
        BatchResultAggregator,
        AHashMap<SegmentOffset, Vec<BatchOffset>>,
    ) {
        let number_segments = search_result.len();
        let batch_size = limits.len();

        // The lowest scored element must be larger or equal to the worst scored element in each segment.
        // Otherwise, the sampling is invalid and some points might be missing.
        // e.g. with 3 segments with the following sampled ranges:
        // s1 - [0.91 -> 0.87]
        // s2 - [0.92 -> 0.86]
        // s3 - [0.93 -> 0.85]
        // If the top merged scores result range is [0.93 -> 0.86] then we do not know if s1 could have contributed more points at the lower part between [0.87 -> 0.86]
        // In that case, we need to re-run the search without sampling on that segment.

        // Initialize result aggregators for each batched request
        let mut result_aggregator = BatchResultAggregator::new(limits.iter().copied());
        result_aggregator.update_point_versions(search_result.iter().flatten().flatten());

        // Therefore we need to track the lowest scored element per segment for each batch
        let mut lowest_scores_per_request: Vec<Vec<ScoreType>> = vec![
            vec![f32::max_value(); batch_size]; // initial max score value for each batch
            number_segments
        ];

        let mut retrieved_points_per_request: Vec<Vec<BatchOffset>> = vec![
            vec![0; batch_size]; // initial max score value for each batch
            number_segments
        ];

        // Batch results merged from all segments
        for (segment_idx, segment_result) in search_result.into_iter().enumerate() {
            // merge results for each batch search request across segments
            for (batch_req_idx, query_res) in segment_result.into_iter().enumerate() {
                retrieved_points_per_request[segment_idx][batch_req_idx] = query_res.len();
                lowest_scores_per_request[segment_idx][batch_req_idx] = query_res
                    .last()
                    .map(|x| x.score)
                    .unwrap_or_else(f32::min_value);
                result_aggregator.update_batch_results(batch_req_idx, query_res);
            }
        }

        // segment id -> list of batch ids
        let mut searches_to_rerun: AHashMap<SegmentOffset, Vec<BatchOffset>> = AHashMap::new();

        // Check if we want to re-run the search without sampling on some segments
        for (batch_id, required_limit) in limits.into_iter().enumerate() {
            let lowest_batch_score_opt = result_aggregator.batch_lowest_scores(batch_id);

            // If there are no results, we do not need to re-run the search
            if let Some(lowest_batch_score) = lowest_batch_score_opt {
                for segment_id in 0..number_segments {
                    let segment_lowest_score = lowest_scores_per_request[segment_id][batch_id];
                    let retrieved_points = retrieved_points_per_request[segment_id][batch_id];
                    let have_further_results = further_results[segment_id][batch_id];

                    if have_further_results
                        && retrieved_points < required_limit
                        && segment_lowest_score >= lowest_batch_score
                    {
                        log::debug!(
                            "Search to re-run without sampling on segment_id: {segment_id} segment_lowest_score: {segment_lowest_score}, lowest_batch_score: {lowest_batch_score}, retrieved_points: {retrieved_points}, required_limit: {required_limit}",
                        );
                        // It is possible, that current segment can have better results than
                        // the lowest score in the batch. In that case, we need to re-run the search
                        // without sampling on that segment.
                        searches_to_rerun
                            .entry(segment_id)
                            .or_default()
                            .push(batch_id);
                    }
                }
            }
        }

        (result_aggregator, searches_to_rerun)
    }

    pub async fn prepare_query_context(
        segments: LockedSegmentHolder,
        batch_request: &CoreSearchRequestBatch,
        collection_config: &CollectionConfigInternal,
        timeout: Duration,
        search_runtime_handle: &AdaptiveSearchHandle,
        is_stopped_guard: &StoppingGuard,
    ) -> CollectionResult<Option<QueryContext>> {
        let indexing_threshold_kb = collection_config
            .optimizer_config
            .indexing_threshold
            .unwrap_or(DEFAULT_INDEXING_THRESHOLD_KB);
        let full_scan_threshold_kb = collection_config.hnsw_config.full_scan_threshold;
        let search_optimized_threshold_kb = indexing_threshold_kb.max(full_scan_threshold_kb);

        let query_context = init_query_context(
            &batch_request.searches,
            search_optimized_threshold_kb,
            is_stopped_guard,
            |vector_name| {
                collection_config
                    .params
                    .get_sparse_vector_params_opt(vector_name)
                    .map(|params| params.modifier == Some(Modifier::Idf))
                    .unwrap_or(false)
            },
        )?;
        Self::fill_query_context(
            query_context,
            segments,
            search_runtime_handle,
            is_stopped_guard,
            timeout,
        )
        .await
    }

    /// Fill `query_context` with the statistics of every segment, one blocking
    /// task per segment when there are IDF or text statistics to collect, a
    /// single sequential task otherwise. Returns `None` when there are no segments.
    async fn fill_query_context(
        mut query_context: QueryContext,
        segments: LockedSegmentHolder,
        runtime_handle: &AdaptiveSearchHandle,
        is_stopped_guard: &StoppingGuard,
        timeout: Duration,
    ) -> CollectionResult<Option<QueryContext>> {
        let start = Instant::now();

        let segments: Vec<_> = {
            let Some(segments_guard) = segments.try_read_for(timeout) else {
                return Err(CollectionError::timeout(timeout, "fill query context"));
            };
            // Collect the segments first so we don't lock the segment holder during the operations.
            segments_guard
                .non_appendable_then_appendable_segments()
                .collect()
        };

        if segments.is_empty() {
            return Ok(None);
        }

        let cpu_utilization = hw::cpu_utilization();

        // Only the available point count is gathered, too little work per
        // segment to pay for a task each.
        if !query_context.has_statistics_to_collect() {
            let is_stopped = is_stopped_guard.get_is_stopped();
            // Do blocking calls in a blocking task: `segment.get().read()` calls might block async runtime
            let handle = runtime_handle.spawn_blocking(move || {
                cpu_utilization.measure(|| {
                    for segment in &segments {
                        let timeout = timeout.saturating_sub(start.elapsed());
                        fill_segment_query_context(
                            &mut query_context,
                            segment,
                            timeout,
                            &is_stopped,
                        )?;
                    }
                    Ok::<_, OperationError>(query_context)
                })
            });
            return Ok(Some(AbortOnDropHandle::new(handle).await??));
        }

        // Every count is still zero, so each clone carries only its own segment.
        debug_assert_eq!(query_context.available_point_count(), 0);
        let mut forks = segments
            .into_iter()
            .map(|segment| {
                let mut fork = query_context.clone();
                let is_stopped = is_stopped_guard.get_is_stopped();
                let timeout = timeout.saturating_sub(start.elapsed());
                let cpu_utilization = cpu_utilization.clone();
                // Do blocking calls in a blocking task: `segment.get().read()` calls might block async runtime
                let handle = runtime_handle.spawn_blocking(move || {
                    cpu_utilization.measure(|| {
                        fill_segment_query_context(&mut fork, &segment, timeout, &is_stopped)?;
                        Ok::<_, OperationError>(fork)
                    })
                });
                AbortOnDropHandle::new(handle)
            })
            .collect::<FuturesUnordered<_>>();

        while let Some(fork) = forks.try_next().await? {
            query_context.merge_statistics(fork?);
        }
        Ok(Some(query_context))
    }

    pub async fn search(
        segments: LockedSegmentHolder,
        batch_request: Arc<CoreSearchRequestBatch>,
        runtime_handle: &AdaptiveSearchHandle,
        sampling_enabled: bool,
        query_context: QueryContext,
        timeout: Duration,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let started = Instant::now();
        let captured = Self::capture_segments(&segments, timeout)?;
        Self::search_captured(
            captured,
            batch_request,
            runtime_handle,
            sampling_enabled,
            query_context,
            timeout.saturating_sub(started.elapsed()),
        )
        .await
    }

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
        use crate::collection_manager::dli_native_route::{
            CandidateResolution, resolve_pinned_candidates, route_pinned,
        };
        use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, VectorInternal};
        use shard::query::query_enum::QueryEnum;
        use std::sync::atomic::Ordering;

        let started = Instant::now();
        let captured = Self::capture_segments(&holder, timeout)?;
        let attempt_segments = captured.clone();
        let attempt_request = batch_request.clone();
        let attempt_holder = holder.clone();
        let attempt_context = query_context.clone();
        let cpu_utilization = attempt_context.handoff().cpu_utilization();
        let handoff = attempt_context.handoff().clone();
        let attempt_config = config.clone();
        let attempt = runtime_handle.spawn_blocking(move || {
            let _scope = handoff.enter_guard();
            cpu_utilization.measure(|| {
                macro_rules! reject {
                    ($reason:expr) => {{
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
                if guards.len() != attempt_config.members.len() {
                    reject!("topology");
                }
                let mut members = Vec::with_capacity(guards.len());
                for segment in &guards {
                    let Some(&level) = attempt_config.members.get(&segment.uuid) else {
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
                let routed = match route_pinned(
                    &members,
                    query,
                    attempt_config.arity,
                    attempt_config.candidate_budget,
                    &vector_context,
                ) {
                    Ok(routed) => routed,
                    Err(error) => {
                        log::debug!("dli_research: routing fallback: {error}");
                        reject!("routing_error");
                    }
                };
                if !routed.plan.buckets.iter().any(|bucket| bucket.level > 0) {
                    reject!("no_learned_bucket");
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
                let resolution_start = Instant::now();
                let resolution = resolve_pinned_candidates(&members, routed, limit);
                attempt_config.resolution_nanos.fetch_add(
                    u64::try_from(resolution_start.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                match resolution {
                    Ok(CandidateResolution::Resolved(hits)) => Some(hits),
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
                        reject!(reason);
                    }
                    Err(error) => {
                        log::debug!("dli_research: routing fallback: {error}");
                        reject!("resolution_error");
                    }
                }
            })
        });
        if let Some(hits) = AbortOnDropHandle::new(attempt).await? {
            config.learned_queries.fetch_add(1, Ordering::Relaxed);
            return Ok(vec![hits]);
        }
        let mut exact_request = (*batch_request).clone();
        for request in &mut exact_request.searches {
            let params = request.params.get_or_insert_default();
            if params.indexed_only {
                return Err(CollectionError::bad_request(
                    "experimental DLI fallback requires complete segment coverage",
                ));
            }
            params.exact = true;
        }
        config.exact_fallbacks.fetch_add(1, Ordering::Relaxed);
        Self::search_captured(
            captured,
            Arc::new(exact_request),
            runtime_handle,
            false,
            query_context,
            timeout.saturating_sub(started.elapsed()),
        )
        .await
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

    /// Pin the complete holder membership for one search execution. The handles
    /// preserve object lifetime, not a point-in-time snapshot of point contents.
    pub(crate) fn capture_segments(
        holder: &LockedSegmentHolder,
        timeout: Duration,
    ) -> CollectionResult<Vec<LockedSegment>> {
        let Some(guard) = holder.try_read_for(timeout) else {
            return Err(CollectionError::timeout(timeout, "search"));
        };
        Ok(guard.non_appendable_then_appendable_segments().collect())
    }

    /// Run native search and sampling reruns on precisely the supplied handles.
    /// A caller must capture the complete holder set and validate any optional
    /// routing metadata against this same set before using this entry point.
    pub(crate) async fn search_captured(
        segments: Vec<LockedSegment>,
        batch_request: Arc<CoreSearchRequestBatch>,
        runtime_handle: &AdaptiveSearchHandle,
        sampling_enabled: bool,
        query_context: QueryContext,
        timeout: Duration,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let start = Instant::now();
        let query_context_arc = Arc::new(query_context);
        let (locked_segments, searches): (Vec<_>, Vec<_>) = {
            // Probabilistic sampling for the `limit` parameter avoids over-fetching points from segments.
            // e.g. 10 segments with limit 1000 would fetch 10000 points in total and discard 9000 points.
            // With probabilistic sampling we determine a smaller sampling limit for each segment.
            // Use probabilistic sampling if:
            // - sampling is enabled
            // - more than 1 segment
            // - segments are not empty
            let use_sampling = sampling_enabled
                && segments.len() > 1
                && query_context_arc.available_point_count() > 0;

            segments
                .into_iter()
                .map(|segment| {
                    let query_context_arc_segment = query_context_arc.clone();
                    // update timeout
                    let timeout = timeout.saturating_sub(start.elapsed());
                    let search = runtime_handle.spawn_blocking({
                        let (segment, batch_request) = (segment.clone(), batch_request.clone());
                        let cpu_utilization = query_context_arc.handoff().cpu_utilization();
                        move || {
                            cpu_utilization.measure(|| {
                                let segment_query_context =
                                    query_context_arc_segment.get_segment_query_context();

                                search_in_segment(
                                    segment,
                                    batch_request,
                                    use_sampling,
                                    &segment_query_context,
                                    timeout,
                                )
                            })
                        }
                    });

                    // We MUST wrap the search handle in AbortOnDropHandle to ensure that we skip
                    // all searches for futures that are already dropped. Not using this allows
                    // users to create a humongous queue of search tasks, even though the searches
                    // are already invalidated.
                    // See: <https://github.com/qdrant/qdrant/pull/7530>
                    let search = AbortOnDropHandle::new(search);

                    (segment, search)
                })
                .unzip()
        };

        // perform search on all segments concurrently
        // the resulting Vec is in the same order as the segment searches were provided.
        let (all_search_results_per_segment, further_results) =
            Self::execute_searches(searches).await?;
        debug_assert!(all_search_results_per_segment.len() == locked_segments.len());

        let (mut result_aggregator, searches_to_rerun) = Self::process_search_result_step1(
            all_search_results_per_segment,
            batch_request
                .searches
                .iter()
                .map(|request| request.limit + request.offset)
                .collect(),
            &further_results,
        );
        // The second step of the search is to re-run the search without sampling on some segments
        // Expected that this stage will be executed rarely
        if !searches_to_rerun.is_empty() {
            // TODO notify telemetry of failing sampling
            // Ensure consistent order of segment ids
            let searches_to_rerun: Vec<(SegmentOffset, Vec<BatchOffset>)> =
                searches_to_rerun.into_iter().collect();

            let secondary_searches: Vec<_> = {
                let mut res = vec![];
                for (segment_id, batch_ids) in searches_to_rerun.iter() {
                    let query_context_arc_segment = query_context_arc.clone();
                    let segment = locked_segments[*segment_id].clone();
                    let partial_batch_request = Arc::new(CoreSearchRequestBatch {
                        searches: batch_ids
                            .iter()
                            .map(|batch_id| batch_request.searches[*batch_id].clone())
                            .collect(),
                    });
                    // update timeout
                    let timeout = timeout.saturating_sub(start.elapsed());
                    let cpu_utilization = query_context_arc.handoff().cpu_utilization();
                    let handle = runtime_handle.spawn_blocking(move || {
                        cpu_utilization.measure(|| {
                            let segment_query_context =
                                query_context_arc_segment.get_segment_query_context();

                            search_in_segment(
                                segment,
                                partial_batch_request,
                                false,
                                &segment_query_context,
                                timeout,
                            )
                        })
                    });

                    // We MUST wrap the search handle in AbortOnDropHandle to ensure that we skip
                    // all searches for futures that are already dropped. Not using this allows
                    // users to create a humongous queue of search tasks, even though the searches
                    // are already invalidated.
                    // See: <https://github.com/qdrant/qdrant/pull/7530>
                    let handle = AbortOnDropHandle::new(handle);

                    res.push(handle);
                }
                res
            };

            let (secondary_search_results_per_segment, _) =
                Self::execute_searches(secondary_searches).await?;

            result_aggregator.update_point_versions(
                secondary_search_results_per_segment
                    .iter()
                    .flatten()
                    .flatten(),
            );

            for ((_segment_id, batch_ids), segments_result) in searches_to_rerun
                .into_iter()
                .zip(secondary_search_results_per_segment)
            {
                for (batch_id, secondary_batch_result) in batch_ids.into_iter().zip(segments_result)
                {
                    result_aggregator.update_batch_results(batch_id, secondary_batch_result);
                }
            }
        }

        let top_scores: Vec<_> = result_aggregator.into_topk();
        Ok(top_scores)
    }

    /// Retrieve records for the given points ids from the segments
    /// - if payload is enabled, payload will be fetched
    /// - if vector is enabled, vector will be fetched
    ///
    /// The points ids can contain duplicates, the records will be fetched only once
    ///
    /// If an id is not found in the segments, it won't be included in the output.
    #[allow(clippy::too_many_arguments)]
    pub async fn retrieve(
        segments: LockedSegmentHolder,
        points: &[PointIdType],
        with_payload: &WithPayload,
        with_vector: &WithVector,
        runtime_handle: &AdaptiveSearchHandle,
        timeout: Duration,
        deferred_behavior: DeferredBehavior,
    ) -> CollectionResult<AHashMap<PointIdType, RecordInternal>> {
        let stopping_guard = StoppingGuard::new();
        let points = runtime_handle.spawn_blocking({
            let segments = segments.clone();
            let points = points.to_vec();
            let with_payload = with_payload.clone();
            let with_vector = with_vector.clone();
            let is_stopped = stopping_guard.get_is_stopped();
            let handoff = ambient::current();
            // TODO create one Task per segment level retrieve
            move || {
                let _scope = handoff.enter_guard();
                retrieve_blocking(
                    segments,
                    &points,
                    &with_payload,
                    &with_vector,
                    timeout,
                    &is_stopped,
                    deferred_behavior,
                )
            }
        });
        Ok(AbortOnDropHandle::new(points).await??)
    }

    /// Byte-blob analogue of [`Self::retrieve`]: returns vectors as
    /// storage-native bytes ([`SegmentRecordRaw`]), avoiding a lossy
    /// quantization round-trip when relocating points during shard transfer.
    pub async fn retrieve_raw(
        segments: LockedSegmentHolder,
        points: &[PointIdType],
        with_vector: &WithVector,
        runtime_handle: &AdaptiveSearchHandle,
        timeout: Duration,
        deferred_behavior: DeferredBehavior,
    ) -> CollectionResult<AHashMap<PointIdType, SegmentRecordRaw>> {
        let stopping_guard = StoppingGuard::new();
        let points = runtime_handle.spawn_blocking({
            let segments = segments.clone();
            let points = points.to_vec();
            let with_vector = with_vector.clone();
            let is_stopped = stopping_guard.get_is_stopped();
            let handoff = ambient::current();
            move || {
                let _scope = handoff.enter_guard();
                retrieve_raw_blocking(
                    segments,
                    &points,
                    &with_vector,
                    timeout,
                    &is_stopped,
                    deferred_behavior,
                )
            }
        });
        Ok(AbortOnDropHandle::new(points).await??)
    }

    pub async fn read_filtered(
        segments: LockedSegmentHolder,
        filter: Option<&Filter>,
        runtime_handle: &AdaptiveSearchHandle,
        timeout: Option<Duration>,
        deferred_behavior: DeferredBehavior,
    ) -> CollectionResult<BTreeSet<PointIdType>> {
        let stopping_guard = StoppingGuard::new();
        // cloning filter spawning task
        let filter = filter.cloned();
        let handoff = ambient::current();
        let points = runtime_handle.spawn_blocking(move || {
            let is_stopped = stopping_guard.get_is_stopped();

            // Collect the segments first so we don't lock the segment holder during the operations.
            let segments: Vec<_> = {
                match timeout {
                    None => Ok(segments.read()),
                    Some(t) => segments
                        .try_read_for(t)
                        .ok_or_else(|| CollectionError::timeout(t, "read_filtered")),
                }?
                .non_appendable_then_appendable_segments()
                .collect()
            };

            let _scope = handoff.enter_guard();

            let work = || -> CollectionResult<_> {
                let all_points: BTreeSet<_> = segments
                    .into_iter()
                    .map(|segment| {
                        segment.get().read().read_filtered(
                            None,
                            None,
                            filter.as_ref(),
                            &is_stopped,
                            deferred_behavior,
                        )
                    })
                    .process_results(|iter| iter.flatten().collect())?;
                Ok(all_points)
            };

            hw::cpu_utilization().measure(work)
        });
        AbortOnDropHandle::new(points).await?
    }

    /// Rescore results with a formula that can reference payload values.
    ///
    /// Aggregates rescores from the segments.
    pub async fn rescore_with_formula(
        segments: LockedSegmentHolder,
        arc_ctx: Arc<FormulaContext>,
        runtime_handle: &AdaptiveSearchHandle,
        timeout: Duration,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        let limit = arc_ctx.limit;

        let mut futures = {
            let segments: Vec<_> = {
                let Some(segments_guard) = segments.try_read_for(timeout) else {
                    return Err(CollectionError::timeout(timeout, "rescore_with_formula"));
                };
                // Collect the segments first so we don't lock the segment holder during the operations.
                segments_guard
                    .non_appendable_then_appendable_segments()
                    .collect()
            };

            segments
                .into_iter()
                .map(|segment| {
                    let handle = runtime_handle.spawn_blocking({
                        let arc_ctx = arc_ctx.clone();
                        let handoff = ambient::current();
                        let cpu_utilization = hw::cpu_utilization();
                        move || {
                            let _scope = handoff.enter_guard();
                            cpu_utilization
                                .measure(|| segment.get().read().rescore_with_formula(arc_ctx))
                        }
                    });
                    AbortOnDropHandle::new(handle)
                })
                .collect::<FuturesUnordered<_>>()
        };

        let mut segments_results = Vec::with_capacity(futures.len());
        while let Some(result) = futures.try_next().await? {
            segments_results.push(result?)
        }

        // use aggregator with only one "batch"
        let mut aggregator = BatchResultAggregator::new(std::iter::once(limit));
        aggregator.update_point_versions(segments_results.iter().flatten());
        aggregator.update_batch_results(0, segments_results.into_iter().flatten());
        let top =
            aggregator.into_topk().into_iter().next().ok_or_else(|| {
                OperationError::service_error("expected first result of aggregator")
            })?;

        Ok(top)
    }

    /// Rank the shard's points by BM25 over the text index of `field` and
    /// return the `limit` best, highest first.
    ///
    /// `terms` must already be tokenized by the field's tokenizer. The text
    /// statistics are gathered over every segment first, among the points
    /// matching `idf_corpus` (all of them when `None`), so a point scores the
    /// same whichever segment holds it; then each segment returns its own
    /// `limit` best, and a point held by several segments keeps only its
    /// highest version. No sampling: every segment is asked for the full
    /// `limit`.
    #[allow(clippy::too_many_arguments)]
    pub async fn score_bm25(
        segments: LockedSegmentHolder,
        field: JsonPath,
        terms: Vec<String>,
        params: Bm25Params,
        filter: Option<Filter>,
        idf_corpus: Option<&Filter>,
        limit: usize,
        with_payload: WithPayload,
        with_vector: WithVector,
        runtime_handle: &AdaptiveSearchHandle,
        is_stopped_guard: &StoppingGuard,
        timeout: Duration,
    ) -> CollectionResult<Vec<ScoredPoint>> {
        let start = Instant::now();

        let query_context = init_text_query_context(
            &field,
            idf_corpus,
            &terms,
            is_stopped_guard.get_is_stopped(),
        );
        let query_context = Self::fill_query_context(
            query_context,
            segments.clone(),
            runtime_handle,
            is_stopped_guard,
            timeout,
        )
        .await?;
        let Some(query_context) = query_context else {
            return Ok(Vec::new());
        };

        let segments: Vec<_> = {
            let timeout = timeout.saturating_sub(start.elapsed());
            let Some(segments_guard) = segments.try_read_for(timeout) else {
                return Err(CollectionError::timeout(timeout, "score_bm25"));
            };
            // Collect the segments first so we don't lock the segment holder during the operations.
            segments_guard
                .non_appendable_then_appendable_segments()
                .collect()
        };

        let query = Arc::new(SegmentBm25Query {
            query_context,
            field,
            terms,
            filter,
            with_payload,
            with_vector,
        });
        let mut futures = segments
            .into_iter()
            .map(|segment| {
                let query = query.clone();
                let timeout = timeout.saturating_sub(start.elapsed());
                let cpu_utilization = hw::cpu_utilization();
                let handle = runtime_handle.spawn_blocking(move || {
                    cpu_utilization.measure(|| {
                        let SegmentBm25Query {
                            query_context,
                            field,
                            terms,
                            filter,
                            with_payload,
                            with_vector,
                        } = &*query;
                        let segment = segment.get();
                        let Some(read_segment) = segment.try_read_for(timeout) else {
                            return Err(CollectionError::timeout(timeout, "score_bm25"));
                        };
                        let segment_query_context = query_context.get_segment_query_context();
                        Ok(read_segment.score_bm25(
                            field,
                            terms,
                            params,
                            with_payload,
                            with_vector,
                            filter.as_ref(),
                            limit,
                            &segment_query_context,
                        )?)
                    })
                });
                AbortOnDropHandle::new(handle)
            })
            .collect::<FuturesUnordered<_>>();

        let mut segments_results = Vec::with_capacity(futures.len());
        while let Some(result) = futures.try_next().await? {
            segments_results.push(result?)
        }

        // use aggregator with only one "batch"
        let mut aggregator = BatchResultAggregator::new(std::iter::once(limit));
        aggregator.update_point_versions(segments_results.iter().flatten());
        aggregator.update_batch_results(0, segments_results.into_iter().flatten());
        let top =
            aggregator.into_topk().into_iter().next().ok_or_else(|| {
                OperationError::service_error("expected first result of aggregator")
            })?;

        Ok(top)
    }
}

/// What every segment's BM25 task shares, behind one `Arc`.
struct SegmentBm25Query {
    query_context: QueryContext,
    field: JsonPath,
    terms: Vec<String>,
    filter: Option<Filter>,
    with_payload: WithPayload,
    with_vector: WithVector,
}

/// Returns suggested search sampling size for a given number of points and required limit.
fn sampling_limit(
    limit: usize,
    ef_limit: Option<usize>,
    segment_points: usize,
    total_points: usize,
) -> usize {
    // shortcut empty segment
    if segment_points == 0 {
        return 0;
    }
    let segment_probability = segment_points as f64 / total_points as f64;
    let poisson_sampling =
        find_search_sampling_over_point_distribution(limit as f64, segment_probability);

    // if no ef_limit was found, it is a plain index => sampling optimization is not needed.
    let effective = ef_limit.map_or(limit, |ef_limit| {
        effective_limit(limit, ef_limit, poisson_sampling)
    });
    log::trace!(
        "sampling: {effective}, poisson: {poisson_sampling} segment_probability: {segment_probability}, segment_points: {segment_points}, total_points: {total_points}",
    );
    effective
}

/// Determines the effective ef limit value for the given parameters.
fn effective_limit(limit: usize, ef_limit: usize, poisson_sampling: usize) -> usize {
    // Prefer the highest of poisson_sampling/ef_limit, but never be higher than limit
    poisson_sampling.max(ef_limit).min(limit)
}

/// Process sequentially contiguous batches
///
/// # Arguments
///
/// * `segment` - Locked segment to search in
/// * `request` - Batch of search requests
/// * `use_sampling` - If true, try to use probabilistic sampling
/// * `segment_query_context` - Additional context for the search
///
/// # Returns
///
/// Collection Result of:
/// * Vector of ScoredPoints for each request in the batch
/// * Vector of boolean indicating if the segment have further points to search
fn search_in_segment(
    segment: LockedSegment,
    request: Arc<CoreSearchRequestBatch>,
    use_sampling: bool,
    segment_query_context: &SegmentQueryContext,
    timeout: Duration,
) -> CollectionResult<(Vec<Vec<ScoredPoint>>, Vec<bool>)> {
    if segment_query_context.is_stopped() {
        return Err(CollectionError::cancelled(
            "Search in segment was cancelled",
        ));
    }

    let batch_size = request.searches.len();

    let mut result: Vec<Vec<ScoredPoint>> = Vec::with_capacity(batch_size);
    let mut further_results: Vec<bool> = Vec::with_capacity(batch_size); // if segment have more points to return

    for group in group_search_batches(&request.searches) {
        let SearchBatchGroup {
            params,
            query_vectors,
        } = group;

        let (mut res, mut further) = execute_batch_search(
            &segment,
            &query_vectors,
            &params,
            use_sampling,
            segment_query_context,
            timeout,
        )?;
        further_results.append(&mut further);
        result.append(&mut res);
    }

    Ok((result, further_results))
}

fn execute_batch_search(
    segment: &LockedSegment,
    vectors_batch: &[QueryVector],
    search_params: &BatchSearchParams,
    use_sampling: bool,
    segment_query_context: &SegmentQueryContext,
    timeout: Duration,
) -> CollectionResult<(Vec<Vec<ScoredPoint>>, Vec<bool>)> {
    let locked_segment = segment.get();
    let Some(read_segment) = locked_segment.try_read_for(timeout) else {
        return Err(CollectionError::timeout(timeout, "batch search"));
    };

    let segment_points = read_segment.available_point_count_without_deferred();
    let segment_config = read_segment.config();

    let top = if use_sampling {
        let ef_limit = search_params
            .params
            .and_then(|p| p.hnsw_ef)
            .or_else(|| get_hnsw_ef_construct(segment_config, search_params.vector_name));
        sampling_limit(
            search_params.top,
            ef_limit,
            segment_points,
            segment_query_context.available_point_count(),
        )
    } else {
        search_params.top
    };

    let vectors_batch = &vectors_batch.iter().collect_vec();
    let res = read_segment.search_batch(
        search_params.vector_name,
        vectors_batch,
        &search_params.with_payload,
        &search_params.with_vector,
        search_params.filter,
        top,
        search_params.params,
        segment_query_context,
    )?;

    drop(read_segment);

    let further_results = res
        .iter()
        .map(|batch_result| batch_result.len() == top)
        .collect();

    Ok((res, further_results))
}

/// Find the HNSW ef_construct for a named vector
///
/// If the given named vector has no HNSW index, `None` is returned.
fn get_hnsw_ef_construct(config: &SegmentConfig, vector_name: &VectorName) -> Option<usize> {
    config
        .vector_data
        .get(vector_name)
        .and_then(|config| match &config.index {
            Indexes::Plain {} | Indexes::Lmi {} | Indexes::LmiTrained(_) => None,
            Indexes::Hnsw(hnsw) => Some(hnsw),
        })
        .map(|hnsw| hnsw.ef_construct)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::wildcard_enum_match_arm, reason = "test code")]

    use std::assert_matches;
    use std::sync::atomic::AtomicBool;

    use ahash::AHashSet;
    use api::rest::SearchRequestInternal;
    use common::ambient::hw::HwMetric;
    use common::ambient::{AmbientContext, AmbientFutureExt, Handoff};
    use segment::data_types::vectors::DEFAULT_VECTOR_NAME;
    use segment::fixtures::index_fixtures::random_vector;
    use segment::index::VectorIndexEnum;
    use segment::types::{Condition, HasIdCondition};
    use shard::optimizers::config::DEFAULT_INDEXING_THRESHOLD_KB;
    use tempfile::Builder;

    use super::*;
    use crate::collection_manager::fixtures::{TEST_TIMEOUT, build_test_holder, random_segment};
    use crate::collection_manager::holders::segment_holder::SegmentHolder;
    use crate::operations::types::CoreSearchRequest;

    #[test]
    fn test_is_indexed_enough_condition() {
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();

        let segment1 = random_segment(dir.path(), 10, 200, 256);

        let vector_index = segment1
            .vector_data
            .get(DEFAULT_VECTOR_NAME)
            .unwrap()
            .vector_index
            .clone();

        let vector_index_borrow = vector_index.borrow();

        let _scope = ambient::test_guard();

        match &*vector_index_borrow {
            VectorIndexEnum::Plain(plain_index) => {
                let res_1 = plain_index
                    .is_small_enough_for_unindexed_search(25, None)
                    .unwrap();
                assert!(!res_1);

                let res_2 = plain_index
                    .is_small_enough_for_unindexed_search(225, None)
                    .unwrap();
                assert!(res_2);

                let ids: AHashSet<_> = vec![1, 2].into_iter().map(PointIdType::from).collect();

                let ids_filter = Filter::new_must(Condition::HasId(HasIdCondition::from(ids)));

                let res_3 = plain_index
                    .is_small_enough_for_unindexed_search(25, Some(&ids_filter))
                    .unwrap();
                assert!(res_3);
            }
            _ => panic!("Expected plain index"),
        }
    }

    #[tokio::test]
    async fn test_segments_search() {
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();

        let segment_holder = build_test_holder(dir.path());

        let query = vec![1.0, 1.0, 1.0, 1.0];

        let req = CoreSearchRequest {
            query: query.into(),
            with_payload: None,
            with_vector: None,
            filter: None,
            params: None,
            limit: 5,
            score_threshold: None,
            offset: 0,
        };

        let batch_request = CoreSearchRequestBatch {
            searches: vec![req],
        };

        let ctx = AmbientContext::new();
        let result = SegmentsSearcher::search(
            segment_holder,
            Arc::new(batch_request),
            &AdaptiveSearchHandle::current_for_tests(),
            true,
            QueryContext::new(DEFAULT_INDEXING_THRESHOLD_KB, Handoff::Measured(ctx)),
            TEST_TIMEOUT,
        )
        .await
        .unwrap()
        .into_iter()
        .next()
        .unwrap();

        // eprintln!("result = {:?}", &result);

        assert_eq!(result.len(), 5);

        assert!(result[0].id == 3.into() || result[0].id == 11.into());
        assert!(result[1].id == 3.into() || result[1].id == 11.into());
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

    #[tokio::test]
    async fn test_segments_search_sampling() {
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();

        let segment1 = random_segment(dir.path(), 10, 2000, 4);
        let segment2 = random_segment(dir.path(), 10, 4000, 4);

        let mut holder = SegmentHolder::default();

        let _sid1 = holder.add_new(segment1);
        let _sid2 = holder.add_new(segment2);

        let segment_holder = LockedSegmentHolder::new(holder);

        let mut rnd = rand::rng();

        for _ in 0..100 {
            let req1 = SearchRequestInternal {
                vector: random_vector(&mut rnd, 4).into(),
                limit: 150, // more than LOWER_SEARCH_LIMIT_SAMPLING
                offset: None,
                with_payload: None,
                with_vector: None,
                filter: None,
                params: None,
                score_threshold: None,
            };
            let req2 = SearchRequestInternal {
                vector: random_vector(&mut rnd, 4).into(),
                limit: 50, // less than LOWER_SEARCH_LIMIT_SAMPLING
                offset: None,
                filter: None,
                params: None,
                with_payload: None,
                with_vector: None,
                score_threshold: None,
            };

            let batch_request = CoreSearchRequestBatch {
                searches: vec![req1.into(), req2.into()],
            };

            let batch_request = Arc::new(batch_request);

            let ctx = AmbientContext::new();
            let query_context = QueryContext::new(
                DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::clone(&ctx)),
            );

            let result_no_sampling = SegmentsSearcher::search(
                segment_holder.clone(),
                batch_request.clone(),
                &AdaptiveSearchHandle::current_for_tests(),
                false,
                query_context,
                TEST_TIMEOUT,
            )
            .await
            .unwrap();

            assert_ne!(ctx.hw_data()[HwMetric::Cpu], 0);

            let ctx = AmbientContext::new();
            let query_context = QueryContext::new(
                DEFAULT_INDEXING_THRESHOLD_KB,
                Handoff::Measured(AmbientContext::clone(&ctx)),
            );

            assert!(!result_no_sampling.is_empty());

            let result_sampling = SegmentsSearcher::search(
                segment_holder.clone(),
                batch_request,
                &AdaptiveSearchHandle::current_for_tests(),
                true,
                query_context,
                TEST_TIMEOUT,
            )
            .await
            .unwrap();
            assert!(!result_sampling.is_empty());

            assert_ne!(ctx.hw_data()[HwMetric::Cpu], 0);

            // assert equivalence in depth
            assert_eq!(result_no_sampling[0].len(), result_sampling[0].len());
            assert_eq!(result_no_sampling[1].len(), result_sampling[1].len());

            for (no_sampling, sampling) in
                result_no_sampling[0].iter().zip(result_sampling[0].iter())
            {
                assert_eq!(no_sampling.score, sampling.score); // different IDs may have same scores
            }
        }
    }

    #[test]
    fn test_retrieve() {
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
        let segment_holder = build_test_holder(dir.path());
        let records = ambient::test(|| {
            retrieve_blocking(
                segment_holder,
                &[1.into(), 2.into(), 3.into()],
                &WithPayload::from(true),
                &true.into(),
                Duration::from_secs(1),
                &AtomicBool::new(false),
                DeferredBehavior::VisibleOnly,
            )
        })
        .unwrap();
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn test_retrieve_timeout() {
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
        let segment_holder = build_test_holder(dir.path());
        // keep write guard to prevent subsequent read
        let _segment_holder_write_guard = segment_holder.write();
        let records = retrieve_blocking(
            segment_holder.clone(),
            &[1.into(), 2.into(), 3.into()],
            &WithPayload::from(true),
            &true.into(),
            Duration::from_secs(1),
            &AtomicBool::new(false),
            DeferredBehavior::VisibleOnly,
        );
        assert_matches!(records, Err(OperationError::Timeout { .. }));
    }

    #[test]
    fn test_sampling_limit() {
        assert_eq!(sampling_limit(1000, None, 464530, 35103551), 1000);
    }

    #[test]
    fn test_sampling_limit_ef() {
        assert_eq!(sampling_limit(1000, Some(100), 464530, 35103551), 100);
    }

    #[test]
    fn test_sampling_limit_high() {
        assert_eq!(sampling_limit(1000000, None, 464530, 35103551), 1000000);
    }

    /// Tests whether calculating the effective ef limit value is correct.
    ///
    /// Because there was confusion about what the effective value should be for some input
    /// combinations, we decided to write this tests to ensure correctness.
    ///
    /// See: <https://github.com/qdrant/qdrant/pull/1694>
    #[test]
    fn test_effective_limit() {
        // Test cases to assert: (limit, ef_limit, poisson_sampling, effective)
        let tests = [
            (1000, 128, 150, 150),
            (1000, 128, 110, 128),
            (130, 128, 150, 130),
            (130, 128, 110, 128),
            (50, 128, 150, 50),
            (50, 128, 110, 50),
            (500, 1000, 300, 500),
            (500, 400, 300, 400),
            (1000, 0, 150, 150),
            (1000, 0, 110, 110),
        ];
        tests.into_iter().for_each(|(limit, ef_limit, poisson_sampling, effective)| assert_eq!(
            effective_limit(limit, ef_limit, poisson_sampling),
            effective,
            "effective limit for [limit: {limit}, ef_limit: {ef_limit}, poisson_sampling: {poisson_sampling}] must be {effective}",
        ));
    }

    /// A text-indexed segment holding `docs`, each `(point id, op num, text)`.
    fn text_segment(
        path: &std::path::Path,
        docs: impl IntoIterator<Item = (u64, u64, String)>,
    ) -> segment::segment::Segment {
        use segment::data_types::index::TextIndexParams;
        use segment::data_types::vectors::only_default_vector;
        use segment::entry::{NonAppendableSegmentEntry as _, SegmentEntry as _};
        use segment::payload_json;
        use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
        use segment::types::{Distance, PayloadFieldSchema, PayloadSchemaParams};

        let _scope = ambient::test_guard();
        let mut segment = build_simple_segment(path, 4, Distance::Dot).unwrap();
        let params = TextIndexParams {
            phrase_matching: Some(true),
            ..TextIndexParams::default()
        };
        segment
            .create_field_index(
                0,
                &JsonPath::new("text"),
                Some(&PayloadFieldSchema::FieldParams(PayloadSchemaParams::Text(
                    params,
                ))),
            )
            .unwrap();
        for (id, op_num, text) in docs {
            let vector = only_default_vector(&[1.0, 0.0, 0.0, 0.0]);
            segment.upsert_point(op_num, id.into(), vector).unwrap();
            segment
                .set_payload(op_num, id.into(), &payload_json! { "text": text }, &None)
                .unwrap();
        }
        segment
    }

    /// Distinct term counts for every `i` below 105, so no two documents tie.
    /// `text_of(90)`, short and heavy in `gamma`, is among the best matches
    /// for `alpha gamma`.
    fn text_of(i: u64) -> String {
        ["alpha"; 5][..(i % 5 + 1) as usize].join(" ")
            + &" beta".repeat((i % 3) as usize)
            + &" gamma".repeat((i % 7) as usize)
    }

    /// The shard's BM25 ranking against each segment scored with statistics
    /// over both, then merged: the statistics are shard-wide, every segment
    /// contributes its best, and a point both segments return is ranked by
    /// its newer copy only.
    ///
    /// Both copies rank here on purpose. Like a vector search, the merge only
    /// sees the versions the segments return, so a stale copy whose newer one
    /// misses its segment's top `limit` would surface. A shard does not hold
    /// such a pair: an upsert deletes the old copy, and during an optimization
    /// the proxy hides it.
    #[tokio::test]
    async fn test_score_bm25_across_segments() {
        use segment::data_types::index::TextIndexParams;
        use segment::entry::ReadSegmentEntry as _;
        use shard::query::text::init_text_query_context;

        let _scoring = TextIndexParams::override_scoring(true);
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
        let moved = 5;
        let older = text_segment(
            &dir.path().join("a"),
            (0..40).map(|i| (i, i + 1, text_of(i))),
        );
        let newer = text_segment(
            &dir.path().join("b"),
            (40..80)
                .map(|i| (i, i + 1, text_of(i)))
                .chain([(moved, 100, text_of(90))]),
        );

        let field = JsonPath::new("text");
        let terms = vec!["alpha".to_owned(), "gamma".to_owned()];
        let params = Bm25Params::default();
        let limit = 15;

        // Reference: each segment ranked in full against both segments'
        // statistics, the older copy of the moved point dropped.
        let mut query_context =
            ambient::test(|| init_text_query_context(&field, None, &terms, Default::default()));
        older.fill_query_context(&mut query_context).unwrap();
        newer.fill_query_context(&mut query_context).unwrap();
        let segment_context = query_context.get_segment_query_context();
        let rank = |segment: &segment::segment::Segment| {
            segment
                .score_bm25(
                    &field,
                    &terms,
                    params,
                    &WithPayload::from(false),
                    &WithVector::Bool(false),
                    None,
                    usize::MAX,
                    &segment_context,
                )
                .unwrap()
        };
        let mut expected: Vec<ScoredPoint> = rank(&older)
            .into_iter()
            .filter(|point| point.id != moved.into())
            .chain(rank(&newer))
            .collect();
        expected.sort_by(|a, b| b.score.total_cmp(&a.score));
        expected.truncate(limit);
        assert!(
            rank(&older)
                .iter()
                .take(limit)
                .any(|point| point.id == moved.into()),
            "its older copy ranks in its segment too, so the merge has both to choose from",
        );

        let mut holder = SegmentHolder::default();
        holder.add_new(older);
        holder.add_new(newer);
        let actual = SegmentsSearcher::score_bm25(
            LockedSegmentHolder::new(holder),
            field.clone(),
            terms.clone(),
            params,
            None,
            None,
            limit,
            WithPayload::from(false),
            WithVector::Bool(false),
            &AdaptiveSearchHandle::current_for_tests(),
            &StoppingGuard::new(),
            TEST_TIMEOUT,
        )
        .measured(AmbientContext::new())
        .await
        .unwrap();

        let ranked = |points: &[ScoredPoint]| -> Vec<(PointIdType, u64, ScoreType)> {
            points
                .iter()
                .map(|point| (point.id, point.version, point.score))
                .collect()
        };
        assert_eq!(ranked(&actual), ranked(&expected));
        assert!(
            actual
                .iter()
                .any(|point| point.id == moved.into() && point.version == 100),
            "the moved point ranks, by its newer copy",
        );
    }
}
