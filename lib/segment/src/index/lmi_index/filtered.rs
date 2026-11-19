//! Adaptive filtered search over a static learned partition.
//!
//! A payload filter is always enforced before top-k selection; the open
//! question is only *where* it is applied. Two physical plans exist:
//!
//! * [`FilteredPlan::ExactMatches`]: few points match. Enumerate them through
//!   the payload index and score each one exactly (the Plain filtered path).
//!   This is exact and needs no routing.
//! * [`FilteredPlan::RoutedIntersect`]: many points match. Route the query,
//!   keep only the posting offsets that satisfy the filter, and score those.
//!
//! The break-even is the number of vectors each plan scores. An unfiltered
//! probe at the configured `nprobe` touches `V ≈ nprobe · N / B` postings on
//! average (the *probe volume*). If the filter matches `c ≤ V` points, scoring
//! all of them exactly costs no more than the routed probe would already
//! inspect, and it is exact; otherwise the routed plan scores about
//! `c · V / N < c` vectors when filter membership is independent of routing.
//! Cardinality comes from Qdrant's payload index estimate; when the estimate
//! straddles the threshold, the same sampling check that HNSW uses decides.
//!
//! Filling `top` does not imply that the best filtered neighbours were found:
//! a matching neighbour may sit in an unprobed bucket. The routed plan doubles
//! `nprobe` while fewer than `min(top, live points)` filtered hits are found;
//! with every bucket probed the postings cover all build-time points, so the
//! last step equals exact filtered search. Recall above that floor is governed
//! by `nprobe`, exactly as in the unfiltered case.

use common::condition_checker::ConditionChecker;
use common::types::{PointOffsetType, ScoredPointOffset};

use super::LmiRoutingState;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, VectorInternal};
use crate::id_tracker::IdTrackerRead;
use crate::index::PayloadIndexRead;
use crate::index::candidate_scoring::score_candidates_counted;
use crate::index::query_estimator::adjust_to_available_vectors;
use crate::index::sample_estimation::sample_check_cardinality;
use crate::types::{Distance, Filter};
use crate::vector_storage::VectorStorageRead;
use crate::vector_storage::raw_scorer::RawScorerBuilder;

/// Physical plan chosen for one filtered LMI search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilteredPlan {
    /// Score every matching point exactly; routing is skipped.
    ExactMatches,
    /// Route, intersect the probed postings with the filter, score the matches.
    RoutedIntersect,
}

/// The planner's inputs and outcome, kept for research logs and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FilteredPlanDecision {
    pub plan: FilteredPlan,
    /// Probe volume at the configured `nprobe` (the break-even cardinality).
    pub threshold: usize,
    pub estimated_min: usize,
    pub estimated_exp: usize,
    pub estimated_max: usize,
    /// Whether the payload-index estimate was inconclusive and sampling decided.
    pub sampled: bool,
}

/// Average number of postings an unfiltered probe at the configured `nprobe` touches.
pub fn probe_volume(state: &LmiRoutingState) -> usize {
    probe_volume_at(state, state.nprobe())
}

/// Average number of postings an unfiltered probe at `nprobe` touches.
pub fn probe_volume_at(state: &LmiRoutingState, nprobe: usize) -> usize {
    let buckets = state.postings().len().max(1);
    state
        .postings()
        .point_count()
        .saturating_mul(nprobe)
        .div_ceil(buckets)
}

/// The payload-index estimate of a filter, as every planner sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FilterEstimate {
    pub min: usize,
    pub exp: usize,
    pub max: usize,
    /// Live vectors the estimate was adjusted to.
    pub available: usize,
    /// Whether exact filtered search can iterate a payload index (primary
    /// clauses) rather than scanning every point with the filter.
    pub primary_iteration: bool,
}

/// Payload-index estimate of `filter`, adjusted to the live vectors.
pub fn estimate_filter<P, I, V>(
    payload_index: &P,
    filter: &Filter,
    id_tracker: &I,
    vector_storage: &V,
) -> OperationResult<FilterEstimate>
where
    P: PayloadIndexRead,
    I: IdTrackerRead,
    V: VectorStorageRead,
{
    let available = vector_storage.available_vector_count();
    let cardinality = adjust_to_available_vectors(
        payload_index.estimate_cardinality(filter)?,
        available,
        id_tracker.available_point_count(),
    );
    Ok(FilterEstimate {
        min: cardinality.min,
        exp: cardinality.exp,
        max: cardinality.max,
        available,
        primary_iteration: !cardinality.primary_clauses.is_empty(),
    })
}

/// Choose between exact scoring of the matching points and routed intersection.
pub fn choose_filtered_plan<P, I, V>(
    state: &LmiRoutingState,
    payload_index: &P,
    filter: &Filter,
    id_tracker: &I,
    vector_storage: &V,
) -> OperationResult<FilteredPlanDecision>
where
    P: PayloadIndexRead,
    I: IdTrackerRead,
    V: VectorStorageRead,
{
    choose_filtered_plan_with_threshold(
        payload_index,
        filter,
        id_tracker,
        vector_storage,
        probe_volume(state),
    )
}

/// The Baseline break-even rule with an explicit threshold (a probe volume).
pub fn choose_filtered_plan_with_threshold<P, I, V>(
    payload_index: &P,
    filter: &Filter,
    id_tracker: &I,
    vector_storage: &V,
    threshold: usize,
) -> OperationResult<FilteredPlanDecision>
where
    P: PayloadIndexRead,
    I: IdTrackerRead,
    V: VectorStorageRead,
{
    let available_vectors = vector_storage.available_vector_count();
    let cardinality = adjust_to_available_vectors(
        payload_index.estimate_cardinality(filter)?,
        available_vectors,
        id_tracker.available_point_count(),
    );
    let decision = |plan, sampled| FilteredPlanDecision {
        plan,
        threshold,
        estimated_min: cardinality.min,
        estimated_exp: cardinality.exp,
        estimated_max: cardinality.max,
        sampled,
    };
    if cardinality.max <= threshold {
        return Ok(decision(FilteredPlan::ExactMatches, false));
    }
    if cardinality.min > threshold {
        return Ok(decision(FilteredPlan::RoutedIntersect, false));
    }
    let filter_context = payload_index.filter_context(filter)?;
    let many = sample_check_cardinality(
        id_tracker.sample_ids(Some(vector_storage.deleted_vector_bitslice())),
        |offset| filter_context.check(offset),
        threshold,
        available_vectors,
    )?;
    let plan = if many {
        FilteredPlan::RoutedIntersect
    } else {
        FilteredPlan::ExactMatches
    };
    Ok(decision(plan, true))
}

/// Query vector as the router sees it: cosine queries are normalised, as the
/// stored vectors were at build time.
pub fn routing_query(distance: Option<Distance>, query: &QueryVector) -> QueryVector {
    match (distance, query) {
        (Some(Distance::Cosine), QueryVector::Nearest(VectorInternal::Dense(vector))) => {
            QueryVector::Nearest(VectorInternal::Dense(
                Distance::Cosine.preprocess_vector::<f32>(vector.clone()),
            ))
        }
        _ => query.clone(),
    }
}

/// Candidate work of one probe, or summed over several probes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProbeCounts {
    /// Posting offsets visited in the selected buckets, before de-duplication.
    pub raw_offsets: usize,
    /// Distinct candidate offsets after de-duplication.
    pub unique_candidates: usize,
    /// Filter evaluations performed (one per unique candidate).
    pub filter_evaluations: usize,
    /// Candidates that satisfied the filter.
    pub surviving: usize,
    /// Surviving candidates that were visible and not deleted, i.e. scored.
    pub scored: usize,
}

impl ProbeCounts {
    fn add(&mut self, other: &Self) {
        self.raw_offsets += other.raw_offsets;
        self.unique_candidates += other.unique_candidates;
        self.filter_evaluations += other.filter_evaluations;
        self.surviving += other.surviving;
        self.scored += other.scored;
    }
}

/// Wall time per stage, summed over all probes. Research builds only: the
/// timers themselves cost time, so these never serve as headline latency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageNanos {
    /// Router inference, bucket ranking, posting gather and de-duplication.
    pub route: u64,
    /// Filter evaluation over the unique candidates.
    pub filter: u64,
    /// Visibility checks, scoring and top-k selection.
    pub score: u64,
}

/// Per-query accounting of the routed plan.
///
/// Each widening re-routes and re-evaluates from scratch, so the work of the
/// query is `cumulative`; `final_probe` is the work of the probe whose result
/// was returned. Without widening the two are equal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RoutedFilterStats {
    /// `nprobe` of the first probe.
    pub initial_nprobe: usize,
    /// `nprobe` of the final (returned) probe.
    pub final_nprobe: usize,
    /// Probe doublings needed to fill `min(top, live points)`.
    pub widenings: usize,
    pub final_probe: ProbeCounts,
    pub cumulative: ProbeCounts,
    /// Present only in `lmi-research` builds.
    pub stage_ns: Option<StageNanos>,
}

/// Route, keep the postings that satisfy `checker`, and score them; widen the
/// probe until `min(top, live points)` filtered hits are found or every bucket
/// has been probed. The first probe uses `initial_nprobe`; the Baseline planner
/// passes the index's configured `nprobe`.
#[allow(clippy::too_many_arguments)]
pub fn routed_filtered_search<V, I, C>(
    state: &LmiRoutingState,
    routed_query: &QueryVector,
    query: &QueryVector,
    top: usize,
    initial_nprobe: usize,
    checker: &C,
    id_tracker: &I,
    vector_storage: &V,
    query_context: &VectorQueryContext,
) -> OperationResult<(Vec<ScoredPointOffset>, RoutedFilterStats)>
where
    V: VectorStorageRead + RawScorerBuilder,
    I: IdTrackerRead,
    C: ConditionChecker<Error = OperationError>,
{
    let needed = top.min(id_tracker.available_point_count());
    let buckets = state.postings().len();
    let initial_nprobe = initial_nprobe.clamp(1, buckets.max(1));
    let mut stats = RoutedFilterStats {
        initial_nprobe,
        final_nprobe: initial_nprobe,
        ..Default::default()
    };
    #[cfg(feature = "lmi-research")]
    let mut stages = StageNanos::default();
    loop {
        #[cfg(feature = "lmi-research")]
        let tick = std::time::Instant::now();
        let (candidates, raw_offsets) = state
            .candidates_with_raw_count(
                routed_query,
                stats.final_nprobe,
                &query_context.is_stopped(),
            )?
            .ok_or_else(|| {
                OperationError::service_error("LMI candidate routing received unsupported query")
            })?;
        #[cfg(feature = "lmi-research")]
        let tick = {
            stages.route += tick.elapsed().as_nanos() as u64;
            std::time::Instant::now()
        };
        let mut matching: Vec<PointOffsetType> = Vec::with_capacity(candidates.len());
        for offset in candidates.iter().copied() {
            if checker.check(offset)? {
                matching.push(offset);
            }
        }
        #[cfg(feature = "lmi-research")]
        let tick = {
            stages.filter += tick.elapsed().as_nanos() as u64;
            std::time::Instant::now()
        };
        let (result, scored) = score_candidates_counted(
            query,
            &matching,
            top,
            id_tracker,
            vector_storage,
            query_context,
        )?;
        #[cfg(feature = "lmi-research")]
        {
            stages.score += tick.elapsed().as_nanos() as u64;
            stats.stage_ns = Some(stages);
        }
        stats.final_probe = ProbeCounts {
            raw_offsets,
            unique_candidates: candidates.len(),
            filter_evaluations: candidates.len(),
            surviving: matching.len(),
            scored,
        };
        stats.cumulative.add(&stats.final_probe);
        if result.len() >= needed || stats.final_nprobe >= buckets {
            return Ok((result, stats));
        }
        stats.final_nprobe = stats.final_nprobe.saturating_mul(2).min(buckets);
        stats.widenings += 1;
    }
}

/// How a research call selects its plan.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilteredSearchMode {
    /// The committed planner, exactly as the production `search()` runs it.
    Baseline,
    /// Score every matching point exactly (the Plain filtered path).
    ForceExact,
    /// Route at `nprobe`, intersect with the filter, widen only to fill.
    ForceRouted { nprobe: usize },
}

/// What one research call did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FilteredSearchRecord {
    pub plan: FilteredPlan,
    /// Initial probe count of a routed plan (the configured one otherwise).
    pub nprobe: usize,
    /// Break-even decision, when the mode consulted it.
    pub decision: Option<FilteredPlanDecision>,
    /// Candidate accounting of a routed plan.
    pub routed: Option<RoutedFilterStats>,
}
