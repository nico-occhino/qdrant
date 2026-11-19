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
use crate::index::candidate_scoring::score_candidates_generic;
use crate::index::query_estimator::adjust_to_available_vectors;
use crate::index::sample_estimation::sample_check_cardinality;
use crate::types::{Distance, Filter};
use crate::vector_storage::VectorStorageRead;
use crate::vector_storage::raw_scorer::RawScorerBuilder;

/// Physical plan chosen for one filtered LMI search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FilteredPlan {
    /// Score every matching point exactly; routing is skipped.
    ExactMatches,
    /// Route, intersect the probed postings with the filter, score the matches.
    RoutedIntersect,
}

/// The planner's inputs and outcome, kept for research logs and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
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
    let buckets = state.postings().len().max(1);
    state
        .postings()
        .point_count()
        .saturating_mul(state.nprobe())
        .div_ceil(buckets)
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
    let threshold = probe_volume(state);
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

/// Per-query accounting of the routed plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct RoutedFilterStats {
    /// `nprobe` of the final (returned) probe.
    pub final_nprobe: usize,
    /// Probe doublings needed to fill `min(top, live points)`.
    pub widenings: usize,
    /// Postings selected by the final probe, before the filter.
    pub selected_offsets: usize,
    /// Postings that passed the filter and were handed to the scorer.
    pub matching_offsets: usize,
}

/// Route, keep the postings that satisfy `checker`, and score them; widen the
/// probe until `min(top, live points)` filtered hits are found or every bucket
/// has been probed.
#[allow(clippy::too_many_arguments)]
pub fn routed_filtered_search<V, I, C>(
    state: &LmiRoutingState,
    routed_query: &QueryVector,
    query: &QueryVector,
    top: usize,
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
    let mut stats = RoutedFilterStats {
        final_nprobe: state.nprobe(),
        ..Default::default()
    };
    loop {
        let candidates = state
            .candidates_for_query_with_nprobe(
                routed_query,
                stats.final_nprobe,
                &query_context.is_stopped(),
            )?
            .ok_or_else(|| {
                OperationError::service_error("LMI candidate routing received unsupported query")
            })?;
        let mut matching: Vec<PointOffsetType> = Vec::with_capacity(candidates.len());
        for offset in candidates.iter().copied() {
            if checker.check(offset)? {
                matching.push(offset);
            }
        }
        stats.selected_offsets = candidates.len();
        stats.matching_offsets = matching.len();
        let result = score_candidates_generic(
            query,
            &matching,
            top,
            id_tracker,
            vector_storage,
            query_context,
        )?;
        if result.len() >= needed || stats.final_nprobe >= buckets {
            return Ok((result, stats));
        }
        stats.final_nprobe = stats.final_nprobe.saturating_mul(2).min(buckets);
        stats.widenings += 1;
    }
}
