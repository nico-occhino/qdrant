//! Scores a caller-provided set of segment-local candidate offsets using Qdrant's
//! authoritative visibility and vector scorer contracts.

use common::types::{DeferredBehavior, PointOffsetType, ScoredPointOffset};

use crate::common::operation_error::OperationResult;
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::QueryVector;
use crate::id_tracker::{IdTrackerEnum, IdTrackerRead};
use crate::index::hnsw_index::point_scorer::BatchFilteredSearcher;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;
use crate::vector_storage::raw_scorer::RawScorerBuilder;
use crate::vector_storage::{VectorStorageEnum, VectorStorageRead};

/// Score unique candidate offsets. This helper does not select the candidates.
///
/// The mapping filter must run before the scorer: the scorer checks point and
/// vector deletion, but it does not by itself enforce deferred visibility.
/// Mutable-segment entry point retained for the existing candidate-scoring seam.
pub fn score_candidates(
    query: &QueryVector,
    candidates: &[PointOffsetType],
    top: usize,
    id_tracker: &IdTrackerEnum,
    vector_storage: &VectorStorageEnum,
    query_context: &VectorQueryContext,
) -> OperationResult<Vec<ScoredPointOffset>> {
    score_candidates_generic(
        query,
        candidates,
        top,
        id_tracker,
        vector_storage,
        query_context,
    )
}

pub fn score_candidates_generic<V, I>(
    query: &QueryVector,
    candidates: &[PointOffsetType],
    top: usize,
    id_tracker: &I,
    vector_storage: &V,
    query_context: &VectorQueryContext,
) -> OperationResult<Vec<ScoredPointOffset>>
where
    V: VectorStorageRead + RawScorerBuilder,
    I: IdTrackerRead,
{
    score_candidates_with_quantization(
        query,
        candidates,
        top,
        id_tracker,
        vector_storage,
        None,
        query_context,
    )
}

/// Native quantized ranking over the same visible segment-local offsets.
/// Approximate scores are diagnostic only: callers must rerank before returning scores.
/// The quantized object must belong to this pinned physical segment generation.
pub fn score_candidates_with_quantization<V, I>(
    query: &QueryVector,
    candidates: &[PointOffsetType],
    top: usize,
    id_tracker: &I,
    vector_storage: &V,
    quantized: Option<&QuantizedVectors>,
    query_context: &VectorQueryContext,
) -> OperationResult<Vec<ScoredPointOffset>>
where
    V: VectorStorageRead + RawScorerBuilder,
    I: IdTrackerRead,
{
    score_candidates_inner(
        query,
        candidates,
        top,
        id_tracker,
        vector_storage,
        quantized,
        query_context,
        None,
    )
}

/// [`score_candidates_generic`] that also returns how many offsets reached the
/// scorer as visible and not deleted, i.e. the vectors actually scored.
pub fn score_candidates_counted<V, I>(
    query: &QueryVector,
    candidates: &[PointOffsetType],
    top: usize,
    id_tracker: &I,
    vector_storage: &V,
    query_context: &VectorQueryContext,
) -> OperationResult<(Vec<ScoredPointOffset>, usize)>
where
    V: VectorStorageRead + RawScorerBuilder,
    I: IdTrackerRead,
{
    let mut scored = 0;
    let result = score_candidates_inner(
        query,
        candidates,
        top,
        id_tracker,
        vector_storage,
        None,
        query_context,
        Some(&mut scored),
    )?;
    Ok((result, scored))
}

#[allow(clippy::too_many_arguments)]
fn score_candidates_inner<V, I>(
    query: &QueryVector,
    candidates: &[PointOffsetType],
    top: usize,
    id_tracker: &I,
    vector_storage: &V,
    quantized: Option<&QuantizedVectors>,
    query_context: &VectorQueryContext,
    scored: Option<&mut usize>,
) -> OperationResult<Vec<ScoredPointOffset>>
where
    V: VectorStorageRead + RawScorerBuilder,
    I: IdTrackerRead,
{
    if top == 0 {
        return Ok(Vec::new());
    }

    let mut candidates = candidates.to_vec();
    candidates.sort_unstable();
    candidates.dedup();
    candidates.retain(|&id| {
        (id as usize) < vector_storage.total_vector_count()
            && (id as usize) < id_tracker.deleted_point_bitslice().len()
    });

    let deleted_points = query_context
        .deleted_points()
        .unwrap_or_else(|| id_tracker.deleted_point_bitslice());
    let scorer = BatchFilteredSearcher::new(
        &[query],
        vector_storage,
        quantized,
        None,
        top,
        deleted_points,
    )?;
    let visible = id_tracker
        .point_mappings()
        .filter_deferred_and_deleted(candidates.into_iter(), DeferredBehavior::VisibleOnly);
    let mut batches = match scored {
        None => scorer.peek_top_iter(visible, &query_context.is_stopped())?,
        Some(scored) => {
            // Count what the scorer will not skip: it rejects deleted points
            // (this mask) and deleted vectors (`NotDeletedChecker`).
            let counted = visible.inspect(|&id| {
                let point_deleted = deleted_points.get(id as usize).is_some_and(|bit| *bit);
                if !point_deleted && !vector_storage.is_deleted_vector(id) {
                    *scored += 1;
                }
            });
            scorer.peek_top_iter(counted, &query_context.is_stopped())?
        }
    };
    Ok(batches
        .pop()
        .expect("one input query yields one result batch"))
}

#[cfg(feature = "lmi-scorer-profile")]
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CandidateScoreProfile {
    pub input_offsets: usize,
    pub bounded_offsets: usize,
    pub deferred_visible_offsets: usize,
    pub scored_vectors: usize,
    pub returned_hits: usize,
    pub preparation_ns: u64,
    pub scorer_init_ns: u64,
    pub scorer_loop_ns: u64,
    pub raw_score_ns: u64,
    pub topk_ns: u64,
}

/// Same native scorer path as `score_candidates_generic`, with opt-in accounting.
#[cfg(feature = "lmi-scorer-profile")]
pub fn score_candidates_profiled<V, I>(
    query: &QueryVector,
    candidates: &[PointOffsetType],
    top: usize,
    id_tracker: &I,
    vector_storage: &V,
    query_context: &VectorQueryContext,
) -> OperationResult<(Vec<ScoredPointOffset>, CandidateScoreProfile)>
where
    V: VectorStorageRead + RawScorerBuilder,
    I: IdTrackerRead,
{
    score_candidates_quantized_profiled(query, candidates, top, id_tracker, vector_storage, None, query_context)
}

/// Request-local successful native scorer accounting, including quantized scoring.
#[cfg(feature = "lmi-scorer-profile")]
pub fn score_candidates_quantized_profiled<V, I>(
    query: &QueryVector,
    candidates: &[PointOffsetType],
    top: usize,
    id_tracker: &I,
    vector_storage: &V,
    quantized: Option<&QuantizedVectors>,
    query_context: &VectorQueryContext,
) -> OperationResult<(Vec<ScoredPointOffset>, CandidateScoreProfile)>
where
    V: VectorStorageRead + RawScorerBuilder,
    I: IdTrackerRead,
{
    use std::time::Instant;
    let tick = Instant::now();
    let input_offsets = candidates.len();
    if top == 0 {
        return Ok((
            Vec::new(),
            CandidateScoreProfile {
                input_offsets,
                bounded_offsets: 0,
                deferred_visible_offsets: 0,
                scored_vectors: 0,
                returned_hits: 0,
                preparation_ns: tick.elapsed().as_nanos() as u64,
                scorer_init_ns: 0,
                scorer_loop_ns: 0,
                raw_score_ns: 0,
                topk_ns: 0,
            },
        ));
    }
    let mut candidates = candidates.to_vec();
    candidates.sort_unstable();
    candidates.dedup();
    candidates.retain(|&id| {
        (id as usize) < vector_storage.total_vector_count()
            && (id as usize) < id_tracker.deleted_point_bitslice().len()
    });
    let bounded_offsets = candidates.len();
    let preparation_ns = tick.elapsed().as_nanos() as u64;
    let tick = Instant::now();
    let deleted_points = query_context
        .deleted_points()
        .unwrap_or_else(|| id_tracker.deleted_point_bitslice());
    let scorer = BatchFilteredSearcher::new(
        &[query],
        vector_storage,
        quantized,
        None,
        top,
        deleted_points,
    )?;
    let scorer_init_ns = tick.elapsed().as_nanos() as u64;
    let mut deferred_visible_offsets = 0usize;
    let visible = id_tracker
        .point_mappings()
        .filter_deferred_and_deleted(candidates.into_iter(), DeferredBehavior::VisibleOnly)
        .inspect(|_| deferred_visible_offsets += 1);
    let (mut batches, scored_vectors, raw_score_ns, topk_ns, scorer_loop_ns) =
        scorer.peek_top_iter_profiled(visible, &query_context.is_stopped())?;
    let hits = batches
        .pop()
        .expect("one input query yields one result batch");
    let profile = CandidateScoreProfile {
        input_offsets,
        bounded_offsets,
        deferred_visible_offsets,
        scored_vectors,
        returned_hits: hits.len(),
        preparation_ns,
        scorer_init_ns,
        scorer_loop_ns,
        raw_score_ns,
        topk_ns,
    };
    Ok((hits, profile))
}
