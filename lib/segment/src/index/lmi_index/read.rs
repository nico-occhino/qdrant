use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{DeferredBehavior, PointOffsetType, ScoredPointOffset, TelemetryDetail};
use sparse::common::types::DimId;

use super::routing::{fake_postings, fake_route};
use super::{LmiCandidateMode, LmiIndex};
use crate::common::operation_error::OperationResult;
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::QueryVector;
use crate::id_tracker::IdTrackerRead;
use crate::index::VectorIndexRead;
use crate::index::hnsw_index::point_scorer::BatchFilteredSearcher;
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Filter, SearchParams};
use crate::vector_storage::VectorStorageRead;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;

impl LmiIndex {
    fn score_candidates_for_query(
        &self,
        query: &QueryVector,
        candidates: impl Iterator<Item = PointOffsetType>,
        top: usize,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if top == 0 {
            return Ok(vec![]);
        }
        let id_tracker = self.id_tracker.borrow();
        let vector_storage = self.vector_storage.borrow();
        // Point and vector deletion are distinct. Never substitute the storage's
        // vector-deletion mask for this point-level mask.
        let deleted_points = query_context
            .deleted_points()
            .unwrap_or_else(|| id_tracker.deleted_point_bitslice());
        let batch_searcher = BatchFilteredSearcher::new(
            &[query],
            &*vector_storage,
            None::<&QuantizedVectors>,
            None,
            top,
            deleted_points,
            query_context.hardware_counter(),
        )?;
        // External postings must also obey deferred visibility. The searcher
        // subsequently applies point/context and vector deletion checks.
        let candidates = id_tracker.point_mappings().filter_deferred_and_deleted(
            candidates.filter(|&id| {
                (id as usize) < vector_storage.total_vector_count()
                    && (id as usize) < id_tracker.deleted_point_bitslice().len()
            }),
            DeferredBehavior::VisibleOnly,
        );
        let results = batch_searcher.peek_top_iter(candidates, &query_context.is_stopped())?;
        Ok(results
            .into_iter()
            .next()
            .expect("one query produces one result"))
    }
}

impl VectorIndexRead for LmiIndex {
    fn search(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        eprintln!(
            "[LMI-DUMMY] search batch={} top={} filtered={} params={}",
            vectors.len(),
            top,
            filter.is_some(),
            params.is_some(),
        );

        // Quantized scoring/rescoring remains owned by Plain in this phase.
        if filter.is_some()
            || params.is_some_and(|p| self.state_path.is_none() || *p != SearchParams::default())
            || self.quantized_vectors.borrow().is_some()
        {
            return self
                .plain
                .search(vectors, filter, top, params, query_context);
        }
        if top == 0 || vectors.is_empty() {
            return Ok(vec![vec![]; vectors.len()]);
        }

        let postings = match self.candidate_mode {
            LmiCandidateMode::AllValidPoints | LmiCandidateMode::StaticLearned => None,
            LmiCandidateMode::DeterministicTwoBuckets => Some(fake_postings(
                self.vector_storage.borrow().total_vector_count(),
                &query_context.is_stopped(),
            )?),
        };
        let mut results = Vec::with_capacity(vectors.len());
        for &query in vectors {
            let candidates = match self.candidate_mode {
                LmiCandidateMode::StaticLearned => match &self.routing_state {
                    Some(state) => {
                        let normalized;
                        let route_query = if let (
                            Some(distance),
                            QueryVector::Nearest(
                                crate::data_types::vectors::VectorInternal::Dense(vector),
                            ),
                        ) = (self.routing_distance, query)
                        {
                            normalized = QueryVector::Nearest(
                                crate::data_types::vectors::VectorInternal::Dense(
                                    distance.preprocess_vector::<f32>(vector.clone()),
                                ),
                            );
                            &normalized
                        } else {
                            query
                        };
                        state.candidates_for_query(route_query, &query_context.is_stopped())?
                    }
                    None => None, // Safe exact fallback when no model was installed.
                },
                LmiCandidateMode::DeterministicTwoBuckets => fake_route(query)
                    .and_then(|bucket| postings.as_ref().map(|p| p[bucket].clone())),
                LmiCandidateMode::AllValidPoints => {
                    eprintln!("[LMI-SCAFFOLD] candidate_source=all_valid_points");
                    let tracker = self.id_tracker.borrow();
                    let deleted = query_context
                        .deleted_points()
                        .unwrap_or_else(|| tracker.deleted_point_bitslice());
                    let candidates = deleted.iter_zeros().map(|id| id as PointOffsetType);
                    results.push(self.score_candidates_for_query(
                        query,
                        candidates,
                        top,
                        query_context,
                    )?);
                    continue;
                }
            };
            if let Some(candidates) = candidates {
                eprintln!(
                    "[LMI-SCAFFOLD] candidate_source={:?} candidate_count={}",
                    self.candidate_mode,
                    candidates.len()
                );
                results.push(self.score_candidates_for_query(
                    query,
                    candidates.into_iter(),
                    top,
                    query_context,
                )?);
            } else {
                results.extend(
                    self.plain
                        .search(&[query], None, top, None, query_context)?,
                );
            }
        }
        Ok(results)
    }

    fn get_telemetry_data(&self, detail: TelemetryDetail) -> VectorIndexSearchesTelemetry {
        self.plain.get_telemetry_data(detail)
    }

    fn indexed_vector_count(&self) -> usize {
        self.routing_state.as_ref().map_or_else(
            || self.plain.indexed_vector_count(),
            |s| s.postings().iter().map(Vec::len).sum(),
        )
    }

    fn size_of_searchable_vectors_in_bytes(&self) -> usize {
        self.plain.size_of_searchable_vectors_in_bytes()
    }

    fn fill_idf_statistics(
        &self,
        idf: &mut HashMap<DimId, usize>,
        corpus: Option<&Filter>,
        is_stopped: &AtomicBool,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<usize> {
        self.plain
            .fill_idf_statistics(idf, corpus, is_stopped, hw_counter)
    }

    fn is_index(&self) -> bool {
        true
    }
}
