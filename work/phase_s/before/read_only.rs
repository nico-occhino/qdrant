use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{DeferredBehavior, ScoredPointOffset, TelemetryDetail};
use common::universal_io::{UniversalReadFs, read_json_via};
use sparse::common::types::DimId;

use super::build::DiskState;
use super::{LMI_STATE_FILE, LmiConfig, LmiRoutingState};
use crate::common::operation_error::OperationResult;
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, VectorInternal};
use crate::id_tracker::IdTrackerRead;
use crate::index::hnsw_index::point_scorer::BatchFilteredSearcher;
use crate::index::plain_vector_index::read_only::ReadOnlyPlainVectorIndex;
use crate::index::read_only::ReadOnlyVectorIndexOpenArgs;
use crate::index::{UniversalReadExt, VectorIndexRead};
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Distance, Filter, SearchParams, VectorDataConfig};
use crate::vector_storage::VectorStorageRead;
use crate::vector_storage::quantized::quantized_vectors::ReadOnlyQuantizedVectors;

/// Native routing over Qdrant's shared read-only storage; owns no vector corpus.
pub struct ReadOnlyLmiIndex<S: UniversalReadExt> {
    plain: ReadOnlyPlainVectorIndex<S>,
    routing: Option<LmiRoutingState>,
    distance: Distance,
}

impl<S: UniversalReadExt + 'static> ReadOnlyLmiIndex<S> {
    pub(crate) fn open<Fs: UniversalReadFs<File = S>>(
        vector_config: &VectorDataConfig,
        config: LmiConfig,
        args: ReadOnlyVectorIndexOpenArgs<'_, S, Fs>,
    ) -> OperationResult<Self> {
        let state: DiskState = read_json_via(args.fs, &args.path.join(LMI_STATE_FILE))?;
        let routing = state.validate(
            vector_config,
            config,
            &*args.id_tracker.borrow(),
            &*args.vector_storage.borrow(),
        )?;
        let plain = ReadOnlyPlainVectorIndex::open(
            args.id_tracker,
            args.vector_storage,
            args.quantized_vectors,
            args.payload_index,
        )?;
        log::info!(
            "LMI read-only open: learned={}; no training",
            routing.is_some()
        );
        Ok(Self {
            plain,
            routing,
            distance: vector_config.distance,
        })
    }
}

impl<S: UniversalReadExt + 'static> VectorIndexRead for ReadOnlyLmiIndex<S> {
    fn search(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        self.plain.with_view(|view| {
            if filter.is_some()
                || params.is_some_and(|p| *p != SearchParams::default())
                || view.quantized_vectors.is_some()
                || self.routing.is_none()
            {
                return view.search(vectors, filter, top, params, context);
            }
            if top == 0 {
                return Ok(vec![vec![]; vectors.len()]);
            }
            let state = self.routing.as_ref().expect("checked above");
            vectors
                .iter()
                .map(|&query| {
                    let normalized;
                    let route_query = if let QueryVector::Nearest(VectorInternal::Dense(v)) = query
                    {
                        normalized = QueryVector::Nearest(VectorInternal::Dense(
                            self.distance.preprocess_vector::<f32>(v.clone()),
                        ));
                        &normalized
                    } else {
                        query
                    };
                    let Some(candidates) =
                        state.candidates_for_query(route_query, &context.is_stopped())?
                    else {
                        return Ok(view.search(&[query], None, top, None, context)?.remove(0));
                    };
                    let deleted = context
                        .deleted_points()
                        .unwrap_or_else(|| view.id_tracker.deleted_point_bitslice());
                    let searcher = BatchFilteredSearcher::new(
                        &[query],
                        view.vector_storage,
                        None::<&ReadOnlyQuantizedVectors<S>>,
                        None,
                        top,
                        deleted,
                        context.hardware_counter(),
                    )?;
                    eprintln!(
                        "[LMI-READ-ONLY] candidate_source=StaticLearned candidate_count={}",
                        candidates.len()
                    );
                    let candidates = view
                        .id_tracker
                        .point_mappings()
                        .filter_deferred_and_deleted(
                            candidates.into_iter().filter(|&id| {
                                (id as usize) < view.vector_storage.total_vector_count()
                                    && (id as usize)
                                        < view.id_tracker.deleted_point_bitslice().len()
                            }),
                            DeferredBehavior::VisibleOnly,
                        );
                    Ok(searcher
                        .peek_top_iter(candidates, &context.is_stopped())?
                        .remove(0))
                })
                .collect()
        })
    }

    fn get_telemetry_data(&self, detail: TelemetryDetail) -> VectorIndexSearchesTelemetry {
        self.plain.get_telemetry_data(detail)
    }
    fn indexed_vector_count(&self) -> usize {
        self.routing
            .as_ref()
            .map_or(0, |s| s.postings().iter().map(Vec::len).sum())
    }
    fn size_of_searchable_vectors_in_bytes(&self) -> usize {
        self.plain.size_of_searchable_vectors_in_bytes()
    }
    fn fill_idf_statistics(
        &self,
        idf: &mut HashMap<DimId, usize>,
        corpus: Option<&Filter>,
        stopped: &AtomicBool,
        hw: &HardwareCounterCell,
    ) -> OperationResult<usize> {
        self.plain.fill_idf_statistics(idf, corpus, stopped, hw)
    }
    fn is_index(&self) -> bool {
        true
    }
}
