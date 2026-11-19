//! Read-only static LMI over Qdrant's read-only tracker and vector storage.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use atomic_refcell::AtomicRefCell;
use common::types::{ScoredPointOffset, TelemetryDetail};
use common::universal_io::UniversalReadFs;
use sparse::common::types::DimId;

use super::filtered::{self, FilteredPlan};
use super::{LmiConfig, LmiRoutingState, state};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, VectorInternal};
use crate::id_tracker::read_only_tracker_enum::ReadOnlyIdTrackerEnum;
use crate::index::candidate_scoring::score_candidates_generic;
use crate::index::plain_vector_index::read_only::ReadOnlyPlainVectorIndex;
use crate::index::struct_payload_index::read_only::ReadOnlyStructPayloadIndex;
use crate::index::{PayloadIndexRead, UniversalReadExt, VectorIndexRead};
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Distance, Filter, SearchParams, VectorDataConfig};
use crate::vector_storage::VectorStorageRead;
use crate::vector_storage::quantized::quantized_vectors::ReadOnlyQuantizedVectors;
use crate::vector_storage::read_only::VectorStorageReadEnum;

pub struct ReadOnlyLmiIndex<S: UniversalReadExt + 'static> {
    plain: ReadOnlyPlainVectorIndex<S>,
    routing: LmiRoutingState,
    distance: Distance,
    id_tracker: Arc<AtomicRefCell<ReadOnlyIdTrackerEnum<S>>>,
    vector_storage: Arc<AtomicRefCell<VectorStorageReadEnum<S>>>,
    quantized_vectors: Arc<AtomicRefCell<Option<ReadOnlyQuantizedVectors<S>>>>,
    payload_index: Arc<AtomicRefCell<ReadOnlyStructPayloadIndex<S>>>,
}

impl<S: UniversalReadExt + 'static> ReadOnlyLmiIndex<S> {
    pub fn open<Fs: UniversalReadFs<File = S>>(
        fs: &Fs,
        path: &Path,
        vector_config: &VectorDataConfig,
        config: LmiConfig,
        id_tracker: Arc<AtomicRefCell<ReadOnlyIdTrackerEnum<S>>>,
        vector_storage: Arc<AtomicRefCell<VectorStorageReadEnum<S>>>,
        quantized_vectors: Arc<AtomicRefCell<Option<ReadOnlyQuantizedVectors<S>>>>,
        payload_index: Arc<AtomicRefCell<ReadOnlyStructPayloadIndex<S>>>,
    ) -> OperationResult<Self> {
        let storage = vector_storage.borrow();
        let routing = state::load_state(
            fs,
            path,
            config,
            vector_config,
            storage.datatype(),
            storage.total_vector_count(),
        )?;
        drop(storage);
        let plain = ReadOnlyPlainVectorIndex::open(
            id_tracker.clone(),
            vector_storage.clone(),
            quantized_vectors.clone(),
            payload_index.clone(),
        )?;
        Ok(Self {
            plain,
            routing,
            distance: vector_config.distance,
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
        })
    }

    fn eligible(&self, queries: &[&QueryVector], params: Option<&SearchParams>) -> bool {
        params.is_none_or(|p| *p == SearchParams::default())
            && self.quantized_vectors.borrow().is_none()
            && queries
                .iter()
                .all(|q| matches!(q, QueryVector::Nearest(VectorInternal::Dense(_))))
    }
}

impl<S: UniversalReadExt + 'static> ReadOnlyLmiIndex<S> {
    /// Same adaptive plan as the mutable index; see [`filtered`].
    fn filtered_search(
        &self,
        vectors: &[&QueryVector],
        filter: &Filter,
        top: usize,
        params: Option<&SearchParams>,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        if top == 0 {
            return Ok(vec![Vec::new(); vectors.len()]);
        }
        let tracker = self.id_tracker.borrow();
        let storage = self.vector_storage.borrow();
        let payload_index = self.payload_index.borrow();
        payload_index.with_view(|view| {
            let decision =
                filtered::choose_filtered_plan(&self.routing, &view, filter, &*tracker, &*storage)?;
            match decision.plan {
                FilteredPlan::ExactMatches => {
                    self.plain
                        .search(vectors, Some(filter), top, params, query_context)
                }
                FilteredPlan::RoutedIntersect => {
                    let checker = view.filter_context(filter)?;
                    vectors
                        .iter()
                        .map(|query| {
                            let routed_query = filtered::routing_query(Some(self.distance), query);
                            filtered::routed_filtered_search(
                                &self.routing,
                                &routed_query,
                                query,
                                top,
                                &checker,
                                &*tracker,
                                &*storage,
                                query_context,
                            )
                            .map(|(result, _)| result)
                        })
                        .collect()
                }
            }
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
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        if !self.eligible(vectors, params) {
            return self
                .plain
                .search(vectors, filter, top, params, query_context);
        }
        if let Some(filter) = filter {
            return self.filtered_search(vectors, filter, top, params, query_context);
        }
        let tracker = self.id_tracker.borrow();
        let storage = self.vector_storage.borrow();
        vectors
            .iter()
            .map(|query| {
                let routed_query = match (self.distance, *query) {
                    (Distance::Cosine, QueryVector::Nearest(VectorInternal::Dense(vector))) => {
                        QueryVector::Nearest(VectorInternal::Dense(
                            Distance::Cosine.preprocess_vector::<f32>(vector.clone()),
                        ))
                    }
                    _ => (*query).clone(),
                };
                let candidates = self
                    .routing
                    .candidates_for_query(&routed_query, &query_context.is_stopped())?
                    .ok_or_else(|| {
                        OperationError::service_error(
                            "LMI candidate routing received unsupported query",
                        )
                    })?;
                let result = score_candidates_generic(
                    query,
                    &candidates,
                    top,
                    &*tracker,
                    &*storage,
                    query_context,
                )?;
                super::widen_until_filled(
                    &self.routing,
                    &routed_query,
                    query,
                    top,
                    result,
                    &*tracker,
                    &*storage,
                    query_context,
                    || {},
                )
            })
            .collect()
    }

    fn get_telemetry_data(&self, detail: TelemetryDetail) -> VectorIndexSearchesTelemetry {
        self.plain.get_telemetry_data(detail)
    }
    fn indexed_vector_count(&self) -> usize {
        self.routing.postings().point_count()
    }
    fn size_of_searchable_vectors_in_bytes(&self) -> usize {
        self.plain.size_of_searchable_vectors_in_bytes()
    }
    fn fill_idf_statistics(
        &self,
        idf: &mut HashMap<DimId, usize>,
        corpus: Option<&Filter>,
        stopped: &AtomicBool,
    ) -> OperationResult<usize> {
        self.plain.fill_idf_statistics(idf, corpus, stopped)
    }
    fn is_index(&self) -> bool {
        true
    }
}
