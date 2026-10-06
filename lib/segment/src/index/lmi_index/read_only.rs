//! Read-only static LMI over Qdrant's read-only tracker and vector storage.
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use atomic_refcell::AtomicRefCell;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{ScoredPointOffset, TelemetryDetail};
use common::universal_io::UniversalReadFs;
use sparse::common::types::DimId;

use super::{LmiConfig, LmiRoutingState, state};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, VectorInternal};
use crate::id_tracker::read_only_tracker_enum::ReadOnlyIdTrackerEnum;
use crate::index::candidate_scoring::score_candidates_generic;
use crate::index::plain_vector_index::read_only::ReadOnlyPlainVectorIndex;
use crate::index::{UniversalReadExt, VectorIndexRead};
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
        payload_index: Arc<
            AtomicRefCell<
                crate::index::struct_payload_index::read_only::ReadOnlyStructPayloadIndex<S>,
            >,
        >,
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
            payload_index,
        )?;
        Ok(Self {
            plain,
            routing,
            distance: vector_config.distance,
            id_tracker,
            vector_storage,
            quantized_vectors,
        })
    }

    fn eligible(
        &self,
        queries: &[&QueryVector],
        filter: Option<&Filter>,
        params: Option<&SearchParams>,
    ) -> bool {
        filter.is_none()
            && params.is_none_or(|p| *p == SearchParams::default())
            && self.quantized_vectors.borrow().is_none()
            && queries
                .iter()
                .all(|q| matches!(q, QueryVector::Nearest(VectorInternal::Dense(_))))
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
        if !self.eligible(vectors, filter, params) {
            return self
                .plain
                .search(vectors, filter, top, params, query_context);
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
                score_candidates_generic(
                    query,
                    &candidates,
                    top,
                    &*tracker,
                    &*storage,
                    query_context,
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
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<usize> {
        self.plain
            .fill_idf_statistics(idf, corpus, stopped, hw_counter)
    }
    fn is_index(&self) -> bool {
        true
    }
}
