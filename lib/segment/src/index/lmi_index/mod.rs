//! Minimal LMI index shell for validating Qdrant 1.19 scoring ownership.
//! Training, routing state, postings, persistence, and optimizer hooks are absent.

mod config;
pub use config::LmiConfig;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use atomic_refcell::AtomicRefCell;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset, TelemetryDetail};
use sparse::common::types::DimId;

use crate::common::operation_error::OperationResult;
use crate::data_types::query_context::VectorQueryContext;
#[cfg(feature = "testing")]
use crate::data_types::vectors::VectorInternal;
use crate::data_types::vectors::{QueryVector, VectorRef};
use crate::id_tracker::IdTrackerEnum;
#[cfg(feature = "testing")]
use crate::index::candidate_scoring::score_candidates;
use crate::index::plain_vector_index::PlainVectorIndex;
use crate::index::{VectorIndex, VectorIndexRead};
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Filter, SearchParams};
use crate::vector_storage::VectorStorageEnum;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;

/// Plain-backed physical-index shell. Synthetic candidates exist only under
/// the test feature; production queries take the exact Plain path.
#[derive(Debug)]
pub struct LmiIndex {
    plain: PlainVectorIndex,
    #[cfg(feature = "testing")]
    id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
    #[cfg(feature = "testing")]
    vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
    #[cfg(feature = "testing")]
    quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
    #[cfg(feature = "testing")]
    synthetic_candidates: parking_lot::Mutex<Option<Vec<PointOffsetType>>>,
}

impl LmiIndex {
    pub fn new(
        id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
        vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
        quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
        payload_index: Arc<AtomicRefCell<crate::index::struct_payload_index::StructPayloadIndex>>,
    ) -> Self {
        let plain = PlainVectorIndex::new(
            id_tracker.clone(),
            vector_storage.clone(),
            quantized_vectors.clone(),
            payload_index,
        );
        Self {
            plain,
            #[cfg(feature = "testing")]
            id_tracker,
            #[cfg(feature = "testing")]
            vector_storage,
            #[cfg(feature = "testing")]
            quantized_vectors,
            #[cfg(feature = "testing")]
            synthetic_candidates: parking_lot::Mutex::new(None),
        }
    }

    /// Install deterministic offsets only in test builds. They are never persisted.
    #[cfg(feature = "testing")]
    pub fn set_synthetic_candidates_for_test(&self, candidates: Vec<PointOffsetType>) {
        *self.synthetic_candidates.lock() = Some(candidates);
    }

    #[cfg(feature = "testing")]
    fn supports_candidates(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        params: Option<&SearchParams>,
    ) -> bool {
        filter.is_none()
            && params.is_none_or(|p| *p == SearchParams::default())
            && self.quantized_vectors.borrow().is_none()
            && vectors
                .iter()
                .all(|q| matches!(q, QueryVector::Nearest(VectorInternal::Dense(_))))
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
        #[cfg(feature = "testing")]
        if self.supports_candidates(vectors, filter, params) {
            if let Some(candidates) = self.synthetic_candidates.lock().clone() {
                let tracker = self.id_tracker.borrow();
                let storage = self.vector_storage.borrow();
                return vectors
                    .iter()
                    .map(|query| {
                        score_candidates(query, &candidates, top, &tracker, &storage, query_context)
                    })
                    .collect();
            }
        }
        self.plain
            .search(vectors, filter, top, params, query_context)
    }

    fn get_telemetry_data(&self, detail: TelemetryDetail) -> VectorIndexSearchesTelemetry {
        self.plain.get_telemetry_data(detail)
    }

    fn indexed_vector_count(&self) -> usize {
        self.plain.indexed_vector_count()
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
        false
    }
}

impl VectorIndex for LmiIndex {
    fn files(&self) -> Vec<PathBuf> {
        Vec::new()
    }

    fn update_vector(
        &mut self,
        id: PointOffsetType,
        vector: Option<VectorRef>,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        self.plain.update_vector(id, vector, hw_counter)
    }

    fn update_vector_raw(
        &mut self,
        id: PointOffsetType,
        vector: Option<&[u8]>,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        self.plain.update_vector_raw(id, vector, hw_counter)
    }
}
