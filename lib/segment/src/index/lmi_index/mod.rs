//! Static LMI index: pure-Rust serving with optional build-only Torch training.
//! Segment optimizer publication is outside this module and remains a separate stage.

#[cfg(feature = "lmi-training")]
mod build;
#[cfg(feature = "lmi-training")]
mod build_plan;
mod config;
mod postings;
pub mod read_only;
mod routing;
#[cfg(feature = "lmi-training")]
mod spherical_kmeans;
mod state;
#[cfg(feature = "lmi-training")]
mod training;
pub use config::LmiConfig;

/// Whether this binary can build new trained LMI segments. Serving does not require it.
pub const fn training_available() -> bool {
    cfg!(feature = "lmi-training")
}
pub use postings::CompactPostings;
pub use routing::{LinearLayer, LmiRoutingState, MlpRouter, RouterLayer};
#[cfg(feature = "testing")]
pub use state::save_fixture_state;
pub use state::{
    LMI_POSTINGS_FILE, LMI_ROUTER_FILE, LMI_STATE_FILE, LmiStateMetadata, RouterPreprocessing,
};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use atomic_refcell::AtomicRefCell;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::{PointOffsetType, ScoredPointOffset, TelemetryDetail};
use sparse::common::types::DimId;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::VectorInternal;
use crate::data_types::vectors::{QueryVector, VectorRef};
use crate::id_tracker::IdTrackerEnum;
use crate::index::candidate_scoring::score_candidates;
use crate::index::plain_vector_index::PlainVectorIndex;
use crate::index::{VectorIndex, VectorIndexRead};
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Distance, Filter, SearchParams, VectorDataConfig};
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;
use crate::vector_storage::{VectorStorageEnum, VectorStorageRead};

/// Plain-backed physical-index shell. Synthetic candidates exist only under
/// the test feature; production queries take the exact Plain path.
#[derive(Debug)]
pub struct LmiIndex {
    plain: PlainVectorIndex,
    id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
    vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
    quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
    #[cfg(feature = "testing")]
    synthetic_candidates: parking_lot::Mutex<Option<Vec<PointOffsetType>>>,
    routing_state: Option<LmiRoutingState>,
    routing_distance: Option<Distance>,
    state_files: Vec<PathBuf>,
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
            id_tracker,
            vector_storage,
            quantized_vectors,
            #[cfg(feature = "testing")]
            synthetic_candidates: parking_lot::Mutex::new(None),
            routing_state: None,
            routing_distance: None,
            state_files: Vec::new(),
        }
    }

    pub(crate) fn open_trained(
        path: &Path,
        vector_config: &VectorDataConfig,
        config: LmiConfig,
        id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
        vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
        quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
        payload_index: Arc<AtomicRefCell<crate::index::struct_payload_index::StructPayloadIndex>>,
    ) -> OperationResult<Self> {
        let storage = vector_storage.borrow();
        let routing_state = state::load_local(
            path,
            config,
            vector_config,
            storage.datatype(),
            storage.total_vector_count(),
        )?;
        drop(storage);
        let mut index = Self::new(id_tracker, vector_storage, quantized_vectors, payload_index);
        index.routing_state = Some(routing_state);
        index.routing_distance = Some(vector_config.distance);
        index.state_files = state::state_files(path);
        Ok(index)
    }

    pub fn routing_state(&self) -> Option<&LmiRoutingState> {
        self.routing_state.as_ref()
    }

    /// Resident auxiliary index data only; authoritative vectors belong to VectorStorage.
    pub(crate) fn auxiliary_heap_bytes(&self) -> u64 {
        self.routing_state
            .as_ref()
            .map_or(0, |state| state.heap_bytes() as u64)
    }

    pub(crate) fn state_file_paths(&self) -> Vec<PathBuf> {
        self.state_files.clone()
    }

    /// Install deterministic offsets only in test builds. They are never persisted.
    #[cfg(feature = "testing")]
    pub fn set_synthetic_candidates_for_test(&self, candidates: Vec<PointOffsetType>) {
        *self.synthetic_candidates.lock() = Some(candidates);
    }

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
        if self.supports_candidates(vectors, filter, params) {
            if let Some(state) = &self.routing_state {
                let tracker = self.id_tracker.borrow();
                let storage = self.vector_storage.borrow();
                return vectors
                    .iter()
                    .map(|query| {
                        let routed_query = match (self.routing_distance, *query) {
                            (
                                Some(Distance::Cosine),
                                QueryVector::Nearest(VectorInternal::Dense(vector)),
                            ) => QueryVector::Nearest(VectorInternal::Dense(
                                Distance::Cosine.preprocess_vector::<f32>(vector.clone()),
                            )),
                            _ => (*query).clone(),
                        };
                        let candidates = state
                            .candidates_for_query(&routed_query, &query_context.is_stopped())?
                            .ok_or_else(|| {
                                OperationError::service_error(
                                    "LMI candidate routing received unsupported query",
                                )
                            })?;
                        score_candidates(
                            query,
                            &candidates,
                            top,
                            &*tracker,
                            &*storage,
                            query_context,
                        )
                    })
                    .collect();
            }
            #[cfg(feature = "testing")]
            if let Some(candidates) = self.synthetic_candidates.lock().clone() {
                let tracker = self.id_tracker.borrow();
                let storage = self.vector_storage.borrow();
                return vectors
                    .iter()
                    .map(|query| {
                        score_candidates(
                            query,
                            &candidates,
                            top,
                            &*tracker,
                            &*storage,
                            query_context,
                        )
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
        self.routing_state.as_ref().map_or_else(
            || self.plain.indexed_vector_count(),
            |state| state.postings().point_count(),
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
        self.routing_state.is_some()
    }
}

impl VectorIndex for LmiIndex {
    fn files(&self) -> Vec<PathBuf> {
        self.state_files.clone()
    }

    fn immutable_files(&self) -> Vec<PathBuf> {
        self.state_files.clone()
    }

    fn update_vector(
        &mut self,
        id: PointOffsetType,
        vector: Option<VectorRef>,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        if self.routing_state.is_some() && vector.is_some() {
            return Err(OperationError::service_error(
                "Trained LMI requires segment rebuild for vector updates",
            ));
        }
        self.plain.update_vector(id, vector, hw_counter)
    }

    fn update_vector_raw(
        &mut self,
        id: PointOffsetType,
        vector: Option<&[u8]>,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        if self.routing_state.is_some() && vector.is_some() {
            return Err(OperationError::service_error(
                "Trained LMI requires segment rebuild for vector updates",
            ));
        }
        self.plain.update_vector_raw(id, vector, hw_counter)
    }
}
