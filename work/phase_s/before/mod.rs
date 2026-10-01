mod build;
mod config;
#[cfg(feature = "lmi-training")]
mod training;
pub use config::LmiConfig;

/// Whether this binary can construct trained LMI indexes.
pub const fn training_available() -> bool {
    cfg!(feature = "lmi-training")
}
pub use build::LMI_STATE_FILE;
mod lifecycle;
mod read;
pub mod read_only;
mod routing;

pub use routing::{
    LinearLayer, LmiCandidateMode, LmiRoutingState, MlpRouter, RouterLayer, build_router_postings,
};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::common::operation_error::{OperationError, OperationResult};

use atomic_refcell::AtomicRefCell;

use crate::id_tracker::IdTrackerEnum;
use crate::index::plain_vector_index::PlainVectorIndex;
use crate::index::struct_payload_index::StructPayloadIndex;
use crate::vector_storage::VectorStorageEnum;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;

#[derive(Debug)]
pub struct LmiIndex {
    plain: PlainVectorIndex,
    state_path: Option<std::path::PathBuf>,
    routing_distance: Option<crate::types::Distance>,
    candidate_mode: LmiCandidateMode,
    routing_state: Option<LmiRoutingState>,
    quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,

    id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
    vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
}

impl LmiIndex {
    /// Select a transient candidate source. StaticLearned without installation
    /// falls back to Plain. This is deliberately absent
    /// from the persisted configuration: legacy fixtures reopen in AllValidPoints;
    /// trained indexes reopen from their saved model.
    /// Deterministic routing is an architectural fixture, not a learned index.
    pub fn set_candidate_mode(&mut self, mode: LmiCandidateMode) -> OperationResult<()> {
        if self.state_path.is_some() {
            return Err(OperationError::service_error(
                "Candidate mode changes are limited to transient LMI fixtures; rebuild a trained index",
            ));
        }
        self.candidate_mode = mode;
        Ok(())
    }

    /// Experimental static installation only: no serialization or update maintenance.
    /// Build fully before publishing, so validation/cancellation errors preserve
    /// the previous mode and state. Reinstall after any vector replacements/additions.
    pub fn install_static_routing(
        &mut self,
        router: MlpRouter,
        nprobe: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<()> {
        if self.state_path.is_some() {
            return Err(OperationError::service_error(
                "Manual model installation is limited to transient LMI fixtures; rebuild a trained index",
            ));
        }
        let count = router.output_dim()?;
        if nprobe == 0 || nprobe > count {
            return Err(OperationError::service_error(
                "LMI nprobe is outside router bucket range",
            ));
        }
        let postings = build_router_postings(&router, &*self.vector_storage.borrow(), stopped)?;
        let state = LmiRoutingState::new(router, postings, nprobe)?;
        self.routing_state = Some(state);
        self.candidate_mode = LmiCandidateMode::StaticLearned;
        Ok(())
    }

    /// Inspect routing state; no mutable access to its invariants.
    pub fn routing_state(&self) -> Option<&LmiRoutingState> {
        self.routing_state.as_ref()
    }

    pub fn new(
        id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
        vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
        quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
        payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
    ) -> Self {
        eprintln!("[LMI-DUMMY] constructed");

        let plain = PlainVectorIndex::new(
            id_tracker.clone(),
            vector_storage.clone(),
            quantized_vectors.clone(),
            payload_index,
        );

        Self {
            plain,
            state_path: None,
            routing_distance: None,
            candidate_mode: LmiCandidateMode::AllValidPoints,
            routing_state: None,
            quantized_vectors,
            id_tracker,
            vector_storage,
        }
    }
}

#[cfg(all(test, feature = "lmi-training"))]
mod evaluation;
