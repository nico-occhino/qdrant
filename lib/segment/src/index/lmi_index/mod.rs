mod lifecycle;
mod read;
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
    candidate_mode: LmiCandidateMode,
    routing_state: Option<LmiRoutingState>,
    quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,

    id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
    vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
}

impl LmiIndex {
    /// Select a transient candidate source. StaticLearned without installation
    /// falls back to Plain. This is deliberately absent
    /// from the persisted configuration: reopening restores AllValidPoints.
    /// Deterministic routing is an architectural fixture, not a learned index.
    pub fn set_candidate_mode(&mut self, mode: LmiCandidateMode) {
        self.candidate_mode = mode;
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

    /// Inspect transient state for experiments; no mutable access to its invariants.
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
            candidate_mode: LmiCandidateMode::AllValidPoints,
            routing_state: None,
            quantized_vectors,
            id_tracker,
            vector_storage,
        }
    }
}
