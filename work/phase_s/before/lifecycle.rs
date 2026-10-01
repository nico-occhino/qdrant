use std::path::PathBuf;

use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;

use super::LmiIndex;
use crate::common::operation_error::OperationResult;
use crate::data_types::vectors::VectorRef;
use crate::index::VectorIndex;

impl VectorIndex for LmiIndex {
    fn files(&self) -> Vec<PathBuf> {
        self.state_path
            .iter()
            .cloned()
            .chain(self.plain.files())
            .collect()
    }

    fn immutable_files(&self) -> Vec<PathBuf> {
        self.state_path
            .iter()
            .cloned()
            .chain(self.plain.immutable_files())
            .collect()
    }

    fn update_vector(
        &mut self,
        id: PointOffsetType,
        vector: Option<VectorRef>,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        if self.state_path.is_some() && vector.is_some() {
            return Err(
                crate::common::operation_error::OperationError::service_error(
                    "Trained LMI requires segment rebuild for vector updates",
                ),
            );
        }
        self.plain.update_vector(id, vector, hw_counter)
    }

    fn update_vector_raw(
        &mut self,
        id: PointOffsetType,
        vector: Option<&[u8]>,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        if self.state_path.is_some() && vector.is_some() {
            return Err(
                crate::common::operation_error::OperationError::service_error(
                    "Trained LMI requires segment rebuild for vector updates",
                ),
            );
        }
        self.plain.update_vector_raw(id, vector, hw_counter)
    }
}
