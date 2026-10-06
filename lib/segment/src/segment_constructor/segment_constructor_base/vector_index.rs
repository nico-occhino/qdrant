use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use atomic_refcell::AtomicRefCell;
use common::budget::ResourcePermit;
use common::flags::FeatureFlags;
use common::progress_tracker::ProgressTracker;
use rand::Rng;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::id_tracker::IdTrackerEnum;
use crate::index::VectorIndexEnum;
use crate::index::hnsw_index::gpu::gpu_devices_manager::LockedGpuDevice;
use crate::index::hnsw_index::hnsw::{HNSWIndex, HnswIndexOpenArgs};
use crate::index::lmi_index::LmiIndex;
use crate::index::plain_vector_index::PlainVectorIndex;
use crate::index::struct_payload_index::StructPayloadIndex;
use crate::types::{HnswGlobalConfig, Indexes, VectorDataConfig, VectorStorageType};
use crate::vector_storage::VectorStorageEnum;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;

pub(crate) struct VectorIndexOpenArgs<'a> {
    pub path: &'a Path,
    pub id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
    pub vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
    pub payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
    pub quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
}

pub struct VectorIndexBuildArgs<'a, R: Rng + ?Sized> {
    pub permit: Arc<ResourcePermit>,
    /// Vector indices from other segments, used to speed up index building.
    /// May or may not contain the same vectors.
    pub old_indices: &'a [Arc<AtomicRefCell<VectorIndexEnum>>],
    pub gpu_device: Option<&'a LockedGpuDevice<'a>>,
    pub rng: &'a mut R,
    pub stopped: &'a AtomicBool,
    pub hnsw_global_config: &'a HnswGlobalConfig,
    pub feature_flags: FeatureFlags,
    /// Write in graph-with-vectors format.
    /// If set, but not supported, the build will fail.
    /// See: [`VectorDataConfig::inline_vectors_in_graph`]
    pub inline_vectors: bool,
    pub progress: ProgressTracker,
}

pub(crate) fn open_vector_index(
    vector_config: &VectorDataConfig,
    open_args: VectorIndexOpenArgs,
) -> OperationResult<VectorIndexEnum> {
    if matches!(
        vector_config.index,
        Indexes::Lmi {} | Indexes::LmiTrained(_)
    ) && (vector_config.storage_type == VectorStorageType::GraphInline
        || vector_config.multivector_config.is_some())
    {
        return Err(OperationError::service_error(
            "LMI shell requires independent dense single-vector storage; GraphInline and multivectors are unsupported",
        ));
    }
    let VectorIndexOpenArgs {
        path,
        id_tracker,
        vector_storage,
        payload_index,
        quantized_vectors,
    } = open_args;
    Ok(match &vector_config.index {
        Indexes::Plain {} => VectorIndexEnum::Plain(PlainVectorIndex::new(
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
        )),
        Indexes::Lmi {} => VectorIndexEnum::Lmi(LmiIndex::new(
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
        )),
        Indexes::LmiTrained(_) => {
            return Err(OperationError::service_error(
                "Trained LMI persistence is not ported to Qdrant 1.19.2",
            ));
        }
        Indexes::Hnsw(hnsw_config) => VectorIndexEnum::Hnsw(HNSWIndex::open(HnswIndexOpenArgs {
            path,
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
            hnsw_config: *hnsw_config,
        })?),
    })
}

pub(crate) fn build_vector_index<R: Rng + ?Sized>(
    vector_config: &VectorDataConfig,
    open_args: VectorIndexOpenArgs,
    build_args: VectorIndexBuildArgs<R>,
) -> OperationResult<VectorIndexEnum> {
    if matches!(
        vector_config.index,
        Indexes::Lmi {} | Indexes::LmiTrained(_)
    ) && (vector_config.storage_type == VectorStorageType::GraphInline
        || vector_config.multivector_config.is_some())
    {
        return Err(OperationError::service_error(
            "LMI shell requires independent dense single-vector storage; GraphInline and multivectors are unsupported",
        ));
    }
    let VectorIndexOpenArgs {
        path,
        id_tracker,
        vector_storage,
        payload_index,
        quantized_vectors,
    } = open_args;
    Ok(match &vector_config.index {
        Indexes::Plain {} => VectorIndexEnum::Plain(PlainVectorIndex::new(
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
        )),
        Indexes::Lmi {} => VectorIndexEnum::Lmi(LmiIndex::new(
            id_tracker,
            vector_storage,
            quantized_vectors,
            payload_index,
        )),
        Indexes::LmiTrained(_) => {
            return Err(OperationError::service_error(
                "Trained LMI build is not ported to Qdrant 1.19.2",
            ));
        }
        Indexes::Hnsw(hnsw_config) => VectorIndexEnum::Hnsw(HNSWIndex::build(
            HnswIndexOpenArgs {
                path,
                id_tracker,
                vector_storage,
                quantized_vectors,
                payload_index,
                hnsw_config: *hnsw_config,
            },
            build_args,
        )?),
    })
}
