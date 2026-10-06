//! Versioned, serving-only persisted LMI state. Files belong to one segment generation.
use std::path::{Path, PathBuf};

#[cfg(feature = "testing")]
use common::fs::{atomic_save_bin, atomic_save_json};
use common::universal_io::{MmapFs, UniversalReadFs, read_bin_via, read_json_via};
use serde::{Deserialize, Serialize};

use super::{CompactPostings, LmiConfig, LmiRoutingState, MlpRouter};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::types::{Distance, VectorDataConfig, VectorStorageDatatype};

pub const LMI_STATE_FILE: &str = "lmi_state.json";
pub const LMI_ROUTER_FILE: &str = "lmi_router.bin";
pub const LMI_POSTINGS_FILE: &str = "lmi_postings.bin";
pub const LMI_STATE_VERSION: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterPreprocessing {
    Identity,
    CosineNormalized,
}

impl RouterPreprocessing {
    pub fn for_distance(distance: Distance) -> Self {
        match distance {
            Distance::Cosine => Self::CosineNormalized,
            _ => Self::Identity,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmiStateMetadata {
    pub version: u32,
    pub config: LmiConfig,
    pub dimension: usize,
    pub distance: Distance,
    pub datatype: VectorStorageDatatype,
    pub preprocessing: RouterPreprocessing,
    pub total_vectors: usize,
    pub router_format: u32,
    pub postings_format: u32,
}

impl LmiStateMetadata {
    pub fn new(
        config: LmiConfig,
        vector_config: &VectorDataConfig,
        datatype: VectorStorageDatatype,
        total_vectors: usize,
    ) -> Self {
        Self {
            version: LMI_STATE_VERSION,
            config,
            dimension: vector_config.size,
            distance: vector_config.distance,
            datatype,
            preprocessing: RouterPreprocessing::for_distance(vector_config.distance),
            total_vectors,
            router_format: 1,
            postings_format: 1,
        }
    }

    pub fn validate(
        &self,
        config: LmiConfig,
        vector_config: &VectorDataConfig,
        datatype: VectorStorageDatatype,
        total_vectors: usize,
        router: &MlpRouter,
        postings: &CompactPostings,
    ) -> OperationResult<()> {
        config.check()?;
        if !matches!(
            datatype,
            VectorStorageDatatype::Float32 | VectorStorageDatatype::Float16
        ) {
            return Err(OperationError::service_error(
                "LMI trained state requires Float32 or Float16 storage",
            ));
        }
        if self.version != LMI_STATE_VERSION {
            return Err(OperationError::service_error(format!(
                "Unsupported LMI state version {} (expected {})",
                self.version, LMI_STATE_VERSION
            )));
        }
        if self.router_format != 1 || self.postings_format != 1 {
            return Err(OperationError::service_error(
                "Unsupported LMI binary format",
            ));
        }
        if self.config != config
            || self.dimension != vector_config.size
            || self.distance != vector_config.distance
            || self.datatype != datatype
            || self.total_vectors != total_vectors
        {
            return Err(OperationError::service_error(
                "LMI persisted state/configuration mismatch",
            ));
        }
        if self.preprocessing != RouterPreprocessing::for_distance(vector_config.distance) {
            return Err(OperationError::service_error(
                "LMI router preprocessing mismatch",
            ));
        }
        let (input, output) = router.validate()?;
        if input != self.dimension
            || output != config.n_buckets
            || postings.len() != config.n_buckets
        {
            return Err(OperationError::service_error(
                "LMI persisted router/postings dimensions mismatch",
            ));
        }
        if !matches!(router.layers.as_slice(),
            [super::RouterLayer::Linear(first), super::RouterLayer::ReLU, super::RouterLayer::Linear(last)]
            if first.out_features == config.hidden_dim && last.in_features == config.hidden_dim)
        {
            return Err(OperationError::service_error(
                "LMI persisted MLP architecture mismatch",
            ));
        }
        postings.validate()?;
        if postings
            .iter()
            .flatten()
            .any(|&id| id as usize >= total_vectors)
        {
            return Err(OperationError::service_error(
                "LMI posting offset exceeds segment vector count",
            ));
        }
        Ok(())
    }
}

pub fn state_files(path: &Path) -> Vec<PathBuf> {
    [LMI_STATE_FILE, LMI_ROUTER_FILE, LMI_POSTINGS_FILE]
        .map(|name| path.join(name))
        .to_vec()
}

pub fn load_state(
    fs: &impl UniversalReadFs,
    path: &Path,
    config: LmiConfig,
    vector_config: &VectorDataConfig,
    datatype: VectorStorageDatatype,
    total_vectors: usize,
) -> OperationResult<LmiRoutingState> {
    let metadata: LmiStateMetadata = read_json_via(fs, path.join(LMI_STATE_FILE))?;
    // Reject incompatible metadata before decoding binary payloads.
    if metadata.version != LMI_STATE_VERSION
        || metadata.router_format != 1
        || metadata.postings_format != 1
    {
        return Err(OperationError::service_error(
            "Unsupported LMI persisted format",
        ));
    }
    let router: MlpRouter = read_bin_via(fs, path.join(LMI_ROUTER_FILE))?;
    let postings: CompactPostings = read_bin_via(fs, path.join(LMI_POSTINGS_FILE))?;
    metadata.validate(
        config,
        vector_config,
        datatype,
        total_vectors,
        &router,
        &postings,
    )?;
    LmiRoutingState::from_compact(router, postings, config.nprobe)
}

pub fn load_local(
    path: &Path,
    config: LmiConfig,
    vector_config: &VectorDataConfig,
    datatype: VectorStorageDatatype,
    total_vectors: usize,
) -> OperationResult<LmiRoutingState> {
    load_state(
        &MmapFs,
        path,
        config,
        vector_config,
        datatype,
        total_vectors,
    )
}

/// Controlled fixture installation only. Future training owns production writes.
#[cfg(feature = "testing")]
pub fn save_fixture_state(
    path: &Path,
    metadata: &LmiStateMetadata,
    routing: &LmiRoutingState,
    vector_config: &VectorDataConfig,
) -> OperationResult<()> {
    metadata.validate(
        metadata.config,
        vector_config,
        metadata.datatype,
        metadata.total_vectors,
        routing.router(),
        routing.postings(),
    )?;
    fs_err::create_dir_all(path)?;
    atomic_save_bin(&path.join(LMI_ROUTER_FILE), routing.router())?;
    atomic_save_bin(&path.join(LMI_POSTINGS_FILE), routing.postings())?;
    atomic_save_json(&path.join(LMI_STATE_FILE), metadata)?;
    Ok(())
}
