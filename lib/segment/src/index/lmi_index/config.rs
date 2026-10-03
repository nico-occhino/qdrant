use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::common::anonymize::Anonymize;
use crate::common::operation_error::{OperationError, OperationResult};

/// Experimental CPU LMI build configuration. Training uses Adam at 1e-3.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Validate, Anonymize,
)]
#[serde(default, deny_unknown_fields)]
#[anonymize(false)]
#[validate(schema(function = "validate_config"))]
pub struct LmiConfig {
    #[validate(range(min = 1, max = 65536))]
    pub n_buckets: usize,
    #[validate(range(min = 1, max = 1000000))]
    pub sample_size: usize,
    #[validate(range(min = 1, max = 4096))]
    pub hidden_dim: usize,
    #[validate(range(min = 1, max = 10000))]
    pub epochs: usize,
    #[validate(range(min = 1, max = 65536))]
    pub batch_size: usize,
    /// Build-only corpus-routing rows kept in native reusable buffers. One
    /// preserves the prior scalar reference path; larger values are explicit
    /// experimental batching choices and do not affect query-time routing.
    #[serde(default = "default_routing_batch_size")]
    #[validate(range(min = 1, max = 65536))]
    pub routing_batch_size: usize,
    #[validate(range(min = 1, max = 1000))]
    pub kmeans_iterations: usize,
    #[validate(range(min = 1))]
    pub nprobe: usize,
    pub seed: u64,
}

impl Default for LmiConfig {
    fn default() -> Self {
        Self {
            n_buckets: 8,
            sample_size: 2048,
            hidden_dim: 64,
            epochs: 30,
            batch_size: 256,
            routing_batch_size: 1,
            kmeans_iterations: 20,
            nprobe: 2,
            seed: 42,
        }
    }
}

const fn default_routing_batch_size() -> usize {
    1
}

fn validate_config(config: &LmiConfig) -> Result<(), ValidationError> {
    if config.nprobe > config.n_buckets || config.sample_size < config.n_buckets {
        let mut err = ValidationError::new("invalid_lmi_config");
        err.message = Some("LMI requires nprobe <= n_buckets <= sample_size".into());
        return Err(err);
    }
    Ok(())
}

impl LmiConfig {
    pub fn check(&self) -> OperationResult<()> {
        self.validate()
            .map_err(|err| OperationError::service_error(format!("Invalid LMI config: {err}")))
    }
}
