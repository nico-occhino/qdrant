use std::path::Path;
use std::sync::atomic::Ordering;

use common::fs::{atomic_save_json, read_json};
use common::generic_consts::Random;
use common::types::{DeferredBehavior, PointOffsetType};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use super::{LmiCandidateMode, LmiConfig, LmiIndex, LmiRoutingState, MlpRouter};
use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};
use crate::data_types::named_vectors::CowVector;
use crate::id_tracker::IdTrackerRead;
use crate::segment_constructor::{VectorIndexBuildArgs, VectorIndexOpenArgs};
use crate::types::{Distance, VectorDataConfig, VectorStorageDatatype};
use crate::vector_storage::VectorStorageRead;

#[cfg(test)]
pub(super) static POSTING_SECONDS: std::sync::Mutex<f64> = std::sync::Mutex::new(0.0);

pub const LMI_STATE_FILE: &str = "lmi_state.json";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DiskState {
    version: u32,
    config: LmiConfig,
    distance: Distance,
    dimension: usize,
    total_vectors: usize,
    sample_offsets: Vec<PointOffsetType>,
    router: Option<MlpRouter>,
    postings: Vec<Vec<PointOffsetType>>,
}

impl LmiIndex {
    pub(crate) fn build_trained<R: Rng + ?Sized>(
        open: VectorIndexOpenArgs,
        vector_config: &VectorDataConfig,
        config: LmiConfig,
        args: VectorIndexBuildArgs<R>,
    ) -> OperationResult<Self> {
        config.check()?;
        check_process_stopped(args.stopped)?;
        if !cfg!(feature = "lmi-training") {
            return Err(OperationError::service_error(
                "LMI construction requires the lmi-training Cargo feature",
            ));
        }
        if vector_config.multivector_config.is_some()
            || vector_config
                .datatype
                .is_some_and(|d| d != VectorStorageDatatype::Float32)
        {
            return Err(OperationError::service_error(
                "LMI training supports dense float32 vectors only",
            ));
        }
        if args.permit.num_cpus == 0 {
            return Err(OperationError::service_error(
                "LMI build requires a CPU permit",
            ));
        }
        let dim = vector_config.size;
        // Bound temporary training/model allocation independently of corpus size.
        let sample_elements = config.sample_size.checked_mul(dim);
        let model_elements = dim
            .checked_add(config.n_buckets)
            .and_then(|n| n.checked_mul(config.hidden_dim));
        if dim == 0
            || sample_elements.is_none_or(|n| n > 32_000_000)
            || model_elements.is_none_or(|n| n > 32_000_000)
        {
            return Err(OperationError::service_error(
                "LMI training matrix/model exceeds the experimental 32M-element limit",
            ));
        }
        let tracker = open.id_tracker.borrow();
        let storage = open.vector_storage.borrow();
        let total = storage.total_vector_count();
        let end = PointOffsetType::try_from(total)
            .map_err(|_| OperationError::service_error("LMI offset range exceeded"))?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut sample = Vec::with_capacity(config.sample_size.min(total));
        let mut eligible = Vec::new();
        let progress = args.progress.track_progress(Some(total as u64));
        for id in tracker
            .point_mappings()
            .filter_deferred_and_deleted(0..end, DeferredBehavior::VisibleOnly)
        {
            check_process_stopped(args.stopped)?;
            if id as usize >= tracker.deleted_point_bitslice().len()
                || storage.is_deleted_vector(id)
            {
                continue;
            }
            eligible.push(id);
            let seen = eligible.len();
            if sample.len() < config.sample_size {
                sample.push(id);
            } else {
                let slot = rng.random_range(0..seen);
                if slot < sample.len() {
                    sample[slot] = id;
                }
            }
            progress.store(u64::from(id) + 1, Ordering::Relaxed);
        }
        sample.sort_unstable();
        #[allow(unused_mut)] // Populated only in builds with the optional trainer.
        let mut state = DiskState {
            version: 1,
            config,
            distance: vector_config.distance,
            dimension: dim,
            total_vectors: total,
            sample_offsets: sample.clone(),
            router: None,
            postings: vec![],
        };
        if sample.len() >= config.n_buckets {
            let mut data = Vec::with_capacity(sample.len() * dim);
            for &id in &sample {
                check_process_stopped(args.stopped)?;
                let Some(CowVector::Dense(row)) = storage.get_vector_opt::<Random>(id) else {
                    return Err(OperationError::service_error(
                        "LMI sample contains no dense vector",
                    ));
                };
                if row.len() != dim || row.iter().any(|x| !x.is_finite()) {
                    return Err(OperationError::service_error(
                        "LMI sample has invalid dimension or non-finite value",
                    ));
                }
                data.extend_from_slice(&row);
            }
            #[cfg(feature = "lmi-training")]
            {
                log::info!(
                    "LMI build: training {} samples, {} dimensions, {} buckets on CPU",
                    sample.len(),
                    dim,
                    config.n_buckets
                );
                let router = super::training::train(&data, dim, &config, args.stopped)?;
                #[cfg(test)]
                let posting_started = std::time::Instant::now();
                let mut postings = vec![Vec::new(); config.n_buckets];
                for id in eligible {
                    check_process_stopped(args.stopped)?;
                    let Some(CowVector::Dense(row)) = storage.get_vector_opt::<Random>(id) else {
                        return Err(OperationError::service_error(
                            "LMI posting contains no dense vector",
                        ));
                    };
                    let bucket = router.top_buckets_with_stop(&row, 1, args.stopped)?[0];
                    postings[bucket].push(id);
                }
                #[cfg(test)]
                {
                    *POSTING_SECONDS.lock().unwrap() = posting_started.elapsed().as_secs_f64();
                }
                state.router = Some(router);
                state.postings = postings;
            }
        } else {
            log::info!(
                "LMI build: {} eligible vectors, fewer than {} buckets; persisting exact fallback",
                sample.len(),
                config.n_buckets
            );
        }
        drop(storage);
        drop(tracker);
        check_process_stopped(args.stopped)?;
        fs_err::create_dir_all(open.path)?;
        atomic_save_json(&open.path.join(LMI_STATE_FILE), &state)?;
        progress.store(total as u64, Ordering::Relaxed);
        Self::open_trained(open, vector_config, config)
    }

    pub(crate) fn open_trained(
        open: VectorIndexOpenArgs,
        vector_config: &VectorDataConfig,
        config: LmiConfig,
    ) -> OperationResult<Self> {
        config.check()?;
        let path = open.path.join(LMI_STATE_FILE);
        let state: DiskState = read_json(&path)?;
        let routing = state.validate(
            vector_config,
            config,
            &*open.id_tracker.borrow(),
            &*open.vector_storage.borrow(),
        )?;
        let mut index = Self::new(
            open.id_tracker,
            open.vector_storage,
            open.quantized_vectors,
            open.payload_index,
        );
        index.routing_state = routing;
        if index.routing_state.is_some() {
            index.candidate_mode = LmiCandidateMode::StaticLearned;
        }
        index.routing_distance = Some(vector_config.distance);
        index.state_path = Some(path);
        log::info!("LMI open: mode={:?}; no training", index.candidate_mode);
        Ok(index)
    }

    /// Persisted native state for snapshots; legacy transient fixtures have no file.
    pub fn state_path(&self) -> Option<&Path> {
        self.state_path.as_deref()
    }
}

impl DiskState {
    /// One validation path for native and universal read-only opening.
    pub(super) fn validate(
        self,
        vector_config: &VectorDataConfig,
        config: LmiConfig,
        tracker: &impl IdTrackerRead,
        storage: &impl VectorStorageRead,
    ) -> OperationResult<Option<LmiRoutingState>> {
        config.check()?;
        let total = storage.total_vector_count();
        if self.version != 1
            || self.config != config
            || self.dimension != vector_config.size
            || self.distance != vector_config.distance
            || self.total_vectors != total
        {
            return Err(OperationError::service_error(
                "LMI persisted state/configuration mismatch",
            ));
        }
        if self.sample_offsets.len() > config.sample_size
            || self.sample_offsets.iter().any(|&id| id as usize >= total)
            || self
                .sample_offsets
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(OperationError::service_error(
                "LMI persisted sample offsets are invalid",
            ));
        }
        let routing = match self.router {
            Some(router) => {
                let (input, output) = router.validate()?;
                if !matches!(router.layers.as_slice(),
                    [super::RouterLayer::Linear(first), super::RouterLayer::ReLU, super::RouterLayer::Linear(last)]
                    if first.out_features == config.hidden_dim && last.in_features == config.hidden_dim)
                {
                    return Err(OperationError::service_error(
                        "LMI persisted architecture/configuration mismatch",
                    ));
                }

                if input != self.dimension || output != config.n_buckets {
                    return Err(OperationError::service_error(
                        "LMI persisted router dimensions mismatch",
                    ));
                }
                let mut seen = std::collections::HashSet::new();
                for posting in &self.postings {
                    for &id in posting {
                        if id as usize >= total || !seen.insert(id) {
                            return Err(OperationError::service_error(
                                "LMI persisted postings contain invalid or duplicate offsets",
                            ));
                        }
                    }
                }
                if self.sample_offsets.len() < config.n_buckets {
                    return Err(OperationError::service_error(
                        "LMI persisted training sample is too small",
                    ));
                }
                if tracker
                    .point_mappings()
                    .filter_deferred_and_deleted(
                        0..PointOffsetType::try_from(total).map_err(|_| {
                            OperationError::service_error("LMI offset range exceeded")
                        })?,
                        DeferredBehavior::VisibleOnly,
                    )
                    .any(|id| !storage.is_deleted_vector(id) && !seen.contains(&id))
                {
                    return Err(OperationError::service_error(
                        "LMI persisted postings omit a live vector",
                    ));
                }
                Some(LmiRoutingState::new(router, self.postings, config.nprobe)?)
            }
            None => {
                let end = PointOffsetType::try_from(total)
                    .map_err(|_| OperationError::service_error("LMI offset range exceeded"))?;
                let live = tracker
                    .point_mappings()
                    .filter_deferred_and_deleted(0..end, DeferredBehavior::VisibleOnly)
                    .filter(|&id| {
                        (id as usize) < tracker.deleted_point_bitslice().len()
                            && !storage.is_deleted_vector(id)
                    })
                    .take(config.n_buckets)
                    .count();
                if live >= config.n_buckets {
                    return Err(OperationError::service_error(
                        "LMI fallback state unexpectedly omits a trained model",
                    ));
                }
                if !self.postings.is_empty() || self.sample_offsets.len() >= config.n_buckets {
                    return Err(OperationError::service_error("Invalid LMI fallback state"));
                }
                None
            }
        };
        Ok(routing)
    }
}
