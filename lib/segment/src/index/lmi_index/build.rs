#[cfg(feature = "lmi-training")]
use std::borrow::Cow;
use std::path::Path;
use std::sync::atomic::Ordering;

use common::fs::{atomic_save_bin, atomic_save_json};
use common::generic_consts::Random;
use common::types::{DeferredBehavior, PointOffsetType};
use common::universal_io::{MmapFs, UniversalReadFs, read_bin_via, read_json_via};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use super::CompactPostings;
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
pub const LMI_POSTINGS_FILE: &str = "lmi_postings.bin";
pub const LMI_ROUTER_FILE: &str = "lmi_router.bin";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DiskState {
    version: u32,
    config: LmiConfig,
    distance: Distance,
    dimension: usize,
    total_vectors: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    sample_offsets: Vec<PointOffsetType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    router: Option<MlpRouter>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    postings: Vec<Vec<PointOffsetType>>,
    #[serde(skip)]
    compact: Option<CompactPostings>,
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
        let tracker = open.id_tracker.borrow();
        let storage = open.vector_storage.borrow();
        let total = storage.total_vector_count();
        // Physical slots provide a conservative pre-scan bound, including holes.
        let plan = super::build_plan::BuildPlan::estimate(total, dim, &config)?;
        log::info!("LMI build plan (physical-slot upper bound): {plan:?}");
        plan.check_budget()?;
        let end = PointOffsetType::try_from(total)
            .map_err(|_| OperationError::service_error("LMI offset range exceeded"))?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut sample = Vec::with_capacity(config.sample_size.min(total));
        // Both borrows remain held through sampling, training and both posting
        // passes. Internal offsets are stable; named-vector tombstones are skipped.
        let eligible = || {
            tracker
                .point_mappings()
                .filter_deferred_and_deleted(0..end, DeferredBehavior::VisibleOnly)
                .filter(|&id| {
                    (id as usize) < tracker.deleted_point_bitslice().len()
                        && !storage.is_deleted_vector(id)
                })
        };
        let mut seen = 0usize;
        let sampling_started = std::time::Instant::now();
        let progress = args.progress.track_progress(Some(total as u64));
        for id in eligible() {
            check_process_stopped(args.stopped)?;
            seen += 1;
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
        let plan = super::build_plan::BuildPlan::estimate(seen, dim, &config)?;
        log::info!("LMI build plan (eligible vectors): {plan:?}");
        plan.check_budget()?;
        log::info!(
            "LMI sampling: segment_slots={total} eligible={seen} sample={} buckets={} seconds={:.6}",
            sample.len(),
            config.n_buckets,
            sampling_started.elapsed().as_secs_f64()
        );
        #[allow(unused_mut)] // Populated only in builds with the optional trainer.
        let mut state = DiskState {
            version: 2,
            config,
            distance: vector_config.distance,
            dimension: dim,
            total_vectors: total,
            sample_offsets: sample.clone(),
            router: None,
            postings: vec![],
            compact: None,
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
                // Explicit experimental build knob; never persisted and never used at open/search.
                let tch_routing = std::env::var("LMI_EXPERIMENTAL_TCH_BUILD_ROUTING")
                    .is_ok_and(|value| value == "1");
                let trained = if tch_routing {
                    Some(super::training::train_with_tch(
                        &data,
                        dim,
                        &config,
                        vector_config.distance,
                        args.stopped,
                    )?)
                } else {
                    None
                };
                // The default native path drops the Torch model immediately after export.
                let native_router = if tch_routing {
                    None
                } else {
                    Some(super::training::train(
                        &data,
                        dim,
                        &config,
                        vector_config.distance,
                        args.stopped,
                    )?)
                };
                let router = trained
                    .as_ref()
                    .map(|t| &t.native)
                    .or(native_router.as_ref())
                    .expect("one routing backend");
                #[cfg(test)]
                let posting_started = std::time::Instant::now();
                // Release the training matrix before allocating corpus postings.
                drop(data);
                let mut predictor = if tch_routing {
                    None
                } else {
                    Some(super::routing::BuildBatchRouter::new(
                        router,
                        config.routing_batch_size,
                        args.stopped,
                    )?)
                };
                let mut tie_verifier = if tch_routing {
                    Some(super::routing::BuildRouter::new(router, args.stopped)?)
                } else {
                    None
                };
                let mut tie_fallbacks = 0usize;
                log::info!(
                    "LMI postings: backend={} batch={} native_batch_workspace_bytes={}",
                    if tch_routing { "tch" } else { "native" },
                    config.routing_batch_size,
                    predictor.as_ref().map_or(0, |p| p.workspace_bytes()),
                );
                let (postings, times) =
                    CompactPostings::build_two_pass(config.n_buckets, args.stopped, |push| {
                        let mut offsets = Vec::new();
                        let mut inputs = Vec::new();
                        offsets
                            .try_reserve_exact(config.routing_batch_size)
                            .map_err(|e| {
                                OperationError::service_error(format!(
                                    "LMI posting routing offsets allocation: {e}"
                                ))
                            })?;
                        inputs
                            .try_reserve_exact(
                                config.routing_batch_size.checked_mul(dim).ok_or_else(|| {
                                    OperationError::service_error(
                                        "LMI posting routing input size overflow",
                                    )
                                })?,
                            )
                            .map_err(|e| {
                                OperationError::service_error(format!(
                                    "LMI posting routing input allocation: {e}"
                                ))
                            })?;
                        let mut flush = |offsets: &mut Vec<PointOffsetType>,
                                         inputs: &mut Vec<f32>|
                         -> OperationResult<()> {
                            if offsets.is_empty() {
                                return Ok(());
                            }
                            let buckets: Cow<'_, [usize]> = if tch_routing {
                                let (buckets, fallbacks) = trained
                                    .as_ref()
                                    .expect("tch backend selected")
                                    .top_buckets_tie_safe(
                                        inputs,
                                        args.stopped,
                                        tie_verifier.as_mut().expect("native verifier"),
                                    )?;
                                tie_fallbacks += fallbacks;
                                Cow::Owned(buckets)
                            } else {
                                Cow::Borrowed(
                                    predictor
                                        .as_mut()
                                        .expect("native backend selected")
                                        .top_buckets(inputs, args.stopped)?,
                                )
                            };
                            for (&id, &bucket) in offsets.iter().zip(buckets.iter()) {
                                push(id, bucket)?;
                            }
                            offsets.clear();
                            inputs.clear();
                            Ok(())
                        };
                        for id in eligible() {
                            check_process_stopped(args.stopped)?;
                            let Some(CowVector::Dense(row)) = storage.get_vector_opt::<Random>(id)
                            else {
                                return Err(OperationError::service_error(
                                    "LMI posting contains no dense vector",
                                ));
                            };
                            if row.len() != dim || row.iter().any(|value| !value.is_finite()) {
                                return Err(OperationError::service_error(
                                    "LMI posting has invalid dimension or non-finite value",
                                ));
                            }
                            offsets.push(id);
                            inputs.extend_from_slice(&row);
                            if offsets.len() == config.routing_batch_size {
                                flush(&mut offsets, &mut inputs)?;
                            }
                        }
                        flush(&mut offsets, &mut inputs)
                    })?;
                if tch_routing {
                    log::info!(
                        "LMI postings: tch native tie verification rows={tie_fallbacks} across both passes"
                    );
                }
                log::info!(
                    "LMI postings: segment_slots={total} eligible={seen} buckets={} postings={} pass1_seconds={:.6} allocation_seconds={:.6} pass2_seconds={:.6}",
                    config.n_buckets,
                    postings.point_count(),
                    times[0],
                    times[1],
                    times[2]
                );
                #[cfg(test)]
                {
                    *POSTING_SECONDS.lock().unwrap() = posting_started.elapsed().as_secs_f64();
                }
                drop(predictor);
                state.router = Some(match (trained, native_router) {
                    (Some(t), None) => t.native,
                    (None, Some(r)) => r,
                    _ => unreachable!("exactly one routing backend"),
                });
                state.compact = Some(postings);
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
        // Binary payloads first, metadata last. SegmentBuilder publishes the
        // entire staging directory only after all indexes and segment state succeed.
        let persistence_started = std::time::Instant::now();
        state.save(open.path)?;
        log::info!(
            "LMI persistence: seconds={:.6}",
            persistence_started.elapsed().as_secs_f64()
        );
        drop(state); // Do not hold a second complete posting array during reopen.
        progress.store(total as u64, Ordering::Relaxed);
        Self::open_trained(open, vector_config, config)
    }

    pub(crate) fn open_trained(
        open: VectorIndexOpenArgs,
        vector_config: &VectorDataConfig,
        config: LmiConfig,
    ) -> OperationResult<Self> {
        config.check()?;
        let reopen_started = std::time::Instant::now();
        let path = open.path.join(LMI_STATE_FILE);
        let state = DiskState::load(&MmapFs, open.path)?;
        let state_files = state.files(open.path);
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
        index.state_files = state_files;
        log::info!(
            "LMI open: mode={:?}; no training; seconds={:.6}",
            index.candidate_mode,
            reopen_started.elapsed().as_secs_f64()
        );
        Ok(index)
    }

    /// Persisted native state for snapshots; legacy transient fixtures have no file.
    pub fn state_path(&self) -> Option<&Path> {
        self.state_path.as_deref()
    }
}

#[derive(Serialize, Deserialize)]
struct DiskRouter {
    sample_offsets: Vec<PointOffsetType>,
    router: Option<MlpRouter>,
}

impl DiskState {
    #[cfg(test)]
    pub(super) fn diagnostic_json(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).expect("serializable state");
        if let Some(postings) = &self.compact {
            value["postings"] = serde_json::json!(postings.to_vec());
        }
        value
    }

    pub(super) fn load(fs: &impl UniversalReadFs, path: &Path) -> OperationResult<Self> {
        let mut state: Self = read_json_via(fs, path.join(LMI_STATE_FILE))?;
        match state.version {
            1 => {}
            2 => {
                if !state.sample_offsets.is_empty()
                    || state.router.is_some()
                    || !state.postings.is_empty()
                {
                    return Err(OperationError::service_error(
                        "LMI v2 contains unexpected inline arrays",
                    ));
                }
                let model: DiskRouter = read_bin_via(fs, path.join(LMI_ROUTER_FILE))?;
                let postings: CompactPostings = read_bin_via(fs, path.join(LMI_POSTINGS_FILE))?;
                postings.validate()?;
                state.sample_offsets = model.sample_offsets;
                state.router = model.router;
                state.compact = Some(postings);
            }
            _ => {
                return Err(OperationError::service_error(
                    "Unsupported LMI state version",
                ));
            }
        }
        Ok(state)
    }

    fn files(&self, path: &Path) -> Vec<std::path::PathBuf> {
        let mut files = vec![path.join(LMI_STATE_FILE)];
        if self.version == 2 {
            files.extend([path.join(LMI_ROUTER_FILE), path.join(LMI_POSTINGS_FILE)]);
        }
        files
    }

    fn save(&mut self, path: &Path) -> OperationResult<()> {
        let router = DiskRouter {
            sample_offsets: std::mem::take(&mut self.sample_offsets),
            router: self.router.take(),
        };
        let postings = self
            .compact
            .take()
            .unwrap_or(CompactPostings::from_buckets(vec![])?);
        atomic_save_bin(&path.join(LMI_ROUTER_FILE), &router)?;
        atomic_save_bin(&path.join(LMI_POSTINGS_FILE), &postings)?;
        atomic_save_json(&path.join(LMI_STATE_FILE), self)?;
        Ok(())
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
        if !matches!(self.version, 1 | 2)
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
        let postings = match self.compact {
            Some(postings) => postings,
            None => CompactPostings::from_buckets(self.postings)?,
        };
        postings.validate()?;
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
                // Exact uniqueness and coverage, at one bit per physical offset.
                // Stale deleted postings remain allowed, as in version 1.
                let mut seen = bitvec::vec::BitVec::<u8, bitvec::order::Lsb0>::repeat(false, total);
                for posting in postings.iter() {
                    for &id in posting {
                        if id as usize >= total || seen.replace(id as usize, true) {
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
                    .any(|id| !storage.is_deleted_vector(id) && !seen[id as usize])
                {
                    return Err(OperationError::service_error(
                        "LMI persisted postings omit a live vector",
                    ));
                }
                Some(LmiRoutingState::from_compact(
                    router,
                    postings,
                    config.nprobe,
                )?)
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
                if !postings.is_empty() || self.sample_offsets.len() >= config.n_buckets {
                    return Err(OperationError::service_error("Invalid LMI fallback state"));
                }
                None
            }
        };
        Ok(routing)
    }
}
