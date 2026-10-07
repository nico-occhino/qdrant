//! Production build from this target segment's authoritative tracker and vector storage.
use std::sync::atomic::Ordering;
use std::time::Instant;

use common::generic_consts::Random;
use common::types::{DeferredBehavior, PointOffsetType};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

use super::{CompactPostings, LmiConfig, LmiIndex, LmiRoutingState, state};
use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};
use crate::data_types::named_vectors::CowVector;
use crate::id_tracker::IdTrackerRead;
use crate::segment_constructor::{VectorIndexBuildArgs, VectorIndexOpenArgs};
use crate::types::{Distance, VectorDataConfig, VectorStorageDatatype};
use crate::vector_storage::VectorStorageRead;

fn eligible_offsets<'a>(
    tracker: &'a impl IdTrackerRead,
    storage: &'a impl VectorStorageRead,
    end: PointOffsetType,
) -> impl Iterator<Item = PointOffsetType> + 'a {
    tracker
        .point_mappings()
        .filter_deferred_and_deleted(0..end, DeferredBehavior::VisibleOnly)
        .filter(move |&id| {
            (id as usize) < tracker.deleted_point_bitslice().len() && !storage.is_deleted_vector(id)
        })
}

fn reservoir_sample_offsets(
    offsets: impl Iterator<Item = PointOffsetType>,
    limit: usize,
    seed: u64,
    stopped: &std::sync::atomic::AtomicBool,
    mut visit: impl FnMut(PointOffsetType),
) -> OperationResult<(Vec<PointOffsetType>, usize)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sample = Vec::new();
    sample
        .try_reserve_exact(limit)
        .map_err(|e| OperationError::service_error(format!("LMI sample allocation: {e}")))?;
    let mut count = 0usize;
    for id in offsets {
        check_process_stopped(stopped)?;
        count = count
            .checked_add(1)
            .ok_or_else(|| OperationError::service_error("LMI eligible count overflow"))?;
        if sample.len() < limit {
            sample.push(id);
        } else {
            let slot = rng.random_range(0..count);
            if slot < limit {
                sample[slot] = id;
            }
        }
        visit(id);
    }
    sample.sort_unstable();
    Ok((sample, count))
}

fn read_dense(
    storage: &impl VectorStorageRead,
    id: PointOffsetType,
    dim: usize,
) -> OperationResult<Vec<f32>> {
    let Some(CowVector::Dense(row)) = storage.get_vector_opt::<Random>(id) else {
        return Err(OperationError::service_error(
            "LMI eligible offset has no dense vector",
        ));
    };
    if row.len() != dim || row.iter().any(|value| !value.is_finite()) {
        return Err(OperationError::service_error(
            "LMI build vector has invalid dimension or non-finite value",
        ));
    }
    Ok(row.into_owned())
}

/// Exact one-bucket/eligible-offset proof. Bitmap size is one bit per physical slot.
fn validate_coverage(
    postings: &CompactPostings,
    eligible: impl Iterator<Item = PointOffsetType>,
    physical_count: usize,
    eligible_count: usize,
) -> OperationResult<()> {
    if postings.point_count() != eligible_count {
        return Err(OperationError::service_error(
            "LMI posting count differs from eligible live vector count",
        ));
    }
    let mut seen = bitvec::vec::BitVec::<u8, bitvec::order::Lsb0>::repeat(false, physical_count);
    for posting in postings.iter() {
        for &id in posting {
            if id as usize >= physical_count || seen.replace(id as usize, true) {
                return Err(OperationError::service_error(
                    "LMI posting offset is out of range or duplicated",
                ));
            }
        }
    }
    for id in eligible {
        if id as usize >= physical_count || !seen[id as usize] {
            return Err(OperationError::service_error(
                "LMI postings omit an eligible live vector",
            ));
        }
    }
    Ok(())
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
        if vector_config.distance != Distance::Cosine {
            return Err(OperationError::service_error(
                "Production LMI training is validated for Cosine only",
            ));
        }
        if vector_config.multivector_config.is_some() {
            return Err(OperationError::service_error(
                "LMI training requires dense single-vector storage",
            ));
        }
        if args.permit.num_cpus == 0 {
            return Err(OperationError::service_error(
                "LMI build requires a CPU permit",
            ));
        }
        let tracker = open.id_tracker.borrow();
        let storage = open.vector_storage.borrow();
        let physical_count = storage.total_vector_count();
        let datatype = storage.datatype();
        if !matches!(
            datatype,
            VectorStorageDatatype::Float32 | VectorStorageDatatype::Float16
        ) {
            return Err(OperationError::service_error(
                "LMI training supports Float32 and Float16 only",
            ));
        }
        let dim = vector_config.size;
        let end = PointOffsetType::try_from(physical_count)
            .map_err(|_| OperationError::service_error("LMI physical offset range exceeded"))?;
        let plan = super::build_plan::BuildPlan::estimate(physical_count, dim, &config)?;
        plan.check_budget()?;
        log::info!("LMI physical-slot build plan: {plan:?}");
        let sample_started = Instant::now();
        let progress = args.progress.track_progress(Some(physical_count as u64));
        let (sample, eligible_count) = reservoir_sample_offsets(
            eligible_offsets(&*tracker, &*storage, end),
            config.sample_size.min(physical_count),
            config.seed,
            args.stopped,
            |id| progress.store(id as u64 + 1, Ordering::Relaxed),
        )?;
        log::info!(
            "LMI sample_offsets: eligible={eligible_count} selected={} seconds={:.6}",
            sample.len(),
            sample_started.elapsed().as_secs_f64()
        );
        if eligible_count < config.n_buckets {
            return Err(OperationError::service_error(format!(
                "LMI needs at least {} eligible live vectors, found {eligible_count}",
                config.n_buckets
            )));
        }
        let live_plan = super::build_plan::BuildPlan::estimate(eligible_count, dim, &config)?;
        live_plan.check_budget()?;
        log::info!("LMI eligible-live build plan: {live_plan:?}");
        let mut data = Vec::new();
        data.try_reserve_exact(
            sample
                .len()
                .checked_mul(dim)
                .ok_or_else(|| OperationError::service_error("LMI sample matrix size overflow"))?,
        )
        .map_err(|e| OperationError::service_error(format!("LMI sample allocation: {e}")))?;
        for &id in &sample {
            check_process_stopped(args.stopped)?;
            data.extend_from_slice(&read_dense(&*storage, id, dim)?);
        }
        let training_started = Instant::now();
        let trained = super::training::train_with_tch(
            &data,
            dim,
            &config,
            vector_config.distance,
            args.stopped,
        )?;
        log::info!(
            "LMI teacher_and_mlp seconds={:.6}",
            training_started.elapsed().as_secs_f64()
        );
        let router = &trained.native;
        drop(data);
        let mut native = super::routing::BuildRouter::new(router, args.stopped)?;
        let batch_size = config.routing_batch_size;
        let mut tie_fallbacks = 0usize;
        let mut route_pass = |push: &mut dyn FnMut(
            PointOffsetType,
            usize,
        ) -> OperationResult<()>|
         -> OperationResult<()> {
            let mut offsets = Vec::with_capacity(batch_size);
            let mut inputs =
                Vec::with_capacity(batch_size.checked_mul(dim).ok_or_else(|| {
                    OperationError::service_error("LMI routing batch size overflow")
                })?);
            let mut flush = |offsets: &mut Vec<PointOffsetType>,
                             inputs: &mut Vec<f32>|
             -> OperationResult<()> {
                if offsets.is_empty() {
                    return Ok(());
                }
                let (buckets, fallbacks) =
                    trained.top_buckets_tie_safe(inputs, args.stopped, &mut native)?;
                tie_fallbacks += fallbacks;
                for (&id, &bucket) in offsets.iter().zip(&buckets) {
                    push(id, bucket)?;
                }
                offsets.clear();
                inputs.clear();
                Ok(())
            };
            for id in eligible_offsets(&*tracker, &*storage, end) {
                check_process_stopped(args.stopped)?;
                offsets.push(id);
                inputs.extend_from_slice(&read_dense(&*storage, id, dim)?);
                if offsets.len() == batch_size {
                    flush(&mut offsets, &mut inputs)?;
                }
            }
            flush(&mut offsets, &mut inputs)
        };
        // LmiConfig currently caps B at 65,536; this covers every valid configuration.
        let routing_started = Instant::now();
        let (postings, times) = CompactPostings::build_cached_u16(
            config.n_buckets,
            eligible_count,
            args.stopped,
            &mut route_pass,
            |push| {
                for id in eligible_offsets(&*tracker, &*storage, end) {
                    check_process_stopped(args.stopped)?;
                    push(id)?;
                }
                Ok(())
            },
        )?;
        log::info!(
            "LMI corpus routing: eligible={eligible_count} physical={physical_count} tie_fallbacks={tie_fallbacks} total_seconds={:.6} count_route_alloc_fill_seconds={times:?}",
            routing_started.elapsed().as_secs_f64()
        );
        let coverage_started = Instant::now();
        validate_coverage(
            &postings,
            eligible_offsets(&*tracker, &*storage, end),
            physical_count,
            eligible_count,
        )?;
        log::info!(
            "LMI exact_coverage seconds={:.6}",
            coverage_started.elapsed().as_secs_f64()
        );
        let routing = LmiRoutingState::from_compact(trained.native, postings, config.nprobe)?;
        let mut metadata =
            state::LmiStateMetadata::new(config, vector_config, datatype, physical_count);
        metadata.indexed_live_count = Some(eligible_count);
        metadata.validate(
            config,
            vector_config,
            datatype,
            physical_count,
            routing.router(),
            routing.postings(),
        )?;
        drop(storage);
        drop(tracker);
        check_process_stopped(args.stopped)?;
        let persistence_started = Instant::now();
        state::save_trained(open.path, &metadata, &routing, vector_config)?;
        log::info!(
            "LMI persistence seconds={:.6}",
            persistence_started.elapsed().as_secs_f64()
        );
        progress.store(physical_count as u64, Ordering::Relaxed);
        Self::open_trained(
            open.path,
            vector_config,
            config,
            open.id_tracker,
            open.vector_storage,
            open.quantized_vectors,
            open.payload_index,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_reservoir_sampling_is_seeded_and_handles_sparse_population() {
        let stop = std::sync::atomic::AtomicBool::new(false);
        let source = [2, 5, 9, 17, 25, 31, 44, 58];
        let sample = |limit, seed| {
            reservoir_sample_offsets(source.into_iter(), limit, seed, &stop, |_| ()).unwrap()
        };
        assert_eq!(sample(0, 42), (Vec::new(), source.len()));
        assert_eq!(sample(16, 42), (source.to_vec(), source.len()));
        assert_eq!(sample(8, 42), (source.to_vec(), source.len()));
        assert_eq!(sample(3, 42), sample(3, 42));
        assert_ne!(sample(3, 42).0, sample(3, 7).0);
        assert_eq!(
            reservoir_sample_offsets(std::iter::empty(), 4, 42, &stop, |_| ()).unwrap(),
            (Vec::new(), 0)
        );
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(reservoir_sample_offsets(source.into_iter(), 3, 42, &stop, |_| ()).is_err());
    }

    #[test]
    fn exact_coverage_rejects_omissions_duplicates_and_out_of_range() {
        let ok = CompactPostings::from_buckets(vec![vec![0, 3], vec![1]]).unwrap();
        validate_coverage(&ok, [0, 1, 3].into_iter(), 5, 3).unwrap();
        assert!(validate_coverage(&ok, [0, 1, 2].into_iter(), 5, 3).is_err());
        let duplicate = CompactPostings::from_buckets(vec![vec![0, 1], vec![1]]).unwrap();
        assert!(validate_coverage(&duplicate, [0, 1, 3].into_iter(), 5, 3).is_err());
        let out_of_range = CompactPostings::from_buckets(vec![vec![0, 5], vec![1]]).unwrap();
        assert!(validate_coverage(&out_of_range, [0, 1, 3].into_iter(), 5, 3).is_err());
    }
}
