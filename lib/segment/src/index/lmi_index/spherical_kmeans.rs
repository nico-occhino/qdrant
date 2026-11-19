//! Scalar spherical KMeans for Qdrant-preprocessed cosine samples.
//! The independent Euclidean Lloyd path remains in `training.rs`.
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use super::LmiConfig;
use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};
use crate::spaces::metric::Metric;
use crate::spaces::simple::DotProductMetric;

#[derive(Debug, Default)]
pub(super) struct ClusterStats {
    pub iterations: usize,
    pub initialization_seconds: f64,
    pub assignment_seconds: Vec<f64>,
    pub accumulation_seconds: Vec<f64>,
    pub normalization_seconds: Vec<f64>,
    pub final_assignment_seconds: f64,
    pub sizes: Vec<usize>,
}

fn error(message: &str) -> OperationError {
    OperationError::service_error(message)
}

fn unit_norm(center: &mut [f64]) -> OperationResult<bool> {
    let squared: f64 = center.iter().map(|v| v * v).sum();
    if !squared.is_finite() {
        return Err(error("LMI spherical centroid norm is non-finite"));
    }
    if squared == 0.0 {
        return Ok(false);
    }
    let norm = squared.sqrt();
    center.iter_mut().for_each(|v| *v /= norm);
    Ok(true)
}

/// Score every centroid in fixed bucket order. Strict comparison makes exact
/// ties choose the smallest bucket ID, including for zero Qdrant vectors.
fn assign(row: &[f32], centers: &[Vec<f64>]) -> usize {
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (bucket, center) in centers.iter().enumerate() {
        let score = row
            .iter()
            .zip(center)
            .map(|(&x, &c)| f64::from(x) * c)
            .sum::<f64>();
        if score > best_score {
            best = bucket;
            best_score = score;
        }
    }
    best
}

/// Reuse the native f32 dot kernel. Reduction order can differ from the
/// f64 reference near boundaries; the scalar path remains the oracle.
fn assign_qdrant(row: &[f32], centers: &[Vec<f32>]) -> usize {
    let mut best = 0;
    let mut best_score = f32::NEG_INFINITY;
    for (bucket, center) in centers.iter().enumerate() {
        let score = DotProductMetric::similarity(row, center);
        if score > best_score {
            best = bucket;
            best_score = score;
        }
    }
    best
}

#[allow(dead_code)] // Scalar oracle is exercised by explicit benchmarks and tests.
#[derive(Clone, Copy)]
enum AssignmentKernel {
    ScalarF64,
    QdrantF32,
}

/// Input rows come from Qdrant's cosine vector storage: nonzero rows are
/// normalized already, while exact zero vectors remain zero. A zero seed or
/// zero-mean cluster may retain a zero centroid; there is no unit direction
/// to assign to it. Nonempty nonzero means are normalized after each update.
#[allow(dead_code)] // Independent scalar oracle; production uses the Qdrant dot kernel.
pub(super) fn cluster_with_centers(
    data: &[f32],
    dim: usize,
    config: &LmiConfig,
    stopped: &AtomicBool,
) -> OperationResult<(Vec<i64>, Vec<Vec<f64>>, ClusterStats)> {
    cluster_impl(data, dim, config, 1, stopped, AssignmentKernel::ScalarF64)
}

/// `threads` parallelises only the per-row assignment steps. Each row's label
/// is computed by the same sequential kernel regardless of the thread count,
/// and accumulation stays sequential in row order, so labels and centers are
/// bit-identical for every `threads >= 1`.
pub(super) fn cluster_with_centers_qdrant(
    data: &[f32],
    dim: usize,
    config: &LmiConfig,
    threads: usize,
    stopped: &AtomicBool,
) -> OperationResult<(Vec<i64>, Vec<Vec<f64>>, ClusterStats)> {
    cluster_impl(
        data,
        dim,
        config,
        threads,
        stopped,
        AssignmentKernel::QdrantF32,
    )
}

/// Label every row with its best centroid, in row order.
fn assign_all(
    data: &[f32],
    dim: usize,
    centers: &[Vec<f64>],
    centers_f32: Option<&[Vec<f32>]>,
    pool: Option<&rayon::ThreadPool>,
    stopped: &AtomicBool,
) -> OperationResult<Vec<usize>> {
    let label = |row: &[f32]| -> OperationResult<usize> {
        check_process_stopped(stopped)?;
        Ok(match centers_f32 {
            Some(c) => assign_qdrant(row, c),
            None => assign(row, centers),
        })
    };
    match pool {
        Some(pool) => pool.install(|| data.par_chunks_exact(dim).map(label).collect()),
        None => data.chunks_exact(dim).map(label).collect(),
    }
}

fn cluster_impl(
    data: &[f32],
    dim: usize,
    config: &LmiConfig,
    threads: usize,
    stopped: &AtomicBool,
    kernel: AssignmentKernel,
) -> OperationResult<(Vec<i64>, Vec<Vec<f64>>, ClusterStats)> {
    check_process_stopped(stopped)?;
    if threads == 0 {
        return Err(error("LMI spherical KMeans needs at least one thread"));
    }
    let pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("lmi-kmeans-{i}"))
                .build()
                .map_err(|e| error(&format!("LMI spherical KMeans thread pool: {e}")))?,
        )
    } else {
        None
    };
    config.check()?;
    if dim == 0 || !data.len().is_multiple_of(dim) {
        return Err(error("LMI spherical sample has invalid dimensions"));
    }
    let count = data.len() / dim;
    if count < config.n_buckets || data.iter().any(|x| !x.is_finite()) {
        return Err(error("LMI spherical sample is too small or non-finite"));
    }
    // Storage has already applied Qdrant cosine preprocessing. Its near-zero
    // length exception is accepted; all other rows must be unit vectors.
    for row in data.chunks_exact(dim) {
        let squared: f64 = row.iter().map(|&x| f64::from(x).powi(2)).sum();
        if squared >= f64::from(f32::EPSILON) && (squared - 1.0).abs() > 1e-3 {
            return Err(error("LMI spherical sample is not cosine-normalized"));
        }
    }
    config
        .n_buckets
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| error("LMI spherical centroid allocation overflow"))?;

    let mut stats = ClusterStats::default();
    let started = Instant::now();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut order: Vec<_> = (0..count).collect();
    order.shuffle(&mut rng);
    let mut centers = Vec::new();
    centers
        .try_reserve_exact(config.n_buckets)
        .map_err(|e| error(&format!("LMI spherical centroid allocation: {e}")))?;
    for &point in order.iter().take(config.n_buckets) {
        let mut center: Vec<f64> = data[point * dim..(point + 1) * dim]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        unit_norm(&mut center)?;
        centers.push(center);
    }
    drop(order);
    stats.initialization_seconds = started.elapsed().as_secs_f64();

    let mut labels = vec![usize::MAX; count];
    for _ in 0..config.kmeans_iterations {
        check_process_stopped(stopped)?;
        let started = Instant::now();
        let mut changed = false;
        let centers_f32 = match kernel {
            AssignmentKernel::ScalarF64 => None,
            AssignmentKernel::QdrantF32 => Some(
                centers
                    .iter()
                    .map(|c| c.iter().map(|&v| v as f32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            ),
        };
        let assigned = assign_all(
            data,
            dim,
            &centers,
            centers_f32.as_deref(),
            pool.as_ref(),
            stopped,
        )?;
        for (label, bucket) in labels.iter_mut().zip(assigned) {
            changed |= *label != bucket;
            *label = bucket;
        }
        drop(centers_f32); // Avoid f32 centers overlapping the f64 accumulator matrix.
        stats
            .assignment_seconds
            .push(started.elapsed().as_secs_f64());
        stats.iterations += 1;
        if !changed {
            stats.accumulation_seconds.push(0.0);
            stats.normalization_seconds.push(0.0);
            break;
        }

        let started = Instant::now();
        let mut sums = vec![vec![0.0f64; dim]; config.n_buckets];
        let mut sizes = vec![0usize; config.n_buckets];
        for (row, &bucket) in data.chunks_exact(dim).zip(&labels) {
            check_process_stopped(stopped)?;
            sizes[bucket] += 1;
            for (sum, &x) in sums[bucket].iter_mut().zip(row) {
                *sum += f64::from(x);
            }
        }
        stats
            .accumulation_seconds
            .push(started.elapsed().as_secs_f64());

        let started = Instant::now();
        for bucket in 0..config.n_buckets {
            if sizes[bucket] == 0 {
                continue; // Keep the previous centroid, including a zero seed.
            }
            let reciprocal = 1.0 / sizes[bucket] as f64;
            sums[bucket].iter_mut().for_each(|v| *v *= reciprocal);
            if unit_norm(&mut sums[bucket])? {
                centers[bucket] = std::mem::take(&mut sums[bucket]);
            } // A zero mean has no direction: retain the previous center.
        }
        stats
            .normalization_seconds
            .push(started.elapsed().as_secs_f64());
    }

    // The last update can change assignments when the iteration cap is hit.
    let started = Instant::now();
    let mut final_labels = Vec::new();
    final_labels
        .try_reserve_exact(count)
        .map_err(|e| error(&format!("LMI spherical label allocation: {e}")))?;
    stats.sizes = vec![0; config.n_buckets];
    let centers_f32 = match kernel {
        AssignmentKernel::ScalarF64 => None,
        AssignmentKernel::QdrantF32 => Some(
            centers
                .iter()
                .map(|c| c.iter().map(|&v| v as f32).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        ),
    };
    let assigned = assign_all(
        data,
        dim,
        &centers,
        centers_f32.as_deref(),
        pool.as_ref(),
        stopped,
    )?;
    for bucket in assigned {
        final_labels.push(bucket as i64);
        stats.sizes[bucket] += 1;
    }
    stats.final_assignment_seconds = started.elapsed().as_secs_f64();
    Ok((final_labels, centers, stats))
}

#[cfg(test)]
pub(super) fn cluster(
    data: &[f32],
    dim: usize,
    config: &LmiConfig,
    stopped: &AtomicBool,
) -> OperationResult<Vec<i64>> {
    cluster_with_centers_qdrant(data, dim, config, 1, stopped).map(|(labels, _, _)| labels)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    fn config(buckets: usize, iterations: usize, seed: u64) -> LmiConfig {
        LmiConfig {
            n_buckets: buckets,
            sample_size: 32,
            nprobe: 1,
            kmeans_iterations: iterations,
            seed,
            ..LmiConfig::default()
        }
    }

    // Independent tiny reference: compute a full score table, then choose the
    // largest value and smallest bucket ID without using production `assign`.
    fn reference_labels(data: &[f32], dim: usize, centers: &[Vec<f64>]) -> Vec<i64> {
        data.chunks_exact(dim)
            .map(|row| {
                let scores: Vec<f64> = centers
                    .iter()
                    .map(|center| {
                        row.iter()
                            .enumerate()
                            .map(|(j, &x)| f64::from(x) * center[j])
                            .sum()
                    })
                    .collect();
                (0..scores.len())
                    .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a)))
                    .unwrap() as i64
            })
            .collect()
    }

    #[test]
    fn normalized_points_dot_assignment_and_seed_are_deterministic() {
        let data = [
            1.0, 0.0, 0.9, 0.4358899, 0.0, 1.0, -1.0, 0.0, -0.9, -0.4358899, 0.0, -1.0,
        ];
        let stopped = AtomicBool::new(false);
        let cfg = config(2, 12, 42);
        let first = cluster_with_centers(&data, 2, &cfg, &stopped).unwrap();
        let second = cluster_with_centers(&data, 2, &cfg, &stopped).unwrap();
        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
        assert_eq!(first.0, reference_labels(&data, 2, &first.1));
        for center in &first.1 {
            let norm = center.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
        assert!(first.2.iterations <= cfg.kmeans_iterations);
    }

    #[test]
    fn zero_ties_empty_cluster_and_single_bucket() {
        let stopped = AtomicBool::new(false);
        let data = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let (labels, centers, stats) =
            cluster_with_centers(&data, 2, &config(3, 4, 7), &stopped).unwrap();
        assert_eq!(labels[0], 0); // Exact zero dots tie at the first bucket.
        assert_eq!(stats.sizes.iter().sum::<usize>(), 3);
        for center in centers {
            let norm = center.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(norm == 0.0 || (norm - 1.0).abs() < 1e-12);
        }
        let (labels, _, _) = cluster_with_centers(&data, 2, &config(1, 3, 7), &stopped).unwrap();
        assert_eq!(labels, [0, 0, 0]);
        let right_angle = [1.0, 0.0, 0.0, 1.0];
        let (labels, centers, _) =
            cluster_with_centers(&right_angle, 2, &config(1, 3, 7), &stopped).unwrap();
        assert_eq!(labels, [0, 0]);
        let expected = 1.0 / 2.0f64.sqrt();
        assert!((centers[0][0] - expected).abs() < 1e-12);
        assert!((centers[0][1] - expected).abs() < 1e-12);
        let all_zero = [0.0, 0.0, 0.0, 0.0];
        let (labels, centers, _) =
            cluster_with_centers(&all_zero, 2, &config(2, 3, 7), &stopped).unwrap();
        assert_eq!(labels, [0, 0]);
        assert!(centers.iter().all(|c| c == &[0.0, 0.0]));
    }

    #[test]
    fn qdrant_dot_matches_scalar_on_deterministic_fixture() {
        let data = [
            1.0, 0.0, 0.9, 0.4358899, 0.0, 1.0, -1.0, 0.0, -0.9, -0.4358899, 0.0, -1.0,
        ];
        let stopped = AtomicBool::new(false);
        let cfg = config(2, 12, 42);
        let scalar = cluster_with_centers(&data, 2, &cfg, &stopped).unwrap();
        let native = cluster_with_centers_qdrant(&data, 2, &cfg, 1, &stopped).unwrap();
        assert_eq!(scalar.0, native.0);
        assert_eq!(scalar.1, native.1);
        assert_eq!(native.0, cluster(&data, 2, &cfg, &stopped).unwrap());
        let zero = [0.0, 0.0, 0.0, 0.0];
        let native = cluster_with_centers_qdrant(&zero, 2, &config(2, 3, 7), 1, &stopped).unwrap();
        assert_eq!(native.0, [0, 0]);
    }

    #[test]
    fn parallel_assignment_is_bit_identical_to_single_thread() {
        // Deterministic pseudo-random unit vectors (no RNG dependency on thread count).
        let (count, dim) = (1_500, 16);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut data = Vec::with_capacity(count * dim);
        for _ in 0..count {
            let row: Vec<f32> = (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
                })
                .collect();
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            data.extend(row.iter().map(|v| v / norm));
        }
        let stopped = AtomicBool::new(false);
        let cfg = LmiConfig {
            sample_size: count,
            ..config(40, 8, 11)
        };
        let serial = cluster_with_centers_qdrant(&data, dim, &cfg, 1, &stopped).unwrap();
        for threads in [2, 3, 8] {
            let parallel =
                cluster_with_centers_qdrant(&data, dim, &cfg, threads, &stopped).unwrap();
            assert_eq!(serial.0, parallel.0, "labels differ at {threads} threads");
            assert_eq!(serial.1, parallel.1, "centers differ at {threads} threads");
            assert_eq!(serial.2.iterations, parallel.2.iterations);
            assert_eq!(serial.2.sizes, parallel.2.sizes);
        }
        assert!(cluster_with_centers_qdrant(&data, dim, &cfg, 0, &stopped).is_err());
        stopped.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(cluster_with_centers_qdrant(&data, dim, &cfg, 4, &stopped).is_err());
    }

    #[test]
    #[ignore = "Explicit bounded parallel KMeans assignment timing (50K x 768, B 2,048)"]
    fn parallel_assignment_synthetic_benchmark() {
        let (count, dim) = (50_000, 768);
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut data = Vec::with_capacity(count * dim);
        for _ in 0..count {
            let row: Vec<f32> = (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
                })
                .collect();
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            data.extend(row.iter().map(|v| v / norm));
        }
        let stopped = AtomicBool::new(false);
        let cfg = LmiConfig {
            sample_size: count,
            ..config(2_048, 2, 5)
        };
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let mut reference = None;
        for t in [1, threads] {
            let started = Instant::now();
            let result = cluster_with_centers_qdrant(&data, dim, &cfg, t, &stopped).unwrap();
            println!(
                "threads={t} total_seconds={:.3} assignment_seconds={:?} accumulation_seconds={:?} final_assignment_seconds={:.3}",
                started.elapsed().as_secs_f64(),
                result.2.assignment_seconds,
                result.2.accumulation_seconds,
                result.2.final_assignment_seconds,
            );
            match &reference {
                None => reference = Some(result),
                Some(r) => {
                    assert_eq!(r.0, result.0);
                    assert_eq!(r.1, result.1);
                }
            }
        }
    }

    #[test]
    fn stable_stop_iteration_cap_malformed_input_and_cancellation() {
        let data = [1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0];
        let stopped = AtomicBool::new(false);
        let (_, _, stable) = cluster_with_centers(&data, 2, &config(2, 20, 3), &stopped).unwrap();
        assert!(stable.iterations < 20);
        let (labels, centers, capped) =
            cluster_with_centers(&data, 2, &config(2, 1, 3), &stopped).unwrap();
        assert_eq!(capped.iterations, 1);
        assert_eq!(labels, reference_labels(&data, 2, &centers));
        assert!(cluster_with_centers(&data, 0, &config(2, 2, 3), &stopped).is_err());
        assert!(cluster_with_centers(&data[..7], 2, &config(2, 2, 3), &stopped).is_err());
        assert!(
            cluster_with_centers(&[f32::NAN, 0.0, 0.0, 1.0], 2, &config(2, 2, 3), &stopped)
                .is_err()
        );
        assert!(cluster_with_centers(&data[..2], 2, &config(2, 2, 3), &stopped).is_err());
        stopped.store(true, Ordering::Relaxed);
        assert!(cluster_with_centers(&data, 2, &config(2, 2, 3), &stopped).is_err());
    }
}
