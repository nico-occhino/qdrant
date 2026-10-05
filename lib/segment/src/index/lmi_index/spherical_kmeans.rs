//! Scalar spherical KMeans for Qdrant-preprocessed cosine samples.
//! The independent Euclidean Lloyd path remains in `training.rs`.
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

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
    cluster_impl(data, dim, config, stopped, AssignmentKernel::ScalarF64)
}

pub(super) fn cluster_with_centers_qdrant(
    data: &[f32],
    dim: usize,
    config: &LmiConfig,
    stopped: &AtomicBool,
) -> OperationResult<(Vec<i64>, Vec<Vec<f64>>, ClusterStats)> {
    cluster_impl(data, dim, config, stopped, AssignmentKernel::QdrantF32)
}

fn cluster_impl(
    data: &[f32],
    dim: usize,
    config: &LmiConfig,
    stopped: &AtomicBool,
    kernel: AssignmentKernel,
) -> OperationResult<(Vec<i64>, Vec<Vec<f64>>, ClusterStats)> {
    check_process_stopped(stopped)?;
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
        for (point, row) in data.chunks_exact(dim).enumerate() {
            check_process_stopped(stopped)?;
            let bucket = match &centers_f32 {
                Some(c) => assign_qdrant(row, c),
                None => assign(row, &centers),
            };
            changed |= labels[point] != bucket;
            labels[point] = bucket;
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
    for row in data.chunks_exact(dim) {
        check_process_stopped(stopped)?;
        let bucket = match &centers_f32 {
            Some(c) => assign_qdrant(row, c),
            None => assign(row, &centers),
        };
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
    cluster_with_centers_qdrant(data, dim, config, stopped).map(|(labels, _, _)| labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

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
        let native = cluster_with_centers_qdrant(&data, 2, &cfg, &stopped).unwrap();
        assert_eq!(scalar.0, native.0);
        assert_eq!(scalar.1, native.1);
        assert_eq!(native.0, cluster(&data, 2, &cfg, &stopped).unwrap());
        let zero = [0.0, 0.0, 0.0, 0.0];
        let native = cluster_with_centers_qdrant(&zero, 2, &config(2, 3, 7), &stopped).unwrap();
        assert_eq!(native.0, [0, 0]);
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
