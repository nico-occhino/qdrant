//! Build-only CPU training. No HDF5, Python orchestration or standalone scoring.
use std::sync::Once;
use std::sync::atomic::AtomicBool;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use tch::nn::{Module, OptimizerConfig};
use tch::{Device, Tensor, nn};

use super::{LinearLayer, LmiConfig, MlpRouter, RouterLayer};
use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};

fn torch_error(err: tch::TchError) -> OperationError {
    OperationError::service_error(format!("LMI training: {err}"))
}

/// Stable-Rust Lloyd clustering with seeded random-sample initialization.
/// Empty clusters keep their previous centroid; ties prefer the first cluster.
pub(super) fn cluster(
    data: &[f32],
    dim: usize,
    config: &LmiConfig,
    stopped: &AtomicBool,
) -> OperationResult<Vec<i64>> {
    let count = data.len() / dim;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut order: Vec<_> = (0..count).collect();
    order.shuffle(&mut rng);
    let mut centers: Vec<Vec<f64>> = order[..config.n_buckets]
        .iter()
        .map(|&i| {
            data[i * dim..(i + 1) * dim]
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        })
        .collect();
    let assign = |row: &[f32], centers: &[Vec<f64>]| {
        centers
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let d: f64 = row
                    .iter()
                    .zip(c)
                    .map(|(&x, &y)| (f64::from(x) - y).powi(2))
                    .sum();
                (i, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .unwrap()
            .0
    };
    let mut labels = vec![usize::MAX; count];
    for _ in 0..config.kmeans_iterations {
        check_process_stopped(stopped)?;
        let mut sums = vec![vec![0.0f64; dim]; config.n_buckets];
        let mut sizes = vec![0usize; config.n_buckets];
        let mut changed = false;
        for (i, row) in data.chunks_exact(dim).enumerate() {
            check_process_stopped(stopped)?;
            let label = assign(row, &centers);
            changed |= labels[i] != label;
            labels[i] = label;
            sizes[label] += 1;
            for (sum, &x) in sums[label].iter_mut().zip(row) {
                *sum += f64::from(x);
            }
        }
        for b in 0..config.n_buckets {
            if sizes[b] > 0 {
                for j in 0..dim {
                    centers[b][j] = sums[b][j] / sizes[b] as f64;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Assign against the final centroids, including when the iteration limit was reached.
    let mut result = Vec::with_capacity(count);
    for row in data.chunks_exact(dim) {
        check_process_stopped(stopped)?;
        result.push(assign(row, &centers) as i64);
    }
    Ok(result)
}

pub(super) fn train(
    data: &[f32],
    dim: usize,
    config: &LmiConfig,
    stopped: &AtomicBool,
) -> OperationResult<MlpRouter> {
    check_process_stopped(stopped)?;
    let labels = cluster(data, dim, config, stopped)?;
    // The only Torch consumer in Qdrant. Set the process-wide inter-op policy once.
    // Intra-op settings are applied on each builder thread; never exceed one CPU.
    static INIT: Once = Once::new();
    INIT.call_once(|| tch::set_num_interop_threads(1));
    tch::set_num_threads(1);
    let store = nn::VarStore::new(Device::Cpu);
    let options = nn::LinearConfig {
        ws_init: nn::Init::Const(0.0),
        bs_init: Some(nn::Init::Const(0.0)),
        bias: true,
    };
    let mut first = nn::linear(
        &store.root() / "hidden",
        dim as i64,
        config.hidden_dim as i64,
        options,
    );
    let mut last = nn::linear(
        &store.root() / "output",
        config.hidden_dim as i64,
        config.n_buckets as i64,
        options,
    );
    // Local Rust RNG avoids process-global Torch seeds and cross-build interference.
    let mut rng = StdRng::seed_from_u64(config.seed);
    for (layer, input, output) in [
        (&mut first, dim, config.hidden_dim),
        (&mut last, config.hidden_dim, config.n_buckets),
    ] {
        let bound = (1.0 / input as f32).sqrt();
        let weights: Vec<f32> = (0..input * output)
            .map(|_| rng.random_range(-bound..bound))
            .collect();
        let biases: Vec<f32> = (0..output)
            .map(|_| rng.random_range(-bound..bound))
            .collect();
        tch::no_grad(|| -> Result<(), tch::TchError> {
            layer.ws.f_copy_(
                &Tensor::f_from_slice(&weights)?.f_reshape([output as i64, input as i64])?,
            )?;
            layer
                .bs
                .as_mut()
                .unwrap()
                .f_copy_(&Tensor::f_from_slice(&biases)?)?;
            Ok(())
        })
        .map_err(torch_error)?;
    }
    let x = Tensor::f_from_slice(data)
        .and_then(|t| t.f_reshape([labels.len() as i64, dim as i64]))
        .map_err(torch_error)?;
    let y = Tensor::f_from_slice(&labels).map_err(torch_error)?;
    let mut optimizer = nn::Adam::default()
        .build(&store, 1e-3)
        .map_err(torch_error)?;
    let mut order: Vec<i64> = (0..labels.len() as i64).collect();
    for epoch in 0..config.epochs {
        check_process_stopped(stopped)?;
        order.shuffle(&mut rng);
        for batch in order.chunks(config.batch_size) {
            check_process_stopped(stopped)?;
            let indices = Tensor::f_from_slice(batch).map_err(torch_error)?;
            let bx = x.f_index_select(0, &indices).map_err(torch_error)?;
            let by = y.f_index_select(0, &indices).map_err(torch_error)?;
            let loss = last
                .forward(&first.forward(&bx).relu())
                .cross_entropy_for_logits(&by);
            if !loss.double_value(&[]).is_finite() {
                return Err(OperationError::service_error(
                    "Non-finite LMI training loss",
                ));
            }
            optimizer.backward_step(&loss);
        }
        log::debug!("LMI training epoch {}/{}", epoch + 1, config.epochs);
    }
    let export = |layer: &nn::Linear, input, output| -> OperationResult<RouterLayer> {
        Ok(RouterLayer::Linear(LinearLayer {
            in_features: input,
            out_features: output,
            weights: Vec::<f32>::try_from(&layer.ws.f_view([-1]).map_err(torch_error)?)
                .map_err(torch_error)?,
            bias: Vec::<f32>::try_from(layer.bs.as_ref().unwrap()).map_err(torch_error)?,
        }))
    };
    let router = MlpRouter {
        layers: vec![
            export(&first, dim, config.hidden_dim)?,
            RouterLayer::ReLU,
            export(&last, config.hidden_dim, config.n_buckets)?,
        ],
    };
    router.validate()?;
    // Validate the ownership/runtime boundary on up to 32 actual sampled inputs.
    for row in data.chunks_exact(dim).take(32) {
        check_process_stopped(stopped)?;
        let input = Tensor::f_from_slice(row).map_err(torch_error)?;
        let reference = tch::no_grad(|| last.forward(&first.forward(&input).relu()));
        let reference = Vec::<f32>::try_from(reference).map_err(torch_error)?;
        let native = router.forward(row)?;
        if reference
            .iter()
            .zip(native)
            .any(|(&a, b)| !a.is_finite() || (a - b).abs() > 1e-4 + 1e-4 * a.abs())
        {
            return Err(OperationError::service_error(
                "LMI native/Torch export mismatch",
            ));
        }
    }
    check_process_stopped(stopped)?;
    Ok(router)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lloyd_separates_clusters_and_obeys_cancellation() {
        let config = LmiConfig {
            n_buckets: 2,
            sample_size: 4,
            nprobe: 1,
            kmeans_iterations: 10,
            ..Default::default()
        };
        let data = [-10.0, -9.0, 9.0, 10.0];
        let labels = cluster(&data, 1, &config, &AtomicBool::new(false)).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
        assert!(cluster(&data, 1, &config, &AtomicBool::new(true)).is_err());
        assert!(train(&data, 1, &config, &AtomicBool::new(true)).is_err());
    }
}
