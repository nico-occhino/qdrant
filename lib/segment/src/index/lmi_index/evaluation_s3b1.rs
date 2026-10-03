//! Opt-in synthetic compute benchmark for the existing build router.
//! Inputs and weights are deterministic; this is not a corpus benchmark.
use std::hint::black_box;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use serde_json::json;

use super::routing::{BuildBatchRouter, BuildRouter, LinearLayer, MlpRouter, RouterLayer};

fn dense(input: usize, output: usize, salt: usize) -> RouterLayer {
    let weights = (0..input * output)
        .map(|i| (((i.wrapping_mul(17).wrapping_add(salt)) % 103) as f32 - 51.0) / 8192.0)
        .collect();
    let bias = (0..output)
        .map(|i| ((i.wrapping_mul(13).wrapping_add(salt) % 37) as f32 - 18.0) / 4096.0)
        .collect();
    RouterLayer::Linear(LinearLayer {
        in_features: input,
        out_features: output,
        weights,
        bias,
    })
}

fn model(d: usize, h: usize, b: usize) -> MlpRouter {
    MlpRouter {
        layers: vec![dense(d, h, 11), RouterLayer::ReLU, dense(h, b, 29)],
    }
}

fn inputs(rows: usize, d: usize) -> Vec<f32> {
    (0..rows * d)
        .map(|i| ((i.wrapping_mul(31) % 211) as f32 - 105.0) / 4096.0)
        .collect()
}

fn process_threads() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| {
            line.strip_prefix("Threads:")
                .map(|v| v.trim().parse().unwrap())
        })
        .unwrap()
}

#[test]
#[ignore = "Explicit S.3B1 deterministic compute microbenchmark"]
fn phase_s3b1_router_microbenchmark() {
    let output = std::env::var("LMI_S3B1_MICRO_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output).exists());
    let stopped = AtomicBool::new(false);
    let shapes = [
        ("A", 128, 64, 64, 4096),
        ("B", 768, 64, 64, 1024),
        ("C", 768, 512, 1024, 256),
        ("D", 768, 512, 10_000, 256),
        ("E", 768, 512, 31_622, 256),
    ];
    let mut records = Vec::new();
    for (name, d, h, b, rows) in shapes {
        let model = model(d, h, b);
        let data = inputs(rows, d);
        let mut scalar = BuildRouter::new(&model, &stopped).unwrap();
        let mut expected = Vec::with_capacity(rows);
        let start = Instant::now();
        for row in data.chunks_exact(d) {
            expected.push(scalar.top_bucket(black_box(row), &stopped).unwrap());
        }
        let scalar_seconds = start.elapsed().as_secs_f64();
        for k in [1, 16, 64, 256] {
            let mut batch = BuildBatchRouter::new(&model, k, &stopped).unwrap();
            let workspace_bytes = batch.workspace_bytes() + k * d * 4 + k * 4;
            let start = Instant::now();
            let mut actual = Vec::with_capacity(rows);
            for chunk in data.chunks(k * d) {
                actual.extend_from_slice(black_box(
                    batch.top_buckets(black_box(chunk), &stopped).unwrap(),
                ));
            }
            let seconds = start.elapsed().as_secs_f64();
            let mismatch = expected.iter().zip(&actual).filter(|(a, b)| a != b).count();
            assert_eq!(mismatch, 0, "shape={name} k={k}");
            black_box(&actual);
            let record = json!({
                "shape":name,"d":d,"h":h,"b":b,"rows":rows,"k":k,
                "seconds":seconds,"rows_per_second":rows as f64 / seconds,
                "microseconds_per_vector":seconds * 1e6 / rows as f64,
                "workspace_bytes":workspace_bytes,"assignment_mismatches":mismatch,
                "native_compute_threads":1,"process_threads":process_threads(),
                "scalar_seconds":scalar_seconds
            });
            eprintln!("S3B1_MICRO {record}");
            records.push(record);
        }
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
}

#[test]
#[ignore = "Explicit S.3B1 large-router stage timing proxy"]
fn phase_s3b1_large_b_stage_profile() {
    let output = std::env::var("LMI_S3B1_PROFILE_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output).exists());
    let (d, h, b, rows) = (768, 512, 10_000, 128);
    let model = model(d, h, b);
    let input = inputs(rows, d);
    let RouterLayer::Linear(first) = &model.layers[0] else {
        unreachable!()
    };
    let RouterLayer::Linear(second) = &model.layers[2] else {
        unreachable!()
    };
    let start = Instant::now();
    let gathered = input.clone();
    let gather_seconds = start.elapsed().as_secs_f64();
    let mut hidden = Vec::with_capacity(rows * h);
    let start = Instant::now();
    for row in gathered.chunks_exact(d) {
        for (weights, &bias) in first.weights.chunks_exact(d).zip(&first.bias) {
            let mut sum = bias;
            for (&weight, &value) in weights.iter().zip(row) {
                sum += weight * value;
            }
            hidden.push(sum);
        }
    }
    let first_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    hidden.iter_mut().for_each(|value| *value = value.max(0.0));
    let relu_seconds = start.elapsed().as_secs_f64();
    let mut logits = Vec::with_capacity(rows * b);
    let start = Instant::now();
    for row in hidden.chunks_exact(h) {
        for (weights, &bias) in second.weights.chunks_exact(h).zip(&second.bias) {
            let mut sum = bias;
            for (&weight, &value) in weights.iter().zip(row) {
                sum += weight * value;
            }
            logits.push(sum);
        }
    }
    let second_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut buckets = Vec::with_capacity(rows);
    for row in logits.chunks_exact(b) {
        let mut best = 0;
        for i in 1..b {
            if row[i] > row[best] {
                best = i;
            }
        }
        buckets.push(best);
    }
    let argmax_seconds = start.elapsed().as_secs_f64();
    black_box(buckets);
    let result = json!({"d":d,"h":h,"b":b,"rows":rows,
        "gather_seconds":gather_seconds,"first_dense_seconds":first_seconds,
        "relu_seconds":relu_seconds,"second_dense_seconds":second_seconds,
        "argmax_seconds":argmax_seconds,"native_compute_threads":1,
        "process_threads":process_threads(),"method":"timed equivalent ordered scalar loops; stage proxy, not instrumented production code"});
    eprintln!("S3B1_PROFILE {result}");
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
}
