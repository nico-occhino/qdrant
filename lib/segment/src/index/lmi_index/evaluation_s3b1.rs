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

pub(super) fn model(d: usize, h: usize, b: usize) -> MlpRouter {
    MlpRouter {
        layers: vec![dense(d, h, 11), RouterLayer::ReLU, dense(h, b, 29)],
    }
}

pub(super) fn inputs(rows: usize, d: usize) -> Vec<f32> {
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

#[test]
#[ignore = "Explicit S.3B2 bounded tch-vs-native corpus-routing benchmark"]
fn phase_s3b2_tch_microbenchmark() {
    let output = std::env::var("LMI_S3B2_MICRO_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output).exists());
    let torch_threads = std::env::var("LMI_S3B2_TORCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(1);
    assert!((1..=8).contains(&torch_threads));
    tch::set_num_threads(torch_threads);
    let stopped = AtomicBool::new(false);
    let shapes = [
        ("A", 128, 64, 64, 4096),
        ("B", 768, 64, 64, 2048),
        ("C", 768, 512, 1024, 1024),
        ("D", 768, 512, 10_000, 2048),
        ("E", 768, 512, 31_622, 1024),
    ];
    let mut records = Vec::new();
    for (name, d, h, b, rows) in shapes {
        let model = model(d, h, b);
        let trained = super::training::TrainedRouter::from_native_for_test(model.clone()).unwrap();
        let data = inputs(rows, d);
        let mut scalar = BuildRouter::new(&model, &stopped).unwrap();
        let started = Instant::now();
        let expected: Vec<_> = data
            .chunks_exact(d)
            .map(|row| scalar.top_bucket(black_box(row), &stopped).unwrap())
            .collect();
        let seconds = started.elapsed().as_secs_f64();
        records.push(json!({"shape":name,"d":d,"h":h,"b":b,"rows":rows,
            "backend":"scalar","k":1,"seconds":seconds,"rows_per_second":rows as f64/seconds,
            "microseconds_per_vector":seconds*1e6/rows as f64,
            "mismatches":0,"torch_threads":torch_threads,"process_threads":process_threads()}));
        let ks: &[usize] = if name == "D" {
            &[32, 64, 128, 256, 512, 1024]
        } else if name == "E" {
            &[32, 64, 128, 256, 512]
        } else {
            &[32, 128, 256, 512]
        };
        for backend in ["native_batch", "tch_batch"] {
            for &k in ks {
                let mut native = (backend == "native_batch")
                    .then(|| BuildBatchRouter::new(&model, k, &stopped).unwrap());
                let started = Instant::now();
                let mut actual = Vec::with_capacity(rows);
                for chunk in data.chunks(k * d) {
                    if let Some(native) = native.as_mut() {
                        actual.extend_from_slice(black_box(
                            native.top_buckets(black_box(chunk), &stopped).unwrap(),
                        ));
                    } else {
                        actual.extend(black_box(
                            trained.top_buckets(black_box(chunk), &stopped).unwrap(),
                        ));
                    }
                }
                let seconds = started.elapsed().as_secs_f64();
                let differing: Vec<_> = expected
                    .iter()
                    .zip(&actual)
                    .enumerate()
                    .filter_map(|(i, (a, b))| (a != b).then_some(i))
                    .collect();
                let mut margins = Vec::with_capacity(differing.len());
                for &i in &differing {
                    let logits = model.forward(&data[i * d..(i + 1) * d]).unwrap();
                    let mut best = f32::NEG_INFINITY;
                    let mut second = f32::NEG_INFINITY;
                    for logit in logits {
                        if logit > best {
                            second = best;
                            best = logit;
                        } else if logit > second {
                            second = logit;
                        }
                    }
                    margins.push(best - second);
                }
                let record = json!({"shape":name,"d":d,"h":h,"b":b,"rows":rows,
                    "backend":backend,"k":k,"seconds":seconds,
                    "rows_per_second":rows as f64/seconds,
                    "microseconds_per_vector":seconds*1e6/rows as f64,
                    "mismatches":differing.len(),"mismatch_rate":differing.len() as f64/rows as f64,
                    "mismatch_rows":differing.iter().take(16).collect::<Vec<_>>(),
                    "mismatch_native_margins":margins.iter().take(16).collect::<Vec<_>>(),
                    "mismatch_margin_min":margins.iter().copied().reduce(f32::min),
                    "mismatch_margin_max":margins.iter().copied().reduce(f32::max),
                    "input_bytes":4*k*d,"hidden_bytes":4*k*h,"logits_bytes":4*k*b,
                    "top1_bytes":8*k,"native_workspace_bytes":native.as_ref().map(|n|n.workspace_bytes()),
                    "torch_threads":torch_threads,"process_threads":process_threads()});
                eprintln!("S3B2_MICRO {record}");
                records.push(record);
            }
        }
    }
    std::fs::write(output, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
}

#[test]
#[ignore = "Explicit S.3B2 corpus parity check against persisted native router"]
fn phase_s3b2_corpus_parity() {
    use crate::types::Distance;
    use std::io::{BufReader, Read};
    let data_root = std::path::PathBuf::from(std::env::var("LMI_S3B2_DATA").unwrap());
    let router_path = std::path::PathBuf::from(std::env::var("LMI_S3B2_ROUTER").unwrap());
    let output = std::env::var("LMI_S3B2_PARITY_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output).exists());
    let k: usize = std::env::var("LMI_S3B2_PARITY_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    assert!((1..=1024).contains(&k));
    tch::set_num_threads(1);
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(data_root.join("dataset.json")).unwrap()).unwrap();
    let n = metadata["corpus_count"].as_u64().unwrap() as usize;
    let d = metadata["dimension"].as_u64().unwrap() as usize;
    let metric: Distance = serde_json::from_value(metadata["metric"].clone()).unwrap();
    #[derive(serde::Deserialize)]
    struct SavedRouter {
        sample_offsets: Vec<u32>,
        router: Option<MlpRouter>,
    }
    let saved: SavedRouter = bincode::deserialize(&std::fs::read(router_path).unwrap()).unwrap();
    let _sample_count = saved.sample_offsets.len();
    let router = saved.router.expect("trained LMI router");
    let (actual_d, b) = router.validate().unwrap();
    assert_eq!(actual_d, d);
    let trained = super::training::TrainedRouter::from_native_for_test(router.clone()).unwrap();
    let stopped = AtomicBool::new(false);
    let mut native = BuildRouter::new(&router, &stopped).unwrap();
    let mut reader = BufReader::new(std::fs::File::open(data_root.join("corpus.f32")).unwrap());
    let mut raw = vec![0u8; d * 4];
    let mut inputs = Vec::with_capacity(k * d);
    let mut native_counts = vec![0usize; b];
    let mut tch_counts = vec![0usize; b];
    let mut mismatch_examples = Vec::new();
    let mut margins = Vec::new();
    let mut native_seconds = 0.0;
    let mut tch_seconds = 0.0;
    for start in (0..n).step_by(k) {
        inputs.clear();
        let rows = (n - start).min(k);
        for _ in 0..rows {
            reader.read_exact(&mut raw).unwrap();
            let row: Vec<f32> = raw
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            inputs.extend(metric.preprocess_vector::<f32>(row));
        }
        let t = Instant::now();
        let native_ids: Vec<_> = inputs
            .chunks_exact(d)
            .map(|row| native.top_bucket(row, &stopped).unwrap())
            .collect();
        native_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let tch_ids = trained.top_buckets(&inputs, &stopped).unwrap();
        tch_seconds += t.elapsed().as_secs_f64();
        for (i, (&a, &c)) in native_ids.iter().zip(&tch_ids).enumerate() {
            native_counts[a] += 1;
            tch_counts[c] += 1;
            if a != c {
                let logits = router.forward(&inputs[i * d..(i + 1) * d]).unwrap();
                let mut best = f32::NEG_INFINITY;
                let mut second = f32::NEG_INFINITY;
                for logit in logits {
                    if logit > best {
                        second = best;
                        best = logit;
                    } else if logit > second {
                        second = logit;
                    }
                }
                let margin = best - second;
                margins.push(margin);
                if mismatch_examples.len() < 32 {
                    mismatch_examples
                        .push(json!({"row":start+i,"native":a,"tch":c,"native_margin":margin}));
                }
            }
        }
    }
    assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    margins.sort_by(|a, b| a.total_cmp(b));
    let result = json!({"n":n,"d":d,"b":b,"k":k,"metric":metric,
        "mismatches":margins.len(),"mismatch_rate":margins.len() as f64/n as f64,
        "mismatch_examples":mismatch_examples,
        "margin_min":margins.first(),"margin_median":margins.get(margins.len()/2),
        "margin_max":margins.last(),
        "native_counts":native_counts,"tch_counts":tch_counts,
        "native_seconds":native_seconds,"tch_seconds":tch_seconds,
        "torch_threads":1,"process_threads":process_threads()});
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    eprintln!("S3B2_CORPUS_PARITY {result}");
}
