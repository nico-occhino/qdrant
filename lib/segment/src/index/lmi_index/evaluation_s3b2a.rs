//! S.3B2a diagnostic: corpus and controlled ULP neighborhoods only.
use std::io::{BufReader, Read};
use std::sync::atomic::AtomicBool;

use serde_json::{Value, json};

use super::routing::{BuildRouter, MlpRouter};
use super::training::TrainedRouter;
use crate::types::Distance;

#[derive(serde::Deserialize)]
struct SavedRouter {
    sample_offsets: Vec<u32>,
    router: Option<MlpRouter>,
}

fn order(scores: &[f32], a: usize, b: usize) -> bool {
    scores[a] > scores[b] || (scores[a] == scores[b] && a < b)
}

fn top_two(scores: &[f32]) -> (usize, usize) {
    assert!(scores.len() >= 2);
    let (mut first, mut second) = if order(scores, 0, 1) { (0, 1) } else { (1, 0) };
    for i in 2..scores.len() {
        if order(scores, i, first) {
            second = first;
            first = i;
        } else if order(scores, i, second) {
            second = i;
        }
    }
    (first, second)
}

fn rank(scores: &[f32], id: usize) -> usize {
    1 + (0..scores.len())
        .filter(|&other| order(scores, other, id))
        .count()
}

fn top_m(scores: &[f32], m: usize) -> Vec<Value> {
    let mut ids: Vec<_> = (0..scores.len()).collect();
    ids.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ids.into_iter()
        .take(m)
        .map(|id| json!({"bucket":id,"score":scores[id]}))
        .collect()
}

const MULTIPLIERS: [f64; 12] = [
    0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
];

#[test]
#[ignore = "Explicit S.3B2a full-corpus tie-boundary diagnostic"]
fn phase_s3b2a_corpus_boundary_diagnostic() {
    let output = std::env::var("LMI_S3B2A_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output).exists());
    let data_root = std::path::PathBuf::from(std::env::var("LMI_S3B2A_DATA").unwrap());
    let router_path = std::path::PathBuf::from(std::env::var("LMI_S3B2A_ROUTER").unwrap());
    let seed_offsets: Vec<usize> = std::env::var("LMI_S3B2A_SEED_OFFSETS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(data_root.join("dataset.json")).unwrap()).unwrap();
    let n = metadata["corpus_count"].as_u64().unwrap() as usize;
    let d = metadata["dimension"].as_u64().unwrap() as usize;
    let metric: Distance = serde_json::from_value(metadata["metric"].clone()).unwrap();
    let saved: SavedRouter = bincode::deserialize(&std::fs::read(router_path).unwrap()).unwrap();
    let _samples = saved.sample_offsets.len();
    let router = saved.router.expect("trained router");
    let (router_d, b) = router.validate().unwrap();
    assert_eq!(router_d, d);
    assert!(b >= 2);
    let trained = TrainedRouter::from_native_for_test(router.clone()).unwrap();
    let stopped = AtomicBool::new(false);
    let mut native = BuildRouter::new(&router, &stopped).unwrap();
    tch::set_num_threads(1);
    let mut reader = BufReader::new(std::fs::File::open(data_root.join("corpus.f32")).unwrap());
    let mut raw = vec![0u8; d * 4];
    let k = 256usize;
    let mut inputs = Vec::with_capacity(k * d);
    let mut fallback_counts = [0usize; MULTIPLIERS.len()];
    let mut remaining_mismatches = [0usize; MULTIPLIERS.len()];
    let mut mismatch_records = Vec::new();
    let mut seed_rows = Vec::new();
    let mut total_mismatches = 0usize;
    let mut tie_safe_mismatches = 0usize;
    let mut tie_safe_fallbacks = 0usize;
    let mut tie_safe_seconds = 0.0f64;
    let mut zero_tch_margins = 0usize;
    let mut smallest_positive_tch_margin = f32::INFINITY;
    for start in (0..n).step_by(k) {
        inputs.clear();
        let rows = (n - start).min(k);
        for i in 0..rows {
            reader.read_exact(&mut raw).unwrap();
            let row: Vec<f32> = raw
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            let row = metric.preprocess_vector::<f32>(row);
            if seed_offsets.contains(&(start + i)) {
                seed_rows.push((start + i, row.clone()));
            }
            inputs.extend_from_slice(&row);
        }
        let native_ids: Vec<_> = inputs
            .chunks_exact(d)
            .map(|row| native.top_bucket(row, &stopped).unwrap())
            .collect();
        let (torch_logits, torch_ids) = trained.logits_for_test(&inputs).unwrap();
        let started = std::time::Instant::now();
        let (tie_safe_ids, fallbacks) = trained
            .top_buckets_tie_safe(&inputs, &stopped, &mut native)
            .unwrap();
        tie_safe_seconds += started.elapsed().as_secs_f64();
        tie_safe_fallbacks += fallbacks;
        tie_safe_mismatches += native_ids
            .iter()
            .zip(&tie_safe_ids)
            .filter(|(a, b)| a != b)
            .count();
        for i in 0..rows {
            let offset = start + i;
            let scores = &torch_logits[i * b..(i + 1) * b];
            let (torch_first, torch_second) = top_two(scores);
            let margin = scores[torch_first] - scores[torch_second];
            if margin == 0.0 {
                zero_tch_margins += 1;
            } else {
                smallest_positive_tch_margin = smallest_positive_tch_margin.min(margin);
            }
            let mismatch = native_ids[i] != torch_ids[i];
            total_mismatches += usize::from(mismatch);
            for (j, multiplier) in MULTIPLIERS.iter().enumerate() {
                let threshold =
                    (*multiplier as f32) * f32::EPSILON * (1.0 + scores[torch_first].abs());
                let fallback = margin <= threshold;
                fallback_counts[j] += usize::from(fallback);
                remaining_mismatches[j] += usize::from(mismatch && !fallback);
            }
            if mismatch {
                let native_scores = router.forward(&inputs[i * d..(i + 1) * d]).unwrap();
                let (native_first, native_second) = top_two(&native_scores);
                mismatch_records.push(json!({
                    "offset":offset,
                    "tch_argmax":torch_ids[i],
                    "tch_top1":torch_first,"tch_top2":torch_second,
                    "tch_score1":scores[torch_first],"tch_score2":scores[torch_second],
                    "tch_margin":margin,
                    "native_top1":native_first,"native_top2":native_second,
                    "native_score1":native_scores[native_first],
                    "native_score2":native_scores[native_second],
                    "native_margin":native_scores[native_first]-native_scores[native_second],
                    "native_winner_rank_in_tch":rank(scores,native_first),
                    "tch_top8":top_m(scores,8),"native_top8":top_m(&native_scores,8)
                }));
            }
        }
    }
    assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    let mut near_rows = Vec::new();
    for (offset, base) in seed_rows {
        let coordinates: Vec<_> = base
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0 && v.is_finite())
            .take(16)
            .map(|(i, _)| i)
            .collect();
        for coordinate in coordinates {
            for delta_ulp in [
                -128i32, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 128,
            ] {
                let mut row = base.clone();
                let bits = row[coordinate].to_bits() as i64 + delta_ulp as i64;
                if bits <= 0 || bits > u32::MAX as i64 {
                    continue;
                }
                row[coordinate] = f32::from_bits(bits as u32);
                if !row[coordinate].is_finite() || row[coordinate] < 0.0 {
                    continue;
                }
                let native_id = native.top_bucket(&row, &stopped).unwrap();
                let native_scores = router.forward(&row).unwrap();
                let (native_first, native_second) = top_two(&native_scores);
                let (torch_logits, torch_ids) = trained.logits_for_test(&row).unwrap();
                let (tie_safe_ids, tie_fallbacks) = trained
                    .top_buckets_tie_safe(&row, &stopped, &mut native)
                    .unwrap();
                let (torch_first, torch_second) = top_two(&torch_logits);
                near_rows.push(json!({
                    "base_offset":offset,"coordinate":coordinate,"delta_ulp":delta_ulp,
                    "native_id":native_id,"tch_id":torch_ids[0],
                    "tie_safe_id":tie_safe_ids[0],"tie_safe_fallback":tie_fallbacks==1,
                    "native_margin":native_scores[native_first]-native_scores[native_second],
                    "tch_margin":torch_logits[torch_first]-torch_logits[torch_second],
                    "tch_score1":torch_logits[torch_first],
                    "native_winner_rank_in_tch":rank(&torch_logits,native_first)
                }));
            }
        }
    }
    let thresholds: Vec<_> = MULTIPLIERS
        .iter()
        .enumerate()
        .map(|(i, &multiplier)| {
            json!({
                "epsilon_multiplier":multiplier,
                "fallback_count":fallback_counts[i],
                "fallback_fraction":fallback_counts[i] as f64/n as f64,
                "remaining_mismatches":remaining_mismatches[i]
            })
        })
        .collect();
    let result = json!({
        "n":n,"d":d,"b":b,"k":k,"metric":metric,
        "mismatch_count":total_mismatches,"mismatches":mismatch_records,
        "tie_safe_mismatches":tie_safe_mismatches,"tie_safe_fallbacks":tie_safe_fallbacks,
        "tie_safe_seconds":tie_safe_seconds,
        "zero_tch_margins":zero_tch_margins,
        "smallest_positive_tch_margin":smallest_positive_tch_margin.is_finite().then_some(smallest_positive_tch_margin),
        "thresholds":thresholds,"seed_offsets":seed_offsets,"near_rows":near_rows
    });
    assert_eq!(tie_safe_mismatches, 0);
    assert!(
        near_rows
            .iter()
            .all(|row| row["native_id"] == row["tie_safe_id"])
    );
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    eprintln!("S3B2A_DIAGNOSTIC n={n} mismatches={total_mismatches}");
}

#[test]
fn controlled_exact_and_small_margin_fixture() {
    use super::routing::{LinearLayer, RouterLayer};
    let stopped = AtomicBool::new(false);
    let mut margins = Vec::new();
    for delta in [0.0f32, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4] {
        let router = MlpRouter {
            layers: vec![
                RouterLayer::Linear(LinearLayer {
                    in_features: 1,
                    out_features: 1,
                    weights: vec![1.0],
                    bias: vec![0.0],
                }),
                RouterLayer::ReLU,
                RouterLayer::Linear(LinearLayer {
                    in_features: 1,
                    out_features: 2,
                    weights: vec![1.0, 1.0],
                    bias: vec![0.0, delta],
                }),
            ],
        };
        let trained = TrainedRouter::from_native_for_test(router.clone()).unwrap();
        let native = router.forward(&[1.0]).unwrap();
        let (torch, ids) = trained.logits_for_test(&[1.0]).unwrap();
        let (a, b) = top_two(&native);
        let (c, d) = top_two(&torch);
        margins.push((native[a] - native[b], torch[c] - torch[d], ids[0]));
        assert_eq!(trained.top_buckets(&[1.0], &stopped).unwrap()[0], ids[0]);
    }
    assert_eq!(margins[0], (0.0, 0.0, 0));
    assert!(margins[5].0 > margins[4].0);
    assert!(margins[5].1 > margins[4].1);
}

#[test]
fn tie_safe_boundary_contract() {
    use super::routing::{LinearLayer, RouterLayer};
    let stopped = AtomicBool::new(false);
    let make = |a: f32, b: f32| MlpRouter {
        layers: vec![
            RouterLayer::Linear(LinearLayer {
                in_features: 1,
                out_features: 1,
                weights: vec![0.0],
                bias: vec![0.0],
            }),
            RouterLayer::ReLU,
            RouterLayer::Linear(LinearLayer {
                in_features: 1,
                out_features: 2,
                weights: vec![0.0, 0.0],
                bias: vec![a, b],
            }),
        ],
    };
    for (a, b, expect_fallback) in [
        (0.0, 0.0, true),
        (0.0, 1e-8, true),
        (10.0, 10.0, true),
        (-10.0, -10.0, true),
        (1.0, 1.0 + f32::EPSILON, true),
        (1.0, 1.001, false),
        (-10.0, -9.0, false),
        (f32::MAX, -f32::MAX, true),
    ] {
        let router = make(a, b);
        let trained = TrainedRouter::from_native_for_test(router.clone()).unwrap();
        let mut native = BuildRouter::new(&router, &stopped).unwrap();
        for rows in [1, 3] {
            let data = vec![1.0; rows];
            let (actual, fallbacks) = trained
                .top_buckets_tie_safe(&data, &stopped, &mut native)
                .unwrap();
            let expected: Vec<_> = data
                .iter()
                .map(|_| native.top_bucket(&[1.0], &stopped).unwrap())
                .collect();
            assert_eq!(actual, expected, "scores {a:?}, {b:?}");
            assert_eq!(fallbacks, if expect_fallback { rows } else { 0 });
        }
    }
    let router = make(0.0, 0.0);
    let trained = TrainedRouter::from_native_for_test(router.clone()).unwrap();
    let mut native = BuildRouter::new(&router, &stopped).unwrap();
    stopped.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(
        trained
            .top_buckets_tie_safe(&[1.0], &stopped, &mut native)
            .is_err()
    );
    stopped.store(false, std::sync::atomic::Ordering::Relaxed);
    assert!(
        trained
            .top_buckets_tie_safe(&[f32::NAN], &stopped, &mut native)
            .is_err()
    );
    assert!(
        trained
            .top_buckets_tie_safe(&[1.0, 2.0], &stopped, &mut native)
            .is_ok()
    );
    let router = MlpRouter {
        layers: vec![
            RouterLayer::Linear(LinearLayer {
                in_features: 1,
                out_features: 1,
                weights: vec![f32::MAX],
                bias: vec![0.0],
            }),
            RouterLayer::ReLU,
            RouterLayer::Linear(LinearLayer {
                in_features: 1,
                out_features: 2,
                weights: vec![1.0, 1.0],
                bias: vec![0.0, 0.0],
            }),
        ],
    };
    let trained = TrainedRouter::from_native_for_test(router.clone()).unwrap();
    let mut native = BuildRouter::new(&router, &stopped).unwrap();
    assert!(
        trained
            .top_buckets_tie_safe(&[2.0], &stopped, &mut native)
            .is_err()
    );
}

#[test]
#[ignore = "Explicit single-CPU B=10k tie-safe routing regression"]
fn phase_s3b2a_large_router_microbenchmark() {
    use super::evaluation_s3b1::{inputs, model};
    use std::time::Instant;
    let output = std::env::var("LMI_S3B2A_MICRO_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output).exists());
    tch::set_num_threads(1);
    let (d, h, b, rows) = (768, 512, 10_000, 2048);
    let router = model(d, h, b);
    let trained = TrainedRouter::from_native_for_test(router.clone()).unwrap();
    let data = inputs(rows, d);
    let stopped = AtomicBool::new(false);
    let mut native = BuildRouter::new(&router, &stopped).unwrap();
    let started = Instant::now();
    let expected: Vec<_> = data
        .chunks_exact(d)
        .map(|row| native.top_bucket(row, &stopped).unwrap())
        .collect();
    let scalar_seconds = started.elapsed().as_secs_f64();
    let mut records = Vec::new();
    for k in [64usize, 128, 256] {
        for backend in ["tch_top1", "tch_tie_safe"] {
            let started = Instant::now();
            let mut got = Vec::with_capacity(rows);
            let mut fallback_count = 0usize;
            for chunk in data.chunks(k * d) {
                if backend == "tch_top1" {
                    got.extend(trained.top_buckets(chunk, &stopped).unwrap());
                } else {
                    let (ids, fallbacks) = trained
                        .top_buckets_tie_safe(chunk, &stopped, &mut native)
                        .unwrap();
                    got.extend(ids);
                    fallback_count += fallbacks;
                }
            }
            let seconds = started.elapsed().as_secs_f64();
            let mismatches = expected.iter().zip(&got).filter(|(a, b)| a != b).count();
            records.push(json!({"d":d,"h":h,"b":b,"rows":rows,"k":k,"backend":backend,
                "seconds":seconds,"rows_per_second":rows as f64/seconds,
                "microseconds_per_vector":seconds*1e6/rows as f64,
                "fallback_count":fallback_count,"fallback_fraction":fallback_count as f64/rows as f64,
                "mismatches":mismatches,"scalar_seconds":scalar_seconds,
                "scalar_rows_per_second":rows as f64/scalar_seconds,
                "top2_output_bytes_per_batch":24*k}));
        }
    }
    std::fs::write(output, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
    eprintln!("S3B2A_MICRO {records:?}");
}
