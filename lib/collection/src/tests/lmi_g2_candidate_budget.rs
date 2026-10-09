//! Query-only candidate-budget extension using the same v3 router and postings.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector};
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexEnum;
use segment::index::candidate_scoring::score_candidates;
use segment::segment_constructor::load_segment;
use segment::types::{Distance, PointIdType};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const N: usize = 10_120_191;
const B: usize = 3_162;
const ROOT: &str = "/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap10m";
const STATE: &str = "/home/nicoo/work/lmi-sisap2023-storage/10m-f16-qdrant-119-sync-attempt2/collection/0/segments/168c6d29-9024-404e-aff5-184913f584e7/vector_index";
const EXPECTED: [&str; 3] = [
    "09c88623def82515b50ddf4f1c85d82a1188e5a29bafbc4fb00ca729dbf8d81f",
    "6789b1aace656c3c475675041433ccdb97bd13190d15edc8ff2b7c0bf8708a94",
    "adcf9f104ef60899e9b922e0bbb155061af3a61532fdcf11705ed55340a7eade",
];

fn hashes(path: &Path) -> Vec<String> {
    ["lmi_state.json", "lmi_router.bin", "lmi_postings.bin"]
        .iter()
        .map(|name| {
            let mut file = File::open(path.join(name)).unwrap();
            let mut digest = Sha256::new();
            let mut buffer = vec![0_u8; 1024 * 1024];
            loop {
                let n = file.read(&mut buffer).unwrap();
                if n == 0 {
                    break;
                }
                digest.update(&buffer[..n]);
            }
            digest
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .collect()
}

fn vector(bytes: &[u8], row: usize) -> Vec<f32> {
    let start = row * 768 * 4;
    bytes[start..start + 768 * 4]
        .chunks_exact(4)
        .map(|part| f32::from_le_bytes(part.try_into().unwrap()))
        .collect()
}

fn gold(bytes: &[u8], row: usize) -> Vec<u64> {
    bytes[row * 40..row * 40 + 40]
        .chunks_exact(4)
        .map(|part| i32::from_le_bytes(part.try_into().unwrap()) as u64)
        .collect()
}

fn selected_count(prefix: &[u32], threshold: u32) -> usize {
    (prefix.partition_point(|&count| count < threshold) + 1).min(prefix.len())
}

fn mean_work(prefixes: &[Vec<u32>], threshold: u32) -> f64 {
    prefixes
        .iter()
        .map(|prefix| prefix[selected_count(prefix, threshold) - 1] as f64)
        .sum::<f64>()
        / prefixes.len() as f64
}

fn calibrated_threshold(prefixes: &[Vec<u32>], target: f64) -> (u32, f64) {
    let mut low = 1_u32;
    let mut high = N as u32;
    while low < high {
        let mid = low + (high - low) / 2;
        if mean_work(prefixes, mid) < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let upper = (low, mean_work(prefixes, low));
    if low > 1 {
        let lower = (low - 1, mean_work(prefixes, low - 1));
        if (lower.1 - target).abs() < (upper.1 - target).abs() {
            return lower;
        }
    }
    upper
}

#[test]
#[ignore = "explicit query-only SISAP10M candidate-budget experiment"]
fn g2_candidate_budget_persisted_state() {
    let root = PathBuf::from(ROOT).join("g2_119");
    let result_root = root.join("candidate_budget");
    assert!(!result_root.exists(), "refuse to overwrite budget results");
    let state_dir = Path::new(STATE);
    assert_eq!(hashes(state_dir), EXPECTED);
    let segment_path = state_dir.parent().unwrap();
    let uuid = Uuid::parse_str(segment_path.file_name().unwrap().to_str().unwrap()).unwrap();
    let segment = load_segment(segment_path, uuid, None, &AtomicBool::new(false), true).unwrap();
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(index) = &*index else {
        panic!("accepted segment is not LMI");
    };
    let state = index.routing_state().unwrap();
    assert_eq!(state.postings().point_count(), N);
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let mut bucket_of_offset = vec![u16::MAX; N];
    let bucket_sizes: Vec<u32> = state
        .postings()
        .iter()
        .enumerate()
        .map(|(bucket, posting)| {
            for &offset in posting {
                let slot = &mut bucket_of_offset[offset as usize];
                assert_eq!(*slot, u16::MAX);
                *slot = bucket as u16;
            }
            posting.len() as u32
        })
        .collect();
    assert_eq!(bucket_sizes.len(), B);
    assert!(!bucket_of_offset.contains(&u16::MAX));

    let input = PathBuf::from(ROOT).join("pre_g2_119_attempt2/full_queries");
    let mut query_bytes = Vec::new();
    File::open(input.join("queries_20_9999.f32"))
        .unwrap()
        .read_to_end(&mut query_bytes)
        .unwrap();
    let mut gold_bytes = Vec::new();
    File::open(input.join("gold_20_9999_top10.i32"))
        .unwrap()
        .read_to_end(&mut gold_bytes)
        .unwrap();
    assert_eq!(query_bytes.len(), 9_980 * 768 * 4);
    assert_eq!(gold_bytes.len(), 9_980 * 40);
    let mut warm_query_bytes = Vec::new();
    File::open(input.join("queries_0_19.f32"))
        .unwrap()
        .read_to_end(&mut warm_query_bytes)
        .unwrap();
    assert_eq!(warm_query_bytes.len(), 20 * 768 * 4);

    // Rank once per query. Only the ordered bucket IDs and posting-size prefix are cached.
    let mut rankings = Vec::<Vec<u16>>::with_capacity(9_980);
    let mut prefixes = Vec::<Vec<u32>>::with_capacity(9_980);
    for row in 0..9_980 {
        let normalized = Distance::Cosine.preprocess_vector::<f32>(vector(&query_bytes, row));
        let rank = state.router().top_buckets(&normalized, B).unwrap();
        let mut cumulative = 0_u32;
        let mut prefix = Vec::with_capacity(B);
        for &bucket in &rank {
            cumulative += bucket_sizes[bucket];
            prefix.push(cumulative);
        }
        assert_eq!(cumulative as usize, N);
        rankings.push(rank.into_iter().map(|bucket| bucket as u16).collect());
        prefixes.push(prefix);
    }

    let anchor_p = [4, 13, 32, 64, 128];
    let mut plans = Vec::new();
    for p in anchor_p {
        let fixed: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("fixed/p{p:04}/summary.json"))).unwrap(),
        )
        .unwrap();
        let target = fixed["mean_candidates"].as_f64().unwrap();
        let (threshold, matched_mean) = calibrated_threshold(&prefixes, target);
        plans.push((p, target, threshold, matched_mean));
    }
    std::fs::create_dir_all(&result_root).unwrap();
    std::fs::write(
        result_root.join("calibration.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "target_fixed_p": anchor_p,
            "plans": plans.iter().map(|(p, target, threshold, matched)| serde_json::json!({
                "fixed_p": p,
                "fixed_mean_candidates": target,
                "budget_threshold": threshold,
                "predicted_mean_candidates": matched,
            })).collect::<Vec<_>>(),
            "calibration_queries": 9_980,
        }))
        .unwrap(),
    )
    .unwrap();
    let ctx = VectorQueryContext::default();

    for (fixed_p, target, threshold, matched_mean) in plans {
        let before = hashes(state_dir);
        assert_eq!(before, EXPECTED);
        let out = result_root.join(format!("target_p{fixed_p:04}"));
        std::fs::create_dir(&out).unwrap();
        std::fs::write(
            out.join("hashes_before.json"),
            serde_json::to_vec_pretty(&before).unwrap(),
        )
        .unwrap();
        let mut writer = BufWriter::new(File::create(out.join("rows.jsonl")).unwrap());
        let mut recall_sum = 0.0;
        let mut candidate_sum = 0_u64;
        let mut coverage_sum = 0.0;
        let started = Instant::now();
        for row in 0..20 {
            let values = vector(&warm_query_bytes, row);
            let normalized = Distance::Cosine.preprocess_vector::<f32>(values.clone());
            let rank = state.router().top_buckets(&normalized, B).unwrap();
            let mut cumulative = 0_u32;
            let mut candidates = Vec::new();
            for bucket in rank {
                cumulative += bucket_sizes[bucket];
                candidates.extend_from_slice(state.postings().get(bucket).unwrap());
                if cumulative >= threshold {
                    break;
                }
            }
            candidates.sort_unstable();
            candidates.dedup();
            let raw: QueryVector = values.into();
            assert_eq!(
                score_candidates(&raw, &candidates, 10, &*tracker, &*storage, &ctx)
                    .unwrap()
                    .len(),
                10
            );
        }
        for row in 0..9_980 {
            let ids = gold(&gold_bytes, row);
            let gold_buckets: Vec<usize> = ids
                .iter()
                .map(|&id| {
                    let offset = tracker
                        .internal_id_with_behavior(
                            PointIdType::NumId(id),
                            common::types::DeferredBehavior::VisibleOnly,
                        )
                        .unwrap();
                    bucket_of_offset[offset as usize] as usize
                })
                .collect();
            let values = vector(&query_bytes, row);
            let timer = Instant::now();
            let normalized = Distance::Cosine.preprocess_vector::<f32>(values.clone());
            let reranked = state.router().top_buckets(&normalized, B).unwrap();
            assert!(
                reranked
                    .iter()
                    .zip(&rankings[row])
                    .all(|(&a, &b)| a == b as usize)
            );
            let count = selected_count(&prefixes[row], threshold);
            let mut candidates = Vec::with_capacity(prefixes[row][count - 1] as usize);
            for &bucket in &rankings[row][..count] {
                candidates.extend_from_slice(state.postings().get(bucket as usize).unwrap());
            }
            candidates.sort_unstable();
            candidates.dedup();
            assert_eq!(candidates.len(), prefixes[row][count - 1] as usize);
            let raw: QueryVector = values.into();
            let hits = score_candidates(&raw, &candidates, 10, &*tracker, &*storage, &ctx).unwrap();
            let latency_ms = timer.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(hits.len(), 10);
            let returned_ids: Vec<u64> = hits
                .iter()
                .map(|hit| match tracker.external_id(hit.idx).unwrap() {
                    PointIdType::NumId(id) => id,
                    _ => panic!("non-numeric SISAP ID"),
                })
                .collect();
            let matched = returned_ids.iter().filter(|id| ids.contains(id)).count();
            let selected: std::collections::HashSet<usize> = rankings[row][..count]
                .iter()
                .map(|&bucket| bucket as usize)
                .collect();
            let covered = gold_buckets
                .iter()
                .filter(|&&bucket| selected.contains(&bucket))
                .count();
            assert!(matched <= covered);
            recall_sum += matched as f64 / 10.0;
            candidate_sum += candidates.len() as u64;
            coverage_sum += covered as f64 / 10.0;
            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "query_row": row + 20,
                    "target_fixed_p": fixed_p,
                    "budget_threshold": threshold,
                    "probes_used": count,
                    "candidate_count": candidates.len(),
                    "recall_at_10": matched as f64 / 10.0,
                    "candidate_gold_coverage": covered as f64 / 10.0,
                    "native_latency_ms": latency_ms,
                    "returned_ids_1_based": returned_ids,
                }),
            )
            .unwrap();
            writer.write_all(b"\n").unwrap();
            if (row + 1) % 100 == 0 {
                writer.flush().unwrap();
            }
            if (row + 1) % 1000 == 0 {
                println!(
                    "G2 budget target P={fixed_p} queries={} elapsed_s={:.1}",
                    row + 1,
                    started.elapsed().as_secs_f64()
                );
            }
        }
        writer.flush().unwrap();
        let after = hashes(state_dir);
        assert_eq!(after, EXPECTED);
        std::fs::write(
            out.join("hashes_after.json"),
            serde_json::to_vec_pretty(&after).unwrap(),
        )
        .unwrap();
        let summary = serde_json::json!({
            "target_fixed_p": fixed_p,
            "fixed_mean_candidates": target,
            "budget_threshold": threshold,
            "calibrated_mean_candidates": matched_mean,
            "query_count": 9_980,
            "mean_recall": recall_sum / 9_980.0,
            "mean_candidates": candidate_sum as f64 / 9_980.0,
            "mean_candidate_fraction": candidate_sum as f64 / 9_980.0 / N as f64,
            "mean_candidate_gold_coverage": coverage_sum / 9_980.0,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "state_hashes_unchanged": true,
        });
        std::fs::write(
            out.join("summary.json"),
            serde_json::to_vec_pretty(&summary).unwrap(),
        )
        .unwrap();
        println!("G2 budget target P={fixed_p} completed: {summary}");
    }
}
