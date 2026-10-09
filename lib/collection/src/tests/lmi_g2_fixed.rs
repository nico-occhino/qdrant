//! Opt-in, query-only G2 sweep over the accepted persisted SISAP10M v3 state.
//! Every output is outside the collection. No optimizer, trainer or state writer is opened.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::types::DeferredBehavior;
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
const STORAGE: &str = "/home/nicoo/work/lmi-sisap2023-storage/10m-f16-qdrant-119-sync-attempt2";
const EXPECTED: [&str; 3] = [
    "09c88623def82515b50ddf4f1c85d82a1188e5a29bafbc4fb00ca729dbf8d81f",
    "6789b1aace656c3c475675041433ccdb97bd13190d15edc8ff2b7c0bf8708a94",
    "adcf9f104ef60899e9b922e0bbb155061af3a61532fdcf11705ed55340a7eade",
];

fn state_hashes(path: &Path) -> Vec<String> {
    ["lmi_state.json", "lmi_router.bin", "lmi_postings.bin"]
        .iter()
        .map(|name| {
            let mut file = File::open(path.join(name)).unwrap();
            let mut hash = Sha256::new();
            let mut buffer = vec![0_u8; 8 * 1024 * 1024];
            loop {
                let n = file.read(&mut buffer).unwrap();
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            hash.finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .collect()
}

fn query(bytes: &[u8], row: usize) -> Vec<f32> {
    bytes[row * 768 * 4..(row + 1) * 768 * 4]
        .chunks_exact(4)
        .map(|part| f32::from_le_bytes(part.try_into().unwrap()))
        .collect()
}

fn gold_ids(bytes: &[u8], row: usize) -> Vec<u64> {
    bytes[row * 40..(row + 1) * 40]
        .chunks_exact(4)
        .map(|part| i32::from_le_bytes(part.try_into().unwrap()) as u64)
        .collect()
}

#[test]
#[ignore = "explicit query-only full SISAP10M G2 point"]
fn g2_fixed_persisted_state() {
    let p: usize = std::env::var("LMI_G2_P").unwrap().parse().unwrap();
    assert!((1..=B).contains(&p));
    let out = PathBuf::from(ROOT)
        .join("g2_119/fixed")
        .join(format!("p{p:04}"));
    assert!(!out.exists(), "refuse to overwrite a G2 condition");

    let baseline = PathBuf::from(ROOT).join("pre_g2_119_attempt2");
    let state_dir = PathBuf::from(STORAGE)
        .join("collection/0/segments/168c6d29-9024-404e-aff5-184913f584e7/vector_index");
    let before = state_hashes(&state_dir);
    assert_eq!(
        before, EXPECTED,
        "accepted state hash mismatch before P={p}"
    );
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join("hashes_before.json"),
        serde_json::to_vec_pretty(&before).unwrap(),
    )
    .unwrap();

    let segment_path = state_dir.parent().unwrap();
    let uuid = Uuid::parse_str(segment_path.file_name().unwrap().to_str().unwrap()).unwrap();
    let segment = load_segment(segment_path, uuid, None, &AtomicBool::new(false), true).unwrap();
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(index) = &*index else {
        panic!("accepted target did not open as LMI");
    };
    let state = index.routing_state().unwrap();
    assert_eq!(state.postings().point_count(), N);
    assert_eq!(state.postings().len(), B);
    let tracker = segment.id_tracker.borrow();
    assert_eq!(tracker.available_point_count(), N);
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();

    let mut bucket_of_offset = vec![u16::MAX; N];
    for (bucket, posting) in state.postings().iter().enumerate() {
        for &offset in posting {
            let slot = &mut bucket_of_offset[offset as usize];
            assert_eq!(*slot, u16::MAX, "duplicate posting offset");
            *slot = bucket as u16;
        }
    }
    assert!(
        !bucket_of_offset.contains(&u16::MAX),
        "missing posting offset"
    );

    let full = baseline.join("full_queries");
    let mut measured_queries = Vec::new();
    File::open(full.join("queries_20_9999.f32"))
        .unwrap()
        .read_to_end(&mut measured_queries)
        .unwrap();
    let mut measured_gold = Vec::new();
    File::open(full.join("gold_20_9999_top10.i32"))
        .unwrap()
        .read_to_end(&mut measured_gold)
        .unwrap();
    let mut warm_queries = Vec::new();
    File::open(full.join("queries_0_19.f32"))
        .unwrap()
        .read_to_end(&mut warm_queries)
        .unwrap();
    assert_eq!(measured_queries.len(), 9_980 * 768 * 4);
    assert_eq!(measured_gold.len(), 9_980 * 40);
    assert_eq!(warm_queries.len(), 20 * 768 * 4);

    let accepted: Vec<serde_json::Value> = if p == 4 {
        BufReader::new(File::open(baseline.join("queries.full.jsonl")).unwrap())
            .lines()
            .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
            .collect()
    } else {
        Vec::new()
    };
    let ctx = VectorQueryContext::default();
    for row in 0..20 {
        let vector = query(&warm_queries, row);
        let normalized = Distance::Cosine.preprocess_vector::<f32>(vector.clone());
        let ranked = state.router().top_buckets(&normalized, p).unwrap();
        let mut candidates = Vec::new();
        for bucket in ranked {
            candidates.extend_from_slice(state.postings().get(bucket).unwrap());
        }
        candidates.sort_unstable();
        candidates.dedup();
        let raw: QueryVector = vector.into();
        let hits = score_candidates(&raw, &candidates, 10, &*tracker, &*storage, &ctx).unwrap();
        assert_eq!(hits.len(), 10);
    }

    let started = Instant::now();
    let mut writer = BufWriter::new(File::create(out.join("rows.jsonl")).unwrap());
    let mut recall_sum = 0.0;
    let mut candidate_sum = 0_u64;
    let mut coverage_sum = 0.0;
    let mut oracle_sum = 0.0;
    for row in 0..9_980 {
        let vector = query(&measured_queries, row);
        let gold = gold_ids(&measured_gold, row);
        let gold_offsets: Vec<_> = gold
            .iter()
            .map(|&id| {
                tracker
                    .internal_id_with_behavior(
                        PointIdType::NumId(id),
                        DeferredBehavior::VisibleOnly,
                    )
                    .expect("official gold ID must map to an internal offset")
            })
            .collect();
        let gold_buckets: Vec<_> = gold_offsets
            .iter()
            .map(|&offset| bucket_of_offset[offset as usize] as usize)
            .collect();
        let normalized = Distance::Cosine.preprocess_vector::<f32>(vector.clone());
        let timer = Instant::now();
        let ranked = state.router().top_buckets(&normalized, B).unwrap();
        let mut rank_of_bucket = vec![0_u16; B];
        for (rank, &bucket) in ranked.iter().enumerate() {
            rank_of_bucket[bucket] = (rank + 1) as u16;
        }
        let gold_ranks: Vec<_> = gold_buckets
            .iter()
            .map(|&bucket| rank_of_bucket[bucket])
            .collect();
        let selected: HashSet<usize> = ranked[..p].iter().copied().collect();
        let candidate_gold_count = gold_buckets
            .iter()
            .filter(|&&bucket| selected.contains(&bucket))
            .count();
        let mut multiplicities = HashMap::<usize, usize>::new();
        for &bucket in &gold_buckets {
            *multiplicities.entry(bucket).or_default() += 1;
        }
        let mut counts: Vec<usize> = multiplicities.values().copied().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let oracle = counts.iter().take(p).sum::<usize>() as f64 / 10.0;
        let mut candidates = Vec::new();
        for &bucket in &ranked[..p] {
            candidates.extend_from_slice(state.postings().get(bucket).unwrap());
        }
        candidates.sort_unstable();
        candidates.dedup();
        let candidate_count = candidates.len();
        let raw: QueryVector = vector.into();
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
        let matched = returned_ids.iter().filter(|id| gold.contains(id)).count();
        assert!(matched <= candidate_gold_count);
        if p == 4 {
            let prior = &accepted[row];
            assert_eq!(prior["query_row"].as_u64(), Some((row + 20) as u64));
            assert_eq!(
                prior["candidate_count"].as_f64(),
                Some(candidate_count as f64)
            );
            let prior_ids: HashSet<u64> = prior["returned_ids_1_based"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(
                returned_ids.iter().copied().collect::<HashSet<_>>(),
                prior_ids
            );
        }
        recall_sum += matched as f64 / 10.0;
        candidate_sum += candidate_count as u64;
        coverage_sum += candidate_gold_count as f64 / 10.0;
        oracle_sum += oracle;
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "query_row": row + 20,
                "nprobe": p,
                "recall_at_10": matched as f64 / 10.0,
                "candidate_count": candidate_count,
                "candidate_gold_coverage": candidate_gold_count as f64 / 10.0,
                "oracle_coverage": oracle,
                "latency_ms": latency_ms,
                "gold_bucket_ranks_1_based": gold_ranks,
                "distinct_gold_buckets": multiplicities.len(),
                "returned_ids_1_based": returned_ids,
            }),
        )
        .unwrap();
        writer.write_all(b"\n").unwrap();
        if (row + 1) % 100 == 0 {
            writer.flush().unwrap();
        }
        if (row + 1) % 1_000 == 0 {
            println!(
                "G2 P={p} queries={} elapsed_s={:.1}",
                row + 1,
                started.elapsed().as_secs_f64()
            );
        }
    }
    writer.flush().unwrap();
    let after = state_hashes(&state_dir);
    assert_eq!(after, EXPECTED, "accepted state hash mismatch after P={p}");
    std::fs::write(
        out.join("hashes_after.json"),
        serde_json::to_vec_pretty(&after).unwrap(),
    )
    .unwrap();
    let summary = serde_json::json!({
        "nprobe": p,
        "query_count": 9_980,
        "mean_recall": recall_sum / 9_980.0,
        "mean_candidates": candidate_sum as f64 / 9_980.0,
        "mean_candidate_fraction": candidate_sum as f64 / 9_980.0 / N as f64,
        "mean_candidate_gold_coverage": coverage_sum / 9_980.0,
        "mean_oracle_coverage": oracle_sum / 9_980.0,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "state_hashes_unchanged": true,
    });
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    println!("G2 P={p} completed: {summary}");
}

#[test]
#[ignore = "explicit persisted-state score diagnostic for duplicated SISAP gold"]
fn g2_raw_scorer_tie_diagnostic() {
    let root = PathBuf::from(ROOT).join("g2_119");
    let output = root.join("raw_scorer_tie_diagnostic.json");
    assert!(!output.exists(), "refuse to overwrite score diagnostic");
    let state_dir = PathBuf::from(STORAGE)
        .join("collection/0/segments/168c6d29-9024-404e-aff5-184913f584e7/vector_index");
    assert_eq!(state_hashes(&state_dir), EXPECTED);
    let segment_path = state_dir.parent().unwrap();
    let uuid = Uuid::parse_str(segment_path.file_name().unwrap().to_str().unwrap()).unwrap();
    let segment = load_segment(segment_path, uuid, None, &AtomicBool::new(false), true).unwrap();
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();

    let baseline = PathBuf::from(ROOT).join("pre_g2_119_attempt2/full_queries");
    let mut query_bytes = Vec::new();
    File::open(baseline.join("queries_20_9999.f32"))
        .unwrap()
        .read_to_end(&mut query_bytes)
        .unwrap();
    let mut gold_bytes = Vec::new();
    File::open(baseline.join("gold_20_9999_top10.i32"))
        .unwrap()
        .read_to_end(&mut gold_bytes)
        .unwrap();
    let query_row = 1269_usize;
    let measured_index = query_row - 20;
    let gold = gold_ids(&gold_bytes, measured_index);
    let result: serde_json::Value =
        BufReader::new(File::open(root.join("fixed/p0160/rows.jsonl")).unwrap())
            .lines()
            .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
            .find(|row: &serde_json::Value| row["query_row"].as_u64() == Some(query_row as u64))
            .unwrap();
    assert_eq!(result["recall_at_10"].as_f64(), Some(0.0));
    assert_eq!(result["candidate_gold_coverage"].as_f64(), Some(1.0));
    let returned: Vec<u64> = result["returned_ids_1_based"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap())
        .collect();
    let ids: Vec<u64> = gold
        .iter()
        .copied()
        .chain(returned.iter().copied())
        .collect();
    assert_eq!(ids.iter().copied().collect::<HashSet<_>>().len(), 20);
    let offsets: Vec<_> = ids
        .iter()
        .map(|&id| {
            tracker
                .internal_id_with_behavior(PointIdType::NumId(id), DeferredBehavior::VisibleOnly)
                .unwrap()
        })
        .collect();
    let raw: QueryVector = query(&query_bytes, measured_index).into();
    let hits = score_candidates(
        &raw,
        &offsets,
        20,
        &*tracker,
        &*storage,
        &VectorQueryContext::default(),
    )
    .unwrap();
    assert_eq!(hits.len(), 20);
    let scored: Vec<_> = hits
        .iter()
        .map(|hit| {
            let PointIdType::NumId(id) = tracker.external_id(hit.idx).unwrap() else {
                panic!("non-numeric SISAP ID");
            };
            serde_json::json!({
                "id_1_based": id,
                "score": hit.score,
                "is_official_gold": gold.contains(&id),
                "is_returned_at_p160": returned.contains(&id),
            })
        })
        .collect();
    let max_gold = hits
        .iter()
        .filter(|hit| matches!(tracker.external_id(hit.idx).unwrap(), PointIdType::NumId(id) if gold.contains(&id)))
        .map(|hit| hit.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_returned = hits
        .iter()
        .filter(|hit| matches!(tracker.external_id(hit.idx).unwrap(), PointIdType::NumId(id) if returned.contains(&id)))
        .map(|hit| hit.score)
        .fold(f32::INFINITY, f32::min);
    assert_eq!(state_hashes(&state_dir), EXPECTED);
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&serde_json::json!({
            "query_row": query_row,
            "official_gold_ids_1_based": gold,
            "p160_returned_ids_1_based": returned,
            "max_gold_score": max_gold,
            "min_returned_score": min_returned,
            "returned_outscores_gold": min_returned >= max_gold,
            "score_gap_returned_min_minus_gold_max": min_returned - max_gold,
            "scores_descending": scored,
            "state_hashes_unchanged": true,
        }))
        .unwrap(),
    )
    .unwrap();
    println!(
        "G2 persisted scorer tie diagnostic saved: {}",
        output.display()
    );
}
