//! Query-only controlled latency panel on the accepted persisted SISAP10M state.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector};
use segment::index::VectorIndexEnum;
use segment::index::candidate_scoring::score_candidates;
use segment::segment_constructor::load_segment;
use segment::types::Distance;
use sha2::{Digest, Sha256};
use uuid::Uuid;

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
            let mut f = File::open(path.join(name)).unwrap();
            let mut hash = Sha256::new();
            let mut buf = vec![0_u8; 1024 * 1024];
            loop {
                let n = f.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                hash.update(&buf[..n]);
            }
            hash.finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .collect()
}

fn vector(bytes: &[u8], query_row: usize) -> Vec<f32> {
    let start = (query_row - 20) * 768 * 4;
    bytes[start..start + 768 * 4]
        .chunks_exact(4)
        .map(|part| f32::from_le_bytes(part.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "explicit controlled SISAP10M G2 native latency panel"]
fn g2_controlled_latency_panel() {
    let root = PathBuf::from(ROOT).join("g2_119");
    let plan: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("latency_panel_design.json")).unwrap())
            .unwrap();
    assert_eq!(plan["seed"].as_u64(), Some(20261009));
    let rows: Vec<usize> = plan["query_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(rows.len(), 1000);
    let warm: Vec<usize> = plan["warm_prefix_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let orders: Vec<Vec<usize>> = plan["probe_orders_by_repetition"]
        .as_array()
        .unwrap()
        .iter()
        .map(|run| {
            run.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    assert_eq!(orders.len(), 3);
    let selected: Option<HashSet<usize>> = std::env::var("LMI_G2_PANEL_PROBES").ok().map(|value| {
        value
            .split(",")
            .map(|item| item.parse::<usize>().unwrap())
            .collect()
    });
    if let Some(selected) = &selected {
        assert!(!selected.is_empty());
        assert!(selected.iter().all(|p| orders[0].contains(p)));
    }
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
    assert_eq!(state.postings().point_count(), 10_120_191);
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let mut queries = Vec::new();
    File::open(Path::new(ROOT).join("pre_g2_119_attempt2/full_queries/queries_20_9999.f32"))
        .unwrap()
        .read_to_end(&mut queries)
        .unwrap();
    assert_eq!(queries.len(), 9_980 * 768 * 4);
    let ctx = VectorQueryContext::default();
    let output = root.join("latency_panel");
    std::fs::create_dir_all(&output).unwrap();

    for (rep_idx, order) in orders.iter().enumerate() {
        assert_eq!(order.len(), 15);
        for &p in order {
            if selected.as_ref().is_some_and(|set| !set.contains(&p)) {
                continue;
            }
            let before = hashes(state_dir);
            assert_eq!(
                before, EXPECTED,
                "state hash changed before panel condition"
            );
            let condition = output.join(format!("rep{}_p{p:04}", rep_idx + 1));
            if condition.exists() {
                let prior: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(condition.join("summary.json"))
                        .expect("partial panel condition exists; preserve it for inspection"),
                )
                .unwrap();
                assert_eq!(prior["nprobe"].as_u64(), Some(p as u64));
                assert_eq!(prior["repetition"].as_u64(), Some((rep_idx + 1) as u64));
                assert_eq!(prior["query_count"].as_u64(), Some(rows.len() as u64));
                let prior_before: Vec<String> = serde_json::from_slice(
                    &std::fs::read(condition.join("hashes_before.json")).unwrap(),
                )
                .unwrap();
                let prior_after: Vec<String> = serde_json::from_slice(
                    &std::fs::read(condition.join("hashes_after.json")).unwrap(),
                )
                .unwrap();
                assert_eq!(prior_before, EXPECTED);
                assert_eq!(prior_after, EXPECTED);
                assert_eq!(
                    BufReader::new(File::open(condition.join("rows.jsonl")).unwrap())
                        .lines()
                        .count(),
                    rows.len()
                );
                println!(
                    "G2 latency rep={} P={} already complete; verified and skipped",
                    rep_idx + 1,
                    p
                );
                continue;
            }
            std::fs::create_dir(&condition).unwrap();
            std::fs::write(
                condition.join("hashes_before.json"),
                serde_json::to_vec_pretty(&before).unwrap(),
            )
            .unwrap();
            for &row in &warm {
                let values = vector(&queries, row);
                let normalized = Distance::Cosine.preprocess_vector::<f32>(values.clone());
                let buckets = state.router().top_buckets(&normalized, p).unwrap();
                let mut candidates = Vec::new();
                for bucket in buckets {
                    candidates.extend_from_slice(state.postings().get(bucket).unwrap());
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
            let mut writer = BufWriter::new(File::create(condition.join("rows.jsonl")).unwrap());
            let mut latencies = Vec::with_capacity(rows.len());
            for &row in &rows {
                let values = vector(&queries, row);
                let normalized = Distance::Cosine.preprocess_vector::<f32>(values.clone());
                let started = Instant::now();
                let buckets = state.router().top_buckets(&normalized, p).unwrap();
                let mut candidates = Vec::new();
                for bucket in buckets {
                    candidates.extend_from_slice(state.postings().get(bucket).unwrap());
                }
                candidates.sort_unstable();
                candidates.dedup();
                let candidate_count = candidates.len();
                let raw: QueryVector = values.into();
                let hits =
                    score_candidates(&raw, &candidates, 10, &*tracker, &*storage, &ctx).unwrap();
                let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(hits.len(), 10);
                latencies.push(latency_ms);
                serde_json::to_writer(
                    &mut writer,
                    &serde_json::json!({
                        "query_row": row,
                        "nprobe": p,
                        "repetition": rep_idx + 1,
                        "candidate_count": candidate_count,
                        "native_search_ms": latency_ms,
                    }),
                )
                .unwrap();
                writer.write_all(b"\n").unwrap();
            }
            writer.flush().unwrap();
            let after = hashes(state_dir);
            assert_eq!(after, EXPECTED, "state hash changed after panel condition");
            std::fs::write(
                condition.join("hashes_after.json"),
                serde_json::to_vec_pretty(&after).unwrap(),
            )
            .unwrap();
            let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
            std::fs::write(
                condition.join("summary.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "nprobe": p,
                    "repetition": rep_idx + 1,
                    "query_count": rows.len(),
                    "mean_native_search_ms": mean,
                    "state_hashes_unchanged": true,
                }))
                .unwrap(),
            )
            .unwrap();
            println!("G2 latency rep={} P={} mean_ms={:.3}", rep_idx + 1, p, mean);
        }
    }
}
