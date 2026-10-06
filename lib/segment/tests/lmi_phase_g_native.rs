//! Read-only native Qdrant scoring across P=1,2,4,8,16,32 for fixed LMI state.
//! Explicit env-controlled integration test; no trained state or production dispatch changes.
use common::types::DeferredBehavior;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector};
use segment::id_tracker::IdTrackerRead;
use segment::index::hnsw_index::point_scorer::BatchFilteredSearcher;
use segment::index::lmi_index::{CompactPostings, LMI_POSTINGS_FILE, LMI_ROUTER_FILE, MlpRouter};
use segment::segment_constructor::load_segment;
use segment::types::Distance;
use segment::vector_storage::VectorStorageRead;
use segment::vector_storage::quantized::quantized_vectors::QuantizedVectors;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

#[test]
#[ignore = "large read-only SISAP nprobe sweep"]
fn native_fixed_state_nprobe_sweep() {
    let dir = PathBuf::from(std::env::var("LMI_G_SEGMENT").expect("LMI_G_SEGMENT"));
    let gold_path = PathBuf::from(std::env::var("LMI_G_GOLD_EXPORT").expect("LMI_G_GOLD_EXPORT"));
    let query_path = PathBuf::from(std::env::var("LMI_G_QUERY_F32").expect("LMI_G_QUERY_F32"));
    let output = PathBuf::from(std::env::var("LMI_G_SWEEP_OUTPUT").expect("LMI_G_SWEEP_OUTPUT"));
    let limit = std::env::var("LMI_G_LIMIT")
        .ok()
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(9980);
    assert!(limit > 0 && limit <= 9980);
    let segment = load_segment(&dir, uuid::Uuid::nil(), None, &AtomicBool::new(false)).unwrap();
    let config = segment
        .segment_config
        .vector_data
        .get(DEFAULT_VECTOR_NAME)
        .unwrap();
    assert_eq!(config.distance, Distance::Cosine);
    let (_sample, router): (Vec<u32>, Option<MlpRouter>) =
        common::fs::read_bin(&dir.join("vector_index").join(LMI_ROUTER_FILE)).unwrap();
    let router = router.expect("persisted router");
    let postings: CompactPostings =
        common::fs::read_bin(&dir.join("vector_index").join(LMI_POSTINGS_FILE)).unwrap();
    let query_bytes = std::fs::read(query_path).unwrap();
    assert_eq!(query_bytes.len(), 9980 * 768 * 4);
    let mapped = BufReader::new(File::open(gold_path).unwrap());
    let root = QueryContext::default();
    let segment_context = root.get_segment_query_context();
    let context = segment_context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let tracker = segment.id_tracker.borrow();
    assert_eq!(storage.total_vector_count(), 10120191);
    let deleted = context
        .deleted_points()
        .unwrap_or_else(|| tracker.deleted_point_bitslice());
    let mut writer = BufWriter::new(File::create(output).unwrap());
    let mut parity_failures = 0usize;
    let started = Instant::now();
    for (i, line) in mapped.lines().take(limit).enumerate() {
        let gold: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let row = gold["query_row"].as_u64().unwrap() as usize;
        assert_eq!(row, i + 20);
        let bytes = &query_bytes[i * 768 * 4..(i + 1) * 768 * 4];
        let vector = bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        let query = QueryVector::from(vector.clone());
        let normalized = Distance::Cosine.preprocess_vector::<f32>(vector);
        let gold_offsets = gold["gold_offsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<std::collections::HashSet<_>>();
        let expected = gold["result_offsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        for p in [1usize, 2, 4, 8, 16, 32] {
            let t = Instant::now();
            let buckets = router.top_buckets(&normalized, p).unwrap();
            let router_ns = t.elapsed().as_nanos() as u64;
            let t = Instant::now();
            let mut candidates = Vec::new();
            for bucket in buckets {
                candidates.extend_from_slice(postings.get(bucket).unwrap());
            }
            candidates.sort_unstable();
            candidates.dedup();
            let posting_ns = t.elapsed().as_nanos() as u64;
            let candidate_count = candidates.len();
            let t = Instant::now();
            let searcher = BatchFilteredSearcher::new(
                &[&query],
                &*storage,
                None::<&QuantizedVectors>,
                None,
                10,
                deleted,
                context.hardware_counter(),
            )
            .unwrap();
            let ids = tracker
                .point_mappings()
                .filter_deferred_and_deleted(candidates.into_iter(), DeferredBehavior::VisibleOnly);
            let result = searcher
                .peek_top_iter(ids, &context.is_stopped())
                .unwrap()
                .remove(0);
            let score_ns = t.elapsed().as_nanos() as u64;
            let offsets = result.iter().map(|x| x.idx).collect::<Vec<_>>();
            let recall = offsets.iter().filter(|x| gold_offsets.contains(x)).count() as f64 / 10.0;
            if p == 4
                && offsets
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    != expected
                        .iter()
                        .copied()
                        .collect::<std::collections::HashSet<_>>()
            {
                parity_failures += 1;
            }
            serde_json::to_writer(&mut writer,&serde_json::json!({"query_row":row,"nprobe":p,"candidate_count":candidate_count,"recall10":recall,"result_offsets":offsets,"router_ns":router_ns,"posting_gather_ns":posting_ns,"scoring_topk_ns":score_ns})).unwrap();
            writer.write_all(b"\n").unwrap();
        }
        if (i + 1) % 500 == 0 {
            writer.flush().unwrap();
            println!(
                "Phase G native sweep: {} / {} queries, elapsed {:.1}s",
                i + 1,
                limit,
                started.elapsed().as_secs_f64()
            );
        }
    }
    writer.flush().unwrap();
    println!(
        "Phase G native sweep done: rows={} parity_failures={} elapsed {:.3}s",
        limit,
        parity_failures,
        started.elapsed().as_secs_f64()
    );
    assert_eq!(
        parity_failures, 0,
        "P=4 native top-10 ID sets must match accepted HTTP results"
    );
}
