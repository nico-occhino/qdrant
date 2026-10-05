//! Explicit read-only S.3D native nprobe sweep over a completed persisted index.
use std::collections::HashSet;
use std::io::{BufWriter, Cursor, Read, Write};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use serde_json::json;

use crate::data_types::query_context::QueryContext;
use crate::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector};
use crate::id_tracker::IdTrackerRead;
use crate::index::VectorIndexEnum;
use crate::segment_constructor::load_segment;
use crate::types::{Distance, PointIdType};

fn u32_le(input: &mut impl Read) -> u32 {
    let mut raw = [0; 4];
    input.read_exact(&mut raw).unwrap();
    u32::from_le_bytes(raw)
}

fn u64_le(input: &mut impl Read) -> u64 {
    let mut raw = [0; 8];
    input.read_exact(&mut raw).unwrap();
    u64::from_le_bytes(raw)
}

struct Query {
    id: u64,
    vector: Vec<f32>,
    first_ten: HashSet<PointIdType>,
    tie_eligible: HashSet<PointIdType>,
}

fn queries(path: &str) -> Vec<Query> {
    let bytes = fs_err::read(path).unwrap();
    let mut input = Cursor::new(bytes);
    let mut magic = [0; 8];
    input.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"LMIQ10M1");
    let count = u32_le(&mut input) as usize;
    let dim = u32_le(&mut input) as usize;
    assert_eq!((count, dim), (4_992, 768));
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let id = u64_le(&mut input);
        let vector = (0..dim)
            .map(|_| f32::from_bits(u32_le(&mut input)))
            .collect::<Vec<_>>();
        assert!(vector.iter().all(|value| value.is_finite()));
        let first_ten = (0..10)
            .map(|_| PointIdType::NumId(u64_le(&mut input)))
            .collect();
        let tied_count = u32_le(&mut input) as usize;
        assert!((10..=4_096).contains(&tied_count));
        let tie_eligible = (0..tied_count)
            .map(|_| PointIdType::NumId(u64_le(&mut input)))
            .collect();
        rows.push(Query {
            id,
            vector,
            first_ten,
            tie_eligible,
        });
    }
    assert_eq!(input.position() as usize, input.get_ref().len());
    rows
}

#[test]
#[ignore = "Explicit 10M persisted native LMI all-query nprobe sweep"]
fn phase_s3d_native_nprobe_sweep() {
    let segment_path = std::env::var("LMI_S3D_SEGMENT_PATH").unwrap();
    let input_path = std::env::var("LMI_S3D_QUERY_INPUT").unwrap();
    let output_path = std::env::var("LMI_S3D_NATIVE_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output_path).exists());
    let rows = queries(&input_path);
    let selected = rows
        .iter()
        .enumerate()
        .skip(35)
        .step_by(25)
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 199);
    let stopped = AtomicBool::new(false);
    let open_started = Instant::now();
    let segment = load_segment(
        std::path::Path::new(&segment_path),
        uuid::Uuid::nil(),
        None,
        &stopped,
    )
    .unwrap();
    let reopen_seconds = open_started.elapsed().as_secs_f64();
    let index_ref = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(index) = &*index_ref else {
        panic!("persisted segment did not reopen as native LMI")
    };
    let state = index.routing_state().expect("trained routing state");
    assert_eq!(state.postings().point_count(), 10_000_000);
    assert_eq!(state.postings().len(), 3_162);
    let tracker = segment.id_tracker.borrow();
    assert_eq!(tracker.available_point_count(), 10_000_000);
    let mut output = BufWriter::new(
        fs_err::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output_path)
            .unwrap(),
    );
    let run = |query: &Query, probe: usize| {
        let raw_query: QueryVector = query.vector.clone().into();
        let normalized = Distance::Cosine.preprocess_vector::<f32>(query.vector.clone());
        let route_query: QueryVector = normalized.clone().into();
        let total_started = Instant::now();
        let started = Instant::now();
        let buckets = state
            .top_buckets_for_test(&normalized, probe, &stopped)
            .unwrap();
        let route_ns = started.elapsed().as_nanos();
        let started = Instant::now();
        let mut candidates = buckets
            .iter()
            .flat_map(|&bucket| state.postings().get(bucket).unwrap().iter().copied())
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        candidates.dedup();
        let candidate_count = candidates.len();
        let gather_ns = started.elapsed().as_nanos();
        if probe == 4 && query.id < 10_000_020 {
            assert_eq!(
                state
                    .candidates_for_query(&route_query, &stopped)
                    .unwrap()
                    .unwrap(),
                candidates
            );
        }
        let root_context = QueryContext::default();
        let context = root_context.get_segment_query_context();
        let context = context.get_vector_context(DEFAULT_VECTOR_NAME, None);
        let started = Instant::now();
        let found = index
            .score_candidates_for_test(&raw_query, candidates.into_iter(), 10, &context)
            .unwrap();
        let scoring_topk_ns = started.elapsed().as_nanos();
        let total_ns = total_started.elapsed().as_nanos();
        let result_ids = found
            .iter()
            .map(|point| tracker.external_id(point.idx).expect("external point ID"))
            .collect::<Vec<_>>();
        assert!(result_ids.len() <= 10);
        let conventional_hits = result_ids
            .iter()
            .filter(|id| query.first_ten.contains(id))
            .count();
        let tie_aware_hits = result_ids
            .iter()
            .filter(|id| query.tie_eligible.contains(id))
            .count();
        json!({
            "query_id":query.id,"nprobe":probe,"candidate_count":candidate_count,
            "candidate_fraction":candidate_count as f64/10_000_000.0,
            "route_ns":route_ns,"gather_ns":gather_ns,
            "scoring_topk_ns":scoring_topk_ns,"total_ns":total_ns,
            "recall10_conventional":conventional_hits as f64/10.0,
            "recall10_tie_aware":tie_aware_hits as f64/10.0,
            "result_ids":result_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
        })
    };
    for query in rows.iter().take(20) {
        let _ = run(query, 4);
    }
    for probe in [1, 2, 4, 8, 16] {
        let probe_started = Instant::now();
        for &(query_row, query) in &selected {
            let mut record = run(query, probe);
            record["query_row"] = json!(query_row);
            writeln!(output, "{record}").unwrap();
        }
        output.flush().unwrap();
        eprintln!(
            "S3D_NATIVE_PROBE nprobe={probe} queries={} seconds={:.3} reopen_seconds={reopen_seconds:.3}",
            selected.len(),
            probe_started.elapsed().as_secs_f64(),
        );
    }
    fs_err::write(
        format!("{output_path}.metadata.json"),
        serde_json::to_vec_pretty(&json!({
            "segment_path":segment_path,
            "query_input":input_path,
            "queries":selected.len(),
            "warmup_queries":20,
            "query_selection":{"start":35,"end":4992,"stride":25},
            "nprobes":[1,2,4,8,16],
            "trials_per_nprobe":1,
            "native_reopen_seconds":reopen_seconds,
            "scorer":"LmiIndex candidate scorer using BatchFilteredSearcher and Qdrant vector storage",
            "timing":"in-process: route; posting gather; scoring and top-k combined; total",
        }))
        .unwrap(),
    )
    .unwrap();
}
