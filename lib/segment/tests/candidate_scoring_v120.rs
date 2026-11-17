use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::bitvec::BitVec;
use common::types::PointOffsetType;
use segment::data_types::query_context::{QueryContext, VectorQueryContext};
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::id_tracker::{IdTracker, IdTrackerRead};
use segment::index::candidate_scoring::score_candidates;
#[cfg(feature = "lmi-research")]
use segment::index::candidate_scoring::score_candidates_profiled;
use segment::segment::Segment;
use segment::segment_constructor::build_segment;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Distance, VectorStorageDatatype};
use segment::vector_storage::{VectorStorage, VectorStorageRead};

fn fixture() -> (tempfile::TempDir, Segment) {
    let root = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(root.path(), 2, Distance::Dot).unwrap();
    for (i, vector) in [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0], [0.8, 0.2]]
        .iter()
        .enumerate()
    {
        segment
            .upsert_point(
                i as u64 + 1,
                (i as u64 + 1).into(),
                only_default_vector(vector),
            )
            .unwrap();
    }
    (root, segment)
}

fn scored(segment: &Segment, candidates: &[PointOffsetType], top: usize) -> Vec<u32> {
    let query: QueryVector = [1.0_f32, 0.0].into();
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    score_candidates(
        &query,
        candidates,
        top,
        &tracker,
        &storage,
        &VectorQueryContext::default(),
    )
    .unwrap()
    .into_iter()
    .map(|point| point.idx)
    .collect()
}

#[test]
fn subset_scores_with_current_qdrant_scorer() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, segment) = fixture();
    assert_eq!(scored(&segment, &[1, 2, 3], 2), vec![1, 2]);
    assert_eq!(scored(&segment, &[1, 2, 3], 0), Vec::<u32>::new());
}

#[test]
fn candidate_order_and_duplicates_do_not_change_results() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, segment) = fixture();
    let first = scored(&segment, &[3, 1, 2, 1], 3);
    let reverse = scored(&segment, &[1, 2, 1, 3], 3);
    assert_eq!(first, reverse);
    assert_eq!(first, vec![1, 2, 3]);
}

#[test]
fn point_and_vector_deletions_are_independent() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, segment) = fixture();
    segment.id_tracker.borrow_mut().drop(2.into()).unwrap();
    assert!(
        !segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow()
            .is_deleted_vector(1)
    );
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow_mut()
        .delete_vector(0)
        .unwrap();
    assert!(segment.id_tracker.borrow().external_id(0).is_some());
    assert_eq!(scored(&segment, &[0, 1, 2, 3], 4), vec![2, 3]);
}

#[test]
fn equal_scores_have_stable_membership_after_candidate_normalization() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, segment) = fixture();
    let left = scored(&segment, &[4, 1, 3], 2);
    let right = scored(&segment, &[3, 1, 4], 2);
    assert_eq!(left, right);
    assert_eq!(left, vec![1, 4]);
}

#[test]
fn context_deleted_mask_and_cancellation_are_preserved() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, segment) = fixture();
    let query: QueryVector = [1.0_f32, 0.0].into();
    let mut deleted = BitVec::repeat(false, 5);
    deleted.set(0, true);
    let stopped = Arc::new(AtomicBool::new(false));
    let context = QueryContext::new(1024, common::ambient::Handoff::Unmeasured(None))
        .with_is_stopped(stopped.clone());
    let segment_context = context
        .get_segment_query_context()
        .with_deleted_points(&deleted);
    let vector_context = segment_context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let result =
        score_candidates(&query, &[0, 1, 2], 3, &tracker, &storage, &vector_context).unwrap();
    assert_eq!(result.iter().map(|p| p.idx).collect::<Vec<_>>(), vec![1, 2]);
    stopped.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(score_candidates(&query, &[0, 1, 2], 3, &tracker, &storage, &vector_context).is_err());
}

#[test]
fn float16_cosine_candidates_use_authoritative_storage() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let root = tempfile::tempdir().unwrap();
    let plain = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
    let mut config = plain.config().clone();
    config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .datatype = Some(VectorStorageDatatype::Float16);
    drop(plain);
    let mut segment = build_segment(root.path(), &config, None, true).unwrap().0;
    for (i, vector) in [[1.0, 0.0], [0.7, 0.7], [-1.0, 0.0]].iter().enumerate() {
        segment
            .upsert_point(
                i as u64 + 1,
                (i as u64 + 1).into(),
                only_default_vector(vector),
            )
            .unwrap();
    }
    assert_eq!(
        segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow()
            .datatype(),
        VectorStorageDatatype::Float16
    );
    let query: QueryVector = [1.0_f32, 0.0].into();
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let result = score_candidates(
        &query,
        &[0, 1, 2],
        2,
        &tracker,
        &storage,
        &VectorQueryContext::default(),
    )
    .unwrap();
    assert_eq!(result.iter().map(|p| p.idx).collect::<Vec<_>>(), vec![0, 1]);
}

#[cfg(feature = "lmi-research")]
#[test]
fn research_profiler_preserves_scores_and_counts_actual_scoring() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI research profiler"));
    let (_root, segment) = fixture();
    let query: QueryVector = [1.0_f32, 0.0].into();
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let context = VectorQueryContext::default();
    let input = [4, 1, 1, 2, 3, 99];
    let expected = score_candidates(&query, &input, 3, &tracker, &storage, &context).unwrap();
    let (observed, profile) =
        score_candidates_profiled(&query, &input, 3, &*tracker, &*storage, &context).unwrap();
    assert_eq!(observed, expected);
    assert_eq!(profile.input_offsets, 6);
    assert_eq!(profile.bounded_offsets, 4);
    assert_eq!(profile.deferred_visible_offsets, 4);
    assert_eq!(profile.scored_vectors, 4);
    assert_eq!(profile.returned_hits, 3);
    assert!(profile.raw_score_ns <= profile.scorer_loop_ns);
    assert!(profile.topk_ns <= profile.scorer_loop_ns);
}

#[cfg(feature = "lmi-research")]
#[test]
fn research_profiler_distinguishes_tracker_and_vector_visibility() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI research profiler"));
    let (_root, segment) = fixture();
    segment.id_tracker.borrow_mut().drop(2.into()).unwrap();
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow_mut()
        .delete_vector(0)
        .unwrap();
    let query: QueryVector = [1.0_f32, 0.0].into();
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let context = VectorQueryContext::default();
    let expected =
        score_candidates(&query, &[0, 1, 2, 3], 4, &tracker, &storage, &context).unwrap();
    let (observed, profile) =
        score_candidates_profiled(&query, &[0, 1, 2, 3], 4, &*tracker, &*storage, &context)
            .unwrap();
    assert_eq!(observed, expected);
    assert_eq!(profile.bounded_offsets, 4);
    assert_eq!(profile.deferred_visible_offsets, 3);
    assert_eq!(profile.scored_vectors, 2);
    assert_eq!(profile.returned_hits, 2);
}

#[test]
fn native_quantized_candidates_rerank_and_delete() {
    use segment::index::candidate_scoring::score_candidates_with_quantization;
    use segment::types::{QuantizationConfig, ScalarQuantizationConfig};
    use segment::vector_storage::quantized::quantized_vectors::{
        QuantizedVectors, QuantizedVectorsStorageType,
    };
    let _scope =
        common::ambient::unmeasured_guard(common::reason::reason("native quantized candidates"));
    let (_root, segment) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let config: QuantizationConfig = ScalarQuantizationConfig {
        memory: None,
        r#type: Default::default(),
        quantile: None,
        always_ram: None,
    }
    .into();
    let quantized = QuantizedVectors::create(
        &*storage,
        &config,
        QuantizedVectorsStorageType::Immutable,
        dir.path(),
        1,
        &AtomicBool::new(false),
    )
    .unwrap();
    let quantized = QuantizedVectors::load(&config, &storage, dir.path(), &AtomicBool::new(false))
        .unwrap()
        .expect("native quantization reopens");
    let missing = tempfile::tempdir().unwrap();
    assert!(
        QuantizedVectors::load(&config, &storage, missing.path(), &AtomicBool::new(false))
            .unwrap()
            .is_none()
    );
    let query: QueryVector = [1.0_f32, 0.0].into();
    let ctx = VectorQueryContext::default();
    let approximate = score_candidates_with_quantization(
        &query,
        &[4, 3, 2, 1, 0, 0],
        5,
        &*tracker,
        &*storage,
        Some(&quantized),
        &ctx,
    )
    .unwrap();
    #[cfg(feature = "lmi-research")]
    {
        let (profiled, profile) = segment::index::candidate_scoring::score_candidates_quantized_profiled(
            &query, &[4,3,2,1,0,0], 5, &*tracker, &*storage, Some(&quantized), &ctx,
        ).unwrap();
        assert_eq!(profiled, approximate);
        assert_eq!(profile.input_offsets, 6);
        assert_eq!(profile.bounded_offsets, 5);
        assert_eq!(profile.scored_vectors, 5);
    }
    let shortlist: Vec<_> = approximate.iter().map(|h| h.idx).collect();
    let reranked = score_candidates(&query, &shortlist, 3, &tracker, &storage, &ctx).unwrap();
    let original = score_candidates(&query, &[0, 1, 2, 3, 4], 3, &tracker, &storage, &ctx).unwrap();
    assert_eq!(reranked, original);
    drop(tracker);
    segment.id_tracker.borrow_mut().drop(1.into()).unwrap();
    let tracker = segment.id_tracker.borrow();
    let approximate = score_candidates_with_quantization(
        &query,
        &[0, 1, 2, 3, 4],
        3,
        &*tracker,
        &*storage,
        Some(&quantized),
        &ctx,
    )
    .unwrap();
    assert_eq!(approximate.len(), 3);
    assert!(approximate.iter().all(|h| h.idx != 0));
}

/// Bounded native scalar control. Reads the previously prepared SISAP subset;
/// does not alter the retained collection or train an LMI model.
#[test]
#[ignore]
fn native_quantized_sisap_subset() {
    use segment::index::candidate_scoring::score_candidates_with_quantization;
    use segment::types::{QuantizationConfig, ScalarQuantizationConfig};
    use segment::vector_storage::quantized::quantized_vectors::{
        QuantizedVectors, QuantizedVectorsStorageType,
    };
    let _scope =
        common::ambient::unmeasured_guard(common::reason::reason("native scalar SISAP control"));
    let input = std::path::PathBuf::from(std::env::var("DLI_TEACHER_CONTROL").unwrap());
    let raw = std::fs::read(input.join("pool.f16")).unwrap();
    assert_eq!(raw.len(), 65536 * 768 * 2);
    let root = tempfile::tempdir().unwrap();
    let segment = if let Ok(path) = std::env::var("DLI_LEARNED_CONTROL_INDEX") {
        let path = std::path::PathBuf::from(path);
        let uuid = path.file_name().unwrap().to_str().unwrap().parse().unwrap();
        segment::segment_constructor::load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap()
    } else {
    let mut segment = build_simple_segment(root.path(), 768, Distance::Cosine).unwrap();
    for (id, bytes) in raw.chunks_exact(768 * 2).enumerate() {
        let v: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect();
        segment
            .upsert_point(1, (id as u64 + 1).into(), only_default_vector(&v))
            .unwrap();
    }
        segment
    };
    let query_bytes = std::fs::read(input.join("queries.f32")).unwrap();
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let config: QuantizationConfig = ScalarQuantizationConfig {
        memory: None,
        r#type: Default::default(),
        quantile: None,
        always_ram: None,
    }
    .into();
    let quantdir = tempfile::tempdir().unwrap();
    let started = std::time::Instant::now();
    let quant = QuantizedVectors::create(
        &*storage,
        &config,
        QuantizedVectorsStorageType::Immutable,
        quantdir.path(),
        2,
        &AtomicBool::new(false),
    )
    .unwrap();
    eprintln!("SCALAR_BUILD_MS={}", started.elapsed().as_millis());
    let candidates: Vec<u32> = (0..65536).collect();
    let ctx = VectorQueryContext::default();
    for (qi, bytes) in query_bytes.chunks_exact(768 * 4).enumerate() {
        let query: QueryVector = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
            .into();
        let candidates = if std::env::var_os("DLI_LEARNED_CONTROL_INDEX").is_some() {
            use segment::index::VectorIndexEnum;
            let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
            let VectorIndexEnum::Lmi(lmi) = &*index else { panic!("trained LMI required") };
            let state = lmi.routing_state().unwrap();
            let values: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let values = Distance::Cosine.preprocess_vector::<f32>(values);
            let logits = state.router().forward(&values).unwrap();
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|a,b| logits[*b].total_cmp(&logits[*a]));
            let mut selected = Vec::new();
            for bucket in order.into_iter().take(16) { selected.extend_from_slice(state.postings().get(bucket).unwrap()); }
            selected.sort_unstable(); selected.dedup(); selected
        } else { candidates.clone() };
        let original_started = std::time::Instant::now();
        let original = score_candidates(&query, &candidates, 30, &tracker, &storage, &ctx).unwrap();
        eprintln!("S0 query={qi} original={} wall_ns={}", candidates.len(), original_started.elapsed().as_nanos());
        if std::env::var_os("DLI_COUNTERBALANCED").is_some() {
            // Warm-only ABBA/BAAB. Baseline above is an explicit untimed-for-this-control warmup.
            let counters = || -> (u64,u64,u64) {
                let stat = std::fs::read_to_string("/proc/thread-self/stat").unwrap();
                let fields: Vec<_> = stat.rsplit_once(')').unwrap().1.split_whitespace().collect();
                let n = |i:usize| fields[i].parse::<u64>().unwrap();
                (n(7), n(9), n(11)+n(12))
            };
            for (pass, approximate_mode) in [false,true,true,false,true,false,false,true].into_iter().enumerate() {
                let before = counters();
                let tick = std::time::Instant::now();
                let (hits, originals) = if approximate_mode {
                    let shortlist = score_candidates_with_quantization(&query, &candidates, 128, &*tracker, &*storage, Some(&quant), &ctx).unwrap();
                    let ids: Vec<_> = shortlist.iter().map(|h|h.idx).collect();
                    (score_candidates(&query, &ids, 30, &tracker, &storage, &ctx).unwrap(), ids.len())
                } else { (score_candidates(&query, &candidates, 30, &tracker, &storage, &ctx).unwrap(), candidates.len()) };
                let ns = tick.elapsed().as_nanos();
                let after = counters();
                let recall = |k:usize| original.iter().take(k).filter(|h| hits.iter().take(k).any(|r|r.idx==h.idx)).count() as f64/k as f64;
                for h in &hits { assert_eq!(h.score, score_candidates(&query,&[h.idx],1,&tracker,&storage,&ctx).unwrap()[0].score); }
                eprintln!("ABBA query={qi} pass={pass} quantized={} original={originals} ns={ns} minor={} major={} cpu_ticks={} recall10={} recall30={}",
                    if approximate_mode {candidates.len()} else {0}, after.0-before.0, after.1-before.1, after.2-before.2, recall(10),recall(30));
            }
            continue;
        }
        for limit in [64, 128, 256, 512, 1024, 2048, 65536] {
            let started = std::time::Instant::now();
            let approximate = score_candidates_with_quantization(
                &query,
                &candidates,
                limit,
                &*tracker,
                &*storage,
                Some(&quant),
                &ctx,
            )
            .unwrap();
            let quantized_ns = started.elapsed().as_nanos();
            let s1_recall = original.iter().filter(|h| approximate.iter().take(30).any(|a| a.idx == h.idx)).count() as f64 / 30.;
            let offsets: Vec<_> = approximate.iter().map(|x| x.idx).collect();
            let reranked =
                score_candidates(&query, &offsets, 30, &tracker, &storage, &ctx).unwrap();
            let execution_ns = started.elapsed().as_nanos();
            let recall = original
                .iter()
                .filter(|h| reranked.iter().any(|r| r.idx == h.idx))
                .count() as f64
                / 30.;
            for hit in &reranked {
                let score =
                    score_candidates(&query, &[hit.idx], 1, &tracker, &storage, &ctx).unwrap();
                assert_eq!(hit.score, score[0].score);
            }
            if limit == 65536 {
                assert_eq!(original, reranked);
            }
            eprintln!(
                "NATIVE_SCALAR query={qi} quantized={} original={} recall30={recall} s1_recall30={s1_recall} quantized_ns={quantized_ns} execution_ns={execution_ns} wall_ms={}",
                candidates.len(), offsets.len(),
                started.elapsed().as_millis()
            );
        }
    }
}
