use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::bitvec::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::ScoredPointOffset;
use segment::data_types::query_context::{QueryContext, VectorQueryContext};
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::id_tracker::{IdTracker, IdTrackerRead};
use segment::index::lmi_index::LmiCandidateMode;
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::segment_constructor::load_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Condition, Distance, Filter, HnswGlobalConfig, Indexes, SearchParams};
use segment::vector_storage::query::RecoQuery;
use segment::vector_storage::{VectorStorage, VectorStorageRead};

struct Fixture {
    _segments: tempfile::TempDir,
    plain: Segment,
    lmi: Segment,
}

fn fixture(count: usize, distance: Distance) -> Fixture {
    let segments = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(segments.path(), 2, distance).unwrap();
    let hw = HardwareCounterCell::new();
    for (i, vector) in [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]]
        .iter()
        .take(count)
        .enumerate()
    {
        plain
            .upsert_point(
                i as u64 + 1,
                (i as u64 + 1).into(),
                only_default_vector(vector),
                &hw,
            )
            .unwrap();
    }
    let mut config = plain.config().clone();
    config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .index = Indexes::Lmi {};
    let mut builder =
        SegmentBuilder::new(staging.path(), &config, &HnswGlobalConfig::default()).unwrap();
    builder
        .update(&[&plain], &AtomicBool::new(false), &hw)
        .unwrap();
    let lmi = builder.build_for_test(segments.path());
    Fixture {
        _segments: segments,
        plain,
        lmi,
    }
}

fn mode(segment: &Segment, mode: LmiCandidateMode) {
    let mut index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow_mut();
    let VectorIndexEnum::Lmi(lmi) = &mut *index else {
        panic!("expected LMI")
    };
    lmi.set_candidate_mode(mode);
}

fn search(
    segment: &Segment,
    queries: &[&QueryVector],
    top: usize,
    filter: Option<&Filter>,
    params: Option<&SearchParams>,
    context: &VectorQueryContext,
) -> Vec<Vec<ScoredPointOffset>> {
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(queries, filter, top, params, context)
        .unwrap()
}

fn ids(results: &[Vec<ScoredPointOffset>]) -> Vec<Vec<u32>> {
    results
        .iter()
        .map(|r| r.iter().map(|p| p.idx).collect())
        .collect()
}

#[test]
fn multi_query_routes_distinct_postings_and_native_top_k() {
    let f = fixture(4, Distance::Dot);
    mode(&f.lmi, LmiCandidateMode::DeterministicTwoBuckets);
    let positive = [1.0, 0.0].into();
    let negative = [-1.0, 0.0].into();
    let context = VectorQueryContext::default();
    let queries = [&positive, &negative];
    let result = search(&f.lmi, &queries, 10, None, None, &context);
    println!("routed batch = {result:?}");
    assert_eq!(
        result,
        vec![
            vec![
                ScoredPointOffset { idx: 0, score: 1.0 },
                ScoredPointOffset { idx: 2, score: 0.0 }
            ],
            vec![
                ScoredPointOffset { idx: 3, score: 1.0 },
                ScoredPointOffset {
                    idx: 1,
                    score: -0.8
                }
            ],
        ]
    );
    for (i, query) in queries.iter().enumerate() {
        assert_eq!(
            search(&f.lmi, &[*query], 10, None, None, &context)[0],
            result[i]
        );
    }
    assert_eq!(
        ids(&search(&f.lmi, &queries, 1, None, None, &context)),
        vec![vec![0], vec![3]]
    );
    assert_eq!(
        search(&f.lmi, &queries, 0, None, None, &context),
        vec![vec![], vec![]]
    );
    assert!(search(&f.lmi, &[], 10, None, None, &context).is_empty());
}

#[test]
fn all_points_batch_matches_plain_across_metrics_and_top_k() {
    for distance in [
        Distance::Dot,
        Distance::Cosine,
        Distance::Euclid,
        Distance::Manhattan,
    ] {
        let f = fixture(4, distance);
        let q1 = [1.0, 0.0].into();
        let q2 = [-1.0, 0.0].into();
        let q3 = [0.0, 0.0].into(); // ties exercise native queue ordering
        for top in [0, 1, 2, 10] {
            let context = VectorQueryContext::default();
            let actual = search(&f.lmi, &[&q1, &q2, &q3], top, None, None, &context);
            assert_eq!(
                actual,
                search(&f.plain, &[&q1, &q2, &q3], top, None, None, &context)
            );
        }
        println!("all-points Plain equivalence: {distance:?}, top=[0,1,2,10], batch=3");
    }
}

#[test]
fn routing_rejects_point_vector_and_context_deletions() {
    let f = fixture(4, Distance::Dot);
    mode(&f.lmi, LmiCandidateMode::DeterministicTwoBuckets);
    // Delete only the point mapping: its vector is deliberately still alive.
    f.lmi.id_tracker.borrow_mut().drop(1.into()).unwrap();
    assert!(
        !f.lmi.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow()
            .is_deleted_vector(0)
    );
    // Delete only a vector: its point mapping deliberately still exists.
    f.lmi.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow_mut()
        .delete_vector(3)
        .unwrap();
    assert!(f.lmi.id_tracker.borrow().external_id(3).is_some());
    let positive = [1.0, 0.0].into();
    let negative = [-1.0, 0.0].into();
    let queries = [&positive, &negative];
    let actual = search(
        &f.lmi,
        &queries,
        10,
        None,
        None,
        &VectorQueryContext::default(),
    );
    assert_eq!(ids(&actual), vec![vec![2], vec![1]]);
    // An oversized context mask must not resurrect tracker deletions, and its
    // extra clear bits must never cause out-of-range vector reads.
    let mut deleted = BitVec::repeat(false, 16);
    deleted.set(2, true);
    let context = QueryContext::default();
    let segment_context = context
        .get_segment_query_context()
        .with_deleted_points(&deleted);
    let vector_context = segment_context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    assert_eq!(
        ids(&search(&f.lmi, &queries, 10, None, None, &vector_context)),
        vec![vec![], vec![1]]
    );
    mode(&f.lmi, LmiCandidateMode::AllValidPoints);
    assert_eq!(
        ids(&search(&f.lmi, &queries, 10, None, None, &vector_context)),
        vec![vec![1], vec![1]]
    );
    // Missing point-mask bits are rejected conservatively.
    let short = BitVec::repeat(false, 1);
    let segment_context = context
        .get_segment_query_context()
        .with_deleted_points(&short);
    let vector_context = segment_context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    mode(&f.lmi, LmiCandidateMode::DeterministicTwoBuckets);
    assert_eq!(
        ids(&search(&f.lmi, &queries, 10, None, None, &vector_context)),
        vec![Vec::<u32>::new(), Vec::<u32>::new()]
    );
    println!(
        "validity: point-only, vector-only, context override, short mask and oversized mask passed"
    );
}

#[test]
fn filtered_parameterized_and_complex_queries_fall_back_to_plain() {
    let f = fixture(4, Distance::Dot);
    mode(&f.lmi, LmiCandidateMode::DeterministicTwoBuckets);
    let q = [1.0, 0.0].into();
    let query_context = QueryContext::new(
        1024,
        common::counter::hardware_accumulator::HwMeasurementAcc::new(),
    );
    let segment_context = query_context.get_segment_query_context();
    let context = segment_context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    let filter = Filter::new_must(Condition::HasId(
        [2.into(), 4.into()]
            .into_iter()
            .collect::<ahash::AHashSet<_>>()
            .into(),
    ));
    assert_eq!(
        search(&f.lmi, &[&q], 10, Some(&filter), None, &context),
        search(&f.plain, &[&q], 10, Some(&filter), None, &context)
    );
    for params in [
        SearchParams::default(),
        SearchParams {
            exact: true,
            ..Default::default()
        },
        SearchParams {
            indexed_only: true,
            ..Default::default()
        },
    ] {
        assert_eq!(
            search(&f.lmi, &[&q], 10, None, Some(&params), &context),
            search(&f.plain, &[&q], 10, None, Some(&params), &context)
        );
    }
    let complex = QueryVector::RecommendBestScore(RecoQuery::new(
        vec![vec![1.0, 0.0].into()],
        vec![vec![-1.0, 0.0].into()],
    ));
    let mixed = search(&f.lmi, &[&q, &complex], 10, None, None, &context);
    assert_eq!(
        mixed[1],
        search(&f.plain, &[&complex], 10, None, None, &context)[0]
    );
    assert_eq!(ids(&mixed)[0], vec![0, 2]);
}

#[test]
fn empty_buckets_and_cancellation_are_well_defined() {
    let f = fixture(1, Distance::Dot);
    mode(&f.lmi, LmiCandidateMode::DeterministicTwoBuckets);
    let negative = [-1.0, 0.0].into();
    assert_eq!(
        search(
            &f.lmi,
            &[&negative],
            10,
            None,
            None,
            &VectorQueryContext::default()
        ),
        vec![vec![]]
    );
    let context = QueryContext::default().with_is_stopped(Arc::new(AtomicBool::new(true)));
    let sc = context.get_segment_query_context();
    let vc = sc.get_vector_context(DEFAULT_VECTOR_NAME, None);
    for selected in [
        LmiCandidateMode::AllValidPoints,
        LmiCandidateMode::DeterministicTwoBuckets,
    ] {
        mode(&f.lmi, selected);
        assert!(
            f.lmi.vector_data[DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow()
                .search(&[&negative], None, 10, None, &vc)
                .is_err()
        );
    }
    let empty = fixture(0, Distance::Dot);
    for selected in [
        LmiCandidateMode::AllValidPoints,
        LmiCandidateMode::DeterministicTwoBuckets,
    ] {
        mode(&empty.lmi, selected);
        assert_eq!(
            search(
                &empty.lmi,
                &[&negative],
                10,
                None,
                None,
                &VectorQueryContext::default()
            ),
            vec![vec![]]
        );
    }
}

#[test]
fn build_reopen_restores_exact_default_and_can_reenable_routing() {
    let f = fixture(4, Distance::Dot);
    let q1 = [1.0, 0.0].into();
    let q2 = [-1.0, 0.0].into();
    let context = VectorQueryContext::default();
    mode(&f.lmi, LmiCandidateMode::DeterministicTwoBuckets);
    let before = search(&f.lmi, &[&q1, &q2], 10, None, None, &context);
    let path = f.lmi.segment_path.clone();
    let uuid = f.lmi.uuid;
    drop(f.lmi);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false)).unwrap();
    assert_eq!(
        search(&reopened, &[&q1, &q2], 10, None, None, &context),
        search(&f.plain, &[&q1, &q2], 10, None, None, &context)
    );
    mode(&reopened, LmiCandidateMode::DeterministicTwoBuckets);
    assert_eq!(
        search(&reopened, &[&q1, &q2], 10, None, None, &context),
        before
    );
    println!("reopen: exact default restored; opt-in deterministic results reproduced");
}
#[test]
fn deferred_postings_are_excluded_before_scoring() {
    use segment::id_tracker::IdTrackerEnum;
    use segment::id_tracker::mutable_id_tracker::MutableIdTracker;
    let f = fixture(4, Distance::Dot);
    let tracker_dir = tempfile::tempdir().unwrap();
    let mut tracker = MutableIdTracker::open(tracker_dir.path(), Some(2)).unwrap();
    for id in 0..4 {
        tracker.set_link((id as u64 + 1).into(), id).unwrap();
        tracker.set_internal_version(id, id as u64 + 1).unwrap();
    }
    *f.lmi.id_tracker.borrow_mut() = IdTrackerEnum::MutableIdTracker(tracker);
    let q1 = [1.0, 0.0].into();
    let q2 = [-1.0, 0.0].into();
    let context = VectorQueryContext::default();
    mode(&f.lmi, LmiCandidateMode::DeterministicTwoBuckets);
    assert_eq!(
        ids(&search(&f.lmi, &[&q1, &q2], 10, None, None, &context)),
        vec![vec![0], vec![1]]
    );
    mode(&f.lmi, LmiCandidateMode::AllValidPoints);
    assert_eq!(
        ids(&search(&f.lmi, &[&q1, &q2], 10, None, None, &context)),
        vec![vec![0, 1], vec![1, 0]]
    );
    println!("deferred cutoff=2: offsets 2 and 3 excluded in both candidate modes");
}

#[test]
fn delegated_updates_are_seen_without_cached_postings() {
    use segment::index::VectorIndex;
    let f = fixture(4, Distance::Dot);
    // The built LMI segment is deliberately non-appendable. This isolated
    // trait-level fixture swaps in a mutable tracker; it does not claim that
    // appending to an optimized LMI segment is supported.
    let mut tracker = segment::id_tracker::in_memory_id_tracker::InMemoryIdTracker::new();
    for id in 0..4 {
        tracker.set_link((id as u64 + 1).into(), id).unwrap();
        tracker.set_internal_version(id, id as u64 + 1).unwrap();
    }
    *f.lmi.id_tracker.borrow_mut() = segment::id_tracker::IdTrackerEnum::InMemoryIdTracker(tracker);
    mode(&f.lmi, LmiCandidateMode::DeterministicTwoBuckets);
    let hw = HardwareCounterCell::new();
    let q = [1.0, 0.0].into();
    let context = VectorQueryContext::default();
    let vector = [2.0_f32, 0.0];
    f.lmi.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow_mut()
        .update_vector(2, Some(vector.as_slice().into()), &hw)
        .unwrap();
    assert_eq!(
        ids(&search(&f.lmi, &[&q], 1, None, None, &context)),
        vec![vec![2]]
    );
    let vector = [3.0_f32, 0.0];
    f.lmi.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow_mut()
        .update_vector(4, Some(vector.as_slice().into()), &hw)
        .unwrap();
    // Storage extent alone does not make this point valid.
    assert_eq!(
        ids(&search(&f.lmi, &[&q], 10, None, None, &context)),
        vec![vec![2, 0]]
    );
    let oversized = BitVec::repeat(false, 16);
    let qc = QueryContext::default();
    let sc = qc
        .get_segment_query_context()
        .with_deleted_points(&oversized);
    let vc = sc.get_vector_context(DEFAULT_VECTOR_NAME, None);
    // A caller mask cannot invent a mapping for the new storage-only offset.
    assert_eq!(
        ids(&search(&f.lmi, &[&q], 10, None, None, &vc)),
        vec![vec![2, 0]]
    );
    f.lmi.id_tracker.borrow_mut().set_link(5.into(), 4).unwrap();
    f.lmi
        .id_tracker
        .borrow_mut()
        .set_internal_version(4, 5)
        .unwrap();
    assert_eq!(
        ids(&search(&f.lmi, &[&q], 10, None, None, &context)),
        vec![vec![4, 2, 0]]
    );
    f.lmi.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow_mut()
        .update_vector(4, None, &hw)
        .unwrap();
    assert_eq!(
        ids(&search(&f.lmi, &[&q], 10, None, None, &context)),
        vec![vec![2, 0]]
    );
}

#[test]
fn quantized_storage_keeps_plain_search_semantics() {
    use segment::index::plain_vector_index::PlainVectorIndex;
    let f = fixture(4, Distance::Dot);
    let staging = tempfile::tempdir().unwrap();
    let mut config = f.lmi.config().clone();
    config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .quantization_config = Some(serde_json::from_str(r#"{"scalar":{"type":"int8"}}"#).unwrap());
    let mut builder =
        SegmentBuilder::new(staging.path(), &config, &HnswGlobalConfig::default()).unwrap();
    builder
        .update(
            &[&f.plain],
            &AtomicBool::new(false),
            &HardwareCounterCell::new(),
        )
        .unwrap();
    let lmi = builder.build_for_test(f._segments.path());
    let data = &lmi.vector_data[DEFAULT_VECTOR_NAME];
    assert!(data.quantized_vectors.borrow().is_some());
    let plain = PlainVectorIndex::new(
        lmi.id_tracker.clone(),
        data.vector_storage.clone(),
        data.quantized_vectors.clone(),
        lmi.payload_index.clone(),
    );
    mode(&lmi, LmiCandidateMode::DeterministicTwoBuckets);
    let q1 = [1.0, 0.0].into();
    let q2 = [-1.0, 0.0].into();
    let context = VectorQueryContext::default();
    let actual = search(&lmi, &[&q1, &q2], 10, None, None, &context);
    assert_eq!(
        actual,
        plain.search(&[&q1, &q2], None, 10, None, &context).unwrap()
    );
    assert_eq!(actual.iter().map(Vec::len).collect::<Vec<_>>(), vec![4, 4]);
    println!("quantized storage: default search matches Plain and bypasses fake routing");
}
#[test]
fn point_mask_is_not_the_vector_deletion_mask() {
    use segment::index::hnsw_index::point_scorer::BatchFilteredSearcher;
    use segment::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
    use segment::vector_storage::quantized::quantized_vectors::QuantizedVectors;
    let mut storage = new_volatile_dense_vector_storage(2, Distance::Dot);
    let hw = HardwareCounterCell::new();
    for (id, vector) in [[1.0_f32, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]]
        .iter()
        .enumerate()
    {
        storage
            .insert_vector(id as u32, vector.as_slice().into(), &hw)
            .unwrap();
    }
    let q = [1.0, 0.0].into();
    let score = |mask| {
        BatchFilteredSearcher::new(
            &[&q],
            &storage,
            None::<&QuantizedVectors>,
            None,
            10,
            mask,
            HardwareCounterCell::new(),
        )
        .unwrap()
        .peek_top_iter([1, 3].into_iter(), &AtomicBool::new(false))
        .unwrap()
    };
    // Reproduce the original failure with the empty mask typical of this storage.
    assert!(storage.deleted_vector_bitslice().is_empty());
    assert!(score(storage.deleted_vector_bitslice())[0].is_empty());
    let alive = BitVec::repeat(false, storage.total_vector_count());
    let fixed = score(alive.as_bitslice());
    assert_eq!(ids(&fixed), vec![vec![1, 3]]);
    println!("mask regression: vector mask -> [[]]; explicit point mask -> {fixed:?}");
}
