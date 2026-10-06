use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::bitvec::BitVec;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use segment::data_types::query_context::{QueryContext, VectorQueryContext};
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::id_tracker::{IdTracker, IdTrackerRead};
use segment::index::candidate_scoring::score_candidates;
use segment::segment::Segment;
use segment::segment_constructor::build_segment;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Distance, VectorStorageDatatype};
use segment::vector_storage::{VectorStorage, VectorStorageRead};

fn fixture() -> (tempfile::TempDir, Segment) {
    let root = tempfile::tempdir().unwrap();
    let mut segment = build_simple_segment(root.path(), 2, Distance::Dot).unwrap();
    let hw = HardwareCounterCell::new();
    for (i, vector) in [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0], [0.8, 0.2]]
        .iter()
        .enumerate()
    {
        segment
            .upsert_point(
                i as u64 + 1,
                (i as u64 + 1).into(),
                only_default_vector(vector),
                &hw,
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
    let (_root, segment) = fixture();
    assert_eq!(scored(&segment, &[1, 2, 3], 2), vec![1, 2]);
    assert_eq!(scored(&segment, &[1, 2, 3], 0), Vec::<u32>::new());
}

#[test]
fn candidate_order_and_duplicates_do_not_change_results() {
    let (_root, segment) = fixture();
    let first = scored(&segment, &[3, 1, 2, 1], 3);
    let reverse = scored(&segment, &[1, 2, 1, 3], 3);
    assert_eq!(first, reverse);
    assert_eq!(first, vec![1, 2, 3]);
}

#[test]
fn point_and_vector_deletions_are_independent() {
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
    let (_root, segment) = fixture();
    let left = scored(&segment, &[4, 1, 3], 2);
    let right = scored(&segment, &[3, 1, 4], 2);
    assert_eq!(left, right);
    assert_eq!(left, vec![1, 4]);
}

#[test]
fn context_deleted_mask_and_cancellation_are_preserved() {
    let (_root, segment) = fixture();
    let query: QueryVector = [1.0_f32, 0.0].into();
    let mut deleted = BitVec::repeat(false, 5);
    deleted.set(0, true);
    let stopped = Arc::new(AtomicBool::new(false));
    let context = QueryContext::new(1024, HwMeasurementAcc::new()).with_is_stopped(stopped.clone());
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
    let hw = HardwareCounterCell::new();
    for (i, vector) in [[1.0, 0.0], [0.7, 0.7], [-1.0, 0.0]].iter().enumerate() {
        segment
            .upsert_point(
                i as u64 + 1,
                (i as u64 + 1).into(),
                only_default_vector(vector),
                &hw,
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
