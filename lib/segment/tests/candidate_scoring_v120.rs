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
