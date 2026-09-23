use common::bitvec::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use common::fixed_length_priority_queue::FixedLengthPriorityQueue;
use common::types::{PointOffsetType, ScoredPointOffset};
use std::sync::atomic::AtomicBool;

use segment::data_types::vectors::QueryVector;
use segment::index::hnsw_index::point_scorer::BatchFilteredSearcher;
use segment::types::Distance;
use segment::vector_storage::VectorStorageRead;
use segment::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
use segment::vector_storage::{VectorStorage, new_raw_scorer};

#[test]
fn routed_bucket_candidates_can_be_scored_and_ranked() {
    let mut storage = new_volatile_dense_vector_storage(2, Distance::Dot);

    let hw_counter = HardwareCounterCell::new();

    // Internal PointOffsetTypes:
    //
    // 0 -> dot([1.0, 0.0], [1.0, 0.0])  =  1.0
    // 1 -> dot([1.0, 0.0], [0.8, 0.2])  =  0.8
    // 2 -> dot([1.0, 0.0], [0.0, 1.0])  =  0.0
    // 3 -> dot([1.0, 0.0], [-1.0, 0.0]) = -1.0

    storage
        .insert_vector(0, (&[1.0_f32, 0.0][..]).into(), &hw_counter)
        .unwrap();

    storage
        .insert_vector(1, (&[0.8_f32, 0.2][..]).into(), &hw_counter)
        .unwrap();

    storage
        .insert_vector(2, (&[0.0_f32, 1.0][..]).into(), &hw_counter)
        .unwrap();

    storage
        .insert_vector(3, (&[-1.0_f32, 0.0][..]).into(), &hw_counter)
        .unwrap();

    let query: QueryVector = [1.0_f32, 0.0].into();

    let scorer = new_raw_scorer(query, &storage, HardwareCounterCell::new()).unwrap();

    // Fake LMI-style bucket postings:
    //
    // bucket 0 -> [0, 2]
    // bucket 1 -> [1, 3]
    let bucket_postings: Vec<Vec<PointOffsetType>> = vec![vec![0, 2], vec![1, 3]];

    // Fake router:
    //
    // Pretend the learned router chose bucket 1 for this query.
    let selected_bucket = 1usize;

    let candidates = &bucket_postings[selected_bucket];

    assert_eq!(candidates, &[1, 3]);

    // RawScorer scores only the routed candidates.
    let mut scores = vec![0.0; candidates.len()];

    scorer.score_points(candidates, &mut scores);

    println!("selected_bucket = {selected_bucket}");
    println!("candidates = {candidates:?}");
    println!("scores = {scores:?}");

    assert_eq!(scores.len(), 2);
    assert!((scores[0] - 0.8).abs() < 1e-6);
    assert!((scores[1] - (-1.0)).abs() < 1e-6);

    // Convert routed candidate scores into the representation expected
    // by the VectorIndex search path and keep only top-1.
    let mut top_k = FixedLengthPriorityQueue::<ScoredPointOffset>::new(1);

    for (&idx, &score) in candidates.iter().zip(&scores) {
        top_k.push(ScoredPointOffset { idx, score });
    }

    let result = top_k.into_sorted_vec();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].idx, 1);
    assert!((result[0].score - 0.8).abs() < 1e-6);
}

#[test]
fn routed_candidates_work_through_qdrant_batch_searcher() {
    let mut storage = new_volatile_dense_vector_storage(2, Distance::Dot);

    let hw_counter = HardwareCounterCell::new();

    storage
        .insert_vector(0, (&[1.0_f32, 0.0][..]).into(), &hw_counter)
        .unwrap();

    storage
        .insert_vector(1, (&[0.8_f32, 0.2][..]).into(), &hw_counter)
        .unwrap();

    storage
        .insert_vector(2, (&[0.0_f32, 1.0][..]).into(), &hw_counter)
        .unwrap();

    storage
        .insert_vector(3, (&[-1.0_f32, 0.0][..]).into(), &hw_counter)
        .unwrap();

    let query: QueryVector = [1.0_f32, 0.0].into();

    // Pretend these are LMI postings.
    let buckets: Vec<Vec<PointOffsetType>> = vec![vec![0, 2], vec![1, 3]];

    // Pretend the learned router selected bucket 1.
    let selected_bucket = 1;
    let candidates = buckets[selected_bucket].clone();

    let queries = [&query];

    let point_deleted = BitVec::repeat(false, storage.total_vector_count());

    let batch_searcher = BatchFilteredSearcher::new(
        &queries,
        &storage,
        None::<&segment::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
        None,
        2,
        point_deleted.as_bitslice(),
        HardwareCounterCell::new(),
    )
    .unwrap();

    let stopped = AtomicBool::new(false);

    let result = batch_searcher
        .peek_top_iter(candidates.into_iter(), &stopped)
        .unwrap();

    println!("result = {result:?}");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 2);

    assert_eq!(result[0][0].idx, 1);
    assert!((result[0][0].score - 0.8).abs() < 1e-6);

    assert_eq!(result[0][1].idx, 3);
    assert!((result[0][1].score - (-1.0)).abs() < 1e-6);
}
