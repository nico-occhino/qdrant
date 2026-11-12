//! Mock-only demonstration of whole-bucket global allocation, not Qdrant integration.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bucket { level: usize, score: f64, count: usize }
fn global(mut buckets: Vec<Bucket>, budget: usize) -> Vec<Bucket> {
    buckets.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.level.cmp(&b.level)));
    let mut n = 0;
    buckets.into_iter().take_while(|b| { if n >= budget { false } else { n += b.count; true } }).collect()
}
#[test]
fn global_whole_bucket_budget_differs_from_fixed_per_level() {
    let buckets = vec![Bucket { level: 0, score: 1.0, count: 2 }, Bucket { level: 1, score: 3.0, count: 4 }, Bucket { level: 2, score: 2.0, count: 3 }];
    let selected = global(buckets, 5);
    assert_eq!(selected.iter().map(|b| b.level).collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(selected.iter().map(|b| b.count).sum::<usize>(), 7);
    // A fixed per-level probe includes level 0, while the global policy does not.
    assert!(!selected.iter().any(|b| b.level == 0));
    assert!(global(vec![], 5).is_empty());
    assert!(global(selected, 0).is_empty());
}

#[test]
fn zero_based_normalization_changes_buffer_order() {
    let arity = 3_f64;
    let probability = 0.4_f64;
    let paper_level_one = probability * arity.powi(1);
    let reference_index_zero = probability * arity.powi(0);
    let buffer = 1.0;
    assert!(paper_level_one > buffer);
    assert!(reference_index_zero < buffer);
    // The two formulas scale every learned-level score by the same factor,
    // but the buffer is not scaled and therefore relative ordering can change.
    assert!((paper_level_one / reference_index_zero - arity).abs() < 1e-12);
}
