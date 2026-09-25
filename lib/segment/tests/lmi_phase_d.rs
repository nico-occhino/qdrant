use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::bitvec::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::ScoredPointOffset;
use segment::data_types::query_context::{QueryContext, VectorQueryContext};
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::id_tracker::IdTracker;
use segment::index::lmi_index::{
    LinearLayer, LmiCandidateMode, LmiRoutingState, MlpRouter, RouterLayer, build_router_postings,
};
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::segment_constructor::load_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Condition, Distance, Filter, HnswGlobalConfig, Indexes, SearchParams};
use segment::vector_storage::VectorStorage;
use segment::vector_storage::query::RecoQuery;

struct Fixture {
    _segments: tempfile::TempDir,
    plain: Segment,
    lmi: Segment,
}

fn fixture(distance: Distance) -> Fixture {
    fixture_vectors(&[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]], distance)
}

fn fixture_vectors(vectors: &[[f32; 2]], distance: Distance) -> Fixture {
    let segments = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(segments.path(), 2, distance).unwrap();
    let hw = HardwareCounterCell::new();
    for (i, vector) in vectors.iter().enumerate() {
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
    lmi.set_candidate_mode(mode).unwrap();
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

// Human-readable 2 -> 4 -> 3 MLP, with logits [x, -x, y].
fn router() -> MlpRouter {
    MlpRouter {
        layers: vec![
            RouterLayer::Linear(LinearLayer {
                in_features: 2,
                out_features: 4,
                weights: vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0],
                bias: vec![0.0; 4],
            }),
            RouterLayer::ReLU,
            RouterLayer::Linear(LinearLayer {
                in_features: 4,
                out_features: 3,
                weights: vec![
                    1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0,
                ],
                bias: vec![0.0; 3],
            }),
        ],
    }
}
fn install(segment: &Segment, nprobe: usize) {
    let mut index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow_mut();
    let VectorIndexEnum::Lmi(lmi) = &mut *index else {
        panic!("expected LMI")
    };
    lmi.install_static_routing(router(), nprobe, &AtomicBool::new(false))
        .unwrap();
}
fn installed_postings(segment: &Segment) -> Option<Vec<Vec<u32>>> {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected LMI")
    };
    lmi.routing_state().map(|s| s.postings().to_vec())
}

#[test]
fn native_mlp_forward_ordering_and_nprobe_are_deterministic() {
    let r = router();
    for (q, expected, order) in [
        ([2.0, -3.0], vec![2.0, -2.0, -3.0], vec![0, 1, 2]),
        ([-2.0, 3.0], vec![-2.0, 2.0, 3.0], vec![2, 1, 0]),
        ([0.0, 0.0], vec![0.0, 0.0, 0.0], vec![0, 1, 2]),
    ] {
        assert_eq!(r.forward(&q).unwrap(), expected);
        for n in 1..=3 {
            assert_eq!(r.top_buckets(&q, n).unwrap(), order[..n]);
        }
        println!("router q={q:?} logits={expected:?} bucket_order={order:?}");
    }
    assert!(r.top_buckets(&[1.0, 0.0], 0).is_err());
    assert!(r.top_buckets(&[1.0, 0.0], 4).is_err());
}

#[test]
fn malformed_router_and_nonfinite_values_return_errors() {
    let linear = |input, output, weights, bias| MlpRouter {
        layers: vec![RouterLayer::Linear(LinearLayer {
            in_features: input,
            out_features: output,
            weights,
            bias,
        })],
    };
    let invalid = vec![
        MlpRouter { layers: vec![] },
        MlpRouter {
            layers: vec![RouterLayer::ReLU],
        },
        linear(0, 1, vec![], vec![0.0]),
        linear(2, 0, vec![], vec![]),
        linear(2, 1, vec![1.0], vec![0.0]),
        linear(2, 1, vec![1.0, 1.0], vec![]),
        linear(usize::MAX, 2, vec![], vec![]),
        linear(2, 1, vec![f32::NAN, 0.0], vec![0.0]),
        linear(2, 1, vec![0.0, 0.0], vec![f32::INFINITY]),
    ];
    for r in invalid {
        assert!(r.validate().is_err());
        assert!(r.forward(&[1.0, 0.0]).is_err());
    }
    let mut mismatch = router();
    if let RouterLayer::Linear(l) = &mut mismatch.layers[2] {
        l.in_features = 3;
    }
    assert!(mismatch.validate().is_err());
    let mut trailing = router();
    trailing.layers.push(RouterLayer::ReLU);
    assert!(trailing.validate().is_err());
    for q in [vec![1.0], vec![f32::NAN, 0.0], vec![0.0, f32::INFINITY]] {
        assert!(router().forward(&q).is_err());
    }
    let overflow = linear(2, 1, vec![f32::MAX, 0.0], vec![0.0]);
    assert!(overflow.forward(&[2.0, 0.0]).is_err());
    // ReLU must not conceal a nonfinite intermediate activation.
    let mut hidden = router();
    if let RouterLayer::Linear(l) = &mut hidden.layers[0] {
        l.weights[0] = f32::MAX;
    }
    assert!(hidden.forward(&[2.0, 0.0]).is_err());
    assert!(LmiRoutingState::new(router(), vec![vec![]; 2], 1).is_err());
    assert!(LmiRoutingState::new(router(), vec![vec![]; 3], 0).is_err());
    assert!(LmiRoutingState::new(router(), vec![vec![]; 3], 4).is_err());
    println!("validation: malformed dimensions/parameters, nonfinite input and overflow rejected");
}

#[test]
fn database_postings_are_built_from_router_predictions() {
    use segment::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
    let mut storage = new_volatile_dense_vector_storage(2, Distance::Dot);
    let hw = HardwareCounterCell::new();
    for (id, v) in [[1.0_f32, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]]
        .iter()
        .enumerate()
    {
        storage
            .insert_vector(id as u32, v.as_slice().into(), &hw)
            .unwrap();
    }
    let postings = build_router_postings(&router(), &storage, &AtomicBool::new(false)).unwrap();
    assert_eq!(postings, vec![vec![0, 1], vec![3], vec![2]]);
    println!("router-built postings = {postings:?}");
    storage.delete_vector(1).unwrap();
    assert_eq!(
        build_router_postings(&router(), &storage, &AtomicBool::new(false)).unwrap(),
        vec![vec![0], vec![3], vec![2]]
    );
    let wrong=segment::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage(3,Distance::Dot);
    let mut wrong = wrong;
    wrong
        .insert_vector(0, (&[1.0_f32, 0.0, 0.0][..]).into(), &hw)
        .unwrap();
    assert!(build_router_postings(&router(), &wrong, &AtomicBool::new(false)).is_err());
}

#[test]
fn nprobe_unions_deduplicates_and_expands_candidates() {
    let f = fixture(Distance::Dot);
    install(&f.lmi, 1);
    let postings = installed_postings(&f.lmi).unwrap();
    let q = [1.0, 0.0].into();
    let mut previous = Vec::new();
    for n in 1..=3 {
        let state = LmiRoutingState::new(router(), postings.clone(), n).unwrap();
        let candidates = state
            .candidates_for_query(&q, &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        assert!(previous.iter().all(|id| candidates.contains(id)));
        assert_eq!(candidates.len(), [2, 3, 4][n - 1]);
        println!("nprobe={n} candidates={candidates:?}");
        previous = candidates;
    }
    // Deliberately overlapping synthetic postings test union robustness only;
    // the database-assignment test above builds postings through inference.
    let overlapping = vec![vec![0, 1, 1], vec![1, 3], vec![2, 0]];
    let state = LmiRoutingState::new(router(), overlapping, 3).unwrap();
    assert_eq!(
        state
            .candidates_for_query(&q, &AtomicBool::new(false))
            .unwrap()
            .unwrap(),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn learned_batch_routes_independently_into_native_top_k() {
    let f = fixture(Distance::Dot);
    install(&f.lmi, 1);
    assert_eq!(
        installed_postings(&f.lmi).unwrap(),
        vec![vec![0, 1], vec![3], vec![2]]
    );
    let q1 = [1.0, 0.0].into();
    let q2 = [-1.0, 0.0].into();
    let vc = VectorQueryContext::default();
    let result = search(&f.lmi, &[&q1, &q2], 10, None, None, &vc);
    assert_eq!(
        result,
        vec![
            vec![
                ScoredPointOffset { idx: 0, score: 1.0 },
                ScoredPointOffset { idx: 1, score: 0.8 }
            ],
            vec![ScoredPointOffset { idx: 3, score: 1.0 }]
        ]
    );
    for (i, q) in [&q1, &q2].into_iter().enumerate() {
        assert_eq!(result[i], search(&f.lmi, &[q], 10, None, None, &vc)[0]);
    }
    println!("learned batch nprobe=1 = {result:?}");
    install(&f.lmi, 2);
    assert_eq!(
        ids(&search(&f.lmi, &[&q1, &q2], 10, None, None, &vc)),
        vec![vec![0, 1, 2], vec![3, 2]]
    );
    assert_eq!(
        ids(&search(&f.lmi, &[&q1, &q2], 1, None, None, &vc)),
        vec![vec![0], vec![3]]
    );
}

#[test]
fn pruning_can_miss_global_best_but_candidate_scores_are_exact() {
    let f = fixture_vectors(
        &[[10.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]],
        Distance::Dot,
    );
    install(&f.lmi, 1);
    let q = [0.5, 1.0].into();
    let vc = VectorQueryContext::default();
    let actual = search(&f.lmi, &[&q], 1, None, None, &vc);
    let global = search(&f.plain, &[&q], 1, None, None, &vc);
    assert_eq!(actual, vec![vec![ScoredPointOffset { idx: 2, score: 1.0 }]]);
    assert_eq!(global, vec![vec![ScoredPointOffset { idx: 0, score: 5.0 }]]);
    println!("pruning: routed={actual:?}; Plain global={global:?}");
    install(&f.lmi, 3);
    assert_eq!(search(&f.lmi, &[&q], 1, None, None, &vc), global);
}

#[test]
fn all_buckets_match_plain_for_four_metrics_and_limits() {
    for distance in [
        Distance::Dot,
        Distance::Cosine,
        Distance::Euclid,
        Distance::Manhattan,
    ] {
        let f = fixture(distance);
        install(&f.lmi, 3);
        let q1 = [1.0, 0.0].into();
        let q2 = [-1.0, 0.0].into();
        let vc = VectorQueryContext::default();
        for top in [0, 1, 2, 10] {
            assert_eq!(
                search(&f.lmi, &[&q1, &q2], top, None, None, &vc),
                search(&f.plain, &[&q1, &q2], top, None, None, &vc)
            );
        }
        assert!(search(&f.lmi, &[], 10, None, None, &vc).is_empty());
        println!("all buckets = Plain: {distance:?}, top=[0,1,2,10]");
    }
}

#[test]
fn static_postings_reuse_point_vector_and_context_validity() {
    let f = fixture(Distance::Dot);
    install(&f.lmi, 1);
    f.lmi.id_tracker.borrow_mut().drop(1.into()).unwrap();
    f.lmi.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow_mut()
        .delete_vector(3)
        .unwrap();
    let q1 = [1.0, 0.0].into();
    let q2 = [-1.0, 0.0].into();
    assert_eq!(
        ids(&search(
            &f.lmi,
            &[&q1, &q2],
            10,
            None,
            None,
            &VectorQueryContext::default()
        )),
        vec![vec![1], vec![]]
    );
    let mut mask = BitVec::repeat(false, 4);
    mask.set(1, true);
    let qc = QueryContext::default();
    let sc = qc.get_segment_query_context().with_deleted_points(&mask);
    let vc = sc.get_vector_context(DEFAULT_VECTOR_NAME, None);
    let result = search(&f.lmi, &[&q1, &q2], 10, None, None, &vc);
    assert!(result.iter().all(Vec::is_empty));
    println!("static postings retain deleted offsets; scoring returns {result:?}");
}

#[test]
fn static_mode_retains_plain_fallbacks_and_safe_missing_state() {
    let f = fixture(Distance::Dot);
    let q = [1.0, 0.0].into();
    let vc = VectorQueryContext::default();
    mode(&f.lmi, LmiCandidateMode::StaticLearned);
    assert!(installed_postings(&f.lmi).is_none());
    assert_eq!(
        search(&f.lmi, &[&q], 10, None, None, &vc),
        search(&f.plain, &[&q], 10, None, None, &vc)
    );
    install(&f.lmi, 1);
    let complex = QueryVector::RecommendBestScore(RecoQuery::new(
        vec![vec![1.0, 0.0].into()],
        vec![vec![-1.0, 0.0].into()],
    ));
    let result = search(&f.lmi, &[&q, &complex], 10, None, None, &vc);
    assert_eq!(ids(&result)[0], vec![0, 1]);
    assert_eq!(
        result[1],
        search(&f.plain, &[&complex], 10, None, None, &vc)[0]
    );
    let filter = Filter::new_must(Condition::HasId(
        [4.into()]
            .into_iter()
            .collect::<ahash::AHashSet<_>>()
            .into(),
    ));
    assert_eq!(
        search(&f.lmi, &[&q], 10, Some(&filter), None, &vc),
        search(&f.plain, &[&q], 10, Some(&filter), None, &vc)
    );
    let params = SearchParams::default();
    assert_eq!(
        search(&f.lmi, &[&q], 10, None, Some(&params), &vc),
        search(&f.plain, &[&q], 10, None, Some(&params), &vc)
    );
}

#[test]
fn reopen_drops_learned_state_and_restores_exact_default() {
    let f = fixture(Distance::Dot);
    install(&f.lmi, 1);
    let q = [1.0, 0.0].into();
    let vc = VectorQueryContext::default();
    assert_eq!(
        ids(&search(&f.lmi, &[&q], 10, None, None, &vc)),
        vec![vec![0, 1]]
    );
    let path = f.lmi.segment_path.clone();
    let uuid = f.lmi.uuid;
    drop(f.lmi);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false)).unwrap();
    assert!(installed_postings(&reopened).is_none());
    assert_eq!(
        search(&reopened, &[&q], 10, None, None, &vc),
        search(&f.plain, &[&q], 10, None, None, &vc)
    );
    install(&reopened, 1);
    assert_eq!(
        ids(&search(&reopened, &[&q], 10, None, None, &vc)),
        vec![vec![0, 1]]
    );
    println!("reopen: no routing state; exact default restored; explicit reinstall works");
}

#[test]
fn installation_is_atomic_and_routing_honors_cancellation() {
    let f = fixture(Distance::Dot);
    install(&f.lmi, 1);
    let before = installed_postings(&f.lmi).unwrap();
    {
        let mut index = f.lmi.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow_mut();
        let VectorIndexEnum::Lmi(lmi) = &mut *index else {
            panic!()
        };
        assert!(
            lmi.install_static_routing(router(), 0, &AtomicBool::new(false))
                .is_err()
        );
        assert!(
            lmi.install_static_routing(router(), 4, &AtomicBool::new(false))
                .is_err()
        );
        let mut wrong = router();
        if let RouterLayer::Linear(layer) = &mut wrong.layers[0] {
            layer.in_features = 3;
            layer.weights = vec![0.0; 12];
        }
        assert!(
            lmi.install_static_routing(wrong, 1, &AtomicBool::new(false))
                .is_err()
        );
        assert!(
            lmi.install_static_routing(router(), 2, &AtomicBool::new(true))
                .is_err()
        );
        assert_eq!(lmi.routing_state().unwrap().nprobe(), 1);
    }
    assert_eq!(installed_postings(&f.lmi).unwrap(), before);
    let qc = QueryContext::default().with_is_stopped(Arc::new(AtomicBool::new(true)));
    let sc = qc.get_segment_query_context();
    let vc = sc.get_vector_context(DEFAULT_VECTOR_NAME, None);
    let q = [1.0, 0.0].into();
    assert!(
        f.lmi.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[&q], None, 10, None, &vc)
            .is_err()
    );
    let state = LmiRoutingState::new(router(), before, 1).unwrap();
    assert!(
        state
            .candidates_for_query(&q, &AtomicBool::new(true))
            .is_err()
    );
    let empty = fixture_vectors(&[], Distance::Dot);
    install(&empty.lmi, 1);
    assert_eq!(
        installed_postings(&empty.lmi).unwrap(),
        vec![Vec::<u32>::new(); 3]
    );
    assert!(
        search(
            &empty.lmi,
            &[&q],
            10,
            None,
            None,
            &VectorQueryContext::default()
        )[0]
        .is_empty()
    );
}
#[test]
fn unsupported_storage_and_bad_dense_queries_fail_clearly() {
    use segment::data_types::vectors::MultiDenseVectorInternal;
    use segment::vector_storage::multi_dense::volatile_multi_dense_vector_storage::new_volatile_multi_dense_vector_storage;
    let mut storage = new_volatile_multi_dense_vector_storage(2, Distance::Dot, Default::default());
    let multi = MultiDenseVectorInternal::new_unchecked(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    storage
        .insert_vector(0, (&multi).into(), &HardwareCounterCell::new())
        .unwrap();
    let error = build_router_postings(&router(), &storage, &AtomicBool::new(false)).unwrap_err();
    assert!(error.to_string().contains("dense vector storage"));
    let f = fixture(Distance::Dot);
    install(&f.lmi, 1);
    let vc = VectorQueryContext::default();
    for query in [vec![1.0].into(), vec![f32::NAN, 0.0].into()] {
        assert!(
            f.lmi.vector_data[DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow()
                .search(&[&query], None, 10, None, &vc)
                .is_err()
        );
    }
    println!("unsupported multivector installation and malformed dense queries return errors");
}
