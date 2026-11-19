use std::sync::atomic::AtomicBool;

use common::flags::FeatureFlags;
use common::universal_io::{MmapFile, MmapFs};
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::id_tracker::IdTracker;
use segment::index::lmi_index::{
    CompactPostings, LinearLayer, LmiConfig, LmiRoutingState, LmiStateMetadata, MlpRouter,
    RouterLayer, save_fixture_state,
};
use segment::index::{VectorIndex, VectorIndexRead};
use segment::segment::Segment;
use segment::segment::read_only::ReadOnlySegment;
use segment::segment_constructor::load_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Condition, Distance, Filter, HnswGlobalConfig, Indexes, SearchParams, VectorStorageDatatype,
};
use segment::vector_storage::VectorStorageRead;

fn router() -> MlpRouter {
    MlpRouter {
        layers: vec![
            RouterLayer::Linear(LinearLayer {
                in_features: 2,
                out_features: 2,
                weights: vec![1.0, 0.0, 0.0, 1.0],
                bias: vec![0.0, 0.0],
            }),
            RouterLayer::ReLU,
            RouterLayer::Linear(LinearLayer {
                in_features: 2,
                out_features: 2,
                weights: vec![1.0, 0.0, 0.0, 1.0],
                bias: vec![0.0, 0.0],
            }),
        ],
    }
}

fn fixture_with_datatype(
    datatype: Option<VectorStorageDatatype>,
) -> (tempfile::TempDir, Segment, Segment, LmiConfig) {
    let root = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(root.path(), 2, Distance::Dot).unwrap();
    if let Some(datatype) = datatype {
        let mut cfg = plain.config().clone();
        cfg.vector_data
            .get_mut(DEFAULT_VECTOR_NAME)
            .unwrap()
            .datatype = Some(datatype);
        drop(plain);
        plain = segment::segment_constructor::build_segment(root.path(), &cfg, None, true)
            .unwrap()
            .0;
    }
    for (i, vector) in [[10.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]
        .iter()
        .enumerate()
    {
        plain
            .upsert_point(
                i as u64 + 1,
                (i as u64 + 1).into(),
                only_default_vector(vector),
            )
            .unwrap();
    }
    let mut cfg = plain.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::Lmi {};
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder.update(&[&plain], &AtomicBool::new(false)).unwrap();
    let lmi = builder.build_for_test(root.path());
    let trained = LmiConfig {
        n_buckets: 2,
        sample_size: 4,
        hidden_dim: 2,
        nprobe: 1,
        ..Default::default()
    };
    (root, plain, lmi, trained)
}

fn fixture() -> (tempfile::TempDir, Segment, Segment, LmiConfig) {
    fixture_with_datatype(None)
}

fn install_state(lmi: &Segment, config: LmiConfig) {
    let vector_config = lmi.config().vector_data[DEFAULT_VECTOR_NAME].clone();
    let storage = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
    let metadata = LmiStateMetadata::new(
        config,
        &vector_config,
        storage.datatype(),
        storage.total_vector_count(),
    );
    drop(storage);
    let routing = LmiRoutingState::from_compact(
        router(),
        CompactPostings::from_buckets(vec![vec![0, 2, 3], vec![1]]).unwrap(),
        config.nprobe,
    )
    .unwrap();
    let index_path = lmi.segment_path.join("vector_index");
    save_fixture_state(&index_path, &metadata, &routing, &vector_config).unwrap();
    let mut state = Segment::load_state(&lmi.segment_path).unwrap();
    state
        .config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .index = Indexes::LmiTrained(config);
    Segment::save_state(&state, &lmi.segment_path).unwrap();
}

/// Top-1 search: the learned bucket for this query holds one live point, so a
/// filled learned answer (offset 1) differs from the exact answer (offset 0).
fn ids(segment: &Segment) -> Vec<u32> {
    ids_top(segment, 1)
}

fn ids_top(segment: &Segment, top: usize) -> Vec<u32> {
    let q: QueryVector = [0.8_f32, 1.0].into();
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[&q], None, top, None, &VectorQueryContext::default())
        .unwrap()[0]
        .iter()
        .map(|p| p.idx)
        .collect()
}

#[test]
fn native_router_logits_ties_and_invalid_inputs() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let model = router();
    assert_eq!(model.forward(&[0.8, 1.0]).unwrap(), vec![0.8, 1.0]);
    assert_eq!(model.top_buckets(&[0.8, 1.0], 2).unwrap(), vec![1, 0]);
    assert_eq!(model.top_buckets(&[1.0, 1.0], 2).unwrap(), vec![0, 1]);
    assert!(model.top_buckets(&[1.0], 1).is_err());
    assert!(model.top_buckets(&[f32::NAN, 0.0], 1).is_err());
    assert!(model.top_buckets(&[0.0, 0.0], 0).is_err());
    assert!(model.top_buckets(&[0.0, 0.0], 3).is_err());
    let mut invalid = model;
    if let RouterLayer::Linear(layer) = &mut invalid.layers[0] {
        layer.weights.pop();
    }
    assert!(invalid.validate().is_err());
}

#[test]
fn compact_postings_rejects_bad_boundaries() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let postings = CompactPostings::from_buckets(vec![vec![0, 3], vec![], vec![1]]).unwrap();
    assert_eq!(postings.get(0), Some(&[0, 3][..]));
    assert_eq!(postings.get(1), Some(&[][..]));
    assert_eq!(postings.get(2), Some(&[1][..]));
    assert!(CompactPostings::from_parts(vec![0, 2, 1], vec![0, 1]).is_err());
    assert!(CompactPostings::from_parts(vec![0, 3], vec![0, 1]).is_err());
}

#[test]
fn persisted_static_search_restarts_without_training() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, plain, lmi, config) = fixture();
    install_state(&lmi, config);
    let path = lmi.segment_path.clone();
    let uuid = lmi.uuid;
    drop(lmi);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    assert_ne!(ids(&plain)[0], ids(&reopened)[0]);
    assert_eq!(ids(&reopened)[0], 1);
    let q: QueryVector = [0.8_f32, 1.0].into();
    let search = |params: Option<&SearchParams>, filter: Option<&Filter>, top: usize| {
        reopened.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[&q], filter, top, params, &VectorQueryContext::default())
            .unwrap()[0]
            .iter()
            .map(|p| p.idx)
            .collect::<Vec<_>>()
    };
    // A filled top-1 learned answer differs from the exact answer.
    assert_eq!(search(Some(&SearchParams::default()), None, 1)[0], 1);
    assert_eq!(
        search(
            Some(&SearchParams {
                exact: true,
                ..Default::default()
            }),
            None,
            2
        )[0],
        0
    );
    let filter = Filter::new_must(Condition::HasId(
        [1.into()]
            .into_iter()
            .collect::<ahash::AHashSet<_>>()
            .into(),
    ));
    assert_eq!(search(None, Some(&filter), 2), vec![0]);
    let files = reopened.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .files();
    assert_eq!(files.len(), 3);
    assert!(files.iter().all(|file| file.exists()));
    drop(reopened);
    let second = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    assert_eq!(ids(&second)[0], 1);
    let read_only = ReadOnlySegment::<MmapFile>::open(&MmapFs, &path, uuid, None, None).unwrap();
    let q: QueryVector = [0.8_f32, 1.0].into();
    let results = read_only.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[&q], None, 1, None, &VectorQueryContext::default())
        .unwrap();
    assert_eq!(results[0][0].idx, 1);
    // The read-only index shares the fill invariant: top 3 widens to exact.
    let widened: Vec<u32> = read_only.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[&q], None, 3, None, &VectorQueryContext::default())
        .unwrap()[0]
        .iter()
        .map(|p| p.idx)
        .collect();
    assert_eq!(widened, ids_top(&plain, 3));
}

#[test]
fn candidate_union_and_probe_count_are_deterministic() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let postings = CompactPostings::from_buckets(vec![vec![3, 1], vec![1, 2]]).unwrap();
    let query: QueryVector = [1.0_f32, 0.0].into();
    let stopped = AtomicBool::new(false);
    let one = LmiRoutingState::from_compact(router(), postings.clone(), 1).unwrap();
    let two = LmiRoutingState::from_compact(router(), postings, 2).unwrap();
    assert_eq!(
        one.candidates_for_query(&query, &stopped).unwrap().unwrap(),
        vec![1, 3]
    );
    assert_eq!(
        two.candidates_for_query(&query, &stopped).unwrap().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
/// Corrupt or mismatched derived LMI state makes segment load fail (fail closed),
/// matching upstream semantics for corrupt derived indexes. Previously named
/// `corrupt_persisted_state_fails_open`, which described the opposite behavior.
fn corrupt_persisted_state_rejects_segment_load() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    for case in [
        "missing_router",
        "missing_postings",
        "truncated_router",
        "truncated_postings",
        "bucket_mismatch",
        "dimension_mismatch",
        "metric_mismatch",
        "preprocessing_mismatch",
        "unsupported_version",
    ] {
        let (_root, plain, lmi, config) = fixture();
        install_state(&lmi, config);
        let index_path = lmi.segment_path.join("vector_index");
        let metadata_path = index_path.join("lmi_state.json");
        match case {
            "missing_router" => std::fs::remove_file(index_path.join("lmi_router.bin")).unwrap(),
            "missing_postings" => {
                std::fs::remove_file(index_path.join("lmi_postings.bin")).unwrap()
            }
            "truncated_router" => {
                std::fs::write(index_path.join("lmi_router.bin"), [0_u8; 3]).unwrap()
            }
            "truncated_postings" => {
                std::fs::write(index_path.join("lmi_postings.bin"), [0_u8; 3]).unwrap()
            }
            other => {
                let mut value: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
                match other {
                    "bucket_mismatch" => value["config"]["n_buckets"] = 3.into(),
                    "dimension_mismatch" => value["dimension"] = 3.into(),
                    "metric_mismatch" => value["distance"] = "Cosine".into(),
                    "preprocessing_mismatch" => value["preprocessing"] = "cosine_normalized".into(),
                    "unsupported_version" => value["version"] = 999.into(),
                    _ => unreachable!(),
                }
                std::fs::write(&metadata_path, serde_json::to_vec(&value).unwrap()).unwrap();
            }
        }
        let path = lmi.segment_path.clone();
        let uuid = lmi.uuid;
        drop(lmi);
        drop(plain);
        assert!(
            load_segment(&path, uuid, None, &AtomicBool::new(false), true).is_err(),
            "{case}"
        );
    }
}

#[test]
fn float16_static_learned_search_uses_real_storage() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, _plain, lmi, config) = fixture_with_datatype(Some(VectorStorageDatatype::Float16));
    install_state(&lmi, config);
    let path = lmi.segment_path.clone();
    let uuid = lmi.uuid;
    drop(lmi);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    assert_eq!(ids(&reopened)[0], 1);
    assert_eq!(
        reopened.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow()
            .datatype(),
        VectorStorageDatatype::Float16
    );
}

#[test]
fn hand_computable_two_three_four_router() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let model = MlpRouter {
        layers: vec![
            RouterLayer::Linear(LinearLayer {
                in_features: 2,
                out_features: 3,
                weights: vec![1.0, 0.0, 0.0, 1.0, -1.0, 1.0],
                bias: vec![0.0; 3],
            }),
            RouterLayer::ReLU,
            RouterLayer::Linear(LinearLayer {
                in_features: 3,
                out_features: 4,
                weights: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                bias: vec![0.0; 4],
            }),
        ],
    };
    assert_eq!(
        model.forward(&[2.0, 3.0]).unwrap(),
        vec![2.0, 3.0, 1.0, 6.0]
    );
    assert_eq!(model.top_buckets(&[2.0, 3.0], 4).unwrap(), vec![3, 1, 0, 2]);
    let mut corrupt = model;
    if let RouterLayer::Linear(layer) = &mut corrupt.layers[2] {
        layer.bias[0] = f32::NAN;
    }
    assert!(corrupt.validate().is_err());
    assert!(
        LmiRoutingState::from_compact(
            router(),
            CompactPostings::from_buckets(vec![vec![1]]).unwrap(),
            1
        )
        .is_err()
    );
}

#[test]
fn deleted_static_candidate_is_filtered_by_current_scorer() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, _plain, lmi, config) = fixture();
    install_state(&lmi, config);
    let path = lmi.segment_path.clone();
    let uuid = lmi.uuid;
    drop(lmi);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    assert_eq!(ids(&reopened), vec![1]);
    reopened.id_tracker.borrow_mut().drop(2.into()).unwrap();
    // The only posting in the learned bucket is now deleted. The search must
    // not return an empty result while live points exist: the probe widens and
    // the deleted offset is still excluded by the current scorer.
    assert_eq!(ids(&reopened), vec![0]);
}

/// Regression: an unfiltered static LMI search must return `min(top, live)`
/// hits. The selected bucket holds one posting; `top = 3` previously returned
/// a single hit from a four-point segment. Widened results equal exact search.
#[test]
fn underfilled_learned_probe_widens_to_fill_top() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, plain, lmi, config) = fixture();
    install_state(&lmi, config);
    let path = lmi.segment_path.clone();
    let uuid = lmi.uuid;
    drop(lmi);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    let exact = ids_top(&plain, 3);
    assert_eq!(exact.len(), 3);
    assert_eq!(ids_top(&reopened, 3), exact);
    let index = reopened.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let segment::index::VectorIndexEnum::Lmi(lmi_index) = &*index else {
        panic!("expected LMI index")
    };
    assert_eq!(lmi_index.underfill_widenings_for_test(), 1);
    // A filled top-1 answer takes the learned route without widening.
    drop(index);
    assert_eq!(ids_top(&reopened, 1), vec![1]);
    // After a deletion the live count is 3: top = 4 fills to 3, never 4.
    reopened.id_tracker.borrow_mut().drop(4.into()).unwrap();
    let widened = ids_top(&reopened, 4);
    assert_eq!(widened.len(), 3);
    assert!(!widened.contains(&3), "deleted offset must stay excluded");
}

fn install_router(lmi: &Segment, config: LmiConfig, router: MlpRouter) {
    let vector_config = lmi.config().vector_data[DEFAULT_VECTOR_NAME].clone();
    let storage = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
    let metadata = LmiStateMetadata::new(
        config,
        &vector_config,
        storage.datatype(),
        storage.total_vector_count(),
    );
    drop(storage);
    let routing = LmiRoutingState::from_compact(
        router,
        CompactPostings::from_buckets(vec![vec![0, 2, 3], vec![1]]).unwrap(),
        config.nprobe,
    )
    .unwrap();
    let index_path = lmi.segment_path.join("vector_index");
    save_fixture_state(&index_path, &metadata, &routing, &vector_config).unwrap();
    let mut state = Segment::load_state(&lmi.segment_path).unwrap();
    state
        .config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .index = Indexes::LmiTrained(config);
    Segment::save_state(&state, &lmi.segment_path).unwrap();
}

/// The supervised-routing artifact shape: one non-centroid `Linear` layer with biases.
/// It persists, reopens without retraining on both open paths, ranks buckets by
/// `bias + W q`, and breaks equal logits toward the smaller bucket id.
#[test]
fn single_linear_router_with_bias_persists_and_ranks_deterministically() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let linear = |bias1: f32| MlpRouter {
        layers: vec![RouterLayer::Linear(LinearLayer {
            in_features: 2,
            out_features: 2,
            // Both rows score q[0]; only the bias separates the buckets.
            weights: vec![1.0, 0.0, 1.0, 0.0],
            bias: vec![0.0, bias1],
        })],
    };
    for (bias1, expected_bucket, expected_top1) in [(0.5_f32, 1_usize, 1_u32), (0.0, 0, 0)] {
        let (_root, _plain, lmi, config) = fixture();
        install_router(&lmi, config, linear(bias1));
        let (path, uuid) = (lmi.segment_path.clone(), lmi.uuid);
        drop(lmi);
        let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
        {
            let index = reopened.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
            let segment::index::VectorIndexEnum::Lmi(lmi_index) = &*index else {
                panic!("expected LMI index")
            };
            let state = lmi_index.routing_state().expect("persisted learned state");
            assert_eq!(state.router(), &linear(bias1), "router reopened bit-identically");
            assert_eq!(
                state.router().top_buckets(&[0.8, 1.0], 2).unwrap(),
                if expected_bucket == 1 { vec![1, 0] } else { vec![0, 1] },
            );
        }
        // nprobe 1 searches only the top bucket: offset 1 alone, or the best of {0, 2, 3}.
        assert_eq!(ids(&reopened), vec![expected_top1]);
        let read_only =
            ReadOnlySegment::<MmapFile>::open(&MmapFs, &path, uuid, None, None).unwrap();
        let q: QueryVector = [0.8_f32, 1.0].into();
        let shared = read_only.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[&q], None, 1, None, &VectorQueryContext::default())
            .unwrap();
        assert_eq!(shared[0][0].idx, expected_top1);
    }
}
