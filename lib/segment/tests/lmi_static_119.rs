use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
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
    let hw = HardwareCounterCell::new();
    for (i, vector) in [[10.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]
        .iter()
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
    let mut cfg = plain.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::Lmi {};
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder
        .update(&[&plain], &AtomicBool::new(false), &hw)
        .unwrap();
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

fn ids(segment: &Segment) -> Vec<u32> {
    let q: QueryVector = [0.8_f32, 1.0].into();
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[&q], None, 2, None, &VectorQueryContext::default())
        .unwrap()[0]
        .iter()
        .map(|p| p.idx)
        .collect()
}

#[test]
fn native_router_logits_ties_and_invalid_inputs() {
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
    let postings = CompactPostings::from_buckets(vec![vec![0, 3], vec![], vec![1]]).unwrap();
    assert_eq!(postings.get(0), Some(&[0, 3][..]));
    assert_eq!(postings.get(1), Some(&[][..]));
    assert_eq!(postings.get(2), Some(&[1][..]));
    assert!(CompactPostings::from_parts(vec![0, 2, 1], vec![0, 1]).is_err());
    assert!(CompactPostings::from_parts(vec![0, 3], vec![0, 1]).is_err());
}

#[test]
fn persisted_static_search_restarts_without_training() {
    let (_root, plain, lmi, config) = fixture();
    install_state(&lmi, config);
    let path = lmi.segment_path.clone();
    let uuid = lmi.uuid;
    drop(lmi);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    assert_ne!(ids(&plain)[0], ids(&reopened)[0]);
    assert_eq!(ids(&reopened)[0], 1);
    let q: QueryVector = [0.8_f32, 1.0].into();
    let search = |params: Option<&SearchParams>, filter: Option<&Filter>| {
        reopened.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[&q], filter, 2, params, &VectorQueryContext::default())
            .unwrap()[0]
            .iter()
            .map(|p| p.idx)
            .collect::<Vec<_>>()
    };
    assert_eq!(search(Some(&SearchParams::default()), None)[0], 1);
    assert_eq!(
        search(
            Some(&SearchParams {
                exact: true,
                ..Default::default()
            }),
            None
        )[0],
        0
    );
    let filter = Filter::new_must(Condition::HasId(
        [1.into()]
            .into_iter()
            .collect::<ahash::AHashSet<_>>()
            .into(),
    ));
    assert_eq!(search(None, Some(&filter)), vec![0]);
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
        .search(&[&q], None, 2, None, &VectorQueryContext::default())
        .unwrap();
    assert_eq!(results[0][0].idx, 1);
}

#[test]
fn candidate_union_and_probe_count_are_deterministic() {
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
fn corrupt_persisted_state_fails_open() {
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
    let (_root, _plain, lmi, config) = fixture();
    install_state(&lmi, config);
    let path = lmi.segment_path.clone();
    let uuid = lmi.uuid;
    drop(lmi);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    assert_eq!(ids(&reopened), vec![1]);
    reopened.id_tracker.borrow_mut().drop(2.into()).unwrap();
    assert!(ids(&reopened).is_empty());
}
