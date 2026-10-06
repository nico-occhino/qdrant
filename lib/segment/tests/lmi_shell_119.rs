use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::flags::FeatureFlags;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::id_tracker::IdTracker;
use segment::index::lmi_index::LmiConfig;
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::segment_constructor::build_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Condition, Distance, Filter, HnswGlobalConfig, Indexes, SearchParams, VectorStorageDatatype,
    VectorStorageType,
};
use segment::vector_storage::VectorStorageRead;

fn fixture(datatype: Option<VectorStorageDatatype>) -> (tempfile::TempDir, Segment, Segment) {
    let root = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(root.path(), 2, Distance::Dot).unwrap();
    if let Some(datatype) = datatype {
        let mut source_config = plain.config().clone();
        source_config
            .vector_data
            .get_mut(DEFAULT_VECTOR_NAME)
            .unwrap()
            .datatype = Some(datatype);
        drop(plain);
        plain =
            segment::segment_constructor::build_segment(root.path(), &source_config, None, true)
                .unwrap()
                .0;
    }
    let hw = HardwareCounterCell::new();
    for (i, vector) in [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]]
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
    let mut config = plain.config().clone();
    config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .index = Indexes::Lmi {};
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &config,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder
        .update(&[&plain], &AtomicBool::new(false), &hw)
        .unwrap();
    let lmi = builder.build_for_test(root.path());
    (root, plain, lmi)
}

fn ids(segment: &Segment, filter: Option<&Filter>, params: Option<&SearchParams>) -> Vec<u32> {
    let query: QueryVector = [1.0_f32, 0.0].into();
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[&query], filter, 2, params, &VectorQueryContext::default())
        .unwrap()[0]
        .iter()
        .map(|point| point.idx)
        .collect()
}

#[test]
fn config_round_trip_retains_historical_fields() {
    let config = Indexes::LmiTrained(LmiConfig::default());
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("lmi_trained"));
    let reopened: Indexes = serde_json::from_str(&json).unwrap();
    assert_eq!(reopened, config);
    assert_eq!(
        serde_json::from_str::<Indexes>(r#"{"type":"lmi","options":{}}"#).unwrap(),
        Indexes::Lmi {}
    );
}

#[test]
fn synthetic_candidates_reach_lmi_runtime_variant() {
    let (_root, plain, lmi) = fixture(None);
    let index = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi_index) = &*index else {
        panic!("LMI-configured segment opened the wrong runtime index");
    };
    lmi_index.set_synthetic_candidates_for_test(vec![3, 1, 1]);
    drop(index);
    assert_eq!(ids(&lmi, None, None), vec![1, 3]);
    assert_eq!(ids(&plain, None, None), vec![0, 1]);
}

#[test]
fn exact_filter_and_nondefault_params_use_plain() {
    let (_root, plain, lmi) = fixture(None);
    let index = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi_index) = &*index else {
        panic!("expected LMI")
    };
    lmi_index.set_synthetic_candidates_for_test(vec![3, 1]);
    drop(index);
    let exact = SearchParams {
        exact: true,
        ..Default::default()
    };
    assert_eq!(
        ids(&lmi, None, Some(&exact)),
        ids(&plain, None, Some(&exact))
    );
    let nondefault = SearchParams {
        hnsw_ef: Some(16),
        ..Default::default()
    };
    assert_eq!(
        ids(&lmi, None, Some(&nondefault)),
        ids(&plain, None, Some(&nondefault))
    );
    let filter = Filter::new_must(Condition::HasId(
        [1.into(), 2.into()]
            .into_iter()
            .collect::<ahash::AHashSet<_>>()
            .into(),
    ));
    assert_eq!(
        ids(&lmi, Some(&filter), None),
        ids(&plain, Some(&filter), None)
    );
}

#[test]
fn deleted_candidate_and_float16_use_real_storage() {
    let (_root, _plain, lmi) = fixture(Some(VectorStorageDatatype::Float16));
    assert_eq!(
        lmi.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow()
            .datatype(),
        VectorStorageDatatype::Float16
    );
    let index = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi_index) = &*index else {
        panic!("expected LMI")
    };
    lmi_index.set_synthetic_candidates_for_test(vec![0, 1, 2]);
    drop(index);
    lmi.id_tracker.borrow_mut().drop(1.into()).unwrap();
    assert_eq!(ids(&lmi, None, None), vec![1, 2]);
}

#[test]
fn unported_trained_and_graph_inline_modes_fail_explicitly() {
    let source_root = tempfile::tempdir().unwrap();
    let source = build_simple_segment(source_root.path(), 2, Distance::Dot).unwrap();
    let mut config = source.config().clone();
    config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .index = Indexes::LmiTrained(LmiConfig::default());
    let target = tempfile::tempdir().unwrap();
    let error = build_segment(target.path(), &config, None, true)
        .err()
        .expect("trained LMI must fail");
    assert!(error.to_string().contains("not ported"));

    config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .index = Indexes::Lmi {};
    config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .storage_type = VectorStorageType::GraphInline;
    let target = tempfile::tempdir().unwrap();
    let error = build_segment(target.path(), &config, None, true).err().expect("GraphInline LMI must fail");
    assert!(error.to_string().contains("GraphInline"));
}
