#![cfg(feature = "lmi-training")]

use std::sync::atomic::AtomicBool;

use common::flags::FeatureFlags;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{NonAppendableSegmentEntry, ReadSegmentEntry, SegmentEntry};
use segment::index::lmi_index::{LmiConfig, LmiStateMetadata};
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::segment_constructor::load_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Distance, HnswGlobalConfig, Indexes, VectorStorageDatatype};

fn config() -> LmiConfig {
    LmiConfig {
        n_buckets: 2,
        sample_size: 8,
        hidden_dim: 4,
        epochs: 4,
        batch_size: 4,
        routing_batch_size: 3,
        kmeans_iterations: 5,
        nprobe: 1,
        seed: 42,
    }
}

fn build(datatype: Option<VectorStorageDatatype>) -> (tempfile::TempDir, Segment) {
    let root = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut source = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
    if let Some(datatype) = datatype {
        let mut cfg = source.config().clone();
        cfg.vector_data
            .get_mut(DEFAULT_VECTOR_NAME)
            .unwrap()
            .datatype = Some(datatype);
        drop(source);
        source = segment::segment_constructor::build_segment(root.path(), &cfg, None, true)
            .unwrap()
            .0;
    }
    let vectors = [
        [1.0, 0.0],
        [0.98, 0.2],
        [0.9, 0.4],
        [0.7, 0.7],
        [0.0, 1.0],
        [-0.2, 0.98],
        [-0.7, 0.7],
        [-1.0, 0.0],
        [-0.98, -0.2],
        [-0.7, -0.7],
        [0.0, -1.0],
        [0.7, -0.7],
    ];
    for (i, vector) in vectors.iter().enumerate() {
        source
            .upsert_point(
                (i + 1) as u64,
                ((i + 1) as u64).into(),
                only_default_vector(vector),
            )
            .unwrap();
    }
    // A deleted source point is excluded before building the target segment.
    source.delete_point(100, 3_u64.into()).unwrap();
    let mut cfg = source.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(config());
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder.update(&[&source], &AtomicBool::new(false)).unwrap();
    let trained = builder.build_for_test(root.path());
    (root, trained)
}

fn result(segment: &Segment) -> Vec<(u32, f32)> {
    let query: QueryVector = [1.0_f32, 0.1].into();
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[&query], None, 4, None, &VectorQueryContext::default())
        .unwrap()[0]
        .iter()
        .map(|p| (p.idx, p.score))
        .collect()
}

fn check_segment(segment: &Segment, expected_datatype: Option<VectorStorageDatatype>) {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected trained LMI")
    };
    let state = lmi
        .routing_state()
        .expect("production build must publish learned state");
    let offsets: Vec<_> = state.postings().iter().flatten().copied().collect();
    let mut unique = offsets.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), offsets.len());
    assert_eq!(offsets.len(), 11);
    assert_eq!(state.postings().len(), 2);
    let metadata: LmiStateMetadata = serde_json::from_slice(
        &std::fs::read(segment.segment_path.join("vector_index/lmi_state.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(metadata.indexed_live_count, Some(11));
    if let Some(datatype) = expected_datatype {
        assert_eq!(metadata.datatype, datatype);
    }
}

#[test]
fn production_float32_build_persists_and_reopens_without_training() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, built) = build(None);
    check_segment(&built, Some(VectorStorageDatatype::Float32));
    let before = result(&built);
    assert!(!before.is_empty());
    let path = built.segment_path.clone();
    let uuid = built.uuid;
    let router_file = path.join("vector_index/lmi_router.bin");
    let saved = std::fs::read(&router_file).unwrap();
    drop(built);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    check_segment(&reopened, Some(VectorStorageDatatype::Float32));
    assert_eq!(result(&reopened), before);
    assert_eq!(std::fs::read(router_file).unwrap(), saved);
}

#[test]
fn production_float16_build_decodes_bounded_samples_and_reopens() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root, built) = build(Some(VectorStorageDatatype::Float16));
    check_segment(&built, Some(VectorStorageDatatype::Float16));
    let before = result(&built);
    assert!(!before.is_empty());
    let path = built.segment_path.clone();
    let uuid = built.uuid;
    drop(built);
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    assert_eq!(result(&reopened), before);
}

#[test]
fn repeated_fixed_seed_builds_have_identical_router_and_postings() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI v1.20 regression"));
    let (_root_a, a) = build(None);
    let (_root_b, b) = build(None);
    for name in ["lmi_router.bin", "lmi_postings.bin"] {
        let left = std::fs::read(a.segment_path.join("vector_index").join(name)).unwrap();
        let right = std::fs::read(b.segment_path.join("vector_index").join(name)).unwrap();
        assert_eq!(left, right, "{name} changed under fixed seed");
    }
    assert_eq!(result(&a), result(&b));
}
