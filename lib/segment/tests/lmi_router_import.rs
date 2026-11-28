#![cfg(all(feature = "lmi-training", feature = "testing"))]

//! Q4 arm (c) known answer: a router exported to LMI1 and imported back rebuilds identical postings.
//! A trained LMI segment's router is written in PostgreSQL's LMI1 layout, parsed and registered, and the segment is
//! rebuilt with `reuse_router` set to that router (no training). Every point id must land in the same bucket.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use common::ambient;
use common::flags::FeatureFlags;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
use segment::entry::ReadSegmentEntry;
use segment::entry::entry_point::{SegmentEntry, StorageSegmentEntry};
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexEnum;
use segment::index::lmi_index::research_import::{parse_lmi1, register_router};
use segment::index::lmi_index::{LmiConfig, MlpRouter, RouterLayer};
use segment::segment::Segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Distance, ExtendedPointId, HnswGlobalConfig, Indexes};

const DIM: usize = 16;

fn build(root: &Path, source: &Segment, model: LmiConfig) -> Segment {
    let mut cfg = source.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(model);
    let staging = tempfile::tempdir_in(root).unwrap();
    let mut builder =
        SegmentBuilder::new(staging.path(), &cfg, &HnswGlobalConfig::default(), FeatureFlags::default()).unwrap();
    builder.update(&[source], &AtomicBool::new(false)).unwrap();
    builder.build_for_test(root)
}

/// (router, point id -> bucket) of a trained LMI segment.
fn routing(segment: &Segment) -> (MlpRouter, BTreeMap<u64, usize>) {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else { panic!("not LMI") };
    let state = lmi.routing_state().unwrap();
    let tracker = segment.id_tracker.borrow();
    let mut map = BTreeMap::new();
    for bucket in 0..state.postings().len() {
        for &offset in state.postings().get(bucket).unwrap() {
            let ExtendedPointId::NumId(id) = tracker.external_id(offset).unwrap() else { panic!() };
            assert!(map.insert(id, bucket).is_none());
        }
    }
    (state.router().clone(), map)
}

fn to_lmi1(router: &MlpRouter) -> Vec<u8> {
    let [RouterLayer::Linear(a), RouterLayer::ReLU, RouterLayer::Linear(b)] = router.layers.as_slice() else {
        panic!("not the trained architecture")
    };
    let mut out = b"LMI1".to_vec();
    for v in [a.in_features, a.out_features, b.out_features] {
        out.extend((v as u32).to_le_bytes());
    }
    for v in a.weights.iter().chain(&a.bias).chain(&b.weights).chain(&b.bias) {
        out.extend(v.to_le_bytes());
    }
    out
}

#[test]
fn lmi1_export_import_rebuild_reproduces_postings() {
    let _scope = ambient::test_guard();
    let dir = tempfile::tempdir().unwrap();
    let mut source = build_simple_segment(&dir.path().join("plain"), DIM, Distance::Cosine).unwrap();
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    for id in 1..=3_000u64 {
        let v: Vec<f32> = (0..DIM)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % 2001) as f32 / 1000.0 - 1.0
            })
            .collect();
        source.upsert_point(id, id.into(), only_default_vector(&v)).unwrap();
    }
    source.flush(true).unwrap();
    let model = LmiConfig {
        n_buckets: 10, sample_size: 2_000, hidden_dim: 8, epochs: 5, batch_size: 128, routing_batch_size: 64,
        kmeans_iterations: 10, nprobe: 2, seed: 7, reuse_router: None,
    };
    let trained = build(dir.path(), &source, model);
    let (router, want) = routing(&trained);
    let imported = parse_lmi1(&to_lmi1(&router)).unwrap();
    assert_eq!(imported, router, "LMI1 round trip");
    let fingerprint = register_router(imported);
    let rebuilt = build(dir.path(), &source, LmiConfig { reuse_router: Some(fingerprint), ..model });
    let (router2, got) = routing(&rebuilt);
    assert_eq!(router2, router);
    assert_eq!(got.len(), 3_000);
    assert_eq!(got, want, "imported router must reproduce every posting");
}
