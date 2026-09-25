#![cfg(feature = "lmi-training")]

use common::budget::ResourcePermit;
use common::counter::hardware_counter::HardwareCounterCell;
use common::progress_tracker::ProgressTracker;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{
    NonAppendableSegmentEntry, ReadSegmentEntry, SegmentEntry, StorageSegmentEntry,
};
use segment::index::lmi_index::{LMI_STATE_FILE, LmiConfig};
use segment::index::{VectorIndex, VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::segment_constructor::{
    load_segment, segment_builder::SegmentBuilder, simple_segment_constructor::build_simple_segment,
};
use segment::types::{Distance, HnswGlobalConfig, Indexes, SearchParams};
use std::sync::atomic::AtomicBool;

fn config(nprobe: usize) -> LmiConfig {
    LmiConfig {
        n_buckets: 2,
        sample_size: 32,
        hidden_dim: 8,
        epochs: 60,
        batch_size: 8,
        kmeans_iterations: 8,
        nprobe,
        seed: 42,
    }
}

fn fixture(
    distance: Distance,
    config: LmiConfig,
    count: usize,
) -> (tempfile::TempDir, Segment, Segment) {
    let root = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(root.path(), 2, distance).unwrap();
    let hw = HardwareCounterCell::new();
    for i in 0..count {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let v = vec![sign * (2.0 + i as f32 / 100.0), 0.1 + (i % 5) as f32 / 20.0];
        plain
            .upsert_point(
                i as u64 + 1,
                (i as u64 + 1).into(),
                only_default_vector(&v),
                &hw,
            )
            .unwrap();
    }
    let mut cfg = plain.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(config);
    let mut builder =
        SegmentBuilder::new(staging.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    builder
        .update(&[&plain], &AtomicBool::new(false), &hw)
        .unwrap();
    let lmi = builder.build_for_test(root.path());
    (root, plain, lmi)
}

fn state(segment: &Segment) -> (std::path::PathBuf, serde_json::Value) {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected LMI")
    };
    let path = lmi.state_path().unwrap().to_owned();
    assert_eq!(path.file_name().unwrap(), LMI_STATE_FILE);
    let json = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    (path, json)
}

fn search(
    segment: &Segment,
    q: &[f32],
    params: Option<&SearchParams>,
) -> Vec<common::types::ScoredPointOffset> {
    let query = QueryVector::from(q.to_vec());
    let root_context = QueryContext::default();
    let context = root_context.get_segment_query_context();
    let context = context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[&query], None, 1000, params, &context)
        .unwrap()
        .remove(0)
}

#[test]
fn builder_trains_persists_and_reopens_without_manual_installation() {
    let (_root, _plain, lmi) = fixture(Distance::Dot, config(1), 64);
    let (path, json) = state(&lmi);
    assert!(!json["router"].is_null());
    assert_eq!(json["sample_offsets"].as_array().unwrap().len(), 32);
    let postings = json["postings"].as_array().unwrap();
    assert_eq!(
        postings
            .iter()
            .map(|p| p.as_array().unwrap().len())
            .sum::<usize>(),
        64
    );
    assert!(postings.iter().all(|p| !p.as_array().unwrap().is_empty()));
    let a = search(&lmi, &[3.0, 0.1], None);
    let b = search(&lmi, &[-3.0, 0.1], None);
    assert!(a.len() < 64 && b.len() < 64);
    assert_ne!(
        a.iter().map(|p| p.idx).collect::<Vec<_>>(),
        b.iter().map(|p| p.idx).collect::<Vec<_>>()
    );
    let bytes = std::fs::read(&path).unwrap();
    let directory = lmi.segment_path.clone();
    drop(lmi);
    let reopened =
        load_segment(&directory, uuid::Uuid::nil(), None, &AtomicBool::new(false)).unwrap();
    assert_eq!(search(&reopened, &[3.0, 0.1], None), a);
    assert_eq!(std::fs::read(path).unwrap(), bytes);
    println!("Phase E: automatic training, two routed candidate sets, unchanged state on reopen");
}

#[test]
fn all_buckets_equal_plain_and_default_params_do_not_bypass_learned_mode() {
    for distance in [
        Distance::Dot,
        Distance::Cosine,
        Distance::Euclid,
        Distance::Manhattan,
    ] {
        let (_root, plain, lmi) = fixture(distance, config(2), 32);
        for q in [[3.0, 0.2], [-3.0, 0.2]] {
            assert_eq!(search(&lmi, &q, None), search(&plain, &q, None));
        }
    }
    let (_root, plain, lmi) = fixture(Distance::Dot, config(1), 32);
    let q = [3.0, 0.2];
    let learned = search(&lmi, &q, None);
    assert!(learned.len() < 32);
    assert_eq!(search(&lmi, &q, Some(&SearchParams::default())), learned);
    let exact = SearchParams {
        exact: true,
        ..Default::default()
    };
    assert_eq!(search(&lmi, &q, Some(&exact)), search(&plain, &q, None));
    println!(
        "Phase E: full-bucket equality across four metrics; default params route; exact params scan"
    );
}

#[test]
fn cosine_router_uses_the_same_representation_as_training() {
    let (_root, _plain, lmi) = fixture(Distance::Cosine, config(1), 32);
    // Query scaling leaves candidate membership and cosine scores unchanged.
    let a = search(&lmi, &[3.0, 0.3], None);
    let b = search(&lmi, &[30.0, 3.0], None);
    assert_eq!(
        a.iter().map(|p| p.idx).collect::<Vec<_>>(),
        b.iter().map(|p| p.idx).collect::<Vec<_>>()
    );
    for (x, y) in a.iter().zip(b) {
        assert!((x.score - y.score).abs() < 1e-6);
    }
}

#[test]
fn seeded_builds_reproduce_samples_and_native_parameters() {
    let (_a, _p, a) = fixture(Distance::Dot, config(1), 40);
    let (_b, _p, b) = fixture(Distance::Dot, config(1), 40);
    assert_eq!(state(&a).1, state(&b).1);
}

#[test]
fn tiny_and_empty_segments_persist_exact_fallback() {
    for count in [0, 1] {
        let (_root, plain, lmi) = fixture(Distance::Dot, config(1), count);
        assert!(state(&lmi).1["router"].is_null());
        assert_eq!(
            search(&lmi, &[1.0, 0.0], None),
            search(&plain, &[1.0, 0.0], None)
        );
    }
}

#[test]
fn deletion_validity_survives_persisted_postings() {
    let (_root, _plain, mut lmi) = fixture(Distance::Dot, config(2), 16);
    let hw = HardwareCounterCell::new();
    lmi.delete_point(100, 1.into(), &hw).unwrap();
    let a = search(&lmi, &[1.0, 0.0], None);
    assert_eq!(a.len(), 15);
    lmi.flush(true).unwrap();
    let directory = lmi.segment_path.clone();
    drop(lmi);
    let reopened =
        load_segment(&directory, uuid::Uuid::nil(), None, &AtomicBool::new(false)).unwrap();
    assert_eq!(search(&reopened, &[1.0, 0.0], None), a);
}

#[test]
fn invalid_persisted_model_and_offsets_are_rejected() {
    for kind in [
        "version",
        "offset",
        "missing",
        "omitted",
        "sample",
        "model",
        "dropped_model",
    ] {
        let (_root, _plain, lmi) = fixture(Distance::Dot, config(1), 8);
        let (path, mut json) = state(&lmi);
        let directory = lmi.segment_path.clone();
        drop(lmi);
        match kind {
            "version" => {
                json["version"] = 999.into();
            }
            "offset" => {
                json["postings"][0] = serde_json::json!([999]);
            }
            "omitted" => {
                json["postings"] = serde_json::json!([[], []]);
            }
            "sample" => {
                json["sample_offsets"] = serde_json::json!([999]);
            }
            "dropped_model" => {
                json["router"] = serde_json::Value::Null;
                json["sample_offsets"] = serde_json::json!([]);
                json["postings"] = serde_json::json!([]);
            }
            "model" => {
                json["router"]["layers"][0]["Linear"]["weights"] = serde_json::json!([]);
            }
            _ => {}
        }
        if kind == "missing" {
            std::fs::remove_file(&path).unwrap();
        } else {
            std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        }
        assert!(
            load_segment(&directory, uuid::Uuid::nil(), None, &AtomicBool::new(false)).is_err()
        );
    }
}

#[test]
fn invalid_config_and_cancelled_build_return_errors() {
    assert!(
        LmiConfig {
            nprobe: 3,
            ..config(1)
        }
        .check()
        .is_err()
    );
    assert!(
        LmiConfig {
            sample_size: 1,
            ..config(1)
        }
        .check()
        .is_err()
    );
    let root = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(root.path(), 2, Distance::Dot).unwrap();
    let hw = HardwareCounterCell::new();
    plain
        .upsert_point(1, 1.into(), only_default_vector(&[1.0, 0.0]), &hw)
        .unwrap();
    let mut cfg = plain.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(config(1));
    let mut builder =
        SegmentBuilder::new(staging.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    builder
        .update(&[&plain], &AtomicBool::new(false), &hw)
        .unwrap();
    assert!(
        builder
            .build(
                root.path(),
                uuid::Uuid::new_v4(),
                None,
                ResourcePermit::dummy(1),
                &AtomicBool::new(true),
                &mut rand::rng(),
                &hw,
                ProgressTracker::new_for_test()
            )
            .is_err()
    );
}

#[test]
fn trained_multi_query_batch_uses_independent_candidates_and_partial_training_batch() {
    let (_root, _plain, lmi) = fixture(Distance::Dot, config(1), 17);
    let queries = [
        QueryVector::from(vec![3.0, 0.1]),
        QueryVector::from(vec![-3.0, 0.1]),
    ];
    let root = QueryContext::default();
    let context = root.get_segment_query_context();
    let context = context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    let index = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    assert!(index.files().contains(&state(&lmi).0));
    assert!(index.immutable_files().contains(&state(&lmi).0));
    let results = index
        .search(&[&queries[0], &queries[1]], None, 1000, None, &context)
        .unwrap();
    assert_eq!(results[0], search(&lmi, &[3.0, 0.1], None));
    assert_eq!(results[1], search(&lmi, &[-3.0, 0.1], None));
    assert!(!results[0].is_empty() && !results[1].is_empty());
    assert!(
        results[0]
            .iter()
            .all(|a| results[1].iter().all(|b| a.idx != b.idx))
    );
}

#[test]
fn persisted_index_rejects_transient_mode_changes() {
    let (_root, _plain, lmi) = fixture(Distance::Dot, config(1), 32);
    let before = search(&lmi, &[3.0, 0.1], None);
    let (path, _) = state(&lmi);
    let bytes = std::fs::read(&path).unwrap();
    {
        let mut index = lmi.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow_mut();
        let VectorIndexEnum::Lmi(index) = &mut *index else {
            panic!("expected LMI")
        };
        assert!(
            index
                .set_candidate_mode(segment::index::lmi_index::LmiCandidateMode::AllValidPoints)
                .is_err()
        );
    }
    assert_eq!(search(&lmi, &[3.0, 0.1], None), before);
    assert_eq!(std::fs::read(path).unwrap(), bytes);
}
