#![cfg(feature = "lmi-training")]
//! Small, deterministic Phase G filter/cardinality experiment.
use common::counter::hardware_counter::HardwareCounterCell;
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::index::lmi_index::LmiConfig;
use segment::index::{PayloadIndex, PayloadIndexRead, VectorIndexRead};
use segment::json_path::JsonPath;
use segment::payload_json;
use segment::segment_constructor::{
    segment_builder::SegmentBuilder, simple_segment_constructor::build_simple_segment,
};
use segment::types::{
    Condition, Distance, FieldCondition, Filter, HnswGlobalConfig, Indexes, PayloadSchemaType,
    SearchParams,
};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

#[test]
fn filtered_lmi_uses_plain_and_preserves_exact_results() {
    let root = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(root.path(), 2, Distance::Dot).unwrap();
    let hw = HardwareCounterCell::new();
    for i in 0..1000u64 {
        let v = vec![i as f32 / 1000.0, 1.0 - (i as f32) / 1000.0];
        plain
            .upsert_point(i + 1, (i + 1).into(), only_default_vector(&v), &hw)
            .unwrap();
        let payload = payload_json! {"all":1,"p50":(i%2==0) as i64,"p10":(i%10==0) as i64,"p1":(i%100==0) as i64,"p01":(i%1000==0) as i64};
        plain
            .set_full_payload(i + 1, (i + 1).into(), &payload, &hw)
            .unwrap();
    }
    for key in ["all", "p50", "p10", "p1", "p01"] {
        plain
            .payload_index
            .borrow_mut()
            .set_indexed(&JsonPath::new(key), PayloadSchemaType::Integer, &hw)
            .unwrap();
    }
    let mut cfg = plain.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(LmiConfig {
        n_buckets: 8,
        sample_size: 64,
        hidden_dim: 16,
        epochs: 5,
        batch_size: 16,
        routing_batch_size: 16,
        kmeans_iterations: 3,
        nprobe: 2,
        seed: 42,
    });
    let mut builder =
        SegmentBuilder::new(staging.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    builder
        .update(&[&plain], &AtomicBool::new(false), &hw)
        .unwrap();
    let lmi = builder.build_for_test(root.path());
    let query = QueryVector::from(vec![0.72, 0.28]);
    let root_context = QueryContext::default();
    let context = root_context.get_segment_query_context();
    let vc = context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    let mut evidence = Vec::new();
    for (key, expected) in [
        ("all", 1000),
        ("p50", 500),
        ("p10", 100),
        ("p1", 10),
        ("p01", 1),
    ] {
        let filter = Filter::new_must(Condition::Field(FieldCondition::new_match(
            JsonPath::new(key),
            1i64.into(),
        )));
        let estimate = lmi
            .payload_index
            .borrow()
            .with_view(|v| v.estimate_cardinality(&filter, &hw))
            .unwrap();
        let points = lmi
            .payload_index
            .borrow()
            .with_view(|v| v.query_points(&filter, &hw, &AtomicBool::new(false)))
            .unwrap();
        assert_eq!(points.len(), expected);
        let exact = SearchParams {
            exact: true,
            ..Default::default()
        };
        let truth = plain.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[&query], Some(&filter), 10, Some(&exact), &vc)
            .unwrap();
        let start = Instant::now();
        let actual = lmi.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow()
            .search(&[&query], Some(&filter), 10, None, &vc)
            .unwrap();
        let elapsed = start.elapsed().as_nanos();
        assert_eq!(
            actual, truth,
            "filtered LMI must equal exact Plain for {key}"
        );
        evidence.push(serde_json::json!({"filter":key,"actual_cardinality":expected,"estimated_min":estimate.min,"estimated_exp":estimate.exp,"estimated_max":estimate.max,"chosen_path":"Plain (LmiIndex::search filter fallback)","recall_at_10_vs_exact":1.0,"returned":actual[0].len(),"one_query_elapsed_ns_descriptive_only":elapsed}));
    }
    if let Ok(path) = std::env::var("LMI_G_FILTER_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    }
}
