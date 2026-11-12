#![cfg(all(feature = "lmi-training", feature = "testing"))]

//! Research-only independent segment probing, not a DLI coordinator or publication protocol.
use common::flags::FeatureFlags;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{
    NonAppendableSegmentEntry, ReadSegmentEntry, SegmentEntry, StorageSegmentEntry,
};
use segment::id_tracker::IdTrackerRead;
use segment::index::lmi_index::LmiConfig;
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::segment_constructor::load_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Distance, HnswGlobalConfig, Indexes};
use std::sync::atomic::AtomicBool;

const DATA: [(u64, [f32; 2]); 9] = [
    (11, [1.0, 0.0]),
    (12, [0.8, 0.6]),
    (13, [0.0, 1.0]),
    (14, [-0.6, 0.8]),
    (21, [-1.0, 0.0]),
    (22, [-0.8, -0.6]),
    (23, [0.0, -1.0]),
    (24, [0.6, -0.8]),
    (31, [0.6, 0.8]),
];

fn lmi(root: &std::path::Path, rows: &[(u64, [f32; 2])], seed: u64) -> Segment {
    let staging = tempfile::tempdir().unwrap();
    let mut src = build_simple_segment(root, 2, Distance::Cosine).unwrap();
    for (id, vector) in rows {
        src.upsert_point(*id, (*id).into(), only_default_vector(vector))
            .unwrap();
    }
    let mut cfg = src.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(LmiConfig {
        n_buckets: 2,
        sample_size: 4,
        hidden_dim: 4,
        epochs: 1,
        batch_size: 4,
        routing_batch_size: 4,
        kmeans_iterations: 2,
        nprobe: 2,
        seed,
    });
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder.update(&[&src], &AtomicBool::new(false)).unwrap();
    builder.build_for_test(root)
}

fn search(s: &Segment, query: &QueryVector) -> Vec<(u64, f32)> {
    let data = &s.vector_data[DEFAULT_VECTOR_NAME];
    let results = data
        .vector_index
        .borrow()
        .search(&[query], None, 20, None, &VectorQueryContext::default())
        .unwrap();
    let tracker = s.id_tracker.borrow();
    results[0]
        .iter()
        .map(|p| {
            let segment::types::PointIdType::NumId(id) = tracker.external_id(p.idx).unwrap() else {
                panic!("unexpected UUID");
            };
            (id, p.score)
        })
        .collect()
}

#[test]
fn two_trained_lmi_levels_and_plain_buffer_match_exact_oracle_and_reopen() {
    let _scope =
        common::ambient::unmeasured_guard(common::reason::reason("DLI native topology research"));
    let root = tempfile::tempdir().unwrap();
    let mut level1 = lmi(root.path(), &DATA[..4], 17);
    let mut level2 = lmi(root.path(), &DATA[4..8], 29);
    let mut buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
    buffer
        .upsert_point(31, 31_u64.into(), only_default_vector(&DATA[8].1))
        .unwrap();
    let q: QueryVector = [1.0_f32, 0.0].into();
    for s in [&level1, &level2] {
        let idx = s.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
        let VectorIndexEnum::Lmi(index) = &*idx else {
            panic!("not LMI")
        };
        assert!(index.routing_state().is_some());
        assert_eq!(index.search_path_counts_for_test(), (0, 0));
    }
    let mut got: Vec<_> = [&level1, &level2, &buffer]
        .into_iter()
        .flat_map(|s| search(s, &q))
        .collect();
    for s in [&level1, &level2] {
        let idx = s.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
        let VectorIndexEnum::Lmi(index) = &*idx else {
            unreachable!()
        };
        assert_eq!(
            index.search_path_counts_for_test(),
            (1, 0),
            "learned path required"
        );
    }
    got.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut expected: Vec<_> = DATA
        .iter()
        .map(|(id, v)| (*id, v[0] / (v[0] * v[0] + v[1] * v[1]).sqrt()))
        .collect();
    expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    assert_eq!(got.len(), DATA.len());
    for (actual, oracle) in got.iter().zip(&expected) {
        assert_eq!(actual.0, oracle.0);
        assert!((actual.1 - oracle.1).abs() < 1e-5);
    }
    // Research-only global bucket planning over the two real router outputs. The
    // selected segment-local postings are scored by Qdrant's native scorer.
    // The buffer is an unmodeled bucket of weight 1.0; this fixed policy is
    // illustrative and does not claim calibration or production integration.
    let levels = [&level1, &level2];
    let mut buckets: Vec<(f32, usize, usize, Vec<u32>)> = Vec::new();
    for (level, s) in levels.iter().enumerate() {
        let index = s.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
        let VectorIndexEnum::Lmi(lmi) = &*index else {
            unreachable!()
        };
        let state = lmi.routing_state().unwrap();
        let logits = state.router().forward(&[1.0, 0.0]).unwrap();
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = logits.iter().map(|v| (v - maximum).exp()).sum();
        for (bucket, logit) in logits.iter().enumerate() {
            let offsets = state.postings().get(bucket).unwrap().to_vec();
            if !offsets.is_empty() {
                buckets.push((
                    (logit - maximum).exp() / denominator * 2_f32.powi(level as i32 + 1),
                    level,
                    bucket,
                    offsets,
                ));
            }
        }
    }
    buckets.push((1.0, 2, 0, vec![0]));
    buckets.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut selected = Vec::new();
    let mut count = 0;
    for bucket in buckets {
        if count >= 2 {
            break;
        }
        count += bucket.3.len();
        selected.push(bucket);
    }
    assert!(count >= 2);
    assert!(selected.iter().any(|b| b.1 < 2));
    println!(
        "global-control budget=2 selected_buckets={} candidate_count={} allocation={:?}",
        selected.len(),
        count,
        selected
            .iter()
            .map(|b| (b.1, b.2, b.3.len()))
            .collect::<Vec<_>>()
    );
    let mut selected_results = Vec::new();
    for (_, level, _, offsets) in &selected {
        if *level == 2 {
            selected_results.extend(search(&buffer, &q));
        } else {
            let s = levels[*level];
            let tracker = s.id_tracker.borrow();
            let storage = s.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
            let scored = segment::index::candidate_scoring::score_candidates(
                &q,
                offsets,
                20,
                &tracker,
                &storage,
                &VectorQueryContext::default(),
            )
            .unwrap();
            for point in scored {
                let segment::types::PointIdType::NumId(id) =
                    tracker.external_id(point.idx).unwrap()
                else {
                    unreachable!()
                };
                selected_results.push((id, point.score));
            }
        }
    }
    assert_eq!(selected_results.len(), count);
    let unique: std::collections::HashSet<_> = selected_results.iter().map(|p| p.0).collect();
    assert_eq!(unique.len(), count);
    for (id, score) in &selected_results {
        let v = DATA.iter().find(|row| row.0 == *id).unwrap().1;
        let oracle = v[0] / (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((score - oracle).abs() < 1e-5);
    }
    // Controlled explicit old-copy deletion followed by writes into the Plain buffer.
    // This does not exercise Qdrant's shard update routing or concurrent replacement.
    level1.delete_point(100, 11_u64.into()).unwrap();
    level2.delete_point(101, 21_u64.into()).unwrap();
    buffer
        .upsert_point(102, 11_u64.into(), only_default_vector(&[0.0, 1.0]))
        .unwrap();
    buffer
        .upsert_point(103, 21_u64.into(), only_default_vector(&[0.0, -1.0]))
        .unwrap();
    let changed: Vec<_> = [&level1, &level2, &buffer]
        .into_iter()
        .flat_map(|s| search(s, &q))
        .collect();
    assert_eq!(changed.len(), DATA.len());
    assert_eq!(changed.iter().filter(|(id, _)| *id == 11).count(), 1);
    assert_eq!(changed.iter().filter(|(id, _)| *id == 21).count(), 1);
    assert!(changed.iter().find(|(id, _)| *id == 11).unwrap().1.abs() < 1e-5);
    assert!(changed.iter().find(|(id, _)| *id == 21).unwrap().1.abs() < 1e-5);
    level1.flush(true).unwrap();
    level2.flush(true).unwrap();
    for s in [level1, level2] {
        let before = search(&s, &q);
        let path = s.segment_path.clone();
        let uuid = s.uuid;
        drop(s);
        let loaded = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
        assert_eq!(search(&loaded, &q), before);
    }
}
