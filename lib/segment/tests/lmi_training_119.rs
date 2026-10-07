#![cfg(feature = "lmi-training")]

use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
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
    let hw = HardwareCounterCell::new();
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
                &hw,
            )
            .unwrap();
    }
    // A deleted source point is excluded before building the target segment.
    source.delete_point(100, 3_u64.into(), &hw).unwrap();
    let mut cfg = source.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(config());
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder
        .update(&[&source], &AtomicBool::new(false), &hw)
        .unwrap();
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
    let (_root_a, a) = build(None);
    let (_root_b, b) = build(None);
    for name in ["lmi_router.bin", "lmi_postings.bin"] {
        let left = std::fs::read(a.segment_path.join("vector_index").join(name)).unwrap();
        let right = std::fs::read(b.segment_path.join("vector_index").join(name)).unwrap();
        assert_eq!(left, right, "{name} changed under fixed seed");
    }
    assert_eq!(result(&a), result(&b));
}

#[test]
#[ignore = "explicit 100K synthetic systems-validation gate"]
fn medium_100k_synthetic_build_restarts_with_complete_postings() {
    use std::time::Instant;
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();
    let root = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut source = build_simple_segment(root.path(), 8, Distance::Cosine).unwrap();
    let hw = HardwareCounterCell::new();
    let insert_started = Instant::now();
    for i in 0..100_000u64 {
        let vector: Vec<f32> = (0..8)
            .map(|j| (((i as usize * 17 + j * 31) % 101) as f32 - 50.0) / 50.0)
            .collect();
        source
            .upsert_point(i + 1, (i + 1).into(), only_default_vector(&vector), &hw)
            .unwrap();
    }
    let insert_seconds = insert_started.elapsed().as_secs_f64();
    let mut cfg = source.config().clone();
    let lmi_cfg = LmiConfig {
        n_buckets: 32,
        sample_size: 2048,
        hidden_dim: 32,
        epochs: 2,
        batch_size: 128,
        routing_batch_size: 1024,
        kmeans_iterations: 5,
        nprobe: 2,
        seed: 42,
    };
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(lmi_cfg);
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder
        .update(&[&source], &AtomicBool::new(false), &hw)
        .unwrap();
    let started = Instant::now();
    let built = builder.build_for_test(root.path());
    let build_seconds = started.elapsed().as_secs_f64();
    let metadata: LmiStateMetadata = serde_json::from_slice(
        &std::fs::read(built.segment_path.join("vector_index/lmi_state.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(metadata.indexed_live_count, Some(100_000));
    let path = built.segment_path.clone();
    let uuid = built.uuid;
    let sizes: Vec<_> = ["lmi_state.json", "lmi_router.bin", "lmi_postings.bin"]
        .iter()
        .map(|name| {
            std::fs::metadata(path.join("vector_index").join(name))
                .unwrap()
                .len()
        })
        .collect();
    drop(built);
    let reopen_started = Instant::now();
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    let reopen_seconds = reopen_started.elapsed().as_secs_f64();
    let query: QueryVector = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into();
    let candidates = reopened.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[&query], None, 10, None, &VectorQueryContext::default())
        .unwrap();
    assert!(!candidates[0].is_empty());
    let vmhwm_kb = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|value| value.parse::<u64>().ok())
        });
    println!(
        "lmi_medium_100k insert_s={insert_seconds:.3} build_s={build_seconds:.3} reopen_s={reopen_seconds:.3} state_bytes={} router_bytes={} posting_bytes={} process_vmhwm_kb={vmhwm_kb:?}",
        sizes[0], sizes[1], sizes[2]
    );
}

#[test]
#[ignore = "explicit real-data SISAP300K gate"]
fn sisap300k_production_build_and_bounded_recall() {
    use half::f16;
    use segment::id_tracker::IdTrackerRead;
    use segment::types::PointIdType;
    use std::io::Read;
    use std::time::Instant;

    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();
    let base =
        std::path::Path::new("/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap300k");
    let output = base.join("synchronized_run");
    assert!(
        !output.exists(),
        "SISAP output already exists; never overwrite a previous run"
    );
    std::fs::create_dir_all(&output).unwrap();
    let root = output.join("segments");
    let staging = output.join("staging");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&staging).unwrap();
    let mut source = build_simple_segment(&root, 768, Distance::Cosine).unwrap();
    let mut source_cfg = source.config().clone();
    source_cfg
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .datatype = Some(VectorStorageDatatype::Float16);
    drop(source);
    source = segment::segment_constructor::build_segment(&root, &source_cfg, None, true)
        .unwrap()
        .0;
    let hw = HardwareCounterCell::new();
    let mut stream =
        std::io::BufReader::new(std::fs::File::open(base.join("vectors.f16")).unwrap());
    let mut raw = vec![0_u8; 768 * 2];
    let load_started = Instant::now();
    for id in 0..300_000_u64 {
        stream.read_exact(&mut raw).unwrap();
        let vector: Vec<f32> = raw
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect();
        source
            .upsert_point(id + 1, (id + 1).into(), only_default_vector(&vector), &hw)
            .unwrap();
        if (id + 1) % 50_000 == 0 {
            println!("sisap_loaded={}", id + 1);
        }
    }
    assert!(stream.read(&mut [0_u8]).unwrap() == 0);
    let load_s = load_started.elapsed().as_secs_f64();
    let mut cfg = source.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(LmiConfig {
        n_buckets: 548,
        sample_size: 32768,
        hidden_dim: 512,
        epochs: 30,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations: 5,
        nprobe: 4,
        seed: 42,
    });
    let mut builder = SegmentBuilder::new(
        &staging,
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder
        .update(&[&source], &AtomicBool::new(false), &hw)
        .unwrap();
    drop(source);
    let build_started = Instant::now();
    let built = builder.build_for_test(&root);
    let build_s = build_started.elapsed().as_secs_f64();
    let path = built.segment_path.clone();
    let uuid = built.uuid;
    let metadata: LmiStateMetadata =
        serde_json::from_slice(&std::fs::read(path.join("vector_index/lmi_state.json")).unwrap())
            .unwrap();
    assert_eq!(metadata.indexed_live_count, Some(300_000));
    let sizes: Vec<_> = ["lmi_state.json", "lmi_router.bin", "lmi_postings.bin"]
        .iter()
        .map(|name| {
            std::fs::metadata(path.join("vector_index").join(name))
                .unwrap()
                .len()
        })
        .collect();
    drop(built);
    let reopen_started = Instant::now();
    let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    let reopen_s = reopen_started.elapsed().as_secs_f64();
    let index = reopened.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("SISAP must open StaticLearned LMI")
    };
    let state = lmi.routing_state().unwrap();
    assert_eq!(state.postings().point_count(), 300_000);
    let mut query_stream =
        std::io::BufReader::new(std::fs::File::open(base.join("queries.f32")).unwrap());
    let mut gold_stream =
        std::io::BufReader::new(std::fs::File::open(base.join("gold.i32")).unwrap());
    let mut query_raw = vec![0_u8; 768 * 4];
    let mut gold_raw = [0_u8; 40];
    let mut matches_zero_based = 0_usize;
    let mut matches_one_based = 0_usize;
    let mut candidates_total = 0_usize;
    let query_started = Instant::now();
    for _ in 0..100 {
        query_stream.read_exact(&mut query_raw).unwrap();
        gold_stream.read_exact(&mut gold_raw).unwrap();
        let vector: Vec<f32> = query_raw
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let gold: Vec<i32> = gold_raw
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let query: QueryVector = vector.clone().into();
        let normalized: QueryVector = Distance::Cosine.preprocess_vector::<f32>(vector).into();
        candidates_total += state
            .candidates_for_query(&normalized, &AtomicBool::new(false))
            .unwrap()
            .unwrap()
            .len();
        let points = index
            .search(&[&query], None, 10, None, &VectorQueryContext::default())
            .unwrap();
        let tracker = reopened.id_tracker.borrow();
        for point in &points[0] {
            let PointIdType::NumId(id) = tracker.external_id(point.idx).unwrap() else {
                panic!("non-numeric SISAP ID")
            };
            if gold.iter().any(|&g| i64::from(g) == id as i64 - 1) {
                matches_zero_based += 1;
            }
            if gold.iter().any(|&g| i64::from(g) == id as i64) {
                matches_one_based += 1;
            }
        }
    }
    let query_s = query_started.elapsed().as_secs_f64();
    println!(
        "sisap300k load_s={load_s:.3} build_s={build_s:.3} reopen_s={reopen_s:.3} query_100_s={query_s:.3} recall_zero_based={:.6} recall_one_based={:.6} mean_candidates={:.1} state_router_postings_bytes={sizes:?} path={}",
        matches_zero_based as f64 / 1000.0,
        matches_one_based as f64 / 1000.0,
        candidates_total as f64 / 100.0,
        path.display()
    );
    assert!(matches_zero_based > 0 || matches_one_based > 0);
}
