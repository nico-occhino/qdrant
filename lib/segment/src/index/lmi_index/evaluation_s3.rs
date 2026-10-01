//! Explicit build-only systems benchmark. Never runs in the normal test suite.
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::counter::hardware_counter::HardwareCounterCell;
use rand::SeedableRng;
use serde_json::{Value, json};

use super::LmiConfig;
use crate::data_types::query_context::QueryContext;
use crate::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use crate::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use crate::index::{VectorIndex, VectorIndexRead};
use crate::segment_constructor::{
    load_segment, segment_builder::SegmentBuilder, simple_segment_constructor::build_simple_segment,
};
use crate::types::{Distance, HnswGlobalConfig, Indexes};

#[test]
#[ignore = "Explicit S.3 build benchmark; requires LMI_S3_DATA and LMI_S3_OUTPUT"]
fn phase_s3_build_benchmark() {
    let _ = env_logger::try_init();
    let data = PathBuf::from(std::env::var("LMI_S3_DATA").unwrap());
    let output = PathBuf::from(std::env::var("LMI_S3_OUTPUT").unwrap());
    assert!(!output.exists(), "never overwrite benchmark output");
    std::fs::create_dir_all(&output).unwrap();
    let meta: Value =
        serde_json::from_slice(&std::fs::read(data.join("dataset.json")).unwrap()).unwrap();
    let dim = meta["dimension"].as_u64().unwrap() as usize;
    let count = meta["corpus_count"].as_u64().unwrap() as usize;
    let distance: Distance = serde_json::from_value(meta["metric"].clone()).unwrap();
    let config = match std::env::var("LMI_S3_CONFIG") {
        Ok(path) => serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap(),
        Err(_) => LmiConfig {
            n_buckets: 64,
            sample_size: 2048,
            hidden_dim: 64,
            epochs: 30,
            batch_size: 256,
            kmeans_iterations: 20,
            nprobe: 4,
            seed: 42,
        },
    };
    let root = tempfile::tempdir().unwrap();
    let mut source = build_simple_segment(root.path(), dim, distance).unwrap();
    let mut reader = BufReader::new(std::fs::File::open(data.join("corpus.f32")).unwrap());
    let mut bytes = vec![0u8; dim * 4];
    let mut row = vec![0f32; dim];
    let mut query = Vec::new();
    let ingest = Instant::now();
    for i in 0..count {
        reader.read_exact(&mut bytes).unwrap();
        for (v, b) in row.iter_mut().zip(bytes.chunks_exact(4)) {
            *v = f32::from_le_bytes(b.try_into().unwrap());
        }
        if i == 0 {
            query = row.clone();
        }
        source
            .upsert_point(
                i as u64 + 1,
                (i as u64).into(),
                only_default_vector(&row),
                &HardwareCounterCell::new(),
            )
            .unwrap();
    }
    assert_eq!(reader.read(&mut [0]).unwrap(), 0);
    let ingest_seconds = ingest.elapsed().as_secs_f64();
    let staging = tempfile::tempdir().unwrap();
    let mut cfg = source.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(config);
    let started = Instant::now();
    let mut builder =
        SegmentBuilder::new(staging.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    builder
        .update(
            &[&source],
            &AtomicBool::new(false),
            &HardwareCounterCell::new(),
        )
        .unwrap();
    let segment = builder
        .build(
            root.path(),
            uuid::Uuid::new_v4(),
            None,
            common::budget::ResourcePermit::dummy(1),
            &AtomicBool::new(false),
            &mut rand::rngs::StdRng::seed_from_u64(42),
            &HardwareCounterCell::new(),
            common::progress_tracker::ProgressTracker::new_for_test(),
        )
        .unwrap();
    let total_build_seconds = started.elapsed().as_secs_f64();
    let stages = *super::training::STAGE_SECONDS.lock().unwrap();
    let path = segment.segment_path.clone();
    let mut sizes = serde_json::Map::new();
    {
        let index = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow();
        for file in index.files() {
            let name = file.file_name().unwrap().to_str().unwrap();
            sizes.insert(name.into(), json!(file.metadata().unwrap().len()));
            std::fs::copy(&file, output.join(name)).unwrap();
        }
    }
    drop(segment);
    let started = Instant::now();
    let reopened = load_segment(&path, uuid::Uuid::nil(), None, &AtomicBool::new(false)).unwrap();
    let reopen_seconds = started.elapsed().as_secs_f64();
    let query: QueryVector = distance.preprocess_vector::<f32>(query).into();
    let root_context = QueryContext::default();
    let context = root_context.get_segment_query_context();
    let context = context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    let results = reopened.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[&query], None, 10, None, &context)
        .unwrap();
    let result = json!({"n":count,"d":dim,"config":config,"ingest_seconds":ingest_seconds,
        "total_build_seconds":total_build_seconds,"clustering_seconds":stages[0],
        "training_export_seconds":stages[1],"reopen_seconds":reopen_seconds,
        "files":sizes,"query_results":results[0].iter().map(|p| (p.idx,p.score)).collect::<Vec<_>>(),"proc_status":std::fs::read_to_string("/proc/self/status").unwrap()});
    std::fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!("S3_RESULT {}", serde_json::to_string(&result).unwrap());
}
