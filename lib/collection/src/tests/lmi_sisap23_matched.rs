//! Opt-in matched SISAP23 Plain/HNSW/LMI benchmark using native Qdrant index paths.
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use ahash::AHashMap;
use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::generic_consts::Random;
use common::storage_version::StorageVersion;
use common::universal_io::MmapFs;
use half::f16;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector};
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexRead;
use segment::index::candidate_scoring::score_candidates;
use segment::index::lmi_index::{LmiConfig, LmiStateMetadata};
use segment::index::{VectorIndex, VectorIndexEnum};
use segment::segment::SegmentVersion;
use segment::segment_constructor::load_segment;
use segment::types::{
    Distance, HnswConfig, Indexes, PointIdType, SearchParams, VectorStorageDatatype,
};
use segment::vector_storage::VectorStorageRead;
use sha2::{Digest, Sha256};
use shard::operations::CollectionUpdateOperations;
use shard::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorStructPersisted,
};
use uuid::Uuid;

use crate::collection::Collection;
use crate::config::{CollectionConfigInternal, CollectionParams};
use crate::operations::OperationWithClockTag;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::operations::types::{Datatype, VectorsConfig};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::optimizers_builder::OptimizersConfig;
use crate::shards::channel_service::ChannelService;
use crate::shards::collection_shard_distribution::CollectionShardDistribution;
use crate::shards::replica_set::replica_set_state::ReplicaState;
use crate::shards::shard_trait::WaitUntil;
use crate::tests::snapshot_test::{
    dummy_abort_shard_transfer, dummy_on_replica_failure, dummy_request_shard_transfer,
};

const N: usize = 10_120_191;
const DIM: usize = 768;
const CHUNK: usize = 100;
const LMI_PROBES: [usize; 10] = [1, 4, 8, 16, 30, 40, 64, 100, 160, 256];
const HNSW_EF: [usize; 7] = [16, 32, 64, 128, 256, 512, 1024];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Plain,
    Lmi,
    HnswA,
    HnswB,
}

impl Variant {
    fn parse(s: &str) -> Self {
        match s {
            "plain" => Self::Plain,
            "lmi" => Self::Lmi,
            "hnsw-a" => Self::HnswA,
            "hnsw-b" => Self::HnswB,
            _ => panic!("SISAP23_VARIANT must be plain, lmi, hnsw-a, or hnsw-b"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Lmi => "lmi",
            Self::HnswA => "hnsw-a",
            Self::HnswB => "hnsw-b",
        }
    }
    fn params(self) -> (usize, usize) {
        match self {
            Self::HnswA => (16, 100),
            Self::HnswB => (32, 200),
            _ => (0, 0),
        }
    }
}
fn root() -> PathBuf {
    PathBuf::from(std::env::var("SISAP23_BENCH_ROOT").expect("SISAP23_BENCH_ROOT required"))
}
fn run_dir(v: Variant) -> PathBuf {
    root().join("runs").join(v.name())
}
fn read_f32(path: &Path, rows: usize) -> Vec<Vec<f32>> {
    let mut bytes = Vec::new();
    File::open(path).unwrap().read_to_end(&mut bytes).unwrap();
    assert_eq!(
        bytes.len(),
        rows * DIM * 4,
        "unexpected input length: {}",
        path.display()
    );
    bytes
        .chunks_exact(DIM * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect()
}
fn write_json_new(path: &Path, value: &serde_json::Value) {
    let tmp = path.with_extension("json.tmp");
    let mut f = File::options()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .expect("refuse an existing or partial metadata file");
    serde_json::to_writer_pretty(&mut f, value).unwrap();
    f.write_all(b"\n").unwrap();
    f.sync_all().unwrap();
    // Atomic create-new; never replace an accepted marker.
    fs::hard_link(&tmp, path).expect("refuse to replace an existing completion marker");
    fs::remove_file(tmp).unwrap();
}
fn find_named(path: &Path, name: &str, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(path).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            find_named(&p, name, out)
        } else if p.file_name().is_some_and(|n| n == name) {
            out.push(p)
        }
    }
}
fn locate_segment(collection: &Path) -> PathBuf {
    let mut found = Vec::new();
    find_named(collection, "segment.json", &mut found);
    assert_eq!(
        found.len(),
        1,
        "expected exactly one optimized segment, found {}",
        found.len()
    );
    found[0].parent().unwrap().to_path_buf()
}
fn load_index(path: &Path) -> segment::segment::Segment {
    let uuid = Uuid::parse_str(path.file_name().unwrap().to_str().unwrap()).unwrap();
    load_segment(path, uuid, None, &AtomicBool::new(false), false).unwrap()
}
fn path_error(path: &Path, action: &str, err: io::Error) -> io::Error {
    io::Error::new(err.kind(), format!("{action} {}: {err}", path.display()))
}

fn count_bytes(path: &Path) -> io::Result<u64> {
    let metadata = fs::symlink_metadata(path).map_err(|e| path_error(path, "stat", e))?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unexpected non-file, non-directory: {}", path.display()),
        ));
    }
    let mut total = 0u64;
    for entry in fs::read_dir(path).map_err(|e| path_error(path, "read directory", e))? {
        let entry = entry.map_err(|e| path_error(path, "read entry in", e))?;
        let child = entry.path();
        total = total.checked_add(count_bytes(&child)?).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("byte count overflow in {}", path.display()),
            )
        })?;
    }
    Ok(total)
}

fn index_directory_bytes(
    segment_path: &Path,
    variant: Variant,
    required_files: &[PathBuf],
) -> io::Result<u64> {
    let path = segment_path.join("vector_index");
    match fs::symlink_metadata(&path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound && variant == Variant::Plain => return Ok(0),
        Err(e) => return Err(path_error(&path, "stat index directory", e)),
        Ok(metadata) if !metadata.is_dir() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("index path is not a directory: {}", path.display()),
            ));
        }
        Ok(_) => {}
    }
    if variant != Variant::Plain {
        if required_files.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} has no persisted index files", path.display()),
            ));
        }
        for file in required_files {
            let metadata = fs::symlink_metadata(file)
                .map_err(|e| path_error(file, "stat required index file", e))?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "required index file is empty or not regular: {}",
                        file.display()
                    ),
                ));
            }
        }
    }
    let bytes = count_bytes(&path)?;
    if variant != Variant::Plain && bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("required index directory is empty: {}", path.display()),
        ));
    }
    Ok(bytes)
}

fn hash_index_tree(segment_path: &Path) -> Vec<(PathBuf, String)> {
    let dir = segment_path.join("vector_index");
    if !dir.exists() {
        return Vec::new();
    }
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let kind = fs::symlink_metadata(&path).unwrap();
            assert!(!kind.file_type().is_symlink(), "index symlink is forbidden");
            if kind.is_dir() {
                walk(&path, files);
            } else {
                assert!(kind.is_file(), "unexpected index file type");
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(&dir, &mut files);
    files.sort();
    files
        .into_iter()
        .map(|file| {
            let relative = file.strip_prefix(segment_path).unwrap().to_path_buf();
            (relative, sha256_file(&file).unwrap())
        })
        .collect()
}

fn verify_persisted_generation(
    collection_path: &Path,
    variant: Variant,
    expected: usize,
) -> serde_json::Value {
    let segment_path = locate_segment(collection_path);
    let segment = load_index(&segment_path);
    let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
    let config = &segment.segment_config.vector_data[DEFAULT_VECTOR_NAME];
    assert_eq!(config.size, DIM);
    assert_eq!(config.distance, Distance::Cosine);
    assert_eq!(config.datatype, Some(VectorStorageDatatype::Float16));
    let index = data.vector_index.borrow();
    let expected_kind = match variant {
        Variant::Plain => "plain",
        Variant::Lmi => "lmi",
        Variant::HnswA | Variant::HnswB => "hnsw",
    };
    assert_eq!(index_kind(&index), expected_kind);
    let storage = data.vector_storage.borrow();
    let tracker = segment.id_tracker.borrow();
    assert_eq!(storage.total_vector_count(), expected);
    assert_eq!(storage.available_vector_count(), expected);
    assert_eq!(storage.deleted_vector_count(), 0);
    assert_eq!(tracker.total_point_count(), expected);
    assert_eq!(tracker.available_point_count(), expected);
    assert_eq!(tracker.deleted_point_count(), 0);
    let mut mappings = MappingAudit::new(expected);
    let mut bad_reverse = 0usize;
    let mut absent = 0usize;
    let mut missing_versions = 0usize;
    for (external, offset) in tracker.point_mappings().iter_from(None) {
        let id = match external {
            PointIdType::NumId(id) => id,
            _ => 0,
        };
        mappings.observe(id, offset, expected);
        bad_reverse += usize::from(tracker.external_id(offset) != Some(external));
        missing_versions += usize::from(tracker.internal_version(offset).is_none());
        if (offset as usize) < expected {
            absent += usize::from(
                storage.is_deleted_vector(offset)
                    || storage.get_vector_opt::<Random>(offset).is_none(),
            );
        }
    }
    let mapping = mappings.finish();
    assert_eq!(mappings.iterated, expected);
    assert_eq!(mapping["missing_external_ids"], 0);
    assert_eq!(mappings.duplicate_ids, 0);
    assert_eq!(mappings.invalid_ids, 0);
    assert_eq!(mappings.missing_slots, 0);
    assert_eq!(bad_reverse, 0);
    assert_eq!(missing_versions, 0);
    assert_eq!(absent, 0);
    let (m, ef) = variant.params();
    match variant {
        Variant::Plain => assert!(matches!(config.index, Indexes::Plain {})),
        Variant::Lmi => {
            assert!(matches!(&config.index, Indexes::LmiTrained(saved) if *saved == lmi_config()));
            let meta: LmiStateMetadata = serde_json::from_slice(
                &fs::read(segment_path.join("vector_index/lmi_state.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(meta.config, lmi_config());
            assert_eq!(meta.indexed_live_count, Some(expected));
            let VectorIndexEnum::Lmi(lmi) = &*index else {
                unreachable!()
            };
            assert_eq!(
                lmi.routing_state().unwrap().postings().point_count(),
                expected
            );
        }
        Variant::HnswA | Variant::HnswB => {
            let Indexes::Hnsw(saved) = &config.index else {
                panic!("missing HNSW segment config")
            };
            assert_eq!(saved.m, m);
            assert_eq!(saved.ef_construct, ef);
            let graph: serde_json::Value = serde_json::from_slice(
                &fs::read(segment_path.join("vector_index/hnsw_config.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(graph["m"], m);
            assert_eq!(graph["ef_construct"], ef);
            assert_eq!(graph["indexed_vector_count"], expected);
        }
    }
    let index_bytes = vector_index_bytes(&segment_path, variant, &index).unwrap();
    serde_json::json!({"points":expected,"mapping":mapping,"index_kind":expected_kind,
        "index_bytes":index_bytes,"index_files":hash_index_tree(&segment_path).len(),
        "complete":true})
}

async fn reopen_for_verification(collection_path: &Path, snapshots: &Path, expected: usize) {
    let pause = serde_json::from_value(serde_json::json!({"max_optimization_threads":0})).unwrap();
    let collection = Collection::load(
        "sisap23_matched".to_string(),
        1,
        collection_path,
        snapshots,
        Arc::new(SharedStorageConfig::default()),
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        Some(pause),
    )
    .await;
    let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
    assert_eq!(
        info.points_count,
        Some(expected),
        "normal restart point count"
    );
    collection.stop_gracefully().await;
}

fn vector_index_bytes(
    segment_path: &Path,
    variant: Variant,
    index: &VectorIndexEnum,
) -> io::Result<u64> {
    let files = match index {
        VectorIndexEnum::Lmi(lmi) => lmi.files(),
        VectorIndexEnum::Hnsw(hnsw) => hnsw.files(),
        _ => Vec::new(),
    };
    index_directory_bytes(segment_path, variant, &files)
}

#[test]
fn sisap23_index_byte_accounting_is_variant_aware() {
    let temp = tempfile::tempdir().unwrap();
    let segment = temp.path();
    assert_eq!(
        index_directory_bytes(segment, Variant::Plain, &[]).unwrap(),
        0
    );
    let missing = index_directory_bytes(
        segment,
        Variant::Lmi,
        &[segment.join("vector_index/lmi_state.json")],
    )
    .unwrap_err();
    assert!(missing.to_string().contains("vector_index"));
    fs::create_dir(segment.join("vector_index")).unwrap();
    let missing_file = segment.join("vector_index/lmi_state.json");
    let err = index_directory_bytes(segment, Variant::Lmi, &[missing_file.clone()]).unwrap_err();
    assert!(err.to_string().contains("lmi_state.json"));
    fs::write(&missing_file, b"metadata").unwrap();
    assert_eq!(
        index_directory_bytes(segment, Variant::Lmi, &[missing_file]).unwrap(),
        8
    );
    let err = index_directory_bytes(segment, Variant::HnswA, &[]).unwrap_err();
    assert!(err.to_string().contains("no persisted index files"));
}
#[test]
fn sisap23_completion_marker_cannot_be_replaced() {
    let temp = tempfile::tempdir().unwrap();
    let marker = temp.path().join("build.complete.json");
    write_json_new(&marker, &serde_json::json!({"verified": true}));
    let first = fs::read(&marker).unwrap();
    let second = std::panic::catch_unwind(|| {
        write_json_new(&marker, &serde_json::json!({"verified": false}));
    });
    assert!(second.is_err(), "existing marker must not be replaced");
    assert_eq!(fs::read(&marker).unwrap(), first);
}

fn index_kind(index: &VectorIndexEnum) -> &'static str {
    match index {
        VectorIndexEnum::Plain(_) => "plain",
        VectorIndexEnum::Lmi(_) => "lmi",
        VectorIndexEnum::Hnsw(_) => "hnsw",
        _ => "other",
    }
}
fn lmi_config() -> LmiConfig {
    LmiConfig {
        n_buckets: 3162,
        sample_size: 250000,
        hidden_dim: 512,
        epochs: 30,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations: 5,
        nprobe: 4,
        seed: 42,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit SISAP23 matched index build; requires the prepared 10M corpus"]
async fn sisap23_matched_build() {
    let variant =
        Variant::parse(&std::env::var("SISAP23_VARIANT").expect("SISAP23_VARIANT required"));
    let run = run_dir(variant);
    assert!(
        run.is_dir(),
        "PBS build script must create a fresh variant run directory"
    );
    let collection_path = run.join("collection");
    assert!(
        !collection_path.exists(),
        "refusing to overwrite an existing collection/index generation"
    );
    fs::create_dir(&collection_path).unwrap();
    let snapshots = run.join("snapshots");
    fs::create_dir(&snapshots).unwrap();
    let (m, ef) = variant.params();
    let mut vp = VectorParamsBuilder::new(DIM as u64, Distance::Cosine).build();
    vp.datatype = Some(Datatype::Float16);
    vp.memory = Some(segment::types::Memory::Cold);
    if variant == Variant::Lmi {
        vp.lmi_config = Some(lmi_config());
    }
    let mut optimizers = OptimizersConfig::fixture();
    optimizers.max_optimization_threads = Some(0);
    optimizers.indexing_threshold = Some(if variant == Variant::Plain {
        usize::MAX
    } else {
        1
    });
    optimizers.max_segment_size = Some(32_000_000);
    optimizers.default_segment_number = 1;
    let mut hnsw = HnswConfig::default();
    if variant == Variant::HnswA || variant == Variant::HnswB {
        hnsw.m = m;
        hnsw.ef_construct = ef;
        hnsw.max_indexing_threads = 1;
        hnsw.memory = Some(segment::types::Memory::Cached);
    }
    let config = CollectionConfigInternal {
        params: CollectionParams {
            vectors: VectorsConfig::Single(vp),
            shard_number: NonZeroU32::new(1).unwrap(),
            ..CollectionParams::empty()
        },
        optimizer_config: optimizers,
        wal_config: Default::default(),
        hnsw_config: hnsw,
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
    };
    let collection = Collection::new(
        "sisap23_matched".to_string(),
        1,
        &collection_path,
        &snapshots,
        &config,
        Arc::new(SharedStorageConfig::default()),
        CollectionShardDistribution {
            shards: AHashMap::from([(0, HashSet::from([1]))]),
        },
        None,
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
    )
    .await
    .unwrap();
    collection
        .set_shard_replica_state(0, 1, ReplicaState::Active, None)
        .await
        .unwrap();
    let shard = collection
        .shards_holder()
        .read()
        .await
        .get_shard(0)
        .unwrap()
        .clone();
    let corpus = PathBuf::from(
        std::env::var("SISAP23_CORPUS_PATH")
            .unwrap_or_else(|_| root().join("corpus.f16le").to_string_lossy().to_string()),
    );
    assert_eq!(fs::metadata(&corpus).unwrap().len() as usize, N * DIM * 2);
    let mut input = BufReader::new(File::open(corpus).unwrap());
    let mut raw = vec![0u8; DIM * 2];
    let mut corpus_hash = Sha256::new();
    let ingest = Instant::now();
    for start in (0..N).step_by(1000) {
        let end = (start + 1000).min(N);
        let mut batch = Vec::with_capacity(end - start);
        for id in (start + 1)..=end {
            input.read_exact(&mut raw).unwrap();
            corpus_hash.update(&raw);
            let v = raw
                .chunks_exact(2)
                .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect();
            batch.push(PointStructPersisted {
                id: (id as u64).into(),
                vector: VectorStructPersisted::Single(v),
                payload: None,
            });
        }
        shard
            .update_local(
                OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                    PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(batch)),
                )),
                WaitUntil::Visible,
                None,
                HwMeasurementAcc::new(),
                false,
            )
            .await
            .unwrap();
        if end % 100000 == 0 || end == N {
            println!(
                "sisap23_ingest_points={end} elapsed_s={:.1}",
                ingest.elapsed().as_secs_f64()
            );
        }
    }
    let ingest_s = ingest.elapsed().as_secs_f64();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root().join("manifest.json")).unwrap()).unwrap();
    let corpus_sha = corpus_hash
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    assert_eq!(
        corpus_sha,
        manifest["prepared"]["corpus"]["sha256"].as_str().unwrap(),
        "corpus input hash mismatch"
    );
    let mut index_build_s = 0.0;
    let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
    assert_eq!(info.points_count, Some(N));
    if variant != Variant::Plain {
        let diff =
            serde_json::from_value(serde_json::json!({"max_optimization_threads":1})).unwrap();
        collection
            .update_optimizer_params_from_diff(diff)
            .await
            .unwrap();
        collection.recreate_optimizers_background();
        let started = Instant::now();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(172800);
        loop {
            let current = collection.info(&ShardSelectorInternal::All).await.unwrap();
            if current.indexed_vectors_count == Some(N) {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "indexing timeout: {:?}",
                current.optimizer_status
            );
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        index_build_s = started.elapsed().as_secs_f64();
        println!("sisap23_index_build_s={:.3}", index_build_s);
    }
    drop(shard);
    collection.stop_gracefully().await;
    drop(collection);
    let segment_path = locate_segment(&collection_path);
    let index_hash_before = hash_index_tree(&segment_path);
    reopen_for_verification(&collection_path, &snapshots, N).await;
    let first_audit = verify_persisted_generation(&collection_path, variant, N);
    assert_eq!(
        hash_index_tree(&segment_path),
        index_hash_before,
        "index changed during verification-only restart"
    );
    reopen_for_verification(&collection_path, &snapshots, N).await;
    let second_audit = verify_persisted_generation(&collection_path, variant, N);
    assert_eq!(
        hash_index_tree(&segment_path),
        index_hash_before,
        "index changed during second verification restart"
    );
    let segment = load_index(&segment_path);
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let expected = match variant {
        Variant::Plain => "plain",
        Variant::Lmi => "lmi",
        Variant::HnswA | Variant::HnswB => "hnsw",
    };
    assert_eq!(index_kind(&index), expected, "wrong persisted index type");
    if variant == Variant::Lmi {
        let VectorIndexEnum::Lmi(lmi) = &*index else {
            unreachable!()
        };
        let metadata_path = segment_path.join("vector_index/lmi_state.json");
        let meta: LmiStateMetadata =
            serde_json::from_slice(&fs::read(metadata_path).unwrap()).unwrap();
        assert_eq!(meta.config, lmi_config());
        assert_eq!(meta.indexed_live_count, Some(N));
        assert_eq!(lmi.routing_state().unwrap().postings().point_count(), N);
    }
    let (m, ef) = variant.params();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root().join("manifest.json")).unwrap()).unwrap();
    let source_commit = manifest["source_commit"]
        .as_str()
        .expect("source commit missing");
    let marker = serde_json::json!({
        "format": "sisap23-build-v1",
        "variant": variant.name(),
        "source_commit": source_commit,
        "point_count": N,
        "dimension": DIM,
        "datatype": "float16",
        "distance": "cosine",
        "segment_path": segment_path,
        "segment_count": 1,
        "index_kind": expected,
        "lmi": if variant == Variant::Lmi { serde_json::json!({"n_buckets":3162,"sample_size":250000,"hidden_dim":512,"epochs":30,"batch_size":256,"routing_batch_size":256,"kmeans_iterations":5,"nprobe":4,"seed":42}) } else { serde_json::Value::Null },
        "hnsw_m": if m == 0 { serde_json::Value::Null } else { m.into() },
        "ef_construct": if ef == 0 { serde_json::Value::Null } else { ef.into() },
        "ingest_seconds": ingest_s,
        "index_build_seconds": index_build_s,
        "corpus_sha256": corpus_sha,
        "collection_bytes": count_bytes(&collection_path).expect("cannot count collection bytes"),
        "segment_bytes": count_bytes(&segment_path).expect("cannot count segment bytes"),
        "vector_index_bytes": vector_index_bytes(&segment_path, variant, &index)
            .expect("cannot count physical vector index bytes"),
        "durability_verification": {"normal_reopens":2,"first":first_audit,"second":second_audit,
            "index_files_unchanged_during_reopen":true},
        "generation": "directory is immutable after this marker; all sweeps reuse this path"
    });
    drop(index);
    drop(segment);
    write_json_new(&run.join("build.complete.json"), &marker);
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut input = BufReader::new(File::open(path).map_err(|e| path_error(path, "open", e))?);
    let mut hash = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = input
            .read(&mut buf)
            .map_err(|e| path_error(path, "read", e))?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Verify the interrupted Plain generation without rebuilding or changing its files.
fn verify_existing_plain() -> serde_json::Value {
    let root = root();
    let run = run_dir(Variant::Plain);
    assert!(
        !run.join("build.complete.json").exists(),
        "Plain build marker already exists"
    );
    let collection_path = run.join("collection");
    assert!(
        collection_path.is_dir(),
        "Plain collection missing: {}",
        collection_path.display()
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["format"], "sisap23-matched-v1");
    let source_commit = manifest["source_commit"]
        .as_str()
        .expect("source commit missing");
    assert_eq!(
        source_commit, "42a163701a51078ec24983b3a93f96ab7d661489",
        "recovery is restricted to the frozen failed generation"
    );
    let prepared = manifest["prepared"]
        .as_object()
        .expect("prepared manifest missing");
    for (name, entry) in prepared {
        let input = root.join(entry["path"].as_str().expect("prepared path missing"));
        assert_eq!(
            fs::metadata(&input).unwrap().len(),
            entry["bytes"].as_u64().unwrap(),
            "prepared input size mismatch: {name}"
        );
        assert_eq!(
            sha256_file(&input).unwrap(),
            entry["sha256"].as_str().unwrap(),
            "prepared input hash mismatch: {name}"
        );
    }
    let corpus_sha = prepared["corpus"]["sha256"].as_str().unwrap();
    let segment_path = locate_segment(&collection_path);
    assert_eq!(
        SegmentVersion::load_universal(&MmapFs, &segment_path).unwrap(),
        Some(SegmentVersion::current()),
        "recovery refuses a segment version migration that could rewrite collection files"
    );
    let segment = load_index(&segment_path);
    assert_eq!(
        segment.segment_config.vector_data.len(),
        1,
        "expected one dense vector field"
    );
    assert!(
        segment.segment_config.sparse_vector_data.is_empty(),
        "unexpected sparse vector field"
    );
    let config = &segment.segment_config.vector_data[DEFAULT_VECTOR_NAME];
    assert_eq!(config.size, DIM, "wrong stored vector dimension");
    assert_eq!(config.distance, Distance::Cosine, "wrong stored distance");
    assert_eq!(
        config.datatype,
        Some(VectorStorageDatatype::Float16),
        "wrong stored datatype"
    );
    assert!(
        matches!(config.index, Indexes::Plain {}),
        "segment config is not Plain"
    );
    assert!(
        config.quantization_config.is_none(),
        "unexpected vector quantization"
    );
    let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
    let index = data.vector_index.borrow();
    assert_eq!(index_kind(&index), "plain", "wrong reopened physical index");
    let storage = data.vector_storage.borrow();
    assert_eq!(storage.datatype(), VectorStorageDatatype::Float16);
    assert_eq!(storage.distance(), Distance::Cosine);
    assert!(storage.is_on_disk(), "Plain vectors are not on disk");
    assert_eq!(
        storage.total_vector_count(),
        N,
        "wrong total stored vectors"
    );
    assert_eq!(
        storage.available_vector_count(),
        N,
        "deleted or unavailable vectors"
    );
    let tracker = segment.id_tracker.borrow();
    assert_eq!(tracker.total_point_count(), N, "wrong total point count");
    assert_eq!(
        tracker.available_point_count(),
        N,
        "deleted or unavailable points"
    );
    let mut seen = vec![false; N];
    let mut mapped = 0usize;
    for (external, internal) in tracker.point_mappings().iter_from(None) {
        let PointIdType::NumId(id) = external else {
            panic!("non-numeric SISAP point ID")
        };
        assert!(
            (1..=N as u64).contains(&id),
            "out-of-range SISAP point ID: {id}"
        );
        let slot = &mut seen[(id - 1) as usize];
        assert!(!*slot, "duplicate SISAP point ID: {id}");
        *slot = true;
        assert_eq!(
            tracker.external_id(internal),
            Some(external),
            "mapping is not reversible"
        );
        assert!(
            tracker.internal_version(internal).is_some(),
            "point lacks persisted version"
        );
        assert!(
            !storage.is_deleted_vector(internal),
            "mapped vector is deleted"
        );
        mapped += 1;
    }
    assert_eq!(mapped, N, "not every one-based SISAP ID is mapped");
    drop(tracker);
    drop(storage);
    drop(index);
    let warm = read_f32(&root.join("queries_warmup.f32le"), 20);
    let (hits, scanned) = search_one(&segment, Variant::Plain, 0, &warm[0]);
    assert_eq!(
        hits.len(),
        10,
        "reopened Plain exact search did not return top ten"
    );
    assert_eq!(scanned, Some(N));
    assert!(hits.iter().all(|id| (1..=N as u64).contains(id)));
    drop(segment);
    let recovery_commit = std::env::var("SISAP23_RECOVERY_COMMIT")
        .expect("SISAP23_RECOVERY_COMMIT must identify the repair executable");
    assert_eq!(
        recovery_commit.len(),
        40,
        "recovery commit must be a full SHA"
    );
    serde_json::json!({
        "format": "sisap23-build-v1", "variant": "plain",
        "source_commit": source_commit, "recovery_commit": recovery_commit,
        "recovered_after_verified_reopen": true,
        "point_count": N, "dimension": DIM, "datatype": "float16", "distance": "cosine",
        "segment_path": segment_path, "segment_count": 1, "index_kind": "plain",
        "lmi": serde_json::Value::Null, "hnsw_m": serde_json::Value::Null,
        "ef_construct": serde_json::Value::Null,
        "ingest_seconds": serde_json::Value::Null,
        "index_build_seconds": serde_json::Value::Null,
        "corpus_sha256": corpus_sha,
        "collection_bytes": count_bytes(&collection_path).expect("cannot count collection bytes"),
        "segment_bytes": count_bytes(&segment_path).expect("cannot count segment bytes"),
        "vector_index_bytes": index_directory_bytes(&segment_path, Variant::Plain, &[])
            .expect("cannot count Plain index bytes"),
        "generation": "original frozen build; recovered metadata after independent validation"
    })
}

#[test]
#[ignore = "explicit validation of the interrupted 10M Plain generation, then marker publication"]
fn sisap23_matched_recover_plain() {
    let marker = verify_existing_plain();
    if std::env::var("SISAP23_RECOVERY_DRY_RUN").as_deref() == Ok("1") {
        println!("sisap23_plain_recovery_verified_dry_run {}", marker);
        return;
    }
    assert_eq!(
        std::env::var("SISAP23_RECOVERY_PUBLISH").as_deref(),
        Ok("1"),
        "set SISAP23_RECOVERY_PUBLISH=1 to create the completion marker"
    );
    let path = run_dir(Variant::Plain).join("build.complete.json");
    write_json_new(&path, &marker);
    println!("sisap23_plain_recovery_complete marker={}", path.display());
}

#[derive(Default)]
struct MappingAudit {
    seen: Vec<bool>,
    iterated: usize,
    duplicate_ids: usize,
    invalid_ids: usize,
    missing_slots: usize,
    outside_offsets: std::collections::HashSet<u32>,
    min_id: Option<u64>,
    max_id: Option<u64>,
    min_offset: Option<u32>,
    max_offset: Option<u32>,
    duplicate_samples: Vec<u64>,
    invalid_samples: Vec<u64>,
    missing_slot_samples: Vec<(u64, u32)>,
}

fn sample_push<T>(items: &mut Vec<T>, value: T) {
    if items.len() < 16 {
        items.push(value);
    }
}

impl MappingAudit {
    fn new(expected: usize) -> Self {
        Self {
            seen: vec![false; expected],
            ..Self::default()
        }
    }
    fn observe(&mut self, id: u64, offset: u32, physical_count: usize) {
        self.iterated += 1;
        self.min_id = Some(self.min_id.map_or(id, |v| v.min(id)));
        self.max_id = Some(self.max_id.map_or(id, |v| v.max(id)));
        self.min_offset = Some(self.min_offset.map_or(offset, |v| v.min(offset)));
        self.max_offset = Some(self.max_offset.map_or(offset, |v| v.max(offset)));
        if !(1..=self.seen.len() as u64).contains(&id) {
            self.invalid_ids += 1;
            sample_push(&mut self.invalid_samples, id);
        } else {
            let seen = &mut self.seen[(id - 1) as usize];
            if *seen {
                self.duplicate_ids += 1;
                sample_push(&mut self.duplicate_samples, id);
            }
            *seen = true;
        }
        if offset as usize >= physical_count {
            self.missing_slots += 1;
            self.outside_offsets.insert(offset);
            sample_push(&mut self.missing_slot_samples, (id, offset));
        }
    }
    fn finish(&self) -> serde_json::Value {
        let mut missing = 0usize;
        let mut missing_samples = Vec::new();
        let mut first_missing = None;
        let mut last_missing = None;
        for (i, seen) in self.seen.iter().enumerate() {
            if !seen {
                let id = (i + 1) as u64;
                missing += 1;
                first_missing.get_or_insert(id);
                last_missing = Some(id);
                sample_push(&mut missing_samples, id);
            }
        }
        let missing_contiguous_tail = match first_missing {
            Some(first) => missing == self.seen.len() + 1 - first as usize,
            None => false,
        };
        let min_outside = self.outside_offsets.iter().min().copied();
        let max_outside = self.outside_offsets.iter().max().copied();
        serde_json::json!({
            "iterated_mappings": self.iterated,
            "min_external_id": self.min_id, "max_external_id": self.max_id,
            "min_internal_offset": self.min_offset, "max_internal_offset": self.max_offset,
            "duplicate_ids": self.duplicate_ids, "duplicate_samples": self.duplicate_samples,
            "invalid_ids": self.invalid_ids, "invalid_samples": self.invalid_samples,
            "missing_external_ids": missing, "missing_external_samples": missing_samples,
            "first_missing_external_id": first_missing, "last_missing_external_id": last_missing,
            "missing_ids_are_contiguous_tail": missing_contiguous_tail,
            "mappings_outside_physical_slots": self.missing_slots,
            "unique_outside_offsets": self.outside_offsets.len(),
            "outside_offset_range":[min_outside,max_outside],
            "outside_offsets_are_contiguous":match (min_outside,max_outside) {
                (Some(lo),Some(hi)) => hi as usize - lo as usize + 1 == self.outside_offsets.len(),
                _ => false
            },
            "missing_slot_samples": self.missing_slot_samples
        })
    }
}

#[test]
fn sisap23_mapping_diagnostic_classifies_tail_and_gaps() {
    let mut tail = MappingAudit::new(5);
    for i in 1..=3 {
        tail.observe(i, (i - 1) as u32, 3);
    }
    let report = tail.finish();
    assert_eq!(report["missing_external_ids"], 2);
    assert_eq!(report["missing_ids_are_contiguous_tail"], true);
    let mut gap = MappingAudit::new(5);
    for i in [1, 2, 4, 5] {
        gap.observe(i, (i - 1) as u32, 3);
    }
    let report = gap.finish();
    assert_eq!(report["missing_external_ids"], 1);
    assert_eq!(report["missing_ids_are_contiguous_tail"], false);
    assert_eq!(report["mappings_outside_physical_slots"], 2);
    gap.observe(4, 1, 3);
    assert_eq!(gap.finish()["duplicate_ids"], 1);
}

fn vector_file_metadata(path: &Path) -> io::Result<Vec<serde_json::Value>> {
    let mut files = Vec::new();
    fn walk(base: &Path, path: &Path, out: &mut Vec<serde_json::Value>) -> io::Result<()> {
        for entry in fs::read_dir(path).map_err(|e| path_error(path, "read directory", e))? {
            let entry = entry.map_err(|e| path_error(path, "read entry in", e))?;
            let child = entry.path();
            let meta = fs::symlink_metadata(&child).map_err(|e| path_error(&child, "stat", e))?;
            if meta.is_dir() {
                walk(base, &child, out)?;
            } else if meta.is_file() {
                out.push(serde_json::json!({
                    "path": child.strip_prefix(base).unwrap().to_string_lossy(),
                    "bytes": meta.len()
                }));
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unexpected symlink or special file: {}", child.display()),
                ));
            }
        }
        Ok(())
    }
    walk(path, path, &mut files)?;
    files.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    Ok(files)
}

fn diagnostic_destination_is_safe(output: &Path, copied_collection: &Path) -> bool {
    !output.exists()
        && !output.starts_with(copied_collection)
        && output
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".json"))
        && !output.to_string_lossy().contains("build.complete")
}

#[test]
fn sisap23_diagnostic_rejects_completion_marker_destinations() {
    let copied = Path::new("/scratch/copied-collection");
    assert!(!diagnostic_destination_is_safe(
        Path::new("/results/runs/plain/build.complete.json"),
        copied
    ));
    assert!(!diagnostic_destination_is_safe(
        Path::new("/scratch/copied-collection/report.json"),
        copied
    ));
    assert!(diagnostic_destination_is_safe(
        Path::new("/results/diagnostics/plain.job.json"),
        copied
    ));
}

#[test]
#[ignore = "explicit diagnostic on a verified scratch copy; never writes a completion marker"]
fn sisap23_matched_diagnose_plain() {
    assert_eq!(std::env::var("SISAP23_DIAG_MODE").as_deref(), Ok("1"));
    let scratch = PathBuf::from(std::env::var("SCRATCHDIR").expect("PBS scratch required"));
    let collection_path = PathBuf::from(
        std::env::var("SISAP23_DIAG_COLLECTION_PATH").expect("verified scratch copy path required"),
    );
    assert!(collection_path.starts_with(&scratch) && collection_path != scratch);
    let output =
        PathBuf::from(std::env::var("SISAP23_DIAG_OUTPUT").expect("diagnostic output required"));
    assert!(
        diagnostic_destination_is_safe(&output, &collection_path),
        "unsafe diagnostic report destination"
    );
    let segment_path = locate_segment(&collection_path);
    assert_eq!(
        SegmentVersion::load_universal(&MmapFs, &segment_path).unwrap(),
        Some(SegmentVersion::current()),
        "refuse segment migration even on scratch copy"
    );
    let segment = load_index(&segment_path);
    let config = &segment.segment_config.vector_data[DEFAULT_VECTOR_NAME];
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let tracker = segment.id_tracker.borrow();
    let physical = storage.total_vector_count();
    let mut mappings = MappingAudit::new(N);
    let mut bad_reverse = 0usize;
    let mut missing_versions = 0usize;
    let mut min_version = None::<u64>;
    let mut max_version = None::<u64>;
    let mut deleted_references = 0usize;
    let mut absent_vectors = 0usize;
    let mut bad_reverse_samples = Vec::new();
    let mut missing_version_samples = Vec::new();
    let mut deleted_samples = Vec::new();
    let mut absent_samples = Vec::new();
    let selected = [
        1usize,
        2,
        3,
        N / 2,
        physical.saturating_sub(2),
        physical.saturating_sub(1),
        physical,
        physical + 1,
        N - 2,
        N - 1,
        N,
    ];
    let mut selected_offsets = Vec::new();
    for (external, internal) in tracker.point_mappings().iter_from(None) {
        let id = match external {
            PointIdType::NumId(v) => v,
            PointIdType::Uuid(_) => 0,
        };
        mappings.observe(id, internal, physical);
        if tracker.external_id(internal) != Some(external) {
            bad_reverse += 1;
            sample_push(&mut bad_reverse_samples, (id, internal));
        }
        if let Some(version) = tracker.internal_version(internal) {
            min_version = Some(min_version.map_or(version, |v| v.min(version)));
            max_version = Some(max_version.map_or(version, |v| v.max(version)));
        } else {
            missing_versions += 1;
            sample_push(&mut missing_version_samples, (id, internal));
        }
        if (internal as usize) < physical && storage.is_deleted_vector(internal) {
            deleted_references += 1;
            sample_push(&mut deleted_samples, (id, internal));
        }
        if (internal as usize) < physical && storage.get_vector_opt::<Random>(internal).is_none() {
            absent_vectors += 1;
            sample_push(&mut absent_samples, (id, internal));
        }
        if selected.contains(&(id as usize)) {
            selected_offsets.push((id, internal));
        }
    }
    let mapping_report = mappings.finish();
    let mut comparisons = Vec::new();
    let corpus_path = root().join("corpus.f16le");
    let mut corpus = File::open(&corpus_path).unwrap();
    for (id, offset) in selected_offsets {
        if id == 0 || id > N as u64 || offset as usize >= physical {
            continue;
        }
        let mut raw = vec![0u8; DIM * 2];
        corpus
            .seek(std::io::SeekFrom::Start((id - 1) * (DIM * 2) as u64))
            .unwrap();
        corpus.read_exact(&mut raw).unwrap();
        let source: Vec<f32> = raw
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect();
        let normalized = Distance::Cosine.preprocess_vector::<f32>(source.clone());
        let expected: Vec<f32> = normalized
            .iter()
            .map(|x| f16::from_f32(*x).to_f32())
            .collect();
        let Some(stored) = storage.get_vector_opt::<Random>(offset) else {
            comparisons.push(serde_json::json!({"id":id,"offset":offset,"status":"vector_absent"}));
            continue;
        };
        let stored = stored.to_owned();
        let segment::data_types::vectors::VectorInternal::Dense(actual) = stored else {
            panic!("diagnostic expected dense vector")
        };
        let max_abs_error = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let raw_max_abs_error = actual
            .iter()
            .zip(source.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        comparisons.push(serde_json::json!({
            "id":id,"offset":offset,"dimension":actual.len(),
            "max_abs_error_vs_cosine_normalized_float16":max_abs_error,
            "max_abs_error_vs_input_float16":raw_max_abs_error,
            "within_normalized_tolerance_0_002":max_abs_error <= 0.002
        }));
    }
    let vector_dir = segment_path.join("vector_storage").join("vectors");
    let persisted_chunk_config = fs::read(vector_dir.join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let status_bytes = fs::read(vector_dir.join("status.dat")).ok();
    let persisted_status_len = status_bytes.as_ref().and_then(|bytes| {
        (bytes.len() == std::mem::size_of::<usize>())
            .then(|| usize::from_ne_bytes(bytes.as_slice().try_into().unwrap()))
    });
    let file_metadata = vector_file_metadata(&segment_path.join("vector_storage"))
        .expect("cannot inspect copied vector storage files");
    let safe_to_search = mappings.missing_slots == 0
        && absent_vectors == 0
        && bad_reverse == 0
        && missing_versions == 0;
    drop(tracker);
    drop(storage);
    drop(index);
    let plain_search = if safe_to_search
        && index_kind(
            &segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow(),
        ) == "plain"
    {
        let warm = read_f32(&root().join("queries_warmup.f32le"), 20);
        let (ids, _) = search_one(&segment, Variant::Plain, 0, &warm[0]);
        let all_ids_valid = ids
            .iter()
            .all(|id| (1..=N as u64).contains(id) && mappings.seen[(*id - 1) as usize]);
        let distinct_ids = ids.iter().collect::<std::collections::HashSet<_>>().len() == ids.len();
        serde_json::json!({"status":"completed","returned_ids_1_based":ids,
            "returned_count":ids.len(),"all_returned_ids_mapped":all_ids_valid,
            "distinct_returned_ids":distinct_ids,
            "storage_available_vectors":physical.saturating_sub(segment.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow().deleted_vector_count()),
            "independent_full_corpus_reference":"not_run",
            "note":"successful native search does not prove corpus completeness or exact recall"})
    } else {
        serde_json::json!({"status":"skipped","reason":"unsafe or inconsistent mappings, or index is not Plain"})
    };
    let report = serde_json::json!({
        "format":"sisap23-plain-diagnostic-v1",
        "source_commit":std::env::var("SISAP23_DIAG_COMMIT").unwrap(),
        "expected_points":N,"segment_path":segment_path,
        "segment_config":{"dimension":config.size,"distance":format!("{:?}",config.distance),
            "datatype":format!("{:?}",config.datatype),"storage_type":format!("{:?}",config.storage_type),
            "index_kind":index_kind(&segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow())},
        "storage":{"total_vectors":physical,"available_vectors":segment.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow().available_vector_count(),
            "deleted_vectors":segment.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow().deleted_vector_count(),
            "offset_range":if physical == 0 { serde_json::Value::Null } else { serde_json::json!([0,physical-1]) },
            "persisted_status_len":persisted_status_len,"persisted_status_bytes":status_bytes.as_ref().map(Vec::len),
            "persisted_chunk_config":persisted_chunk_config,"files":file_metadata},
        "tracker":{"total_points":segment.id_tracker.borrow().total_point_count(),
            "available_points":segment.id_tracker.borrow().available_point_count(),
            "deleted_points":segment.id_tracker.borrow().deleted_point_count(),
            "mapping_audit":mapping_report,
            "reverse_mapping_failures":bad_reverse,"reverse_mapping_samples":bad_reverse_samples,
            "missing_versions":missing_versions,"missing_version_samples":missing_version_samples,
            "point_version_range":[min_version,max_version],
            "deleted_vector_references":deleted_references,"deleted_vector_samples":deleted_samples,
            "absent_in_range_vectors":absent_vectors,"absent_vector_samples":absent_samples},
        "vector_comparisons":{"selection":"first, midpoint, physical boundary, and final IDs",
            "compared":comparisons.len(),"items":comparisons},
        "plain_search":plain_search,
        "conclusion":"diagnostic only; no completion marker; acceptance requires independent review"
    });
    drop(segment);
    write_json_new(&output, &report);
    println!("sisap23_plain_diagnostic_report={}", output.display());
}

fn selected_settings(v: Variant) -> Vec<(String, usize)> {
    match v {
        Variant::Plain => vec![("full".to_string(), 0)],
        Variant::Lmi => LMI_PROBES
            .iter()
            .map(|p| (format!("nprobe-{p}"), *p))
            .collect(),
        Variant::HnswA | Variant::HnswB => HNSW_EF
            .iter()
            .map(|e| (format!("hnsw-ef-{e}"), *e))
            .collect(),
    }
}
fn search_one(
    segment: &segment::segment::Segment,
    v: Variant,
    setting: usize,
    values: &[f32],
) -> (Vec<u64>, Option<usize>) {
    let idx = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let tracker = segment.id_tracker.borrow();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let query: QueryVector = values.to_vec().into();
    let ctx = VectorQueryContext::default();
    let hits = match v {
        Variant::Lmi => {
            let VectorIndexEnum::Lmi(lmi) = &*idx else {
                panic!("persisted LMI index not found")
            };
            let state = lmi.routing_state().expect("LMI routing state missing");
            let normalized = Distance::Cosine.preprocess_vector::<f32>(values.to_vec());
            let buckets = state.router().top_buckets(&normalized, setting).unwrap();
            let mut offsets = Vec::new();
            for b in buckets {
                offsets.extend_from_slice(state.postings().get(b).unwrap());
            }
            offsets.sort_unstable();
            offsets.dedup();
            let count = offsets.len();
            let scored =
                score_candidates(&query, &offsets, 10, &*tracker, &*storage, &ctx).unwrap();
            let ids = scored
                .iter()
                .map(|h| match tracker.external_id(h.idx).unwrap() {
                    PointIdType::NumId(id) => id,
                    _ => panic!("expected numeric SISAP ids"),
                })
                .collect();
            return (ids, Some(count));
        }
        Variant::HnswA | Variant::HnswB => {
            let VectorIndexEnum::Hnsw(hnsw) = &*idx else {
                panic!("persisted HNSW graph index not found")
            };
            let p = SearchParams {
                hnsw_ef: Some(setting),
                ..Default::default()
            };
            hnsw.search(&[&query], None, 10, Some(&p), &ctx)
                .unwrap()
                .remove(0)
        }
        Variant::Plain => {
            let VectorIndexEnum::Plain(plain) = &*idx else {
                panic!("persisted Plain exact index not found")
            };
            plain
                .search(&[&query], None, 10, None, &ctx)
                .unwrap()
                .remove(0)
        }
    };
    let ids = hits
        .iter()
        .map(|h| match tracker.external_id(h.idx).unwrap() {
            PointIdType::NumId(id) => id,
            _ => panic!("expected numeric SISAP ids"),
        })
        .collect();
    (ids, if v == Variant::Plain { Some(N) } else { None })
}
fn write_chunk(path: &Path, rows: &[serde_json::Value]) {
    let tmp = path.with_extension("jsonl.tmp");
    if tmp.exists() {
        fs::remove_file(&tmp).expect("cannot remove incomplete chunk temporary");
    }
    let mut out = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .expect("partial chunk exists; preserve it"),
    );
    for r in rows {
        serde_json::to_writer(&mut out, r).unwrap();
        out.write_all(b"\n").unwrap();
    }
    out.flush().unwrap();
    out.get_ref().sync_all().unwrap();
    fs::rename(tmp, path).unwrap();
}
fn recall(ids: &[u64], gold: &[i32]) -> f64 {
    ids.iter()
        .filter(|id| gold.contains(&(**id as i32)))
        .count() as f64
        / 10.0
}
fn run_setting(
    segment: &segment::segment::Segment,
    v: Variant,
    label: &str,
    effort: usize,
    query_rows: &[(usize, Vec<f32>)],
    gold: &[Vec<i32>],
    dest: &Path,
) {
    if dest.join("condition.complete.json").exists() {
        let c: serde_json::Value =
            serde_json::from_slice(&fs::read(dest.join("condition.complete.json")).unwrap())
                .unwrap();
        assert_eq!(c["setting"], label);
        assert_eq!(c["query_count"], query_rows.len());
        let saved = fs::read_to_string(dest.join("rows.jsonl")).unwrap();
        let records: Vec<serde_json::Value> = saved
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), query_rows.len());
        for (record, (row, _)) in records.iter().zip(query_rows.iter()) {
            assert_eq!(record["query_row"], *row);
            assert_eq!(record["parameter"], label);
        }
        return;
    }
    fs::create_dir_all(dest).unwrap();
    let warm = read_f32(&root().join("queries_warmup.f32le"), 20);
    for q in &warm {
        let _ = search_one(segment, v, effort, q);
    }
    let chunks = dest.join("chunks");
    fs::create_dir_all(&chunks).unwrap();
    for (chunk_idx, part) in query_rows.chunks(CHUNK).enumerate() {
        let path = chunks.join(format!("rows-{chunk_idx:05}.jsonl"));
        if path.exists() {
            let existing = fs::read_to_string(&path).expect("cannot inspect saved chunk");
            let saved: Vec<serde_json::Value> = existing
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert_eq!(saved.len(), part.len(), "saved chunk row count mismatch");
            for (r, (expected, _)) in saved.iter().zip(part.iter()) {
                assert_eq!(r["query_row"], *expected, "saved chunk query mismatch");
                assert_eq!(r["parameter"], label, "saved chunk parameter mismatch");
            }
            continue;
        }
        let mut records = Vec::with_capacity(part.len());
        for (row, values) in part {
            let started = Instant::now();
            let (ids, candidates) = search_one(segment, v, effort, values);
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let g = &gold[row - 20];
            records.push(serde_json::json!({"query_row":row,"parameter":label,"recall_at_10":recall(&ids,g),"candidate_count":candidates,"hnsw_ef":if v==Variant::HnswA||v==Variant::HnswB{Some(effort)}else{None},"native_search_ms":ms,"returned_ids_1_based":ids}));
        }
        write_chunk(&path, &records);
    }
    let rows_path = dest.join("rows.jsonl");
    if rows_path.exists() {
        let saved = fs::read_to_string(&rows_path).expect("cannot inspect assembled rows");
        assert_eq!(
            saved.lines().count(),
            query_rows.len(),
            "assembled rows count mismatch"
        );
        for (line, (row, _)) in saved.lines().zip(query_rows.iter()) {
            let r: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(r["query_row"], *row);
            assert_eq!(r["parameter"], label);
        }
        write_json_new(
            &dest.join("condition.complete.json"),
            &serde_json::json!({"setting":label,"query_count":query_rows.len(),"complete":true,"recovered_after_assembly":true}),
        );
        return;
    }
    let tmp = dest.join("rows.jsonl.tmp");
    if tmp.exists() {
        fs::remove_file(&tmp).expect("cannot remove incomplete assembly temporary");
    }
    let mut out = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .expect("partial assembled result exists"),
    );
    let n_chunks = query_rows.len().div_ceil(CHUNK);
    for i in 0..n_chunks {
        let mut f =
            File::open(chunks.join(format!("rows-{i:05}.jsonl"))).expect("missing completed chunk");
        std::io::copy(&mut f, &mut out).unwrap();
    }
    out.flush().unwrap();
    out.get_ref().sync_all().unwrap();
    fs::rename(tmp, rows_path).unwrap();
    write_json_new(
        &dest.join("condition.complete.json"),
        &serde_json::json!({"setting":label,"query_count":query_rows.len(),"chunks":n_chunks,"complete":true}),
    );
}

#[test]
#[ignore = "explicit SISAP23 matched native evaluation sweeps; requires built 10M generation"]
fn sisap23_matched_evaluate() {
    let v = Variant::parse(&std::env::var("SISAP23_VARIANT").expect("SISAP23_VARIANT required"));
    let run = run_dir(v);
    let imported = v == Variant::Plain && run.join("imported.generation.json").exists();
    let marker_path = if imported {
        run.join("imported.generation.json")
    } else {
        run.join("build.complete.json")
    };
    let build: serde_json::Value = serde_json::from_slice(
        &fs::read(&marker_path).expect("verified generation marker required"),
    )
    .unwrap();
    assert_eq!(build["variant"], v.name());
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root().join("manifest.json")).unwrap()).unwrap();
    if imported {
        assert_eq!(build["format"], "sisap23-imported-plain-v1");
        assert_eq!(
            build["source_commit"],
            "42a163701a51078ec24983b3a93f96ab7d661489"
        );
        let generation_path = PathBuf::from(build["generation_path"].as_str().unwrap());
        let generation_bytes = fs::read(generation_path.join("generation.json")).unwrap();
        assert_eq!(
            Sha256::digest(&generation_bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            build["generation_json_sha256"].as_str().unwrap()
        );
        let generation: serde_json::Value = serde_json::from_slice(&generation_bytes).unwrap();
        assert_eq!(generation["accepted"], true);
        assert_eq!(generation["preservation_commit"], manifest["source_commit"]);
        assert_eq!(
            build["segment_path"],
            locate_segment(&generation_path.join("collection"))
                .to_string_lossy()
                .as_ref()
        );
    } else {
        assert_eq!(build["source_commit"], manifest["source_commit"]);
    }
    assert_eq!(
        manifest["source_commit"].as_str(),
        std::env::var("SISAP23_REVIEWED_COMMIT").ok().as_deref()
    );
    let segment_path = PathBuf::from(build["segment_path"].as_str().unwrap());
    let segment = load_index(&segment_path);
    let idx = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let expected = if v == Variant::Plain {
        "plain"
    } else if v == Variant::Lmi {
        "lmi"
    } else {
        "hnsw"
    };
    assert_eq!(
        index_kind(&idx),
        expected,
        "index generation does not match requested algorithm"
    );
    assert_eq!(manifest["format"], "sisap23-matched-v1");
    let queries = read_f32(&root().join("queries_20_9999.f32le"), 9980);
    let warm = read_f32(&root().join("queries_warmup.f32le"), 20);
    let gold_bytes = fs::read(root().join("gold_20_9999_top10.i32le")).unwrap();
    assert_eq!(gold_bytes.len(), 9980 * 10 * 4);
    let gold: Vec<Vec<i32>> = gold_bytes
        .chunks_exact(40)
        .map(|r| {
            r.chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect();
    for q in &warm {
        let _ = search_one(&segment, v, selected_settings(v)[0].1, q);
    }
    let measured: Vec<(usize, Vec<f32>)> = queries
        .into_iter()
        .enumerate()
        .map(|(i, q)| (i + 20, q))
        .collect();
    let full = run.join("full");
    fs::create_dir_all(&full).unwrap();
    for (label, effort) in selected_settings(v) {
        run_setting(
            &segment,
            v,
            &label,
            effort,
            &measured,
            &gold,
            &full.join(&label),
        );
    }
    // Three paired warm panels use one shared query order; rotate setting order to reduce
    // systematic cache-order bias. Panel query IDs are frozen in the prepared manifest.
    let panel_bytes = fs::read(root().join("panel_rows.u32le")).unwrap();
    assert_eq!(panel_bytes.len(), 1000 * 4);
    let panel_ids: Vec<usize> = panel_bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect();
    let panel: Vec<(usize, Vec<f32>)> = panel_ids
        .iter()
        .map(|row| {
            (
                *row,
                read_query_at(&root().join("queries_20_9999.f32le"), row - 20),
            )
        })
        .collect();
    let settings = selected_settings(v);
    for rep in 0..3 {
        let mut rotated = settings.clone();
        let shift = rep % rotated.len();
        rotated.rotate_left(shift);
        for (label, effort) in rotated {
            let dest = run
                .join("panel")
                .join(format!("rep-{}", rep + 1))
                .join(&label);
            run_setting(&segment, v, &label, effort, &panel, &gold, &dest);
        }
    }
    println!("sisap23_native_evaluation_complete variant={}", v.name());
}
fn read_query_at(path: &Path, row: usize) -> Vec<f32> {
    let mut f = File::open(path).unwrap();
    f.seek(std::io::SeekFrom::Start((row * DIM * 4) as u64))
        .unwrap();
    let mut bytes = vec![0u8; DIM * 4];
    f.read_exact(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[path = "lmi_sisap23_matched_wal_replay.rs"]
mod wal_replay;
