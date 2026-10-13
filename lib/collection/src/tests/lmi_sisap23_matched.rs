//! Opt-in matched SISAP23 Plain/HNSW/LMI benchmark using native Qdrant index paths.
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ahash::AHashMap;
use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use half::f16;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{QueryVector, DEFAULT_VECTOR_NAME};
use segment::id_tracker::IdTrackerRead;
use segment::index::candidate_scoring::score_candidates;
use segment::index::lmi_index::{LmiConfig, LmiStateMetadata};
use segment::index::VectorIndexEnum;
use segment::index::VectorIndexRead;
use segment::segment_constructor::load_segment;
use segment::types::{Distance, HnswConfig, PointIdType, SearchParams};
use sha2::{Digest, Sha256};
use shard::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorStructPersisted,
};
use shard::operations::CollectionUpdateOperations;
use uuid::Uuid;

use crate::collection::Collection;
use crate::config::{CollectionConfigInternal, CollectionParams};
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::operations::types::{Datatype, VectorsConfig};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::operations::OperationWithClockTag;
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
    fs::rename(tmp, path).unwrap();
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
fn count_bytes(path: &Path) -> u64 {
    if path.is_file() {
        return fs::metadata(path).unwrap().len();
    }
    fs::read_dir(path)
        .unwrap()
        .map(|e| count_bytes(&e.unwrap().path()))
        .sum()
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
        "collection_bytes": count_bytes(&collection_path),
        "segment_bytes": count_bytes(&segment_path),
        "vector_index_bytes": count_bytes(&segment_path.join("vector_index")),
        "generation": "directory is immutable after this marker; all sweeps reuse this path"
    });
    drop(index);
    drop(segment);
    write_json_new(&run.join("build.complete.json"), &marker);
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
    let build: serde_json::Value = serde_json::from_slice(
        &fs::read(run.join("build.complete.json")).expect("build completion marker required"),
    )
    .unwrap();
    assert_eq!(build["variant"], v.name());
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(build["source_commit"], manifest["source_commit"]);
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
