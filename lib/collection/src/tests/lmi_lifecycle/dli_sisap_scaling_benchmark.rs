//! Long-running matched SISAP23 comparison. Invoke explicitly with
//! `--ignored --exact tests::lmi_lifecycle::dli_sisap300k_benchmark::sisap23_scaling_static_lmi_vs_temporal_dli`.
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use half::f16;
use segment::entry::ReadSegmentEntry;
use segment::index::VectorIndexEnum;

use super::*;
use crate::config::DliExperimentalConfig;
use crate::operations::types::Datatype;

const DIMENSIONS: usize = 768;
const FULL_COUNT: usize = 10_120_191;
const BATCH_SIZE: usize = 250_000;
const DLI_BUFFER_CAPACITY: usize = 250_000;
const QUERY_COUNT: usize = 100;
const DYNAMIC_QUERY_COUNT: usize = 100;
const DYNAMIC_UPSERTS: usize = 128;
const DYNAMIC_DELETIONS: usize = 128;
const SMALL_NATIVE_POPULATION: usize = 65_536;
const SMALL_NATIVE_BUFFER: usize = 20_000;

async fn replica(collection: &Collection) -> Arc<crate::shards::replica_set::ShardReplicaSet> {
    collection
        .shards_holder()
        .read()
        .await
        .get_shard(0)
        .unwrap()
        .clone()
}

fn model_config() -> LmiConfig {
    LmiConfig {
        n_buckets: 548,
        sample_size: 32_768,
        hidden_dim: 512,
        epochs: 30,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations: 5,
        nprobe: 4,
        seed: 42,
        reuse_router: None,
    }
}

/// Preregistered arm (`research/dli/emma_path/PREREGISTRATION.md`); unset means G0,
/// the original configuration, so earlier campaigns are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    G0,
    E1,
    E1R,
}

fn arm() -> Arm {
    match std::env::var("DLI_ARM").as_deref() {
        Err(_) | Ok("G0") => Arm::G0,
        Ok("E1") => Arm::E1,
        Ok("E1R") => Arm::E1R,
        Ok(other) => panic!("DLI_ARM must be G0, E1 or E1R, got {other}"),
    }
}

/// (buffer capacity, arity, base model, per-level rule, collection routing policy).
fn arm_config() -> (
    usize,
    usize,
    LmiConfig,
    crate::collection_manager::dli_schedule::LevelModelPolicy,
    crate::collection_manager::dli_bucket_plan::RoutingPolicy,
) {
    use crate::collection_manager::dli_bucket_plan::RoutingPolicy;
    use crate::collection_manager::dli_schedule::{LevelBuckets, LevelModelPolicy};
    match arm() {
        Arm::G0 => (
            DLI_BUFFER_CAPACITY,
            2,
            model_config(),
            LevelModelPolicy::default(),
            RoutingPolicy::SlotWeighted,
        ),
        // Sommerová `bs_10_5k.yml` + her code defaults (hidden 128, Faiss 25 iterations,
        // 100K preferred sample with the 40·k floor).
        arm @ (Arm::E1 | Arm::E1R) => (
            5_000,
            10,
            emma_model_config(),
            LevelModelPolicy {
                buckets: LevelBuckets::ArityPower,
                min_samples_per_bucket: 40,
                reuse_target_router: arm == Arm::E1,
            },
            RoutingPolicy::ModelDrivenNormalizedCode,
        ),
    }
}

/// Base model for the E arms; `n_buckets` is replaced by arity^level per level.
fn emma_model_config() -> LmiConfig {
    LmiConfig {
        n_buckets: 10,
        sample_size: 100_000,
        hidden_dim: 128,
        epochs: 10,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations: 25,
        nprobe: 1,
        seed: 42,
        reuse_router: None,
    }
}

fn directory_file_bytes(path: &Path) -> u64 {
    std::fs::read_dir(path)
        .unwrap_or_else(|error| panic!("cannot inspect {}: {error}", path.display()))
        .map(|entry| entry.unwrap())
        .map(|entry| {
            let path = entry.path();
            let kind = entry.file_type().unwrap();
            if kind.is_dir() {
                directory_file_bytes(&path)
            } else if kind.is_file() {
                entry.metadata().unwrap().len()
            } else {
                0
            }
        })
        .sum()
}

#[test]
fn recursive_storage_measurement_includes_nested_mmap_chunks() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("vectors")).unwrap();
    std::fs::write(root.path().join("quantized.data"), [0; 7]).unwrap();
    std::fs::write(root.path().join("vectors/chunk_0.mmap"), [0; 13]).unwrap();
    assert_eq!(directory_file_bytes(root.path()), 20);
}

fn collection_config(dli: bool) -> CollectionConfigInternal {
    let mut vectors = VectorParamsBuilder::new(DIMENSIONS as u64, Distance::Cosine)
        .with_on_disk(true)
        .with_datatype(Datatype::Float16)
        .build();
    let (buffer_capacity, arity, model, level_model, routing_policy) = arm_config();
    vectors.lmi_config = (!dli).then_some(model.clone());
    let mut optimizer_config = OptimizersConfig::fixture();
    // The comparison baseline uses Qdrant's ordinary optimizer. DLI owns its
    // learned-level transitions and must not have a second optimizer rewrite
    // its logical level segments into unrelated HNSW generations.
    optimizer_config.indexing_threshold = (!dli).then_some(20_000);
    optimizer_config.default_segment_number = 1;
    optimizer_config.max_optimization_threads = if dli { Some(0) } else { Some(1) };
    CollectionConfigInternal {
        params: CollectionParams {
            vectors: VectorsConfig::Single(vectors),
            shard_number: NonZeroU32::new(1).unwrap(),
            ..CollectionParams::empty()
        },
        hnsw_config: Default::default(),
        optimizer_config,
        wal_config: Default::default(),
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
        created_at: None,
        dli_experimental: dli.then_some(DliExperimentalConfig {
            materialize_topk: false,
            quantized_shortlist: None,
            routing_policy,
            maintenance_policy: if std::env::var("DLI_BENCH_POLICY").as_deref() == Ok("size-tiered")
            {
                crate::collection_manager::dli_schedule::MaintenancePolicy::SizeTiered
            } else {
                Default::default()
            },
            level_model,
            greedy_rebuild: std::env::var("DLI_GREEDY_REBUILD").as_deref() == Ok("1"),
            vacuum_dead_permille: 0,
            buffer_capacity,
            arity,
            candidate_budget: std::env::var("DLI_BENCH_BUDGET")
                .map_or(2_000, |v| v.parse().expect("DLI_BENCH_BUDGET")),
            model,
        }),
    }
}

async fn create_collection(name: &str, root: &Path, dli: bool) -> Collection {
    create_collection_with_config(name, root, collection_config(dli)).await
}

async fn create_collection_with_config(
    name: &str,
    root: &Path,
    config: CollectionConfigInternal,
) -> Collection {
    let data = root.join("storage");
    let snapshots = root.join("snapshots");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&snapshots).unwrap();
    let collection = Collection::new(
        name.to_owned(),
        1,
        &data,
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
    collection
}

fn point_from_row(row: usize, id: u64, vectors: &[u8]) -> PointStructPersisted {
    let begin = row * DIMENSIONS * 2;
    let source = &vectors[begin..begin + DIMENSIONS * 2];
    let vector = source
        .chunks_exact(2)
        .map(|half| f16::from_bits(u16::from_le_bytes([half[0], half[1]])).to_f32())
        .collect();
    PointStructPersisted {
        id: id.into(),
        vector: VectorStructPersisted::Single(vector),
        payload: None,
    }
}

async fn apply_points(collection: &Collection, points: Vec<PointStructPersisted>) -> Duration {
    let started = Instant::now();
    let result = replica(collection)
        .await
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(points)),
            )),
            WaitUntil::Visible,
            None,
            false,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        result.status,
        crate::operations::types::UpdateStatus::Completed
    );
    started.elapsed()
}

async fn apply_delete(collection: &Collection, ids: Vec<u64>) -> Duration {
    let started = Instant::now();
    let result = replica(collection)
        .await
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::DeletePoints {
                    ids: ids.into_iter().map(Into::into).collect(),
                },
            )),
            WaitUntil::Visible,
            None,
            false,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        result.status,
        crate::operations::types::UpdateStatus::Completed
    );
    started.elapsed()
}

async fn wait_for_dli_publications(
    replica: &crate::shards::replica_set::ShardReplicaSet,
    target: usize,
) {
    let started = Instant::now();
    let mut last_report = Duration::ZERO;
    // Unset keeps 30 min. A fresh level with a^i buckets can need longer (E1 at 10M:
    // publication 1100 trains 10,000 buckets, projected 45 min).
    let limit = std::env::var("DLI_BENCH_PUBLICATION_TIMEOUT_S")
        .map_or(30 * 60, |v| v.parse().expect("DLI_BENCH_PUBLICATION_TIMEOUT_S"));
    let result = tokio::time::timeout(Duration::from_secs(limit), async {
        loop {
            let diagnostics = replica.local_dli_diagnostics_for_test().await.unwrap();
            if diagnostics.1 >= target {
                break;
            }
            if !diagnostics.4.is_empty() {
                panic!(
                    "DLI maintenance failed before publication {target}: diagnostics={diagnostics:?}"
                );
            }
            if started.elapsed().saturating_sub(last_report) >= Duration::from_secs(30) {
                println!(
                    "SISAP23_DLI_WAIT target={target} elapsed_s={} diagnostics={diagnostics:?}",
                    started.elapsed().as_secs()
                );
                last_report = started.elapsed();
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    if result.is_err() {
        let diagnostics = replica.local_dli_diagnostics_for_test().await;
        panic!("timed out waiting for DLI publication {target}: diagnostics={diagnostics:?}")
    }
}

async fn wait_for_static_index(collection: &Collection, target: usize) -> Duration {
    let started = Instant::now();
    let mut last_report = Duration::ZERO;
    tokio::time::timeout(Duration::from_secs(30 * 60), async {
        loop {
            let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
            let indexed = info.indexed_vectors_count.unwrap_or_default();
            let lmi = active_static_lmi_stats(collection).await;
            if indexed >= target && lmi.0 > 0 && lmi.1 >= target {
                return;
            }
            if started.elapsed().saturating_sub(last_report) >= Duration::from_secs(30) {
                println!(
                    "SISAP23_STATIC_WAIT target={target} indexed={indexed} points={:?} active_lmi={lmi:?} elapsed_s={}",
                    info.points_count,
                    started.elapsed().as_secs()
                );
                last_report = started.elapsed();
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for static LMI indexing through {target}"));
    started.elapsed()
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * p).ceil() as usize;
    values[index]
}

fn query_process_counters() -> Option<(u64, u64, u64)> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let (_, tail) = stat.rsplit_once(')')?;
    let fields: Vec<_> = tail.split_whitespace().collect();
    let n = |i: usize| fields.get(i)?.parse::<u64>().ok();
    Some((n(7)?, n(9)?, n(11)?.checked_add(n(12)?)?))
}

async fn run_queries(
    collection: &Collection,
    query_data: &[u8],
    gold: &[u8],
    query_count: usize,
    deleted_ids: &HashSet<u32>,
) -> (f64, f64, f64, f64, f64) {
    run_queries_params(collection, query_data, gold, query_count, deleted_ids, None).await
}

async fn run_queries_params(
    collection: &Collection,
    query_data: &[u8],
    gold: &[u8],
    query_count: usize,
    deleted_ids: &HashSet<u32>,
    params: Option<segment::types::SearchParams>,
) -> (f64, f64, f64, f64, f64) {
    let detailed = std::env::var_os("DLI_DETAILED_QUERY_STATS").is_some();
    let local_profile = if detailed {
        Some(replica(collection).await)
    } else {
        None
    };
    let mut timings = Vec::with_capacity(query_count);
    let mut hits10 = 0usize;
    let mut hits30 = 0usize;
    for query_idx in 0..query_count {
        let begin = query_idx * DIMENSIONS * 4;
        let vector: Vec<_> = query_data[begin..begin + DIMENSIONS * 4]
            .chunks_exact(4)
            .map(|f| f32::from_le_bytes([f[0], f[1], f[2], f[3]]))
            .collect();
        let profile_before = if let Some(local) = &local_profile {
            local.local_dli_named_profile_for_test().await
        } else {
            None
        };
        let cpu_before = if detailed {
            query_process_counters()
        } else {
            None
        };
        let work_before = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
        let started = Instant::now();
        let result = collection
            .search(
                CoreSearchRequest {
                    query: QueryEnum::Nearest(NamedQuery::default_dense(vector)),
                    filter: None,
                    params: params.clone(),
                    limit: 30,
                    offset: 0,
                    with_payload: None,
                    with_vector: None,
                    score_threshold: None,
                },
                None,
                None,
                &ShardSelectorInternal::All,
                // Unset keeps the collection default (60 s). Cold on-disk segments
                // after a 10M ingestion can exceed it on the first queries.
                std::env::var("DLI_BENCH_QUERY_TIMEOUT_S").ok().map(|v| {
                    Duration::from_secs(v.parse().expect("DLI_BENCH_QUERY_TIMEOUT_S"))
                }),
            )
            .await
            .unwrap();
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cpu_after = if detailed {
            query_process_counters()
        } else {
            None
        };
        let cpu_delta = cpu_before.zip(cpu_after).map(|(a, b)| {
            (
                b.0.saturating_sub(a.0),
                b.1.saturating_sub(a.1),
                b.2.saturating_sub(a.2),
            )
        });
        let profile_after = if let Some(local) = &local_profile {
            local.local_dli_named_profile_for_test().await
        } else {
            None
        };
        let mut phases = serde_json::Map::new();
        if let Some((a, b)) = profile_before.zip(profile_after) {
            for (key, v) in b.as_object().unwrap() {
                if let Some((before, after)) = a[key].as_u64().zip(v.as_u64()) {
                    phases.insert(key.clone(), serde_json::json!(after.saturating_sub(before)));
                }
            }
        }
        timings.push(latency_ms);
        if let Ok(path) = std::env::var("DLI_QUERY_EVENTS") {
            use std::io::Write;
            let after = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
            let event = serde_json::json!({"query":query_idx,"latency_ms":latency_ms,
                "scored_vectors":after[2]-work_before[2],"scorer_calls":after[0]-work_before[0],"returned":result.len(),"phases":phases,"process_cpu_ticks":cpu_delta.map(|x|x.2),"process_minor_faults":cpu_delta.map(|x|x.0),"process_major_faults":cpu_delta.map(|x|x.1),"hits":result.iter().map(|h|serde_json::json!({"id":h.id,"score":h.score,"version":h.version})).collect::<Vec<_>>()});
            writeln!(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .unwrap(),
                "{event}"
            )
            .unwrap();
        }

        let mut expected = Vec::with_capacity(30);
        for rank in 0..1_000 {
            let offset = (query_idx * 1_000 + rank) * 4;
            let id = i32::from_le_bytes([
                gold[offset],
                gold[offset + 1],
                gold[offset + 2],
                gold[offset + 3],
            ]);
            assert!(id > 0, "ground truth IDs must be official 1-based values");
            if !deleted_ids.contains(&(id as u32)) {
                expected.push(id as u64);
                if expected.len() == 30 {
                    break;
                }
            }
        }
        assert_eq!(
            expected.len(),
            30,
            "official top-1000 must cover live top-30 after 128 deletions"
        );
        let actual: Vec<u64> = result
            .iter()
            .map(|hit| match hit.id {
                segment::types::PointIdType::NumId(id) => id,
                _ => panic!("SISAP IDs must be numeric"),
            })
            .collect();
        let expected10: HashSet<_> = expected[..10].iter().copied().collect();
        let expected30: HashSet<_> = expected.iter().copied().collect();
        hits10 += actual
            .iter()
            .take(10)
            .filter(|id| expected10.contains(id))
            .count();
        hits30 += actual
            .iter()
            .take(30)
            .filter(|id| expected30.contains(id))
            .count();
    }
    let p50 = percentile(&mut timings, 0.50);
    let p95 = percentile(&mut timings, 0.95);
    (
        hits10 as f64 / (query_count * 10) as f64,
        hits30 as f64 / (query_count * 30) as f64,
        p50,
        p95,
        timings.iter().sum::<f64>() / query_count as f64,
    )
}

async fn active_static_lmi_stats(collection: &Collection) -> (usize, usize, u64, u64) {
    let replica = replica(collection).await;
    let holder = replica.local_segments_for_test().await.unwrap();
    let guard = holder.read();
    let mut lmi_segments = 0;
    let mut indexed_points = 0;
    let mut learned = 0;
    let mut candidates = 0;
    for (_, member) in guard.iter_original() {
        let segment = member.read();
        let Some(data) = segment
            .vector_data
            .get(segment::data_types::vectors::DEFAULT_VECTOR_NAME)
        else {
            continue;
        };
        let index = data.vector_index.borrow();
        if let VectorIndexEnum::Lmi(lmi) = &*index {
            lmi_segments += 1;
            indexed_points += segment.available_point_count_without_deferred();
            let (count, _) = lmi.search_path_counts_for_test();
            learned += count;
            candidates += lmi.selected_candidate_offsets_for_test();
        }
    }
    (lmi_segments, indexed_points, learned, candidates)
}

fn auxiliary_index_bytes(root: &Path) -> u64 {
    fn visit(path: &Path, total: &mut u64) {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, total);
            } else if path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("lmi_"))
            {
                *total += entry
                    .metadata()
                    .map(|metadata| metadata.len())
                    .unwrap_or_default();
            }
        }
    }
    let mut total = 0;
    visit(root, &mut total);
    total
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse().ok())
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "isolated scaling campaign; requires population-specific validated oracle"]
async fn sisap23_scaling_static_lmi_vs_temporal_dli() {
    use std::io::{Read, Seek, SeekFrom};
    let inputs = PathBuf::from(std::env::var("DLI_SISAP_INPUTS").unwrap());
    let output = PathBuf::from(std::env::var("DLI_SISAP_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let population: usize = std::env::var("DLI_BENCH_POINTS").unwrap().parse().unwrap();
    let oracle_population: usize = std::env::var("DLI_ORACLE_POPULATION")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(population, oracle_population);
    // Below 1M the caller must stage population-specific gold as ground_truth.i32.
    assert!(population >= 100_000 && population <= FULL_COUNT);
    let (buffer_capacity, _, _, _, _) = arm_config();
    let chunk = BATCH_SIZE.min(buffer_capacity);
    let system = std::env::var("DLI_SINGLE_SYSTEM").unwrap();
    assert!(["static", "geometric", "size-tiered"].contains(&system.as_str()));
    assert_eq!(
        std::env::var("DLI_BENCH_POLICY").unwrap_or_default() == "size-tiered",
        system == "size-tiered"
    );
    let dli = system != "static";
    let mut source = std::fs::File::open(inputs.join("vectors.f16")).unwrap();
    assert_eq!(
        source.metadata().unwrap().len(),
        (FULL_COUNT * DIMENSIONS * 2) as u64
    );
    let queries = std::fs::read(inputs.join("queries.f32")).unwrap();
    let gold = std::fs::read(inputs.join("ground_truth.i32")).unwrap();
    assert_eq!(queries.len(), 10_000 * DIMENSIONS * 4);
    assert_eq!(gold.len(), 10_000 * 1_000 * 4);
    let root = output.join("collection");
    assert!(
        !root.exists(),
        "preserve previous collection; use a new output directory"
    );
    std::fs::create_dir(&root).unwrap();
    let collection = create_collection(&system, &root, dli).await;
    let local = replica(&collection).await;
    let ingestion = Instant::now();
    let mut checkpoints = Vec::new();
    for first in (0..population).step_by(chunk) {
        let end = (first + chunk).min(population);
        let mut block = vec![0; (end - first) * DIMENSIONS * 2];
        source
            .seek(SeekFrom::Start((first * DIMENSIONS * 2) as u64))
            .unwrap();
        source.read_exact(&mut block).unwrap();
        let points = (first..end)
            .map(|row| point_from_row(row - first, (row + 1) as u64, &block))
            .collect();
        drop(block);
        let old_publications = if dli {
            local
                .local_dli_publication_events_for_test()
                .await
                .unwrap()
                .len()
        } else {
            0
        };
        let write_seconds = apply_points(&collection, points).await.as_secs_f64();
        if dli {
            if end % buffer_capacity == 0 {
                wait_for_dli_publications(&local, old_publications + 1).await;
            }
        } else {
            wait_for_static_index(&collection, end).await;
        }
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        checkpoints.push(
            serde_json::json!({"population":end,"write_seconds":write_seconds,
            "elapsed_seconds":ingestion.elapsed().as_secs_f64(),"points":info.points_count}),
        );
        assert_eq!(info.points_count, Some(end));
        if end % BATCH_SIZE.max(chunk) == 0 || end == population {
            println!(
                "SCALING_CHECKPOINT system={system} population={end} elapsed={:.3}",
                ingestion.elapsed().as_secs_f64()
            );
        }
    }
    let ingestion_seconds = ingestion.elapsed().as_secs_f64();
    let work = segment::index::hnsw_index::point_scorer::scoring_work_snapshot;
    let before = work();
    let metrics = run_queries(&collection, &queries, &gold, QUERY_COUNT, &HashSet::new()).await;
    let after = work();
    let attribution = if dli {
        Some(local.local_dli_search_metrics_for_test().await.unwrap())
    } else {
        None
    };
    let events = if dli {
        local.local_dli_publication_events_for_test().await.unwrap()
    } else {
        Vec::new()
    };
    let (buffer_capacity, arity, model, level_model, routing_policy) = arm_config();
    let result = serde_json::json!({"source_head":std::env::var("DLI_BENCH_HEAD").ok(),"system":system,
        "arm":format!("{:?}", arm()),"buffer_capacity":buffer_capacity,"arity":arity,"model":model,
        "level_model":level_model,"routing_policy":routing_policy,"publication_events":events,
        "population":population,"oracle_population":oracle_population,"checkpoints":checkpoints,
        "ingestion_to_checkpoint_seconds":ingestion_seconds,"recall10":metrics.0,"recall30":metrics.1,
        "query_p50_ms":metrics.2,"query_p95_ms":metrics.3,"query_mean_ms":metrics.4,"query_count":QUERY_COUNT,
        "scorer_calls":after[0]-before[0],"scored_vectors":after[2]-before[2],
        "dli_attribution":attribution,"dli_publications":events.len(),
        "auxiliary_index_bytes":auxiliary_index_bytes(&root),"peak_rss_kib":peak_rss_kib()});
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    println!("SCALING_RESULT {result}");
    collection.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "reopen retained frozen collection and profile two identical query rounds"]
async fn dli_frozen_scaling_reopen_profile() {
    let _ambient = common::ambient::test_guard();
    let root = PathBuf::from(std::env::var("DLI_REOPEN_COLLECTION_ROOT").unwrap());
    let inputs = PathBuf::from(std::env::var("DLI_SISAP_INPUTS").unwrap());
    let output = PathBuf::from(std::env::var("DLI_SISAP_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let population: usize = std::env::var("DLI_ORACLE_POPULATION")
        .unwrap()
        .parse()
        .unwrap();
    let collection = Collection::load(
        "geometric_profile".into(),
        1,
        &root.join("storage"),
        &root.join("snapshots"),
        Arc::new(SharedStorageConfig::default()),
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
    )
    .await;
    let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
    assert_eq!(info.points_count, Some(population));
    let local = replica(&collection).await;
    let topk = std::env::var("DLI_PROFILE_TOPK").as_deref() == Ok("1");
    let reserved = std::env::var("DLI_PROFILE_RESERVED").as_deref() == Ok("1");
    local
        .set_local_dli_search_profile_for_test(topk, reserved)
        .await;
    let queries = std::fs::read(inputs.join("queries.f32")).unwrap();
    let gold = std::fs::read(inputs.join("ground_truth.i32")).unwrap();
    let query_count = std::env::var("DLI_PROFILE_QUERY_COUNT")
        .ok()
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(QUERY_COUNT);
    assert!((1..=QUERY_COUNT).contains(&query_count));
    let query_start = std::env::var("DLI_PROFILE_QUERY_START")
        .ok()
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(0);
    assert!(
        query_start
            .checked_add(query_count)
            .is_some_and(|n| n <= 10000)
    );
    let queries = &queries[query_start * DIMENSIONS * 4..];
    let gold = &gold[query_start * 1000 * 4..];
    let mut rounds = Vec::new();
    let compare_topk = std::env::var("DLI_PROFILE_COMPARE_TOPK").as_deref() == Ok("1");
    for round in 0..if compare_topk { 4 } else { 2 } {
        let topk = if compare_topk { round % 2 == 1 } else { topk };
        local
            .set_local_dli_search_profile_for_test(topk, reserved)
            .await;
        let before = local.local_dli_named_profile_for_test().await.unwrap();
        let counts_before = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
        let result = run_queries(&collection, &queries, &gold, query_count, &HashSet::new()).await;
        let counts_after = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
        let after = local.local_dli_named_profile_for_test().await.unwrap();
        rounds.push(serde_json::json!({"round":round,"materialize_topk":topk,"reserved":reserved,"population":population,"query_count":query_count,"query_start":query_start,
            "recall10":result.0,"recall30":result.1,"p50_ms":result.2,"p95_ms":result.3,"mean_ms":result.4,
            "scored_vectors":counts_after[2]-counts_before[2],"before":before,"after":after}));
        std::fs::write(
            output.join("rounds.json"),
            serde_json::to_vec_pretty(&rounds).unwrap(),
        )
        .unwrap();
        assert_eq!(
            after["learned_queries"].as_u64().unwrap()
                - before["learned_queries"].as_u64().unwrap(),
            query_count as u64
        );
        assert_eq!(
            after["exact_fallbacks"], before["exact_fallbacks"],
            "frozen physical membership changed or visibility unsupported: {after}"
        );
        println!("FROZEN_PROFILE {}", rounds.last().unwrap());
    }
    collection.stop_gracefully().await;
}

/// Optimized frozen-topology pilot. Budgets chosen only from validation rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "explicit frozen10M held-out routing study; no ingestion or training"]
async fn dli_frozen_matched_work_study() {
    use crate::collection_manager::dli_bucket_plan::RoutingPolicy;
    let _ambient = common::ambient::test_guard();
    let root = PathBuf::from(std::env::var("DLI_REOPEN_COLLECTION_ROOT").unwrap());
    let inputs = PathBuf::from(std::env::var("DLI_SISAP_INPUTS").unwrap());
    let output = PathBuf::from(std::env::var("DLI_SISAP_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    assert!(
        !output.join("study.json").exists(),
        "refuse to overwrite completed evidence"
    );
    let collection = Collection::load(
        "frozen_matched_work".into(),
        1,
        &root.join("storage"),
        &root.join("snapshots"),
        Arc::new(SharedStorageConfig::default()),
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
    )
    .await;
    assert_eq!(
        collection
            .info(&ShardSelectorInternal::All)
            .await
            .unwrap()
            .points_count,
        Some(FULL_COUNT)
    );
    let local = replica(&collection).await;
    let queries = std::fs::read(inputs.join("queries.f32")).unwrap();
    let gold = std::fs::read(inputs.join("ground_truth.i32")).unwrap();
    let validation_count = std::env::var("DLI_STUDY_VALIDATION_COUNT")
        .ok()
        .map(|x| x.parse::<usize>().unwrap())
        .unwrap_or(12);
    let test_count = std::env::var("DLI_STUDY_TEST_COUNT")
        .ok()
        .map(|x| x.parse::<usize>().unwrap())
        .unwrap_or(24);
    assert!((1..=100).contains(&validation_count) && (1..=100).contains(&test_count));
    let policies = [
        RoutingPolicy::SlotWeighted,
        RoutingPolicy::ReservedPerLevel,
        RoutingPolicy::PopulationDensity,
    ];
    let mut cases = Vec::<serde_json::Value>::new();
    let mut ordinal = 0usize;
    for policy in policies {
        for budget in [10000usize, 20000, 30000] {
            local
                .configure_local_dli_query_for_test(true, policy, budget)
                .await;
            // Rehearsal and measured validation use ONLY rows100..199.
            for pass in ["validation_warmup", "validation"] {
                let row = study_case(
                    &collection,
                    &local,
                    &queries,
                    &gold,
                    100,
                    validation_count,
                    policy,
                    budget,
                    true,
                    pass,
                    &mut ordinal,
                )
                .await;
                cases.push(row);
                std::fs::write(
                    output.join("study-partial.json"),
                    serde_json::to_vec_pretty(&cases).unwrap(),
                )
                .unwrap();
            }
        }
    }
    // Lock selection BEFORE any query from rows200..299 is executed.
    let mut selected = Vec::new();
    for policy in policies {
        let relevant: Vec<_> = cases
            .iter()
            .filter(|x| x["pass"] == "validation" && x["policy"] == serde_json::json!(policy))
            .collect();
        for objective in ["work30000", "recall060"] {
            let loss = |x: &&serde_json::Value| {
                if objective == "work30000" {
                    (x["mean_scored"].as_f64().unwrap() - 30000.).abs()
                } else {
                    (x["recall10"].as_f64().unwrap() - 0.60).abs()
                }
            };
            let chosen = relevant
                .iter()
                .min_by(|a, b| loss(a).total_cmp(&loss(b)))
                .unwrap();
            let budget = chosen["budget"].as_u64().unwrap() as usize;
            if !selected.iter().any(|&(p, b)| p == policy && b == budget) {
                selected.push((policy, budget));
            }
        }
    }
    std::fs::write(output.join("locked-selection.json"),serde_json::to_vec_pretty(&serde_json::json!({"validation_rows":[100,100+validation_count],"test_rows":[200,200+test_count],"work_target":30000,"recall10_target":0.60,"selected":selected,"rule":"closest validation mean actual scored work / closest validation Recall10; no test tuning"})).unwrap()).unwrap();
    // Fixed alternating policy order; first-touch timing is never called warm.
    for (policy, budget) in selected {
        local
            .configure_local_dli_query_for_test(true, policy, budget)
            .await;
        let cleared = local.clear_local_dli_cold_vectors_for_test().await;
        for pass in ["test_cache_clear_attempt", "test_warm"] {
            let mut row = study_case(
                &collection,
                &local,
                &queries,
                &gold,
                200,
                test_count,
                policy,
                budget,
                true,
                pass,
                &mut ordinal,
            )
            .await;
            row["cleared_cold_vectors"] = serde_json::json!(cleared);
            cases.push(row);
            std::fs::write(
                output.join("study-partial.json"),
                serde_json::to_vec_pretty(&cases).unwrap(),
            )
            .unwrap();
        }
    }
    // Independent materialization factor on the exact same slot/budget topology.
    local
        .configure_local_dli_query_for_test(true, RoutingPolicy::SlotWeighted, 20000)
        .await;
    cases.push(
        study_case(
            &collection,
            &local,
            &queries,
            &gold,
            200,
            test_count,
            RoutingPolicy::SlotWeighted,
            20000,
            true,
            "materialization_warmup",
            &mut ordinal,
        )
        .await,
    );
    for topk in [false, true, false, true] {
        local
            .configure_local_dli_query_for_test(topk, RoutingPolicy::SlotWeighted, 20000)
            .await;
        cases.push(
            study_case(
                &collection,
                &local,
                &queries,
                &gold,
                200,
                test_count,
                RoutingPolicy::SlotWeighted,
                20000,
                topk,
                "materialization_control",
                &mut ordinal,
            )
            .await,
        );
        std::fs::write(
            output.join("study-partial.json"),
            serde_json::to_vec_pretty(&cases).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        output.join("study.json"),
        serde_json::to_vec_pretty(&cases).unwrap(),
    )
    .unwrap();
    collection.stop_gracefully().await;
}

async fn study_case(
    collection: &Collection,
    local: &crate::shards::replica_set::ShardReplicaSet,
    queries: &[u8],
    gold: &[u8],
    start: usize,
    count: usize,
    policy: crate::collection_manager::dli_bucket_plan::RoutingPolicy,
    budget: usize,
    topk: bool,
    pass: &str,
    ordinal: &mut usize,
) -> serde_json::Value {
    let before = local.local_dli_named_profile_for_test().await.unwrap();
    let work_before = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
    let result = run_queries(
        collection,
        &queries[start * DIMENSIONS * 4..],
        &gold[start * 1000 * 4..],
        count,
        &HashSet::new(),
    )
    .await;
    let work_after = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
    let after = local.local_dli_named_profile_for_test().await.unwrap();
    assert_eq!(
        after["learned_queries"].as_u64().unwrap() - before["learned_queries"].as_u64().unwrap(),
        count as u64
    );
    assert_eq!(after["exact_fallbacks"], before["exact_fallbacks"]);
    let row = serde_json::json!({"policy":policy,"budget":budget,"topk":topk,"pass":pass,"query_start":start,"query_count":count,"ordinal_begin":*ordinal,"recall10":result.0,"recall30":result.1,"p50_ms":result.2,"p95_ms":result.3,"mean_ms":result.4,"scored_vectors":work_after[2]-work_before[2],"mean_scored":(work_after[2]-work_before[2]) as f64/count as f64,"before":before,"after":after});
    *ordinal += count;
    println!("STUDY_CASE {row}");
    row
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "bounded release recall frontier on retained10M; no ingestion or model training"]
async fn dli_frozen_recall_frontier() {
    use crate::collection_manager::dli_bucket_plan::RoutingPolicy;
    let _scope = common::ambient::test_guard();
    let root = PathBuf::from(std::env::var("DLI_REOPEN_COLLECTION_ROOT").unwrap());
    let inputs = PathBuf::from(std::env::var("DLI_SISAP_INPUTS").unwrap());
    let output = PathBuf::from(std::env::var("DLI_SISAP_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let plan: Vec<serde_json::Value> = serde_json::from_slice(
        &std::fs::read(std::env::var("DLI_FRONTIER_PLAN").unwrap()).unwrap(),
    )
    .unwrap();
    let count = std::env::var("DLI_FRONTIER_QUERY_COUNT")
        .ok()
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(12);
    assert!((1..=100).contains(&count));
    assert!(!output.join("frontier.json").exists());
    // Budget/temperature choices originate from validation400..499 only.
    std::fs::write(
        output.join("locked-selection.json"),
        serde_json::to_vec_pretty(&plan).unwrap(),
    )
    .unwrap();
    let collection = Collection::load(
        "frozen_recall_frontier".into(),
        1,
        &root.join("storage"),
        &root.join("snapshots"),
        Arc::new(SharedStorageConfig::default()),
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
    )
    .await;
    assert_eq!(
        collection
            .info(&ShardSelectorInternal::All)
            .await
            .unwrap()
            .points_count,
        Some(FULL_COUNT)
    );
    let local = replica(&collection).await;
    let queries = std::fs::read(inputs.join("queries.f32")).unwrap();
    let gold = std::fs::read(inputs.join("ground_truth.i32")).unwrap();
    let mut cases = Vec::new();
    let mut ordinal = 0;
    for selection in plan {
        let policy: RoutingPolicy = serde_json::from_value(selection["policy"].clone()).unwrap();
        let budget = selection["budget"].as_u64().unwrap() as usize;
        local
            .configure_local_dli_query_for_test(true, policy, budget)
            .await;
        cases.push(
            study_case(
                &collection,
                &local,
                &queries,
                &gold,
                400,
                count,
                policy,
                budget,
                true,
                "frontier_validation",
                &mut ordinal,
            )
            .await,
        );
        // Do not repeatedly fault9GB back in at every large budget. Explicit
        // cold repetition is opt-in; ordinary first-touch and warm are distinct.
        let cold = std::env::var("DLI_FRONTIER_COLD").as_deref() == Ok("1");
        let cleared = if cold {
            local.clear_local_dli_cold_vectors_for_test().await
        } else {
            Vec::new()
        };
        for pass in [
            if cold {
                "test_cache_clear_attempt"
            } else {
                "test_first_touch"
            },
            "test_warm",
        ] {
            let mut row = study_case(
                &collection,
                &local,
                &queries,
                &gold,
                600,
                count,
                policy,
                budget,
                true,
                pass,
                &mut ordinal,
            )
            .await;
            row["cleared_cold_vectors"] = serde_json::json!(cleared);
            cases.push(row);
            std::fs::write(
                output.join("frontier-partial.json"),
                serde_json::to_vec_pretty(&cases).unwrap(),
            )
            .unwrap();
        }
    }
    // Same-binary old-comparator/cache control, low budget, four alternating repeats.
    for (i, policy) in [
        RoutingPolicy::PopulationDensityUncached,
        RoutingPolicy::PopulationDensityUncached,
        RoutingPolicy::PopulationDensity,
        RoutingPolicy::PopulationDensity,
        RoutingPolicy::PopulationDensityUncached,
    ]
    .into_iter()
    .enumerate()
    {
        local
            .configure_local_dli_query_for_test(true, policy, 20000)
            .await;
        cases.push(
            study_case(
                &collection,
                &local,
                &queries,
                &gold,
                600,
                count,
                policy,
                20000,
                true,
                if i == 0 {
                    "priority_rehearsal"
                } else {
                    "priority_control"
                },
                &mut ordinal,
            )
            .await,
        );
    }
    // Four full-population native exact controls identify execution/numerical ceiling.
    let exact_count = count.min(4);
    let before = local.local_dli_named_profile_for_test().await.unwrap();
    let wb = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
    let result = run_queries_params(
        &collection,
        &queries[600 * DIMENSIONS * 4..],
        &gold[600 * 1000 * 4..],
        exact_count,
        &HashSet::new(),
        Some(segment::types::SearchParams {
            exact: true,
            ..Default::default()
        }),
    )
    .await;
    let wa = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
    let after = local.local_dli_named_profile_for_test().await.unwrap();
    assert_eq!(before["learned_queries"], after["learned_queries"]);
    assert_eq!(
        after["exact_fallbacks"].as_u64().unwrap() - before["exact_fallbacks"].as_u64().unwrap(),
        exact_count as u64
    );
    cases.push(serde_json::json!({"policy":"NativeExact","budget":FULL_COUNT,"topk":false,"exact":true,"pass":"exact_control","query_start":600,"query_count":exact_count,"ordinal_begin":ordinal,"recall10":result.0,"recall30":result.1,"p50_ms":result.2,"p95_ms":result.3,"mean_ms":result.4,"scored_vectors":wa[2]-wb[2],"mean_scored":(wa[2]-wb[2]) as f64/exact_count as f64,"before":before,"after":after}));
    std::fs::write(
        output.join("frontier.json"),
        serde_json::to_vec_pretty(&cases).unwrap(),
    )
    .unwrap();
    collection.stop_gracefully().await;
}

#[test]
#[ignore = "one private64K teacher/router control, never a10M hierarchy rebuild"]
fn dli_small_teacher_router_control() {
    use crate::collection_manager::dli_maintenance::DliOptimizationStrategy;
    use segment::data_types::vectors::DEFAULT_VECTOR_NAME;
    use segment::entry::{SegmentEntry, StorageSegmentEntry};
    use shard::optimize::OptimizationStrategy;
    let _scope = common::ambient::test_guard();
    let output = PathBuf::from(std::env::var("DLI_TEACHER_CONTROL").unwrap());
    let mut model = model_config();
    if let Ok(buckets) = std::env::var("DLI_CONTROL_BUCKETS") {
        model.n_buckets = buckets.parse().unwrap();
        assert!([128, 256, 548].contains(&model.n_buckets));
    }

    let bytes = std::fs::read(output.join("pool.f16")).unwrap();
    let id_bytes = std::fs::read(output.join("pool_ids.u64")).unwrap();
    let ids: Vec<u64> = id_bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(ids.len(), 65536);
    assert_eq!(bytes.len(), ids.len() * DIMENSIONS * 2);
    std::fs::create_dir_all(output.join("segments")).unwrap();
    std::fs::create_dir_all(output.join("staging")).unwrap();
    let base = segment::segment_constructor::simple_segment_constructor::build_simple_segment(
        &output.join("segments"),
        DIMENSIONS,
        Distance::Cosine,
    )
    .unwrap();
    let mut config = base.config().clone();
    drop(base);
    config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .datatype = Some(segment::types::VectorStorageDatatype::Float16);
    let (mut source, _token) =
        segment::segment_constructor::build_segment(&output.join("segments"), &config, None, true)
            .unwrap();
    for (row, id) in ids.iter().enumerate() {
        let point = point_from_row(row, *id, &bytes);
        let VectorStructPersisted::Single(v) = point.vector else {
            unreachable!()
        };
        source
            .upsert_point(
                1,
                point.id,
                segment::data_types::vectors::only_default_vector(&v),
            )
            .unwrap();
    }
    let source_uuid = source.uuid;
    let inputs = vec![shard::locked_segment::LockedSegment::from(source)];
    let adapter = DliOptimizationStrategy {
        staging: output.join("staging"),
        segments_path: output.join("segments"),
        segment_config: config,
        destination_on_disk: false,
        model: model.clone(),
        expected_sources: HashSet::from([source_uuid]),
        expected_topology: None,
        validation_hook: None,
    };
    let mut builder = adapter.create_segment_builder(&inputs).unwrap();
    {
        let shard::locked_segment::LockedSegment::Original(source) = &inputs[0] else {
            unreachable!()
        };
        let source = source.read();
        builder
            .update(&[&*source], &std::sync::atomic::AtomicBool::new(false))
            .unwrap();
    }
    let segment = builder.build_for_test(&output.join("segments"));
    assert_eq!(segment.available_point_count(), 65536);
    segment.flush(true).unwrap();
    let path = segment.segment_path.clone();
    let uuid = segment.uuid;
    drop(segment);
    let reopened = segment::segment_constructor::load_segment(
        &path,
        uuid,
        None,
        &std::sync::atomic::AtomicBool::new(false),
        true,
    )
    .unwrap();
    assert_eq!(reopened.available_point_count(), 65536);
    std::fs::write(output.join("index.json"),serde_json::to_vec_pretty(&serde_json::json!({"path":path,"uuid":uuid,"population":65536,"model":model,"scope":"disposable subset, not reconstruction of original6M teacher"})).unwrap()).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "native 65K multi-level Collection S0/S2 benchmark; requires DLI_65K_INPUT, DLI_65K_ORACLE and DLI_65K_OUTPUT"]
async fn sisap23_65k_multilevel_collection_s0_vs_s2() {
    use std::io::{Read, Seek, SeekFrom, Write};

    use segment::entry::ReadSegmentEntry;
    use segment::id_tracker::IdTrackerRead;
    use segment::types::ExtendedPointId;

    let _ambient = common::ambient::test_guard();
    // Four 16K drains produce a level-2 run of 48K followed by a separate
    // level-1 run of 16K under the geometric scheduler. The final 1,536 IDs
    // remain in Plain, giving a real three-role topology at exactly 65,536.
    const MULTILEVEL_BUFFER: usize = 16_000;
    for name in [
        "DLI_REQUEST_SCORING_EVENTS",
        "DLI_ROUTING_EVENTS",
        "DLI_QUANTIZATION_BUILD_EVENTS",
    ] {
        assert!(
            std::env::var_os(name).is_some(),
            "required measurement path {name}"
        );
    }
    assert!(
        std::env::var_os("DLI_QUANTIZED_SHORTLIST").is_none(),
        "use the shard-local override"
    );
    let inputs = PathBuf::from(std::env::var("DLI_65K_INPUT").unwrap());
    let oracle = PathBuf::from(std::env::var("DLI_65K_ORACLE").unwrap());
    let output = PathBuf::from(std::env::var("DLI_65K_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let mut vectors = std::fs::File::open(inputs.join("pool.f16")).unwrap();
    assert_eq!(
        vectors.metadata().unwrap().len(),
        (SMALL_NATIVE_POPULATION * DIMENSIONS * 2) as u64
    );
    let id_bytes = std::fs::read(inputs.join("pool_ids.u64")).unwrap();
    assert_eq!(id_bytes.len(), SMALL_NATIVE_POPULATION * 8);
    let pool_ids: Vec<u64> = id_bytes
        .chunks_exact(8)
        .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let expected_ids: HashSet<_> = pool_ids.iter().copied().collect();
    assert_eq!(expected_ids.len(), SMALL_NATIVE_POPULATION);
    let queries = std::fs::read(inputs.join("queries.f32")).unwrap();
    assert_eq!(queries.len(), 64 * DIMENSIONS * 4);
    let gold_bytes = std::fs::read(oracle.join("gold_ids.u64")).unwrap();
    assert_eq!(gold_bytes.len(), 64 * 30 * 8);
    let gold: Vec<u64> = gold_bytes
        .chunks_exact(8)
        .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let gold_score_bytes = std::fs::read(oracle.join("gold_cosines.f64")).unwrap();
    assert_eq!(gold_score_bytes.len(), 64 * 30 * 8);
    let gold_scores: Vec<f64> = gold_score_bytes
        .chunks_exact(8)
        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert!(gold.iter().all(|id| expected_ids.contains(id)));

    let mut config = collection_config(true);
    let dli = config.dli_experimental.as_mut().unwrap();
    dli.buffer_capacity = MULTILEVEL_BUFFER;
    dli.materialize_topk = true;
    dli.quantized_shortlist = Some(128); // built natively; S0 overrides query use only
    // Slot weighting ranks Plain at priority 1.0, so include its whole bucket
    // plus a bounded learned allowance or this fixture would exact-fallback.
    dli.candidate_budget = 10_000;
    let root = output.join("collection");
    assert!(!root.exists(), "use a new output root; preserve prior runs");
    std::fs::create_dir(&root).unwrap();
    let collection = create_collection_with_config("sisap23_65k_multilevel", &root, config).await;
    let local = replica(&collection).await;
    let mut publications = Vec::new();
    for (batch, first, end) in [
        (0usize, 0usize, 16_000usize),
        (1, 16_000, 32_000),
        (2, 32_000, 48_000),
        (3, 48_000, 64_000),
    ] {
        let mut block = vec![0u8; (end - first) * DIMENSIONS * 2];
        vectors
            .seek(SeekFrom::Start((first * DIMENSIONS * 2) as u64))
            .unwrap();
        vectors.read_exact(&mut block).unwrap();
        let points = (first..end)
            .map(|row| point_from_row(row - first, pool_ids[row], &block))
            .collect();
        let before = local
            .local_dli_publication_events_for_test()
            .await
            .unwrap()
            .len();
        let write = apply_points(&collection, points).await;
        wait_for_dli_publications(&local, before + 1).await;
        let events = local.local_dli_publication_events_for_test().await.unwrap();
        let event = events.last().unwrap();
        publications.push(serde_json::json!({"batch":batch,"first":first,"end":end,"insert_wait_ms":write.as_secs_f64()*1000.0,
            "target_level":event.target_level,"source_uuids":event.source_uuids,"destination_uuid":event.destination_uuid,
            "expected_output_points":event.expected_output_points,"published_output_points":event.published_output_points,
            "optimizer_micros":event.optimizer_micros}));
    }
    // 1,536 new points remain in the native appendable Plain buffer.
    let first = 64_000;
    let end = SMALL_NATIVE_POPULATION;
    let mut block = vec![0u8; (end - first) * DIMENSIONS * 2];
    vectors
        .seek(SeekFrom::Start((first * DIMENSIONS * 2) as u64))
        .unwrap();
    vectors.read_exact(&mut block).unwrap();
    let points = (first..end)
        .map(|row| point_from_row(row - first, pool_ids[row], &block))
        .collect();
    let _insert_wait = apply_points(&collection, points).await;
    assert_eq!(
        collection
            .info(&ShardSelectorInternal::All)
            .await
            .unwrap()
            .points_count,
        Some(SMALL_NATIVE_POPULATION)
    );

    // Inspect the active holder, not scheduler expectations. Prove every live
    // external ID is owned exactly once and each learned member owns its own index.
    let holder = local.local_segments_for_test().await.unwrap();
    let guard = holder.read();
    assert_eq!(
        guard.iter().count(),
        guard.iter_original().count(),
        "no active optimizer proxies"
    );
    let (hierarchy_generation, roles) = local.local_dli_roles_for_test().await;
    let roles_by_uuid: std::collections::HashMap<_, _> = roles.iter().copied().collect();
    assert_eq!(roles_by_uuid.len(), guard.iter_original().count());
    assert_eq!(
        roles_by_uuid.values().copied().collect::<HashSet<_>>(),
        HashSet::from([0, 1, 2])
    );
    let mut actual_ids = HashSet::new();
    let mut topology = Vec::new();
    let mut learned = 0;
    let mut plain = 0;
    for (_, member) in guard.iter_original() {
        let segment = member.read();
        let level = *roles_by_uuid
            .get(&segment.uuid)
            .expect("active physical UUID has logical role");
        let data = &segment.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME];
        let index = data.vector_index.borrow();
        let (kind, learned_population) = match &*index {
            VectorIndexEnum::Lmi(lmi) => {
                assert!(lmi.routing_state().is_some());
                assert!(!segment.is_appendable());
                assert!(level > 0);
                assert_eq!(lmi.routing_state().unwrap().postings().len(), 548);
                let quant = data.quantized_vectors.borrow();
                let quant = quant
                    .as_ref()
                    .expect("each learned physical generation has native scalar data");
                let expected_path = segment::segment_constructor::get_vector_storage_path(
                    &segment.segment_path,
                    segment::data_types::vectors::DEFAULT_VECTOR_NAME,
                );
                assert_eq!(
                    quant.storage_path(),
                    expected_path,
                    "quantizer belongs to active segment generation"
                );
                learned += 1;
                ("lmi", segment.available_point_count())
            }
            VectorIndexEnum::Plain(_) => {
                assert!(segment.is_appendable());
                assert!(
                    segment.available_point_count() > 0,
                    "Plain member is not empty"
                );
                assert_eq!(level, 0);
                plain += 1;
                ("plain", segment.available_point_count())
            }
            _ => panic!("unexpected active index type"),
        };
        let tracker = segment.id_tracker.borrow();
        for (point_id, offset) in tracker.point_mappings().iter_from(None) {
            let ExtendedPointId::NumId(id) = point_id else {
                panic!("expected numeric IDs")
            };
            if !segment.has_point(id.into(), common::types::DeferredBehavior::VisibleOnly) {
                continue;
            }
            assert!(actual_ids.insert(id), "duplicate active physical ID {id}");
            assert_eq!(tracker.external_id(offset), Some(id.into()));
        }
        let vector_storage_path = segment::segment_constructor::get_vector_storage_path(
            &segment.segment_path,
            segment::data_types::vectors::DEFAULT_VECTOR_NAME,
        );
        // Include mmap chunks under `vectors/`; a one-level directory scan
        // counted only top-level quantized files and understated original
        // vector storage by nearly the entire Float16 corpus.
        let vector_storage_bytes = directory_file_bytes(&vector_storage_path);
        let quantized_files: Vec<_> = [
            "quantized.config.json",
            "quantized.data",
            "quantized.meta.json",
            "quantized.offsets.data",
        ]
        .into_iter()
        .filter_map(|name| {
            let path = vector_storage_path.join(name);
            path.metadata()
                .ok()
                .map(|meta| serde_json::json!({"name":name,"bytes":meta.len()}))
        })
        .collect();
        let quantized_bytes: u64 = quantized_files
            .iter()
            .filter_map(|file| file["bytes"].as_u64())
            .sum();
        let index_path = segment.segment_path.join("vector_index");
        let learned_index_bytes: u64 = std::fs::read_dir(index_path)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("lmi_"))
            .filter_map(|entry| entry.metadata().ok().map(|meta| meta.len()))
            .sum();
        topology.push(serde_json::json!({"uuid":segment.uuid,"logical_level":level,"kind":kind,"population":learned_population,
            "segment_path":segment.segment_path,"vector_storage_path":vector_storage_path,"vector_storage_total_bytes":vector_storage_bytes,
            "scalar_quantized_files":quantized_files,"scalar_quantized_bytes":quantized_bytes,"learned_model_posting_bytes":learned_index_bytes}));
    }
    assert_eq!(learned, 2, "expected exactly two learned physical levels");
    assert_eq!(plain, 1, "expected one nonempty Plain buffer");
    assert!(
        topology
            .iter()
            .filter(|row| row["kind"] == "lmi")
            .all(|row| row["scalar_quantized_bytes"].as_u64().unwrap_or(0) > 0),
        "quantized storage files must be present on learned members"
    );
    assert_eq!(
        actual_ids, expected_ids,
        "active Collection population differs from oracle"
    );
    drop(guard);
    drop(holder);
    assert_eq!(
        local.local_dli_counts_for_test().await.unwrap().0,
        4,
        "four automatic native publications"
    );

    let scorer_events = output.join("request-scoring.jsonl");
    let routing_events = output.join("routing.jsonl");
    let query_events = output.join("collection-queries.jsonl");
    for p in [&scorer_events, &routing_events, &query_events] {
        let _ = std::fs::remove_file(p);
    }
    // Warm each mode first. Profiling runs are separate from latency runs so
    // per-stage timers and JSONL writes do not contaminate Collection latency.
    local
        .configure_local_dli_scoring_profile_for_test(false)
        .await;
    for mode in [Some(128), None] {
        local
            .configure_local_dli_quantized_shortlist_for_test(mode)
            .await;
        for qid in 0..4 {
            let _ = run_65k_query(&collection, &queries, qid, 30).await;
        }
    }
    let mut profiles = Vec::new();
    for qid in 0..64 {
        for mode in [None, Some(128)] {
            local
                .configure_local_dli_quantized_shortlist_for_test(mode)
                .await;
            local
                .configure_local_dli_scoring_profile_for_test(true)
                .await;
            let before_profile = local.local_dli_named_profile_for_test().await.unwrap();
            let before_route_bytes = std::fs::metadata(&routing_events)
                .map(|m| m.len())
                .unwrap_or(0);
            let before_score_bytes = std::fs::metadata(&scorer_events)
                .map(|m| m.len())
                .unwrap_or(0);
            let hits = run_65k_query(&collection, &queries, qid, 30).await;
            assert_native_scores_match_oracle(&hits, &gold, &gold_scores, qid);
            let after_profile = local.local_dli_named_profile_for_test().await.unwrap();
            let route_data = std::fs::read(&routing_events).unwrap_or_default();
            assert!(
                route_data.len() as u64 > before_route_bytes,
                "profiled request omitted its routing plan"
            );
            let route_line = route_data[before_route_bytes as usize..]
                .split(|b| *b == b'\n')
                .next()
                .unwrap();
            let plan = serde_json::from_slice::<serde_json::Value>(route_line).unwrap();
            let score_data = std::fs::read(&scorer_events).unwrap_or_default();
            assert!(
                score_data.len() as u64 > before_score_bytes,
                "profiled request omitted its request-local event"
            );
            let score_line = score_data[before_score_bytes as usize..]
                .split(|b| *b == b'\n')
                .next()
                .unwrap();
            let request_profile = serde_json::from_slice::<serde_json::Value>(score_line).unwrap();
            let learned_delta = after_profile["learned_queries"].as_u64().unwrap()
                - before_profile["learned_queries"].as_u64().unwrap();
            let fallback_delta = after_profile["exact_fallbacks"].as_u64().unwrap()
                - before_profile["exact_fallbacks"].as_u64().unwrap();
            assert_eq!(
                learned_delta, 1,
                "this configuration must execute learned routing"
            );
            assert_eq!(
                fallback_delta, 0,
                "do not count exact fallback as learned success"
            );
            // Final-outcome event schema (one event per logical request).
            assert_eq!(request_profile["final_outcome"], "learned");
            assert_eq!(request_profile["attempt"]["attempt_resolution"], "resolved");
            assert!(request_profile["exact_fallback"].is_null());
            let lp = &request_profile["learned_scoring"];
            assert!(
                lp["plain_scored"].as_u64().unwrap() > 0,
                "the selected plan must account for Plain scoring"
            );
            assert!(lp["original_scored"].as_u64().unwrap() > 0);
            assert_eq!(lp["quantized_scored"].as_u64().unwrap() > 0, mode.is_some());
            let unique_offsets = request_profile["attempt"]["unique_offsets"].as_u64().unwrap();
            assert!(request_profile["attempt"]["posting_entries"].as_u64().unwrap() >= unique_offsets);
            if mode.is_some() {
                assert_eq!(lp["quantized_scored"].as_u64().unwrap(), unique_offsets);
                assert!(lp["original_scored"].as_u64().unwrap() <= 2 * 128);
            } else {
                assert_eq!(lp["original_scored"].as_u64().unwrap(), unique_offsets);
                assert_eq!(lp["quantized_scored"].as_u64().unwrap(), 0);
            }
            assert_eq!(lp["plain_scored"].as_u64().unwrap(), 1_536);
            let attributed = lp["quantized_scored"].as_u64().unwrap()
                + lp["original_scored"].as_u64().unwrap()
                + lp["plain_scored"].as_u64().unwrap();
            let profile_row = serde_json::json!({"query":qid,"mode":if mode.is_some(){"S2"}else{"S0"},"learned_delta":learned_delta,"exact_fallback_delta":fallback_delta,"attributed_successful_scorer_inputs":attributed,"request_profile":request_profile,"plan":plan,"profile_pass_returned":hits});
            profiles.push(profile_row);
        }
    }
    // Scoring-stage instrumentation, including its event writes, is disabled
    // during latency passes. Only the per-shard scoring override changes.
    local
        .configure_local_dli_scoring_profile_for_test(false)
        .await;
    let mut query_rows = Vec::new();
    for qid in 0..64 {
        let order = if qid % 2 == 0 {
            [None, Some(128), Some(128), None]
        } else {
            [Some(128), None, None, Some(128)]
        };
        for (pass, mode) in order.into_iter().enumerate() {
            local
                .configure_local_dli_quantized_shortlist_for_test(mode)
                .await;
            let before_task = query_process_counters();
            let started = Instant::now();
            let hits = run_65k_query(&collection, &queries, qid, 30).await;
            let elapsed_ns = started.elapsed().as_nanos();
            let after_task = query_process_counters();
            assert_native_scores_match_oracle(&hits, &gold, &gold_scores, qid);
            let expected = &gold[qid * 30..(qid + 1) * 30];
            let recall10 = hits
                .iter()
                .take(10)
                .filter(|hit| expected[..10].contains(&hit["id"].as_u64().unwrap()))
                .count();
            let recall30 = hits
                .iter()
                .take(30)
                .filter(|hit| expected.contains(&hit["id"].as_u64().unwrap()))
                .count();
            let faults = before_task.zip(after_task).map(|(a,b)| serde_json::json!({"minor_faults":b.0.saturating_sub(a.0),"major_faults":b.1.saturating_sub(a.1),"process_cpu_ticks":b.2.saturating_sub(a.2)}));
            let profile = profiles
                .iter()
                .find(|row| {
                    row["query"] == qid && row["mode"] == if mode.is_some() { "S2" } else { "S0" }
                })
                .unwrap();
            let row = serde_json::json!({"query":qid,"split":if qid<32{"validation"}else{"test"},"pass":pass,"mode":if mode.is_some(){"S2"}else{"S0"},"shortlist":mode,"warm_elapsed_ns":elapsed_ns,"recall10_hits":recall10,"recall30_hits":recall30,"process_faults_cpu":faults,"matched_profile":profile["request_profile"],"hits":hits});
            writeln!(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&query_events)
                    .unwrap(),
                "{row}"
            )
            .unwrap();
            query_rows.push(row);
        }
    }
    assert_eq!(profiles.len(), 64 * 2);
    for qid in 0..64 {
        let a = profiles
            .iter()
            .find(|row| row["query"] == qid && row["mode"] == "S0")
            .unwrap();
        let b = profiles
            .iter()
            .find(|row| row["query"] == qid && row["mode"] == "S2")
            .unwrap();
        assert_eq!(a["plan"], b["plan"], "S0/S2 postings must be identical");
    }
    let (generation, mut roles) = (hierarchy_generation, roles);
    roles.sort_unstable();
    let quantization_events_path =
        PathBuf::from(std::env::var("DLI_QUANTIZATION_BUILD_EVENTS").unwrap());
    let quantization_events: Vec<serde_json::Value> =
        std::fs::read_to_string(&quantization_events_path)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
    assert_eq!(
        quantization_events.len(),
        4,
        "each native destination quantizes once"
    );
    let quantized_bytes_total: u64 = topology
        .iter()
        .map(|row| row["scalar_quantized_bytes"].as_u64().unwrap_or(0))
        .sum();
    let original_vector_storage_bytes: u64 = topology
        .iter()
        .map(|row| {
            row["vector_storage_total_bytes"]
                .as_u64()
                .unwrap_or(0)
                .saturating_sub(row["scalar_quantized_bytes"].as_u64().unwrap_or(0))
        })
        .sum();
    let mut summary = serde_json::json!({"population":SMALL_NATIVE_POPULATION,"dimensions":DIMENSIONS,"model":model_config(),"buffer_capacity":MULTILEVEL_BUFFER,"candidate_budget":10_000,"hierarchy_generation":generation,"logical_roles":roles,"topology":topology,"publications":publications,"profiled_queries":profiles,"timed_queries":query_rows,"warm_only":true,"counterbalance":"per-query ABBA/BAAB","official_query_rows":"900-963; validation 900-931; test 932-963 (selection manifest says not used in continuations 26-38)","oracle":"continuation-39 full-pool FP16 corpus; Float64 cosine, ID tie break; exact oracle generated outside timed search","source_sha":std::process::Command::new("git").args(["rev-parse","HEAD"]).output().ok().map(|o|String::from_utf8_lossy(&o.stdout).trim().to_owned()),"raw_original_vector_bytes":SMALL_NATIVE_POPULATION*DIMENSIONS*2,"persisted_original_vector_storage_bytes":original_vector_storage_bytes,"persisted_scalar_quantized_bytes":quantized_bytes_total,"quantization_build_events":quantization_events,"process_peak_rss_kib":peak_rss_kib()});
    local.force_flush_local_for_test().await;
    drop(local);
    collection.stop_gracefully().await;

    let reopen_started = Instant::now();
    let reopened = Collection::load(
        "sisap23_65k_multilevel".into(),
        1,
        &root.join("storage"),
        &root.join("snapshots"),
        Arc::new(SharedStorageConfig::default()),
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        None,
    )
    .await;
    let reopen_elapsed = reopen_started.elapsed();
    assert_eq!(
        reopened
            .info(&ShardSelectorInternal::All)
            .await
            .unwrap()
            .points_count,
        Some(SMALL_NATIVE_POPULATION)
    );
    let reopened_local = replica(&reopened).await;
    let reopened_holder = reopened_local.local_segments_for_test().await.unwrap();
    for (_, member) in reopened_holder.read().iter_original() {
        let segment = member.read();
        let data = &segment.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME];
        if matches!(&*data.vector_index.borrow(), VectorIndexEnum::Lmi(_)) {
            assert!(
                data.quantized_vectors.borrow().is_some(),
                "quantized files loaded for the right recovered generation"
            );
        }
    }
    reopened_local
        .configure_local_dli_scoring_profile_for_test(false)
        .await;
    reopened_local
        .configure_local_dli_quantized_shortlist_for_test(Some(128))
        .await;
    let before = reopened_local.local_dli_counts_for_test().await.unwrap();
    let reopened_hits = run_65k_query(&reopened, &queries, 63, 30).await;
    let after = reopened_local.local_dli_counts_for_test().await.unwrap();
    assert_eq!(
        after.1 - before.1,
        1,
        "reopened collection must resume learned routing"
    );
    assert_eq!(
        after.2 - before.2,
        0,
        "reopen query must not use exact fallback"
    );
    let reopen_record = serde_json::json!({"load_ms":reopen_elapsed.as_secs_f64()*1000.0,"points":SMALL_NATIVE_POPULATION,"learned_query_delta":after.1-before.1,"fallback_delta":after.2-before.2,"sample_query":63,"sample_result_count":reopened_hits.len()});
    summary["reopen"] = reopen_record;
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    reopened.stop_gracefully().await;
}

fn dli_65k_query_vector(bytes: &[u8], query: usize) -> Vec<f32> {
    let begin = query * DIMENSIONS * 4;
    bytes[begin..begin + DIMENSIONS * 4]
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

fn assert_native_scores_match_oracle(
    hits: &[serde_json::Value],
    gold_ids: &[u64],
    gold_scores: &[f64],
    query: usize,
) {
    let ids = &gold_ids[query * 30..(query + 1) * 30];
    let scores = &gold_scores[query * 30..(query + 1) * 30];
    for hit in hits {
        let id = hit["id"].as_u64().unwrap();
        if let Some(rank) = ids.iter().position(|candidate| *candidate == id) {
            let actual = hit["score"].as_f64().unwrap();
            assert!(
                (actual - scores[rank]).abs() < 2e-3,
                "native cosine score differs from independent Float16 oracle: query={query}, id={id}, native={actual}, oracle={}",
                scores[rank]
            );
        }
    }
}

async fn run_65k_query(
    collection: &Collection,
    query_data: &[u8],
    query: usize,
    limit: usize,
) -> Vec<serde_json::Value> {
    let result = collection
        .search(
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(dli_65k_query_vector(
                    query_data, query,
                ))),
                filter: None,
                params: None,
                limit,
                offset: 0,
                with_payload: None,
                with_vector: None,
                score_threshold: None,
            },
            None,
            None,
            &ShardSelectorInternal::All,
            None,
        )
        .await
        .unwrap();
    result.into_iter().map(|hit| serde_json::json!({
        "id": match hit.id { segment::types::ExtendedPointId::NumId(id) => id, _ => panic!("numeric IDs expected") },
        "score": hit.score,
        "version": hit.version,
    })).collect()
}

/// Router identity and bucket count of every learned member, keyed by DLI level.
async fn learned_members(
    collection: &Collection,
    root: &Path,
) -> std::collections::BTreeMap<usize, (uuid::Uuid, u64, usize, usize)> {
    let Ok(bytes) = std::fs::read(root.join("storage/0/dli_hierarchy.json")) else {
        return Default::default();
    };
    let descriptor: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let roles: std::collections::HashMap<uuid::Uuid, usize> = descriptor["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| (m[0].as_str().unwrap().parse().unwrap(), m[1].as_u64().unwrap() as usize))
        .collect();
    let holder = replica(collection).await.local_segments_for_test().await.unwrap();
    let guard = holder.read();
    let mut members = std::collections::BTreeMap::new();
    for (_, member) in guard.iter_original() {
        let segment = member.read();
        let level = roles[&segment.uuid];
        if level == 0 {
            continue;
        }
        let index = segment.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow();
        let VectorIndexEnum::Lmi(lmi) = &*index else {
            panic!("learned member without LMI index")
        };
        let state = lmi.routing_state().unwrap();
        assert!(
            members
                .insert(
                    level,
                    (
                        segment.uuid,
                        state.router().fingerprint(),
                        state.postings().len(),
                        segment.available_point_count(),
                    ),
                )
                .is_none(),
            "two members on level {level}"
        );
    }
    members
}

/// Sommerová-faithful maintenance end to end in a configured collection: aⁱ buckets on
/// every learned level; with reuse, a compaction into a non-empty level keeps that
/// level's router (no training) and without reuse it retrains; full-budget search
/// equals exact search over all points.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn emma_level_rule_and_router_reuse_in_configured_collection() {
    let _scope = common::ambient::test_guard();
    use rand::{RngExt, SeedableRng};
    use crate::collection_manager::dli_bucket_plan::RoutingPolicy;
    use crate::collection_manager::dli_schedule::{
        HierarchyConfig, LevelBuckets, LevelModelPolicy, MaintenancePolicy,
    };
    const B: usize = 40;
    const A: usize = 4;
    const BATCHES: usize = 22;
    let mut rng = rand::rngs::StdRng::seed_from_u64(20261120);
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..DIMENSIONS).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();
    let rows: Vec<Vec<f32>> = (0..B * BATCHES)
        .map(|i| {
            centers[i % centers.len()]
                .iter()
                .map(|c| c + rng.random_range(-0.3f32..0.3))
                .collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..6)
        .map(|i| centers[i].iter().map(|c| c + rng.random_range(-0.3f32..0.3)).collect())
        .collect();
    let stored = |v: &[f32]| -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| f16::from_f32(x / norm).to_f32()).collect()
    };
    let corpus: Vec<Vec<f32>> = rows.iter().map(|v| stored(v)).collect();

    for reuse in [true, false] {
        let root = tempfile::tempdir().unwrap();
        let mut config = collection_config(true);
        let dli = config.dli_experimental.as_mut().unwrap();
        dli.buffer_capacity = B;
        dli.arity = A;
        dli.candidate_budget = usize::MAX / 2;
        dli.routing_policy = RoutingPolicy::ModelDrivenNormalizedCode;
        dli.level_model = LevelModelPolicy {
            buckets: LevelBuckets::ArityPower,
            min_samples_per_bucket: 4,
            reuse_target_router: reuse,
        };
        dli.model = LmiConfig {
            n_buckets: A,
            sample_size: 64,
            hidden_dim: 8,
            epochs: 2,
            batch_size: 16,
            routing_batch_size: 16,
            kmeans_iterations: 3,
            nprobe: 1,
            seed: 5,
            reuse_router: None,
        };
        let collection = create_collection_with_config("emma", root.path(), config).await;
        let local = replica(&collection).await;
        let hierarchy = HierarchyConfig {
            buffer_capacity: B,
            arity: A,
            policy: MaintenancePolicy::Geometric,
        };
        let mut counts: Vec<usize> = Vec::new();
        let mut previous = learned_members(&collection, root.path()).await;
        let (mut reused, mut retrained) = (0, 0);
        for batch in 0..BATCHES {
            let points = (batch * B..(batch + 1) * B)
                .map(|row| PointStructPersisted {
                    id: ((row + 1) as u64).into(),
                    vector: VectorStructPersisted::Single(rows[row].clone()),
                    payload: None,
                })
                .collect();
            apply_points(&collection, points).await;
            wait_for_dli_publications(&local, batch + 1).await;
            let plan = hierarchy.plan(B, &counts).unwrap().unwrap();
            counts.resize(counts.len().max(plan.target_level), 0);
            for level in &plan.source_levels {
                counts[level - 1] = 0;
            }
            counts[plan.target_level - 1] = plan.output_points;
            let current = learned_members(&collection, root.path()).await;
            for (level, (_, _, buckets, points)) in &current {
                assert_eq!(*buckets, A.pow(*level as u32), "level {level} bucket count");
                assert_eq!(*points, counts[level - 1], "level {level} population");
            }
            let before = previous.get(&plan.target_level);
            let after = current[&plan.target_level];
            if plan.target_existing_points > 0 {
                let before = before.expect("non-empty target existed");
                assert_ne!(before.0, after.0, "target must be a new physical generation");
                if reuse {
                    assert_eq!(before.1, after.1, "reuse must keep the target router");
                    reused += 1;
                } else {
                    assert_ne!(before.1, after.1, "rebuild must retrain the target router");
                    retrained += 1;
                }
            }
            previous = current;
        }
        assert_eq!(counts.iter().sum::<usize>(), B * BATCHES);
        assert!(reused + retrained >= 8, "too few non-empty-target compactions");
        let diagnostics = local.local_dli_diagnostics_for_test().await.unwrap();
        assert!(diagnostics.4.is_empty(), "maintenance errors: {diagnostics:?}");

        for query in &queries {
            let result = collection
                .search(
                    CoreSearchRequest {
                        query: QueryEnum::Nearest(NamedQuery::default_dense(query.clone())),
                        filter: None,
                        params: None,
                        limit: 10,
                        offset: 0,
                        with_payload: None,
                        with_vector: None,
                        score_threshold: None,
                    },
                    None,
                    None,
                    &ShardSelectorInternal::All,
                    None,
                )
                .await
                .unwrap();
            let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut exact: Vec<f32> = corpus
                .iter()
                .map(|v| v.iter().zip(query).map(|(a, b)| a * b / norm).sum())
                .collect();
            exact.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(result.len(), 10);
            for (hit, oracle) in result.iter().zip(&exact) {
                assert!((hit.score - oracle).abs() < 2e-3, "{} vs {oracle}", hit.score);
            }
        }
        let attribution = local.local_dli_search_metrics_for_test().await.unwrap();
        println!("EMMA_TEST reuse={reuse} reused={reused} retrained={retrained} search={attribution:?}");
        collection.stop_gracefully().await;
    }
}

/// Drives maintenance to a fixed point after deletes: re-plans without an update (rebuild
/// cascades have none) and waits until nothing is scheduled or running. Fails on any
/// optimizer error or after `limit`.
async fn settle_dli_maintenance(
    replica: &crate::shards::replica_set::ShardReplicaSet,
    limit: Duration,
) -> usize {
    let started = Instant::now();
    let initial = replica.local_dli_diagnostics_for_test().await.unwrap().1;
    loop {
        assert!(started.elapsed() < limit, "DLI maintenance did not settle");
        let before = replica.local_dli_diagnostics_for_test().await.unwrap();
        assert!(before.4.is_empty(), "DLI maintenance failed: {before:?}");
        if before.3 || before.0 > before.1 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            continue;
        }
        replica.local_dli_observe_for_test().await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        let after = replica.local_dli_diagnostics_for_test().await.unwrap();
        if after.0 == before.0 && !after.3 && after.0 == after.1 {
            return after.1 - initial;
        }
    }
}

/// Phase 3 M4: Greedy underutilisation rebuild through the native optimizer, after
/// random and oldest-first deletes, in a configured collection (b = 40, a = 4). After
/// maintenance settles, no level is below Cmin, every rebuilt destination holds no
/// deleted points, and full-budget learned search equals exact search over live points.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn greedy_rebuild_after_deletes_in_configured_collection() {
    let _scope = common::ambient::test_guard();
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};
    use crate::collection_manager::dli_bucket_plan::RoutingPolicy;
    use crate::collection_manager::dli_schedule::{
        HierarchyConfig, LevelBuckets, LevelModelPolicy, MaintenancePolicy,
    };
    const B: usize = 40;
    const A: usize = 4;
    const BATCHES: usize = 12;
    let mut rng = rand::rngs::StdRng::seed_from_u64(20261122);
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..DIMENSIONS).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();
    let rows: Vec<Vec<f32>> = (0..B * BATCHES)
        .map(|i| {
            centers[i % centers.len()]
                .iter()
                .map(|c| c + rng.random_range(-0.3f32..0.3))
                .collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..6)
        .map(|i| centers[i].iter().map(|c| c + rng.random_range(-0.3f32..0.3)).collect())
        .collect();
    let stored = |v: &[f32]| -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| f16::from_f32(x / norm).to_f32()).collect()
    };
    let hierarchy = HierarchyConfig {
        buffer_capacity: B,
        arity: A,
        policy: MaintenancePolicy::Geometric,
    };
    for oldest in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut config = collection_config(true);
        let dli = config.dli_experimental.as_mut().unwrap();
        dli.buffer_capacity = B;
        dli.arity = A;
        dli.candidate_budget = usize::MAX / 2;
        dli.routing_policy = RoutingPolicy::ModelDrivenNormalizedCode;
        dli.greedy_rebuild = true;
        dli.level_model = LevelModelPolicy {
            buckets: LevelBuckets::ArityPower,
            min_samples_per_bucket: 4,
            reuse_target_router: true,
        };
        dli.model = LmiConfig {
            n_buckets: A,
            sample_size: 64,
            hidden_dim: 8,
            epochs: 2,
            batch_size: 16,
            routing_batch_size: 16,
            kmeans_iterations: 3,
            nprobe: 1,
            seed: 5,
            reuse_router: None,
        };
        let collection = create_collection_with_config("rebuild", root.path(), config).await;
        let local = replica(&collection).await;
        for batch in 0..BATCHES {
            let points = (batch * B..(batch + 1) * B)
                .map(|row| PointStructPersisted {
                    id: ((row + 1) as u64).into(),
                    vector: VectorStructPersisted::Single(rows[row].clone()),
                    payload: None,
                })
                .collect();
            apply_points(&collection, points).await;
            wait_for_dli_publications(&local, batch + 1).await;
        }
        // Delete 70% of the points: the oldest IDs, or a seeded random subset.
        let mut ids: Vec<u64> = (1..=(B * BATCHES) as u64).collect();
        if !oldest {
            ids.shuffle(&mut rand::rngs::StdRng::seed_from_u64(7));
        }
        let deleted: HashSet<u64> = ids[..B * BATCHES * 7 / 10].iter().copied().collect();
        let before_events = local.local_dli_publication_events_for_test().await.unwrap().len();
        apply_delete(&collection, deleted.iter().copied().collect()).await;
        let rebuilds = settle_dli_maintenance(&local, Duration::from_secs(120)).await;
        let events = local.local_dli_publication_events_for_test().await.unwrap();
        let rebuilt: HashSet<uuid::Uuid> =
            events[before_events..].iter().map(|e| e.destination_uuid).collect();
        assert_eq!(rebuilt.len(), rebuilds);

        // Fixed point: no level below Cmin; rebuilt destinations dropped dead points.
        let members = learned_members(&collection, root.path()).await;
        let mut live = vec![0usize; members.keys().max().copied().unwrap_or(0)];
        for (level, (_, _, _, points)) in &members {
            live[level - 1] = *points;
        }
        assert_eq!(hierarchy.plan_rebuild(&live).unwrap(), None, "levels {live:?}");
        let holder = replica(&collection).await.local_segments_for_test().await.unwrap();
        for (_, member) in holder.read().iter_original() {
            let segment = member.read();
            if rebuilt.contains(&segment.uuid) {
                assert_eq!(segment.deleted_point_count(), 0, "rebuild kept dead points");
            }
        }
        println!("REBUILD_TEST oldest={oldest} rebuilds={rebuilds} levels={live:?}");
        if !oldest {
            assert!(rebuilds >= 1, "70% random deletes must trigger a rebuild");
        }

        let corpus: Vec<(u64, Vec<f32>)> = rows
            .iter()
            .enumerate()
            .map(|(i, v)| ((i + 1) as u64, stored(v)))
            .filter(|(id, _)| !deleted.contains(id))
            .collect();
        let (learned_before, fallback_before) =
            local.local_dli_search_counts_for_test().await.unwrap();
        for query in &queries {
            let result = collection
                .search(
                    CoreSearchRequest {
                        query: QueryEnum::Nearest(NamedQuery::default_dense(query.clone())),
                        filter: None,
                        params: None,
                        limit: 10,
                        offset: 0,
                        with_payload: None,
                        with_vector: None,
                        score_threshold: None,
                    },
                    None,
                    None,
                    &ShardSelectorInternal::All,
                    None,
                )
                .await
                .unwrap();
            let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut exact: Vec<f32> = corpus
                .iter()
                .map(|(_, v)| v.iter().zip(query).map(|(a, b)| a * b / norm).sum())
                .collect();
            exact.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(result.len(), 10);
            for (hit, oracle) in result.iter().zip(&exact) {
                assert!((hit.score - oracle).abs() < 2e-3, "{} vs {oracle}", hit.score);
                let id = match hit.id {
                    segment::types::ExtendedPointId::NumId(n) => n,
                    other => panic!("unexpected id {other:?}"),
                };
                assert!(!deleted.contains(&id), "deleted point {id} returned");
            }
        }
        let (learned, fallback) = local.local_dli_search_counts_for_test().await.unwrap();
        assert_eq!(learned - learned_before, queries.len() as u64, "learned routing");
        assert_eq!(fallback - fallback_before, 0, "exact fallbacks");
        collection.stop_gracefully().await;
    }
}

/// Phase 3 arm E1V: vacuum rebuild through the native optimizer after 50% random deletes
/// (b = 40, a = 4). After maintenance settles, no learned level is more than 20% dead,
/// vacuumed destinations hold no deleted points, and full-budget learned search equals
/// exact search over live points with no exact fallback.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn vacuum_rebuild_after_deletes_in_configured_collection() {
    let _scope = common::ambient::test_guard();
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};
    use crate::collection_manager::dli_bucket_plan::RoutingPolicy;
    use crate::collection_manager::dli_schedule::{LevelBuckets, LevelModelPolicy};
    const B: usize = 40;
    const A: usize = 4;
    const BATCHES: usize = 12;
    let mut rng = rand::rngs::StdRng::seed_from_u64(20261122);
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..DIMENSIONS).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();
    let rows: Vec<Vec<f32>> = (0..B * BATCHES)
        .map(|i| {
            centers[i % centers.len()]
                .iter()
                .map(|c| c + rng.random_range(-0.3f32..0.3))
                .collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..6)
        .map(|i| centers[i].iter().map(|c| c + rng.random_range(-0.3f32..0.3)).collect())
        .collect();
    let stored = |v: &[f32]| -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| f16::from_f32(x / norm).to_f32()).collect()
    };
    for oldest in [false] {
        let root = tempfile::tempdir().unwrap();
        let mut config = collection_config(true);
        let dli = config.dli_experimental.as_mut().unwrap();
        dli.buffer_capacity = B;
        dli.arity = A;
        dli.candidate_budget = usize::MAX / 2;
        dli.routing_policy = RoutingPolicy::ModelDrivenNormalizedCode;
        dli.vacuum_dead_permille = 200;
        dli.level_model = LevelModelPolicy {
            buckets: LevelBuckets::ArityPower,
            min_samples_per_bucket: 4,
            reuse_target_router: true,
        };
        dli.model = LmiConfig {
            n_buckets: A,
            sample_size: 64,
            hidden_dim: 8,
            epochs: 2,
            batch_size: 16,
            routing_batch_size: 16,
            kmeans_iterations: 3,
            nprobe: 1,
            seed: 5,
            reuse_router: None,
        };
        let collection = create_collection_with_config("vacuum", root.path(), config).await;
        let local = replica(&collection).await;
        for batch in 0..BATCHES {
            let points = (batch * B..(batch + 1) * B)
                .map(|row| PointStructPersisted {
                    id: ((row + 1) as u64).into(),
                    vector: VectorStructPersisted::Single(rows[row].clone()),
                    payload: None,
                })
                .collect();
            apply_points(&collection, points).await;
            wait_for_dli_publications(&local, batch + 1).await;
        }
        // Delete 70% of the points: the oldest IDs, or a seeded random subset.
        let mut ids: Vec<u64> = (1..=(B * BATCHES) as u64).collect();
        if !oldest {
            ids.shuffle(&mut rand::rngs::StdRng::seed_from_u64(7));
        }
        let deleted: HashSet<u64> = ids[..B * BATCHES / 2].iter().copied().collect();
        let before_events = local.local_dli_publication_events_for_test().await.unwrap().len();
        apply_delete(&collection, deleted.iter().copied().collect()).await;
        let rebuilds = settle_dli_maintenance(&local, Duration::from_secs(120)).await;
        let events = local.local_dli_publication_events_for_test().await.unwrap();
        let rebuilt: HashSet<uuid::Uuid> =
            events[before_events..].iter().map(|e| e.destination_uuid).collect();
        assert_eq!(rebuilt.len(), rebuilds);

        // Fixed point: no learned level above 20% dead; vacuumed destinations hold none.
        let holder = replica(&collection).await.local_segments_for_test().await.unwrap();
        let mut worst = 0f64;
        for (_, member) in holder.read().iter_original() {
            let segment = member.read();
            let (live, dead) = (segment.available_point_count(), segment.deleted_point_count());
            if !segment.is_appendable() && live + dead > 0 {
                worst = worst.max(dead as f64 / (live + dead) as f64);
            }
            if rebuilt.contains(&segment.uuid) {
                assert_eq!(dead, 0, "vacuum kept dead points");
            }
        }
        assert!(worst <= 0.2, "a learned level is {worst:.2} dead after vacuum");
        assert!(rebuilds >= 1, "50% random deletes must trigger a vacuum");
        println!("VACUUM_TEST rebuilds={rebuilds} worst_dead={worst:.3}");

        let corpus: Vec<(u64, Vec<f32>)> = rows
            .iter()
            .enumerate()
            .map(|(i, v)| ((i + 1) as u64, stored(v)))
            .filter(|(id, _)| !deleted.contains(id))
            .collect();
        let (learned_before, fallback_before) =
            local.local_dli_search_counts_for_test().await.unwrap();
        for query in &queries {
            let result = collection
                .search(
                    CoreSearchRequest {
                        query: QueryEnum::Nearest(NamedQuery::default_dense(query.clone())),
                        filter: None,
                        params: None,
                        limit: 10,
                        offset: 0,
                        with_payload: None,
                        with_vector: None,
                        score_threshold: None,
                    },
                    None,
                    None,
                    &ShardSelectorInternal::All,
                    None,
                )
                .await
                .unwrap();
            let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut exact: Vec<f32> = corpus
                .iter()
                .map(|(_, v)| v.iter().zip(query).map(|(a, b)| a * b / norm).sum())
                .collect();
            exact.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(result.len(), 10);
            for (hit, oracle) in result.iter().zip(&exact) {
                assert!((hit.score - oracle).abs() < 2e-3, "{} vs {oracle}", hit.score);
                let id = match hit.id {
                    segment::types::ExtendedPointId::NumId(n) => n,
                    other => panic!("unexpected id {other:?}"),
                };
                assert!(!deleted.contains(&id), "deleted point {id} returned");
            }
        }
        let (learned, fallback) = local.local_dli_search_counts_for_test().await.unwrap();
        assert_eq!(learned - learned_before, queries.len() as u64, "learned routing");
        assert_eq!(fallback - fallback_before, 0, "exact fallbacks");
        collection.stop_gracefully().await;
    }
}

/// DLI-best (research/dli/emma_path/DLI_BEST_RULE.md) in one configured collection:
/// consolidation rho = 0.25 and vacuum above 20% dead run together in the maintenance
/// planner, and searches are served by the native `PopulationProportional` policy. After
/// inserts and 50% random deletes the maintenance reaches a fixed point with no learned
/// level above 20% dead, vacuumed destinations hold no deleted points, and full-budget
/// proportional search equals exact search over live points with no exact fallback.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dli_best_maintenance_and_proportional_serving_in_configured_collection() {
    let _scope = common::ambient::test_guard();
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};
    use crate::collection_manager::dli_bucket_plan::RoutingPolicy;
    use crate::collection_manager::dli_schedule::{LevelBuckets, LevelModelPolicy};
    const B: usize = 40;
    const A: usize = 4;
    const BATCHES: usize = 12;
    let mut rng = rand::rngs::StdRng::seed_from_u64(20261122);
    let centers: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..DIMENSIONS).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();
    let rows: Vec<Vec<f32>> = (0..B * BATCHES)
        .map(|i| {
            centers[i % centers.len()]
                .iter()
                .map(|c| c + rng.random_range(-0.3f32..0.3))
                .collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..6)
        .map(|i| centers[i].iter().map(|c| c + rng.random_range(-0.3f32..0.3)).collect())
        .collect();
    let stored = |v: &[f32]| -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| f16::from_f32(x / norm).to_f32()).collect()
    };
    for oldest in [false] {
        let root = tempfile::tempdir().unwrap();
        let mut config = collection_config(true);
        let dli = config.dli_experimental.as_mut().unwrap();
        dli.buffer_capacity = B;
        dli.arity = A;
        dli.candidate_budget = usize::MAX / 2;
        dli.routing_policy = RoutingPolicy::PopulationProportional;
        dli.vacuum_dead_permille = 200;
        dli.maintenance_policy = crate::collection_manager::dli_schedule::MaintenancePolicy::GeometricConsolidating { rho_permille: 250 };
        dli.level_model = LevelModelPolicy {
            buckets: LevelBuckets::ArityPower,
            min_samples_per_bucket: 4,
            reuse_target_router: true,
        };
        dli.model = LmiConfig {
            n_buckets: A,
            sample_size: 64,
            hidden_dim: 8,
            epochs: 2,
            batch_size: 16,
            routing_batch_size: 16,
            kmeans_iterations: 3,
            nprobe: 1,
            seed: 5,
            reuse_router: None,
        };
        let collection = create_collection_with_config("dlibest", root.path(), config).await;
        let local = replica(&collection).await;
        for batch in 0..BATCHES {
            let points = (batch * B..(batch + 1) * B)
                .map(|row| PointStructPersisted {
                    id: ((row + 1) as u64).into(),
                    vector: VectorStructPersisted::Single(rows[row].clone()),
                    payload: None,
                })
                .collect();
            apply_points(&collection, points).await;
            wait_for_dli_publications(&local, batch + 1).await;
        }
        // Delete 70% of the points: the oldest IDs, or a seeded random subset.
        let mut ids: Vec<u64> = (1..=(B * BATCHES) as u64).collect();
        if !oldest {
            ids.shuffle(&mut rand::rngs::StdRng::seed_from_u64(7));
        }
        let deleted: HashSet<u64> = ids[..B * BATCHES / 2].iter().copied().collect();
        let before_events = local.local_dli_publication_events_for_test().await.unwrap().len();
        apply_delete(&collection, deleted.iter().copied().collect()).await;
        let rebuilds = settle_dli_maintenance(&local, Duration::from_secs(120)).await;
        let events = local.local_dli_publication_events_for_test().await.unwrap();
        let rebuilt: HashSet<uuid::Uuid> =
            events[before_events..].iter().map(|e| e.destination_uuid).collect();
        assert_eq!(rebuilt.len(), rebuilds);

        // Fixed point: no learned level above 20% dead; vacuumed destinations hold none.
        let holder = replica(&collection).await.local_segments_for_test().await.unwrap();
        let mut worst = 0f64;
        for (_, member) in holder.read().iter_original() {
            let segment = member.read();
            let (live, dead) = (segment.available_point_count(), segment.deleted_point_count());
            if !segment.is_appendable() && live + dead > 0 {
                worst = worst.max(dead as f64 / (live + dead) as f64);
            }
            if rebuilt.contains(&segment.uuid) {
                assert_eq!(dead, 0, "vacuum kept dead points");
            }
        }
        assert!(worst <= 0.2, "a learned level is {worst:.2} dead after vacuum");
        assert!(rebuilds >= 1, "50% random deletes must trigger a vacuum");
        let merges = events
            .iter()
            .filter(|e| e.source_uuids.len() >= 2)
            .count();
        println!("DLI_BEST_TEST rebuilds={rebuilds} worst_dead={worst:.3} multi_source_publications={merges}");

        let corpus: Vec<(u64, Vec<f32>)> = rows
            .iter()
            .enumerate()
            .map(|(i, v)| ((i + 1) as u64, stored(v)))
            .filter(|(id, _)| !deleted.contains(id))
            .collect();
        let (learned_before, fallback_before) =
            local.local_dli_search_counts_for_test().await.unwrap();
        for query in &queries {
            let result = collection
                .search(
                    CoreSearchRequest {
                        query: QueryEnum::Nearest(NamedQuery::default_dense(query.clone())),
                        filter: None,
                        params: None,
                        limit: 10,
                        offset: 0,
                        with_payload: None,
                        with_vector: None,
                        score_threshold: None,
                    },
                    None,
                    None,
                    &ShardSelectorInternal::All,
                    None,
                )
                .await
                .unwrap();
            let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut exact: Vec<f32> = corpus
                .iter()
                .map(|(_, v)| v.iter().zip(query).map(|(a, b)| a * b / norm).sum())
                .collect();
            exact.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(result.len(), 10);
            for (hit, oracle) in result.iter().zip(&exact) {
                assert!((hit.score - oracle).abs() < 2e-3, "{} vs {oracle}", hit.score);
                let id = match hit.id {
                    segment::types::ExtendedPointId::NumId(n) => n,
                    other => panic!("unexpected id {other:?}"),
                };
                assert!(!deleted.contains(&id), "deleted point {id} returned");
            }
        }
        let (learned, fallback) = local.local_dli_search_counts_for_test().await.unwrap();
        assert_eq!(learned - learned_before, queries.len() as u64, "learned routing");
        assert_eq!(fallback - fallback_before, 0, "exact fallbacks");
        collection.stop_gracefully().await;
    }
}

/// Arm S of `research/dli/static_qdrant_vs_dli/`: wait until Qdrant's ordinary
/// optimizers are idle (collection Green, optimizer status Ok) for two consecutive
/// polls one second apart. Nothing is triggered or forced; an optimizer error fails.
async fn settle_static_optimizers(collection: &Collection, limit: Duration) -> f64 {
    use crate::operations::types::{CollectionStatus, OptimizersStatus};
    let started = Instant::now();
    let mut green_polls = 0;
    loop {
        assert!(started.elapsed() < limit, "static optimizers did not settle");
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        if let OptimizersStatus::Error(error) = &info.optimizer_status {
            panic!("static optimizer error: {error}");
        }
        if info.status == CollectionStatus::Green {
            green_polls += 1;
            if green_polls >= 2 {
                return started.elapsed().as_secs_f64();
            }
        } else {
            green_polls = 0;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// Segment topology of the local shard: kind, live and deleted points, learned buckets.
async fn segment_topology(collection: &Collection) -> Vec<serde_json::Value> {
    let holder = replica(collection).await.local_segments_for_test().await.unwrap();
    let guard = holder.read();
    let mut out: Vec<serde_json::Value> = guard
        .iter_original()
        .map(|(_, member)| {
            let segment = member.read();
            let index = segment.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow();
            let buckets = match &*index {
                VectorIndexEnum::Lmi(lmi) => lmi.routing_state().map(|st| st.postings().len()),
                _ => None,
            };
            serde_json::json!({"uuid": segment.uuid, "appendable": segment.is_appendable(),
                "live": segment.available_point_count(), "deleted": segment.deleted_point_count(),
                "learned_buckets": buckets})
        })
        .collect();
    out.sort_by_key(|v| v["uuid"].as_str().unwrap_or_default().to_owned());
    out
}

fn env_list(name: &str) -> Option<Vec<String>> {
    std::env::var(name)
        .ok()
        .map(|v| v.split(',').filter(|s| !s.is_empty()).map(str::to_owned).collect())
}

/// Sorted live point ids of the local shard (all segments) and their SHA-256 over
/// little-endian u32, the digest `phase3_plan.py` records per batch.
async fn live_id_digest(collection: &Collection) -> (usize, String) {
    use sha2::{Digest, Sha256};
    let holder = replica(collection).await.local_segments_for_test().await.unwrap();
    let mut ids: Vec<u32> = Vec::new();
    let mut available = 0usize;
    for (_, member) in holder.read().iter_original() {
        let segment = member.read();
        available += segment.available_point_count();
        for id in segment.iter_points() {
            match id {
                segment::types::ExtendedPointId::NumId(n) => ids.push(u32::try_from(n).unwrap()),
                other => panic!("unexpected id {other:?}"),
            }
        }
    }
    ids.sort_unstable();
    assert_eq!(ids.len(), available, "iterated ids differ from available counts");
    let before = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), before, "a point is live in two segments");
    let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    let hex: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    (ids.len(), hex)
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// Collection configuration of one arm of `research/dli/static_qdrant_vs_dli/` (and of the
/// Phase 3 DLI variants), shared by the trajectory driver and the service driver.
/// Returns (config, static_system, static_model, arm_label, buffer_capacity, arity, base_model).
/// `DLI_P3_STATIC_SERVE_NPROBE` sets the static model's serving nprobe (default 1; the replay
/// sweeps nprobe itself); the DLI serving budget is `DLI_BENCH_BUDGET` (collection_config).
#[allow(clippy::type_complexity)]
fn p3_setup() -> (CollectionConfigInternal, bool, Option<LmiConfig>, String, usize, usize, LmiConfig) {
    use crate::collection_manager::dli_schedule::MaintenancePolicy;
    let variant = std::env::var("DLI_P3_VARIANT").unwrap_or_else(|_| "none".into());
    let (buffer_capacity, arity, base_model, _, _) = arm_config();
    // DLI_P3_SYSTEM = dli (default) | static. `static` is arm S of
    // research/dli/static_qdrant_vs_dli: ordinary Qdrant optimizers with one fixed
    // static-LMI model on every optimized segment (the arm's model rule with
    // `DLI_P3_STATIC_BUCKETS` buckets), `DLI_P3_STATIC_SEGMENTS` target segments,
    // `DLI_P3_STATIC_INDEXING_KB` indexing threshold, one optimization job at a time.
    // `hnsw` is the final benchmark's HNSW arm (research/dli/final_benchmark/PREREGISTRATION.md
    // §3.3): the same ordinary-optimizer lifecycle as `static`, with no LMI model, so optimized
    // segments become Qdrant HNSW indexes (`DLI_P3_HNSW_M`, `DLI_P3_HNSW_EF_CONSTRUCT`, default 100).
    let system = std::env::var("DLI_P3_SYSTEM").unwrap_or_else(|_| "dli".into());
    let (static_system, hnsw_system) = match system.as_str() {
        "dli" => (false, false),
        "static" => (true, false),
        "hnsw" => (true, true),
        other => panic!("DLI_P3_SYSTEM must be dli, static or hnsw, got {other}"),
    };
    let env_usize = |name: &str| -> usize {
        std::env::var(name).unwrap_or_else(|_| panic!("{name} required")).parse().expect(name)
    };
    let static_model = (static_system && !hnsw_system).then(|| {
        let buckets = env_usize("DLI_P3_STATIC_BUCKETS");
        let mut model = base_model;
        model.n_buckets = buckets;
        model.sample_size = buckets.max(100_000.max(40 * buckets).min(1_000_000));
        model.nprobe = std::env::var("DLI_P3_STATIC_SERVE_NPROBE").map_or(1, |v| v.parse().expect("DLI_P3_STATIC_SERVE_NPROBE"));
        model.reuse_router = None;
        model
    });
    let mut config = collection_config(!static_system);
    if hnsw_system {
        let crate::operations::types::VectorsConfig::Single(vectors) = &mut config.params.vectors else {
            unreachable!()
        };
        vectors.lmi_config = None;
        config.hnsw_config.m = env_usize("DLI_P3_HNSW_M");
        config.hnsw_config.ef_construct =
            std::env::var("DLI_P3_HNSW_EF_CONSTRUCT").map_or(100, |v| v.parse().expect("DLI_P3_HNSW_EF_CONSTRUCT"));
        config.optimizer_config.default_segment_number = env_usize("DLI_P3_STATIC_SEGMENTS");
        config.optimizer_config.indexing_threshold = Some(env_usize("DLI_P3_STATIC_INDEXING_KB"));
        config.optimizer_config.max_optimization_threads = Some(1);
        assert_eq!(variant, "none", "variants apply to the DLI system only");
    }
    if let Some(model) = static_model {
        let crate::operations::types::VectorsConfig::Single(vectors) = &mut config.params.vectors else {
            unreachable!()
        };
        vectors.lmi_config = Some(model);
        config.optimizer_config.default_segment_number = env_usize("DLI_P3_STATIC_SEGMENTS");
        config.optimizer_config.indexing_threshold = Some(env_usize("DLI_P3_STATIC_INDEXING_KB"));
        config.optimizer_config.max_optimization_threads = Some(1);
        assert_eq!(variant, "none", "variants apply to the DLI system only");
    } else if !hnsw_system {
        let dli = config.dli_experimental.as_mut().unwrap();
        match variant.as_str() {
            "none" => {}
            "greedy" => dli.greedy_rebuild = true,
            "vacuum200" => dli.vacuum_dead_permille = 200,
            "consolidate250" => {
                dli.maintenance_policy = MaintenancePolicy::GeometricConsolidating { rho_permille: 250 }
            }
            "consolidate500" => {
                dli.maintenance_policy = MaintenancePolicy::GeometricConsolidating { rho_permille: 500 }
            }
            // DLI-best (research/dli/emma_path/DLI_BEST_RULE.md): vacuum above 20% dead,
            // consolidation rho = 0.25 and population-proportional serving allocation.
            "dlibest" => {
                dli.vacuum_dead_permille = 200;
                dli.maintenance_policy = MaintenancePolicy::GeometricConsolidating { rho_permille: 250 };
                dli.routing_policy = crate::collection_manager::dli_bucket_plan::RoutingPolicy::PopulationProportional;
            }
            other => panic!("unknown DLI_P3_VARIANT {other}"),
        }
    }
    let arm_label = if hnsw_system {
        format!("HNSW/m{}/efc{}/seg{}", config.hnsw_config.m, config.hnsw_config.ef_construct, env_usize("DLI_P3_STATIC_SEGMENTS"))
    } else if static_system {
        format!("S/b{}/seg{}", static_model.unwrap().n_buckets, env_usize("DLI_P3_STATIC_SEGMENTS"))
    } else {
        format!("{:?}/{variant}", arm())
    };
    (config, static_system, static_model, arm_label, buffer_capacity, arity, base_model)
}

/// Phase 3 workload driver (PHASE3_PREREGISTRATION.md §2–§7, Part B §10). Executes a plan
/// written by `research/dli/emma_path/phase3_plan.py`: per batch, inserts (in chunks of the
/// buffer capacity), then deletes, then drives maintenance to a fixed point, checks the live
/// set against the plan's ledger, and replays the selection frontier on a flushed copy of the
/// shard. Environment:
/// * `DLI_P3_PLAN` (plan dir), `DLI_P3_CORPUS` (Float16 rows), `DLI_P3_OUTPUT`, `DLI_P3_SCRATCH`;
/// * `DLI_ARM` = E1 | E1R and `DLI_P3_VARIANT` = none | greedy | vacuum200 | consolidate250 | consolidate500 | dlibest;
/// * `DLI_P3_CALIB_SET` (plan set fitted per batch for the `calibrated` policy, E1L),
/// * `DLI_P3_EVAL` (batch indices, default all), `DLI_P3_STATIC_BATCHES` (default 5,10,15,20),
///   `DLI_P3_SETS` (set names, default every set of the batch), `DLI_FRONTIER_BUDGETS`,
///   `DLI_FRONTIER_POLICIES`, `DLI_FRONTIER_MAX_QUERIES`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Phase 3 workload driver; requires DLI_P3_PLAN, DLI_P3_CORPUS, DLI_P3_OUTPUT, DLI_P3_SCRATCH"]
async fn dli_phase3_workload() {
    use std::io::{Read, Seek, SeekFrom, Write};
    use super::dli_selection_frontier::{FrontierRun, read_set, run_frontier};
    let _scope = common::ambient::test_guard();
    let plan_dir = PathBuf::from(std::env::var("DLI_P3_PLAN").unwrap());
    let output = PathBuf::from(std::env::var("DLI_P3_OUTPUT").unwrap());
    let scratch = PathBuf::from(std::env::var("DLI_P3_SCRATCH").unwrap());
    assert!(!output.join("summary.json").exists(), "refusing to overwrite");
    std::fs::create_dir_all(&output).unwrap();
    let plan: serde_json::Value =
        serde_json::from_slice(&std::fs::read(plan_dir.join("plan.json")).unwrap()).unwrap();
    let (config, static_system, static_model, arm_label, buffer_capacity, arity, base_model) = p3_setup();
    let hnsw_system = std::env::var("DLI_P3_SYSTEM").as_deref() == Ok("hnsw");
    let hnsw_efs: Vec<usize> = if hnsw_system {
        env_list("DLI_P3_HNSW_EFS").expect("DLI_P3_HNSW_EFS").iter().map(|s| s.parse().unwrap()).collect()
    } else {
        Vec::new()
    };
    let static_nprobes: Vec<usize> = if static_system && !hnsw_system {
        env_list("DLI_P3_STATIC_NPROBES").expect("DLI_P3_STATIC_NPROBES").iter().map(|s| s.parse().unwrap()).collect()
    } else {
        Vec::new()
    };
    let max_batches = std::env::var("DLI_P3_MAX_BATCHES").map_or(usize::MAX, |v| v.parse().unwrap());
    let eval: Option<HashSet<usize>> =
        env_list("DLI_P3_EVAL").map(|v| v.iter().map(|s| s.parse().unwrap()).collect());
    let static_batches: HashSet<usize> = env_list("DLI_P3_STATIC_BATCHES")
        .unwrap_or_else(|| ["5", "10", "15", "20"].map(String::from).to_vec())
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let set_filter: Option<HashSet<String>> = env_list("DLI_P3_SETS").map(|v| v.into_iter().collect());
    let budgets: Vec<usize> = env_list("DLI_FRONTIER_BUDGETS")
        .or_else(|| static_system.then(Vec::new))
        .expect("DLI_FRONTIER_BUDGETS")
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    let policies: Option<HashSet<String>> =
        env_list("DLI_FRONTIER_POLICIES").map(|v| v.into_iter().collect());
    let query_cap = std::env::var("DLI_FRONTIER_MAX_QUERIES").map_or(usize::MAX, |v| v.parse().unwrap());

    let root = tempfile::tempdir_in(&scratch).unwrap();
    let collection = create_collection_with_config("phase3", root.path(), config).await;
    let local = replica(&collection).await;
    let mut corpus = std::fs::File::open(std::env::var("DLI_P3_CORPUS").unwrap()).unwrap();
    let mut read_rows = |ids: &[u32]| -> Vec<PointStructPersisted> {
        ids.iter()
            .map(|&id| {
                let row = id as usize - 1;
                let mut bytes = vec![0u8; DIMENSIONS * 2];
                corpus.seek(SeekFrom::Start((row * DIMENSIONS * 2) as u64)).unwrap();
                corpus.read_exact(&mut bytes).unwrap();
                point_from_row(0, id as u64, &bytes)
            })
            .collect()
    };
    let read_u32 = |path: PathBuf| -> Vec<u32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    let mut log = std::fs::File::create(output.join("batches.jsonl")).unwrap();
    let started = Instant::now();
    let batches = plan["batches"].as_array().unwrap();
    for batch in batches.iter().take(max_batches) {
        let index = batch["index"].as_u64().unwrap() as usize;
        let dir = plan_dir.join(format!("batch_{index:02}"));
        let batch_started = Instant::now();
        let events_before = local.local_dli_publication_events_for_test().await.map_or(0, |e| e.len());
        let inserts = read_u32(dir.join("insert.u32"));
        // DLI drains synchronously after every buffer-sized chunk (Phase 3 protocol); the
        // static system lets Qdrant's optimizers run concurrently and settles at batch end.
        for chunk in inserts.chunks(buffer_capacity) {
            apply_points(&collection, read_rows(chunk)).await;
            if !static_system {
                settle_dli_maintenance(&local, Duration::from_secs(4 * 3600)).await;
            }
        }
        let deletes = read_u32(dir.join("delete.u32"));
        for chunk in deletes.chunks(10_000) {
            apply_delete(&collection, chunk.iter().map(|&i| i as u64).collect()).await;
            if !static_system {
                settle_dli_maintenance(&local, Duration::from_secs(4 * 3600)).await;
            }
        }
        let writes_seconds = batch_started.elapsed().as_secs_f64();
        let settle_seconds = if static_system {
            settle_static_optimizers(&collection, Duration::from_secs(4 * 3600)).await
        } else {
            0.0
        };
        let ingest_seconds = batch_started.elapsed().as_secs_f64();
        let (live, digest) = live_id_digest(&collection).await;
        assert_eq!(live as u64, batch["live_count"].as_u64().unwrap(), "batch {index} live count");
        assert_eq!(digest, batch["live_sha256"].as_str().unwrap(), "batch {index} live set");
        let events = if static_system {
            Vec::new()
        } else {
            local.local_dli_publication_events_for_test().await.unwrap()
        };
        let members = learned_members(&collection, root.path()).await;
        let segments = segment_topology(&collection).await;
        let mut record = serde_json::json!({
            "batch": index, "arm": arm_label, "live": live, "live_sha256": digest,
            "ingest_seconds": ingest_seconds, "writes_seconds": writes_seconds, "settle_seconds": settle_seconds,
            "elapsed_seconds": started.elapsed().as_secs_f64(), "segments": segments,
            "publications": &events[events_before..],
            "levels": members.iter().map(|(l, (_, fp, b, n))| serde_json::json!({
                "level": l, "live": n, "buckets": b, "router": format!("{fp:016x}")})).collect::<Vec<_>>(),
        });
        if eval.as_ref().is_none_or(|e| e.contains(&index)) {
            local.force_flush_local_for_test().await;
            let copy = scratch.join(format!("frozen-{index:02}"));
            copy_dir(&root.path().join("storage/0"), &copy);
            let sets: Vec<_> = batch["sets"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(name, _)| set_filter.as_ref().is_none_or(|f| f.contains(*name)))
                .map(|(name, s)| {
                    let set = read_set(
                        name,
                        &plan_dir.join(s["queries"].as_str().unwrap()),
                        &plan_dir.join(s["gold"].as_str().unwrap()),
                    );
                    // Final-benchmark plans (fb_plan.py) carry G1 tie sets; Phase 3 plans do not.
                    match (s["tie"].as_str(), s["tie_offsets"].as_str()) {
                        (Some(t), Some(o)) => set.with_ties(&plan_dir.join(t), &plan_dir.join(o)),
                        _ => set,
                    }
                })
                .collect();
            let static_model = static_batches.contains(&index).then(|| {
                // Matched bucket total with the arm's model rule (Phase 1 §2, Amendment 1.5).
                let b: usize = members.keys().map(|&l| arity.pow(l as u32)).sum();
                let mut model = base_model;
                model.n_buckets = b.max(1);
                model.sample_size = b.max(live.min(100_000.max(40 * b)).min(1_000_000));
                model.nprobe = 1;
                model
            });
            let calibration_set = std::env::var("DLI_P3_CALIB_SET").ok().map(|name| {
                let spec = &batch["sets"][&name];
                read_set(
                    &name,
                    &plan_dir.join(spec["queries"].as_str().expect("calibration set in plan")),
                    &plan_dir.join(spec["gold"].as_str().unwrap()),
                )
            });
            let frontier_started = Instant::now();
            if hnsw_system {
                super::dli_selection_frontier::run_hnsw_frontier(
                    &copy, &output.join(format!("batch_{index:02}")), &sets, &hnsw_efs, query_cap,
                );
            } else if static_system {
                super::dli_selection_frontier::run_static_frontier(
                    &copy, &output.join(format!("batch_{index:02}")), &sets, &static_nprobes, query_cap,
                );
            } else {
            run_frontier(&FrontierRun {
                segments_dir: &copy,
                output: &output.join(format!("batch_{index:02}")),
                sets: &sets,
                budgets: &budgets,
                policy_filter: policies.as_ref(),
                query_cap,
                static_reference: static_model.is_some(),
                static_model,
                calibration_set: calibration_set.as_ref(),
            });
            }
            std::fs::remove_dir_all(&copy).unwrap();
            record["frontier_seconds"] = frontier_started.elapsed().as_secs_f64().into();
        }
        writeln!(log, "{record}").unwrap();
        println!("PHASE3_BATCH {record}");
    }
    let diagnostics = if static_system {
        (0, 0, 0, false, Vec::new())
    } else {
        local.local_dli_diagnostics_for_test().await.unwrap()
    };
    assert!(diagnostics.4.is_empty(), "maintenance errors: {diagnostics:?}");
    let summary = serde_json::json!({
        "system": if hnsw_system { "hnsw" } else if static_system { "static" } else { "dli" }, "static_model": static_model,
        "arm": arm_label, "scenario": plan["scenario"], "rows": plan["rows"],
        "buffer_capacity": buffer_capacity, "arity": arity, "model": base_model,
        "publications": if static_system { Vec::new() } else { local.local_dli_publication_events_for_test().await.unwrap() },
        "diagnostics": format!("{diagnostics:?}"), "seconds": started.elapsed().as_secs_f64(),
        "source_head": std::env::var("DLI_BENCH_HEAD").ok(),
    });
    std::fs::write(output.join("summary.json"), serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
    collection.stop_gracefully().await;
}

/// Process and cgroup counters for the service phase. CPU is in clock ticks of
/// `/proc/self/stat` (user, system; `CLK_TCK` is recorded as 100, the Linux default);
/// I/O from `/proc/self/io`; memory from `/proc/self/status` and cgroup v2 files.
fn service_counters() -> serde_json::Value {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<u64> = stat
        .rsplit_once(')')
        .map(|(_, t)| t.split_whitespace().map(|f| f.parse().unwrap_or(0)).collect())
        .unwrap_or_default();
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kib = |key: &str| {
        status.lines().find_map(|l| l.strip_prefix(key)).and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
    };
    let io = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    let io_field = |key: &str| io.lines().find_map(|l| l.strip_prefix(key)).and_then(|v| v.trim().parse::<u64>().ok());
    let cgroup = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|c| c.lines().find_map(|l| l.strip_prefix("0::").map(str::to_owned)));
    let cg = |file: &str| {
        cgroup.as_ref().and_then(|p| std::fs::read_to_string(format!("/sys/fs/cgroup{p}/{file}")).ok()).and_then(|v| v.trim().parse::<u64>().ok())
    };
    serde_json::json!({
        "utime_ticks": fields.get(11), "stime_ticks": fields.get(12), "clk_tck": 100,
        "rss_kib": kib("VmRSS:"), "hwm_kib": kib("VmHWM:"),
        "read_bytes": io_field("read_bytes:"), "write_bytes": io_field("write_bytes:"),
        "cgroup_memory_current": cg("memory.current"), "cgroup_memory_peak": cg("memory.peak"),
    })
}

fn percentiles(mut v: Vec<f64>) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::json!(null);
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
    serde_json::json!({"n": v.len(), "p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": v[v.len() - 1],
        "mean": v.iter().sum::<f64>() / v.len() as f64})
}

/// Concurrent-service phase of `research/dli/static_qdrant_vs_dli/` (DESIGN.md §6, H4).
/// Same plan and arm configuration as `dli_phase3_workload` (`p3_setup`), but through the
/// live `Collection::search` path while the batch is written at a fixed rate and maintenance
/// (DLI) or the ordinary optimizers (static) run in the background — no synchronous drain.
///
/// Per batch:
/// * writer: inserts then deletes in chunks of `DLI_SVC_CHUNK` at `DLI_SVC_WRITE_RATE`
///   points/s (sleep-paced, never faster);
/// * readers: open-loop arrivals at `DLI_SVC_QPS` with at most `DLI_SVC_INFLIGHT` requests
///   in flight (arrivals beyond the cap are counted as dropped), per-request timeout
///   `DLI_SVC_TIMEOUT_MS`, from the writer's start until maintenance has settled;
/// * after settling: one sequential pass of the panel (settled latency and recall against
///   the plan's gold for this batch), and the live-set ledger check;
/// * correctness: any returned id whose deletion completed before the request started is a
///   violation (counted, and the run fails at the end if any occurred).
///
/// Serving operating point: `DLI_BENCH_BUDGET` (DLI) or `DLI_P3_STATIC_SERVE_NPROBE` (static).
/// Query panel: `DLI_SVC_SET` from the plan (validation in pilots). No request-path tracing
/// is enabled; `DLI_QUERY_EVENTS` must be unset.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "static-vs-DLI service phase; requires DLI_P3_PLAN, DLI_P3_CORPUS, DLI_P3_OUTPUT, DLI_P3_SCRATCH, DLI_SVC_*"]
async fn dli_service_workload() {
    use std::collections::HashMap;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use common::ambient::AmbientFutureExt as _;
    use super::dli_selection_frontier::read_set;
    /// Releases one in-flight slot even if the request task panics.
    struct Slot(Arc<AtomicUsize>);
    impl Drop for Slot {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::AcqRel);
        }
    }
    let _scope = common::ambient::test_guard();
    assert!(std::env::var_os("DLI_QUERY_EVENTS").is_none(), "no request-path tracing in the service phase");
    let env = |n: &str| std::env::var(n).unwrap_or_else(|_| panic!("{n} required"));
    let num = |n: &str| -> f64 { env(n).parse().expect(n) };
    let plan_dir = PathBuf::from(env("DLI_P3_PLAN"));
    let output = PathBuf::from(env("DLI_P3_OUTPUT"));
    let scratch = PathBuf::from(env("DLI_P3_SCRATCH"));
    assert!(!output.join("service-summary.json").exists(), "refusing to overwrite");
    std::fs::create_dir_all(&output).unwrap();
    let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(plan_dir.join("plan.json")).unwrap()).unwrap();
    let (config, static_system, static_model, arm_label, _buffer_capacity, _, _) = p3_setup();
    let (qps, inflight_cap, timeout_ms) = (num("DLI_SVC_QPS"), num("DLI_SVC_INFLIGHT") as usize, num("DLI_SVC_TIMEOUT_MS"));
    let (write_rate, chunk) = (num("DLI_SVC_WRITE_RATE"), num("DLI_SVC_CHUNK") as usize);
    let warmup = std::env::var("DLI_SVC_WARMUP").map_or(20, |v| v.parse().unwrap());
    let set_name = env("DLI_SVC_SET");
    let max_batches = std::env::var("DLI_P3_MAX_BATCHES").map_or(usize::MAX, |v| v.parse().unwrap());

    let root = tempfile::tempdir_in(&scratch).unwrap();
    let collection = Arc::new(create_collection_with_config("service", root.path(), config).await);
    let local = replica(&collection).await;
    let mut corpus = std::fs::File::open(env("DLI_P3_CORPUS")).unwrap();
    let mut read_rows = |ids: &[u32]| -> Vec<PointStructPersisted> {
        ids.iter()
            .map(|&id| {
                let mut bytes = vec![0u8; DIMENSIONS * 2];
                corpus.seek(SeekFrom::Start(((id as usize - 1) * DIMENSIONS * 2) as u64)).unwrap();
                corpus.read_exact(&mut bytes).unwrap();
                point_from_row(0, id as u64, &bytes)
            })
            .collect()
    };
    let read_u32 = |path: PathBuf| -> Vec<u32> {
        std::fs::read(path).unwrap().chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    };
    let request = |query: &Vec<f32>| CoreSearchRequest {
        query: QueryEnum::Nearest(NamedQuery::default_dense(query.clone())),
        filter: None, params: None, limit: 30, offset: 0, with_payload: None, with_vector: None, score_threshold: None,
    };
    let id_of = |p: &segment::types::ScoredPoint| match p.id {
        segment::types::ExtendedPointId::NumId(n) => n,
        other => panic!("unexpected id {other:?}"),
    };
    // Deletion generation: id -> generation after which it must never be returned.
    let deleted: Arc<parking_lot::RwLock<HashMap<u64, u64>>> = Default::default();
    let generation = Arc::new(AtomicU64::new(0));
    let mut total_violations = 0u64;
    let mut log = std::fs::File::create(output.join("service-batches.jsonl")).unwrap();
    for batch in plan["batches"].as_array().unwrap().iter().take(max_batches) {
        let index = batch["index"].as_u64().unwrap() as usize;
        let dir = plan_dir.join(format!("batch_{index:02}"));
        let spec = &batch["sets"][&set_name];
        let mut panel = read_set(&set_name, &plan_dir.join(spec["queries"].as_str().unwrap()), &plan_dir.join(spec["gold"].as_str().unwrap()));
        if let (Some(t), Some(o)) = (spec["tie"].as_str(), spec["tie_offsets"].as_str()) {
            panel = panel.with_ties(&plan_dir.join(t), &plan_dir.join(o));
        }
        let panel = Arc::new(panel);
        let queries: Arc<Vec<Vec<f32>>> = Arc::new(panel.queries.clone());
        for q in queries.iter().cycle().take(warmup) {
            let _ = collection.search(request(q), None, None, &ShardSelectorInternal::All, Some(Duration::from_millis(timeout_ms as u64))).await;
        }
        let counts_before = local.local_dli_search_counts_for_test().await;
        let counters_before = service_counters();

        // Readers: open-loop arrivals until `stop`.
        let stop = Arc::new(AtomicBool::new(false));
        let inflight = Arc::new(AtomicUsize::new(0));
        let results: Arc<parking_lot::Mutex<Vec<(f64, &'static str)>>> = Default::default();
        let violations = Arc::new(AtomicU64::new(0));
        let dropped = Arc::new(AtomicU64::new(0));
        let reader = {
            let (collection, stop, inflight, results, violations, dropped, deleted, generation, queries) = (
                collection.clone(), stop.clone(), inflight.clone(), results.clone(), violations.clone(),
                dropped.clone(), deleted.clone(), generation.clone(), queries.clone());
            tokio::spawn(async move {
                let period = Duration::from_secs_f64(1.0 / qps);
                let mut next = tokio::time::Instant::now();
                let mut i = 0usize;
                let mut tasks = Vec::new();
                while !stop.load(Ordering::Acquire) {
                    tokio::time::sleep_until(next).await;
                    next += period;
                    if inflight.load(Ordering::Acquire) >= inflight_cap {
                        dropped.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    inflight.fetch_add(1, Ordering::AcqRel);
                    let q = queries[i % queries.len()].clone();
                    i += 1;
                    let (collection, inflight, results, violations, deleted, generation) = (
                        collection.clone(), inflight.clone(), results.clone(), violations.clone(), deleted.clone(), generation.clone());
                    tasks.push(tokio::spawn(async move {
                        let _slot = Slot(inflight);
                        let start_gen = generation.load(Ordering::Acquire);
                        let started = Instant::now();
                        let outcome = collection
                            .search(
                                CoreSearchRequest {
                                    query: QueryEnum::Nearest(NamedQuery::default_dense(q)),
                                    filter: None, params: None, limit: 30, offset: 0,
                                    with_payload: None, with_vector: None, score_threshold: None,
                                },
                                None, None, &ShardSelectorInternal::All,
                                Some(Duration::from_millis(timeout_ms as u64)),
                            )
                            .await;
                        let ms = started.elapsed().as_secs_f64() * 1000.0;
                        let kind = match &outcome {
                            Ok(hits) => {
                                let map = deleted.read();
                                for hit in hits {
                                    let id = match hit.id {
                                        segment::types::ExtendedPointId::NumId(n) => n,
                                        _ => continue,
                                    };
                                    if map.get(&id).is_some_and(|&g| g <= start_gen) {
                                        violations.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                                "ok"
                            }
                            Err(e) if e.to_string().contains("imed out") => "timeout",
                            Err(_) => "error",
                        };
                        results.lock().push((ms, kind));
                    }.in_current_ambient()));
                }
                for t in tasks {
                    let _ = t.await;
                }
            }.in_current_ambient())
        };

        // Backlog sampler (1 s).
        let sampler = {
            let (collection, local, stop) = (collection.clone(), local.clone(), stop.clone());
            tokio::spawn(async move {
                let (mut max_backlog, mut busy_seconds) = (0usize, 0usize);
                while !stop.load(Ordering::Acquire) {
                    if static_system {
                        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
                        if info.status != crate::operations::types::CollectionStatus::Green {
                            busy_seconds += 1;
                        }
                    } else if let Some(d) = local.local_dli_diagnostics_for_test().await {
                        max_backlog = max_backlog.max(d.0.saturating_sub(d.1) + usize::from(d.3));
                        busy_seconds += usize::from(d.3 || d.0 > d.1);
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                (max_backlog, busy_seconds)
            }.in_current_ambient())
        };

        // Writer: paced inserts, then paced deletes.
        let window_started = Instant::now();
        let pace = |done: usize, started: Instant| {
            let due = Duration::from_secs_f64(done as f64 / write_rate);
            due.saturating_sub(started.elapsed())
        };
        let inserts = read_u32(dir.join("insert.u32"));
        let mut written = 0usize;
        for part in inserts.chunks(chunk) {
            apply_points(&collection, read_rows(part)).await;
            written += part.len();
            tokio::time::sleep(pace(written, window_started)).await;
        }
        for part in read_u32(dir.join("delete.u32")).chunks(chunk) {
            apply_delete(&collection, part.iter().map(|&i| i as u64).collect()).await;
            let g = generation.fetch_add(1, Ordering::AcqRel) + 1;
            {
                let mut map = deleted.write();
                for &i in part {
                    map.insert(i as u64, g);
                }
            }
            written += part.len();
            tokio::time::sleep(pace(written, window_started)).await;
        }
        let writes_seconds = window_started.elapsed().as_secs_f64();
        let settle_seconds = if static_system {
            settle_static_optimizers(&collection, Duration::from_secs(4 * 3600)).await
        } else {
            let s = Instant::now();
            settle_dli_maintenance(&local, Duration::from_secs(4 * 3600)).await;
            s.elapsed().as_secs_f64()
        };
        let window_seconds = window_started.elapsed().as_secs_f64();
        stop.store(true, Ordering::Release);
        reader.await.unwrap();
        let (max_backlog, busy_seconds) = sampler.await.unwrap();
        let window: Vec<(f64, &str)> = std::mem::take(&mut *results.lock());
        let ok: Vec<f64> = window.iter().filter(|r| r.1 == "ok").map(|r| r.0).collect();
        let batch_violations = violations.load(Ordering::Relaxed);
        total_violations += batch_violations;

        // Settled pass: sequential, recall against this batch's gold.
        let mut settled_ms = Vec::new();
        let mut recall30 = 0usize;
        let mut recall30_g1: Option<usize> = panel.ties.as_ref().map(|_| 0);
        for (qi, (q, gold)) in panel.queries.iter().zip(&panel.gold).enumerate() {
            let started = Instant::now();
            let hits = collection
                .search(request(q), None, None, &ShardSelectorInternal::All, Some(Duration::from_millis(timeout_ms as u64)))
                .await
                .expect("settled request must succeed");
            settled_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let ids: Vec<u64> = hits.iter().map(id_of).collect();
            recall30 += gold[..30].iter().filter(|g| ids.contains(g)).count();
            if let (Some(total), Some(ties)) = (recall30_g1.as_mut(), panel.ties.as_ref()) {
                *total += super::dli_selection_frontier::recall_g1(&ids, &ties[qi]);
            }
        }
        let (live, digest) = live_id_digest(&collection).await;
        assert_eq!(live as u64, batch["live_count"].as_u64().unwrap(), "batch {index} live count");
        assert_eq!(digest, batch["live_sha256"].as_str().unwrap(), "batch {index} live set");
        let counts_after = local.local_dli_search_counts_for_test().await;
        let record = serde_json::json!({
            "batch": index, "arm": arm_label, "live": live, "set": set_name,
            "load": {"qps_target": qps, "inflight_cap": inflight_cap, "timeout_ms": timeout_ms,
                     "write_rate_points_s": write_rate, "chunk": chunk},
            "window": {"seconds": window_seconds, "writes_seconds": writes_seconds, "settle_seconds": settle_seconds,
                       "issued": window.len(), "ok": ok.len(),
                       "timeouts": window.iter().filter(|r| r.1 == "timeout").count(),
                       "errors": window.iter().filter(|r| r.1 == "error").count(),
                       "dropped_at_cap": dropped.load(Ordering::Relaxed),
                       "achieved_qps": ok.len() as f64 / window_seconds.max(1e-9),
                       "latency_ms": percentiles(ok),
                       "deleted_point_violations": batch_violations,
                       "maintenance_busy_seconds": busy_seconds, "dli_max_backlog": max_backlog},
            "settled": {"latency_ms": percentiles(settled_ms),
                        "recall30": recall30 as f64 / (30 * panel.queries.len()) as f64,
                        "recall30_g1": recall30_g1.map(|r| r as f64 / (30 * panel.queries.len()) as f64)},
            "dli_learned_fallback_before_after": [counts_before, counts_after],
            "counters_before": counters_before, "counters_after": service_counters(),
            "segments": segment_topology(&collection).await,
        });
        writeln!(log, "{record}").unwrap();
        println!("SERVICE_BATCH {record}");
    }
    let summary = serde_json::json!({"arm": arm_label, "system": if static_system { "static" } else { "dli" },
        "static_model": static_model, "deleted_point_violations": total_violations,
        "source_head": std::env::var("DLI_BENCH_HEAD").ok()});
    std::fs::write(output.join("service-summary.json"), serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
    assert_eq!(total_violations, 0, "deleted points were returned");
    match Arc::try_unwrap(collection) {
        Ok(c) => c.stop_gracefully().await,
        Err(_) => panic!("collection still shared"),
    }
}
