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
    }
}

fn collection_config(dli: bool) -> CollectionConfigInternal {
    let mut vectors = VectorParamsBuilder::new(DIMENSIONS as u64, Distance::Cosine)
        .with_on_disk(true)
        .with_datatype(Datatype::Float16)
        .build();
    let model = model_config();
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
            routing_policy: Default::default(),
            maintenance_policy: if std::env::var("DLI_BENCH_POLICY").as_deref() == Ok("size-tiered")
            {
                crate::collection_manager::dli_schedule::MaintenancePolicy::SizeTiered
            } else {
                Default::default()
            },
            buffer_capacity: DLI_BUFFER_CAPACITY,
            arity: 2,
            candidate_budget: 2_000,
            model,
        }),
    }
}

async fn create_collection(name: &str, root: &Path, dli: bool) -> Collection {
    let data = root.join("storage");
    let snapshots = root.join("snapshots");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::create_dir_all(&snapshots).unwrap();
    let collection = Collection::new(
        name.to_owned(),
        1,
        &data,
        &snapshots,
        &collection_config(dli),
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
    let result = tokio::time::timeout(Duration::from_secs(30 * 60), async {
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

async fn run_queries(
    collection: &Collection,
    query_data: &[u8],
    gold: &[u8],
    query_count: usize,
    deleted_ids: &HashSet<u32>,
) -> (f64, f64, f64, f64, f64) {
    let mut timings = Vec::with_capacity(query_count);
    let mut hits10 = 0usize;
    let mut hits30 = 0usize;
    for query_idx in 0..query_count {
        let begin = query_idx * DIMENSIONS * 4;
        let vector: Vec<_> = query_data[begin..begin + DIMENSIONS * 4]
            .chunks_exact(4)
            .map(|f| f32::from_le_bytes([f[0], f[1], f[2], f[3]]))
            .collect();
        let work_before = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
        let started = Instant::now();
        let result = collection
            .search(
                CoreSearchRequest {
                    query: QueryEnum::Nearest(NamedQuery::default_dense(vector)),
                    filter: None,
                    params: None,
                    limit: 30,
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
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        timings.push(latency_ms);
        if let Ok(path) = std::env::var("DLI_QUERY_EVENTS") {
            use std::io::Write;
            let after = segment::index::hnsw_index::point_scorer::scoring_work_snapshot();
            let event = serde_json::json!({"query":query_idx,"latency_ms":latency_ms,
                "scored_vectors":after[2]-work_before[2],"scorer_calls":after[0]-work_before[0],"returned":result.len()});
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
    assert!(population >= 1_000_000 && population <= FULL_COUNT);
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
    for first in (0..population).step_by(BATCH_SIZE) {
        let end = (first + BATCH_SIZE).min(population);
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
            if end % DLI_BUFFER_CAPACITY == 0 {
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
        println!(
            "SCALING_CHECKPOINT system={system} population={end} elapsed={:.3}",
            ingestion.elapsed().as_secs_f64()
        );
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
    let result = serde_json::json!({"source_head":std::env::var("DLI_BENCH_HEAD").ok(),"system":system,
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
        rounds.push(serde_json::json!({"round":round,"materialize_topk":topk,"reserved":reserved,"population":population,"query_count":query_count,
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
