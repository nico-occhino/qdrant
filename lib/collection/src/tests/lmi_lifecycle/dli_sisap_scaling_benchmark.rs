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
                None,
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
