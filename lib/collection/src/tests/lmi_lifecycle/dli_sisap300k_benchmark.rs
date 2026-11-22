//! Long-running matched SISAP23 comparison. Invoke explicitly with
//! `--ignored --exact tests::lmi_lifecycle::dli_sisap300k_benchmark::sisap23_300k_static_lmi_vs_temporal_dli`.
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
const FULL_COUNT: usize = 300_000;
const BATCH_SIZE: usize = 50_000;
const DLI_BUFFER_CAPACITY: usize = 50_000;
const QUERY_COUNT: usize = 1_000;
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
        reuse_router: None,
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
            quantized_shortlist: None,
            routing_policy: Default::default(),
            maintenance_policy: if std::env::var("DLI_BENCH_POLICY").as_deref() == Ok("size-tiered")
            {
                crate::collection_manager::dli_schedule::MaintenancePolicy::SizeTiered
            } else {
                Default::default()
            },
            level_model: Default::default(),
            greedy_rebuild: false,
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
) -> (f64, f64, f64, f64, usize) {
    let mut timings = Vec::with_capacity(query_count);
    let mut hits10 = 0usize;
    let mut hits30 = 0usize;
    for query_idx in 0..query_count {
        let begin = query_idx * DIMENSIONS * 4;
        let vector: Vec<_> = query_data[begin..begin + DIMENSIONS * 4]
            .chunks_exact(4)
            .map(|f| f32::from_le_bytes([f[0], f[1], f[2], f[3]]))
            .collect();
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
        timings.push(started.elapsed().as_secs_f64() * 1_000.0);

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
        query_count,
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
#[ignore = "full canonical 300K SISAP23 dynamic benchmark; requires DLI_SISAP_INPUTS and DLI_SISAP_OUTPUT"]
async fn sisap23_300k_static_lmi_vs_temporal_dli() {
    let inputs = PathBuf::from(std::env::var("DLI_SISAP_INPUTS").expect("set DLI_SISAP_INPUTS"));
    let output = PathBuf::from(std::env::var("DLI_SISAP_OUTPUT").expect("set DLI_SISAP_OUTPUT"));
    std::fs::create_dir_all(&output).unwrap();
    let vectors = std::fs::read(inputs.join("vectors.f16")).unwrap();
    let queries = std::fs::read(inputs.join("queries.f32")).unwrap();
    let gold = std::fs::read(inputs.join("ground_truth.i32")).unwrap();
    assert_eq!(vectors.len(), FULL_COUNT * DIMENSIONS * 2);
    assert_eq!(queries.len(), 10_000 * DIMENSIONS * 4);
    assert_eq!(gold.len(), 10_000 * 1_000 * 4);

    let parent = tempfile::tempdir().unwrap();
    let static_root = parent.path().join("static-lmi");
    let dli_root = parent.path().join("temporal-dli");
    std::fs::create_dir(&static_root).unwrap();
    std::fs::create_dir(&dli_root).unwrap();
    let static_collection = create_collection("sisap23_static_lmi", &static_root, false).await;
    let dli_collection = create_collection("sisap23_temporal_dli", &dli_root, true).await;
    let dli_replica = replica(&dli_collection).await;

    let mut static_write_ms = Vec::new();
    let mut static_maintenance_ms = Vec::new();
    let mut dli_write_ms = Vec::new();
    let mut dli_publication_targets = Vec::new();
    let mut dli_inserted = 0usize;
    let population = std::env::var("DLI_BENCH_POINTS")
        .ok()
        .map(|s| s.parse::<usize>().expect("positive checkpoint size"))
        .unwrap_or(FULL_COUNT);
    assert!(population >= 100_000 && population <= FULL_COUNT && population % BATCH_SIZE == 0);
    let total_batches = population.div_ceil(BATCH_SIZE);
    for batch in 0..total_batches {
        let first = batch * BATCH_SIZE;
        let end = (first + BATCH_SIZE).min(population);
        let points: Vec<_> = (first..end)
            .map(|row| point_from_row(row, (row + 1) as u64, &vectors))
            .collect();

        static_write_ms.push(
            apply_points(&static_collection, points.clone())
                .await
                .as_secs_f64()
                * 1_000.0,
        );
        static_maintenance_ms.push(
            wait_for_static_index(&static_collection, end)
                .await
                .as_secs_f64()
                * 1_000.0,
        );

        dli_write_ms.push(apply_points(&dli_collection, points).await.as_secs_f64() * 1_000.0);
        if end >= DLI_BUFFER_CAPACITY && end % DLI_BUFFER_CAPACITY == 0 {
            let events = dli_replica
                .local_dli_publication_events_for_test()
                .await
                .unwrap();
            wait_for_dli_publications(&dli_replica, events.len() + 1).await;
            dli_inserted += 1;
            dli_publication_targets = dli_replica
                .local_dli_publication_events_for_test()
                .await
                .unwrap()
                .iter()
                .map(|event| event.target_level)
                .collect();
        }

        if end % BATCH_SIZE == 0 {
            let static_stats = active_static_lmi_stats(&static_collection).await;
            let dli_events = dli_replica
                .local_dli_publication_events_for_test()
                .await
                .unwrap();
            println!(
                "SISAP23_CHECKPOINT points={end} static_lmi={static_stats:?} dli_publications={} dli_targets={:?}",
                dli_events.len(),
                dli_publication_targets
            );
        }
    }
    assert!(dli_inserted >= 2);
    let (static_lmi_segments, static_indexed_points, static_learned_queries, static_candidates) =
        active_static_lmi_stats(&static_collection).await;
    assert!(
        static_lmi_segments > 0 && static_indexed_points > 0,
        "static baseline must have active LMI segments"
    );

    let empty_deleted = HashSet::new();
    let scorer_snapshot = segment::index::hnsw_index::point_scorer::scoring_work_snapshot;
    let static_score_start = scorer_snapshot();
    let static_before = run_queries(
        &static_collection,
        &queries,
        &gold,
        QUERY_COUNT,
        &empty_deleted,
    )
    .await;
    let static_scored =
        std::array::from_fn::<_, 3, _>(|i| scorer_snapshot()[i] - static_score_start[i]);
    let dli_score_start = scorer_snapshot();
    let dli_before = run_queries(
        &dli_collection,
        &queries,
        &gold,
        QUERY_COUNT,
        &empty_deleted,
    )
    .await;
    let dli_scored = std::array::from_fn::<_, 3, _>(|i| scorer_snapshot()[i] - dli_score_start[i]);
    let static_after_query_stats = active_static_lmi_stats(&static_collection).await;
    let dli_before_counts = dli_replica
        .local_dli_search_metrics_for_test()
        .await
        .unwrap();
    assert!(
        static_after_query_stats.2 > static_learned_queries,
        "static LMI baseline must execute learned search"
    );
    assert!(
        dli_before_counts.0 > 0,
        "DLI benchmark must execute learned search"
    );

    // Identical mixed update stream: 128 same-vector upserts and 128 deletes.
    // Official top-1000 remains a complete live oracle for top-30 after deleting
    // at most 128 IDs; same-vector upserts do not change cosine rankings.
    let upserts: Vec<_> = (0..DYNAMIC_UPSERTS)
        .map(|row| point_from_row(row, (row + 1) as u64, &vectors))
        .collect();
    let static_upsert_ms = apply_points(&static_collection, upserts.clone())
        .await
        .as_secs_f64()
        * 1_000.0;
    let dli_upsert_ms = apply_points(&dli_collection, upserts).await.as_secs_f64() * 1_000.0;
    let deleted: HashSet<u32> =
        (DYNAMIC_UPSERTS as u32 + 1..=(DYNAMIC_UPSERTS + DYNAMIC_DELETIONS) as u32).collect();
    let delete_ids: Vec<_> = deleted.iter().copied().map(u64::from).collect();
    let static_delete_ms = apply_delete(&static_collection, delete_ids.clone())
        .await
        .as_secs_f64()
        * 1_000.0;
    let dli_delete_ms = apply_delete(&dli_collection, delete_ids)
        .await
        .as_secs_f64()
        * 1_000.0;
    let static_after = run_queries(
        &static_collection,
        &queries,
        &gold,
        DYNAMIC_QUERY_COUNT,
        &deleted,
    )
    .await;
    let dli_after = run_queries(
        &dli_collection,
        &queries,
        &gold,
        DYNAMIC_QUERY_COUNT,
        &deleted,
    )
    .await;
    let dli_after_counts = dli_replica
        .local_dli_search_metrics_for_test()
        .await
        .unwrap();
    assert!(
        dli_after_counts.0 > dli_before_counts.0,
        "supported post-update DLI must execute learned routing"
    );

    let dli_events = dli_replica
        .local_dli_publication_events_for_test()
        .await
        .unwrap();
    let dli_rebuilt: usize = dli_events
        .iter()
        .map(|event| event.published_output_points)
        .sum();
    let dli_optimizer_micros: u128 = dli_events.iter().map(|event| event.optimizer_micros).sum();
    let static_auxiliary = auxiliary_index_bytes(&static_root);
    let dli_auxiliary = auxiliary_index_bytes(&dli_root);
    let result = serde_json::json!({
        "source_head": std::env::var("DLI_BENCH_HEAD").unwrap_or_else(|_| "unset".into()),
        "optimized_build": !cfg!(debug_assertions),
        "vectors": population,
        "dimensions": DIMENSIONS,
        "batch_size": BATCH_SIZE,
        "queries_before": QUERY_COUNT,
        "queries_after_mixed_updates": DYNAMIC_QUERY_COUNT,
        "static_lmi": {
            "batch_scorer_work_calls_attempted_successful": static_scored,
            "active_lmi_segments": static_lmi_segments,
            "indexed_points": static_indexed_points,
            "learned_queries": static_after_query_stats.2,
            "candidate_offsets": static_after_query_stats.3,
            "candidate_offsets_before_queries": static_candidates,
            "insert_ms": static_write_ms,
            "maintenance_until_indexed_ms": static_maintenance_ms,
            "recall10": static_before.0,
            "recall30": static_before.1,
            "query_p50_ms": static_before.2,
            "query_p95_ms": static_before.3,
            "mixed_recall10": static_after.0,
            "mixed_recall30": static_after.1,
            "mixed_query_p50_ms": static_after.2,
            "mixed_query_p95_ms": static_after.3,
            "upsert_128_ms": static_upsert_ms,
            "delete_128_ms": static_delete_ms,
            "auxiliary_index_bytes": static_auxiliary,
        },
        "temporal_dli": {
            "batch_scorer_work_calls_attempted_successful": dli_scored,
            "publications": dli_events.iter().map(|event| serde_json::json!({
                "target_level": event.target_level,
                "source_uuids": event.source_uuids,
                "destination_uuid": event.destination_uuid,
                "expected_output_points": event.expected_output_points,
                "published_output_points": event.published_output_points,
                "optimizer_micros": event.optimizer_micros,
            })).collect::<Vec<_>>(),
            "publication_count": dli_events.len(),
            "vectors_rebuilt": dli_rebuilt,
            "cumulative_optimizer_micros": dli_optimizer_micros,
            "learned_queries": dli_before_counts.0,
            "exact_fallback_queries_after_updates": dli_after_counts.1 - dli_before_counts.1,
            "learned_queries_after_updates": dli_after_counts.0 - dli_before_counts.0,
            "posting_candidates": dli_before_counts.2,
            "unique_candidates": dli_before_counts.3,
            "recall10": dli_before.0,
            "recall30": dli_before.1,
            "query_p50_ms": dli_before.2,
            "query_p95_ms": dli_before.3,
            "mixed_recall10": dli_after.0,
            "mixed_recall30": dli_after.1,
            "mixed_query_p50_ms": dli_after.2,
            "mixed_query_p95_ms": dli_after.3,
            "insert_ms": dli_write_ms,
            "upsert_128_ms": dli_upsert_ms,
            "delete_128_ms": dli_delete_ms,
            "auxiliary_index_bytes": dli_auxiliary,
        },
        "process_peak_rss_kib": peak_rss_kib(),
        "limitations": ["single process sequential query timing", "static maintenance time is insert-to-indexed wall time", "official top-1000 is used for post-delete top-30 because at most 128 of its IDs are removed"],
    });
    let report_path = output.join("summary.json");
    std::fs::write(&report_path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    println!(
        "SISAP23_300K_RESULT {}",
        serde_json::to_string(&result).unwrap()
    );

    static_collection.stop_gracefully().await;
    dli_collection.stop_gracefully().await;
}
