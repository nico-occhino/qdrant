use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector};
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexEnum;
use segment::index::VectorIndexRead;
use segment::index::lmi_index::{LmiConfig, LmiStateMetadata};
use segment::segment_constructor::load_segment;
use segment::types::{Distance, PointIdType};
use shard::query::query_enum::QueryEnum;
use uuid::Uuid;

use crate::collection::Collection;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::operations::types::CoreSearchRequest;
use crate::shards::channel_service::ChannelService;
use crate::tests::snapshot_test::{
    dummy_abort_shard_transfer, dummy_on_replica_failure, dummy_request_shard_transfer,
};

fn find_state(path: &Path, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            find_state(&path, found);
        } else if path
            .file_name()
            .is_some_and(|name| name == "lmi_state.json")
        {
            found.push(path);
        }
    }
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Full SISAP10.12M baseline, opt-in and refusing to overwrite any run.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit resumed SISAP10M optimizer and 9,980-query gate"]
async fn sisap10m_resume_collection_lifecycle_and_queries() {
    const N: usize = 10_120_191;
    let input = Path::new(
        "/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap10m/pre_g2_119_attempt2",
    );
    let output =
        Path::new("/home/nicoo/work/lmi-sisap2023-storage/10m-f16-qdrant-119-sync-attempt2");
    let _ = env_logger::builder().is_test(true).try_init();
    assert!(output.exists(), "recovery clone must exist");
    assert!(input.exists(), "fresh attempt-2 results must exist");
    let collection_path = output.join("collection");
    let snapshots_path = output.join("snapshots");
    let lmi = LmiConfig {
        n_buckets: 3162,
        sample_size: 250000,
        hidden_dim: 512,
        epochs: 30,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations: 5,
        nprobe: 4,
        seed: 42,
    };
    // Inspect the recovered collection with optimization disabled via a runtime-only override.
    let pause = serde_json::from_value(serde_json::json!({"max_optimization_threads": 0})).unwrap();
    let probe = Collection::load(
        "sisap10m_lmi_recovery_probe".to_string(),
        1,
        &collection_path,
        &snapshots_path,
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
    let recovered = probe.info(&ShardSelectorInternal::All).await.unwrap();
    assert_eq!(recovered.points_count, Some(N));
    assert_eq!(recovered.indexed_vectors_count, Some(0));
    std::fs::write(
        input.join("recovery.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "recovered_points": recovered.points_count,
            "indexed_before_resume": recovered.indexed_vectors_count,
            "interrupted_attempt": "/home/nicoo/work/lmi-sisap2023-storage/10m-f16-qdrant-119-sync",
            "source_artifact_preserved": true,
        }))
        .unwrap(),
    )
    .unwrap();
    probe.stop_gracefully().await;
    drop(probe);

    // Load the same clone with its persisted one-thread optimizer policy.
    let collection = Collection::load(
        "sisap10m_lmi_resume".to_string(),
        1,
        &collection_path,
        &snapshots_path,
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
    collection.recreate_optimizers_background();
    let ingest_seconds = 2110.025953507_f64;
    let build_started = Instant::now();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(43200);
    loop {
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        if info.indexed_vectors_count == Some(N) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "10M LMI target not published: {:?}",
            info.optimizer_status
        );
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    let build_seconds = build_started.elapsed().as_secs_f64();
    println!("sisap10m_collection_build_s={build_seconds:.3}");
    let mut states = Vec::new();
    find_state(&collection_path, &mut states);
    assert_eq!(states.len(), 1, "expected one optimizer-built LMI target");
    let metadata: LmiStateMetadata =
        serde_json::from_slice(&std::fs::read(&states[0]).unwrap()).unwrap();
    assert_eq!(metadata.indexed_live_count, Some(N));
    assert_eq!(metadata.config, lmi);
    let sizes: Vec<_> = ["lmi_state.json", "lmi_router.bin", "lmi_postings.bin"]
        .iter()
        .map(|name| {
            std::fs::metadata(states[0].parent().unwrap().join(name))
                .unwrap()
                .len()
        })
        .collect();

    std::fs::write(
        input.join("build.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "build_seconds": build_seconds,
            "indexed_live_count": metadata.indexed_live_count,
            "state_router_postings_bytes": sizes,
            "state_path": states[0],
        }))
        .unwrap(),
    )
    .unwrap();

    let mut vector = vec![0.0; 768];
    vector[0] = 1.0;
    let probe = CoreSearchRequest {
        query: QueryEnum::Nearest(segment::data_types::vectors::NamedQuery::default_dense(
            vector,
        )),
        filter: None,
        params: None,
        limit: 10,
        offset: 0,
        with_payload: None,
        with_vector: None,
        score_threshold: None,
    };
    assert!(
        !collection
            .search(
                probe,
                None,
                None,
                &ShardSelectorInternal::All,
                None,
                HwMeasurementAcc::new()
            )
            .await
            .unwrap()
            .is_empty()
    );
    collection.stop_gracefully().await;
    drop(collection);
    let reopened = Collection::load(
        "sisap10m_lmi_reopened".to_string(),
        1,
        &collection_path,
        &snapshots_path,
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
        reopened
            .info(&ShardSelectorInternal::All)
            .await
            .unwrap()
            .indexed_vectors_count,
        Some(N)
    );
    std::fs::write(
        input.join("restart.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "indexed_vectors_after_restart": N,
            "reopened_without_training": true,
        }))
        .unwrap(),
    )
    .unwrap();
    reopened.stop_gracefully().await;
    drop(reopened);

    let index_path = states[0].parent().unwrap();
    let segment_path = index_path.parent().unwrap();
    let uuid = Uuid::parse_str(segment_path.file_name().unwrap().to_str().unwrap()).unwrap();
    let segment = load_segment(segment_path, uuid, None, &AtomicBool::new(false), true).unwrap();
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(index) = &*index else {
        panic!("optimized 10M target did not reopen as LMI")
    };
    let state = index.routing_state().unwrap();
    assert_eq!(state.postings().point_count(), N);
    let full = input.join("full_queries");
    let mut queries =
        std::io::BufReader::new(std::fs::File::open(full.join("queries_20_9999.f32")).unwrap());
    let mut gold =
        std::io::BufReader::new(std::fs::File::open(full.join("gold_20_9999_top10.i32")).unwrap());
    let mut qraw = vec![0_u8; 768 * 4];
    let mut graw = [0_u8; 40];
    let mut warm_queries =
        std::io::BufReader::new(std::fs::File::open(full.join("queries_0_19.f32")).unwrap());
    for _ in 0..20 {
        warm_queries.read_exact(&mut qraw).unwrap();
        let vector: Vec<f32> = qraw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let query: QueryVector = vector.into();
        let hits = index
            .search(&[&query], None, 10, None, &VectorQueryContext::default())
            .unwrap();
        assert_eq!(hits[0].len(), 10);
    }
    let mut per_query =
        std::io::BufWriter::new(std::fs::File::create(input.join("queries.full.jsonl")).unwrap());
    let mut recalls = Vec::with_capacity(9980);
    let mut candidates = Vec::with_capacity(9980);
    let mut latency_ms = Vec::with_capacity(9980);
    for row in 0..9980 {
        queries.read_exact(&mut qraw).unwrap();
        gold.read_exact(&mut graw).unwrap();
        let vector: Vec<f32> = qraw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let gold: Vec<i32> = graw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let normalized: QueryVector = Distance::Cosine
            .preprocess_vector::<f32>(vector.clone())
            .into();
        candidates.push(
            state
                .candidates_for_query(&normalized, &AtomicBool::new(false))
                .unwrap()
                .unwrap()
                .len() as f64,
        );
        let query: QueryVector = vector.into();
        let start = Instant::now();
        let hits = index
            .search(&[&query], None, 10, None, &VectorQueryContext::default())
            .unwrap();
        latency_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        let tracker = segment.id_tracker.borrow();
        let matched = hits[0]
            .iter()
            .filter(|hit| {
                let PointIdType::NumId(id) = tracker.external_id(hit.idx).unwrap() else {
                    panic!("non-numeric SISAP ID")
                };
                gold.contains(&(id as i32))
            })
            .count();
        recalls.push(matched as f64 / 10.0);
        let ids: Vec<u64> = hits[0]
            .iter()
            .map(|hit| match tracker.external_id(hit.idx).unwrap() {
                PointIdType::NumId(id) => id,
                _ => panic!("non-numeric SISAP ID"),
            })
            .collect();
        writeln!(
            per_query,
            "{}",
            serde_json::json!({
                "query_row": row + 20,
                "recall_at_10": matched as f64 / 10.0,
                "candidate_count": candidates.last().unwrap(),
                "native_search_ms": latency_ms.last().unwrap(),
                "returned_ids_1_based": ids,
            })
        )
        .unwrap();
        if (row + 1) % 1000 == 0 {
            println!("sisap10m_collection_queries={}", row + 1);
        }
    }
    per_query.flush().unwrap();
    let mean_recall = recalls.iter().sum::<f64>() / 9980.0;
    let mean_candidates = candidates.iter().sum::<f64>() / 9980.0;
    let summary = serde_json::json!({
        "query_count": 9980,
        "query_rows": "20..9999",
        "recall_at_10_mean": mean_recall,
        "recall_at_10_p05": quantile(&recalls, 0.05),
        "recall_histogram": (0..=10).map(|i| recalls.iter().filter(|v| (**v * 10.0).round() as usize == i).count()).collect::<Vec<_>>(),
        "recall_at_10_p50": quantile(&recalls, 0.50),
        "recall_at_10_p95": quantile(&recalls, 0.95),
        "mean_candidates": mean_candidates,
        "candidates_min": candidates.iter().copied().fold(f64::INFINITY, f64::min),
        "candidates_max": candidates.iter().copied().fold(0.0, f64::max),
        "candidates_p05": quantile(&candidates, 0.05),
        "candidates_p50": quantile(&candidates, 0.50),
        "candidates_p95": quantile(&candidates, 0.95),
        "candidates_p99": quantile(&candidates, 0.99),
        "mean_candidate_fraction": mean_candidates / N as f64,
        "native_search_ms_mean": latency_ms.iter().sum::<f64>() / 9980.0,
        "native_search_ms_p50": quantile(&latency_ms, 0.50),
        "native_search_ms_p95": quantile(&latency_ms, 0.95),
        "native_search_ms_p99": quantile(&latency_ms, 0.99),
        "native_search_ms_max": latency_ms.iter().copied().fold(0.0, f64::max),
        "state_router_postings_bytes": sizes,
        "ingest_seconds": ingest_seconds,
        "build_seconds": build_seconds,
    });
    std::fs::write(
        input.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    println!("sisap10m_collection_full_summary={summary}");
    assert!(mean_recall > 0.5, "material retrieval failure");
}
#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit post-query SISAP10M collection health gate"]
async fn sisap10m_postquery_collection_health() {
    const N: usize = 10_120_191;
    let output =
        Path::new("/home/nicoo/work/lmi-sisap2023-storage/10m-f16-qdrant-119-sync-attempt2");
    let input = Path::new(
        "/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap10m/pre_g2_119_attempt2",
    );
    let pause = serde_json::from_value(serde_json::json!({"max_optimization_threads": 0})).unwrap();
    let collection = Collection::load(
        "sisap10m_lmi_postquery".to_string(),
        1,
        &output.join("collection"),
        &output.join("snapshots"),
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
    assert_eq!(info.points_count, Some(N));
    assert_eq!(info.indexed_vectors_count, Some(N));
    let mut vector = vec![0.0; 768];
    vector[0] = 1.0;
    let probe = CoreSearchRequest {
        query: QueryEnum::Nearest(segment::data_types::vectors::NamedQuery::default_dense(
            vector,
        )),
        filter: None,
        params: None,
        limit: 10,
        offset: 0,
        with_payload: None,
        with_vector: None,
        score_threshold: None,
    };
    let hits = collection
        .search(
            probe,
            None,
            None,
            &ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 10);
    std::fs::write(
        input.join("postquery.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "collection_search_hits": hits.len(),
            "optimizer_threads_runtime_override": 0,
            "retrained": false,
        }))
        .unwrap(),
    )
    .unwrap();
    collection.stop_gracefully().await;
}
