use std::collections::HashSet;
use std::io::Read;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use ahash::AHashMap;
use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use half::f16;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector};
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexEnum;
use segment::index::VectorIndexRead;
use segment::index::lmi_index::{LmiConfig, LmiStateMetadata};
use segment::segment_constructor::load_segment;
use segment::types::{Distance, PointIdType};
use shard::operations::CollectionUpdateOperations;
use shard::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorStructPersisted,
};
use shard::query::query_enum::QueryEnum;
use uuid::Uuid;

use crate::collection::Collection;
use crate::config::{CollectionConfigInternal, CollectionParams};
use crate::operations::OperationWithClockTag;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::operations::types::{CoreSearchRequest, Datatype, VectorsConfig};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::optimizers_builder::OptimizersConfig;
use crate::shards::channel_service::ChannelService;
use crate::shards::collection_shard_distribution::CollectionShardDistribution;
use crate::shards::replica_set::replica_set_state::ReplicaState;
use crate::shards::shard_trait::WaitUntil;
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

/// Expensive and intentionally opt-in. Refuses to overwrite any earlier run.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit full SISAP300K collection-lifecycle and 9,980-query gate"]
async fn sisap300k_full_collection_lifecycle_and_queries() {
    let input = Path::new("/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap300k");
    let output = input.join("collection_full_2026_10_07_attempt2");
    assert!(!output.exists(), "fresh output path required");
    std::fs::create_dir_all(&output).unwrap();
    let collection_path = output.join("collection");
    let snapshots_path = output.join("snapshots");
    std::fs::create_dir_all(&collection_path).unwrap();
    std::fs::create_dir_all(&snapshots_path).unwrap();

    let lmi = LmiConfig {
        n_buckets: 548,
        sample_size: 32768,
        hidden_dim: 512,
        epochs: 30,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations: 5,
        nprobe: 4,
        seed: 42,
    };
    let mut vector_params = VectorParamsBuilder::new(768, Distance::Cosine).build();
    vector_params.datatype = Some(Datatype::Float16);
    vector_params.lmi_config = Some(lmi);
    let mut optimizers = OptimizersConfig::fixture();
    optimizers.max_optimization_threads = Some(0);
    optimizers.indexing_threshold = Some(1);
    optimizers.max_segment_size = Some(1_000_000);
    optimizers.default_segment_number = 1;
    let config = CollectionConfigInternal {
        params: CollectionParams {
            vectors: VectorsConfig::Single(vector_params),
            shard_number: NonZeroU32::new(1).unwrap(),
            ..CollectionParams::empty()
        },
        optimizer_config: optimizers,
        wal_config: Default::default(),
        hnsw_config: Default::default(),
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
    };
    let collection = Collection::new(
        "sisap300k_lmi".to_string(),
        1,
        &collection_path,
        &snapshots_path,
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
    let mut stream =
        std::io::BufReader::new(std::fs::File::open(input.join("vectors.f16")).unwrap());
    let mut raw = vec![0_u8; 768 * 2];
    let ingest_started = Instant::now();
    for start in (0..300_000_u64).step_by(1000) {
        let mut batch = Vec::with_capacity(1000);
        for id in start + 1..=start + 1000 {
            stream.read_exact(&mut raw).unwrap();
            let vector = raw
                .chunks_exact(2)
                .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
                .collect();
            batch.push(PointStructPersisted {
                id: id.into(),
                vector: VectorStructPersisted::Single(vector),
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
        if (start + 1000) % 50_000 == 0 {
            println!(
                "sisap_collection_ingested={} elapsed_s={:.2}",
                start + 1000,
                ingest_started.elapsed().as_secs_f64()
            );
        }
    }
    assert_eq!(stream.read(&mut [0_u8]).unwrap(), 0);
    let ingest_seconds = ingest_started.elapsed().as_secs_f64();
    let before = collection.info(&ShardSelectorInternal::All).await.unwrap();
    assert_eq!(before.points_count, Some(300_000));
    assert_eq!(before.indexed_vectors_count, Some(0));

    let diff = serde_json::from_value(serde_json::json!({"max_optimization_threads": 1})).unwrap();
    collection
        .update_optimizer_params_from_diff(diff)
        .await
        .unwrap();
    collection.recreate_optimizers_background();
    let build_started = Instant::now();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(1200);
    loop {
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        if info.indexed_vectors_count == Some(300_000) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "300K LMI target not published: {:?}",
            info.optimizer_status
        );
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    let build_seconds = build_started.elapsed().as_secs_f64();
    println!("sisap_collection_build_s={build_seconds:.3}");
    let mut states = Vec::new();
    find_state(&collection_path, &mut states);
    assert_eq!(states.len(), 1, "expected one optimizer-built LMI target");
    let metadata: LmiStateMetadata =
        serde_json::from_slice(&std::fs::read(&states[0]).unwrap()).unwrap();
    assert_eq!(metadata.indexed_live_count, Some(300_000));
    assert_eq!(metadata.config, lmi);
    let sizes: Vec<_> = ["lmi_state.json", "lmi_router.bin", "lmi_postings.bin"]
        .iter()
        .map(|name| {
            std::fs::metadata(states[0].parent().unwrap().join(name))
                .unwrap()
                .len()
        })
        .collect();

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
    drop(shard);
    collection.stop_gracefully().await;
    drop(collection);
    let reopened = Collection::load(
        "sisap300k_lmi_reopened".to_string(),
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
        Some(300_000)
    );
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
        panic!("optimized 300K target did not reopen as LMI")
    };
    let state = index.routing_state().unwrap();
    assert_eq!(state.postings().point_count(), 300_000);
    let full = input.join("full_queries_2026_10_07");
    let mut queries =
        std::io::BufReader::new(std::fs::File::open(full.join("queries_20_9999.f32")).unwrap());
    let mut gold =
        std::io::BufReader::new(std::fs::File::open(full.join("gold_20_9999_top10.i32")).unwrap());
    let mut qraw = vec![0_u8; 768 * 4];
    let mut graw = [0_u8; 40];
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
        if (row + 1) % 1000 == 0 {
            println!("sisap_collection_queries={}", row + 1);
        }
    }
    let mean_recall = recalls.iter().sum::<f64>() / 9980.0;
    let mean_candidates = candidates.iter().sum::<f64>() / 9980.0;
    let summary = serde_json::json!({
        "query_count": 9980,
        "query_rows": "20..9999",
        "recall_at_10_mean": mean_recall,
        "recall_at_10_p05": quantile(&recalls, 0.05),
        "recall_at_10_p50": quantile(&recalls, 0.50),
        "recall_at_10_p95": quantile(&recalls, 0.95),
        "mean_candidates": mean_candidates,
        "candidates_p50": quantile(&candidates, 0.50),
        "candidates_p95": quantile(&candidates, 0.95),
        "candidates_p99": quantile(&candidates, 0.99),
        "mean_candidate_fraction": mean_candidates / 300000.0,
        "native_search_ms_p50": quantile(&latency_ms, 0.50),
        "native_search_ms_p95": quantile(&latency_ms, 0.95),
        "state_router_postings_bytes": sizes,
        "ingest_seconds": ingest_seconds,
        "build_seconds": build_seconds,
    });
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    println!("sisap_collection_full_summary={summary}");
    assert!(mean_recall > 0.5, "material retrieval failure");
}
