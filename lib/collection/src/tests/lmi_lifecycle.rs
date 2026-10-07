use std::collections::HashSet;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
use common::budget::ResourceBudget;
use common::counter::hardware_accumulator::HwMeasurementAcc;
use segment::data_types::vectors::NamedQuery;
use segment::index::lmi_index::{LmiConfig, LmiStateMetadata};
use segment::types::{Condition, Distance, Filter, SearchParams};
use sha2::{Digest, Sha256};
use shard::operations::CollectionUpdateOperations;
use shard::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorStructPersisted,
};
use shard::query::query_enum::QueryEnum;
use shard::snapshots::snapshot_data::SnapshotData;
use tempfile::Builder;

use crate::collection::Collection;
use crate::config::{CollectionConfigInternal, CollectionParams};
use crate::operations::OperationWithClockTag;
use crate::operations::shard_selector_internal::ShardSelectorInternal;
use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::operations::types::{CoreSearchRequest, VectorsConfig};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::optimizers_builder::OptimizersConfig;
use crate::shards::channel_service::ChannelService;
use crate::shards::collection_shard_distribution::CollectionShardDistribution;
use crate::shards::replica_set::replica_set_state::ReplicaState;
use crate::shards::shard_trait::WaitUntil;
use crate::tests::snapshot_test::{
    dummy_abort_shard_transfer, dummy_on_replica_failure, dummy_request_shard_transfer,
};

fn lmi_generations(root: &Path) -> Vec<(usize, [Vec<u8>; 3])> {
    fn visit(path: &Path, out: &mut Vec<PathBuf>) {
        if !path.is_dir() {
            return;
        }
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, out);
            } else if path
                .file_name()
                .is_some_and(|name| name == "lmi_state.json")
            {
                out.push(path);
            }
        }
    }
    let mut states = Vec::new();
    visit(root, &mut states);
    states
        .into_iter()
        .map(|state_path| {
            let index_dir = state_path.parent().unwrap();
            let metadata: LmiStateMetadata =
                serde_json::from_slice(&std::fs::read(&state_path).unwrap()).unwrap();
            let names = ["lmi_state.json", "lmi_router.bin", "lmi_postings.bin"];
            let hashes = names.map(|name| {
                let bytes = std::fs::read(index_dir.join(name)).unwrap();
                Sha256::digest(bytes).to_vec()
            });
            (metadata.indexed_live_count.unwrap(), hashes)
        })
        .collect()
}

fn point(id: u64, axis: usize) -> PointStructPersisted {
    let mut vector = vec![0.0; 16];
    vector[axis] = 1.0;
    PointStructPersisted {
        id: id.into(),
        vector: VectorStructPersisted::Single(vector),
        payload: None,
    }
}

async fn search_with(
    collection: &Collection,
    axis: usize,
    filter: Option<Filter>,
    params: Option<SearchParams>,
) -> Vec<u64> {
    let mut vector = vec![0.0; 16];
    vector[axis] = 1.0;
    collection
        .search(
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(vector)),
                filter,
                params,
                limit: 50,
                offset: 0,
                with_payload: None,
                with_vector: None,
                score_threshold: None,
            },
            None,
            None,
            &ShardSelectorInternal::All,
            None,
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap()
        .into_iter()
        .filter_map(|point| match point.id {
            segment::types::ExtendedPointId::NumId(id) => Some(id),
            _ => None,
        })
        .collect()
}

async fn search(collection: &Collection, axis: usize) -> Vec<u64> {
    search_with(collection, axis, None, None).await
}

#[tokio::test(flavor = "multi_thread")]
async fn lmi_collection_builds_and_searches_with_fresh_plain_writes() {
    let collection_dir = Builder::new().prefix("lmi_collection").tempdir().unwrap();
    let snapshots_dir = Builder::new().prefix("lmi_snapshots").tempdir().unwrap();
    let mut vector_params = VectorParamsBuilder::new(16, Distance::Cosine).build();
    vector_params.lmi_config = Some(LmiConfig {
        n_buckets: 2,
        sample_size: 32,
        hidden_dim: 8,
        epochs: 1,
        batch_size: 8,
        routing_batch_size: 8,
        kmeans_iterations: 1,
        nprobe: 2,
        seed: 7,
    });
    let mut optimizer_config = OptimizersConfig::fixture();
    optimizer_config.indexing_threshold = Some(1);
    optimizer_config.max_optimization_threads = Some(1);
    optimizer_config.default_segment_number = 2;
    let config = CollectionConfigInternal {
        params: CollectionParams {
            vectors: VectorsConfig::Single(vector_params),
            shard_number: NonZeroU32::new(1).unwrap(),
            ..CollectionParams::empty()
        },
        optimizer_config,
        wal_config: Default::default(),
        hnsw_config: Default::default(),
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
    };
    let collection = Collection::new(
        "lmi_lifecycle".to_string(),
        1,
        collection_dir.path(),
        snapshots_dir.path(),
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
    let initial: Vec<_> = (0..32).map(|id| point(id, (id % 16) as usize)).collect();
    shard
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(initial)),
            )),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .unwrap();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        if info.indexed_vectors_count.unwrap_or(0) >= 32 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "LMI optimizer did not publish: {:?}",
            info.optimizer_status
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(search(&collection, 0).await.contains(&0));
    let first_generations = lmi_generations(collection_dir.path());
    assert!(first_generations.iter().any(|(count, _)| *count == 32));

    shard
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(vec![
                    point(100, 0),
                ])),
            )),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .unwrap();
    assert!(search(&collection, 0).await.contains(&100));
    let filtered = search_with(
        &collection,
        0,
        Some(Filter::new_must(Condition::HasId(
            AHashSet::from([0_u64.into(), 100_u64.into()]).into(),
        ))),
        None,
    )
    .await;
    assert_eq!(filtered.len(), 2);
    assert!(filtered.contains(&0) && filtered.contains(&100));
    assert!(
        search_with(
            &collection,
            0,
            None,
            Some(SearchParams {
                exact: true,
                ..Default::default()
            })
        )
        .await
        .contains(&0)
    );
    assert!(
        search_with(
            &collection,
            0,
            None,
            Some(SearchParams {
                hnsw_ef: Some(16),
                ..Default::default()
            })
        )
        .await
        .contains(&100)
    );

    // Update one old logical ID and delete another while the first LMI
    // generation is already published. Both operations must remain visible
    // through collection search and the next target rebuild.
    shard
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(vec![
                    point(0, 1),
                ])),
            )),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .unwrap();
    shard
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::DeletePoints {
                    ids: vec![1_u64.into()],
                },
            )),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .unwrap();
    assert!(!search(&collection, 0).await.contains(&1));
    assert!(search(&collection, 1).await.contains(&0));

    // Fill another appendable segment so the optimizer publishes a second
    // trained generation through its normal scheduling path.
    let second_batch: Vec<_> = (101..133).map(|id| point(id, (id % 16) as usize)).collect();
    shard
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(
                    second_batch,
                )),
            )),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        if info.indexed_vectors_count.unwrap_or(0) >= 63 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "second LMI optimization did not publish: {:?}",
            info.optimizer_status
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let second_generations = lmi_generations(collection_dir.path());
    assert!(
        second_generations.iter().any(|(count, _)| *count >= 63),
        "second generation did not include the fresh mutable vectors: {second_generations:?}"
    );
    assert!(
        second_generations.iter().any(|(_, hashes)| {
            !first_generations
                .iter()
                .any(|(_, prior_hashes)| prior_hashes == hashes)
        }),
        "second generation reused the complete old LMI state"
    );
    let before_restart = search(&collection, 1).await;
    assert!(before_restart.contains(&0));
    assert!(!before_restart.contains(&1));
    assert!(before_restart.contains(&113));

    let third_batch: Vec<_> = (133..165).map(|id| point(id, (id % 16) as usize)).collect();
    shard
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(
                    third_batch,
                )),
            )),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .unwrap();
    shard
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(vec![
                    point(100, 2),
                ])),
            )),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .unwrap();
    shard
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::DeletePoints {
                    ids: vec![2_u64.into()],
                },
            )),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        if info.indexed_vectors_count.unwrap_or(0) >= 95 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "third LMI optimization did not publish: {:?}",
            info.optimizer_status
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let third_generations = lmi_generations(collection_dir.path());
    assert!(
        third_generations.iter().any(|(count, _)| *count >= 95),
        "third generation did not index all current vectors: {third_generations:?}"
    );
    assert!(third_generations.iter().any(|(_, hashes)| {
        !second_generations
            .iter()
            .any(|(_, prior_hashes)| prior_hashes == hashes)
    }));
    let before_restart = search(&collection, 2).await;
    assert!(before_restart.contains(&100));
    assert!(!before_restart.contains(&2));

    let snapshot_temp = Builder::new()
        .prefix("lmi_snapshot_temp")
        .tempdir()
        .unwrap();
    let snapshot = collection
        .create_snapshot(snapshot_temp.path(), 0)
        .await
        .unwrap();
    let restore_dir = Builder::new().prefix("lmi_restored").tempdir().unwrap();
    Collection::restore_snapshot(
        SnapshotData::new_packed_persistent(snapshots_dir.path().join(snapshot.name)),
        restore_dir.path(),
        0,
        true,
    )
    .unwrap();

    drop(shard);
    collection.stop_gracefully().await;
    drop(collection);
    let reopened = Collection::load(
        "lmi_restarted".to_string(),
        1,
        collection_dir.path(),
        snapshots_dir.path(),
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
    assert_eq!(search(&reopened, 2).await, before_restart);
    assert!(matches!(
        reopened.vectors_config().await,
        VectorsConfig::Single(ref params) if params.lmi_config.is_some()
    ));
    assert!(
        lmi_generations(collection_dir.path())
            .iter()
            .any(|(count, _)| *count >= 95)
    );
    reopened.stop_gracefully().await;

    let restored = Collection::load(
        "lmi_restored".to_string(),
        1,
        restore_dir.path(),
        snapshots_dir.path(),
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
    assert_eq!(search(&restored, 2).await, before_restart);
    let restored_generations = lmi_generations(restore_dir.path());
    assert!(
        third_generations.iter().any(|(count, hashes)| {
            *count >= 95
                && restored_generations
                    .iter()
                    .any(|(restored_count, restored_hashes)| {
                        restored_count == count && restored_hashes == hashes
                    })
        }),
        "snapshot restore changed or lost the trained LMI files"
    );
    restored.stop_gracefully().await;
}
