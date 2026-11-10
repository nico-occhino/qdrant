//! Native collection mixed-index lifecycle experiment; synthetic data only.
use super::*;
use crate::operations::types::Datatype;
use common::generic_consts::Random;
use common::types::DeferredBehavior;
use segment::data_types::vectors::VectorInternal;
use segment::entry::ReadSegmentEntry;
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexEnum;
use segment::types::VectorStorageDatatype;
use segment::vector_storage::VectorStorageRead;
use shard::segment_holder::locked::LockedSegmentHolder;
use std::collections::BTreeMap;

const N: u64 = 10_000;
const BASE_ID: u64 = 1_000_003;
const NEW_ID: u64 = 7_000_019;
const KEEP_ID: u64 = 9_000_031;
const DIM: usize = 16;

async fn shard(c: &Collection) -> Arc<crate::shards::replica_set::ShardReplicaSet> {
    c.shards_holder().read().await.get_shard(0).unwrap().clone()
}

async fn mutate(c: &Collection, op: PointOperations) {
    let result = shard(c)
        .await
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(op)),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .expect("normal collection mutation must apply")
        .expect("active local replica must apply operation");
    assert_eq!(
        result.status,
        crate::operations::types::UpdateStatus::Completed
    );
    println!("NATIVE_LMI_APPLIED {result:?}");
}

async fn put(c: &Collection, points: Vec<PointStructPersisted>) {
    mutate(
        c,
        PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(points)),
    )
    .await;
}

async fn holder(c: &Collection) -> LockedSegmentHolder {
    shard(c).await.local_segments_for_test().await.unwrap()
}

// Physical ownership uses external IDs; postings are never treated as IDs.
fn ownership(h: &LockedSegmentHolder) -> BTreeMap<u64, (String, u32, &'static str)> {
    let mut out = BTreeMap::new();
    let h = h.read();
    assert_eq!(
        h.iter().count(),
        h.iter_original().count(),
        "unexpected active optimizer proxy"
    );
    for (_, locked) in h.iter_original() {
        let s = locked.read();
        let tracker = s.id_tracker.borrow();
        let data = &s.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME];
        let idx = data.vector_index.borrow();
        let kind = match &*idx {
            VectorIndexEnum::Lmi(lmi) => {
                assert!(lmi.routing_state().is_some());
                assert!(!s.is_appendable());
                "lmi"
            }
            VectorIndexEnum::Plain(_) => {
                assert!(s.is_appendable());
                "plain"
            }
            _ => panic!("unexpected index variant"),
        };
        let storage = data.vector_storage.borrow();
        assert_eq!(storage.datatype(), VectorStorageDatatype::Float16);
        for (id, off) in tracker.point_mappings().iter_from(None) {
            let segment::types::PointIdType::NumId(id) = id else {
                panic!("numeric fixture")
            };
            if !s.has_point(id.into(), DeferredBehavior::VisibleOnly) {
                continue;
            }
            assert_eq!(tracker.external_id(off), Some(id.into()));
            assert!(!storage.is_deleted_vector(off));
            assert!(
                out.insert(id, (s.uuid.to_string(), off, kind)).is_none(),
                "duplicate visible external ID {id}"
            );
        }
    }
    out
}

fn counters(h: &LockedSegmentHolder) -> BTreeMap<String, (u64, u64)> {
    h.read()
        .iter_original()
        .filter_map(|(_, locked)| {
            let s = locked.read();
            let idx = s.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow();
            match &*idx {
                VectorIndexEnum::Lmi(lmi) => {
                    Some((s.uuid.to_string(), lmi.search_path_counts_for_test()))
                }
                _ => None,
            }
        })
        .collect()
}

async fn query(c: &Collection, axes: &[usize], limit: usize) -> Vec<(u64, f32)> {
    let h = holder(c).await;
    let before = counters(&h);
    assert!(!before.is_empty(), "fixture must retain learned index");
    let mut q = vec![0.0; DIM];
    for &a in axes {
        q[a] = 1.0;
    }
    let found = c
        .search(
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(q)),
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
            HwMeasurementAcc::new(),
        )
        .await
        .unwrap();
    let after = counters(&h);
    for (uuid, counts) in &before {
        let now = after.get(uuid).expect("query changed topology");
        assert!(
            now.0 > counts.0,
            "normal collection search did not execute learned routing"
        );
        assert_eq!(now.1, counts.1, "eligible query fell back to Plain");
        println!(
            "NATIVE_LMI_ROUTE segment={uuid} learned_delta={} fallback_delta=0",
            now.0 - counts.0
        );
    }
    let hits: Vec<_> = found
        .into_iter()
        .map(|p| {
            let segment::types::PointIdType::NumId(id) = p.id else {
                panic!("numeric fixture")
            };
            assert!(p.score.is_finite());
            (id, p.score)
        })
        .collect();
    assert_eq!(
        hits.iter().map(|p| p.0).collect::<HashSet<_>>().len(),
        hits.len()
    );
    assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
    hits
}

// All fixture vectors are unit coordinate axes. A single-axis query therefore
// has exact native Float16 scores 0 or 1, without reduction/tolerance ambiguity.
async fn exact_axis_oracle(c: &Collection, expected: &BTreeMap<u64, usize>, axis: usize) {
    let hits = query(c, &[axis], 10).await;
    assert_eq!(hits.len(), 10.min(expected.len()));
    let h = holder(c).await;
    let h = h.read();
    let mut actual = BTreeMap::new();
    for (_, locked) in h.iter_original() {
        let segment = locked.read();
        let tracker = segment.id_tracker.borrow();
        let storage = segment.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow();
        for (id, offset) in tracker.point_mappings().iter_from(None) {
            if !segment.has_point(id, DeferredBehavior::VisibleOnly) {
                continue;
            }
            let segment::types::PointIdType::NumId(id) = id else {
                panic!("numeric fixture")
            };
            let VectorInternal::Dense(vector) =
                storage.get_vector_opt::<Random>(offset).unwrap().to_owned()
            else {
                panic!("dense fixture")
            };
            assert_eq!(vector.len(), DIM);
            for (i, &value) in vector.iter().enumerate() {
                assert_eq!(value, if expected[&id] == i { 1.0 } else { 0.0 });
            }
            assert!(actual.insert(id, vector[axis]).is_none());
        }
    }
    assert_eq!(actual.len(), expected.len());
    let mut oracle: Vec<f32> = actual.values().copied().collect();
    oracle.sort_by(|a, b| b.total_cmp(a));
    for ((id, score), reference) in hits.iter().zip(&oracle) {
        assert!(expected.contains_key(id), "unexpected or deleted ID");
        assert_eq!(*score, if expected[id] == axis { 1.0 } else { 0.0 });
        assert_eq!(
            *score, *reference,
            "incorrect score rank; equal-score IDs may permute"
        );
    }
}

async fn open(path: &Path, snapshots: &Path, optimize: bool) -> Collection {
    let c = Collection::load(
        "native_segment_delta".into(),
        1,
        path,
        snapshots,
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
    // Update the real persisted collection configuration, then synchronously
    // recreate native workers. Load-time optimizer overrides do not change
    // the shared config read by LocalShard::new for worker thread limits.
    c.update_optimizer_params_from_diff(
        serde_json::from_value(
            serde_json::json!({"max_optimization_threads":if optimize {1} else {0}}),
        )
        .unwrap(),
    )
    .await
    .unwrap();
    shard(&c).await.on_optimizer_config_update().await.unwrap();
    c
}

async fn wait_indexed(c: &Collection, expected: usize) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    loop {
        let info = c.info(&ShardSelectorInternal::All).await.unwrap();
        assert!(
            !matches!(
                info.optimizer_status,
                crate::operations::types::OptimizersStatus::Error(_)
            ),
            "optimizer error: {:?}",
            info.optimizer_status
        );
        let h = holder(c).await;
        let ready = {
            let h = h.read();
            h.iter().count() == h.iter_original().count()
                && h.iter_original().all(|(_, locked)| {
                    let s = locked.read();
                    if s.id_tracker.borrow().available_point_count() == 0 {
                        return true;
                    }
                    matches!(
                        &*s.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME]
                            .vector_index
                            .borrow(),
                        VectorIndexEnum::Lmi(_)
                    )
                })
        };
        if ready && info.indexed_vectors_count.is_some_and(|n| n >= expected) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "optimizer timeout: {info:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_lmi_mixed_segments_mutation_restart_and_optimizer() {
    let root = Builder::new()
        .prefix("native_lmi_delta_")
        .tempdir()
        .unwrap();
    let path = root.path().join("collection");
    let snapshots = root.path().join("snapshots");
    std::fs::create_dir(&path).unwrap();
    std::fs::create_dir(&snapshots).unwrap();
    let mut vp = VectorParamsBuilder::new(DIM as u64, Distance::Cosine).build();
    vp.datatype = Some(Datatype::Float16);
    vp.lmi_config = Some(LmiConfig {
        n_buckets: 2,
        sample_size: 256,
        hidden_dim: 8,
        epochs: 1,
        batch_size: 64,
        routing_batch_size: 64,
        kmeans_iterations: 2,
        nprobe: 2,
        seed: 7,
    });
    let mut opt = OptimizersConfig::fixture();
    opt.indexing_threshold = Some(1);
    opt.max_optimization_threads = Some(0);
    opt.default_segment_number = 1;
    opt.max_segment_size = Some(1_000_000);
    let cfg = CollectionConfigInternal {
        params: CollectionParams {
            vectors: VectorsConfig::Single(vp),
            shard_number: NonZeroU32::new(1).unwrap(),
            ..CollectionParams::empty()
        },
        optimizer_config: opt,
        wal_config: Default::default(),
        hnsw_config: Default::default(),
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
    };
    let c = Collection::new(
        "native_segment_delta".into(),
        1,
        &path,
        &snapshots,
        &cfg,
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
    c.set_shard_replica_state(0, 1, ReplicaState::Active, None)
        .await
        .unwrap();
    let mut expected: BTreeMap<u64, usize> =
        (0..N).map(|i| (10_000 + i * 3, (i % 8) as usize)).collect();
    expected.insert(BASE_ID, 15);
    put(
        &c,
        expected
            .iter()
            .map(|(&id, &axis)| point(id, axis))
            .collect(),
    )
    .await;
    shard(&c).await.force_flush_local_for_test().await;
    c.stop_gracefully().await;
    drop(c);
    let c = open(&path, &snapshots, true).await;
    c.trigger_optimizers().await;
    wait_indexed(&c, expected.len()).await;
    shard(&c).await.force_flush_local_for_test().await;
    c.stop_gracefully().await;
    drop(c);
    // Pause by normal load override so topology cannot race with assertions.
    let c = open(&path, &snapshots, false).await;
    let before = ownership(&holder(&c).await);
    assert_eq!(before.len(), expected.len());
    assert_eq!(before[&BASE_ID].2, "lmi");
    let base_owner = before[&BASE_ID].clone();
    let immutable_base_files = lmi_generations(&path);
    println!("NATIVE_LMI_STAGE static_ready points={}", expected.len());
    exact_axis_oracle(&c, &expected, 15).await;
    // A+B: normal insertion must land in a different appendable Plain segment.
    put(&c, vec![point(NEW_ID, 14), point(KEEP_ID, 12)]).await;
    expected.insert(NEW_ID, 14);
    expected.insert(KEEP_ID, 12);
    let mixed = ownership(&holder(&c).await);
    assert_eq!(mixed[&NEW_ID].2, "plain");
    assert_ne!(mixed[&NEW_ID].0, base_owner.0);
    let pair = query(&c, &[14, 15], 2).await;
    assert_eq!(
        pair.iter().map(|p| p.0).collect::<HashSet<_>>(),
        HashSet::from([BASE_ID, NEW_ID])
    );
    println!("NATIVE_LMI_STAGE mixed_search passed");
    // C: upsert existing static ID must move, not update trained postings in place.
    put(&c, vec![point(BASE_ID, 13)]).await;
    expected.insert(BASE_ID, 13);
    let moved = ownership(&holder(&c).await);
    assert_eq!(moved.len(), expected.len());
    assert_eq!(moved[&BASE_ID].2, "plain");
    assert_ne!(moved[&BASE_ID].0, base_owner.0);
    let pair = query(&c, &[13, 14], 2).await;
    assert_eq!(
        pair.iter().map(|p| p.0).collect::<HashSet<_>>(),
        HashSet::from([BASE_ID, NEW_ID])
    );
    println!(
        "NATIVE_LMI_STAGE upsert old={base_owner:?} new={:?}",
        moved[&BASE_ID]
    );
    // D: delete a point still in LMI and one in mutable Plain.
    let deleted_static = 10_000;
    assert_eq!(moved[&deleted_static].2, "lmi");
    mutate(
        &c,
        PointOperations::DeletePoints {
            ids: vec![deleted_static.into(), NEW_ID.into()],
        },
    )
    .await;
    expected.remove(&deleted_static);
    expected.remove(&NEW_ID);
    assert_eq!(
        ownership(&holder(&c).await)
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        expected.keys().copied().collect::<Vec<_>>()
    );
    exact_axis_oracle(&c, &expected, 13).await;
    let all = query(&c, &[0], expected.len() + 10).await;
    assert_eq!(
        all.iter().map(|p| p.0).collect::<HashSet<_>>(),
        expected.keys().copied().collect()
    );
    println!("NATIVE_LMI_STAGE deletion passed points={}", expected.len());
    // E: persist/reopen through native collection load, retain mixed ownership.
    shard(&c).await.force_flush_local_for_test().await;
    let pre_restart = ownership(&holder(&c).await);
    assert_eq!(
        lmi_generations(&path),
        immutable_base_files,
        "mutations changed the static learned artifacts"
    );
    c.stop_gracefully().await;
    drop(c);
    let c = open(&path, &snapshots, false).await;
    assert_eq!(ownership(&holder(&c).await), pre_restart);
    exact_axis_oracle(&c, &expected, 13).await;
    println!("NATIVE_LMI_STAGE restart passed");
    // The normal 1 KiB indexing threshold intentionally leaves tiny buffers
    // Plain. Add a bounded batch to make an optimizer transition eligible.
    let batch: Vec<_> = (0..64u64)
        .map(|i| {
            let id = 20_000_000 + i * 7;
            let axis = (i % 8) as usize;
            expected.insert(id, axis);
            point(id, axis)
        })
        .collect();
    put(&c, batch).await;
    c.stop_gracefully().await;
    drop(c);
    // F: optimizer replaces/merges native segments; offsets are regenerated.
    let c = open(&path, &snapshots, true).await;
    c.trigger_optimizers().await;
    wait_indexed(&c, expected.len()).await;
    shard(&c).await.force_flush_local_for_test().await;
    c.stop_gracefully().await;
    drop(c);
    let c = open(&path, &snapshots, false).await;
    let optimized = ownership(&holder(&c).await);
    assert_eq!(
        optimized.keys().copied().collect::<Vec<_>>(),
        expected.keys().copied().collect::<Vec<_>>()
    );
    assert!(optimized.values().all(|v| v.2 == "lmi"));
    assert_ne!(optimized[&BASE_ID].0, pre_restart[&BASE_ID].0);
    println!(
        "NATIVE_LMI_REMAP id={BASE_ID} before={:?} after={:?}",
        pre_restart[&BASE_ID], optimized[&BASE_ID]
    );
    exact_axis_oracle(&c, &expected, 13).await;
    let all = query(&c, &[0], expected.len() + 10).await;
    assert_eq!(
        all.iter().map(|p| p.0).collect::<HashSet<_>>(),
        expected.keys().copied().collect()
    );
    println!(
        "NATIVE_LMI_STAGE optimizer passed points={}",
        expected.len()
    );
    c.stop_gracefully().await;
}
