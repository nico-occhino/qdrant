//! Native collection mixed-index lifecycle experiment; synthetic data only.
use std::collections::{BTreeMap, HashMap};

use common::generic_consts::Random;
use common::types::DeferredBehavior;
use segment::data_types::vectors::VectorInternal;
use segment::entry::ReadSegmentEntry;
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexEnum;
use segment::types::VectorStorageDatatype;
use segment::vector_storage::VectorStorageRead;
use shard::segment_holder::locked::LockedSegmentHolder;

use super::*;
use crate::operations::types::Datatype;

const N: u64 = 128;
const BASE_ID: u64 = 1_000_003;
const NEW_ID: u64 = 7_000_019;
const KEEP_ID: u64 = 9_000_031;
const DIM: usize = 16;

async fn shard(c: &Collection) -> Arc<crate::shards::replica_set::ShardReplicaSet> {
    c.shards_holder().read().await.get_shard(0).unwrap().clone()
}

async fn wait_dli_publications(
    replica: &crate::shards::replica_set::ShardReplicaSet,
    target: usize,
) {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if replica
                .local_dli_counts_for_test()
                .await
                .is_some_and(|(published, _, _)| published >= target)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("automatic DLI publication did not complete");
}

async fn mutate(c: &Collection, op: PointOperations) {
    let result = shard(c)
        .await
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(op)),
            WaitUntil::Visible,
            None,
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

fn counters(h: &LockedSegmentHolder) -> BTreeMap<String, (u64, u64, u64)> {
    h.read()
        .iter_original()
        .filter_map(|(_, locked)| {
            let s = locked.read();
            let idx = s.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow();
            match &*idx {
                VectorIndexEnum::Lmi(lmi) => {
                    let (learned, fallback) = lmi.search_path_counts_for_test();
                    Some((
                        s.uuid.to_string(),
                        (learned, fallback, lmi.selected_candidate_offsets_for_test()),
                    ))
                }
                _ => None,
            }
        })
        .collect()
}

// Experimental validator: captures every current native holder member while locked.
// Not a production publication or concurrency boundary.
fn test_descriptor(h: &LockedSegmentHolder) -> serde_json::Value {
    let h = h.read();
    let mut members = Vec::new();
    for (_, segment) in h.iter() {
        let s = segment.get().read();
        let cfg = s.config().clone();
        let uuid = s.segment_uuid().to_string();
        let artifacts = ["lmi_state.json", "lmi_router.bin", "lmi_postings.bin"].map(|name| {
            std::fs::read(s.data_path().join("vector_index").join(name))
                .ok()
                .map(|data| Sha256::digest(data).to_vec())
        });
        members.push(serde_json::json!({"physical":uuid,"config":cfg,"artifacts":artifacts}));
    }
    members.sort_by(|a, b| a["physical"].as_str().cmp(&b["physical"].as_str()));
    serde_json::json!({"version":1,"generation":1,"members":members})
}
fn test_descriptor_valid(bytes: &[u8], h: &LockedSegmentHolder) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return false;
    };
    value == test_descriptor(h)
}

async fn query(c: &Collection, axes: &[usize], limit: usize) -> Vec<(u64, f32)> {
    let h = holder(c).await;
    let before = counters(&h);
    assert!(!before.is_empty(), "fixture must retain learned index");
    let found = raw_query(c, axes, limit).await;
    let after = counters(&h);
    for (uuid, counts) in &before {
        let now = after.get(uuid).expect("query changed topology");
        assert!(
            now.0 > counts.0,
            "normal collection search did not execute learned routing"
        );
        assert_eq!(now.1, counts.1, "eligible query fell back to Plain");
        assert!(
            now.2 > counts.2,
            "learned route selected no posting candidates"
        );
        println!(
            "NATIVE_LMI_ROUTE segment={uuid} learned_delta={} fallback_delta=0 selected_offsets_delta={}",
            now.0 - counts.0,
            now.2 - counts.2
        );
    }
    found
}

async fn raw_query(c: &Collection, axes: &[usize], limit: usize) -> Vec<(u64, f32)> {
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
        )
        .await
        .unwrap();
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

async fn open_configured_dli(path: &Path, snapshots: &Path) -> Collection {
    let collection = Collection::load(
        "configured_dli_collection".into(),
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
    collection
        .set_shard_replica_state(0, 1, ReplicaState::Active, None)
        .await
        .unwrap();
    collection
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

async fn dli_recovery_control_body() {
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
        created_at: None,
        #[cfg(feature = "dli-research")]
        dli_experimental: None,
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
    // Experimental descriptor, deliberately restricted to vector-bearing owners.
    // Coverage validation against every active holder member is still required.
    let stale_members: std::collections::BTreeSet<String> =
        before.values().map(|v| v.0.clone()).collect();
    let metadata_path = root.path().join("dli-test-metadata.json");
    std::fs::write(&metadata_path, serde_json::to_vec(&stale_members).unwrap()).unwrap();
    let full_descriptor = serde_json::to_vec(&test_descriptor(&holder(&c).await)).unwrap();
    assert!(test_descriptor_valid(&full_descriptor, &holder(&c).await));
    assert!(!test_descriptor_valid(b"{", &holder(&c).await));
    let mut wrong: serde_json::Value = serde_json::from_slice(&full_descriptor).unwrap();
    wrong["version"] = serde_json::json!(999);
    assert!(!test_descriptor_valid(
        &serde_json::to_vec(&wrong).unwrap(),
        &holder(&c).await
    ));
    let mut missing: serde_json::Value = serde_json::from_slice(&full_descriptor).unwrap();
    missing["members"].as_array_mut().unwrap().pop();
    assert!(!test_descriptor_valid(
        &serde_json::to_vec(&missing).unwrap(),
        &holder(&c).await
    ));
    let mut duplicate: serde_json::Value = serde_json::from_slice(&full_descriptor).unwrap();
    let first = duplicate["members"][0].clone();
    duplicate["members"].as_array_mut().unwrap().push(first);
    assert!(!test_descriptor_valid(
        &serde_json::to_vec(&duplicate).unwrap(),
        &holder(&c).await
    ));
    let mut bad_generation: serde_json::Value = serde_json::from_slice(&full_descriptor).unwrap();
    bad_generation["generation"] = serde_json::json!(2);
    assert!(!test_descriptor_valid(
        &serde_json::to_vec(&bad_generation).unwrap(),
        &holder(&c).await
    ));
    let mut bad_artifact: serde_json::Value = serde_json::from_slice(&full_descriptor).unwrap();
    bad_artifact["members"][0]["artifacts"][0] = serde_json::json!([0, 1]);
    assert!(!test_descriptor_valid(
        &serde_json::to_vec(&bad_artifact).unwrap(),
        &holder(&c).await
    ));
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
    assert!(
        !test_descriptor_valid(&full_descriptor, &holder(&c).await),
        "topology change must invalidate the old descriptor"
    );
    let restored: std::collections::BTreeSet<String> =
        serde_json::from_slice(&std::fs::read(metadata_path).unwrap()).unwrap();
    let recovered_members: std::collections::BTreeSet<String> =
        optimized.values().map(|v| v.0.clone()).collect();
    assert_ne!(
        restored, recovered_members,
        "stale descriptor must be rejected after native reopen"
    );
    println!("DLI_STALE_DESCRIPTOR_REJECTED old={restored:?} recovered={recovered_members:?}");
    // Test-only dispatch: topology is quiescent; no atomicity with search is claimed.
    let fallback = !test_descriptor_valid(&full_descriptor, &holder(&c).await);
    assert!(fallback);
    let mut query_vector = vec![0.0; DIM];
    query_vector[0] = 1.0;
    let exact = c
        .search(
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(query_vector)),
                filter: None,
                params: Some(SearchParams {
                    exact: fallback,
                    ..Default::default()
                }),
                limit: expected.len() + 10,
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
    assert_eq!(exact.len(), expected.len());
    let mut seen = HashSet::new();
    for hit in exact {
        let segment::types::PointIdType::NumId(id) = hit.id else {
            panic!("numeric fixture")
        };
        assert!(seen.insert(id));
        assert_eq!(hit.score, if expected[&id] == 0 { 1.0 } else { 0.0 });
    }
    assert_eq!(seen, expected.keys().copied().collect());
    println!("DLI_NATIVE_EXACT_FALLBACK_ORACLE_PASS count={}", seen.len());
    #[cfg(feature = "dli-research")]
    {
        let mut vector = vec![0.0; DIM];
        vector[0] = 1.0;
        let batch = shard::search::CoreSearchRequestBatch {
            searches: vec![CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(vector)),
                filter: None,
                params: None,
                limit: expected.len() + 10,
                offset: 0,
                with_payload: None,
                with_vector: None,
                score_threshold: None,
            }],
        };
        let ctx = segment::data_types::query_context::QueryContext::new(
            shard::optimizers::config::DEFAULT_INDEXING_THRESHOLD_KB,
            common::ambient::Handoff::Unmeasured(None),
        );
        let result = crate::collection_manager::segments_searcher::SegmentsSearcher::search_dli_exact_fallback(
            holder(&c).await, None, Arc::new(batch),
            &crate::common::adaptive_handle::AdaptiveSearchHandle::current_for_tests(),
            ctx, Duration::from_secs(30),
        ).await.unwrap();
        let mut ids = HashSet::new();
        assert_eq!(
            result.1,
            crate::collection_manager::segments_searcher::DliFallbackReason::MissingHint
        );
        for hit in &result.0[0] {
            let segment::types::PointIdType::NumId(id) = hit.id else {
                panic!("numeric fixture")
            };
            assert!(ids.insert(id));
            assert_eq!(hit.score, if expected[&id] == 0 { 1.0 } else { 0.0 });
        }
        assert_eq!(ids, expected.keys().copied().collect());
        println!("DLI_CAPTURED_EXACT_ORACLE_PASS count={}", ids.len());
        // Use the same routing primitive as the two-level native test on the
        // recovered collection holder. The result remains an experimental
        // per-segment candidate set, not a client-facing collection search.
        use crate::collection_manager::dli_native_route::route_pinned;
        use segment::data_types::query_context::VectorQueryContext;
        let h = holder(&c).await;
        let guard = h.read();
        let pinned: Vec<_> = guard
            .iter_original()
            .map(|(_, locked)| locked.read())
            .collect();
        let members: Vec<_> = pinned
            .iter()
            .map(|s| {
                let index = s.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME]
                    .vector_index
                    .borrow();
                let level = if matches!(&*index, VectorIndexEnum::Lmi(_)) {
                    1
                } else {
                    0
                };
                (level, &**s)
            })
            .collect();
        let mut query = vec![0.0; DIM];
        query[0] = 1.0;
        let result = route_pinned(&members, &query, 2, 2, &VectorQueryContext::default()).unwrap();
        let mut count = 0;
        for (_, hits) in &result.per_segment {
            for point in hits {
                let segment::types::PointIdType::NumId(id) = point.id else {
                    panic!("numeric fixture")
                };
                assert_eq!(point.score, if expected[&id] == 0 { 1.0 } else { 0.0 });
                count += 1;
            }
        }
        assert!(count > 0);
        println!(
            "DLI_ROUTE_RECOVERED_HOLDER budget=2 estimated={} postings={} unique_offsets_plus_buffer_estimate={} returned={}",
            result.plan.estimated_candidates,
            result.posting_candidates,
            result.unique_posting_offsets + result.buffer_point_estimate,
            count
        );
        drop(pinned);
        drop(guard);

        // Exercise the normal collection -> LocalShard -> searcher path with
        // process-local research configuration. A new native member invalidates
        // coverage and must force exact fallback on the same captured handles.
        use crate::collection_manager::segments_searcher::DliResearchConfig;
        use std::sync::atomic::Ordering;
        let roles = {
            let guard = h.read();
            guard
                .iter_original()
                .map(|(_, locked)| {
                    let segment = locked.read();
                    let index = segment.vector_data
                        [segment::data_types::vectors::DEFAULT_VECTOR_NAME]
                        .vector_index
                        .borrow();
                    (
                        segment.uuid,
                        if matches!(&*index, VectorIndexEnum::Lmi(_)) {
                            1
                        } else {
                            0
                        },
                    )
                })
                .collect::<Vec<_>>()
        };
        let config = Arc::new(DliResearchConfig::new(roles, 2, 1000).unwrap());
        let local = shard(&c).await;
        assert!(
            local
                .configure_local_dli_for_test(Some(config.clone()))
                .await
        );
        let search = || async {
            let mut vector = vec![0.0; DIM];
            vector[0] = 1.0;
            c.search(
                CoreSearchRequest {
                    query: QueryEnum::Nearest(NamedQuery::default_dense(vector)),
                    filter: None,
                    params: None,
                    limit: expected.len() + 10,
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
            .unwrap()
        };
        let learned = search().await;
        assert_eq!(config.learned_queries.load(Ordering::Relaxed), 1);
        assert_eq!(config.exact_fallbacks.load(Ordering::Relaxed), 0);
        assert_eq!(learned.len(), expected.len());
        for hit in &learned {
            let segment::types::PointIdType::NumId(id) = hit.id else {
                panic!("numeric fixture")
            };
            assert_eq!(hit.score, if expected[&id] == 0 { 1.0 } else { 0.0 });
        }
        let extra = tempfile::tempdir().unwrap();
        let mut new_buffer =
            segment::segment_constructor::simple_segment_constructor::build_simple_segment(
                extra.path(),
                DIM,
                segment::types::Distance::Cosine,
            )
            .unwrap();
        use segment::entry::SegmentEntry;
        let victim = *expected.iter().find(|(_, axis)| **axis != 0).unwrap().0;
        let mut updated_vector = vec![0.0; DIM];
        updated_vector[0] = 1.0;
        new_buffer
            .upsert_point(
                1000,
                victim.into(),
                segment::data_types::vectors::only_default_vector(&updated_vector),
            )
            .unwrap();
        let new_id = h.write().add_new(new_buffer);
        let fallback = search().await;
        assert_eq!(config.exact_fallbacks.load(Ordering::Relaxed), 1);
        assert_eq!(config.fallback_reasons.lock().get("topology"), Some(&1));
        assert_eq!(fallback.len(), expected.len());
        assert_eq!(
            fallback.iter().map(|p| p.id).collect::<HashSet<_>>(),
            learned.iter().map(|p| p.id).collect()
        );
        for hit in &fallback {
            let segment::types::PointIdType::NumId(id) = hit.id else {
                panic!("numeric fixture")
            };
            assert_eq!(
                hit.score,
                if id == victim || expected[&id] == 0 {
                    1.0
                } else {
                    0.0
                }
            );
            if id == victim {
                assert_eq!(hit.version, 1000);
            }
        }
        h.write().remove(&[new_id]);
        assert!(local.configure_local_dli_for_test(None).await);
        println!(
            "DLI_LOCAL_COLLECTION deleted_fallback=1 topology_fallback=1 oracle={}",
            learned.len()
        );

        // Temporarily install a fresh, manually constructed two-level topology
        // into the real LocalShard. This is not optimizer publication or WAL
        // recovery. Restore the original holder before the test exits.
        use common::flags::FeatureFlags;
        use segment::segment_constructor::segment_builder::SegmentBuilder;
        use segment::types::{HnswGlobalConfig, Indexes, VectorStorageDatatype};
        use std::sync::atomic::AtomicBool;
        let experiment = tempfile::tempdir().unwrap();
        let make_level = |base: u64, seed: u64| {
            let template =
                segment::segment_constructor::simple_segment_constructor::build_simple_segment(
                    experiment.path(),
                    DIM,
                    segment::types::Distance::Cosine,
                )
                .unwrap();
            let mut source_config = template.config().clone();
            source_config
                .vector_data
                .get_mut(segment::data_types::vectors::DEFAULT_VECTOR_NAME)
                .unwrap()
                .datatype = Some(VectorStorageDatatype::Float16);
            let (mut source, _) = segment::segment_constructor::build_segment(
                experiment.path(),
                &source_config,
                None,
                true,
            )
            .unwrap();
            for i in 0..4_u64 {
                let mut vector = vec![0.0; DIM];
                vector[i as usize] = 1.0;
                source
                    .upsert_point(
                        base + i,
                        (base + i).into(),
                        segment::data_types::vectors::only_default_vector(&vector),
                    )
                    .unwrap();
            }
            let mut segment_config = source.config().clone();
            let vector_config = segment_config
                .vector_data
                .get_mut(segment::data_types::vectors::DEFAULT_VECTOR_NAME)
                .unwrap();
            vector_config.datatype = Some(VectorStorageDatatype::Float16);
            vector_config.index = Indexes::LmiTrained(LmiConfig {
                n_buckets: 2,
                sample_size: 4,
                hidden_dim: 4,
                epochs: 1,
                batch_size: 4,
                routing_batch_size: 4,
                kmeans_iterations: 2,
                nprobe: 2,
                seed,
            });
            let staging = tempfile::tempdir().unwrap();
            let mut builder = SegmentBuilder::new(
                staging.path(),
                &segment_config,
                &HnswGlobalConfig::default(),
                FeatureFlags::default(),
            )
            .unwrap();
            builder.update(&[&source], &AtomicBool::new(false)).unwrap();
            builder.build_for_test(experiment.path())
        };
        let level1 = make_level(101, 17);
        let level2 = make_level(201, 29);
        let mut plain =
            segment::segment_constructor::simple_segment_constructor::build_simple_segment(
                experiment.path(),
                DIM,
                segment::types::Distance::Cosine,
            )
            .unwrap();
        let mut plain_vector = vec![0.0; DIM];
        let mut plain_config = plain.config().clone();
        plain_config
            .vector_data
            .get_mut(segment::data_types::vectors::DEFAULT_VECTOR_NAME)
            .unwrap()
            .datatype = Some(VectorStorageDatatype::Float16);
        plain = segment::segment_constructor::build_segment(
            experiment.path(),
            &plain_config,
            None,
            true,
        )
        .unwrap()
        .0;
        plain_vector[1] = 1.0;
        plain
            .upsert_point(
                99,
                99_u64.into(),
                segment::data_types::vectors::only_default_vector(&plain_vector),
            )
            .unwrap();
        let mut axis_query = vec![0.0; DIM];
        axis_query[0] = 1.0;
        let route = route_pinned(
            &[(1, &level1), (2, &level2), (0, &plain)],
            &axis_query,
            2,
            100,
            &VectorQueryContext::default(),
        )
        .unwrap();
        let old = route
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits)
            .find(|hit| hit.id == 101_u64.into())
            .unwrap()
            .clone();
        let uuids = (level1.uuid, level2.uuid, plain.uuid);
        let previous_ids: Vec<_> = h.read().iter().map(|(id, _)| id).collect();
        let previous = h.write().remove(&previous_ids);
        let (_id1, _id2, buffer_id) = {
            let mut holder = h.write();
            (
                holder.add_new(level1),
                holder.add_new(level2),
                holder.add_new(plain),
            )
        };
        let config = Arc::new(
            DliResearchConfig::new(vec![(uuids.0, 1), (uuids.1, 2), (uuids.2, 0)], 2, 100).unwrap(),
        );
        assert!(
            local
                .configure_local_dli_for_test(Some(config.clone()))
                .await
        );
        let fixture_search = || async {
            c.search(
                CoreSearchRequest {
                    query: QueryEnum::Nearest(NamedQuery::default_dense(axis_query.clone())),
                    filter: None,
                    params: None,
                    limit: 40,
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
            .unwrap()
        };
        let routed = fixture_search().await;
        assert_eq!(
            config.learned_queries.load(Ordering::Relaxed),
            1,
            "{:?}",
            config.fallback_reasons.lock()
        );
        assert!(
            routed
                .iter()
                .any(|hit| hit.id == old.id && hit.version == old.version)
        );
        assert!(config.selected_postings.load(Ordering::Relaxed) > 0);
        let new_axis = if old.score > 0.5 { 1 } else { 0 };
        let mut replacement = vec![0.0; DIM];
        replacement[new_axis] = 1.0;
        let buffer_handle = h.read().get(buffer_id).unwrap().clone();
        buffer_handle
            .get()
            .write()
            .upsert_point(
                1000,
                old.id,
                segment::data_types::vectors::only_default_vector(&replacement),
            )
            .unwrap();
        let repaired = fixture_search().await;
        assert_eq!(config.exact_fallbacks.load(Ordering::Relaxed), 1);
        assert_eq!(config.fallback_reasons.lock().get("duplicate"), Some(&1));
        assert_eq!(repaired.len(), 9);
        let hit = repaired.iter().find(|hit| hit.id == old.id).unwrap();
        assert_eq!(hit.version, 1000);
        assert_eq!(hit.score, if new_axis == 0 { 1.0 } else { 0.0 });
        assert_eq!(
            repaired
                .iter()
                .map(|hit| hit.id)
                .collect::<HashSet<_>>()
                .len(),
            9
        );
        assert!(local.configure_local_dli_for_test(None).await);
        use crate::collection_manager::dli_maintenance::DliMaintenance;
        let private_root = tempfile::tempdir().unwrap();
        let initial_roles = HashMap::from([(uuids.0, 1), (uuids.1, 3), (uuids.2, 0)]);
        let maintenance = DliMaintenance::new(
            crate::collection_manager::dli_schedule::HierarchyConfig {
                policy: Default::default(),
                buffer_capacity: 3,
                arity: 2,
            },
            initial_roles.clone(),
            LmiConfig {
                n_buckets: 2,
                sample_size: 16,
                hidden_dim: 4,
                epochs: 1,
                batch_size: 4,
                routing_batch_size: 4,
                kmeans_iterations: 2,
                nprobe: 2,
                seed: 71,
            },
            private_root.path().to_owned(),
        );
        assert!(
            local
                .configure_local_dli_maintenance_for_test(Some(maintenance.clone()))
                .await
        );
        put(&c, vec![point(301, 2)]).await;
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let notified = maintenance.completed.notified();
                if !maintenance.builds.lock().is_empty() {
                    break;
                }
                notified.await;
            }
        })
        .await
        .unwrap();
        assert_eq!(maintenance.scheduled.load(Ordering::Relaxed), 1);
        // A repeated observation of the identical native generation is suppressed.
        maintenance.observe(&h);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let state = maintenance.clone();
                let holder = h.clone();
                scope.spawn(move || {
                    let _scope = common::ambient::test_guard();
                    state.observe(&holder);
                });
            }
        });
        assert_eq!(maintenance.scheduled.load(Ordering::Relaxed), 1);
        let built = maintenance.builds.lock().pop().unwrap();
        assert_eq!(built.plan.target_level, 2);
        assert!(built.sources.contains(&uuids.0));
        assert!(built.sources.contains(&uuids.2));
        assert!(!built.sources.contains(&uuids.1));
        let output = built.output.unwrap();
        let hits = route_pinned(
            &[(2, &output)],
            &axis_query,
            2,
            1000,
            &VectorQueryContext::default(),
        )
        .unwrap();
        assert!(hits.per_segment[0].1.iter().any(|p| p.id == 301_u64.into()));
        let mut private_oracle = std::collections::HashMap::from([
            (101_u64, 1.0_f32),
            (102, 0.0),
            (103, 0.0),
            (104, 0.0),
            (99, 0.0),
            (301, 0.0),
        ]);
        let segment::types::PointIdType::NumId(old_id) = old.id else {
            panic!("numeric fixture")
        };
        private_oracle.insert(old_id, if new_axis == 0 { 1.0 } else { 0.0 });
        assert_eq!(hits.per_segment[0].1.len(), private_oracle.len());
        for hit in &hits.per_segment[0].1 {
            let segment::types::PointIdType::NumId(id) = hit.id else {
                panic!("numeric fixture")
            };
            assert_eq!(hit.score, private_oracle[&id]);
        }
        use segment::entry::StorageSegmentEntry;
        output.flush(true).unwrap();
        let path = output.segment_path.clone();
        let uuid = output.uuid;
        drop(output);
        let reopened = segment::segment_constructor::load_segment(
            &path,
            uuid,
            None,
            &AtomicBool::new(false),
            true,
        )
        .unwrap();
        let reloaded = route_pinned(
            &[(2, &reopened)],
            &axis_query,
            2,
            1000,
            &VectorQueryContext::default(),
        )
        .unwrap();
        assert_eq!(reloaded.per_segment[0].1.len(), private_oracle.len());
        for hit in &reloaded.per_segment[0].1 {
            let segment::types::PointIdType::NumId(id) = hit.id else {
                panic!("numeric fixture")
            };
            assert_eq!(hit.score, private_oracle[&id]);
        }
        assert!(
            !hits.per_segment[0]
                .1
                .iter()
                .any(|p| p.id == 201_u64.into() && p.id != old.id)
        );
        println!(
            "DLI_INSERT_TRIGGER_PRIVATE_BUILD target=2 input_estimate={} live_output={}",
            built.plan.output_points,
            hits.per_segment[0].1.len()
        );
        // An ordinary subsequent insert requests the experimental native
        // optimizer path; this fixture is disposable and manually assembled.
        let active_config = Arc::new(
            DliResearchConfig::new(vec![(uuids.0, 1), (uuids.1, 3), (uuids.2, 0)], 2, 1000)
                .unwrap(),
        );
        maintenance.attach_search(active_config.clone());
        assert!(
            local
                .configure_local_dli_for_test(Some(active_config.clone()))
                .await
        );
        maintenance.native_enabled.store(true, Ordering::Release);
        put(&c, vec![point(302, 3)]).await;
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let notified = maintenance.completed.notified();
                if maintenance.published.load(Ordering::Acquire) > 0
                    || !maintenance.optimizer_errors.lock().is_empty()
                {
                    break;
                }
                notified.await;
            }
        })
        .await
        .unwrap();
        let errors = maintenance.optimizer_errors.lock().clone();
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(maintenance.published.load(Ordering::Relaxed), 1);
        let destination = maintenance.published_uuids.lock()[0];
        let after_native = h
            .read()
            .iter_original()
            .map(|(_, segment)| segment.read().uuid)
            .collect::<HashSet<_>>();
        assert!(!after_native.contains(&uuids.0));
        assert!(!after_native.contains(&uuids.2));
        assert!(after_native.contains(&uuids.1));
        assert!(after_native.contains(&destination));
        // Submit the old scheduler generation to the real optimizer after its
        // selected physical sources were already replaced. The topology hook
        // must reject before temporary COW creation or proxy installation.
        let (stale_ids, source_config, segments_path) = {
            let guard = h.read();
            let ids = guard.iter().map(|(id, _)| id).collect::<Vec<_>>();
            let source = guard.iter_original().next().unwrap().1.read();
            (
                ids,
                source.config().clone(),
                source.data_path().parent().unwrap().to_owned(),
            )
        };
        let stale_staging = tempfile::tempdir_in(private_root.path()).unwrap();
        let stale_plan = crate::collection_manager::dli_schedule::HierarchyConfig {
            policy: Default::default(),
            buffer_capacity: 3,
            arity: 2,
        }
        .plan(3, &[4, 0, 4])
        .unwrap()
        .unwrap();
        let stale_strategy = crate::collection_manager::dli_maintenance::DliOptimizationStrategy {
            staging: stale_staging.path().to_owned(),
            segments_path: segments_path.clone(),
            segment_config: source_config,
            destination_on_disk: true,
            model: maintenance.model,
            expected_sources: HashSet::from([uuids.0, uuids.2]),
            expected_topology: Some((maintenance.hierarchy, initial_roles, stale_plan)),
            validation_hook: None,
        };
        let budget = common::budget::ResourceBudget::new(1, 1);
        let stale_result = shard::optimize::execute_optimization(
            "dli_stale_generation",
            h.clone(),
            stale_ids,
            uuid::Uuid::new_v4(),
            None,
            &shard::optimize::OptimizationPaths {
                segments_path,
                temp_path: stale_staging.path().to_owned(),
            },
            budget.try_acquire(0, 1).unwrap(),
            budget,
            &AtomicBool::new(false),
            common::progress_tracker::ProgressTracker::new_for_test(),
            &segment::common::operation_time_statistics::OperationDurationsAggregator::new(),
            &stale_strategy,
            Box::new(|| {}),
        );
        assert!(stale_result.is_err());
        let after_stale = h
            .read()
            .iter_original()
            .map(|(_, segment)| segment.read().uuid)
            .collect::<HashSet<_>>();
        assert_eq!(after_stale, after_native);
        assert_eq!(h.read().iter().count(), after_native.len());
        println!(
            "DLI_STALE_SCHEDULED_GENERATION_REJECTED active={:?}",
            after_native
        );
        {
            let guard = h.read();
            let (_, output) = guard
                .iter_original()
                .find(|(_, segment)| segment.read().uuid == destination)
                .unwrap();
            let output = output.read();
            let index = output.vector_data[segment::data_types::vectors::DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow();
            let VectorIndexEnum::Lmi(lmi) = &*index else {
                panic!("published segment must contain LMI")
            };
            assert!(lmi.routing_state().is_some());
            assert!(
                guard
                    .iter_original()
                    .any(|(_, segment)| segment.read().is_appendable())
            );
        }
        println!(
            "DLI_AUTOMATIC_NATIVE_PUBLICATION old={:?} destination={} current={:?}",
            uuids, destination, after_native
        );
        let after_search = fixture_search().await;
        assert_eq!(after_search.len(), 11);
        let mut seen = HashSet::new();
        for hit in &after_search {
            let segment::types::PointIdType::NumId(id) = hit.id else {
                panic!("numeric fixture")
            };
            assert!(seen.insert(id));
            let score = if hit.id == old.id {
                if new_axis == 0 { 1.0 } else { 0.0 }
            } else if id == 101 || id == 201 {
                1.0
            } else {
                0.0
            };
            assert_eq!(hit.score, score);
        }
        assert_eq!(maintenance.generation.load(Ordering::Relaxed), 1);
        assert_eq!(active_config.members.read().len(), after_native.len());
        assert_eq!(active_config.members.read().get(&destination), Some(&2));
        assert!(
            active_config.learned_queries.load(Ordering::Relaxed) > 0,
            "{:?}",
            active_config.fallback_reasons.lock()
        );
        let first_learned = active_config.learned_queries.load(Ordering::Relaxed);
        // The optimizer's new appendable COW member is the next insertion
        // buffer. A normal update must trigger a second, independently trained
        // target without rebuilding either existing learned level.
        let first_buffer = *active_config
            .members
            .read()
            .iter()
            .find_map(|(uuid, level)| (*level == 0).then_some(uuid))
            .unwrap();
        put(&c, vec![point(303, 1), point(304, 2), point(305, 3)]).await;
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let notified = maintenance.completed.notified();
                if maintenance.published.load(Ordering::Acquire) >= 2
                    || !maintenance.optimizer_errors.lock().is_empty()
                {
                    break;
                }
                notified.await;
            }
        })
        .await
        .unwrap();
        let errors = maintenance.optimizer_errors.lock().clone();
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(maintenance.published.load(Ordering::Relaxed), 2);
        assert_eq!(maintenance.generation.load(Ordering::Relaxed), 2);
        let second_destination = maintenance.published_uuids.lock()[1];
        let second_members = active_config.members.read().clone();
        assert!(!second_members.contains_key(&first_buffer));
        assert_eq!(second_members.get(&second_destination), Some(&1));
        assert_eq!(second_members.get(&destination), Some(&2));
        assert_eq!(second_members.get(&uuids.1), Some(&3));
        let second_search = fixture_search().await;
        assert_eq!(second_search.len(), 14);
        assert_eq!(
            second_search
                .iter()
                .map(|hit| hit.id)
                .collect::<HashSet<_>>()
                .len(),
            14
        );
        for hit in &second_search {
            let segment::types::PointIdType::NumId(id) = hit.id else {
                panic!("numeric fixture")
            };
            let expected_score = if id == 201 { 1.0 } else { 0.0 };
            assert_eq!(hit.score, expected_score, "id={id}");
        }
        assert!(
            active_config.learned_queries.load(Ordering::Relaxed) > first_learned,
            "{:?}",
            active_config.fallback_reasons.lock()
        );
        println!(
            "DLI_TWO_AUTOMATIC_PUBLICATIONS first={} second={} members={:?} learned={}",
            destination,
            second_destination,
            second_members,
            active_config.learned_queries.load(Ordering::Relaxed)
        );
        let descriptor_path = private_root.path().join("dli_hierarchy.json");
        let valid_descriptor = std::fs::read(&descriptor_path).unwrap();
        let restored = maintenance.load_validated_descriptor(&h).unwrap();
        assert_eq!(restored.generation, 2);
        // Derived metadata cannot override the active native holder. Corrupt
        // metadata disables the learned map and selects exact fallback.
        std::fs::write(&descriptor_path, b"{corrupt").unwrap();
        assert!(maintenance.load_validated_descriptor(&h).is_err());
        assert!(active_config.members.read().is_empty());
        let fallbacks_before = active_config.exact_fallbacks.load(Ordering::Relaxed);
        assert_eq!(fixture_search().await.len(), 14);
        assert!(active_config.exact_fallbacks.load(Ordering::Relaxed) > fallbacks_before);
        std::fs::write(&descriptor_path, valid_descriptor).unwrap();
        maintenance.load_validated_descriptor(&h).unwrap();
        let learned_before = active_config.learned_queries.load(Ordering::Relaxed);
        assert_eq!(fixture_search().await.len(), 14);
        assert!(active_config.learned_queries.load(Ordering::Relaxed) > learned_before);
        // Cancel a third job after an ordinary insert batch has scheduled it,
        // but before the native optimizer protects its sources. The buffer
        // remains authoritative, and a later ordinary insert may retry.
        use crate::collection_manager::dli_maintenance::DliBeforeNativeHook;
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        *maintenance.before_native.lock() = Some(Arc::new(DliBeforeNativeHook {
            entered: entered_tx,
            resume: parking_lot::Mutex::new(resume_rx),
        }));
        put(&c, vec![point(306, 1), point(307, 2), point(308, 3)]).await;
        entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        maintenance.request_cancel();
        resume_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(30), maintenance.cancel_and_wait())
            .await
            .unwrap();
        assert_eq!(maintenance.published.load(Ordering::Relaxed), 2);
        let cancellation_errors = maintenance.optimizer_errors.lock().clone();
        assert!(
            cancellation_errors
                .iter()
                .any(|error| error.to_ascii_lowercase().contains("cancel")),
            "{cancellation_errors:?}"
        );
        assert_eq!(fixture_search().await.len(), 17);
        *maintenance.before_native.lock() = None;
        maintenance.optimizer_errors.lock().clear();
        maintenance.resume_after_cancel();
        put(&c, vec![point(309, 0)]).await;
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let notified = maintenance.completed.notified();
                if maintenance.published.load(Ordering::Acquire) >= 3
                    || !maintenance.optimizer_errors.lock().is_empty()
                {
                    break;
                }
                notified.await;
            }
        })
        .await
        .unwrap();
        let retry_errors = maintenance.optimizer_errors.lock().clone();
        assert!(retry_errors.is_empty(), "{retry_errors:?}");
        assert_eq!(maintenance.published.load(Ordering::Relaxed), 3);
        let learned_before_retry = active_config.learned_queries.load(Ordering::Relaxed);
        let retried = fixture_search().await;
        assert_eq!(retried.len(), 18);
        assert_eq!(
            retried
                .iter()
                .map(|hit| hit.id)
                .collect::<HashSet<_>>()
                .len(),
            18
        );
        for hit in &retried {
            let segment::types::PointIdType::NumId(id) = hit.id else {
                panic!("numeric fixture")
            };
            assert_eq!(
                hit.score,
                if id == 201 || id == 309 { 1.0 } else { 0.0 },
                "id={id}"
            );
        }
        assert!(
            active_config.learned_queries.load(Ordering::Relaxed) > learned_before_retry,
            "{:?}",
            active_config.fallback_reasons.lock()
        );
        // Reopen the active physical generations into an independent holder.
        // This checks derived metadata against loaded segments, without
        // claiming a full Collection::load or WAL recovery.
        let reloaded_holder = shard::segment_holder::locked::LockedSegmentHolder::new(
            shard::segment_holder::SegmentHolder::default(),
        );
        for (_, locked) in h.read().iter_original() {
            let segment = locked.read();
            segment.flush(true).unwrap();
            let reopened = segment::segment_constructor::load_segment(
                &segment.segment_path,
                segment.uuid,
                None,
                &AtomicBool::new(false),
                true,
            )
            .unwrap();
            reloaded_holder.write().add_new(reopened);
        }
        assert_eq!(
            maintenance
                .load_validated_descriptor(&reloaded_holder)
                .unwrap()
                .generation,
            3
        );
        // Hold the real optimizer after it validated the scheduler plan but
        // before it creates proxies. A native upsert must finish in this gap;
        // after resuming, proxy/COW propagation and the final holder must keep
        // the newer vector visible.
        let (validated_tx, validated_rx) = std::sync::mpsc::channel();
        let (continue_tx, continue_rx) = std::sync::mpsc::channel();
        *maintenance.after_validation.lock() = Some(Arc::new(DliBeforeNativeHook {
            entered: validated_tx,
            resume: parking_lot::Mutex::new(continue_rx),
        }));
        put(&c, vec![point(310, 0), point(311, 2), point(312, 3)]).await;
        validated_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let before_write = fixture_search().await;
        assert_eq!(
            before_write
                .iter()
                .find(|hit| hit.id == 310_u64.into())
                .unwrap()
                .score,
            1.0
        );
        tokio::time::timeout(Duration::from_secs(10), put(&c, vec![point(310, 2)]))
            .await
            .expect("update after plan validation must complete before proxy installation");
        let after_write = fixture_search().await;
        assert_eq!(
            after_write
                .iter()
                .find(|hit| hit.id == 310_u64.into())
                .unwrap()
                .score,
            0.0
        );
        continue_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let notified = maintenance.completed.notified();
                if maintenance.published.load(Ordering::Acquire) >= 4
                    || !maintenance.optimizer_errors.lock().is_empty()
                {
                    break;
                }
                notified.await;
            }
        })
        .await
        .unwrap();
        let interval_errors = maintenance.optimizer_errors.lock().clone();
        assert!(interval_errors.is_empty(), "{interval_errors:?}");
        *maintenance.after_validation.lock() = None;
        let interval_results = fixture_search().await;
        assert_eq!(interval_results.len(), 21);
        assert_eq!(
            interval_results
                .iter()
                .map(|hit| hit.id)
                .collect::<HashSet<_>>()
                .len(),
            21
        );
        let updated = interval_results
            .iter()
            .find(|hit| hit.id == 310_u64.into())
            .unwrap();
        assert_eq!(updated.score, 0.0);
        assert!(local.configure_local_dli_for_test(None).await);
        assert!(local.configure_local_dli_maintenance_for_test(None).await);
        drop(buffer_handle);
        let current_ids: Vec<_> = h.read().iter().map(|(id, _)| id).collect();
        h.write().remove(&current_ids);
        for segment in previous {
            h.write().add_new_locked(segment);
        }
        println!("DLI_TWO_LEVEL_LOCAL_COLLECTION learned=1 duplicate_fallback=1 live=9");
    }
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

    // A separate opt-in collection proves startup activation from persisted
    // experimental collection config; no research helper installs the search
    // dispatcher or maintenance observer.
    let dli_path = root.path().join("configured_dli_collection");
    let dli_snapshots = root.path().join("configured_dli_snapshots");
    std::fs::create_dir(&dli_path).unwrap();
    std::fs::create_dir(&dli_snapshots).unwrap();
    let mut dli_cfg = cfg.clone();
    if let crate::operations::types::VectorsConfig::Single(params) = &mut dli_cfg.params.vectors {
        params.lmi_config = None;
    }
    dli_cfg.optimizer_config.indexing_threshold = None;
    dli_cfg.dli_experimental = Some(crate::config::DliExperimentalConfig {
        maintenance_policy: Default::default(),
        buffer_capacity: 3,
        arity: 2,
        candidate_budget: 64,
        model: LmiConfig {
            n_buckets: 2,
            sample_size: 16,
            hidden_dim: 8,
            epochs: 1,
            batch_size: 8,
            routing_batch_size: 8,
            kmeans_iterations: 2,
            nprobe: 2,
            seed: 73,
        },
    });
    let configured_dli = Collection::new(
        "configured_dli_collection".into(),
        1,
        &dli_path,
        &dli_snapshots,
        &dli_cfg,
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
    configured_dli
        .set_shard_replica_state(0, 1, ReplicaState::Active, None)
        .await
        .unwrap();
    assert!(
        shard(&configured_dli)
            .await
            .local_dli_runtime_ready_for_test()
            .await,
        "configured LocalShard must install learned search and maintenance automatically"
    );
    assert!(dli_path.join("0").join("dli_hierarchy.json").exists());
    let dli_replica = shard(&configured_dli).await;
    let mut dli_expected = BTreeMap::new();
    for batch in 0..4 {
        let first_id = 90_000 + batch * 3;
        let new_points = (0..3)
            .map(|offset| point(first_id + offset, (batch * 3 + offset) as usize))
            .collect::<Vec<_>>();
        for offset in 0..3 {
            dli_expected.insert(first_id + offset, ((batch * 3 + offset) as usize) % DIM);
        }
        put(&configured_dli, new_points).await;
        wait_dli_publications(&dli_replica, (batch + 1) as usize).await;

        let mut query_vector = vec![0.0; DIM];
        query_vector[0] = 1.0;
        let hits = configured_dli
            .search(
                CoreSearchRequest {
                    query: QueryEnum::Nearest(NamedQuery::default_dense(query_vector)),
                    filter: None,
                    params: None,
                    limit: 16,
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
        assert_eq!(hits.len(), dli_expected.len());
        let ids: HashSet<_> = hits.iter().map(|hit| hit.id).collect();
        assert_eq!(ids.len(), dli_expected.len());
        for hit in hits {
            let segment::types::PointIdType::NumId(id) = hit.id else {
                panic!("numeric DLI fixture")
            };
            assert_eq!(hit.score, if dli_expected[&id] == 0 { 1.0 } else { 0.0 });
        }
        let (_, learned, fallbacks) = dli_replica.local_dli_counts_for_test().await.unwrap();
        assert!(
            learned > 0,
            "normal search should use newly trained DLI levels"
        );
        println!(
            "DLI_CONFIGURED_RUNTIME transition={} learned_queries={learned} exact_fallbacks={fallbacks} live={}",
            batch + 1,
            dli_expected.len()
        );
    }
    let published_descriptor: serde_json::Value =
        common::fs::read_json(&dli_path.join("0").join("dli_hierarchy.json")).unwrap();
    let published_levels: HashSet<usize> = published_descriptor["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|member| member[1].as_u64().unwrap() as usize)
        .collect();
    assert!(
        published_levels.contains(&0),
        "active Plain buffer must be covered"
    );
    assert!(published_levels.contains(&1), "level 1 must be active");
    assert!(published_levels.contains(&2), "level 2 must be active");
    println!("DLI_CONFIGURED_ACTIVE_LEVELS {published_levels:?}");
    let publication_events = dli_replica
        .local_dli_publication_events_for_test()
        .await
        .unwrap();
    assert_eq!(publication_events.len(), 4);
    assert_eq!(
        publication_events
            .iter()
            .map(|event| event.published_output_points)
            .collect::<Vec<_>>(),
        vec![3, 6, 9, 3]
    );
    println!("DLI_CONFIGURED_PUBLICATION_EVENTS {publication_events:?}");
    shard(&configured_dli)
        .await
        .force_flush_local_for_test()
        .await;
    configured_dli.stop_gracefully().await;
    drop(configured_dli);

    // Full collection restart: native segment recovery runs first, then the
    // configured DLI descriptor is validated against the recovered holder.
    let reopened_dli = open_configured_dli(&dli_path, &dli_snapshots).await;
    let reopened_replica = shard(&reopened_dli).await;
    assert!(reopened_replica.local_dli_runtime_ready_for_test().await);
    let recovered_hits = raw_query(&reopened_dli, &[0], 16).await;
    assert_eq!(recovered_hits.len(), dli_expected.len());
    assert_eq!(
        recovered_hits
            .iter()
            .map(|hit| hit.0)
            .collect::<HashSet<_>>(),
        dli_expected.keys().copied().collect()
    );
    for (id, score) in &recovered_hits {
        assert_eq!(*score, if dli_expected[id] == 0 { 1.0 } else { 0.0 });
    }
    let (_, recovered_learned, recovered_fallbacks) =
        reopened_replica.local_dli_counts_for_test().await.unwrap();
    assert!(
        recovered_learned > 0,
        "reopened descriptor must route learned queries"
    );
    println!(
        "DLI_CONFIGURED_CLEAN_REOPEN learned_queries={recovered_learned} exact_fallbacks={recovered_fallbacks} hits={}",
        recovered_hits.len(),
    );
    shard(&reopened_dli)
        .await
        .force_flush_local_for_test()
        .await;
    reopened_dli.stop_gracefully().await;
    drop(reopened_dli);

    // Invalid derived state must fail closed while native recovered vectors
    // remain searchable, and must not prevent a later safe reconstruction.
    std::fs::write(dli_path.join("0").join("dli_hierarchy.json"), b"{").unwrap();
    let invalid_dli = open_configured_dli(&dli_path, &dli_snapshots).await;
    let invalid_replica = shard(&invalid_dli).await;
    assert!(!invalid_replica.local_dli_runtime_ready_for_test().await);
    let before_fallback = invalid_replica
        .local_dli_search_counts_for_test()
        .await
        .unwrap();
    let mut query_vector = vec![0.0; DIM];
    query_vector[0] = 1.0;
    let fallback_hits = invalid_dli
        .search(
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(query_vector)),
                filter: None,
                params: None,
                limit: 16,
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
    assert_eq!(fallback_hits.len(), dli_expected.len());
    let fallback_ids: HashSet<_> = fallback_hits.iter().map(|hit| hit.id).collect();
    assert_eq!(fallback_ids.len(), dli_expected.len());
    for hit in fallback_hits {
        let segment::types::PointIdType::NumId(id) = hit.id else {
            panic!("numeric DLI fallback fixture")
        };
        assert_eq!(hit.score, if dli_expected[&id] == 0 { 1.0 } else { 0.0 });
    }
    let after_fallback = invalid_replica
        .local_dli_search_counts_for_test()
        .await
        .unwrap();
    assert!(after_fallback.1 > before_fallback.1);
    println!(
        "DLI_CONFIGURED_CORRUPT_DESCRIPTOR_EXACT_FALLBACK ids={}",
        dli_expected.len()
    );

    // A later ordinary update triggers a full native rebuild from the
    // authoritative recovered holder. Search remains exact until publication;
    // the successful optimizer transition writes a fresh descriptor and
    // reactivates learned routing.
    let rebuild_point = point(91_000, 9);
    dli_expected.insert(91_000, 9);
    put(&invalid_dli, vec![rebuild_point]).await;
    wait_dli_publications(&invalid_replica, 1).await;
    assert!(invalid_replica.local_dli_runtime_ready_for_test().await);
    let rebuilt_hits = raw_query(&invalid_dli, &[0], 16).await;
    assert_eq!(rebuilt_hits.len(), dli_expected.len());
    assert_eq!(
        rebuilt_hits.iter().map(|hit| hit.0).collect::<HashSet<_>>(),
        dli_expected.keys().copied().collect()
    );
    for (id, score) in &rebuilt_hits {
        assert_eq!(*score, if dli_expected[id] == 0 { 1.0 } else { 0.0 });
    }
    let (_, rebuilt_learned, rebuilt_fallbacks) =
        invalid_replica.local_dli_counts_for_test().await.unwrap();
    assert!(rebuilt_learned > 0);
    println!(
        "DLI_CONFIGURED_RECONSTRUCTED learned_queries={rebuilt_learned} exact_fallbacks={rebuilt_fallbacks} hits={}",
        rebuilt_hits.len()
    );
    invalid_dli.stop_gracefully().await;
    c.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dli_recovery_control() {
    use common::ambient::AmbientFutureExt;
    dli_recovery_control_body()
        .in_ambient(common::ambient::Handoff::Unmeasured(None))
        .await;
}
