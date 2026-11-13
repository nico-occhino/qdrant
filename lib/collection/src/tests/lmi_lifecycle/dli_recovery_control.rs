//! Native collection mixed-index lifecycle experiment; synthetic data only.
use std::collections::BTreeMap;

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
        assert_eq!(config.learned_queries.load(Ordering::Relaxed), 0);
        assert_eq!(config.exact_fallbacks.load(Ordering::Relaxed), 1);
        assert_eq!(config.fallback_reasons.lock().get("deleted"), Some(&1));
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
        assert_eq!(config.exact_fallbacks.load(Ordering::Relaxed), 2);
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
            source_config.vector_data.get_mut(segment::data_types::vectors::DEFAULT_VECTOR_NAME).unwrap().datatype = Some(VectorStorageDatatype::Float16);
            let (mut source, _) = segment::segment_constructor::build_segment(experiment.path(), &source_config, None, true).unwrap();
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
            100,
            1,
            &VectorQueryContext::default(),
        )
        .unwrap();
        let old = route
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits)
            .next()
            .unwrap()
            .clone();
        let uuids = (level1.uuid, level2.uuid, plain.uuid);
        let previous_ids: Vec<_> = h.read().iter().map(|(id, _)| id).collect();
        let previous = h.write().remove(&previous_ids);
        let (id1, id2, buffer_id) = {
            let mut holder = h.write();
            (
                holder.add_new(level1),
                holder.add_new(level2),
                holder.add_new(plain),
            )
        };
        let config = Arc::new(
            DliResearchConfig::new(vec![(uuids.0, 1), (uuids.1, 2), (uuids.2, 0)], 100, 1).unwrap(),
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
                    limit: 20,
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
        drop(buffer_handle);
        h.write().remove(&[id1, id2, buffer_id]);
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
    c.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dli_recovery_control() {
    use common::ambient::AmbientFutureExt;
    dli_recovery_control_body()
        .in_ambient(common::ambient::Handoff::Unmeasured(None))
        .await;
}
