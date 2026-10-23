//! Disposable native Float16 lifecycle diagnostics. Never reads SISAP inputs.
use super::*;
use segment::data_types::named_vectors::NamedVectors;
use segment::data_types::vectors::NamedQuery;
use segment::data_types::vectors::VectorInternal;
use segment::entry::{SegmentEntry, StorageSegmentEntry};
use segment::segment::Segment;
use segment::segment_constructor::build_segment;
use segment::types::{SegmentConfig, VectorDataConfig, VectorStorageType};
use shard::query::query_enum::QueryEnum;
use shard::segment_holder::FlushMode;
use shard::segment_holder::SegmentHolder;

use crate::operations::types::CoreSearchRequest;
use crate::shards::replica_set::ShardReplicaSet;

const SMALL_N: usize = 64;
const SMALL_DEADLINE: Duration = Duration::from_secs(120);

fn small_scratch() -> PathBuf {
    let allocation = PathBuf::from(std::env::var("SCRATCHDIR").unwrap())
        .canonicalize()
        .unwrap();
    let scratch = PathBuf::from(std::env::var("SISAP23_SMALL_SCRATCH").unwrap())
        .canonicalize()
        .unwrap();
    assert!(scratch.starts_with(&allocation) && scratch != allocation);
    scratch
}

fn physical_dirs(collection_path: &Path) -> Vec<String> {
    let mut names: Vec<_> = fs::read_dir(collection_path.join("0/segments"))
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            let name = path.file_name()?.to_str()?.to_owned();
            (path.is_dir() && Uuid::parse_str(&name).is_ok()).then_some(name)
        })
        .collect();
    names.sort();
    names
}

fn holder_rows(holder: &SegmentHolder) -> Vec<serde_json::Value> {
    let mut rows: Vec<_> = holder
        .iter_original()
        .map(|(holder_id, locked)| {
            let segment = locked.read();
            let tracker = segment.id_tracker.borrow();
            let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
            let index = data.vector_index.borrow();
            serde_json::json!({
                "holder_id": holder_id,
                "uuid": segment.uuid.to_string(),
                "live": tracker.available_point_count(),
                "deleted": tracker.deleted_point_count(),
                "total": tracker.total_point_count(),
                "index_kind": index_kind(&index),
                "indexed_vectors": index.indexed_vector_count(),
            })
        })
        .collect();
    rows.sort_by(|a, b| a["uuid"].as_str().cmp(&b["uuid"].as_str()));
    rows
}

fn lmi_search_counts(
    holder: &shard::segment_holder::locked::LockedSegmentHolder,
) -> Vec<(String, (u64, u64))> {
    let holder = holder.read();
    let mut counts: Vec<_> = holder
        .iter_original()
        .filter_map(|(_, locked)| {
            let segment = locked.read();
            let index = segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow();
            match &*index {
                VectorIndexEnum::Lmi(lmi)
                    if segment.id_tracker.borrow().available_point_count() > 0 =>
                {
                    Some((segment.uuid.to_string(), lmi.search_path_counts_for_test()))
                }
                _ => None,
            }
        })
        .collect();
    counts.sort_by(|a, b| a.0.cmp(&b.0));
    counts
}

fn lmi_segment_members(
    holder: &shard::segment_holder::locked::LockedSegmentHolder,
) -> Vec<(String, Vec<u64>)> {
    let holder = holder.read();
    let mut members = Vec::new();
    for (_, locked) in holder.iter_original() {
        let segment = locked.read();
        let tracker = segment.id_tracker.borrow();
        if tracker.available_point_count() == 0 {
            continue;
        }
        let index = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow();
        assert!(
            matches!(&*index, VectorIndexEnum::Lmi(_)),
            "live segment is not LMI"
        );
        let ids = tracker
            .point_mappings()
            .iter_from(None)
            .map(|(id, _)| match id {
                PointIdType::NumId(id) => id,
                _ => panic!("non-numeric external ID"),
            })
            .collect();
        members.push((segment.uuid.to_string(), ids));
    }
    members.sort_by(|a, b| a.0.cmp(&b.0));
    members
}

async fn exercise_collection_query(
    collection: &Collection,
    shard: &Arc<ShardReplicaSet>,
    expected_live_lmi_segments: usize,
    cross_segment_top_k: bool,
) -> serde_json::Value {
    let holder = shard.local_segments_for_test().await.unwrap();
    let before = lmi_search_counts(&holder);
    assert_eq!(before.len(), expected_live_lmi_segments);
    let partitions = lmi_segment_members(&holder);
    assert_eq!(partitions.len(), expected_live_lmi_segments);
    if cross_segment_top_k {
        assert_eq!(
            partitions.iter().map(|(_, ids)| ids.len()).sum::<usize>(),
            SMALL_N
        );
        assert!(partitions.iter().any(|(_, ids)| ids.contains(&1)));
        assert!(partitions.iter().any(|(_, ids)| ids.contains(&33)));
        let segment_for_1 = partitions
            .iter()
            .find(|(_, ids)| ids.contains(&1))
            .unwrap()
            .0
            .clone();
        let segment_for_33 = partitions
            .iter()
            .find(|(_, ids)| ids.contains(&33))
            .unwrap()
            .0
            .clone();
        assert_ne!(
            segment_for_1, segment_for_33,
            "reference neighbors must come from different segments"
        );
        let mut ids: Vec<_> = partitions
            .iter()
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (1..=SMALL_N as u64).collect::<Vec<_>>());
    }
    drop(holder);

    let mut vector = vec![0.0; DIM];
    vector[1] = 1.0;
    if cross_segment_top_k {
        vector[33] = 1.0;
    }
    let found = collection
        .search(
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(vector)),
                filter: None,
                params: None,
                limit: if cross_segment_top_k { 2 } else { 10 },
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
    let scored: Vec<(u64, f32)> = found
        .into_iter()
        .map(|point| match point.id {
            segment::types::ExtendedPointId::NumId(id) => (id, point.score),
            _ => panic!("unexpected nonnumeric result ID"),
        })
        .collect();
    let ids: Vec<_> = scored.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids.iter().collect::<HashSet<_>>().len(), ids.len());
    if cross_segment_top_k {
        assert_eq!(ids.len(), 2);
        assert_eq!(
            ids.iter().copied().collect::<HashSet<_>>(),
            HashSet::from([1, 33])
        );
        for (_, score) in &scored {
            assert!((*score - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.002);
        }
    } else {
        assert_eq!(ids.first(), Some(&1));
    }

    let holder = shard.local_segments_for_test().await.unwrap();
    let after = lmi_search_counts(&holder);
    assert_eq!(after.len(), before.len());
    let mut learned_deltas = Vec::new();
    for ((before_uuid, before_counts), (after_uuid, after_counts)) in before.iter().zip(&after) {
        assert_eq!(before_uuid, after_uuid);
        assert_eq!(
            after_counts.0,
            before_counts.0 + 1,
            "segment {before_uuid} missed learned routing"
        );
        assert_eq!(
            after_counts.1, before_counts.1,
            "segment {before_uuid} fell back to Plain"
        );
        learned_deltas.push(
            serde_json::json!({"segment":after_uuid,"learned_delta":1,"plain_fallback_delta":0}),
        );
    }
    drop(holder);
    serde_json::json!({"results":ids,"segments":learned_deltas,"partitions":partitions})
}

async fn phase(collection: &Collection, collection_path: &Path, label: &str) -> serde_json::Value {
    let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
    let shard = collection
        .shards_holder()
        .read()
        .await
        .get_shard(0)
        .unwrap()
        .clone();
    let segments = shard.local_segments_for_test().await.unwrap();
    let holder = segments.read();
    let original_count = holder.iter_original().count();
    let active_count = holder.iter().count();
    let value = serde_json::json!({
        "phase": label,
        "status": format!("{:?}", info.status),
        "optimizer_status": format!("{:?}", info.optimizer_status),
        "indexed": info.indexed_vectors_count,
        "points": info.points_count,
        "reported_segments": info.segments_count,
        "active_holder_entries": active_count,
        "active_proxies": active_count - original_count,
        "active_originals": holder_rows(&holder),
        "physical_directories": physical_dirs(collection_path),
    });
    println!("SISAP23_SMALL_PHASE {}", value);
    value
}

fn audit_active(segments: &shard::segment_holder::locked::LockedSegmentHolder) -> String {
    let holder = segments.read();
    let mut seen = vec![false; SMALL_N];
    let mut fingerprint = [0u8; 32];
    let mut count = 0;
    assert_eq!(
        holder.iter().count(),
        holder.iter_original().count(),
        "proxy remains"
    );
    for (_, locked) in holder.iter_original() {
        let segment = locked.read();
        let config = &segment.segment_config.vector_data[DEFAULT_VECTOR_NAME];
        assert_eq!(config.size, DIM);
        assert_eq!(config.distance, Distance::Cosine);
        assert_eq!(config.datatype, Some(VectorStorageDatatype::Float16));
        let tracker = segment.id_tracker.borrow();
        let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow();
        assert_eq!(tracker.deleted_point_count(), 0);
        for (external, offset) in tracker.point_mappings().iter_from(None) {
            let PointIdType::NumId(id) = external else {
                panic!("non-numeric external ID");
            };
            assert!((1..=SMALL_N as u64).contains(&id));
            assert!(
                !std::mem::replace(&mut seen[id as usize - 1], true),
                "duplicate ID"
            );
            assert_eq!(tracker.external_id(offset), Some(external));
            assert!(tracker.internal_version(offset).is_some());
            assert!(!storage.is_deleted_vector(offset));
            let VectorInternal::Dense(vector) = storage
                .get_vector_opt::<Random>(offset)
                .expect("missing vector")
                .to_owned()
            else {
                panic!("non-dense vector");
            };
            assert_eq!(vector.len(), DIM);
            assert_eq!(vector[id as usize], 1.0);
            assert_eq!(vector.iter().filter(|&&value| value != 0.0).count(), 1);
            let mut hash = Sha256::new();
            hash.update(id.to_le_bytes());
            for value in vector {
                hash.update(f16::from_f32(value).to_bits().to_le_bytes());
            }
            for (out, byte) in fingerprint.iter_mut().zip(hash.finalize()) {
                *out ^= byte;
            }
            count += 1;
        }
    }
    assert_eq!(count, SMALL_N);
    assert!(seen.iter().all(|present| *present));
    fingerprint
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

async fn wait_small(collection: &Collection, path: &Path, phases: &mut Vec<serde_json::Value>) {
    let deadline = tokio::time::Instant::now() + SMALL_DEADLINE;
    loop {
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        assert!(
            !matches!(
                info.optimizer_status,
                crate::operations::types::OptimizersStatus::Error(_)
            ),
            "optimizer error: {:?}",
            info.optimizer_status
        );
        assert!(info.points_count.unwrap_or(0) <= SMALL_N);
        assert!(info.indexed_vectors_count.unwrap_or(0) <= SMALL_N);
        if info.indexed_vectors_count == Some(SMALL_N) {
            phases.push(phase(collection, path, "first_aggregate_indexed_count").await);
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "small indexing deadline"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let mut green_samples = 0;
    while green_samples < 3 {
        let info = collection.info(&ShardSelectorInternal::All).await.unwrap();
        let ready = optimizer_complete(
            &info.optimizer_status,
            info.status,
            info.indexed_vectors_count,
            info.points_count,
            SMALL_N,
        )
        .unwrap();
        green_samples = if ready { green_samples + 1 } else { 0 };
        assert!(
            tokio::time::Instant::now() < deadline,
            "optimizer quiescence deadline"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    phases.push(phase(collection, path, "optimizer_quiescent").await);
}

async fn run_small_lifecycle(variant: Variant, multi_segment: bool) {
    let scratch = small_scratch();

    let tmp = tempfile::tempdir_in(&scratch).unwrap();
    let path = tmp.path().join("collection");
    let snapshots = tmp.path().join("snapshots");
    fs::create_dir(&path).unwrap();
    fs::create_dir(&snapshots).unwrap();

    let mut vp = VectorParamsBuilder::new(DIM as u64, Distance::Cosine).build();
    vp.datatype = Some(Datatype::Float16);
    vp.memory = Some(segment::types::Memory::Cold);
    let mut optimizers = OptimizersConfig::fixture();
    optimizers.max_optimization_threads = Some(0);
    optimizers.indexing_threshold = Some(usize::MAX);
    optimizers.default_segment_number = 1;
    // 48 KiB keeps two 32-point source segments separate during the targeted
    // multi-segment test. The production-sized lifecycle keeps the normal cap.
    optimizers.max_segment_size = Some(if multi_segment { 48 } else { 32_000_000 });
    let config = CollectionConfigInternal {
        params: CollectionParams {
            vectors: VectorsConfig::Single(vp),
            shard_number: NonZeroU32::new(1).unwrap(),
            ..CollectionParams::empty()
        },
        optimizer_config: optimizers,
        wal_config: Default::default(),
        hnsw_config: HnswConfig::default(),
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
    };
    let collection = Collection::new(
        "sisap23_small_lifecycle".to_string(),
        1,
        &path,
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
    let shard = collection
        .shards_holder()
        .read()
        .await
        .get_shard(0)
        .unwrap()
        .clone();
    let points: Vec<_> = (1..=SMALL_N)
        .map(|id| {
            let mut values = vec![0f32; DIM];
            values[id] = 1.0;
            PointStructPersisted {
                id: (id as u64).into(),
                vector: VectorStructPersisted::Single(values),
                payload: None,
            }
        })
        .collect();
    let batches: Vec<Vec<_>> = if multi_segment {
        points
            .chunks(SMALL_N / 2)
            .map(|chunk| chunk.to_vec())
            .collect()
    } else {
        vec![points]
    };
    for batch in batches {
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
    }
    shard.force_flush_local_for_test().await;
    drop(shard);
    collection.stop_gracefully().await;
    drop(collection);

    let before = super::memory_recovery::audit_data(&path, SMALL_N, None);
    let collection = open_paused(&path, &snapshots).await;
    let plain_baseline = phase(&collection, &path, "plain_before_index").await;
    if multi_segment {
        let live: Vec<_> = plain_baseline["active_originals"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["live"].as_u64().unwrap() > 0)
            .collect();
        assert_eq!(
            live.len(),
            2,
            "fixture did not create two Plain source segments: {plain_baseline}"
        );
        assert!(
            live.iter()
                .all(|row| row["index_kind"] == "plain" && row["live"] == (SMALL_N / 2) as u64)
        );
    }
    let mut phases = vec![plain_baseline];
    if variant == Variant::Lmi {
        let diagnostic = LmiConfig {
            n_buckets: 2,
            sample_size: 32,
            hidden_dim: 8,
            epochs: 1,
            batch_size: 8,
            routing_batch_size: 8,
            kmeans_iterations: 1,
            nprobe: 2,
            seed: 7,
        };
        collection
            .update_vectors_from_diff(
                &serde_json::from_value(serde_json::json!({"":{"lmi_config":diagnostic}})).unwrap(),
            )
            .await
            .unwrap();
        let max_segment_size = if multi_segment { 48 } else { 32_000_000 };
        collection
            .update_optimizer_params_from_diff(
                serde_json::from_value(serde_json::json!({
                    "max_optimization_threads":1,"indexing_threshold":1,
                    "default_segment_number":1,"max_segment_size":max_segment_size
                }))
                .unwrap(),
            )
            .await
            .unwrap();
        collection.recreate_optimizers_background();
    } else {
        super::memory_recovery::configure_index(&collection, Variant::HnswA).await;
    }
    wait_small(&collection, &path, &mut phases).await;
    if multi_segment {
        assert!(
            super::active_indexes_ready(&collection, Variant::Lmi, SMALL_N)
                .await
                .unwrap(),
            "each live segment must be fully LMI-indexed before the fixture is considered ready"
        );
        let topology = phase(&collection, &path, "two_lmi_segments_quiescent").await;
        let live: Vec<_> = topology["active_originals"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["live"].as_u64().unwrap() > 0)
            .collect();
        assert_eq!(
            live.len(),
            2,
            "expected two simultaneously active segments: {topology}"
        );
        for row in &live {
            assert_eq!(row["index_kind"], "lmi");
            assert_eq!(row["live"], (SMALL_N / 2) as u64);
            assert_eq!(row["indexed_vectors"], (SMALL_N / 2) as u64);
        }
    }
    let shard = collection
        .shards_holder()
        .read()
        .await
        .get_shard(0)
        .unwrap()
        .clone();
    let holder = shard.local_segments_for_test().await.unwrap();
    let active_fingerprint = audit_active(&holder);
    assert_eq!(
        active_fingerprint,
        before["vector_fingerprint_xor_sha256"].as_str().unwrap()
    );
    let pre_flush_query = if variant == Variant::Lmi {
        Some(
            exercise_collection_query(
                &collection,
                &shard,
                if multi_segment { 2 } else { 1 },
                multi_segment,
            )
            .await,
        )
    } else {
        None
    };
    phases.push(serde_json::json!({"phase":"pre_flush_query","query_path":pre_flush_query}));
    shard.force_flush_local_for_test().await;
    let flushed = phase(&collection, &path, "synchronous_flush").await;
    if multi_segment {
        let live: Vec<_> = flushed["active_originals"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["live"].as_u64().unwrap() > 0)
            .collect();
        assert_eq!(
            live.len(),
            2,
            "flush changed the two-live-segment topology: {flushed}"
        );
        assert!(live.iter().all(
            |row| row["index_kind"] == "lmi" && row["indexed_vectors"] == (SMALL_N / 2) as u64
        ));
    }
    let post_flush_query = if multi_segment {
        Some(exercise_collection_query(&collection, &shard, 2, true).await)
    } else {
        None
    };
    phases.push(serde_json::json!({"phase":"synchronous_flush","query_path":post_flush_query}));
    drop(holder);
    drop(shard);
    collection.stop_gracefully().await;
    drop(collection);
    let physical_after_stop = physical_dirs(&path);
    let shutdown_topology: Vec<_> = segment_topology(&path)
        .into_iter()
        .map(|(_, summary)| summary)
        .collect();
    println!(
        "SISAP23_SMALL_SHUTDOWN {}",
        serde_json::json!({
            "physical": physical_after_stop, "topology": shutdown_topology
        })
    );
    let live_paths: Vec<_> = segment_topology(&path)
        .into_iter()
        .filter(|(_, row)| row["live_points"].as_u64().unwrap() > 0)
        .map(|(path, _)| path)
        .collect();
    let expected_live = if multi_segment { 2 } else { 1 };
    assert_eq!(live_paths.len(), expected_live);
    let index_hashes: Vec<_> = live_paths
        .iter()
        .map(|path| {
            let hashes = hash_index_tree(path);
            assert!(!hashes.is_empty(), "native index absent after flush");
            let persisted = load_index(path);
            let indexed = persisted.vector_data[DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow();
            assert_eq!(
                index_kind(&indexed),
                if variant == Variant::Lmi {
                    "lmi"
                } else {
                    "hnsw"
                }
            );
            assert_eq!(indexed.indexed_vector_count(), SMALL_N / expected_live);
            hashes
        })
        .collect();
    let collection = open_paused(&path, &snapshots).await;
    let restart = phase(&collection, &path, "clean_restart").await;
    let live: Vec<_> = restart["active_originals"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["live"].as_u64().unwrap() > 0)
        .collect();
    assert_eq!(
        live.len(),
        expected_live,
        "restart changed live segment topology"
    );
    for row in &live {
        assert_eq!(
            row["index_kind"],
            if variant == Variant::Lmi {
                "lmi"
            } else {
                "hnsw"
            }
        );
        assert_eq!(row["live"], (SMALL_N / expected_live) as u64);
        assert_eq!(row["indexed_vectors"], (SMALL_N / expected_live) as u64);
    }
    phases.push(restart);
    let shard = collection
        .shards_holder()
        .read()
        .await
        .get_shard(0)
        .unwrap()
        .clone();
    let holder = shard.local_segments_for_test().await.unwrap();
    assert_eq!(audit_active(&holder), active_fingerprint);
    let before_search = (variant == Variant::Lmi).then(|| lmi_search_counts(&holder));
    drop(holder);
    let query_path = if let Some(before_search) = before_search {
        assert_eq!(before_search.len(), expected_live);
        let query_path =
            exercise_collection_query(&collection, &shard, expected_live, multi_segment).await;
        println!("SISAP23_SMALL_QUERY_PATH {}", query_path);
        Some(query_path)
    } else {
        None
    };
    drop(shard);
    collection.stop_gracefully().await;
    drop(collection);
    let after_hashes: Vec<_> = segment_topology(&path)
        .into_iter()
        .filter(|(_, row)| row["live_points"].as_u64().unwrap() > 0)
        .map(|(path, _)| hash_index_tree(&path))
        .collect();
    assert_eq!(
        after_hashes, index_hashes,
        "query or restart changed LMI files"
    );
    let after = super::memory_recovery::audit_data(&path, SMALL_N, None);
    assert_eq!(
        before["vector_fingerprint_xor_sha256"],
        after["vector_fingerprint_xor_sha256"]
    );
    let report_path = PathBuf::from(std::env::var("SISAP23_SMALL_REPORT").unwrap());
    let results = PathBuf::from(std::env::var("SISAP23_SMALL_RESULTS").unwrap())
        .canonicalize()
        .unwrap();
    assert_eq!(
        report_path.parent().unwrap().canonicalize().unwrap(),
        results
    );
    assert!(!report_path.exists(), "refuse existing report");
    write_json_new(
        &report_path,
        &serde_json::json!({
            "format": "sisap23-small-lifecycle-v1",
            "source_commit": std::env::var("SISAP23_REVIEWED_COMMIT").unwrap(),
            "accepted": true,
            "points": SMALL_N,
            "phases": phases,
            "physical_after_shutdown": physical_after_stop,
            "topology_after_shutdown": shutdown_topology,
            "fingerprint": active_fingerprint,
            "variant": variant.name(),
            "index_files": index_hashes.iter().map(|hashes| hashes.len()).sum::<usize>(),
            "query_path": query_path,
            "ingested_once": true,
            "reused_vectors_for_index": true
        }),
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "opt-in disposable Float16 native HNSW collection; no SISAP input"]
async fn sisap23_small_float16_hnsw_lifecycle() {
    run_small_lifecycle(Variant::HnswA, false).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "opt-in disposable Float16 native LMI collection; no SISAP input"]
async fn sisap23_small_float16_lmi_lifecycle() {
    run_small_lifecycle(Variant::Lmi, false).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "opt-in two-active-segment Float16 LMI collection search; no SISAP input"]
async fn sisap23_small_float16_lmi_two_segment_search() {
    run_small_lifecycle(Variant::Lmi, true).await;
}

fn assert_small_segment(path: &Path, expected_id: u64) {
    let segment = load_index(path);
    let config = &segment.segment_config.vector_data[DEFAULT_VECTOR_NAME];
    assert_eq!(config.size, DIM);
    assert_eq!(config.distance, Distance::Cosine);
    assert_eq!(config.datatype, Some(VectorStorageDatatype::Float16));
    let tracker = segment.id_tracker.borrow();
    assert_eq!(tracker.available_point_count(), 1);
    assert_eq!(tracker.deleted_point_count(), 0);
    let mappings: Vec<_> = tracker.point_mappings().iter_from(None).collect();
    assert_eq!(mappings.len(), 1);
    let (external, offset) = mappings[0];
    assert_eq!(external, PointIdType::NumId(expected_id));
    assert_eq!(tracker.external_id(offset), Some(external));
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    assert_eq!(storage.datatype(), VectorStorageDatatype::Float16);
    let VectorInternal::Dense(vector) =
        storage.get_vector_opt::<Random>(offset).unwrap().to_owned()
    else {
        panic!("non-dense vector");
    };
    assert_eq!(vector.len(), DIM);
    assert_eq!(vector[expected_id as usize], 1.0);
    assert_eq!(vector.iter().filter(|&&value| value != 0.0).count(), 1);
}

fn small_segment(path: &Path, id: u64) -> Segment {
    let config = SegmentConfig {
        vector_data: std::collections::HashMap::from([(
            DEFAULT_VECTOR_NAME.to_owned(),
            VectorDataConfig {
                size: DIM,
                distance: Distance::Cosine,
                storage_type: VectorStorageType::ChunkedMmap,
                index: Indexes::Plain {},
                quantization_config: None,
                multivector_config: None,
                datatype: Some(VectorStorageDatatype::Float16),
            },
        )]),
        sparse_vector_data: Default::default(),
        payload_storage_type: Default::default(),
        id_tracker_memory: None,
    };
    let mut segment = build_segment(path, &config, None, true).unwrap().0;
    let mut values = vec![0f32; DIM];
    values[id as usize] = 1.0;
    let mut vectors = NamedVectors::default();
    vectors.insert(DEFAULT_VECTOR_NAME.to_owned(), values.into());
    segment
        .upsert_point(
            1,
            id.into(),
            vectors,
            &common::counter::hardware_counter::HardwareCounterCell::new(),
        )
        .unwrap();
    segment.flush(true).unwrap();
    segment
}

#[test]
#[ignore = "opt-in disposable native holder retirement; no SISAP input"]
fn sisap23_small_float16_holder_retirement() {
    let scratch = small_scratch();

    let tmp = tempfile::tempdir_in(&scratch).unwrap();
    let collection_path = tmp.path().join("collection");
    let path = collection_path.join("0/segments");
    fs::create_dir_all(&path).unwrap();
    let mut holder = SegmentHolder::default();
    holder.add_new(small_segment(&path, 1));
    holder.add_new(small_segment(&path, 2));
    assert_eq!(holder.iter_original().count(), 2);
    assert_eq!(physical_dirs(&collection_path).len(), 2);
    let paths_and_ids: Vec<_> = holder
        .iter_original()
        .map(|(_, locked)| {
            let segment = locked.read();
            let id = segment
                .id_tracker
                .borrow()
                .point_mappings()
                .iter_from(None)
                .next()
                .unwrap()
                .0;
            let PointIdType::NumId(id) = id else {
                panic!("non-numeric ID");
            };
            (segment.segment_path.clone(), id)
        })
        .collect();
    let ids: Vec<_> = paths_and_ids.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&1) && ids.contains(&2));
    println!(
        "SISAP23_SMALL_TWO_ACTIVE {}",
        serde_json::json!({
            "active": holder_rows(&holder), "physical": physical_dirs(&collection_path)
        })
    );
    drop(holder);
    for (segment_path, id) in paths_and_ids {
        assert_small_segment(&segment_path, id);
    }
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| locate_segment(
            &collection_path
        )))
        .is_err()
    );

    // Reproduce Qdrant's deferred source-drop protocol in a separate directory.
    let retired_collection_path = tmp.path().join("retired");
    let path = retired_collection_path.join("0/segments");
    fs::create_dir_all(&path).unwrap();
    let source = small_segment(&path, 1);
    let source_uuid = source.uuid.to_string();
    let replacement = small_segment(&path, 1);
    let replacement_uuid = replacement.uuid.to_string();
    let mut holder = SegmentHolder::default();
    let source_id = holder.add_new(source);
    let retired = holder.remove(&[source_id]).pop().unwrap();
    holder.add_new(replacement);
    holder.register_segment_drop(1, 1, retired);
    assert_eq!(holder.iter_original().count(), 1);
    assert!(physical_dirs(&retired_collection_path).contains(&source_uuid));
    assert!(physical_dirs(&retired_collection_path).contains(&replacement_uuid));
    println!(
        "SISAP23_SMALL_RETIRED_BEFORE_FLUSH {}",
        serde_json::json!({
            "active": holder_rows(&holder), "physical": physical_dirs(&retired_collection_path)
        })
    );
    holder.flush_all(FlushMode::Sync, true).unwrap();
    assert!(!physical_dirs(&retired_collection_path).contains(&source_uuid));
    assert!(physical_dirs(&retired_collection_path).contains(&replacement_uuid));
    println!(
        "SISAP23_SMALL_RETIRED_AFTER_FLUSH {}",
        serde_json::json!({
            "active": holder_rows(&holder), "physical": physical_dirs(&retired_collection_path)
        })
    );
    drop(holder);
    assert_small_segment(&path.join(&replacement_uuid), 1);
}
