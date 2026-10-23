//! Opt-in, process-separated recovery of vector data and index-only construction.
use super::*;
use segment::data_types::vectors::VectorInternal;
use segment::entry::ReadSegmentEntry;
use segment::utils::mem::Mem;

/// Full audit across all active segments after normal Collection startup has
/// reconciled temporary/deleted optimizer output. No synthetic inserts occur.
pub(super) fn audit_data(path: &Path, expected: usize, corpus: Option<&Path>) -> serde_json::Value {
    let mut paths: Vec<_> = fs::read_dir(path.join("0/segments"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|p| p.is_dir() && Uuid::parse_str(p.file_name().unwrap().to_str().unwrap()).is_ok())
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no active segments after normal recovery"
    );
    let mut seen = vec![false; expected];
    let mut count = 0usize;
    let mut fingerprint = [0u8; 32];
    let mut source = corpus.map(|p| File::open(p).unwrap());
    let mut raw = vec![0u8; DIM * 2];
    let mut max_error = 0f32;
    let mut kinds = Vec::new();
    for path in paths {
        let segment = load_index(&path);
        assert_eq!(segment.segment_config.vector_data.len(), 1);
        assert!(segment.segment_config.sparse_vector_data.is_empty());
        let config = &segment.segment_config.vector_data[DEFAULT_VECTOR_NAME];
        assert_eq!(config.size, DIM);
        assert_eq!(config.distance, Distance::Cosine);
        assert_eq!(config.datatype, Some(VectorStorageDatatype::Float16));
        assert!(config.quantization_config.is_none());
        let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
        let storage = data.vector_storage.borrow();
        let tracker = segment.id_tracker.borrow();
        if tracker.available_point_count() == 0 {
            assert_eq!(tracker.total_point_count(), tracker.deleted_point_count());
            assert!(storage.total_vector_count() <= tracker.total_point_count());
            for offset in 0..storage.total_vector_count() as u32 {
                assert!(tracker.is_deleted_point(offset), "orphan live vector slot");
                assert!(tracker.external_id(offset).is_none());
            }
            continue; // Native optimizer may retain an entirely tombstoned appendable segment.
        }
        assert_eq!(
            storage.deleted_vector_count(),
            0,
            "deleted vector slots require investigation"
        );
        assert_eq!(tracker.deleted_point_count(), 0);
        assert_eq!(
            tracker.available_point_count(),
            storage.available_vector_count()
        );
        let index = data.vector_index.borrow();
        if tracker.available_point_count() > 0 {
            kinds.push(index_kind(&index));
        }
        for (id, offset) in tracker.point_mappings().iter_from(None) {
            let PointIdType::NumId(numeric) = id else {
                panic!("UUID in numeric SISAP corpus");
            };
            assert!(
                (1..=expected as u64).contains(&numeric),
                "out-of-range external ID"
            );
            assert!(
                !std::mem::replace(&mut seen[numeric as usize - 1], true),
                "duplicate external ID"
            );
            assert_eq!(
                tracker.external_id(offset),
                Some(id),
                "reverse mapping mismatch"
            );
            assert!(
                tracker.internal_version(offset).is_some(),
                "missing point version"
            );
            assert!(
                (offset as usize) < storage.total_vector_count(),
                "mapping beyond stored vectors"
            );
            assert!(!storage.is_deleted_vector(offset));
            let value = storage
                .get_vector_opt::<Random>(offset)
                .expect("absent vector")
                .to_owned();
            let VectorInternal::Dense(actual) = value else {
                panic!("non-dense vector");
            };
            assert_eq!(actual.len(), DIM);
            assert!(actual.iter().all(|v| v.is_finite()), "non-finite vector");
            if let Some(source) = source.as_mut() {
                source
                    .seek(std::io::SeekFrom::Start((numeric - 1) * (DIM * 2) as u64))
                    .unwrap();
                source.read_exact(&mut raw).unwrap();
                let values = raw
                    .chunks_exact(2)
                    .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                    .collect();
                let normalized = Distance::Cosine.preprocess_vector::<f32>(values);
                for (a, b) in actual.iter().zip(normalized) {
                    let error = (a - f16::from_f32(b).to_f32()).abs();
                    assert!(
                        error <= 0.002,
                        "stored/corpus mismatch for external ID {numeric}"
                    );
                    max_error = max_error.max(error);
                }
            }
            // Commutative fingerprint of (unique external ID, Float16 vector).
            // ID range/uniqueness/count are independently checked above.
            let mut hash = Sha256::new();
            hash.update(numeric.to_le_bytes());
            let bytes: Vec<u8> = actual
                .iter()
                .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
                .collect();
            hash.update(bytes);
            for (sum, byte) in fingerprint.iter_mut().zip(hash.finalize()) {
                *sum ^= byte;
            }
            count += 1;
        }
    }
    assert_eq!(count, expected);
    assert!(seen.iter().all(|v| *v), "missing external IDs");
    serde_json::json!({
        "accepted":true,"points":count,"dimension":DIM,"datatype":"float16","distance":"cosine",
        "missing_ids":0,"duplicate_ids":0,"invalid_mappings":0,"absent_vectors":0,
        "corpus_vectors_compared":if corpus.is_some() {expected} else {0},
        "corpus_max_abs_error":max_error,"physical_index_kinds":kinds,
        "vector_fingerprint_xor_sha256":fingerprint.iter().map(|v| format!("{v:02x}")).collect::<String>()
    })
}

fn checked_collection() -> PathBuf {
    let path = PathBuf::from(std::env::var("SISAP23_RECOVERY_COLLECTION").unwrap())
        .canonicalize()
        .unwrap();
    let scratch = PathBuf::from(std::env::var("SCRATCHDIR").unwrap())
        .canonicalize()
        .unwrap();
    let staging = PathBuf::from(std::env::var("SISAP23_RECOVERY_ROOT").unwrap()).join(".staging");
    let under_staging = staging.exists() && path.starts_with(staging.canonicalize().unwrap());
    assert!(
        (path.starts_with(&scratch) && path != scratch) || under_staging,
        "recovery may only open verified scratch or new persistent staging copies"
    );
    assert!(
        !path.starts_with(root().canonicalize().unwrap()),
        "original experiment may never be opened"
    );
    path
}

fn report(value: serde_json::Value) {
    let path = PathBuf::from(std::env::var("SISAP23_RECOVERY_REPORT").unwrap());
    write_json_new(&path, &value);
}

fn print_memory(label: &str) {
    println!(
        "SISAP23_MEMORY {label} {:?}",
        Mem::new().admission_snapshot()
    );
}

fn config_is_compatible(path: &Path) {
    let config = CollectionConfigInternal::load(path).unwrap();
    assert!(
        !config
            .optimizer_config
            .prevent_unoptimized
            .unwrap_or_default(),
        "deferred WAL replay is not supported by this recovery protocol"
    );
    assert_eq!(config.params.shard_number.get(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit copied-data WAL recovery and exhaustive corpus verification"]
async fn sisap23_recovery_vectors() {
    let path = checked_collection();
    config_is_compatible(&path);
    let snapshots = path.parent().unwrap().join("snapshots-recovery");
    fs::create_dir_all(&snapshots).unwrap();
    print_memory("before_recovery");
    reopen_for_verification(&path, &snapshots, N).await;
    let corpus = PathBuf::from(std::env::var("SISAP23_CORPUS_PATH").unwrap());
    let first = audit_data(&path, N, Some(&corpus));
    assert!(
        first["physical_index_kinds"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v == "plain"),
        "source already contains an index; do not discard or rebuild it automatically"
    );
    reopen_for_verification(&path, &snapshots, N).await;
    let second = audit_data(&path, N, None);
    assert_eq!(
        first["vector_fingerprint_xor_sha256"],
        second["vector_fingerprint_xor_sha256"]
    );
    print_memory("after_recovery");
    report(
        serde_json::json!({"format":"sisap23-recovered-data-v2","accepted":true,
        "first":first,"second":second,"normal_restarts":2,"ingested_points":0}),
    );
}

pub(super) async fn configure_index(collection: &Collection, variant: Variant) {
    let lmi = if variant == Variant::Lmi {
        serde_json::to_value(lmi_config()).unwrap()
    } else {
        serde_json::json!("Disabled")
    };
    collection
        .update_vectors_from_diff(
            &serde_json::from_value(serde_json::json!({"":{"lmi_config":lmi}})).unwrap(),
        )
        .await
        .unwrap();
    if matches!(variant, Variant::HnswA | Variant::HnswB) {
        let (m, ef) = variant.params();
        collection
            .update_hnsw_config_from_diff(
                serde_json::from_value(serde_json::json!({
                    "m":m,"ef_construct":ef,"max_indexing_threads":1,"memory":"cached"
                }))
                .unwrap(),
            )
            .await
            .unwrap();
    }
    collection
        .update_optimizer_params_from_diff(
            serde_json::from_value(serde_json::json!({
                "max_optimization_threads":1,"indexing_threshold":1,
                "default_segment_number":1,"max_segment_size":32000000
            }))
            .unwrap(),
        )
        .await
        .unwrap();
    collection.recreate_optimizers_background();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit index-only construction on verified recovered data"]
async fn sisap23_recovery_index() {
    let path = checked_collection();
    config_is_compatible(&path);
    let variant = Variant::parse(&std::env::var("SISAP23_VARIANT").unwrap());
    assert_ne!(variant, Variant::Plain);
    let before = audit_data(&path, N, None);
    assert!(
        before["physical_index_kinds"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v == "plain")
    );
    let snapshots = path.parent().unwrap().join("snapshots-index");
    fs::create_dir_all(&snapshots).unwrap();
    print_memory("before_index");
    // Early admission before enabling the optimizer. Training repeats this at
    // the real allocation boundary, after segment copying and live sampling.
    if variant == Variant::Lmi {
        segment::index::lmi_index::check_build_memory(N, DIM, &lmi_config()).unwrap();
    }
    let collection = open_paused(&path, &snapshots).await;
    configure_index(&collection, variant).await;
    let result = wait_for_index(&collection, variant, N).await;
    if result.is_ok() {
        let shard = collection
            .shards_holder()
            .read()
            .await
            .get_shard(0)
            .unwrap()
            .clone();
        shard.force_flush_local_for_test().await;
    }
    collection.stop_gracefully().await;
    drop(collection);
    let seconds = result.unwrap_or_else(|e| panic!("{e}"));
    let index_before: Vec<_> = live_segment_paths(&path)
        .into_iter()
        .map(|segment| {
            let hash = hash_index_tree(&segment);
            (segment, hash)
        })
        .collect();
    reopen_for_verification(&path, &snapshots, N).await;
    let persisted = verify_persisted_generation(&path, variant, N);
    assert_eq!(
        index_before,
        live_segment_paths(&path)
            .into_iter()
            .map(|segment| {
                let hash = hash_index_tree(&segment);
                (segment, hash)
            })
            .collect::<Vec<_>>(),
        "verification rebuilt index"
    );
    let after = audit_data(&path, N, None);
    assert_eq!(
        before["vector_fingerprint_xor_sha256"], after["vector_fingerprint_xor_sha256"],
        "index construction changed vectors"
    );
    print_memory("after_index");
    report(
        serde_json::json!({"format":"sisap23-index-only-v2","accepted":true,
        "variant":variant.name(),"data":after,"index":persisted,
        "index_build_seconds":seconds,"ingest_seconds":null,"ingested_points":0,
        "lmi":if variant == Variant::Lmi {serde_json::to_value(lmi_config()).unwrap()} else {serde_json::Value::Null},
        "hnsw_params":variant.params()}),
    );
}

#[test]
#[ignore = "opt-in topology inventory on a separately verified scratch copy; no optimization"]
fn sisap23_recovery_inspect_topology() {
    let path = checked_collection();
    let topology = segment_topology(&path);
    let live = topology
        .iter()
        .filter(|(_, row)| row["live_points"].as_u64().unwrap() > 0)
        .count();
    println!(
        "SISAP23_SEGMENT_TOPOLOGY {}",
        serde_json::json!({
            "physical_segments": topology.len(),
            "live_segments": live,
            "segments": topology.into_iter().map(|(_, row)| row).collect::<Vec<_>>(),
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit verification of new persistent staging; no index building"]
async fn sisap23_recovery_verify() {
    let path = checked_collection();
    config_is_compatible(&path);
    let variant = std::env::var("SISAP23_VERIFY_VARIANT").unwrap();
    let snapshots = path.parent().unwrap().join("snapshots-verify");
    fs::create_dir_all(&snapshots).unwrap();
    let before = audit_data(&path, N, None);
    let mut index_files = Vec::new();
    find_named(&path, "segment.json", &mut index_files);
    let hashes: Vec<_> = index_files
        .iter()
        .map(|p| hash_index_tree(p.parent().unwrap()))
        .collect();
    for _ in 0..2 {
        reopen_for_verification(&path, &snapshots, N).await;
    }
    let after = audit_data(&path, N, None);
    assert_eq!(
        before["vector_fingerprint_xor_sha256"],
        after["vector_fingerprint_xor_sha256"]
    );
    for (p, old) in index_files.iter().zip(&hashes) {
        assert_eq!(
            old,
            &hash_index_tree(p.parent().unwrap()),
            "index changed during verification"
        );
    }
    let index_audit = if variant != "data" {
        Some(verify_persisted_generation(
            &path,
            Variant::parse(&variant),
            N,
        ))
    } else {
        None
    };
    let ordinary_lmi_query_path = if variant == "lmi" {
        Some(verify_normal_lmi_query(&path, &snapshots).await)
    } else {
        None
    };
    for (p, old) in index_files.iter().zip(&hashes) {
        assert_eq!(
            old,
            &hash_index_tree(p.parent().unwrap()),
            "query changed index"
        );
    }
    report(
        serde_json::json!({"format":"sisap23-persistent-verified-v2","accepted":true,
        "data":after,"normal_restarts":2,"index_files_unchanged":true,"ingested_points":0,
        "index":index_audit,"ordinary_lmi_query_path":ordinary_lmi_query_path}),
    );
}

async fn verify_normal_lmi_query(path: &Path, snapshots: &Path) -> serde_json::Value {
    use segment::data_types::vectors::NamedQuery;
    use shard::query::query_enum::QueryEnum;

    use crate::operations::types::CoreSearchRequest;

    let collection = open_paused(path, snapshots).await;
    let shard = collection
        .shards_holder()
        .read()
        .await
        .get_shard(0)
        .unwrap()
        .clone();
    let counts = |segments: &shard::segment_holder::locked::LockedSegmentHolder| {
        let holder = segments.read();
        assert_eq!(holder.iter().count(), holder.iter_original().count());
        let mut found: Vec<_> = holder
            .iter_original()
            .filter_map(|(_, locked)| {
                let segment = locked.read();
                let live = segment.available_point_count();
                if live == 0 {
                    return None;
                }
                assert!(matches!(
                    &segment.segment_config.vector_data[DEFAULT_VECTOR_NAME].index,
                    Indexes::LmiTrained(_)
                ));
                let index = segment.vector_data[DEFAULT_VECTOR_NAME]
                    .vector_index
                    .borrow();
                let VectorIndexEnum::Lmi(lmi) = &*index else {
                    panic!("active live segment is not an LMI index")
                };
                assert_eq!(index.indexed_vector_count(), live);
                Some((
                    segment.uuid.to_string(),
                    live,
                    lmi.search_path_counts_for_test(),
                ))
            })
            .collect();
        assert!(!found.is_empty(), "no active live LMI segments");
        found.sort_by(|a, b| a.0.cmp(&b.0));
        found
    };
    let holder = shard.local_segments_for_test().await.unwrap();
    let before = counts(&holder);
    drop(holder);
    assert_eq!(
        before.iter().map(|(_, points, _)| points).sum::<usize>(),
        N,
        "active LMI segments do not account for the complete generation"
    );
    let physical_live: Vec<_> = super::live_segment_paths(path)
        .iter()
        .map(|segment| segment.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let active_before: Vec<_> = before.iter().map(|(uuid, _, _)| uuid.clone()).collect();
    assert_eq!(
        active_before, physical_live,
        "physical and active live segment topology differs"
    );
    let mut vector = vec![0.0; DIM];
    vector[0] = 1.0;
    let found = collection
        .search(
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(vector)),
                filter: None,
                params: None,
                limit: 10,
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
    let ids: Vec<_> = found
        .into_iter()
        .map(|point| match point.id {
            segment::types::ExtendedPointId::NumId(id) => id,
            _ => panic!("unexpected point ID type"),
        })
        .collect();
    assert_eq!(ids.len(), 10, "ordinary LMI search returned too few points");
    assert!(ids.iter().all(|id| (1..=N as u64).contains(id)));
    assert_eq!(ids.iter().collect::<HashSet<_>>().len(), ids.len());
    let holder = shard.local_segments_for_test().await.unwrap();
    let after = counts(&holder);
    assert_eq!(after.len(), before.len());
    let mut segment_deltas = Vec::new();
    for ((before_uuid, before_live, before_counts), (after_uuid, after_live, after_counts)) in
        before.iter().zip(&after)
    {
        assert_eq!(before_uuid, after_uuid);
        assert_eq!(before_live, after_live);
        assert_eq!(
            after_counts.0,
            before_counts.0 + 1,
            "ordinary search bypassed learned routing for {before_uuid}"
        );
        assert_eq!(
            after_counts.1, before_counts.1,
            "ordinary search fell back to Plain for {before_uuid}"
        );
        segment_deltas.push(serde_json::json!({
            "segment":after_uuid,"live_points":after_live,
            "learned_delta":after_counts.0 - before_counts.0,
            "plain_fallback_delta":after_counts.1 - before_counts.1
        }));
    }
    drop(holder);
    drop(shard);
    collection.stop_gracefully().await;
    println!("SISAP23_NORMAL_QUERY_PATH segments={segment_deltas:?}");
    serde_json::json!({"live_segment_count":after.len(),"active_segments":segment_deltas,
        "learned_delta":after.len(),"plain_fallback_delta":0,"returned_ids_1_based":ids})
}

#[test]
#[ignore = "opt-in memory admission probe; no data ingestion or training"]
fn sisap23_recovery_memory_probe() {
    print_memory("probe");
    let plan = segment::index::lmi_index::check_build_memory(N, DIM, &lmi_config()).unwrap();
    report(serde_json::json!({"accepted":true,"plan":plan,
        "memory":Mem::new().admission_snapshot().unwrap()}));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "small native Float16 recovery/index-only regression; no SISAP inputs needed"]
async fn sisap23_reuse_vectors_without_reingestion() {
    let tmp = tempfile::tempdir().unwrap();
    let collection_path = tmp.path().join("collection");
    let snapshots = tmp.path().join("snapshots");
    fs::create_dir(&collection_path).unwrap();
    fs::create_dir(&snapshots).unwrap();
    let variant = Variant::Plain;
    let (m, ef) = variant.params();
    let mut vp = VectorParamsBuilder::new(DIM as u64, Distance::Cosine).build();
    vp.datatype = Some(Datatype::Float16);
    vp.memory = Some(segment::types::Memory::Cold);
    if variant == Variant::Lmi {
        vp.lmi_config = Some(lmi_config());
    }
    let mut optimizers = OptimizersConfig::fixture();
    optimizers.max_optimization_threads = Some(0);
    optimizers.indexing_threshold = Some(if variant == Variant::Plain {
        usize::MAX
    } else {
        1
    });
    optimizers.max_segment_size = Some(32_000_000);
    optimizers.default_segment_number = 1;
    let mut hnsw = HnswConfig::default();
    if variant == Variant::HnswA || variant == Variant::HnswB {
        hnsw.m = m;
        hnsw.ef_construct = ef;
        hnsw.max_indexing_threads = 1;
        hnsw.memory = Some(segment::types::Memory::Cached);
    }
    let config = CollectionConfigInternal {
        params: CollectionParams {
            vectors: VectorsConfig::Single(vp),
            shard_number: NonZeroU32::new(1).unwrap(),
            ..CollectionParams::empty()
        },
        optimizer_config: optimizers,
        wal_config: Default::default(),
        hnsw_config: hnsw,
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
    };
    let collection = Collection::new(
        "sisap23_matched".to_string(),
        1,
        &collection_path,
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

    let count = 64;
    let mut corpus = File::create(tmp.path().join("corpus")).unwrap();
    let points = (1..=count)
        .map(|id| {
            let mut values = vec![0f32; DIM];
            values[id] = 1.0;
            for value in &values {
                corpus
                    .write_all(&f16::from_f32(*value).to_bits().to_le_bytes())
                    .unwrap();
            }
            PointStructPersisted {
                id: (id as u64).into(),
                vector: VectorStructPersisted::Single(values),
                payload: None,
            }
        })
        .collect();
    corpus.sync_all().unwrap();
    shard
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(points)),
            )),
            WaitUntil::Visible,
            None,
            HwMeasurementAcc::new(),
            false,
        )
        .await
        .unwrap();
    drop(shard);
    collection.stop_gracefully().await;
    drop(collection);
    reopen_for_verification(&collection_path, &snapshots, count).await;
    let before = audit_data(&collection_path, count, Some(&tmp.path().join("corpus")));
    // A claimed count larger than the complete stored data must fail.
    assert!(std::panic::catch_unwind(|| audit_data(&collection_path, count + 1, None)).is_err());
    let collection = open_paused(&collection_path, &snapshots).await;
    // Everything after this point uses the existing vectors: no upsert API.
    configure_index(&collection, Variant::HnswA).await;
    let result = tokio::time::timeout(
        Duration::from_secs(60),
        wait_for_index(&collection, Variant::HnswA, count),
    )
    .await;
    collection.stop_gracefully().await;
    drop(collection);
    result.unwrap().unwrap();
    for _ in 0..2 {
        reopen_for_verification(&collection_path, &snapshots, count).await;
    }
    verify_persisted_generation(&collection_path, Variant::HnswA, count);
    let after = audit_data(&collection_path, count, Some(&tmp.path().join("corpus")));
    assert_eq!(
        before["vector_fingerprint_xor_sha256"],
        after["vector_fingerprint_xor_sha256"]
    );
}
