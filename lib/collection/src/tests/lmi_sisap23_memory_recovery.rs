//! Opt-in, process-separated recovery of vector data and index-only construction.
use segment::data_types::vectors::VectorInternal;
use segment::entry::ReadSegmentEntry;
use segment::utils::mem::Mem;

use super::*;

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
    let query_proof_copy = std::env::var("SISAP23_QUERY_PROOF_COLLECTION")
        .ok()
        .map(|raw| {
            let proof = PathBuf::from(raw).canonicalize().unwrap();
            let scratch = PathBuf::from(std::env::var("SCRATCHDIR").expect("PBS scratch required"))
                .canonicalize()
                .unwrap();
            assert_ne!(proof, path, "query proof must use an isolated copy");
            assert!(
                proof.starts_with(&scratch),
                "query proof copy must be inside PBS scratch"
            );
            config_is_compatible(&proof);
            proof
        });
    let ordinary_lmi_query_path = if variant == "lmi" {
        let proof_path = query_proof_copy.as_deref().unwrap_or(&path);
        let proof_snapshots = if query_proof_copy.is_some() {
            proof_path.parent().unwrap().join("snapshots-query-proof")
        } else {
            snapshots.clone()
        };
        fs::create_dir_all(&proof_snapshots).unwrap();
        Some(verify_normal_lmi_query(proof_path, &proof_snapshots).await)
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
        "index":index_audit,"ordinary_lmi_query_path":ordinary_lmi_query_path,
        "query_proof_storage":if query_proof_copy.is_some() {"isolated_pbs_scratch"} else {"persistent_nfs"},
        "query_proof_collection":query_proof_copy.as_deref().unwrap_or(&path)}),
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

/// Run only the persistent learned-query proof and retain per-segment stage
/// diagnostics even when Qdrant's ordinary search deadline expires.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "bounded search-path diagnosis on an isolated verified collection copy"]
async fn sisap23_recovery_query_diagnostic() {
    use segment::data_types::vectors::NamedQuery;
    use shard::query::query_enum::QueryEnum;

    use crate::operations::types::CoreSearchRequest;

    let path = checked_collection();
    config_is_compatible(&path);
    let snapshots = path.parent().unwrap().join("snapshots-query-diagnostic");
    let load_started = std::time::Instant::now();
    let collection = open_paused(&path, &snapshots).await;
    let load_elapsed = load_started.elapsed().as_secs_f64();
    let shard = collection
        .shards_holder()
        .read()
        .await
        .get_shard(0)
        .unwrap()
        .clone();

    let (query, query_source) = if let Ok(query_path) = std::env::var("SISAP23_QUERY_VECTOR_PATH") {
        let bytes = fs::read(&query_path).unwrap();
        assert_eq!(
            bytes.len(),
            DIM * std::mem::size_of::<f32>(),
            "query vector file must contain exactly one little-endian dense vector"
        );
        let query = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        (
            query,
            serde_json::json!({"kind":"prepared_query","path":query_path}),
        )
    } else {
        let mut query = vec![0.0f32; DIM];
        query[0] = 1.0;
        (query, serde_json::json!({"kind":"one_hot"}))
    };
    let normalized = Distance::Cosine.preprocess_vector::<f32>(query.clone());
    let holder = shard.local_segments_for_test().await.unwrap();
    let active: Vec<_> = holder
        .read()
        .iter_original()
        .filter_map(|(_, locked)| {
            let segment = locked.read();
            let points = segment.available_point_count();
            if points == 0 {
                return None;
            }
            let index = segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow();
            let VectorIndexEnum::Lmi(lmi) = &*index else {
                panic!("active nonempty segment is not LMI");
            };
            let state = lmi.routing_state().expect("trained LMI state missing");
            let buckets = state
                .router()
                .top_buckets(&normalized, state.nprobe())
                .unwrap();
            let bucket_sizes: Vec<_> = buckets
                .iter()
                .map(|bucket| state.postings().get(*bucket).unwrap().len())
                .collect();
            let (learned, fallbacks) = lmi.search_path_counts_for_test();
            let (routed, candidates, route_ns, scored, score_ns) =
                lmi.search_stage_counts_for_test();
            Some(serde_json::json!({
                "segment":segment.uuid.to_string(),
                "live_points":points,
                "indexed_points":index.indexed_vector_count(),
                "nprobe":state.nprobe(),
                "selected_buckets":buckets,
                "selected_bucket_sizes":bucket_sizes,
                "counters_before":{
                    "learned":learned,"plain_fallback":fallbacks,
                    "routed":routed,"candidates":candidates,
                    "routing_ns":route_ns,"scored":scored,"scoring_ns":score_ns
                }
            }))
        })
        .collect();
    let active_points: usize = active
        .iter()
        .map(|segment| segment["live_points"].as_u64().unwrap() as usize)
        .sum();
    assert_eq!(
        active_points, N,
        "active LMI segments do not account for all points"
    );
    drop(holder);

    let query_started = std::time::Instant::now();
    let result = collection
        .search(
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(query)),
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
        .await;
    let query_elapsed = query_started.elapsed().as_secs_f64();
    let (returned_ids, error) = match result {
        Ok(points) => (
            points
                .into_iter()
                .map(|point| match point.id {
                    segment::types::ExtendedPointId::NumId(id) => id,
                    _ => panic!("unexpected point ID type"),
                })
                .collect::<Vec<_>>(),
            None,
        ),
        Err(error) => (Vec::new(), Some(error.to_string())),
    };

    let holder = shard.local_segments_for_test().await.unwrap();
    let after: Vec<_> = holder
        .read()
        .iter_original()
        .filter_map(|(_, locked)| {
            let segment = locked.read();
            if segment.available_point_count() == 0 {
                return None;
            }
            let index = segment.vector_data[DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow();
            let VectorIndexEnum::Lmi(lmi) = &*index else {
                panic!("active nonempty segment changed index kind during search");
            };
            let (learned, fallbacks) = lmi.search_path_counts_for_test();
            let (routed, candidates, route_ns, scored, score_ns) =
                lmi.search_stage_counts_for_test();
            Some(serde_json::json!({
                "segment":segment.uuid.to_string(),
                "counters_after":{
                    "learned":learned,"plain_fallback":fallbacks,
                    "routed":routed,"candidates":candidates,
                    "routing_ns":route_ns,"scored":scored,"scoring_ns":score_ns
                }
            }))
        })
        .collect();
    drop(holder);
    drop(shard);
    collection.stop_gracefully().await;

    let diagnostic = serde_json::json!({
        "format":"sisap23-query-diagnostic-v1",
        "source_commit":std::env::var("SISAP23_REVIEWED_COMMIT").unwrap_or_default(),
        "points":N,
        "collection_path":path,
        "query_source":query_source,
        "load_seconds":load_elapsed,
        "query_seconds":query_elapsed,
        "query_error":error,
        "returned_ids_1_based":returned_ids,
        "active_segments_before":active,
        "active_segments_after":after
    });
    println!("SISAP23_QUERY_DIAGNOSTIC {diagnostic}");
    if let Ok(report_path) = std::env::var("SISAP23_DIAGNOSTIC_REPORT") {
        write_json_new(Path::new(&report_path), &diagnostic);
    }
}

/// Reproduce the matched evaluator's twenty direct-index warmup calls, then
/// independently exercise an ordinary collection search on measured row 20.
/// This is diagnostic evidence only and cannot publish a generation.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "bounded official-query warmup on an isolated verified collection copy"]
async fn sisap23_recovery_official_warmup_diagnostic() {
    use segment::data_types::vectors::NamedQuery;
    use shard::query::query_enum::QueryEnum;

    use crate::operations::types::CoreSearchRequest;

    let collection_path = checked_collection();
    config_is_compatible(&collection_path);
    let root = PathBuf::from(std::env::var("SISAP23_OFFICIAL_QUERY_ROOT").unwrap());
    let output = PathBuf::from(std::env::var("SISAP23_WARMUP_DIAGNOSTIC_JSONL").unwrap());
    let warm = read_f32(&root.join("queries_warmup.f32le"), 20);
    let measured_path = root.join("queries_20_9999.f32le");
    let mut measured_file = File::open(&measured_path).unwrap();
    let mut measured_bytes = vec![0u8; DIM * 4];
    measured_file.read_exact(&mut measured_bytes).unwrap();
    let measured: Vec<f32> = measured_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let mut gold_file = File::open(root.join("gold_20_9999_top10.i32le")).unwrap();
    let mut gold_bytes = [0u8; 40];
    gold_file.read_exact(&mut gold_bytes).unwrap();
    let gold: Vec<i32> = gold_bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let mut log = File::options()
        .write(true)
        .create_new(true)
        .open(&output)
        .unwrap();
    let mut record = |value: serde_json::Value| {
        serde_json::to_writer(&mut log, &value).unwrap();
        log.write_all(b"\n").unwrap();
        log.sync_data().unwrap();
        println!("SISAP23_WARMUP_DIAGNOSTIC {value}");
    };

    let load_start = Instant::now();
    let collection = open_paused(
        &collection_path,
        &collection_path
            .parent()
            .unwrap()
            .join("snapshots-official-warmup"),
    )
    .await;
    record(serde_json::json!({"phase":"reopen","seconds":load_start.elapsed().as_secs_f64()}));
    let shard = collection
        .shards_holder()
        .read()
        .await
        .get_shard(0)
        .unwrap()
        .clone();
    let holder = shard.local_segments_for_test().await.unwrap();
    let active: Vec<_> = holder
        .read()
        .iter_original()
        .filter_map(|(_, locked)| {
            let segment = locked.read();
            (segment.available_point_count() > 0).then(|| locked.clone())
        })
        .collect();
    // The direct-index benchmark implementation accepts one Segment. Fail
    // explicitly if this artifact has a topology it cannot evaluate.
    assert_eq!(
        active.len(),
        1,
        "direct-index warmup requires exactly one active nonempty segment"
    );
    let segment = active[0].read();
    assert_eq!(segment.available_point_count(), N);
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("active segment is not LMI")
    };
    let nprobe = lmi
        .routing_state()
        .expect("trained LMI state missing")
        .nprobe();
    record(serde_json::json!({"phase":"topology","active_segments":1,
        "segment":segment.uuid.to_string(),"live_points":N,"nprobe":nprobe,
        "indexed_points":index.indexed_vector_count()}));
    drop(index);

    for (row, query) in warm.iter().enumerate() {
        let normalized = Distance::Cosine.preprocess_vector::<f32>(query.clone());
        let index = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow();
        let VectorIndexEnum::Lmi(lmi) = &*index else {
            unreachable!()
        };
        let buckets = lmi
            .routing_state()
            .unwrap()
            .router()
            .top_buckets(&normalized, nprobe)
            .unwrap();
        let (before_learned, before_fallback) = lmi.search_path_counts_for_test();
        drop(index);
        let began = Instant::now();
        record(serde_json::json!({"phase":"direct_start","query_row":row,
            "utc":chrono::Utc::now().to_rfc3339()}));
        let (ids, candidates) = search_one(&segment, Variant::Lmi, nprobe, query);
        let index = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow();
        let VectorIndexEnum::Lmi(lmi) = &*index else {
            unreachable!()
        };
        let (after_learned, after_fallback) = lmi.search_path_counts_for_test();
        drop(index);
        record(serde_json::json!({"phase":"warmup","query_row":row,
            "method":"matched_evaluator_manual_lmi_routing_and_candidate_scoring",
            "seconds":began.elapsed().as_secs_f64(),"selected_buckets":buckets,
            "candidate_count":candidates,"returned_ids_1_based":ids,
            "learned_delta":after_learned-before_learned,
            "plain_fallback_delta":after_fallback-before_fallback}));
    }
    let began = Instant::now();
    let normalized = Distance::Cosine.preprocess_vector::<f32>(measured.clone());
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        unreachable!()
    };
    let measured_buckets = lmi
        .routing_state()
        .unwrap()
        .router()
        .top_buckets(&normalized, nprobe)
        .unwrap();
    drop(index);
    record(serde_json::json!({"phase":"direct_start","query_row":20,
        "utc":chrono::Utc::now().to_rfc3339()}));
    let (direct_ids, candidates) = search_one(&segment, Variant::Lmi, nprobe, &measured);
    let direct_overlap_raw = direct_ids
        .iter()
        .filter(|id| gold.contains(&(**id as i32)))
        .count();
    let direct_overlap_zero_based = direct_ids
        .iter()
        .filter(|id| **id > 0 && gold.contains(&((**id - 1) as i32)))
        .count();
    record(serde_json::json!({"phase":"measured_direct","query_row":20,
        "method":"matched_evaluator_manual_lmi_routing_and_candidate_scoring",
        "seconds":began.elapsed().as_secs_f64(),"selected_buckets":measured_buckets,
        "candidate_count":candidates,
        "returned_ids_1_based":direct_ids,"overlap_gold_raw":direct_overlap_raw,
        "overlap_gold_if_zero_based":direct_overlap_zero_based,"ground_truth_raw":gold}));
    drop(segment);
    drop(holder);

    let shard_holder = shard.local_segments_for_test().await.unwrap();
    let segment = active[0].read();
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        unreachable!()
    };
    let before = lmi.search_path_counts_for_test();
    drop(index);
    drop(segment);
    drop(shard_holder);
    let began = Instant::now();
    record(
        serde_json::json!({"phase":"collection_start","query_row":20,
        "utc":chrono::Utc::now().to_rfc3339()}),
    );
    let result = collection
        .search(
            CoreSearchRequest {
                query: QueryEnum::Nearest(NamedQuery::default_dense(measured)),
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
        .await;
    let segment = active[0].read();
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        unreachable!()
    };
    let after = lmi.search_path_counts_for_test();
    let ids: Vec<u64> = result
        .as_ref()
        .map(|points| {
            points
                .iter()
                .map(|point| match point.id {
                    segment::types::ExtendedPointId::NumId(id) => id,
                    _ => panic!("unexpected point ID type"),
                })
                .collect()
        })
        .unwrap_or_default();
    let overlap_raw = ids
        .iter()
        .filter(|id| gold.contains(&(**id as i32)))
        .count();
    let overlap_zero_based = ids
        .iter()
        .filter(|id| **id > 0 && gold.contains(&((**id - 1) as i32)))
        .count();
    record(
        serde_json::json!({"phase":"measured_collection","query_row":20,
        "seconds":began.elapsed().as_secs_f64(),"query_error":result.err().map(|e| e.to_string()),
        "returned_ids_1_based":ids,"overlap_gold_raw":overlap_raw,
        "overlap_gold_if_zero_based":overlap_zero_based,
        "learned_delta":after.0-before.0,"plain_fallback_delta":after.1-before.1}),
    );
    drop(index);
    drop(segment);
    drop(active);
    drop(shard);
    collection.stop_gracefully().await;
}

/// Diagnose whether exact row-20 neighbors were routed into nprobe=4, and
/// retain scored candidate rankings without changing the index or query.
#[test]
#[ignore = "bounded candidate-coverage diagnosis on an isolated full-scale copy"]
fn sisap23_recovery_row20_candidate_coverage() {
    use common::types::DeferredBehavior;

    let path = checked_collection();
    config_is_compatible(&path);
    let root = PathBuf::from(std::env::var("SISAP23_OFFICIAL_QUERY_ROOT").unwrap());
    let mut query_file = File::open(root.join("queries_20_9999.f32le")).unwrap();
    let mut query_bytes = vec![0u8; DIM * 4];
    query_file.read_exact(&mut query_bytes).unwrap();
    let values: Vec<f32> = query_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let mut gold_file = File::open(root.join("gold_20_9999_top10.i32le")).unwrap();
    let mut gold_bytes = [0u8; 40];
    gold_file.read_exact(&mut gold_bytes).unwrap();
    let gold: Vec<u64> = gold_bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()) as u64)
        .collect();

    let segment_path = locate_segment(&path);
    let segment = load_index(&segment_path);
    assert_eq!(segment.available_point_count(), N);
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("active segment is not LMI")
    };
    let state = lmi.routing_state().expect("trained LMI state missing");
    let normalized = Distance::Cosine.preprocess_vector::<f32>(values.clone());
    let buckets = state
        .router()
        .top_buckets(&normalized, state.nprobe())
        .unwrap();
    let mut offsets: Vec<_> = buckets
        .iter()
        .flat_map(|bucket| state.postings().get(*bucket).unwrap().iter().copied())
        .collect();
    offsets.sort_unstable();
    offsets.dedup();
    assert_eq!(
        state.nprobe(),
        4,
        "fixed full-scale LMI configuration changed"
    );

    let tracker = segment.id_tracker.borrow();
    let gold_coverage: Vec<_> = gold
        .iter()
        .map(|&id| {
            assert!((1..=N as u64).contains(&id));
            let offset = tracker
                .internal_id_with_behavior(PointIdType::NumId(id), DeferredBehavior::VisibleOnly)
                .expect("ground-truth point is absent from the index segment");
            serde_json::json!({"id_1_based":id,"segment_offset":offset,
                "selected_candidate":offsets.binary_search(&offset).is_ok()})
        })
        .collect();
    let storage = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let query: QueryVector = values.into();
    let scored = score_candidates(
        &query,
        &offsets,
        10,
        &*tracker,
        &*storage,
        &VectorQueryContext::default(),
    )
    .unwrap();
    let ranked: Vec<_> = scored
        .iter()
        .map(|hit| {
            serde_json::json!({
                "id_1_based":match tracker.external_id(hit.idx).unwrap() {
                    PointIdType::NumId(id) => id,
                    _ => panic!("unexpected point ID type"),
                },
                "segment_offset":hit.idx,"cosine_score":hit.score
            })
        })
        .collect();
    assert_eq!(ranked.len(), 10);
    assert!(scored.windows(2).all(|pair| pair[0].score >= pair[1].score));
    let covered = gold_coverage
        .iter()
        .filter(|item| item["selected_candidate"] == true)
        .count();
    let report_path = PathBuf::from(std::env::var("SISAP23_CANDIDATE_REPORT").unwrap());
    let report = serde_json::json!({
        "format":"sisap23-row20-candidate-coverage-v1",
        "query_row":20,"segment":segment.uuid.to_string(),
        "active_points":N,"nprobe":state.nprobe(),"selected_buckets":buckets,
        "candidate_count":offsets.len(),"ground_truth_coverage":gold_coverage,
        "ground_truth_candidates_selected":covered,
        "ranked_candidate_top10":ranked,
        "scored_in_descending_order":true,
        "scientific_acceptance":false
    });
    write_json_new(&report_path, &report);
    println!("SISAP23_ROW20_CANDIDATE_COVERAGE {report}");
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
    if matches!(result, Ok(Ok(_))) {
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
