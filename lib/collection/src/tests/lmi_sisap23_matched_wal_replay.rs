//! Opt-in WAL inspection and collection replay on a verified scratch copy only.
use super::*;
use shard::wal::{SerdeWal, WalRawRecord};
use std::collections::{BTreeMap, HashMap};

struct WalAudit {
    report: serde_json::Value,
    complete: bool,
}

fn source_vector(corpus: &mut File, id: u64) -> Vec<f32> {
    let mut raw = vec![0u8; DIM * 2];
    corpus
        .seek(std::io::SeekFrom::Start((id - 1) * (DIM * 2) as u64))
        .unwrap();
    corpus.read_exact(&mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect()
}

fn inspect_wal(collection_path: &Path, missing: &[u64], corpus_path: &Path) -> WalAudit {
    let config = CollectionConfigInternal::load(collection_path).unwrap();
    let wal_path = collection_path.join("0/wal");
    let raw_first_index: serde_json::Value =
        serde_json::from_slice(&fs::read(wal_path.join("first-index")).unwrap()).unwrap();
    let wal: SerdeWal<OperationWithClockTag> =
        SerdeWal::new(&wal_path, (&config.wal_config).into()).unwrap();
    inspect_wal_records(&wal, raw_first_index, missing, corpus_path, N)
}

fn inspect_wal_records(
    wal: &SerdeWal<OperationWithClockTag>,
    raw_first_index: serde_json::Value,
    missing: &[u64],
    corpus_path: &Path,
    expected_count: usize,
) -> WalAudit {
    let physical_start = wal.first_closed_index();
    let physical_end = physical_start + wal.len(true);
    let logical_start = wal.first_index();
    let logical_end = logical_start + wal.len(false);
    let missing_set: HashSet<u64> = missing.iter().copied().collect();
    let mut covered = HashSet::new();
    let mut bad_vectors = HashSet::new();
    let mut kinds = BTreeMap::<String, usize>::new();
    let mut bad_records = Vec::new();
    let mut read_error_count = 0usize;
    let mut unexpected_samples = Vec::new();
    let mut unexpected_count = 0usize;
    let mut upsert_points = 0usize;
    let mut records_read = 0usize;
    let mut min_upsert_id = None::<u64>;
    let mut max_upsert_id = None::<u64>;
    let mut corpus = File::open(corpus_path).unwrap();
    for entry in wal.read_range(logical_start..logical_end) {
        let (op_num, update) = match entry {
            Ok(v) => v,
            Err(err) => {
                read_error_count += 1;
                sample_push(&mut bad_records, err.to_string());
                continue;
            }
        };
        records_read += 1;
        match update.operation {
            CollectionUpdateOperations::PointOperation(PointOperations::UpsertPoints(
                PointInsertOperationsInternal::PointsList(points),
            )) => {
                *kinds.entry("upsert_points_list".into()).or_default() += 1;
                for point in points {
                    upsert_points += 1;
                    let PointIdType::NumId(id) = point.id else {
                        unexpected_count += 1;
                        sample_push(&mut unexpected_samples, format!("uuid at WAL {op_num}"));
                        continue;
                    };
                    min_upsert_id = Some(min_upsert_id.map_or(id, |v| v.min(id)));
                    max_upsert_id = Some(max_upsert_id.map_or(id, |v| v.max(id)));
                    if id == 0 || id > expected_count as u64 {
                        unexpected_count += 1;
                        sample_push(
                            &mut unexpected_samples,
                            format!("out-of-range ID {id} at WAL {op_num}"),
                        );
                    }
                    if missing_set.contains(&id) {
                        match point.vector {
                            VectorStructPersisted::Single(vector) if vector.len() == DIM => {
                                let expected = source_vector(&mut corpus, id);
                                if vector
                                    .iter()
                                    .zip(expected.iter())
                                    .all(|(a, b)| a.to_bits() == b.to_bits())
                                {
                                    covered.insert(id);
                                    bad_vectors.remove(&id);
                                } else {
                                    bad_vectors.insert(id);
                                    covered.remove(&id);
                                }
                            }
                            _ => {
                                bad_vectors.insert(id);
                                covered.remove(&id);
                            }
                        }
                    }
                }
            }
            other => {
                let name = other.operation_name().to_string();
                *kinds.entry(name.clone()).or_default() += 1;
                unexpected_count += 1;
                sample_push(&mut unexpected_samples, format!("{name} at WAL {op_num}"));
            }
        }
    }
    let uncovered: Vec<u64> = missing
        .iter()
        .copied()
        .filter(|id| !covered.contains(id))
        .collect();
    let complete = records_read as u64 == wal.len(false)
        && read_error_count == 0
        && unexpected_count == 0
        && bad_vectors.is_empty()
        && uncovered.is_empty();
    WalAudit {
        report: serde_json::json!({
            "raw_first_index":raw_first_index,
            "physical_range_exclusive":[physical_start,physical_end],
            "physical_records":wal.len(true),
            "logical_replay_range_exclusive":[logical_start,logical_end],
            "logical_records":wal.len(false),
            "last_index_api":wal.last_index(),
            "records_read":records_read,
            "record_types":kinds,
            "upsert_points":upsert_points,
            "upsert_id_range":[min_upsert_id,max_upsert_id],
            "missing_ids_expected":missing.len(),
            "missing_ids_with_matching_wal_vectors":covered.len(),
            "uncovered_count":uncovered.len(),
            "uncovered_samples":uncovered.iter().take(16).collect::<Vec<_>>(),
            "bad_vector_count":bad_vectors.len(),
            "bad_vector_samples":bad_vectors.iter().take(16).collect::<Vec<_>>(),
            "read_error_count":read_error_count,
            "read_error_samples":bad_records,
            "unexpected_operation_count":unexpected_count,
            "unexpected_operation_samples":unexpected_samples,
            "safe_for_replay":complete
        }),
        complete,
    }
}

struct SegmentAudit {
    report: serde_json::Value,
    missing: Vec<u64>,
    valid: bool,
}

fn audit_segment_collection(
    collection_path: &Path,
    corpus_path: &Path,
    recovered: &[u64],
) -> SegmentAudit {
    let segment_path = locate_segment(collection_path);
    let segment = load_index(&segment_path);
    let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
    let storage = data.vector_storage.borrow();
    let tracker = segment.id_tracker.borrow();
    let physical = storage.total_vector_count();
    let mut mappings = MappingAudit::new(N);
    let mut bad_reverse = 0usize;
    let mut missing_versions = 0usize;
    let mut deleted_references = 0usize;
    let mut absent_vectors = 0usize;
    let recovered_set: HashSet<u64> = recovered.iter().copied().collect();
    let mut recovered_offsets = HashMap::new();
    for (external, offset) in tracker.point_mappings().iter_from(None) {
        let id = match external {
            PointIdType::NumId(v) => v,
            PointIdType::Uuid(_) => 0,
        };
        mappings.observe(id, offset, physical);
        if tracker.external_id(offset) != Some(external) {
            bad_reverse += 1;
        }
        if tracker.internal_version(offset).is_none() {
            missing_versions += 1;
        }
        if (offset as usize) < physical {
            if storage.is_deleted_vector(offset) {
                deleted_references += 1;
            }
            if storage.get_vector_opt::<Random>(offset).is_none() {
                absent_vectors += 1;
            }
        }
        if recovered_set.contains(&id) {
            recovered_offsets.insert(id, offset);
        }
    }
    let missing: Vec<u64> = mappings
        .seen
        .iter()
        .enumerate()
        .filter_map(|(i, seen)| (!*seen).then_some((i + 1) as u64))
        .collect();
    let mut compared = 0usize;
    let mut mismatched = 0usize;
    let mut maximum_error = 0.0f32;
    let mut mismatch_samples = Vec::new();
    let mut corpus = File::open(corpus_path).unwrap();
    for id in recovered {
        let Some(&offset) = recovered_offsets.get(id) else {
            mismatched += 1;
            sample_push(&mut mismatch_samples, (*id, "missing ID".to_string()));
            continue;
        };
        let Some(stored) = storage.get_vector_opt::<Random>(offset) else {
            mismatched += 1;
            sample_push(&mut mismatch_samples, (*id, "missing vector".to_string()));
            continue;
        };
        let segment::data_types::vectors::VectorInternal::Dense(actual) = stored.to_owned() else {
            mismatched += 1;
            sample_push(&mut mismatch_samples, (*id, "not dense".to_string()));
            continue;
        };
        let source = source_vector(&mut corpus, *id);
        let expected: Vec<f32> = Distance::Cosine
            .preprocess_vector::<f32>(source)
            .iter()
            .map(|v| f16::from_f32(*v).to_f32())
            .collect();
        let error = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        maximum_error = maximum_error.max(error);
        compared += 1;
        if actual.len() != DIM || error > 0.002 {
            mismatched += 1;
            sample_push(&mut mismatch_samples, (*id, format!("max error {error}")));
        }
    }
    let available_vectors = storage.available_vector_count();
    let deleted_vectors = storage.deleted_vector_count();
    let total_points = tracker.total_point_count();
    let available_points = tracker.available_point_count();
    let deleted_points = tracker.deleted_point_count();
    let mapping_report = mappings.finish();
    let valid = physical == N
        && available_vectors == N
        && deleted_vectors == 0
        && total_points == N
        && available_points == N
        && deleted_points == 0
        && mappings.iterated == N
        && missing.is_empty()
        && mappings.duplicate_ids == 0
        && mappings.invalid_ids == 0
        && mappings.missing_slots == 0
        && bad_reverse == 0
        && missing_versions == 0
        && deleted_references == 0
        && absent_vectors == 0
        && mismatched == 0
        && compared == recovered.len();
    SegmentAudit {
        report: serde_json::json!({
            "segment_path":segment_path,
            "total_vectors":physical,"available_vectors":available_vectors,"deleted_vectors":deleted_vectors,
            "total_points":total_points,"available_points":available_points,"deleted_points":deleted_points,
            "mapping_audit":mapping_report,
            "reverse_mapping_failures":bad_reverse,"missing_versions":missing_versions,
            "deleted_vector_references":deleted_references,"absent_vectors":absent_vectors,
            "restored_vectors_expected":recovered.len(),"restored_vectors_compared":compared,
            "restored_vector_mismatches":mismatched,"restored_vector_max_abs_error":maximum_error,
            "restored_vector_mismatch_samples":mismatch_samples,
            "complete_and_valid":valid
        }),
        missing,
        valid,
    }
}

/// Independent one-query reference streamed from the original prepared Float16 corpus.
fn independent_plain_top10(corpus_path: &Path, query: &[f32], count: usize) -> Vec<u64> {
    let query = Distance::Cosine.preprocess_vector::<f32>(query.to_vec());
    let mut corpus = BufReader::new(File::open(corpus_path).unwrap());
    let mut raw = vec![0u8; DIM * 2];
    let mut values = vec![0f32; DIM];
    let mut top = Vec::<(f32, u64)>::new();
    for id in 1..=count as u64 {
        corpus.read_exact(&mut raw).unwrap();
        let mut norm_square = 0f32;
        for (i, two) in raw.chunks_exact(2).enumerate() {
            let value = f16::from_bits(u16::from_le_bytes([two[0], two[1]])).to_f32();
            values[i] = value;
            norm_square += value * value;
        }
        let norm = norm_square.sqrt();
        let score = values
            .iter()
            .zip(&query)
            .map(|(v, q)| {
                let normalized = if norm > 0.0 { *v / norm } else { *v };
                f16::from_f32(normalized).to_f32() * q
            })
            .sum::<f32>();
        if top.len() < 10 || score > top.last().unwrap().0 {
            top.push((score, id));
            top.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            top.truncate(10);
        }
    }
    top.into_iter().map(|(_, id)| id).collect()
}

#[test]
fn sisap23_wal_inspection_detects_covered_and_truncated_ids() {
    let temp = tempfile::tempdir().unwrap();
    let corpus_path = temp.path().join("small.f16le");
    let mut corpus = File::create(&corpus_path).unwrap();
    for id in 1..=2 {
        for dim in 0..DIM {
            let value = if dim == id - 1 { 1.0f32 } else { 0.0 };
            corpus
                .write_all(&f16::from_f32(value).to_bits().to_le_bytes())
                .unwrap();
        }
    }
    drop(corpus);
    let wal_dir = temp.path().join("wal");
    fs::create_dir(&wal_dir).unwrap();
    let mut wal: SerdeWal<OperationWithClockTag> =
        SerdeWal::new(&wal_dir, (&crate::config::WalConfig::default()).into()).unwrap();
    let mut last = 0;
    for id in 1..=2u64 {
        let mut vector = vec![0f32; DIM];
        vector[(id - 1) as usize] = 1.0;
        let point = PointStructPersisted {
            id: id.into(),
            vector: VectorStructPersisted::Single(vector),
            payload: None,
        };
        let op = OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
            PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(vec![point])),
        ));
        last = wal.write(&WalRawRecord::new(&op).unwrap()).unwrap();
    }
    wal.flush().unwrap();
    let covered = inspect_wal_records(&wal, serde_json::Value::Null, &[1, 2], &corpus_path, 2);
    assert!(covered.complete);
    assert_eq!(covered.report["logical_records"], 2);
    wal.ack(last).unwrap();
    let truncated = inspect_wal_records(&wal, serde_json::Value::Null, &[1, 2], &corpus_path, 2);
    assert!(!truncated.complete);
    assert_eq!(truncated.report["uncovered_count"], 1);
}

#[test]
fn sisap23_wal_independent_reference_uses_float16_corpus() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("small.f16le");
    let mut output = File::create(&path).unwrap();
    for (x, y) in [(1.0f32, 0.0f32), (0.0, 1.0), (-1.0, 0.0)] {
        for i in 0..DIM {
            let value = if i == 0 {
                x
            } else if i == 1 {
                y
            } else {
                0.0
            };
            output
                .write_all(&f16::from_f32(value).to_bits().to_le_bytes())
                .unwrap();
        }
    }
    drop(output);
    let mut query = vec![0f32; DIM];
    query[0] = 1.0;
    assert_eq!(independent_plain_top10(&path, &query, 3), vec![1, 2, 3]);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit scratch-copy WAL inspection and optional normal collection replay"]
async fn sisap23_matched_inspect_or_replay_plain_wal() {
    let mode = std::env::var("SISAP23_WAL_MODE").unwrap();
    assert!(mode == "inspect" || mode == "replay");
    let scratch = PathBuf::from(std::env::var("SCRATCHDIR").unwrap());
    let collection_path = PathBuf::from(std::env::var("SISAP23_WAL_COLLECTION_PATH").unwrap());
    let inspection_path = PathBuf::from(std::env::var("SISAP23_WAL_INSPECTION_PATH").unwrap());
    assert!(collection_path.starts_with(&scratch) && collection_path != scratch);
    assert!(inspection_path.starts_with(&scratch) && inspection_path != scratch);
    assert_ne!(collection_path, inspection_path);
    let pre_path = PathBuf::from(std::env::var("SISAP23_WAL_PRE_REPORT").unwrap());
    let post_path = PathBuf::from(std::env::var("SISAP23_WAL_POST_REPORT").unwrap());
    assert!(diagnostic_destination_is_safe(&pre_path, &collection_path));
    assert!(diagnostic_destination_is_safe(&post_path, &collection_path));
    assert!(!pre_path.to_string_lossy().contains("build.complete"));
    assert!(!post_path.to_string_lossy().contains("build.complete"));
    let corpus_path = root().join("corpus.f16le");
    let initial = audit_segment_collection(&inspection_path, &corpus_path, &[]);
    let wal = inspect_wal(&inspection_path, &initial.missing, &corpus_path);
    let pre = serde_json::json!({
        "format":"sisap23-plain-wal-pre-v1",
        "source_commit":std::env::var("SISAP23_WAL_COMMIT").unwrap(),
        "mode":mode,
        "initial_segment":initial.report,
        "wal":wal.report,
        "original_collection_never_loaded":true,
        "collection_loaded_from_verified_scratch_copy":true
    });
    write_json_new(&pre_path, &pre);
    println!("sisap23_wal_pre_report={}", pre_path.display());
    if mode == "inspect" {
        return;
    }
    assert_eq!(
        initial.missing.len(),
        1679,
        "unexpected initial missing-ID count"
    );
    assert!(
        wal.complete,
        "WAL inspection did not prove all missing vectors are present"
    );
    let config = CollectionConfigInternal::load(&collection_path).unwrap();
    assert!(
        !config
            .optimizer_config
            .prevent_unoptimized
            .unwrap_or_default(),
        "async WAL tail would require additional completion handling"
    );
    let snapshots = scratch.join("wal-replay-snapshots");
    fs::create_dir(&snapshots).unwrap();
    let pause = serde_json::from_value(serde_json::json!({"max_optimization_threads":0})).unwrap();
    let collection = Collection::load(
        "sisap23_matched".to_string(),
        1,
        &collection_path,
        &snapshots,
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
    // LocalShard::load_from_wal synchronously replays the full logical range and
    // forces a synchronous segment flush when prevent_unoptimized is false.
    let visible = collection.info(&ShardSelectorInternal::All).await.unwrap();
    let visible_points = visible.points_count;
    collection.stop_gracefully().await;
    drop(collection);
    let persisted = audit_segment_collection(&collection_path, &corpus_path, &initial.missing);
    let reopened = Collection::load(
        "sisap23_matched".to_string(),
        1,
        &collection_path,
        &snapshots,
        Arc::new(SharedStorageConfig::default()),
        ChannelService::default(),
        dummy_on_replica_failure(),
        dummy_request_shard_transfer(),
        dummy_abort_shard_transfer(),
        None,
        None,
        ResourceBudget::default(),
        Some(serde_json::from_value(serde_json::json!({"max_optimization_threads":0})).unwrap()),
    )
    .await;
    let reopened_points = reopened
        .info(&ShardSelectorInternal::All)
        .await
        .unwrap()
        .points_count;
    reopened.stop_gracefully().await;
    drop(reopened);
    let restarted = audit_segment_collection(&collection_path, &corpus_path, &initial.missing);
    let count_and_mapping_valid = visible_points == Some(N)
        && reopened_points == Some(N)
        && persisted.valid
        && restarted.valid;
    let reference = if count_and_mapping_valid {
        let query = read_f32(&root().join("queries_warmup.f32le"), 20).remove(0);
        let segment = load_index(&locate_segment(&collection_path));
        let (native, _) = search_one(&segment, Variant::Plain, 0, &query);
        drop(segment);
        let independent = independent_plain_top10(&corpus_path, &query, N);
        let overlap = native.iter().filter(|id| independent.contains(id)).count();
        serde_json::json!({
            "query_row":0,"native_ids_1_based":native,"independent_ids_1_based":independent,
            "top10_set_overlap":overlap,
            "reference_method":"stream prepared Float16 corpus; normalize, round to Float16, dot with normalized warmup query",
            "match":overlap == 10
        })
    } else {
        serde_json::json!({"status":"skipped","reason":"mapping or persistence invariants failed"})
    };
    let valid = count_and_mapping_valid && reference["match"] == true;
    let post = serde_json::json!({
        "format":"sisap23-plain-wal-post-v1",
        "visible_points_after_normal_replay":visible_points,
        "persisted_after_first_shutdown":persisted.report,
        "visible_points_after_restart":reopened_points,
        "persisted_after_restart":restarted.report,
        "independent_exact_search_reference":reference,
        "accepted":valid,
        "completion_marker_created":false,
        "original_hash_verification":"PBS exit trap compares before and after hashes"
    });
    write_json_new(&post_path, &post);
    println!("sisap23_wal_post_report={}", post_path.display());
    assert!(
        valid,
        "scratch-copy WAL replay did not satisfy full consistency invariants"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit persistent staging collection reopen; requires verified WAL replay"]
async fn sisap23_matched_verify_persistent_plain_staging() {
    assert_eq!(
        std::env::var("SISAP23_PERSISTENT_VERIFY").as_deref(),
        Ok("1")
    );
    let staging = PathBuf::from(std::env::var("SISAP23_PERSISTENT_STAGING_COLLECTION").unwrap());
    let inspection = PathBuf::from(std::env::var("SISAP23_WAL_INSPECTION_PATH").unwrap());
    let report_path = PathBuf::from(std::env::var("SISAP23_PERSISTENT_REPORT").unwrap());
    assert!(staging.is_absolute() && staging.ends_with("collection"));
    assert!(staging.starts_with(root().join("recovered-generations/.staging")));
    assert!(inspection.starts_with(PathBuf::from(std::env::var("SCRATCHDIR").unwrap())));
    assert!(diagnostic_destination_is_safe(&report_path, &staging));
    let post: serde_json::Value = serde_json::from_slice(
        &fs::read(std::env::var("SISAP23_WAL_POST_REPORT").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(post["accepted"], true, "scratch replay must have passed");
    let corpus = root().join("corpus.f16le");
    let initial = audit_segment_collection(&inspection, &corpus, &[]);
    assert_eq!(initial.missing.len(), 1679);
    let snapshots = staging.parent().unwrap().join("verification-snapshots");
    fs::create_dir(&snapshots).unwrap();
    let index_before = hash_index_tree(&locate_segment(&staging));
    reopen_for_verification(&staging, &snapshots, N).await;
    let first = audit_segment_collection(&staging, &corpus, &initial.missing);
    assert!(
        first.valid,
        "persisted staging failed complete mapping/vector audit"
    );
    reopen_for_verification(&staging, &snapshots, N).await;
    let second = audit_segment_collection(&staging, &corpus, &initial.missing);
    assert!(
        second.valid,
        "persistent staging changed after a second restart"
    );
    let index_after = hash_index_tree(&locate_segment(&staging));
    assert_eq!(
        index_before, index_after,
        "Plain index changed during verification"
    );
    let query = read_f32(&root().join("queries_warmup.f32le"), 20).remove(0);
    let segment = load_index(&locate_segment(&staging));
    let (native, _) = search_one(&segment, Variant::Plain, 0, &query);
    drop(segment);
    let reference = independent_plain_top10(&corpus, &query, N);
    let overlap = native.iter().filter(|id| reference.contains(id)).count();
    let accepted = overlap == 10;
    let report = serde_json::json!({
        "format":"sisap23-persistent-plain-verification-v1",
        "source_replay_report":std::env::var("SISAP23_WAL_POST_REPORT").unwrap(),
        "first_reopen":first.report,"second_reopen":second.report,
        "recovered_vectors_compared":initial.missing.len(),
        "independent_top10_overlap":overlap,
        "index_files_unchanged":true,
        "accepted":accepted
    });
    write_json_new(&report_path, &report);
    assert!(
        accepted,
        "persistent staging exact-search comparison failed"
    );
}
