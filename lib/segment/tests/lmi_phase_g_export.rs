//! Read-only Phase G export through Qdrant's real segment and IdTracker APIs.
//! Run explicitly with LMI_G_SEGMENT, LMI_G_QUERIES_JSONL and LMI_G_OUTPUT.
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use common::types::DeferredBehavior;
use segment::data_types::vectors::DEFAULT_VECTOR_NAME;
use segment::id_tracker::IdTrackerRead;
use segment::index::VectorIndexEnum;
use segment::index::lmi_index::{CompactPostings, LMI_POSTINGS_FILE, LMI_ROUTER_FILE, MlpRouter};
use segment::segment_constructor::load_segment;
use segment::vector_storage::VectorStorageRead;

#[test]
#[ignore = "explicit read-only export of the persisted SISAP segment"]
fn export_real_id_mapping_and_router() {
    let dir = PathBuf::from(std::env::var("LMI_G_SEGMENT").expect("LMI_G_SEGMENT"));
    let raw = PathBuf::from(std::env::var("LMI_G_QUERIES_JSONL").expect("LMI_G_QUERIES_JSONL"));
    let out = PathBuf::from(std::env::var("LMI_G_OUTPUT").expect("LMI_G_OUTPUT"));
    std::fs::create_dir_all(&out).unwrap();
    let segment = load_segment(&dir, uuid::Uuid::nil(), None, &AtomicBool::new(false)).unwrap();
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected trained LMI");
    };
    assert!(lmi.state_path().is_some());
    let _routing = lmi.routing_state().expect("persisted learned router");
    let postings: CompactPostings =
        common::fs::read_bin(&dir.join("vector_index").join(LMI_POSTINGS_FILE)).unwrap();
    let (_, router): (Vec<u32>, Option<MlpRouter>) =
        common::fs::read_bin(&dir.join("vector_index").join(LMI_ROUTER_FILE)).unwrap();
    let router = router.expect("trained router");
    let count = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow()
        .total_vector_count();
    let mut labels = vec![u16::MAX; count];
    let mut assigned = 0usize;
    for (bucket, offsets) in postings.iter().enumerate() {
        for &offset in offsets {
            let label = labels
                .get_mut(offset as usize)
                .expect("posting offset in storage");
            assert_eq!(*label, u16::MAX, "duplicate posting offset");
            *label = bucket as u16;
            assigned += 1;
        }
    }
    assert_eq!(assigned, count, "full corpus posting coverage");
    let tracker = segment.id_tracker.borrow();
    let mut output = BufWriter::new(File::create(out.join("gold-buckets.jsonl")).unwrap());
    let mut rows = 0usize;
    for line in BufReader::new(File::open(&raw).unwrap()).lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let ids = row["gold_ids_1_based"].as_array().unwrap();
        assert_eq!(ids.len(), 10);
        let mut offsets = Vec::with_capacity(10);
        let mut buckets = Vec::with_capacity(10);
        for id in ids {
            let external = id.as_u64().unwrap();
            let offset = tracker
                .internal_id_with_behavior(external.into(), DeferredBehavior::VisibleOnly)
                .expect("official gold ID absent from live segment");
            offsets.push(offset);
            buckets.push(labels[offset as usize]);
        }
        let mut result_offsets = Vec::with_capacity(10);
        for id in row["result_ids_1_based"].as_array().unwrap() {
            let external = id.as_u64().unwrap();
            result_offsets.push(
                tracker
                    .internal_id_with_behavior(external.into(), DeferredBehavior::VisibleOnly)
                    .expect("returned ID absent from live segment"),
            );
        }
        serde_json::to_writer(
            &mut output,
            &serde_json::json!({
                "query_row": row["query_row"], "gold_ids": ids,
                "gold_offsets": offsets, "gold_buckets": buckets, "result_offsets": result_offsets
            }),
        )
        .unwrap();
        output.write_all(b"\n").unwrap();
        rows += 1;
    }
    output.flush().unwrap();
    assert_eq!(rows, 9980);
    std::fs::write(
        out.join("router.json"),
        serde_json::to_vec(&router).unwrap(),
    )
    .unwrap();
    std::fs::write(
        out.join("export-summary.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "segment": dir, "source_queries": raw, "measured_rows": rows,
            "points": count, "posting_count": assigned, "bucket_count": postings.len(),
            "mapping_method": "IdTrackerRead::internal_id_with_behavior(VisibleOnly)",
            "rebuild": false, "state_mutation": false
        }))
        .unwrap(),
    )
    .unwrap();
}
