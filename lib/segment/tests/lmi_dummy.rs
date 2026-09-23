use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::index::{VectorIndex, VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::segment_constructor::load_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{Condition, Distance, Filter, HnswGlobalConfig, Indexes, WithPayload};

fn assert_lmi(segment: &Segment) {
    assert!(matches!(
        segment.config().vector_data[DEFAULT_VECTOR_NAME].index,
        Indexes::Lmi {}
    ));
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    assert!(matches!(&*index, VectorIndexEnum::Lmi(_)));
    assert!(index.is_index());
    assert!(!index.is_on_disk());
    assert!(index.as_hnsw().is_none());
    assert!(index.files().is_empty());
    assert!(index.immutable_files().is_empty());
    index.populate().unwrap();
    index.clear_cache().unwrap();
}

fn assert_search_matches_plain(lmi: &Segment, plain: &Segment) {
    let excluded = Filter::new_must_not(Condition::HasId(
        [1.into()]
            .into_iter()
            .collect::<ahash::AHashSet<_>>()
            .into(),
    ));
    let empty = Filter::new_must(Condition::HasId(
        [99.into()]
            .into_iter()
            .collect::<ahash::AHashSet<_>>()
            .into(),
    ));
    for (filter, expected_ids) in [
        (None, vec![1, 2, 3, 4]),
        (Some(&excluded), vec![2, 3, 4]),
        (Some(&empty), vec![]),
    ] {
        let query = vec![1.0, 0.0].into();
        for top in [1, 10] {
            let search = |segment: &Segment| {
                segment
                    .search(
                        DEFAULT_VECTOR_NAME,
                        &query,
                        &WithPayload::default(),
                        &false.into(),
                        filter,
                        top,
                        None,
                    )
                    .unwrap()
            };
            let actual = search(lmi);
            let reference = search(plain);
            assert_eq!(actual, reference);
            let ids: Vec<_> = actual.iter().map(|point| point.id).collect();
            let expected: Vec<_> = expected_ids
                .iter()
                .take(top)
                .map(|id| (*id as u64).into())
                .collect();
            assert_eq!(ids, expected);
        }
    }
}

#[test]
fn lmi_dummy_build_reopen_and_search_matches_plain() {
    let segments = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let stopped = AtomicBool::new(false);
    let hw_counter = HardwareCounterCell::new();
    let mut plain = build_simple_segment(segments.path(), 2, Distance::Dot).unwrap();
    for (id, vector) in [
        (1, [1.0, 0.0]),
        (2, [0.75, 0.25]),
        (3, [0.25, 0.75]),
        (4, [-1.0, 0.0]),
    ] {
        plain
            .upsert_point(id, id.into(), only_default_vector(&vector), &hw_counter)
            .unwrap();
    }

    let mut config = plain.config().clone();
    config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .index = Indexes::Lmi {};
    let mut builder =
        SegmentBuilder::new(staging.path(), &config, &HnswGlobalConfig::default()).unwrap();
    builder.update(&[&plain], &stopped, &hw_counter).unwrap();
    let lmi = builder.build_for_test(segments.path());
    assert_eq!(lmi.available_point_count(), 4);
    assert_lmi(&lmi);
    assert_search_matches_plain(&lmi, &plain);

    let path = lmi.segment_path.clone();
    let uuid = lmi.uuid;
    drop(lmi);
    let reopened = load_segment(&path, uuid, None, &stopped).unwrap();
    assert_eq!(reopened.available_point_count(), 4);
    assert_lmi(&reopened);
    assert_search_matches_plain(&reopened, &plain);
}
