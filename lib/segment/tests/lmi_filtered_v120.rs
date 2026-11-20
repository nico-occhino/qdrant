//! Adaptive filtered search over a static learned partition.
//!
//! The fixture is a real spherical IVF partition (the centroid router of the
//! thesis comparison): 32 clustered 16-d cosine clusters of 200 points, a
//! one-layer router whose rows are the unit centroids, Voronoi postings and
//! `nprobe = 2`, so the probe volume (break-even cardinality) is 400 points.
//! Two indexed integer payloads drive the filters: `r` is independent of the
//! clusters (selectivity sweep), `far` marks four clusters away from the queries.

use std::collections::HashSet;
use std::str::FromStr;
use std::sync::atomic::AtomicBool;

use common::flags::FeatureFlags;
use common::types::{DeferredBehavior, ScoredPointOffset};
use common::universal_io::{MmapFile, MmapFs};
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::NonAppendableSegmentEntry;
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::id_tracker::{IdTracker, IdTrackerRead};
use segment::index::lmi_index::{
    CompactPostings, FilteredPlan, LinearLayer, LmiConfig, LmiRoutingState, LmiStateMetadata,
    MlpRouter, RouterLayer, save_fixture_state,
};
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::json_path::JsonPath;
use segment::segment::Segment;
use segment::segment::read_only::ReadOnlySegment;
use segment::segment_constructor::load_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Condition, Distance, FieldCondition, Filter, HnswGlobalConfig, Indexes, Match, Payload,
    PayloadSchemaType, Range, SearchParams, ValueVariants,
};
use segment::vector_storage::VectorStorageRead;

const DIM: usize = 16;
const BUCKETS: usize = 32;
const PER_BUCKET: usize = 200;
const POINTS: usize = BUCKETS * PER_BUCKET;
const NPROBE: usize = 2;
const PROBE_VOLUME: usize = POINTS * NPROBE / BUCKETS;
const FAR_CLUSTERS: [usize; 4] = [28, 29, 30, 31];
const TOP: usize = 10;

/// Deterministic xorshift64* so the fixture does not depend on a RNG crate API.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Fixture {
    _root: tempfile::TempDir,
    _staging: tempfile::TempDir,
    lmi: Segment,
    centers: Vec<Vec<f32>>,
}

fn r_value(i: usize) -> i64 {
    ((i * 7919) % 1000) as i64
}

fn build_fixture() -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let centers: Vec<Vec<f32>> = (0..BUCKETS)
        .map(|_| normalized((0..DIM).map(|_| rng.uniform()).collect()))
        .collect();
    let mut plain = build_simple_segment(root.path(), DIM, Distance::Cosine).unwrap();
    let mut vectors = Vec::with_capacity(POINTS);
    for i in 0..POINTS {
        let center = &centers[i % BUCKETS];
        let raw: Vec<f32> = center.iter().map(|c| c + 0.35 * rng.uniform()).collect();
        let id = (i as u64 + 1).into();
        plain
            .upsert_point(i as u64 + 1, id, only_default_vector(&raw))
            .unwrap();
        let far = i64::from(FAR_CLUSTERS.contains(&(i % BUCKETS)));
        let payload: Payload =
            serde_json::from_value(serde_json::json!({ "r": r_value(i), "far": far })).unwrap();
        plain
            .set_full_payload(POINTS as u64 + i as u64 + 1, id, &payload)
            .unwrap();
        vectors.push(normalized(raw));
    }
    for (op, key) in [(3 * POINTS as u64, "r"), (3 * POINTS as u64 + 1, "far")] {
        plain
            .create_field_index(
                op,
                &JsonPath::from_str(key).unwrap(),
                Some(&PayloadSchemaType::Integer.into()),
            )
            .unwrap();
    }

    let mut cfg = plain.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::Lmi {};
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder.update(&[&plain], &AtomicBool::new(false)).unwrap();
    let lmi = builder.build_for_test(root.path());

    // Spherical IVF: centroid router, Voronoi postings over segment offsets.
    let config = LmiConfig {
        n_buckets: BUCKETS,
        sample_size: POINTS,
        hidden_dim: DIM,
        nprobe: NPROBE,
        ..Default::default()
    };
    let mut buckets = vec![Vec::new(); BUCKETS];
    {
        let tracker = lmi.id_tracker.borrow();
        for (i, vector) in vectors.iter().enumerate() {
            let offset = tracker
                .internal_id_with_behavior((i as u64 + 1).into(), DeferredBehavior::WithDeferred)
                .unwrap();
            let best = (0..BUCKETS)
                .max_by(|&a, &b| dot(&centers[a], vector).total_cmp(&dot(&centers[b], vector)))
                .unwrap();
            buckets[best].push(offset);
        }
    }
    let router = MlpRouter {
        layers: vec![RouterLayer::Linear(LinearLayer {
            in_features: DIM,
            out_features: BUCKETS,
            weights: centers.iter().flatten().copied().collect(),
            bias: vec![0.0; BUCKETS],
        })],
    };
    let routing = LmiRoutingState::from_compact(
        router,
        CompactPostings::from_buckets(buckets).unwrap(),
        NPROBE,
    )
    .unwrap();
    let vector_config = lmi.config().vector_data[DEFAULT_VECTOR_NAME].clone();
    let storage = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
    let metadata = LmiStateMetadata::new(
        config,
        &vector_config,
        storage.datatype(),
        storage.total_vector_count(),
    );
    drop(storage);
    save_fixture_state(
        &lmi.segment_path.join("vector_index"),
        &metadata,
        &routing,
        &vector_config,
    )
    .unwrap();
    let mut state = Segment::load_state(&lmi.segment_path).unwrap();
    state
        .config
        .vector_data
        .get_mut(DEFAULT_VECTOR_NAME)
        .unwrap()
        .index = Indexes::LmiTrained(config);
    Segment::save_state(&state, &lmi.segment_path).unwrap();
    let (path, uuid) = (lmi.segment_path.clone(), lmi.uuid);
    drop(lmi);
    let lmi = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
    Fixture {
        _root: root,
        _staging: staging,
        lmi,
        centers,
    }
}

fn r_below(limit: i64) -> Filter {
    Filter::new_must(Condition::Field(FieldCondition::new_range(
        JsonPath::new("r"),
        Range {
            lt: Some((limit as f64).into()),
            ..Default::default()
        },
    )))
}

fn far_filter() -> Filter {
    Filter::new_must(Condition::Field(FieldCondition::new_match(
        JsonPath::new("far"),
        Match::new_value(ValueVariants::Integer(1)),
    )))
}

/// Queries near the first 28 cluster centres (never the `far` ones).
fn queries(fixture: &Fixture, count: usize) -> Vec<QueryVector> {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    (0..count)
        .map(|q| {
            let center = &fixture.centers[q % 28];
            let v: Vec<f32> = center.iter().map(|c| c + 0.35 * rng.uniform()).collect();
            v.into()
        })
        .collect()
}

fn search(
    segment: &Segment,
    query: &QueryVector,
    filter: Option<&Filter>,
    params: Option<&SearchParams>,
) -> Vec<ScoredPointOffset> {
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(
            &[query],
            filter,
            TOP,
            params,
            &VectorQueryContext::default(),
        )
        .unwrap()
        .remove(0)
}

fn exact(
    segment: &Segment,
    query: &QueryVector,
    filter: Option<&Filter>,
) -> Vec<ScoredPointOffset> {
    let params = SearchParams {
        exact: true,
        ..Default::default()
    };
    search(segment, query, filter, Some(&params))
}

fn recall(found: &[ScoredPointOffset], gold: &[ScoredPointOffset]) -> f64 {
    if gold.is_empty() {
        return 1.0;
    }
    let gold: HashSet<_> = gold.iter().map(|p| p.idx).collect();
    found.iter().filter(|p| gold.contains(&p.idx)).count() as f64 / gold.len() as f64
}

fn external(segment: &Segment, offset: u32) -> usize {
    match segment.id_tracker.borrow().external_id(offset).unwrap() {
        segment::types::ExtendedPointId::NumId(n) => n as usize - 1,
        other => panic!("unexpected id {other:?}"),
    }
}

fn plan_counts(segment: &Segment) -> (u64, u64) {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected LMI index")
    };
    lmi.filtered_plan_counts_for_test()
}

fn plan(segment: &Segment, filter: &Filter) -> segment::index::lmi_index::FilteredPlanDecision {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected LMI index")
    };
    lmi.filtered_plan(filter).unwrap().unwrap()
}

/// Selectivity sweep: the filter is always enforced, the result always fills
/// `min(top, matching)`, the plan follows the probe volume, and exact plans
/// return exactly the exact-search answer. Recall is printed for the report.
#[test]
fn filtered_search_enforces_filter_and_plans_by_cardinality() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering"));
    let fixture = build_fixture();
    let lmi = &fixture.lmi;
    assert_eq!(PROBE_VOLUME, 400);
    let queries = queries(&fixture, 112);

    let unfiltered: f64 = queries
        .iter()
        .map(|q| recall(&search(lmi, q, None, None), &exact(lmi, q, None)))
        .sum::<f64>()
        / queries.len() as f64;
    eprintln!(
        "selectivity  matching  plan              sampled  recall@10  (unfiltered recall@10 {unfiltered:.3})"
    );

    // r < limit matches limit/1000 of the points, independently of the clusters.
    for limit in [1000_i64, 500, 100, 50, 10, 1] {
        let filter = r_below(limit);
        let matching = (0..POINTS).filter(|&i| r_value(i) < limit).count();
        let decision = plan(lmi, &filter);
        let before = plan_counts(lmi);
        let mut total = 0.0;
        for q in &queries {
            let found = search(lmi, q, Some(&filter), None);
            let gold = exact(lmi, q, Some(&filter));
            assert_eq!(
                found.len(),
                TOP.min(matching),
                "result must fill min(top, matching)"
            );
            for hit in &found {
                assert!(
                    r_value(external(lmi, hit.idx)) < limit,
                    "returned point violates the filter"
                );
            }
            if decision.plan == FilteredPlan::ExactMatches {
                assert_eq!(
                    found, gold,
                    "the exact plan must equal exact filtered search"
                );
            }
            total += recall(&found, &gold);
        }
        let after = plan_counts(lmi);
        let served = match decision.plan {
            FilteredPlan::ExactMatches => after.0 - before.0,
            FilteredPlan::RoutedIntersect => after.1 - before.1,
        };
        assert_eq!(
            served,
            queries.len() as u64,
            "every query must take the decided plan"
        );
        let mean = total / queries.len() as f64;
        eprintln!(
            "{:>10.1}%  {matching:>8}  {:<16}  {:<7}  {mean:.3}",
            limit as f64 / 10.0,
            format!("{:?}", decision.plan),
            decision.sampled,
        );
        // Clear-cut cases on either side of the 400-point break-even.
        if matching >= 2 * PROBE_VOLUME {
            assert_eq!(
                decision.plan,
                FilteredPlan::RoutedIntersect,
                "limit {limit}"
            );
            // Independent filters keep roughly the unfiltered routing quality.
            assert!(
                mean >= unfiltered - 0.15,
                "limit {limit}: recall {mean:.3} vs {unfiltered:.3}"
            );
        }
        if matching <= PROBE_VOLUME / 2 {
            assert_eq!(decision.plan, FilteredPlan::ExactMatches, "limit {limit}");
            assert!((mean - 1.0).abs() < 1e-12);
        }
    }
}

/// All matching points sit in four clusters the queries never route to first:
/// the configured probe finds no match, so the routed plan must widen, and with
/// widening the answer equals exact filtered search.
#[test]
fn routed_filter_widens_when_matches_sit_in_unprobed_buckets() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering"));
    let fixture = build_fixture();
    let lmi = &fixture.lmi;
    let filter = far_filter();
    let decision = plan(lmi, &filter);
    assert_eq!(decision.plan, FilteredPlan::RoutedIntersect, "{decision:?}");
    let widenings = |segment: &Segment| {
        let index = segment.vector_data[DEFAULT_VECTOR_NAME]
            .vector_index
            .borrow();
        let VectorIndexEnum::Lmi(lmi) = &*index else {
            panic!("expected LMI index")
        };
        lmi.underfill_widenings_for_test()
    };
    let before = widenings(lmi);
    // A query at a cluster centre whose `NPROBE` nearest centroids are not
    // `far`, so the configured probe holds no matching posting.
    let start = (0..BUCKETS)
        .find(|&c| {
            let mut order: Vec<usize> = (0..BUCKETS).collect();
            order.sort_by(|&a, &b| {
                dot(&fixture.centers[b], &fixture.centers[c])
                    .total_cmp(&dot(&fixture.centers[a], &fixture.centers[c]))
            });
            order[..NPROBE].iter().all(|b| !FAR_CLUSTERS.contains(b))
        })
        .expect("some cluster routes away from the far clusters");
    let query: QueryVector = fixture.centers[start].clone().into();
    let found = search(lmi, &query, Some(&filter), None);
    assert_eq!(found.len(), TOP);
    assert!(widenings(lmi) > before, "the probe must widen");
    for hit in &found {
        assert!(FAR_CLUSTERS.contains(&(external(lmi, hit.idx) % BUCKETS)));
    }
    assert_eq!(found, exact(lmi, &query, Some(&filter)));
}

/// The read-only open path serves the same plans and the same answers.
#[test]
fn read_only_filtered_search_matches_mutable() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering"));
    let fixture = build_fixture();
    let lmi = &fixture.lmi;
    let read_only =
        ReadOnlySegment::<MmapFile>::open(&MmapFs, &lmi.segment_path, lmi.uuid, None, None)
            .unwrap();
    let filters = [r_below(1000), r_below(500), r_below(10), far_filter()];
    for query in queries(&fixture, 20) {
        for filter in &filters {
            let mutable = search(lmi, &query, Some(filter), None);
            let shared = read_only.vector_data[DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow()
                .search(
                    &[&query],
                    Some(filter),
                    TOP,
                    None,
                    &VectorQueryContext::default(),
                )
                .unwrap()
                .remove(0);
            assert_eq!(mutable, shared);
        }
    }
}

/// A deleted point never comes back through the routed filtered plan, and the
/// result still fills `top`.
#[test]
fn deleted_point_is_excluded_from_routed_filtered_results() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering"));
    let fixture = build_fixture();
    let lmi = &fixture.lmi;
    let filter = r_below(500);
    assert_eq!(plan(lmi, &filter).plan, FilteredPlan::RoutedIntersect);
    let query = queries(&fixture, 1).remove(0);
    let first = search(lmi, &query, Some(&filter), None);
    let victim = first[0].idx;
    let external_id = lmi.id_tracker.borrow().external_id(victim).unwrap();
    lmi.id_tracker.borrow_mut().drop(external_id, None).unwrap();
    let second = search(lmi, &query, Some(&filter), None);
    assert_eq!(second.len(), TOP);
    assert!(second.iter().all(|p| p.idx != victim));
    // Same probe minus the deleted offset: the old ranks 2..=10 move up by one.
    assert_eq!(second[..TOP - 1], first[1..]);
}

/// FNV-1a 64: a stable, dependency-free digest for regression fingerprints.
fn fnv(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

/// Outputs (offsets and exact score bits) and plan decisions of the Baseline
/// planner, recorded at `a89ab1fd3` before the probing-policy changes. Every
/// later change must leave them bit-identical, on both the mutable and the
/// read-only index paths.
const BASELINE_DIGESTS: &[(&str, u64)] = &[
    ("none/top1", 0xc7873ae1f5c26a0e),
    ("none/top10", 0x2a7337b25d960be4),
    ("none/top50", 0x2b3df395f174eb4a),
    ("r<1000/top1", 0x929d6f10ffdafc64),
    ("r<1000/top10", 0x1942904b989cbe02),
    ("r<1000/top50", 0xc0e1f56dee3f7a68),
    ("r<500/top1", 0x785c7f9cd1b97f12),
    ("r<500/top10", 0xd658d4bc31520f82),
    ("r<500/top50", 0x6e14ce767f98e63c),
    ("r<100/top1", 0x3e87d7874dd14483),
    ("r<100/top10", 0x37925614f755d8b2),
    ("r<100/top50", 0x518705b630b1095a),
    ("r<50/top1", 0xe4e3080a6d582f54),
    ("r<50/top10", 0xb603f124c06e3a39),
    ("r<50/top50", 0xc34353a384545ceb),
    ("r<10/top1", 0x8b2f407ff0cf27f1),
    ("r<10/top10", 0x74ffdc8d4a8e88b5),
    ("r<10/top50", 0x33dfe4af28089d8e),
    ("r<1/top1", 0x48ceddf22d6fc721),
    ("r<1/top10", 0x099b0f59e52224c8),
    ("r<1/top50", 0x099b0f59e52224c8),
    ("far/top1", 0xa78b29d0c5b5792d),
    ("far/top10", 0xabbf16fa329ad737),
    ("far/top50", 0xfabe27084e02400d),
    ("plans_exact", 0x0000000000000510),
    ("plans_routed", 0x00000000000006c0),
    ("widenings", 0x00000000000002f9),
];

#[test]
fn baseline_planner_outputs_are_unchanged() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering"));
    let fixture = build_fixture();
    let lmi = &fixture.lmi;
    let read_only =
        ReadOnlySegment::<MmapFile>::open(&MmapFs, &lmi.segment_path, lmi.uuid, None, None)
            .unwrap();
    let mut queries = queries(&fixture, 112);
    queries.extend(fixture.centers.iter().map(|c| QueryVector::from(c.clone())));
    let mut cases: Vec<(String, Option<Filter>)> = vec![("none".into(), None)];
    for limit in [1000_i64, 500, 100, 50, 10, 1] {
        cases.push((format!("r<{limit}"), Some(r_below(limit))));
    }
    cases.push(("far".into(), Some(far_filter())));

    let mut digests = Vec::new();
    for (name, filter) in &cases {
        for top in [1_usize, 10, 50] {
            let mut hash = 0xcbf2_9ce4_8422_2325_u64;
            if let Some(filter) = filter {
                fnv(&mut hash, format!("{:?}", plan(lmi, filter)).as_bytes());
            }
            for query in &queries {
                let run = |index: &dyn VectorIndexRead| {
                    index
                        .search(
                            &[query],
                            filter.as_ref(),
                            top,
                            None,
                            &VectorQueryContext::default(),
                        )
                        .unwrap()
                        .remove(0)
                };
                let mutable = run(&*lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow());
                let shared = run(&*read_only.vector_data[DEFAULT_VECTOR_NAME]
                    .vector_index
                    .borrow());
                assert_eq!(
                    mutable, shared,
                    "read-only path diverged for {name} top {top}"
                );
                for hit in &mutable {
                    fnv(&mut hash, &hit.idx.to_le_bytes());
                    fnv(&mut hash, &hit.score.to_bits().to_le_bytes());
                }
                fnv(&mut hash, b";");
            }
            digests.push((format!("{name}/top{top}"), hash));
        }
    }
    // Plan counters and widenings accumulated over all of the above.
    let (plans, widenings) = {
        let index = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
        let VectorIndexEnum::Lmi(index) = &*index else {
            panic!("expected LMI index")
        };
        (
            index.filtered_plan_counts_for_test(),
            index.underfill_widenings_for_test(),
        )
    };
    digests.push(("plans_exact".into(), plans.0));
    digests.push(("plans_routed".into(), plans.1));
    digests.push(("widenings".into(), widenings));

    if BASELINE_DIGESTS.is_empty() || std::env::var_os("LMI_PRINT_BASELINE_DIGESTS").is_some() {
        for (name, value) in &digests {
            eprintln!("    (\"{name}\", 0x{value:016x}),");
        }
    }
    assert!(
        !BASELINE_DIGESTS.is_empty(),
        "baseline digests not recorded"
    );
    let expected: Vec<(String, u64)> = BASELINE_DIGESTS
        .iter()
        .map(|(name, value)| (name.to_string(), *value))
        .collect();
    assert_eq!(digests, expected);
}

// ---------------------------------------------------------------------------
// Research entry point, counters and probing policies.
// ---------------------------------------------------------------------------

use segment::index::lmi_index::filtered::{
    CalibrationBin, CalibrationTable, CostModel, FilterProbePolicy, FilteredSearchMode,
    FilteredSearchRecord, LinearCost,
};

fn research(
    segment: &Segment,
    query: &QueryVector,
    filter: &Filter,
    top: usize,
    mode: &FilteredSearchMode,
) -> (Vec<ScoredPointOffset>, FilteredSearchRecord) {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected LMI index")
    };
    lmi.filtered_search_research(query, filter, top, mode, &VectorQueryContext::default())
        .unwrap()
}

fn all_filters() -> Vec<Filter> {
    let mut filters: Vec<Filter> = [1000_i64, 500, 100, 50, 10, 1]
        .into_iter()
        .map(r_below)
        .collect();
    filters.push(far_filter());
    filters
}

/// Research `Baseline` mode, and `Policy(Baseline)`, run exactly the planner
/// and search of the production `search()`.
#[test]
fn research_baseline_mode_equals_production_search() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering"));
    let fixture = build_fixture();
    let lmi = &fixture.lmi;
    let mut queries = queries(&fixture, 40);
    queries.extend(fixture.centers.iter().map(|c| QueryVector::from(c.clone())));
    for filter in all_filters() {
        let decision = plan(lmi, &filter);
        for top in [1_usize, 10, 50] {
            for query in &queries {
                let production = lmi.vector_data[DEFAULT_VECTOR_NAME]
                    .vector_index
                    .borrow()
                    .search(
                        &[query],
                        Some(&filter),
                        top,
                        None,
                        &VectorQueryContext::default(),
                    )
                    .unwrap()
                    .remove(0);
                let (hits, record) =
                    research(lmi, query, &filter, top, &FilteredSearchMode::Baseline);
                assert_eq!(hits, production);
                assert_eq!(record.decision, Some(decision));
                assert_eq!(record.plan, decision.plan);
                let (policy_hits, policy_record) = research(
                    lmi,
                    query,
                    &filter,
                    top,
                    &FilteredSearchMode::Policy(FilterProbePolicy::Baseline),
                );
                assert_eq!(policy_hits, production);
                assert_eq!(policy_record.plan, decision.plan);
                assert_eq!(policy_record.decision, Some(decision));
            }
        }
    }
}

/// Forced plans: exact equals exact search; routed at the configured probe
/// equals the Baseline routed plan; `FixedProbe` at the configured probe
/// decides like Baseline and its threshold is the probe volume at its nprobe.
#[test]
fn forced_plans_and_fixed_probe_match_their_definitions() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering"));
    let fixture = build_fixture();
    let lmi = &fixture.lmi;
    let queries = queries(&fixture, 30);
    for filter in all_filters() {
        let decision = plan(lmi, &filter);
        let fixed = {
            let index = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
            let VectorIndexEnum::Lmi(index) = &*index else {
                panic!("expected LMI index")
            };
            (
                index.filtered_plan_at(&filter, NPROBE).unwrap().unwrap(),
                index.filtered_plan_at(&filter, 8).unwrap().unwrap(),
            )
        };
        assert_eq!(fixed.0, decision);
        assert_eq!(fixed.1.threshold, POINTS * 8 / BUCKETS);
        for query in &queries {
            let (exact_hits, record) =
                research(lmi, query, &filter, TOP, &FilteredSearchMode::ForceExact);
            assert_eq!(record.plan, FilteredPlan::ExactMatches);
            assert!(record.routed.is_none());
            assert_eq!(exact_hits, exact(lmi, query, Some(&filter)));
            let (routed_hits, record) = research(
                lmi,
                query,
                &filter,
                TOP,
                &FilteredSearchMode::ForceRouted { nprobe: NPROBE },
            );
            assert_eq!(record.plan, FilteredPlan::RoutedIntersect);
            if decision.plan == FilteredPlan::RoutedIntersect {
                assert_eq!(routed_hits, search(lmi, query, Some(&filter), None));
            }
            let (policy_hits, record) = research(
                lmi,
                query,
                &filter,
                TOP,
                &FilteredSearchMode::Policy(FilterProbePolicy::FixedProbe { nprobe: NPROBE }),
            );
            assert_eq!(record.plan, decision.plan);
            assert_eq!(policy_hits, search(lmi, query, Some(&filter), None));
        }
    }
}

/// Counter definitions, checked against an independent recomputation:
/// raw offsets are the selected buckets' posting lengths, filter evaluations
/// equal unique candidates, surviving candidates are those whose payload
/// matches, scored excludes deleted points, and cumulative work is the sum of
/// every probe of the query (each widening re-probes from scratch).
#[test]
fn routed_counters_match_independent_recomputation() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering"));
    let fixture = build_fixture();
    let lmi = &fixture.lmi;
    let (state, tracker_ids): (LmiRoutingState, Vec<usize>) = {
        let index = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
        let VectorIndexEnum::Lmi(index) = &*index else {
            panic!("expected LMI index")
        };
        let tracker = lmi.id_tracker.borrow();
        let ids = (0..POINTS as u32)
            .map(|offset| match tracker.external_id(offset).unwrap() {
                segment::types::ExtendedPointId::NumId(n) => n as usize - 1,
                other => panic!("unexpected id {other:?}"),
            })
            .collect();
        (index.routing_state().unwrap().clone(), ids)
    };
    let stopped = AtomicBool::new(false);
    let probe = |query: &QueryVector, nprobe: usize, keep: &dyn Fn(usize) -> bool| {
        let normalized: QueryVector = match query {
            QueryVector::Nearest(segment::data_types::vectors::VectorInternal::Dense(v)) => {
                normalized(v.clone()).into()
            }
            _ => unreachable!(),
        };
        let (unique, raw) = state
            .candidates_with_raw_count(&normalized, nprobe, &stopped)
            .unwrap()
            .unwrap();
        let surviving = unique
            .iter()
            .filter(|&&o| keep(tracker_ids[o as usize]))
            .count();
        (raw, unique.len(), surviving)
    };

    // No widening: final probe equals cumulative.
    let filter = r_below(500);
    let keep = |i: usize| r_value(i) < 500;
    for (q, query) in queries(&fixture, 20).iter().enumerate() {
        for nprobe in [1_usize, 2, 4, 8] {
            let (_, record) = research(
                lmi,
                query,
                &filter,
                TOP,
                &FilteredSearchMode::ForceRouted { nprobe },
            );
            let stats = record.routed.unwrap();
            assert_eq!(stats.widenings, 0, "query {q} nprobe {nprobe}");
            let (raw, unique, surviving) = probe(query, nprobe, &keep);
            let counts = stats.final_probe;
            assert_eq!(counts.raw_offsets, raw);
            assert_eq!(counts.unique_candidates, unique);
            assert_eq!(counts.filter_evaluations, unique);
            assert_eq!(counts.surviving, surviving);
            assert_eq!(counts.scored, surviving, "nothing is deleted");
            assert_eq!(stats.cumulative, stats.final_probe);
            // Static postings hold each point once: raw equals unique.
            assert_eq!(counts.raw_offsets, counts.unique_candidates);
        }
    }

    // Widening: cumulative is the sum over the probe sequence p, 2p, 4p, ...
    let filter = far_filter();
    let keep = |i: usize| FAR_CLUSTERS.contains(&(i % BUCKETS));
    let query: QueryVector = fixture.centers[0].clone().into();
    let (_, record) = research(
        lmi,
        &query,
        &filter,
        TOP,
        &FilteredSearchMode::ForceRouted { nprobe: 1 },
    );
    let stats = record.routed.unwrap();
    assert!(stats.widenings > 0);
    let mut expected = segment::index::lmi_index::filtered::ProbeCounts::default();
    let mut nprobe = 1;
    for step in 0..=stats.widenings {
        let (raw, unique, surviving) = probe(&query, nprobe, &keep);
        expected.raw_offsets += raw;
        expected.unique_candidates += unique;
        expected.filter_evaluations += unique;
        expected.surviving += surviving;
        expected.scored += surviving;
        if step == stats.widenings {
            assert_eq!(stats.final_nprobe, nprobe);
            assert_eq!(stats.final_probe.unique_candidates, unique);
        } else {
            nprobe = (nprobe * 2).min(BUCKETS);
        }
    }
    assert_eq!(stats.initial_nprobe, 1);
    assert_eq!(stats.cumulative, expected);
    assert!(stats.cumulative.unique_candidates > stats.final_probe.unique_candidates);

    // A deleted surviving candidate is evaluated and survives the filter but is not scored.
    let filter = r_below(1000);
    let query = queries(&fixture, 1).remove(0);
    let mode = FilteredSearchMode::ForceRouted { nprobe: 2 };
    let (hits, before) = research(lmi, &query, &filter, TOP, &mode);
    let victim = lmi.id_tracker.borrow().external_id(hits[0].idx).unwrap();
    lmi.id_tracker.borrow_mut().drop(victim, None).unwrap();
    let (_, after) = research(lmi, &query, &filter, TOP, &mode);
    let (before, after) = (before.routed.unwrap(), after.routed.unwrap());
    assert_eq!(after.final_probe.surviving, before.final_probe.surviving);
    assert_eq!(after.final_probe.scored, before.final_probe.scored - 1);
}

fn costs(exact_per_match_ns: f64, route_fixed_ns: f64) -> CostModel {
    CostModel {
        exact_iteration: LinearCost {
            fixed_ns: 0.0,
            per_unit_ns: exact_per_match_ns,
        },
        exact_scan: LinearCost {
            fixed_ns: 0.0,
            per_unit_ns: exact_per_match_ns,
        },
        route_fixed_ns,
        route_per_candidate_ns: 0.0,
        route_per_scored_ns: 0.0,
        unique_candidates_at: [1_usize, 2, 4, 8, 16, 32]
            .iter()
            .map(|&p| (p, (p * PER_BUCKET) as f64))
            .collect(),
        exact_table: None,
        routed_table: None,
    }
}

/// Empirical tables replace the linear terms and interpolate log-log.
#[test]
fn empirical_cost_tables_interpolate_log_log_and_clamp() {
    use segment::index::lmi_index::filtered::FilterEstimate;
    let mut model = costs(1.0, 1.0);
    model.exact_table = Some(vec![(100.0, 1_000.0), (10_000.0, 100_000.0)]);
    model.routed_table = Some(vec![(4, vec![(0.01, 2_000.0), (1.0, 8_000.0)])]);
    let estimate = |exp: usize| FilterEstimate {
        min: exp,
        exp,
        max: exp,
        available: 100_000,
        primary_iteration: true,
    };
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6 * b.abs().max(1.0);
    assert!(
        close(model.exact_ns(&estimate(1_000)), 10_000.0),
        "log-log midpoint"
    );
    assert!(
        close(model.exact_ns(&estimate(10)), 1_000.0),
        "clamped below"
    );
    assert!(
        close(model.exact_ns(&estimate(50_000)), 100_000.0),
        "clamped above"
    );
    assert!(
        close(model.routed_ns(4, &estimate(10_000)).unwrap(), 4_000.0),
        "s = 0.1"
    );
    assert!(
        model.routed_ns(8, &estimate(10_000)).is_none(),
        "uncalibrated nprobe"
    );
}

/// Calibration tables bin conservatively and fall back to exact.
#[test]
fn calibration_table_bins_round_selectivity_down() {
    let table = CalibrationTable {
        target_recall: 0.95,
        bins: vec![
            CalibrationBin {
                selectivity: 0.01,
                nprobe: None,
            },
            CalibrationBin {
                selectivity: 0.1,
                nprobe: Some(8),
            },
            CalibrationBin {
                selectivity: 1.0,
                nprobe: Some(2),
            },
        ],
    };
    let bin = |s: f64| table.bin_for(s).unwrap().selectivity;
    assert_eq!(bin(1.0), 1.0);
    assert_eq!(
        bin(0.99),
        0.1,
        "between bins: the lower selectivity, more probes"
    );
    assert_eq!(bin(0.1), 0.1);
    assert_eq!(bin(0.05), 0.01);
    assert_eq!(bin(0.0001), 0.01, "below the grid: the smallest bin");
    assert!(
        CalibrationTable {
            target_recall: 0.9,
            bins: vec![]
        }
        .bin_for(0.5)
        .is_none()
    );
}

/// Policy decisions use the estimate only and route to the forced plans.
#[test]
fn probing_policies_decide_from_the_estimate_and_execute_the_forced_plan() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering"));
    let fixture = build_fixture();
    let lmi = &fixture.lmi;
    let table = CalibrationTable {
        target_recall: 0.95,
        bins: vec![
            CalibrationBin {
                selectivity: 0.001,
                nprobe: None,
            },
            CalibrationBin {
                selectivity: 0.05,
                nprobe: Some(16),
            },
            CalibrationBin {
                selectivity: 0.5,
                nprobe: Some(4),
            },
        ],
    };
    // Exact is expensive per match and routing is cheap: route whenever the table allows.
    let cheap_routing = costs(1_000.0, 1.0);
    // Routing is prohibitively expensive: always exact.
    let costly_routing = costs(1.0, 1e12);
    let query = queries(&fixture, 1).remove(0);
    // r < 500 matches 3,199 of 6,400 points: just below the 0.5 bin, so it rounds
    // down to the 0.05 bin (more probes), as conservative binning requires.
    for (limit, expected_nprobe) in [
        (1000_i64, Some(4)),
        (500, Some(16)),
        (100, Some(16)),
        (50, Some(16)),
        (1, None),
    ] {
        let filter = r_below(limit);
        let calibrated = FilterProbePolicy::Calibrated {
            table: table.clone(),
            costs: cheap_routing.clone(),
        };
        let (hits, record) = research(
            lmi,
            &query,
            &filter,
            TOP,
            &FilteredSearchMode::Policy(calibrated),
        );
        let decision = record.policy.unwrap();
        match expected_nprobe {
            Some(nprobe) => {
                assert_eq!(
                    decision.plan,
                    FilteredPlan::RoutedIntersect,
                    "limit {limit}"
                );
                assert_eq!(decision.nprobe, nprobe, "limit {limit}");
                let (forced, _) = research(
                    lmi,
                    &query,
                    &filter,
                    TOP,
                    &FilteredSearchMode::ForceRouted { nprobe },
                );
                assert_eq!(hits, forced);
                assert_eq!(record.routed.unwrap().initial_nprobe, nprobe);
            }
            None => {
                assert_eq!(decision.plan, FilteredPlan::ExactMatches, "limit {limit}");
                assert_eq!(hits, exact(lmi, &query, Some(&filter)));
            }
        }
        let costly = FilterProbePolicy::Calibrated {
            table: table.clone(),
            costs: costly_routing.clone(),
        };
        let (hits, record) = research(
            lmi,
            &query,
            &filter,
            TOP,
            &FilteredSearchMode::Policy(costly),
        );
        assert_eq!(record.policy.unwrap().plan, FilteredPlan::ExactMatches);
        assert_eq!(hits, exact(lmi, &query, Some(&filter)));
        // C1 at a fixed probe count follows the same cost comparison.
        let c1 = FilterProbePolicy::CostModel {
            nprobe: 8,
            costs: cheap_routing.clone(),
        };
        let (_, record) = research(lmi, &query, &filter, TOP, &FilteredSearchMode::Policy(c1));
        let decision = record.policy.unwrap();
        assert_eq!(decision.nprobe, 8);
        assert!(decision.predicted_exact_ns.unwrap() > decision.predicted_routed_ns.unwrap());
        assert_eq!(decision.plan, FilteredPlan::RoutedIntersect);
    }
    // A probe count without calibration is an error, never a silent default.
    let bad = FilterProbePolicy::CostModel {
        nprobe: 3,
        costs: cheap_routing,
    };
    let index = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(index) = &*index else {
        panic!("expected LMI index")
    };
    assert!(
        index
            .filtered_search_research(
                &query,
                &r_below(500),
                TOP,
                &FilteredSearchMode::Policy(bad),
                &VectorQueryContext::default(),
            )
            .is_err()
    );
}
