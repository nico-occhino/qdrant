#![cfg(all(feature = "lmi-training", feature = "testing"))]

//! SISAP23 300K filtered-search study for the adaptive LMI planner.
//!
//! Builds one Float16 cosine source segment from the canonical 300K subset,
//! attaches an indexed integer payload `r` drawn from a hash of the point id
//! (independent of the geometry), and natively trains two indexes from it:
//! spherical IVF (`LMI_RESEARCH_ROUTER=centroid`) and the MLP LMI, both with
//! the same teacher, bucket count and `nprobe`. For a sweep of filter
//! selectivities it reports the chosen plan, filtered Recall@10 against
//! native exact filtered search over the same segment, and per-query latency
//! of the learned path and of exact filtered search.
//!
//! Run (release-like build, single test thread):
//! `LMI_FILTER_SISAP_INPUTS=<dir with vectors.f16, queries.f32, ground_truth.i32>
//!  LMI_FILTER_SISAP_OUTPUT=<result.json> cargo test --profile perf -p segment
//!  --features testing,lmi-training --test lmi_filtered_sisap300k -- --ignored --nocapture`
//! Optional: `LMI_FILTER_NPROBE` (default 16), `LMI_FILTER_QUERIES` (default 1000).

use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::flags::FeatureFlags;
use common::types::ScoredPointOffset;
use half::f16;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::NonAppendableSegmentEntry;
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use segment::id_tracker::IdTrackerRead;
use segment::index::lmi_index::{FilteredPlan, LmiConfig};
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::json_path::JsonPath;
use segment::segment::Segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Condition, Distance, ExtendedPointId, FieldCondition, Filter, HnswGlobalConfig, Indexes,
    Payload, PayloadSchemaType, Range, SearchParams, VectorStorageDatatype,
};

const DIM: usize = 768;
const POINTS: usize = 300_000;
const TOP: usize = 10;
const R_RANGE: u64 = 1_000_000;
/// Fractions of the corpus matched by `r < fraction · R_RANGE`.
const SELECTIVITIES: [f64; 9] = [1.0, 0.5, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001, 0.0001];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |v| v.parse().expect(name))
}

/// SplitMix64: a fixed hash, so the payload is reproducible and unrelated to the vectors.
fn r_value(id: u64) -> u64 {
    let mut z = id.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) % R_RANGE
}

fn model(nprobe: usize) -> LmiConfig {
    // The accepted 300K static configuration (548 ≈ √300K buckets), at a
    // higher default probe so the unfiltered baseline is in the high-recall regime.
    LmiConfig {
        n_buckets: 548,
        sample_size: 32_768,
        hidden_dim: 512,
        epochs: 30,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations: 5,
        nprobe,
        seed: 42,
    }
}

fn r_below(fraction: f64) -> Filter {
    Filter::new_must(Condition::Field(FieldCondition::new_range(
        JsonPath::new("r"),
        Range {
            lt: Some((fraction * R_RANGE as f64).into()),
            ..Default::default()
        },
    )))
}

fn index_search(
    segment: &Segment,
    query: &QueryVector,
    filter: Option<&Filter>,
    params: Option<&SearchParams>,
) -> (Vec<ScoredPointOffset>, f64) {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let began = Instant::now();
    let result = index
        .search(&[query], filter, TOP, params, &VectorQueryContext::default())
        .unwrap()
        .remove(0);
    (result, began.elapsed().as_secs_f64() * 1e3)
}

fn external_ids(segment: &Segment, hits: &[ScoredPointOffset]) -> Vec<u64> {
    let tracker = segment.id_tracker.borrow();
    hits.iter()
        .map(|hit| match tracker.external_id(hit.idx).unwrap() {
            ExtendedPointId::NumId(n) => n,
            other => panic!("unexpected id {other:?}"),
        })
        .collect()
}

fn recall(found: &[u64], gold: &[u64]) -> f64 {
    if gold.is_empty() {
        return 1.0;
    }
    let gold: HashSet<_> = gold.iter().collect();
    found.iter().filter(|id| gold.contains(id)).count() as f64 / gold.len() as f64
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * p).round() as usize]
}

fn lmi_counters(segment: &Segment) -> ((u64, u64), u64) {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected a trained LMI index")
    };
    (lmi.filtered_plan_counts_for_test(), lmi.underfill_widenings_for_test())
}

fn build_index(
    source: &Segment,
    root: &std::path::Path,
    router: &str,
    config: LmiConfig,
) -> (Segment, f64) {
    // SAFETY: this ignored benchmark runs on one test thread and sets the
    // research router before the build that reads it.
    unsafe { std::env::set_var("LMI_RESEARCH_ROUTER", router) };
    let staging = tempfile::tempdir().unwrap();
    let mut cfg = source.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(config);
    let began = Instant::now();
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &cfg,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder.update(&[source], &AtomicBool::new(false)).unwrap();
    let segment = builder.build_for_test(root);
    let seconds = began.elapsed().as_secs_f64();
    unsafe { std::env::remove_var("LMI_RESEARCH_ROUTER") };
    (segment, seconds)
}

#[test]
#[ignore = "SISAP23 300K filtered study; requires LMI_FILTER_SISAP_INPUTS and LMI_FILTER_SISAP_OUTPUT"]
fn sisap23_300k_adaptive_filtering() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filtering 300K"));
    let inputs = PathBuf::from(std::env::var("LMI_FILTER_SISAP_INPUTS").expect("inputs"));
    let output = PathBuf::from(std::env::var("LMI_FILTER_SISAP_OUTPUT").expect("output"));
    let nprobe = env_usize("LMI_FILTER_NPROBE", 16);
    let query_count = env_usize("LMI_FILTER_QUERIES", 1_000);

    let vectors = std::fs::read(inputs.join("vectors.f16")).unwrap();
    let queries = std::fs::read(inputs.join("queries.f32")).unwrap();
    let gold = std::fs::read(inputs.join("ground_truth.i32")).unwrap();
    assert_eq!(vectors.len(), POINTS * DIM * 2);
    assert_eq!(queries.len(), 10_000 * DIM * 4);
    assert_eq!(gold.len(), 10_000 * 1_000 * 4);

    // Source: Float16 cosine Plain segment with an indexed integer payload.
    let root = tempfile::tempdir().unwrap();
    let began = Instant::now();
    let mut source = {
        let plain = build_simple_segment(root.path(), DIM, Distance::Cosine).unwrap();
        let mut cfg = plain.config().clone();
        cfg.vector_data
            .get_mut(DEFAULT_VECTOR_NAME)
            .unwrap()
            .datatype = Some(VectorStorageDatatype::Float16);
        drop(plain);
        segment::segment_constructor::build_segment(root.path(), &cfg, None, true)
            .unwrap()
            .0
    };
    let mut row = vec![0.0_f32; DIM];
    for i in 0..POINTS {
        for (d, value) in row.iter_mut().enumerate() {
            let at = (i * DIM + d) * 2;
            *value = f16::from_le_bytes([vectors[at], vectors[at + 1]]).to_f32();
        }
        let id = i as u64 + 1;
        source
            .upsert_point(id, id.into(), only_default_vector(&row))
            .unwrap();
        let payload: Payload =
            serde_json::from_value(serde_json::json!({ "r": r_value(id) })).unwrap();
        source
            .set_full_payload(POINTS as u64 + id, id.into(), &payload)
            .unwrap();
    }
    source
        .create_field_index(
            3 * POINTS as u64,
            &JsonPath::from_str("r").unwrap(),
            Some(&PayloadSchemaType::Integer.into()),
        )
        .unwrap();
    let source_seconds = began.elapsed().as_secs_f64();
    drop(vectors);
    eprintln!("source segment: {POINTS} points in {source_seconds:.1} s");

    let query_vectors: Vec<QueryVector> = (0..query_count)
        .map(|q| {
            let v: Vec<f32> = (0..DIM)
                .map(|d| {
                    let at = (q * DIM + d) * 4;
                    f32::from_le_bytes(queries[at..at + 4].try_into().unwrap())
                })
                .collect();
            v.into()
        })
        .collect();
    let official_top10 = |q: usize| -> Vec<u64> {
        (0..TOP)
            .map(|k| {
                let at = (q * 1_000 + k) * 4;
                i32::from_le_bytes(gold[at..at + 4].try_into().unwrap()) as u64
            })
            .collect()
    };

    // Exact filtered gold, native semantics, computed once per selectivity on the source.
    let exact = SearchParams {
        exact: true,
        ..Default::default()
    };
    let mut exact_gold: Vec<Vec<Vec<u64>>> = Vec::new();
    let mut exact_ms: Vec<Vec<f64>> = Vec::new();
    let mut matching_counts = Vec::new();
    for &fraction in &SELECTIVITIES {
        let filter = r_below(fraction);
        let limit = (fraction * R_RANGE as f64) as u64;
        matching_counts.push((1..=POINTS as u64).filter(|&id| r_value(id) < limit).count());
        let (mut ids, mut times) = (Vec::new(), Vec::new());
        for query in &query_vectors {
            let (hits, ms) = index_search(&source, query, Some(&filter), Some(&exact));
            ids.push(external_ids(&source, &hits));
            times.push(ms);
        }
        exact_gold.push(ids);
        exact_ms.push(times);
    }

    let mut systems = Vec::new();
    for router in ["centroid", "mlp"] {
        let build_root = tempfile::tempdir().unwrap();
        let (segment, build_seconds) = build_index(&source, build_root.path(), router, model(nprobe));
        eprintln!("\n{router}: built in {build_seconds:.1} s, nprobe {nprobe}");

        // Unfiltered baseline: official recall and native-exact recall.
        let (mut official, mut native, mut unfiltered_ms) = (0.0, 0.0, Vec::new());
        for (q, query) in query_vectors.iter().enumerate() {
            let (hits, ms) = index_search(&segment, query, None, None);
            let found = external_ids(&segment, &hits);
            official += recall(&found, &official_top10(q));
            native += recall(&found, &exact_gold[0][q]);
            unfiltered_ms.push(ms);
        }
        let n = query_vectors.len() as f64;
        let unfiltered = serde_json::json!({
            "official_recall_at_10": official / n,
            "native_exact_recall_at_10": native / n,
            "p50_ms": percentile(&mut unfiltered_ms.clone(), 0.5),
            "p95_ms": percentile(&mut unfiltered_ms, 0.95),
        });
        eprintln!("unfiltered: {unfiltered}");
        eprintln!(
            "{:>9} {:>8} {:<16} {:>7} {:>9} {:>9} {:>9} {:>9} {:>8}",
            "select", "matching", "plan", "sampled", "recall@10", "p50 ms", "exact p50", "speedup", "widen/q"
        );

        let mut rows = Vec::new();
        for (s, &fraction) in SELECTIVITIES.iter().enumerate() {
            let filter = r_below(fraction);
            let decision = {
                let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
                let VectorIndexEnum::Lmi(lmi) = &*index else {
                    panic!("expected a trained LMI index")
                };
                lmi.filtered_plan(&filter).unwrap().unwrap()
            };
            let (plans_before, widen_before) = lmi_counters(&segment);
            let (mut total, mut times) = (0.0, Vec::new());
            for (q, query) in query_vectors.iter().enumerate() {
                let (hits, ms) = index_search(&segment, query, Some(&filter), None);
                let found = external_ids(&segment, &hits);
                assert_eq!(found.len(), TOP.min(matching_counts[s]), "fill");
                let limit = (fraction * R_RANGE as f64) as u64;
                assert!(found.iter().all(|&id| r_value(id) < limit), "filter violated");
                total += recall(&found, &exact_gold[s][q]);
                times.push(ms);
            }
            let (plans_after, widen_after) = lmi_counters(&segment);
            let served = (plans_after.0 - plans_before.0, plans_after.1 - plans_before.1);
            let p50 = percentile(&mut times.clone(), 0.5);
            let p95 = percentile(&mut times, 0.95);
            let exact_p50 = percentile(&mut exact_ms[s].clone(), 0.5);
            let widenings = (widen_after - widen_before) as f64 / n;
            let mean = total / n;
            eprintln!(
                "{:>8.2}% {:>8} {:<16} {:>7} {:>9.4} {:>9.3} {:>9.3} {:>8.1}x {:>8.2}",
                fraction * 100.0,
                matching_counts[s],
                format!("{:?}", decision.plan),
                decision.sampled,
                mean,
                p50,
                exact_p50,
                exact_p50 / p50,
                widenings,
            );
            if decision.plan == FilteredPlan::ExactMatches {
                assert!((mean - 1.0).abs() < 1e-9, "exact plan must equal exact search");
            }
            rows.push(serde_json::json!({
                "selectivity": fraction,
                "matching_points": matching_counts[s],
                "decision": decision,
                "served_exact_routed": [served.0, served.1],
                "recall_at_10_vs_native_exact_filtered": mean,
                "p50_ms": p50,
                "p95_ms": p95,
                "exact_filtered_p50_ms": exact_p50,
                "widenings_per_query": widenings,
            }));
        }
        systems.push(serde_json::json!({
            "router": router,
            "build_seconds": build_seconds,
            "unfiltered": unfiltered,
            "filtered": rows,
        }));
    }

    let report = serde_json::json!({
        "dataset": "SISAP23 LAION CLIP-768, canonical 300K subset (Float16, cosine)",
        "inputs": inputs,
        "points": POINTS,
        "queries": query_count,
        "query_rows": format!("0..{query_count}"),
        "top": TOP,
        "model": model(nprobe),
        "probe_volume": POINTS * nprobe / 548,
        "payload": "r = splitmix64(id) mod 1e6, Integer index; filter r < selectivity * 1e6",
        "gold": "native exact filtered search over the same Float16 source segment",
        "timing": "wall time of VectorIndexRead::search, one query per call, one thread, warm",
        "source_build_seconds": source_seconds,
        "systems": systems,
    });
    std::fs::write(&output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("\nwrote {}", output.display());
}
