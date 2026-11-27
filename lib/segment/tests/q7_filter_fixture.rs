#![cfg(all(feature = "lmi-training", feature = "testing"))]

//! Q7 shared filtered-search fixture (research/dli/uniform/Q7_PREREGISTRATION.md; PostgreSQL
//! `FILTER_FIXTURE_SPEC.md` @ 3ec4041). One ignored test, driven by environment variables:
//!
//! * `Q7_INPUTS`: the directory written by `research/dli/uniform/q7_fixture.py`, holding `vectors.f16`,
//!   `queries.f16` and `filter_rank.u32`.
//! * `Q7_OUT`: a new output directory (it must not exist). It receives `rows.jsonl`, `gold.jsonl`,
//!   `topology.json` and `summary.json`.
//! * `Q7_ROWS` (optional): a prefix of the fixture, for the labelled local smoke. The default is all 60,000.
//! * `Q7_THREADS` (optional): query parallelism. Work is counted per query on its own thread.
//! * `QDRANT_LMI_SAMPLE_BUDGET_BYTES=2000000000`, as in the Phase 3 jobs: the 100,000-row teacher sample needs it.
//!
//! The arms are:
//! * **S-LMI.** A static MLP LMI over all rows, 250 buckets, native nprobe 16.
//! * **DLI.** E1 members (buffer 5,000, arity 10, aⁱ buckets, nprobe 1), reconstructed by the Bentley–Saxe
//!   schedule of inserting ids 1..N in order; the remaining buffer is a plain segment.
//! * **HNSW.** m 16, ef_construct 100, default full-scan threshold.
//!
//! The work for every search is counted by `TaskScoringProfileGuard`.
//!
//! Run: `Q7_INPUTS=… Q7_OUT=… cargo test --profile perf -p segment --features testing,lmi-training
//!  --test q7_filter_fixture -- --ignored --nocapture`

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::flags::FeatureFlags;
use common::types::ScoredPointOffset;
use half::f16;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::NonAppendableSegmentEntry;
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry, StorageSegmentEntry};
use segment::id_tracker::IdTrackerRead;
use segment::index::hnsw_index::point_scorer::{TaskScoringProfileGuard, scoring_task_success_snapshot};
use segment::index::lmi_index::LmiConfig;
use segment::index::lmi_index::filtered::FilteredSearchMode;
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::json_path::JsonPath;
use segment::segment::Segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Condition, Distance, ExtendedPointId, FieldCondition, Filter, HnswConfig, HnswGlobalConfig, Indexes, Payload,
    PayloadSchemaType, Range, SearchParams, VectorStorageDatatype,
};
use serde_json::json;

const DIM: usize = 768;
const K: usize = 30;
const FULL_ROWS: usize = 60_000;
const PREDICATES: [(&str, u64); 4] = [("0.1%", 60), ("1%", 600), ("10%", 6_000), ("50%", 30_000)];
const BUFFER: usize = 5_000;
const ARITY: usize = 10;
const HNSW_EFS: [usize; 11] = [30, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1024];
const HNSW_NATIVE_EF: usize = 128;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

fn read_f16_rows(path: &Path, rows: usize) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len() >= rows * DIM * 2, "{}: too short", path.display());
    bytes[..rows * DIM * 2]
        .chunks_exact(DIM * 2)
        .map(|r| r.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect())
        .collect()
}

fn filter(max_rank: u64) -> Filter {
    Filter::new_must(Condition::Field(FieldCondition::new_range(
        JsonPath::new("filter_rank"),
        Range { lte: Some((max_rank as f64).into()), ..Default::default() },
    )))
}

fn plain_segment(root: &Path, name: &str, rows: &[Vec<f32>], ranks: &[u32], first_id: u64) -> Segment {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let plain = build_simple_segment(&dir, DIM, Distance::Cosine).unwrap();
    let mut cfg = plain.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().datatype = Some(VectorStorageDatatype::Float16);
    let stale = plain.segment_path.clone();
    drop(plain);
    std::fs::remove_dir_all(stale).unwrap();
    let mut segment = segment::segment_constructor::build_segment(&dir, &cfg, None, true).unwrap().0;
    let n = rows.len() as u64;
    for (i, row) in rows.iter().enumerate() {
        let id = first_id + i as u64;
        segment.upsert_point(i as u64 + 1, id.into(), only_default_vector(row)).unwrap();
        let payload: Payload = serde_json::from_value(json!({ "filter_rank": ranks[(id - 1) as usize] })).unwrap();
        segment.set_full_payload(n + i as u64 + 1, id.into(), &payload).unwrap();
    }
    segment
        .create_field_index(2 * n + 1, &JsonPath::from_str("filter_rank").unwrap(), Some(&PayloadSchemaType::Integer.into()))
        .unwrap();
    segment.flush(true).unwrap();
    segment
}

fn build(root: &Path, name: &str, source: &Segment, index: Indexes) -> (Segment, f64) {
    let mut cfg = source.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = index;
    let staging = tempfile::tempdir_in(root).unwrap();
    let target = root.join(name);
    std::fs::create_dir_all(&target).unwrap();
    let began = Instant::now();
    let mut builder =
        SegmentBuilder::new(staging.path(), &cfg, &HnswGlobalConfig::default(), FeatureFlags::default()).unwrap();
    builder.update(&[source], &AtomicBool::new(false)).unwrap();
    let built = builder.build_for_test(&target);
    (built, began.elapsed().as_secs_f64())
}

fn lmi_config(buckets: usize, nprobe: usize) -> LmiConfig {
    LmiConfig {
        n_buckets: buckets,
        sample_size: 100_000.max(40 * buckets),
        hidden_dim: 128,
        epochs: 10,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations: 25,
        nprobe,
        seed: 42,
        reuse_router: None,
    }
}

/// Bentley–Saxe members after inserting ids 1..=n in 5,000-point drains: (level, first id, last id), oldest
/// first, plus the remaining buffer range. A drain merges the buffer, every level below the target and the
/// target into the smallest level that can hold them (capacity 5,000 · 10^level).
fn hierarchy(n: usize) -> (Vec<(usize, u64, u64)>, Option<(u64, u64)>) {
    let mut levels: Vec<Option<(u64, u64)>> = Vec::new();
    let drains = n / BUFFER;
    for d in 0..drains {
        let (mut lo, hi) = ((d * BUFFER + 1) as u64, ((d + 1) * BUFFER) as u64);
        let mut count = BUFFER;
        let mut level = 1;
        loop {
            if levels.len() < level {
                levels.push(None);
            }
            let here = levels[level - 1].map_or(0, |(a, b)| (b - a + 1) as usize);
            if count + here <= BUFFER * ARITY.pow(level as u32) {
                if let Some((a, _)) = levels[level - 1] {
                    lo = lo.min(a);
                }
                for l in levels.iter_mut().take(level - 1) {
                    *l = None;
                }
                levels[level - 1] = Some((lo, hi));
                break;
            }
            count += here;
            if let Some((a, _)) = levels[level - 1] {
                lo = lo.min(a);
            }
            level += 1;
        }
    }
    let members = levels
        .iter()
        .enumerate()
        .rev()
        .filter_map(|(i, r)| r.map(|(a, b)| (i + 1, a, b)))
        .collect();
    let buffer = (drains * BUFFER < n).then(|| ((drains * BUFFER + 1) as u64, n as u64));
    (members, buffer)
}

fn external(segment: &Segment, hits: &[ScoredPointOffset]) -> Vec<(u64, f32)> {
    let tracker = segment.id_tracker.borrow();
    hits.iter()
        .map(|h| match tracker.external_id(h.idx).unwrap() {
            ExtendedPointId::NumId(n) => (n, h.score),
            other => panic!("unexpected id {other:?}"),
        })
        .collect()
}

fn index_of(segment: &Segment) -> atomic_refcell::AtomicRef<'_, VectorIndexEnum> {
    segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow()
}

/// One counted search. Returns (hits as (id, score), vectors scored, plan label, initial nprobe).
fn lmi_search(segment: &Segment, query: &QueryVector, f: &Filter, mode: &FilteredSearchMode) -> (Vec<(u64, f32)>, u64, String, usize) {
    let index = index_of(segment);
    let VectorIndexEnum::Lmi(lmi) = &*index else { panic!("expected LMI") };
    let guard = TaskScoringProfileGuard::begin();
    let (hits, record) = lmi.filtered_search_research(query, f, K, mode, &VectorQueryContext::default()).unwrap();
    let work = scoring_task_success_snapshot();
    drop(guard);
    (external(segment, &hits), work, format!("{:?}", record.plan), record.nprobe)
}

fn plain_search(segment: &Segment, query: &QueryVector, f: &Filter, params: Option<&SearchParams>, top: usize) -> (Vec<(u64, f32)>, u64) {
    let index = index_of(segment);
    let guard = TaskScoringProfileGuard::begin();
    let hits = index.search(&[query], Some(f), top, params, &VectorQueryContext::default()).unwrap().remove(0);
    let work = scoring_task_success_snapshot();
    drop(guard);
    (external(segment, &hits), work)
}

fn merge(parts: Vec<Vec<(u64, f32)>>) -> Vec<(u64, f32)> {
    let mut all: Vec<(u64, f32)> = parts.into_iter().flatten().collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(K);
    all
}

#[test]
#[ignore]
fn q7_filter_fixture() {
    let inputs = env_path("Q7_INPUTS");
    let out = env_path("Q7_OUT");
    std::fs::create_dir(&out).expect("Q7_OUT must not exist");
    let n: usize = std::env::var("Q7_ROWS").map_or(FULL_ROWS, |v| v.parse().unwrap());
    let threads: usize = std::env::var("Q7_THREADS").map_or(8, |v| v.parse().unwrap());
    let rows = read_f16_rows(&inputs.join("vectors.f16"), n);
    let queries: Vec<QueryVector> = read_f16_rows(&inputs.join("queries.f16"), 32).into_iter().map(Into::into).collect();
    let ranks: Vec<u32> = std::fs::read(inputs.join("filter_rank.u32"))
        .unwrap()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(ranks.len(), FULL_ROWS);
    let work = out.join("segments");
    std::fs::create_dir_all(&work).unwrap();

    // Segments.
    let source = plain_segment(&work, "source", &rows, &ranks, 1);
    let (slmi, slmi_s) = build(&work, "slmi", &source, Indexes::LmiTrained(lmi_config(250, 16)));
    let (hnsw, hnsw_s) = build(
        &work,
        "hnsw",
        &source,
        Indexes::Hnsw(HnswConfig { m: 16, ef_construct: 100, max_indexing_threads: threads, ..HnswConfig::default() }),
    );
    let (members, buffer) = hierarchy(n);
    let mut dli: Vec<(usize, Segment)> = Vec::new();
    let mut topo = vec![];
    for (level, a, b) in &members {
        let src = plain_segment(&work, &format!("dli-src-l{level}"), &rows[(*a - 1) as usize..*b as usize], &ranks, *a);
        let buckets = ARITY.pow(*level as u32);
        let (seg, secs) = build(&work, &format!("dli-l{level}"), &src, Indexes::LmiTrained(lmi_config(buckets, 1)));
        topo.push(json!({"level": level, "ids": [a, b], "buckets": buckets, "build_seconds": secs}));
        dli.push((*level, seg));
    }
    let dli_buffer = buffer.map(|(a, b)| plain_segment(&work, "dli-buffer", &rows[(a - 1) as usize..b as usize], &ranks, a));
    std::fs::write(
        out.join("topology.json"),
        serde_json::to_vec_pretty(&json!({"rows": n, "dli_members": topo, "dli_buffer": buffer,
            "slmi_build_seconds": slmi_s, "hnsw_build_seconds": hnsw_s})).unwrap(),
    )
    .unwrap();

    // Gold over the stored representation: exact filtered search on the plain source, top 31 for the boundary.
    let exact = SearchParams { exact: true, ..Default::default() };
    let mut gold_lines = Vec::new();
    let mut gold: Vec<Vec<Vec<u64>>> = Vec::new();
    for (name, max_rank) in PREDICATES {
        let f = filter(max_rank);
        let qualifying = ranks[..n].iter().filter(|&&r| u64::from(r) <= max_rank).count();
        let mut per_q = Vec::new();
        for (qi, q) in queries.iter().enumerate() {
            let (hits, work) = plain_search(&source, q, &f, Some(&exact), K + 1);
            let ids: Vec<u64> = hits.iter().take(K).map(|h| h.0).collect();
            let (s30, s31) = (hits.get(K - 1).map(|h| h.1), hits.get(K).map(|h| h.1));
            gold_lines.push(json!({"predicate": name, "query": qi, "qualifying": qualifying, "gold": ids,
                "s30": s30, "s31": s31, "tie_at_boundary": s30.is_some() && s30 == s31, "exact_work": work}).to_string());
            per_q.push(ids);
        }
        gold.push(per_q);
    }
    std::fs::write(out.join("gold.jsonl"), gold_lines.join("\n") + "\n").unwrap();

    // Strategies.
    let mut strategies: Vec<(String, String)> = vec![("S-LMI".into(), "native".into()), ("S-LMI".into(), "exact".into()),
        ("DLI".into(), "native".into()), ("DLI".into(), "exact".into())];
    let mut p = 1;
    while p <= 250 {
        strategies.push(("S-LMI".into(), format!("nprobe={p}")));
        p *= 2;
    }
    strategies.push(("S-LMI".into(), "nprobe=250".into()));
    for p in [1usize, 2, 4, 8, 16, 32, 64, 100] {
        strategies.push(("DLI".into(), format!("nprobe={p}")));
    }
    strategies.push(("HNSW".into(), format!("native ef={HNSW_NATIVE_EF}")));
    for ef in HNSW_EFS {
        strategies.push(("HNSW".into(), format!("ef={ef}")));
    }
    let lmi_mode = |s: &str, cap: usize| -> FilteredSearchMode {
        match s {
            "native" => FilteredSearchMode::Baseline,
            "exact" => FilteredSearchMode::ForceExact,
            p => FilteredSearchMode::ForceRouted { nprobe: p.trim_start_matches("nprobe=").parse::<usize>().unwrap().min(cap) },
        }
    };
    let mut jobs = Vec::new();
    for (arm, strat) in &strategies {
        for (pi, _) in PREDICATES.iter().enumerate() {
            for qi in 0..queries.len() {
                jobs.push((arm.clone(), strat.clone(), pi, qi));
            }
        }
    }
    let began = Instant::now();
    let lines: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|w| {
                let (jobs, queries, gold, slmi, hnsw, dli, dli_buffer, ranks, lmi_mode) =
                    (&jobs, &queries, &gold, &slmi, &hnsw, &dli, &dli_buffer, &ranks, &lmi_mode);
                scope.spawn(move || {
                    let mut lines = Vec::new();
                    for (arm, strat, pi, qi) in jobs.iter().skip(w).step_by(threads) {
                        let (pname, max_rank) = PREDICATES[*pi];
                        let f = filter(max_rank);
                        let q = &queries[*qi];
                        let started = Instant::now();
                        let (found, work, plans) = match arm.as_str() {
                            "S-LMI" => {
                                let (h, w, plan, np) = lmi_search(slmi, q, &f, &lmi_mode(strat, 250));
                                (h, w, json!([{"plan": plan, "nprobe": np}]))
                            }
                            "DLI" => {
                                let (mut parts, mut total, mut plans) = (Vec::new(), 0, Vec::new());
                                for (level, seg) in dli {
                                    let (h, w, plan, np) = lmi_search(seg, q, &f, &lmi_mode(strat, ARITY.pow(*level as u32)));
                                    parts.push(h);
                                    total += w;
                                    plans.push(json!({"level": level, "plan": plan, "nprobe": np, "work": w}));
                                }
                                if let Some(buf) = dli_buffer {
                                    let (h, w) = plain_search(buf, q, &f, None, K);
                                    parts.push(h);
                                    total += w;
                                    plans.push(json!({"level": 0, "plan": "Plain", "work": w}));
                                }
                                (merge(parts), total, json!(plans))
                            }
                            _ => {
                                let ef = strat.rsplit('=').next().unwrap().parse::<usize>().unwrap();
                                let params = SearchParams { hnsw_ef: Some(ef), ..Default::default() };
                                let (h, w) = plain_search(hnsw, q, &f, Some(&params), K);
                                (h, w, json!([{"plan": "HnswFiltered", "ef": ef}]))
                            }
                        };
                        let ns = started.elapsed().as_nanos() as u64;
                        let g = &gold[*pi][*qi];
                        let mut ids: Vec<u64> = found.iter().map(|h| h.0).collect();
                        let returned = ids.len();
                        ids.sort_unstable();
                        ids.dedup();
                        let duplicates = returned - ids.len();
                        let nonqualifying = ids.iter().filter(|&&id| u64::from(ranks[(id - 1) as usize]) > max_rank).count();
                        let qualifying_returned = ids.len() - nonqualifying;
                        let hit = found.iter().filter(|h| g.contains(&h.0)).count();
                        lines.push(json!({"arm": arm, "strategy": strat, "predicate": pname, "query": qi,
                            "returned": found.iter().map(|h| h.0).collect::<Vec<_>>(), "recall30": hit as f64 / g.len().max(1) as f64,
                            "gold_size": g.len(), "underfill": K.saturating_sub(qualifying_returned),
                            "duplicates": duplicates, "nonqualifying": nonqualifying, "work": work,
                            "plans": plans, "elapsed_ns": ns}).to_string());
                    }
                    lines
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    std::fs::write(out.join("rows.jsonl"), lines.join("\n") + "\n").unwrap();
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&json!({"rows": n, "queries": queries.len(), "records": lines.len(),
            "search_seconds": began.elapsed().as_secs_f64(), "threads": threads})).unwrap(),
    )
    .unwrap();
    println!("Q7_DONE records={}", lines.len());
}

#[test]
fn q7_hierarchy_matches_the_bentley_saxe_schedule() {
    // 60,000 rows: 12 drains. Drains 1–10 fill level 1 (50,000); drain 11 overflows into level 2 (55,000);
    // drain 12 starts level 1 again.
    assert_eq!(hierarchy(60_000), (vec![(2, 1, 55_000), (1, 55_001, 60_000)], None));
    assert_eq!(hierarchy(6_000), (vec![(1, 1, 5_000)], Some((5_001, 6_000))));
    assert_eq!(hierarchy(4_000), (vec![], Some((1, 4_000))));
    let (m, _) = hierarchy(500_000);
    // 99 drains fill level 2 with 495,000 (every 11th drain merges a full level 1 into it); the 100th starts
    // level 1 again. This is PostgreSQL's ledger at batch 11 of the Phase 3 Shift plan (Q4).
    assert_eq!(m, vec![(2, 1, 495_000), (1, 495_001, 500_000)]);
}
