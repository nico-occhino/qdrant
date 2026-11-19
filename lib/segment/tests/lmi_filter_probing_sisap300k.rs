#![cfg(all(feature = "lmi-training", feature = "testing"))]

//! Harness for `research/lmi/filtering/PREREGISTRATION_FILTER_PROBING.md`.
//!
//! One ignored test, driven by environment variables, run phase by phase:
//!
//! | `LMI_PROBE_PHASE` | What it does |
//! |---|---|
//! | `prepare`  | Source segment (Float16, cosine, payloads `r` and `g`), native IVF and MLP builds, gold. |
//! | `grid`     | Counter and recall pass: every query, filter and cell; parallel; no latency. |
//! | `timing`   | Single-thread timing of every cell on the seeded timing subsamples. |
//! | `stages`   | Stage-cost pass (build with `lmi-research`), validation timing subsample. |
//! | `policies` | Native run of the frozen policies (`LMI_PROBE_POLICIES`), test timing subsample. |
//! | `b0repro`  | B0 on rows 0..999 against the committed 300K baseline result. |
//!
//! `LMI_PROBE_INPUTS` holds `vectors.f16` and `queries.f32`; `LMI_PROBE_WORK` receives
//! segments and records (outside Git). `LMI_PROBE_PILOT=1` restricts every phase to
//! 100 validation queries and three selectivities. `LMI_PROBE_THREADS` sets the
//! grid-pass parallelism (default: available cores − 2).

use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Write};
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
use segment::index::lmi_index::filtered::{FilterProbePolicy, FilteredSearchMode};
use segment::index::lmi_index::{FilteredPlan, LmiConfig};
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::json_path::JsonPath;
use segment::segment::Segment;
use segment::segment_constructor::load_segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
use segment::types::{
    Condition, Distance, ExtendedPointId, FieldCondition, Filter, HnswGlobalConfig, Indexes,
    Payload, PayloadSchemaType, Range, SearchParams, VectorStorageDatatype,
};
use serde_json::{Value, json};

const DIM: usize = 768;
const POINTS: usize = 300_000;
const TOP: usize = 10;
const SCALE: f64 = 1_000_000.0;
const P_GRID: [usize; 13] = [1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 128, 256];
const S_GRID: [f64; 12] = [
    1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0001,
];
const PILOT_S: [f64; 3] = [1.0, 0.03, 0.001];
/// The committed baseline's selectivities (B0 reproduction).
const BASELINE_S: [f64; 9] = [1.0, 0.5, 0.1, 0.05, 0.03, 0.01, 0.003, 0.001, 0.0001];
const FAMILIES: [&str; 2] = ["ind", "slab"];
const ROUTERS: [&str; 2] = ["ivf", "mlp"];
const TIMING_SAMPLE: usize = 500;
const WARMUP_QUERIES: usize = 50;
const REPETITIONS: usize = 3;

// ---------------------------------------------------------------- utilities

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
}

fn pilot() -> bool {
    std::env::var("LMI_PROBE_PILOT").is_ok_and(|v| v == "1")
}

fn selectivities() -> &'static [f64] {
    if pilot() { &PILOT_S } else { &S_GRID }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Independent payload: identical to the 300K baseline.
fn r_value(id: u64) -> u64 {
    splitmix(id) % SCALE as u64
}

/// Seeded Fisher–Yates shuffle.
fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed;
    for i in (1..items.len()).rev() {
        state = splitmix(state);
        items.swap(i, (state % (i as u64 + 1)) as usize);
    }
}

fn filter_for(family: &str, s: f64) -> Filter {
    let (key, range) = match family {
        "ind" => (
            "r",
            Range {
                lt: Some((s * SCALE).into()),
                ..Default::default()
            },
        ),
        "slab" => (
            "g",
            Range {
                gte: Some(((1.0 - s) * SCALE).into()),
                ..Default::default()
            },
        ),
        other => panic!("unknown family {other}"),
    };
    Filter::new_must(Condition::Field(FieldCondition::new_range(
        JsonPath::new(key),
        range,
    )))
}

fn matches(family: &str, s: f64, r: u64, g: u64) -> bool {
    match family {
        "ind" => (r as f64) < s * SCALE,
        _ => (g as f64) >= (1.0 - s) * SCALE,
    }
}

fn model() -> LmiConfig {
    LmiConfig {
        n_buckets: 548,
        sample_size: 32_768,
        hidden_dim: 512,
        epochs: 30,
        batch_size: 256,
        routing_batch_size: 256,
        kmeans_iterations: 5,
        nprobe: 16,
        seed: 42,
    }
}

fn read_queries(inputs: &Path) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(inputs.join("queries.f32")).unwrap();
    assert_eq!(bytes.len(), 10_000 * DIM * 4);
    bytes
        .chunks_exact(DIM * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect()
}

/// Query rows of a split; the pilot uses the first 100 validation rows.
fn split_rows(split: &str) -> Vec<usize> {
    match (split, pilot()) {
        ("val", false) => (1000..3000).collect(),
        ("test", false) => (4000..6000).collect(),
        ("val", true) => (1000..1100).collect(),
        ("test", true) => Vec::new(),
        ("repro", _) => (0..1000).collect(),
        _ => panic!("unknown split {split}"),
    }
}

fn timing_rows(split: &str) -> Vec<usize> {
    let mut rows = split_rows(split);
    shuffle(&mut rows, if split == "val" { 11 } else { 13 });
    rows.truncate(TIMING_SAMPLE);
    rows
}

fn fnv(hits: &[(u64, f32)]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for (id, score) in hits {
        for b in id
            .to_le_bytes()
            .iter()
            .chain(score.to_bits().to_le_bytes().iter())
        {
            hash ^= u64::from(*b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

fn external(segment: &Segment, hits: &[ScoredPointOffset]) -> Vec<(u64, f32)> {
    let tracker = segment.id_tracker.borrow();
    hits.iter()
        .map(|hit| match tracker.external_id(hit.idx).unwrap() {
            ExtendedPointId::NumId(n) => (n, hit.score),
            other => panic!("unexpected id {other:?}"),
        })
        .collect()
}

/// Recall@10 with k_eff = |gold| = min(10, live matches); `None` when nothing matches.
fn recall(found: &[(u64, f32)], gold: &[u64]) -> Option<f64> {
    if gold.is_empty() {
        return None;
    }
    let hit = found.iter().filter(|(id, _)| gold.contains(id)).count();
    Some(hit as f64 / gold.len() as f64)
}

fn write_jsonl(path: &Path, lines: &[String], append: bool) {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(append)
        .write(true)
        .truncate(!append)
        .open(path)
        .unwrap();
    let mut out = BufWriter::new(file);
    for line in lines {
        writeln!(out, "{line}").unwrap();
    }
    out.flush().unwrap();
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    let file = std::fs::File::open(path).unwrap();
    std::io::BufReader::new(file)
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect()
}

fn threads() -> usize {
    std::env::var("LMI_PROBE_THREADS").map_or_else(
        |_| {
            std::thread::available_parallelism()
                .map_or(4, |n| n.get())
                .saturating_sub(2)
                .max(1)
        },
        |v| v.parse().unwrap(),
    )
}

// ---------------------------------------------------------------- segments

struct Work {
    root: PathBuf,
    manifest: Value,
}

impl Work {
    fn open() -> Self {
        let root = env_path("LMI_PROBE_WORK");
        let manifest = std::fs::read(root.join("manifest.json"))
            .map(|bytes| serde_json::from_slice(&bytes).unwrap())
            .unwrap_or_else(|_| json!({}));
        Self { root, manifest }
    }

    fn load(&self, name: &str) -> Segment {
        let entry = &self.manifest["segments"][name];
        let path = PathBuf::from(entry["path"].as_str().unwrap());
        let uuid = uuid::Uuid::from_str(entry["uuid"].as_str().unwrap()).unwrap();
        load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap()
    }

    fn payloads(&self) -> (Vec<u64>, Vec<u64>) {
        let bytes = std::fs::read(self.root.join("payload_g.u32")).unwrap();
        let g: Vec<u64> = bytes
            .chunks_exact(4)
            .map(|b| u64::from(u32::from_le_bytes(b.try_into().unwrap())))
            .collect();
        let r = (1..=POINTS as u64).map(r_value).collect();
        (r, g)
    }

    fn gold(&self) -> HashMap<String, Vec<u64>> {
        let mut gold = HashMap::new();
        for record in read_jsonl(&self.root.join("gold.jsonl")) {
            let key = format!(
                "{}|{}|{}|{}",
                record["split"].as_str().unwrap(),
                record["family"].as_str().unwrap(),
                record["s"],
                record["q"],
            );
            let ids = record["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            gold.insert(key, ids);
        }
        gold
    }
}

fn gold_key(split: &str, family: &str, s: f64, q: usize) -> String {
    format!("{split}|{family}|{}|{q}", json!(s))
}

fn lmi(segment: &Segment) -> atomic_refcell::AtomicRef<'_, VectorIndexEnum> {
    segment.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
}

fn research(
    segment: &Segment,
    query: &QueryVector,
    filter: &Filter,
    mode: &FilteredSearchMode,
) -> (
    Vec<ScoredPointOffset>,
    segment::index::lmi_index::filtered::FilteredSearchRecord,
    u64,
) {
    let index = lmi(segment);
    let VectorIndexEnum::Lmi(index) = &*index else {
        panic!("expected trained LMI")
    };
    let began = Instant::now();
    let (hits, record) = index
        .filtered_search_research(query, filter, TOP, mode, &VectorQueryContext::default())
        .unwrap();
    let ns = began.elapsed().as_nanos() as u64;
    (hits, record, ns)
}

fn cell_mode(cell: &str) -> FilteredSearchMode {
    match cell {
        "exact" => FilteredSearchMode::ForceExact,
        p => FilteredSearchMode::ForceRouted {
            nprobe: p.parse().unwrap(),
        },
    }
}

fn cells() -> Vec<String> {
    P_GRID
        .iter()
        .map(|p| p.to_string())
        .chain(std::iter::once("exact".to_string()))
        .collect()
}

// ---------------------------------------------------------------- phases

fn prepare() {
    let inputs = env_path("LMI_PROBE_INPUTS");
    let root = env_path("LMI_PROBE_WORK");
    std::fs::create_dir_all(&root).unwrap();
    let began = Instant::now();
    let vectors = std::fs::read(inputs.join("vectors.f16")).unwrap();
    assert_eq!(vectors.len(), POINTS * DIM * 2);
    let rows: Vec<Vec<f32>> = vectors
        .chunks_exact(DIM * 2)
        .map(|row| {
            row.chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()
        })
        .collect();
    drop(vectors);

    // Correlated payload: quantile rank of the normalized vector's projection
    // on a fixed random unit direction (seed 7).
    let direction: Vec<f32> = {
        let mut state = 7_u64;
        let raw: Vec<f32> = (0..DIM)
            .map(|_| {
                state = splitmix(state);
                (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect();
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    };
    let projection: Vec<f64> = rows
        .iter()
        .map(|v| {
            let norm = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            v.iter()
                .zip(&direction)
                .map(|(x, d)| f64::from(*x) * f64::from(*d))
                .sum::<f64>()
                / norm
        })
        .collect();
    let mut order: Vec<usize> = (0..POINTS).collect();
    order.sort_by(|&a, &b| projection[a].total_cmp(&projection[b]).then(a.cmp(&b)));
    let mut g = vec![0_u32; POINTS];
    for (rank, &i) in order.iter().enumerate() {
        g[i] = (rank as f64 * SCALE / POINTS as f64) as u32;
    }
    std::fs::write(
        root.join("payload_g.u32"),
        g.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();

    // Source segment.
    let source_root = root.join("segments/source");
    std::fs::create_dir_all(&source_root).unwrap();
    let mut source = {
        let plain = build_simple_segment(&source_root, DIM, Distance::Cosine).unwrap();
        let mut cfg = plain.config().clone();
        cfg.vector_data
            .get_mut(DEFAULT_VECTOR_NAME)
            .unwrap()
            .datatype = Some(VectorStorageDatatype::Float16);
        let stale = plain.segment_path.clone();
        drop(plain);
        std::fs::remove_dir_all(stale).unwrap();
        segment::segment_constructor::build_segment(&source_root, &cfg, None, true)
            .unwrap()
            .0
    };
    for (i, row) in rows.iter().enumerate() {
        let id = i as u64 + 1;
        source
            .upsert_point(id, id.into(), only_default_vector(row))
            .unwrap();
        let payload: Payload =
            serde_json::from_value(json!({ "r": r_value(id), "g": g[i] })).unwrap();
        source
            .set_full_payload(POINTS as u64 + id, id.into(), &payload)
            .unwrap();
    }
    for (op, key) in [(3 * POINTS as u64, "r"), (3 * POINTS as u64 + 1, "g")] {
        source
            .create_field_index(
                op,
                &JsonPath::from_str(key).unwrap(),
                Some(&PayloadSchemaType::Integer.into()),
            )
            .unwrap();
    }
    source.flush(true).unwrap();
    let source_seconds = began.elapsed().as_secs_f64();

    // Native builds, one per router, persisted.
    let mut segments = json!({
        "source": { "path": source.segment_path, "uuid": source.uuid.to_string() },
    });
    let mut builds = json!({});
    for router in ROUTERS {
        let research_router = if router == "ivf" { "centroid" } else { "mlp" };
        // SAFETY: single-threaded phase; the build reads it once.
        unsafe { std::env::set_var("LMI_RESEARCH_ROUTER", research_router) };
        let staging = tempfile::tempdir_in(&root).unwrap();
        let target_root = root.join(format!("segments/{router}"));
        std::fs::create_dir_all(&target_root).unwrap();
        let mut cfg = source.config().clone();
        cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = Indexes::LmiTrained(model());
        let began = Instant::now();
        let mut builder = SegmentBuilder::new(
            staging.path(),
            &cfg,
            &HnswGlobalConfig::default(),
            FeatureFlags::default(),
        )
        .unwrap();
        builder.update(&[&source], &AtomicBool::new(false)).unwrap();
        let built = builder.build_for_test(&target_root);
        let seconds = began.elapsed().as_secs_f64();
        unsafe { std::env::remove_var("LMI_RESEARCH_ROUTER") };
        segments[router] = json!({ "path": built.segment_path, "uuid": built.uuid.to_string() });
        builds[router] = json!({ "seconds": seconds });
        eprintln!("{router}: built in {seconds:.1} s");
    }

    // Gold: native exact filtered search on the source, parallel over queries.
    let queries = read_queries(&inputs);
    let exact = SearchParams {
        exact: true,
        ..Default::default()
    };
    let mut jobs: Vec<(&str, &str, f64, usize)> = Vec::new();
    for split in ["val", "test", "repro"] {
        for family in FAMILIES {
            if split == "repro" && family != "ind" {
                continue;
            }
            let grid: &[f64] = if split == "repro" {
                &BASELINE_S
            } else {
                selectivities()
            };
            for &s in grid {
                for q in split_rows(split) {
                    jobs.push((split, family, s, q));
                }
            }
        }
    }
    let began = Instant::now();
    let workers = threads();
    let lines: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let (jobs, queries, source, exact) = (&jobs, &queries, &source, &exact);
                scope.spawn(move || {
                    let mut lines = Vec::new();
                    for (split, family, s, q) in jobs.iter().skip(w).step_by(workers) {
                        let filter = filter_for(family, *s);
                        let query: QueryVector = queries[*q].clone().into();
                        let hits = lmi(source)
                            .search(
                                &[&query],
                                Some(&filter),
                                TOP,
                                Some(exact),
                                &VectorQueryContext::default(),
                            )
                            .unwrap()
                            .remove(0);
                        let ids: Vec<u64> =
                            external(source, &hits).iter().map(|(id, _)| *id).collect();
                        lines.push(
                            json!({ "split": split, "family": family, "s": s, "q": q, "ids": ids })
                                .to_string(),
                        );
                    }
                    lines
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    write_jsonl(&root.join("gold.jsonl"), &lines, false);
    let gold_seconds = began.elapsed().as_secs_f64();

    let manifest = json!({
        "segments": segments,
        "builds": builds,
        "model": model(),
        "pilot": pilot(),
        "source_seconds": source_seconds,
        "gold_seconds": gold_seconds,
        "gold_records": lines.len(),
        "threads": workers,
    });
    std::fs::write(
        root.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    eprintln!(
        "prepare: source {source_seconds:.1} s, gold {gold_seconds:.1} s ({} records)",
        lines.len()
    );
}

/// Per-filter planner inputs: estimate, true matches and break-even decisions at every grid p.
fn filter_records(work: &Work, segment: &Segment, router: &str) -> Vec<String> {
    let (r, g) = work.payloads();
    let mut lines = Vec::new();
    for family in FAMILIES {
        for &s in selectivities() {
            let filter = filter_for(family, s);
            let m = (0..POINTS)
                .filter(|&i| matches(family, s, r[i], g[i]))
                .count();
            let index = lmi(segment);
            let VectorIndexEnum::Lmi(index) = &*index else {
                panic!("expected trained LMI")
            };
            let estimate = index.filter_estimate(&filter).unwrap();
            let decisions: Vec<Value> = P_GRID
                .iter()
                .map(|&p| {
                    let d = index.filtered_plan_at(&filter, p).unwrap().unwrap();
                    json!({ "p": p, "plan": d.plan, "threshold": d.threshold, "sampled": d.sampled })
                })
                .collect();
            lines.push(
                json!({
                    "router": router, "family": family, "s": s, "matching": m,
                    "estimate": estimate, "break_even": decisions,
                })
                .to_string(),
            );
        }
    }
    lines
}

fn grid() {
    let work = Work::open();
    let inputs = env_path("LMI_PROBE_INPUTS");
    let queries = read_queries(&inputs);
    let gold = work.gold();
    let out = work.root.join("grid");
    std::fs::create_dir_all(&out).unwrap();
    let records_path = out.join("records.jsonl");
    let _ = std::fs::remove_file(&records_path);
    let mut filter_lines = Vec::new();
    let cells = cells();
    let workers = threads();
    let mut summary = json!({});
    for router in ROUTERS {
        let segment = work.load(router);
        filter_lines.extend(filter_records(&work, &segment, router));
        for family in FAMILIES {
            let began = Instant::now();
            let mut jobs: Vec<(&str, f64, usize)> = Vec::new();
            for split in ["val", "test"] {
                for &s in selectivities() {
                    for q in split_rows(split) {
                        jobs.push((split, s, q));
                    }
                }
            }
            let lines: Vec<String> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..workers)
                    .map(|w| {
                        let (jobs, queries, segment, gold, cells) =
                            (&jobs, &queries, &segment, &gold, &cells);
                        scope.spawn(move || {
                            let mut lines = Vec::new();
                            for (split, s, q) in jobs.iter().skip(w).step_by(workers) {
                                let filter = filter_for(family, *s);
                                let query: QueryVector = queries[*q].clone().into();
                                let gold = &gold[&gold_key(split, family, *s, *q)];
                                for cell in cells {
                                    let (hits, record, _) =
                                        research(segment, &query, &filter, &cell_mode(cell));
                                    let found = external(segment, &hits);
                                    assert!(
                                        found.len() == gold.len(),
                                        "fill: {router} {family} {s} {q} {cell}: {} vs {}",
                                        found.len(),
                                        gold.len()
                                    );
                                    lines.push(
                                        json!({
                                            "router": router, "family": family, "split": split,
                                            "s": s, "q": q, "cell": cell,
                                            "k_eff": gold.len(),
                                            "recall": recall(&found, gold),
                                            "hash": fnv(&found),
                                            "plan": record.plan,
                                            "routed": record.routed,
                                        })
                                        .to_string(),
                                    );
                                }
                            }
                            lines
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect()
            });
            write_jsonl(&records_path, &lines, true);
            let seconds = began.elapsed().as_secs_f64();
            summary[format!("{router}/{family}")] =
                json!({ "records": lines.len(), "seconds": seconds });
            eprintln!(
                "grid {router}/{family}: {} records in {seconds:.1} s",
                lines.len()
            );
        }
    }
    write_jsonl(&out.join("filters.jsonl"), &filter_lines, false);
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
}

/// Single-thread timing of `modes` over the timing subsample of `split`.
/// Warm-up: every mode over the first `WARMUP_QUERIES` rows, discarded. Then
/// `REPETITIONS` passes, each over all modes in a seeded shuffled order.
fn time_modes(
    segment: &Segment,
    queries: &[Vec<f32>],
    rows: &[usize],
    filter: &Filter,
    modes: &[(String, FilteredSearchMode)],
    seed: u64,
    mut emit: impl FnMut(
        &str,
        usize,
        usize,
        u64,
        &[(u64, f32)],
        &segment::index::lmi_index::filtered::FilteredSearchRecord,
    ),
) {
    let prepared: Vec<QueryVector> = rows.iter().map(|&q| queries[q].clone().into()).collect();
    for (_, mode) in modes {
        for query in prepared.iter().take(WARMUP_QUERIES) {
            let _ = research(segment, query, filter, mode);
        }
    }
    for rep in 0..REPETITIONS {
        let mut order: Vec<usize> = (0..modes.len()).collect();
        shuffle(&mut order, seed.wrapping_add(rep as u64));
        for &m in &order {
            let (name, mode) = &modes[m];
            for (query, &q) in prepared.iter().zip(rows) {
                let (hits, record, ns) = research(segment, query, filter, mode);
                emit(name, rep, q, ns, &external(segment, &hits), &record);
            }
        }
    }
}

fn timing() {
    let work = Work::open();
    let queries = read_queries(&env_path("LMI_PROBE_INPUTS"));
    let out = work.root.join("timing");
    std::fs::create_dir_all(&out).unwrap();
    let path = out.join("timing.jsonl");
    let _ = std::fs::remove_file(&path);
    let modes: Vec<(String, FilteredSearchMode)> = cells()
        .into_iter()
        .map(|c| {
            let m = cell_mode(&c);
            (c, m)
        })
        .collect();
    let splits: &[&str] = if pilot() { &["val"] } else { &["val", "test"] };
    for router in ROUTERS {
        let segment = work.load(router);
        for family in FAMILIES {
            let began = Instant::now();
            let mut lines = Vec::new();
            for (si, &s) in selectivities().iter().enumerate() {
                let filter = filter_for(family, s);
                for split in splits {
                    let rows = timing_rows(split);
                    let seed = splitmix(
                        (si as u64) << 8
                            | u64::from(router == "mlp") << 4
                            | u64::from(family == "slab"),
                    );
                    time_modes(
                        &segment,
                        &queries,
                        &rows,
                        &filter,
                        &modes,
                        seed,
                        |cell, rep, q, ns, _, _| {
                            lines.push(
                                json!({
                                    "router": router, "family": family, "split": split, "s": s,
                                    "q": q, "cell": cell, "rep": rep, "ns": ns,
                                })
                                .to_string(),
                            );
                        },
                    );
                }
            }
            write_jsonl(&path, &lines, true);
            eprintln!(
                "timing {router}/{family}: {} records in {:.1} s",
                lines.len(),
                began.elapsed().as_secs_f64()
            );
        }
    }
}

fn stages() {
    assert!(
        cfg!(feature = "lmi-research"),
        "the stage pass needs a build with --features lmi-research"
    );
    let work = Work::open();
    let queries = read_queries(&env_path("LMI_PROBE_INPUTS"));
    let out = work.root.join("stages");
    std::fs::create_dir_all(&out).unwrap();
    let path = out.join("stages.jsonl");
    let _ = std::fs::remove_file(&path);
    let rows = timing_rows("val");
    for router in ROUTERS {
        let segment = work.load(router);
        for family in FAMILIES {
            let began = Instant::now();
            let mut lines = Vec::new();
            for &s in selectivities() {
                let filter = filter_for(family, s);
                for &p in &P_GRID {
                    let mode = FilteredSearchMode::ForceRouted { nprobe: p };
                    let prepared: Vec<QueryVector> =
                        rows.iter().map(|&q| queries[q].clone().into()).collect();
                    for query in prepared.iter().take(WARMUP_QUERIES) {
                        let _ = research(&segment, query, &filter, &mode);
                    }
                    for (query, &q) in prepared.iter().zip(&rows) {
                        let (_, record, ns) = research(&segment, query, &filter, &mode);
                        lines.push(
                            json!({
                                "router": router, "family": family, "s": s, "q": q,
                                "cell": p.to_string(), "ns": ns, "routed": record.routed,
                            })
                            .to_string(),
                        );
                    }
                }
            }
            write_jsonl(&path, &lines, true);
            eprintln!(
                "stages {router}/{family}: {} records in {:.1} s",
                lines.len(),
                began.elapsed().as_secs_f64()
            );
        }
    }
}

fn policies() {
    let work = Work::open();
    let queries = read_queries(&env_path("LMI_PROBE_INPUTS"));
    let spec: Value =
        serde_json::from_slice(&std::fs::read(env_path("LMI_PROBE_POLICIES")).unwrap()).unwrap();
    // Simulation reference: grid hashes of the test split.
    // The pilot has no test split; it exercises this phase on validation rows.
    let eval_split = if pilot() { "val" } else { "test" };
    let mut grid_hash: HashMap<String, u64> = HashMap::new();
    for record in read_jsonl(&work.root.join("grid/records.jsonl")) {
        if record["split"] != eval_split {
            continue;
        }
        let key = format!(
            "{}|{}|{}|{}|{}",
            record["router"].as_str().unwrap(),
            record["family"].as_str().unwrap(),
            record["s"],
            record["q"],
            record["cell"].as_str().unwrap()
        );
        grid_hash.insert(key, record["hash"].as_u64().unwrap());
    }
    let gold = work.gold();
    let out = work.root.join("policies");
    std::fs::create_dir_all(&out).unwrap();
    let path = out.join("policies.jsonl");
    let _ = std::fs::remove_file(&path);
    let rows = timing_rows(eval_split);
    let mut mismatches = 0_usize;
    for router in ROUTERS {
        let segment = work.load(router);
        let modes: Vec<(String, FilteredSearchMode)> = spec["policies"][router]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, policy)| {
                let policy: FilterProbePolicy = serde_json::from_value(policy.clone()).unwrap();
                (name.clone(), FilteredSearchMode::Policy(policy))
            })
            .collect();
        for family in FAMILIES {
            let began = Instant::now();
            let mut lines = Vec::new();
            for (si, &s) in selectivities().iter().enumerate() {
                let filter = filter_for(family, s);
                let seed = splitmix(
                    0xA5 ^ (si as u64) << 8
                        | u64::from(router == "mlp") << 4
                        | u64::from(family == "slab"),
                );
                time_modes(
                    &segment,
                    &queries,
                    &rows,
                    &filter,
                    &modes,
                    seed,
                    |name, rep, q, ns, found, record| {
                        let cell = match record.plan {
                            FilteredPlan::ExactMatches => "exact".to_string(),
                            FilteredPlan::RoutedIntersect => record.nprobe.to_string(),
                        };
                        let key = format!("{router}|{family}|{}|{q}|{cell}", json!(s));
                        let simulated = grid_hash.get(&key).copied();
                        let matched = simulated == Some(fnv(found));
                        if !matched {
                            mismatches += 1;
                        }
                        lines.push(
                            json!({
                                "router": router, "family": family, "s": s, "q": q, "policy": name,
                                "rep": rep, "ns": ns, "cell": cell, "plan": record.plan,
                                "split": eval_split,
                                "recall": recall(found, &gold[&gold_key(eval_split, family, s, q)]),
                                "matches_simulation": matched,
                                "decision": record.policy, "routed": record.routed,
                            })
                            .to_string(),
                        );
                    },
                );
            }
            write_jsonl(&path, &lines, true);
            eprintln!(
                "policies {router}/{family}: {} records in {:.1} s",
                lines.len(),
                began.elapsed().as_secs_f64()
            );
        }
    }
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&json!({ "simulation_mismatches": mismatches })).unwrap(),
    )
    .unwrap();
    eprintln!("policies: {mismatches} native/simulation mismatches");
}

fn b0repro() {
    let work = Work::open();
    let queries = read_queries(&env_path("LMI_PROBE_INPUTS"));
    let gold = work.gold();
    let baseline: Value =
        serde_json::from_slice(&std::fs::read(env_path("LMI_PROBE_BASELINE")).unwrap()).unwrap();
    let rows = split_rows("repro");
    let mut report = Vec::new();
    let mut mismatches = 0;
    for (router, label) in [("ivf", "centroid"), ("mlp", "mlp")] {
        let segment = work.load(router);
        let system = baseline["systems"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["router"] == label)
            .unwrap();
        for (si, &s) in BASELINE_S.iter().enumerate() {
            let filter = filter_for("ind", s);
            let expected = &system["filtered"][si];
            assert_eq!(expected["selectivity"].as_f64().unwrap(), s);
            let decision = {
                let index = lmi(&segment);
                let VectorIndexEnum::Lmi(index) = &*index else {
                    panic!("expected trained LMI")
                };
                index.filtered_plan(&filter).unwrap().unwrap()
            };
            let (mut total, mut widenings) = (0.0, 0_usize);
            for &q in &rows {
                let query: QueryVector = queries[q].clone().into();
                let (hits, record, _) =
                    research(&segment, &query, &filter, &FilteredSearchMode::Baseline);
                let production = lmi(&segment)
                    .search(
                        &[&query],
                        Some(&filter),
                        TOP,
                        None,
                        &VectorQueryContext::default(),
                    )
                    .unwrap()
                    .remove(0);
                assert_eq!(hits, production, "research Baseline must equal production");
                let found = external(&segment, &hits);
                total += recall(&found, &gold[&gold_key("repro", "ind", s, q)]).unwrap();
                widenings += record.routed.map_or(0, |r| r.widenings);
            }
            let mean = total / rows.len() as f64;
            let widen = widenings as f64 / rows.len() as f64;
            let same_plan = json!(decision) == expected["decision"];
            let same_recall = (mean
                - expected["recall_at_10_vs_native_exact_filtered"]
                    .as_f64()
                    .unwrap())
            .abs()
                < 1e-12;
            let same_widen =
                (widen - expected["widenings_per_query"].as_f64().unwrap()).abs() < 1e-12;
            if !(same_plan && same_recall && same_widen) {
                mismatches += 1;
            }
            report.push(json!({
                "router": router, "s": s, "recall": mean, "expected_recall": expected["recall_at_10_vs_native_exact_filtered"],
                "decision": decision, "expected_decision": expected["decision"], "widenings_per_query": widen,
                "same_plan": same_plan, "same_recall": same_recall, "same_widenings": same_widen,
            }));
        }
    }
    std::fs::write(
        work.root.join("b0repro.json"),
        serde_json::to_vec_pretty(&json!({ "mismatches": mismatches, "rows": report })).unwrap(),
    )
    .unwrap();
    eprintln!("b0repro: {mismatches} mismatching cells");
    assert_eq!(mismatches, 0, "B0 must reproduce the committed baseline");
}

#[test]
#[ignore = "filtered-probing study; see the module docs for LMI_PROBE_* variables"]
fn lmi_filter_probing_sisap300k() {
    let _scope = common::ambient::unmeasured_guard(common::reason::reason("LMI filter probing"));
    let phase = std::env::var("LMI_PROBE_PHASE").expect("set LMI_PROBE_PHASE");
    let began = Instant::now();
    match phase.as_str() {
        "prepare" => prepare(),
        "grid" => grid(),
        "timing" => timing(),
        "stages" => stages(),
        "policies" => policies(),
        "b0repro" => b0repro(),
        other => panic!("unknown phase {other}"),
    }
    eprintln!(
        "phase {phase} done in {:.1} s",
        began.elapsed().as_secs_f64()
    );
}
