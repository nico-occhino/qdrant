//! Native Qdrant v1.20 index-level ANN benchmark: Plain, HNSW and LMI (MLP or centroid router).
//!
//! Research harness, ignored by default. One phase per invocation, selected by `ANN_PHASE`:
//!
//! * `source`: ingest `ANN_N` rows of a little-endian Float16 corpus (`ANN_CORPUS`) into one
//!   appendable Plain Float16 segment under `ANN_ROOT/source` (ids 1..=N).
//! * `build`: build `ANN_ROOT/build-$ANN_NAME` from the source with `ANN_INDEX` (JSON:
//!   `{"plain":{}}`, `{"hnsw":{"m":16,"ef_construct":100,"threads":16}}` or `{"lmi":LmiConfig}`,
//!   optionally with a top-level `"threads"` CPU permit for LMI)
//!   through `SegmentBuilder`, recording wall time, index bytes and peak RSS.
//! * `search`: load a built segment and time index-level search for every setting in
//!   `ANN_SETTINGS` (nprobe for LMI, ef for HNSW, ignored for Plain) over raw Float32 queries
//!   (`ANN_QUERIES`, unit-normalized rows), `ANN_REPS` timed repetitions after one untimed
//!   warm pass, then throughput at each thread count in `ANN_THREADS`.
//!
//! The LMI path is the native learned path decomposed so nprobe can vary per call: router
//! `top_buckets`, posting union (sorted, deduplicated), then Qdrant `score_candidates`. It
//! omits only the underfill widening, which the analysis checks for (result length).
//! Output files are created with `create_new`; nothing existing is overwritten.

use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::flags::FeatureFlags;
use half::f16;
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use segment::entry::entry_point::{ReadSegmentEntry, SegmentEntry, StorageSegmentEntry};
use segment::id_tracker::IdTrackerRead;
use segment::index::candidate_scoring::score_candidates;
use segment::index::lmi_index::LmiConfig;
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::{build_segment, load_segment};
use segment::types::{
    Distance, ExtendedPointId, HnswConfig, HnswGlobalConfig, Indexes, SearchParams, SegmentConfig,
    VectorDataConfig, VectorStorageDatatype, VectorStorageType,
};

const DIM: usize = 768;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} required"))
}

fn peak_rss_kib() -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn dir_bytes(path: &Path, filter: &dyn Fn(&Path) -> bool) -> u64 {
    let mut total = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for e in entries.flatten() {
            let p = e.path();
            let m = e.metadata().unwrap();
            if m.is_dir() {
                total += dir_bytes(&p, filter);
            } else if filter(&p) {
                total += m.len();
            }
        }
    }
    total
}

fn write_new(path: &Path, value: &serde_json::Value) {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    writeln!(f, "{value}").unwrap();
}

fn only_segment_dir(root: &Path) -> PathBuf {
    let dirs: Vec<PathBuf> = fs::read_dir(root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("segment.json").exists())
        .collect();
    assert_eq!(
        dirs.len(),
        1,
        "expected exactly one segment under {}",
        root.display()
    );
    dirs.into_iter().next().unwrap()
}

fn load(root: &Path) -> Segment {
    let dir = only_segment_dir(root);
    let uuid = dir.file_name().unwrap().to_str().unwrap().parse().unwrap();
    load_segment(&dir, uuid, None, &AtomicBool::new(false), true).unwrap()
}

fn source_config() -> SegmentConfig {
    SegmentConfig {
        vector_data: HashMap::from([(
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
    }
}

fn phase_source(root: &Path) {
    let out = root.join("source");
    fs::create_dir(&out).expect("source must not exist");
    let n: usize = env("ANN_N").parse().unwrap();
    let corpus = PathBuf::from(env("ANN_CORPUS"));
    assert!(fs::metadata(&corpus).unwrap().len() as usize >= n * DIM * 2);
    let (mut segment, _) = build_segment(&out, &source_config(), None, true).unwrap();
    let mut input = BufReader::with_capacity(1 << 24, fs::File::open(&corpus).unwrap());
    let mut raw = vec![0u8; DIM * 2];
    let started = Instant::now();
    for id in 1..=n as u64 {
        input.read_exact(&mut raw).unwrap();
        let v: Vec<f32> = raw
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect();
        segment
            .upsert_point(id, ExtendedPointId::NumId(id), only_default_vector(&v))
            .unwrap();
        if id % 1_000_000 == 0 {
            println!(
                "SOURCE_PROGRESS rows={id} seconds={:.1}",
                started.elapsed().as_secs_f64()
            );
        }
    }
    segment.flush(true).unwrap();
    write_new(
        &out.join("source.json"),
        &serde_json::json!({"rows": n, "seconds": started.elapsed().as_secs_f64(),
            "available": segment.available_point_count(), "peak_rss_kib": peak_rss_kib(),
            "corpus": corpus}),
    );
}

fn phase_build(root: &Path) {
    let name = env("ANN_NAME");
    let out = root.join(format!("build-{name}"));
    fs::create_dir(&out).expect("build directory must not exist");
    let spec: serde_json::Value = serde_json::from_str(&env("ANN_INDEX")).unwrap();
    let source = load(&root.join("source"));
    let mut config = source.config().clone();
    let vector = config.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap();
    vector.storage_type = VectorStorageType::Mmap;
    let mut threads = 1usize;
    vector.index = if let Some(h) = spec.get("hnsw") {
        threads = h["threads"].as_u64().unwrap() as usize;
        Indexes::Hnsw(HnswConfig {
            m: h["m"].as_u64().unwrap() as usize,
            ef_construct: h["ef_construct"].as_u64().unwrap() as usize,
            full_scan_threshold: 1,
            max_indexing_threads: threads,
            ..HnswConfig::default()
        })
    } else if let Some(l) = spec.get("lmi") {
        let cfg: LmiConfig = serde_json::from_value(l.clone()).unwrap();
        // Optional top-level "threads": the CPU permit (parallel KMeans assignment).
        if let Some(t) = spec.get("threads") {
            threads = t.as_u64().unwrap() as usize;
        }
        Indexes::LmiTrained(cfg)
    } else {
        Indexes::Plain {}
    };
    let staging = tempfile::tempdir_in(&out).unwrap();
    let started = Instant::now();
    let mut builder = SegmentBuilder::new(
        staging.path(),
        &config,
        &HnswGlobalConfig::default(),
        FeatureFlags::default(),
    )
    .unwrap();
    builder.update(&[&source], &AtomicBool::new(false)).unwrap();
    let copied = started.elapsed().as_secs_f64();
    drop(source);
    let segment_root = out.join("segment");
    fs::create_dir(&segment_root).unwrap();
    let built = {
        builder
            .build(
                &segment_root,
                uuid::Uuid::new_v4(),
                None,
                true,
                common::budget::ResourcePermit::dummy(threads as u32),
                &AtomicBool::new(false),
                &mut rand::rng(),
                common::progress_tracker::ProgressTracker::new_for_test(),
            )
            .unwrap()
    };
    let seconds = started.elapsed().as_secs_f64();
    let dir = only_segment_dir(&segment_root);
    // Index-only bytes: files under the segment's vector index directory (not vector storage).
    let index_bytes = dir_bytes(&dir, &|p: &Path| {
        let rel = p.strip_prefix(&dir).unwrap().to_string_lossy().to_string();
        rel.contains("vector_index")
    });
    let listing: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    let total_bytes = dir_bytes(&dir, &|_| true);
    write_new(
        &out.join("build.json"),
        &serde_json::json!({"name": name, "spec": spec, "seconds_total": seconds,
            "seconds_copy": copied, "seconds_index": seconds - copied, "threads": threads,
            "index_bytes": index_bytes, "segment_bytes": total_bytes, "segment_entries": listing,
            "available": built.available_point_count(), "peak_rss_kib": peak_rss_kib()}),
    );
    println!("BUILD_DONE name={name} seconds={seconds:.1} index_bytes={index_bytes}");
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Plain,
    Hnsw,
    Lmi,
}

struct Outcome {
    ids: Vec<u64>,
    candidates: usize,
    route_ns: u64,
}

fn search_once(segment: &Segment, kind: Kind, setting: usize, q: &[f32], top: usize) -> Outcome {
    let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
    let index = data.vector_index.borrow();
    let tracker = segment.id_tracker.borrow();
    let query: QueryVector = q.to_vec().into();
    let ctx = VectorQueryContext::default();
    let (hits, candidates, route_ns) = match (kind, &*index) {
        (Kind::Lmi, VectorIndexEnum::Lmi(lmi)) => {
            let storage = data.vector_storage.borrow();
            let state = lmi.routing_state().expect("LMI routing state");
            let started = Instant::now();
            let normalized = Distance::Cosine.preprocess_vector::<f32>(q.to_vec());
            let buckets = state.router().top_buckets(&normalized, setting).unwrap();
            let mut offsets = Vec::new();
            for b in buckets {
                offsets.extend_from_slice(state.postings().get(b).unwrap());
            }
            offsets.sort_unstable();
            offsets.dedup();
            let route_ns = started.elapsed().as_nanos() as u64;
            let count = offsets.len();
            let hits = score_candidates(&query, &offsets, top, &tracker, &storage, &ctx).unwrap();
            (hits, count, route_ns)
        }
        (Kind::Hnsw, VectorIndexEnum::Hnsw(hnsw)) => {
            let params = SearchParams {
                hnsw_ef: Some(setting),
                ..Default::default()
            };
            (
                hnsw.search(&[&query], None, top, Some(&params), &ctx)
                    .unwrap()
                    .remove(0),
                0,
                0,
            )
        }
        (Kind::Plain, VectorIndexEnum::Plain(plain)) => (
            plain
                .search(&[&query], None, top, None, &ctx)
                .unwrap()
                .remove(0),
            segment.available_point_count(),
            0,
        ),
        _ => panic!("index kind does not match ANN_KIND"),
    };
    let ids = hits
        .iter()
        .map(|h| match tracker.external_id(h.idx).unwrap() {
            ExtendedPointId::NumId(id) => id,
            other => panic!("unexpected id {other:?}"),
        })
        .collect();
    Outcome {
        ids,
        candidates,
        route_ns,
    }
}

fn phase_search(root: &Path) {
    let name = env("ANN_NAME");
    let out = root.join(format!("search-{name}-{}", env("ANN_TAG")));
    fs::create_dir(&out).expect("search directory must not exist");
    let kind = match env("ANN_KIND").as_str() {
        "plain" => Kind::Plain,
        "hnsw" => Kind::Hnsw,
        "lmi" => Kind::Lmi,
        k => panic!("unknown kind {k}"),
    };
    let settings: Vec<usize> = env("ANN_SETTINGS")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let reps: usize = env("ANN_REPS").parse().unwrap();
    let threads: Vec<usize> = env("ANN_THREADS")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let top: usize = std::env::var("ANN_TOP").map_or(30, |v| v.parse().unwrap());
    let raw = fs::read(env("ANN_QUERIES")).unwrap();
    let queries: Vec<Vec<f32>> = raw
        .chunks_exact(DIM * 4)
        .map(|r| {
            r.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect();
    let load_started = Instant::now();
    let segment = load(&root.join(format!("build-{name}")).join("segment"));
    let load_seconds = load_started.elapsed().as_secs_f64();
    // Untimed warm pass over all queries at every setting (vectors and index pages in memory).
    let warm_started = Instant::now();
    for &s in &settings {
        for q in &queries {
            search_once(&segment, kind, s, q, top);
        }
    }
    let warm_seconds = warm_started.elapsed().as_secs_f64();
    let mut rows = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("queries.jsonl"))
        .unwrap();
    let mut summary = Vec::new();
    for &s in &settings {
        for rep in 0..reps {
            let mut lat = Vec::with_capacity(queries.len());
            for (qi, q) in queries.iter().enumerate() {
                let started = Instant::now();
                let o = search_once(&segment, kind, s, q, top);
                let ns = started.elapsed().as_nanos() as u64;
                lat.push(ns);
                if rep == 0 {
                    writeln!(
                        rows,
                        "{}",
                        serde_json::json!({"setting": s, "q": qi, "ns": ns,
                        "route_ns": o.route_ns, "candidates": o.candidates, "ids": o.ids})
                    )
                    .unwrap();
                } else {
                    writeln!(
                        rows,
                        "{}",
                        serde_json::json!({"setting": s, "q": qi, "rep": rep, "ns": ns})
                    )
                    .unwrap();
                }
            }
            lat.sort_unstable();
            let pct = |p: f64| lat[((lat.len() - 1) as f64 * p).round() as usize];
            summary.push(serde_json::json!({"setting": s, "rep": rep, "p50_ns": pct(0.5),
                "p95_ns": pct(0.95), "p99_ns": pct(0.99), "mean_ns": lat.iter().sum::<u64>() / lat.len() as u64}));
            println!(
                "SEARCH {name} setting={s} rep={rep} p50_ms={:.3}",
                pct(0.5) as f64 / 1e6
            );
        }
    }
    let mut throughput = Vec::new();
    for &s in &settings {
        for &t in &threads {
            let started = Instant::now();
            std::thread::scope(|scope| {
                for w in 0..t {
                    let segment = &segment;
                    let queries = &queries;
                    scope.spawn(move || {
                        for q in queries.iter().skip(w).step_by(t) {
                            search_once(segment, kind, s, q, top);
                        }
                    });
                }
            });
            let secs = started.elapsed().as_secs_f64();
            throughput.push(
                serde_json::json!({"setting": s, "threads": t, "qps": queries.len() as f64 / secs}),
            );
            println!(
                "THROUGHPUT {name} setting={s} threads={t} qps={:.1}",
                queries.len() as f64 / secs
            );
        }
    }
    write_new(
        &out.join("summary.json"),
        &serde_json::json!({"name": name, "kind": env("ANN_KIND"), "queries": queries.len(),
            "settings": settings, "reps": reps, "top": top, "load_seconds": load_seconds,
            "warm_seconds": warm_seconds, "latency": summary, "throughput": throughput,
            "peak_rss_kib": peak_rss_kib()}),
    );
}

#[test]
#[ignore = "explicit 10M ANN benchmark; requires ANN_PHASE and ANN_ROOT"]
fn ann_matched_v120() {
    let root = PathBuf::from(env("ANN_ROOT"));
    match env("ANN_PHASE").as_str() {
        "source" => phase_source(&root),
        "build" => phase_build(&root),
        "search" => phase_search(&root),
        p => panic!("unknown phase {p}"),
    }
}
