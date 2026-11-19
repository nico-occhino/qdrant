//! Native harness for `research/lmi/supervised_routing/PREREGISTRATION.md`.
//!
//! Ignored by default; one phase per invocation (`SR_PHASE`), rooted at `SR_ROOT`:
//!
//! * `source`  ingest `SR_N` rows of the Float16 corpus `SR_CORPUS` into a Plain Float16 cosine
//!   segment (ids 1..=N), exactly as the Tier 3 harness (`ann_matched_v120.rs`).
//! * `build`   native IVF through `SegmentBuilder` with `SR_INDEX` (`{"lmi": LmiConfig}`) and the
//!   CPU permit `SR_THREADS`; the router kind comes from `LMI_RESEARCH_ROUTER`.
//! * `export`  the built router (arm A0, artifact format below), `bucket_of_id.u32`,
//!   `occupancy.u32` and `postings_canonical.u64` (buckets in order, each bucket's external ids
//!   sorted; its sha256 is the partition fingerprint, also written to `partition.json`).
//! * `gold`    native exact top-`SR_K` (Plain exact path) for the unit-normalised Float32 query
//!   file `SR_QUERIES`; writes `<SR_OUT>.ids.u64` and `<SR_OUT>.scores.f32`. Multi-threaded.
//! * `consistency` native top-64 buckets for the first `SR_CONSISTENCY_N` queries per arm.
//! * `evaluate` arms (`SR_ARMS`, one router.json path per line) x nprobe grid (`SR_SETTINGS`) x
//!   query sets (`SR_SETS` = `name:file:reps,...`): one untimed warm pass per (arm, nprobe), then
//!   `reps` timed repetitions whose arm order is a seeded permutation per repetition. The timed
//!   path is the Tier 3 decomposed LMI path: `router.top_buckets`, sorted de-duplicated posting
//!   union, native `score_candidates`. Every result must hold min(`SR_TOP`, candidates) hits;
//!   shorter results (probes holding fewer than `SR_TOP` points) are recorded, not widened.
//!
//! Router artifact (`router.json`): `{"arm": .., "layers": [{"kind": "linear", "in": d, "out": m,
//! "weights": "<file>", "bias": "<file>", "weights_sha256": .., "bias_sha256": ..},
//! {"kind": "relu"}, ...]}` with raw little-endian f32 files relative to the JSON. Hashes,
//! `MlpRouter::validate`, input 768 and output = posting count are checked before use.
//! Output files are created with `create_new`; nothing existing is overwritten.

use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
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
use segment::index::lmi_index::{LinearLayer, LmiConfig, MlpRouter, RouterLayer};
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::segment_constructor::segment_builder::SegmentBuilder;
use segment::segment_constructor::{build_segment, load_segment};
use segment::types::{
    Distance, ExtendedPointId, HnswGlobalConfig, Indexes, SearchParams, SegmentConfig,
    VectorDataConfig, VectorStorageDatatype, VectorStorageType,
};
use sha2::{Digest, Sha256};

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

fn write_new(path: &Path, value: &serde_json::Value) {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    writeln!(f, "{value}").unwrap();
}

fn create_new(path: &Path) -> BufWriter<fs::File> {
    BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap_or_else(|e| panic!("create {}: {e}", path.display())),
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn only_segment_dir(root: &Path) -> PathBuf {
    let dirs: Vec<PathBuf> = fs::read_dir(root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("segment.json").exists())
        .collect();
    assert_eq!(dirs.len(), 1, "expected one segment under {}", root.display());
    dirs.into_iter().next().unwrap()
}

fn load(root: &Path) -> Segment {
    let dir = only_segment_dir(root);
    let uuid = dir.file_name().unwrap().to_str().unwrap().parse().unwrap();
    load_segment(&dir, uuid, None, &AtomicBool::new(false), true).unwrap()
}

fn read_f32(path: &Path) -> Vec<f32> {
    fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn write_f32(path: &Path, values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut f = create_new(path);
    f.write_all(&bytes).unwrap();
    sha256_hex(&bytes)
}

fn read_queries(path: &Path) -> Vec<Vec<f32>> {
    let q = read_f32(path);
    assert_eq!(q.len() % DIM, 0, "query file {} is not n x {DIM}", path.display());
    q.chunks_exact(DIM).map(<[f32]>::to_vec).collect()
}

/// Load and fully check a router artifact. Panics (aborting the phase) on any mismatch.
fn load_router(path: &Path, buckets: usize) -> (String, MlpRouter, serde_json::Value) {
    let spec: serde_json::Value =
        serde_json::from_slice(&fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
            .unwrap();
    let dir = path.parent().unwrap();
    let mut layers = Vec::new();
    for layer in spec["layers"].as_array().expect("layers") {
        match layer["kind"].as_str().unwrap() {
            "relu" => layers.push(RouterLayer::ReLU),
            "linear" => {
                let (inf, outf) = (
                    layer["in"].as_u64().unwrap() as usize,
                    layer["out"].as_u64().unwrap() as usize,
                );
                let wbytes = fs::read(dir.join(layer["weights"].as_str().unwrap())).unwrap();
                let bbytes = fs::read(dir.join(layer["bias"].as_str().unwrap())).unwrap();
                assert_eq!(sha256_hex(&wbytes), layer["weights_sha256"].as_str().unwrap(), "weights hash");
                assert_eq!(sha256_hex(&bbytes), layer["bias_sha256"].as_str().unwrap(), "bias hash");
                let to_f32 = |b: &[u8]| -> Vec<f32> {
                    b.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                };
                let (weights, bias) = (to_f32(&wbytes), to_f32(&bbytes));
                assert_eq!(weights.len(), inf * outf, "weights shape");
                assert_eq!(bias.len(), outf, "bias shape");
                layers.push(RouterLayer::Linear(LinearLayer {
                    in_features: inf,
                    out_features: outf,
                    weights,
                    bias,
                }));
            }
            other => panic!("unknown layer kind {other}"),
        }
    }
    let router = MlpRouter { layers };
    let (input, output) = router.validate().expect("router validates");
    assert_eq!(input, DIM, "router input dimension");
    assert_eq!(output, buckets, "router output must equal the posting count");
    let name = spec["arm"].as_str().unwrap().to_string();
    (name, router, spec)
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
    let n: usize = env("SR_N").parse().unwrap();
    let corpus = PathBuf::from(env("SR_CORPUS"));
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
    let out = root.join("build");
    fs::create_dir(&out).expect("build directory must not exist");
    let spec: serde_json::Value = serde_json::from_str(&env("SR_INDEX")).unwrap();
    let cfg: LmiConfig = serde_json::from_value(spec["lmi"].clone()).unwrap();
    let threads: usize = env("SR_THREADS").parse().unwrap();
    let source = load(&root.join("source"));
    let mut config = source.config().clone();
    let vector = config.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap();
    vector.storage_type = VectorStorageType::Mmap;
    vector.index = Indexes::LmiTrained(cfg);
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
    drop(source);
    let segment_root = out.join("segment");
    fs::create_dir(&segment_root).unwrap();
    let built = builder
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
        .unwrap();
    write_new(
        &out.join("build.json"),
        &serde_json::json!({"spec": spec, "threads": threads,
            "seconds": started.elapsed().as_secs_f64(),
            "router_env": std::env::var("LMI_RESEARCH_ROUTER").ok(),
            "available": built.available_point_count(), "peak_rss_kib": peak_rss_kib()}),
    );
}

fn built(root: &Path) -> Segment {
    load(&root.join("build").join("segment"))
}

/// Router layers as an artifact directory (used for A0 export).
fn write_router_artifact(dir: &Path, arm: &str, router: &MlpRouter) {
    fs::create_dir_all(dir).unwrap();
    let mut layers = Vec::new();
    for (i, layer) in router.layers.iter().enumerate() {
        match layer {
            RouterLayer::ReLU => layers.push(serde_json::json!({"kind": "relu"})),
            RouterLayer::Linear(l) => {
                let w = format!("layer{i}.weights.f32");
                let b = format!("layer{i}.bias.f32");
                let ws = write_f32(&dir.join(&w), &l.weights);
                let bs = write_f32(&dir.join(&b), &l.bias);
                layers.push(serde_json::json!({"kind": "linear", "in": l.in_features,
                    "out": l.out_features, "weights": w, "bias": b,
                    "weights_sha256": ws, "bias_sha256": bs}));
            }
        }
    }
    write_new(
        &dir.join("router.json"),
        &serde_json::json!({"arm": arm, "layers": layers, "origin": "native build export"}),
    );
}

/// Bucket of every external id, occupancy, and the canonical postings fingerprint.
fn partition(segment: &Segment) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
    let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected a trained LMI segment")
    };
    let state = lmi.routing_state().expect("trained routing state");
    let tracker = segment.id_tracker.borrow();
    let available = segment.available_point_count();
    let mut bucket_of = vec![u32::MAX; available];
    let mut occupancy = Vec::with_capacity(state.postings().len());
    let mut canonical = Vec::new();
    for (b, posting) in state.postings().iter().enumerate() {
        let mut ids: Vec<u64> = posting
            .iter()
            .map(|&o| match tracker.external_id(o).expect("posting offset is live") {
                ExtendedPointId::NumId(id) => id,
                other => panic!("unexpected id {other:?}"),
            })
            .collect();
        ids.sort_unstable();
        for &id in &ids {
            let slot = &mut bucket_of[(id - 1) as usize];
            assert_eq!(*slot, u32::MAX, "point {id} appears in two postings");
            *slot = b as u32;
            canonical.extend_from_slice(&id.to_le_bytes());
        }
        canonical.extend_from_slice(&u64::MAX.to_le_bytes()); // bucket separator
        occupancy.push(ids.len() as u32);
    }
    assert!(bucket_of.iter().all(|&b| b != u32::MAX), "every live point has a bucket");
    assert_eq!(occupancy.iter().map(|&n| n as usize).sum::<usize>(), available);
    (bucket_of, occupancy, canonical)
}

fn phase_export(root: &Path) {
    let out = root.join("export");
    fs::create_dir(&out).expect("export must not exist");
    let segment = built(root);
    let (bucket_of, occupancy, canonical) = partition(&segment);
    let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected a trained LMI segment")
    };
    write_router_artifact(&out.join("A0"), "A0", lmi.routing_state().unwrap().router());
    let to_bytes = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    create_new(&out.join("bucket_of_id.u32")).write_all(&to_bytes(&bucket_of)).unwrap();
    create_new(&out.join("occupancy.u32")).write_all(&to_bytes(&occupancy)).unwrap();
    create_new(&out.join("postings_canonical.u64")).write_all(&canonical).unwrap();
    write_new(
        &out.join("partition.json"),
        &serde_json::json!({"points": bucket_of.len(), "buckets": occupancy.len(),
            "fingerprint_sha256": sha256_hex(&canonical),
            "occupancy_min": occupancy.iter().min(), "occupancy_max": occupancy.iter().max(),
            "empty_buckets": occupancy.iter().filter(|&&n| n == 0).count()}),
    );
}

fn phase_gold(root: &Path) {
    let segment = built(root);
    let queries = read_queries(Path::new(&env("SR_QUERIES")));
    let k: usize = env("SR_K").parse().unwrap();
    let out = PathBuf::from(env("SR_OUT"));
    let threads: usize = env("SR_THREADS").parse().unwrap();
    let exact = SearchParams {
        exact: true,
        ..Default::default()
    };
    let started = Instant::now();
    let results: Vec<(usize, Vec<u64>, Vec<f32>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|w| {
                let (segment, queries, exact) = (&segment, &queries, &exact);
                scope.spawn(move || {
                    let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
                    let tracker = segment.id_tracker.borrow();
                    let ctx = VectorQueryContext::default();
                    let mut rows = Vec::new();
                    for (qi, q) in queries.iter().enumerate().skip(w).step_by(threads) {
                        let query: QueryVector = q.clone().into();
                        let hits = index
                            .search(&[&query], None, k, Some(exact), &ctx)
                            .unwrap()
                            .remove(0);
                        assert_eq!(hits.len(), k, "exact search must fill k");
                        let ids = hits
                            .iter()
                            .map(|h| match tracker.external_id(h.idx).unwrap() {
                                ExtendedPointId::NumId(id) => id,
                                other => panic!("unexpected id {other:?}"),
                            })
                            .collect();
                        rows.push((qi, ids, hits.iter().map(|h| h.score).collect()));
                    }
                    rows
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut ordered = results;
    ordered.sort_by_key(|r| r.0);
    assert_eq!(ordered.len(), queries.len());
    let ids: Vec<u8> = ordered.iter().flat_map(|r| r.1.iter().flat_map(|v| v.to_le_bytes())).collect();
    let scores: Vec<u8> = ordered.iter().flat_map(|r| r.2.iter().flat_map(|v| v.to_le_bytes())).collect();
    let path = |ext: &str| PathBuf::from(format!("{}.{ext}", out.display()));
    create_new(&path("ids.u64")).write_all(&ids).unwrap();
    create_new(&path("scores.f32")).write_all(&scores).unwrap();
    write_new(
        &path("json"),
        &serde_json::json!({"queries": queries.len(), "k": k, "threads": threads,
            "seconds": started.elapsed().as_secs_f64(), "ids_sha256": sha256_hex(&ids),
            "source": "native exact (Plain exact path) over the built Float16 segment"}),
    );
}

fn arms(buckets: usize) -> Vec<(String, MlpRouter, serde_json::Value)> {
    let list = fs::read_to_string(env("SR_ARMS")).unwrap();
    let arms: Vec<_> = list
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| load_router(Path::new(l.trim()), buckets))
        .collect();
    let mut names: Vec<&str> = arms.iter().map(|a| a.0.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), arms.len(), "arm names must be unique");
    arms
}

fn phase_consistency(root: &Path) {
    let segment = built(root);
    let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected a trained LMI segment")
    };
    let buckets = lmi.routing_state().unwrap().postings().len();
    let queries = read_queries(Path::new(&env("SR_QUERIES")));
    let n: usize = env("SR_CONSISTENCY_N").parse().unwrap();
    let mut out = create_new(Path::new(&env("SR_OUT")));
    for (name, router, _) in arms(buckets) {
        for (qi, q) in queries.iter().take(n).enumerate() {
            let normalized = Distance::Cosine.preprocess_vector::<f32>(q.clone());
            let top = router.top_buckets(&normalized, 64.min(buckets)).unwrap();
            writeln!(out, "{}", serde_json::json!({"arm": name, "q": qi, "buckets": top})).unwrap();
        }
    }
}

struct Timed {
    ids: Vec<u64>,
    candidates: usize,
    route_ns: u64,
    union_ns: u64,
    score_ns: u64,
}

/// The Tier 3 decomposed native LMI path with an explicit router.
fn search_routed(
    segment: &Segment,
    router: &MlpRouter,
    q: &[f32],
    nprobe: usize,
    top: usize,
) -> Timed {
    let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
    let index = data.vector_index.borrow();
    let VectorIndexEnum::Lmi(lmi) = &*index else {
        panic!("expected a trained LMI segment")
    };
    let state = lmi.routing_state().expect("LMI routing state");
    let storage = data.vector_storage.borrow();
    let tracker = segment.id_tracker.borrow();
    let query: QueryVector = q.to_vec().into();
    let ctx = VectorQueryContext::default();
    let t0 = Instant::now();
    let normalized = Distance::Cosine.preprocess_vector::<f32>(q.to_vec());
    let buckets = router.top_buckets(&normalized, nprobe).unwrap();
    let t1 = Instant::now();
    let mut offsets = Vec::new();
    for b in buckets {
        offsets.extend_from_slice(state.postings().get(b).unwrap());
    }
    offsets.sort_unstable();
    offsets.dedup();
    let t2 = Instant::now();
    let hits = score_candidates(&query, &offsets, top, &tracker, &storage, &ctx).unwrap();
    let t3 = Instant::now();
    let ids = hits
        .iter()
        .map(|h| match tracker.external_id(h.idx).unwrap() {
            ExtendedPointId::NumId(id) => id,
            other => panic!("unexpected id {other:?}"),
        })
        .collect();
    Timed {
        ids,
        candidates: offsets.len(),
        route_ns: (t1 - t0).as_nanos() as u64,
        union_ns: (t2 - t1).as_nanos() as u64,
        score_ns: (t3 - t2).as_nanos() as u64,
    }
}

fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, (next() % (i as u64 + 1)) as usize);
    }
    order
}

fn phase_evaluate(root: &Path) {
    let out = PathBuf::from(env("SR_OUT"));
    fs::create_dir(&out).expect("evaluation directory must not exist");
    let load_started = Instant::now();
    let segment = built(root);
    let load_seconds = load_started.elapsed().as_secs_f64();
    let buckets = {
        let index = segment.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
        let VectorIndexEnum::Lmi(lmi) = &*index else {
            panic!("expected a trained LMI segment")
        };
        lmi.routing_state().unwrap().postings().len()
    };
    let fingerprint_before = sha256_hex(&partition(&segment).2);
    let arms = arms(buckets);
    let settings: Vec<usize> = env("SR_SETTINGS").split(',').map(|s| s.parse().unwrap()).collect();
    let top: usize = env("SR_TOP").parse().unwrap();
    let seed: u64 = env("SR_SEED").parse().unwrap();
    // Short results (< top hits) arise only when the probed buckets hold fewer than `top`
    // candidates: the decomposed path has no underfill widening (as in Tier 3). They are counted
    // and scored as misses. A hit count other than min(top, candidates) is a correctness failure.
    let mut short_results = 0usize;
    let mut unexplained = 0usize;
    let mut summary_sets = Vec::new();
    for set in env("SR_SETS").split(',') {
        let parts: Vec<&str> = set.split(':').collect();
        let (name, file, reps) = (parts[0], parts[1], parts[2].parse::<usize>().unwrap());
        let queries = read_queries(Path::new(file));
        let mut rows = create_new(&out.join(format!("{name}.jsonl")));
        let warm = Instant::now();
        for (_, router, _) in &arms {
            for &p in &settings {
                for q in &queries {
                    search_routed(&segment, router, q, p, top);
                }
            }
        }
        let warm_seconds = warm.elapsed().as_secs_f64();
        let timed = Instant::now();
        for rep in 1..=reps {
            let order = permutation(arms.len(), seed ^ (rep as u64) << 32 ^ name.len() as u64);
            for &a in &order {
                let (arm, router, _) = &arms[a];
                for &p in &settings {
                    for (qi, q) in queries.iter().enumerate() {
                        let started = Instant::now();
                        let r = search_routed(&segment, router, q, p, top);
                        let ns = started.elapsed().as_nanos() as u64;
                        if r.ids.len() != top {
                            short_results += 1;
                        }
                        if r.ids.len() != top.min(r.candidates) {
                            unexplained += 1;
                        }
                        let mut row = serde_json::json!({"arm": arm, "nprobe": p, "q": qi,
                            "rep": rep, "ns": ns, "route_ns": r.route_ns, "union_ns": r.union_ns,
                            "score_ns": r.score_ns, "candidates": r.candidates,
                            "hits": r.ids.len()});
                        if rep == 1 {
                            row["ids"] = serde_json::json!(r.ids);
                        }
                        writeln!(rows, "{row}").unwrap();
                    }
                }
            }
        }
        rows.flush().unwrap();
        summary_sets.push(serde_json::json!({"set": name, "queries": queries.len(), "reps": reps,
            "warm_seconds": warm_seconds, "timed_seconds": timed.elapsed().as_secs_f64()}));
    }
    let fingerprint_after = sha256_hex(&partition(&segment).2);
    assert_eq!(fingerprint_before, fingerprint_after, "evaluation must not alter postings");
    write_new(
        &out.join("summary.json"),
        &serde_json::json!({"arms": arms.iter().map(|a| a.0.clone()).collect::<Vec<_>>(),
            "arm_specs": arms.iter().map(|a| a.2.clone()).collect::<Vec<_>>(),
            "settings": settings, "top": top, "seed": seed, "sets": summary_sets,
            "load_seconds": load_seconds, "fingerprint_sha256": fingerprint_before,
            "short_results": short_results, "unexplained_short_results": unexplained,
            "peak_rss_kib": peak_rss_kib()}),
    );
    assert_eq!(unexplained, 0, "every query must return min(top, candidates) hits");
}

#[test]
#[ignore = "supervised-routing harness; requires SR_PHASE and SR_ROOT"]
fn supervised_routing_v120() {
    let root = PathBuf::from(env("SR_ROOT"));
    match env("SR_PHASE").as_str() {
        "source" => phase_source(&root),
        "build" => phase_build(&root),
        "export" => phase_export(&root),
        "gold" => phase_gold(&root),
        "consistency" => phase_consistency(&root),
        "evaluate" => phase_evaluate(&root),
        p => panic!("unknown phase {p}"),
    }
}
