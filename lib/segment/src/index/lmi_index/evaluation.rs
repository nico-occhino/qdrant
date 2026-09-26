//! Opt-in scientific harness. Compiled only into the segment unit-test binary.
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::counter::hardware_counter::HardwareCounterCell;
use common::generic_consts::Random;
use common::types::{PointOffsetType, ScoredPointOffset, TelemetryDetail};
use serde_json::{Value, json};

use super::{LmiConfig, LmiRoutingState, MlpRouter};
use crate::data_types::query_context::QueryContext;
use crate::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector, only_default_vector};
use crate::entry::entry_point::{ReadSegmentEntry, SegmentEntry};
use crate::id_tracker::IdTrackerRead;
use crate::index::hnsw_index::point_scorer::BatchFilteredSearcher;
use crate::index::{VectorIndex, VectorIndexEnum, VectorIndexRead};
use crate::segment::Segment;
use crate::segment_constructor::{
    load_segment, segment_builder::SegmentBuilder, simple_segment_constructor::build_simple_segment,
};
use crate::types::{Distance, HnswConfig, HnswGlobalConfig, Indexes, PointIdType, SearchParams};
use crate::vector_storage::VectorStorageRead;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;

fn order(q: &[f32], centers: &[Vec<f64>], affine: bool) -> Vec<usize> {
    let mut scores: Vec<_> = centers
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let score: f64 = if affine {
                q.iter()
                    .zip(c)
                    .map(|(&x, &y)| y * f64::from(x))
                    .sum::<f64>()
                    + c[q.len()]
            } else {
                -q.iter()
                    .zip(c)
                    .map(|(&x, &y)| (f64::from(x) - y).powi(2))
                    .sum::<f64>()
            };
            (i, score)
        })
        .collect();
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scores.into_iter().map(|(i, _)| i).collect()
}

fn affine_parameters(centers: &[Vec<f64>]) -> Vec<Vec<f64>> {
    centers
        .iter()
        .map(|c| {
            let mut weights: Vec<_> = c.iter().map(|x| 2.0 * x).collect();
            weights.push(-c.iter().map(|x| x * x).sum::<f64>());
            weights
        })
        .collect()
}

#[test]
fn centroid_affine_equivalence() {
    let centers = vec![
        vec![1.0, 2.0],
        vec![-3.0, 0.5],
        vec![1.0, 2.0],
        vec![0.0, 0.0],
    ];
    for i in -100..100 {
        let q = [i as f32 / 17.0, (i * i % 31) as f32 / 11.0];
        assert_eq!(
            order(&q, &centers, false),
            order(&q, &affine_parameters(&centers), true)
        );
    }
}

fn floats(path: PathBuf) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn ids(s: &Segment, rows: &[ScoredPointOffset]) -> HashSet<PointIdType> {
    let tracker = s.id_tracker.borrow();
    rows.iter()
        .map(|p| tracker.external_id(p.idx).unwrap())
        .collect()
}
fn search(
    s: &Segment,
    q: &QueryVector,
    k: usize,
    params: Option<&SearchParams>,
) -> Vec<ScoredPointOffset> {
    let root = QueryContext::default();
    let context = root.get_segment_query_context();
    let context = context.get_vector_context(DEFAULT_VECTOR_NAME, None);
    s.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .search(&[q], None, k, params, &context)
        .unwrap()
        .remove(0)
}
fn build(source: &Segment, path: &Path, index: Indexes) -> (Segment, f64) {
    let staging = tempfile::tempdir().unwrap();
    let mut cfg = source.config().clone();
    cfg.vector_data.get_mut(DEFAULT_VECTOR_NAME).unwrap().index = index;
    let start = Instant::now();
    let mut builder =
        SegmentBuilder::new(staging.path(), &cfg, &HnswGlobalConfig::default()).unwrap();
    builder
        .update(
            &[source],
            &AtomicBool::new(false),
            &HardwareCounterCell::new(),
        )
        .unwrap();
    (
        builder
            .build(
                path,
                uuid::Uuid::new_v4(),
                None,
                common::budget::ResourcePermit::dummy(1),
                &AtomicBool::new(false),
                &mut <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42),
                &HardwareCounterCell::new(),
                common::progress_tracker::ProgressTracker::new_for_test(),
            )
            .unwrap(),
        start.elapsed().as_secs_f64(),
    )
}
fn size(s: &Segment) -> u64 {
    s.vector_data[DEFAULT_VECTOR_NAME]
        .vector_index
        .borrow()
        .files()
        .iter()
        .map(|p| p.metadata().unwrap().len())
        .sum()
}
fn reopened(s: Segment) -> (Segment, f64) {
    let path = s.segment_path.clone();
    drop(s);
    let start = Instant::now();
    let s = load_segment(&path, uuid::Uuid::nil(), None, &AtomicBool::new(false)).unwrap();
    (s, start.elapsed().as_secs_f64())
}

#[test]
#[ignore = "Explicit Phase F release benchmark; requires LMI_PHASE_F_DIR"]
fn phase_f_benchmark() {
    assert!(
        std::process::Command::new("taskset")
            .args(["-apc", "0", &std::process::id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let dir = PathBuf::from(std::env::var("LMI_PHASE_F_DIR").expect("dataset directory"));
    let meta: Value =
        serde_json::from_slice(&std::fs::read(dir.join("dataset.json")).unwrap()).unwrap();
    let dim = meta["dimension"].as_u64().unwrap() as usize;
    let count = meta["corpus_count"].as_u64().unwrap() as usize;
    let warmup = meta["warmup"].as_u64().unwrap() as usize;
    let trials = meta["trials"].as_u64().unwrap() as usize;
    let k = 10;
    let config = LmiConfig {
        n_buckets: 64,
        sample_size: 2048.min(count),
        hidden_dim: 64,
        epochs: 30,
        batch_size: 256,
        kmeans_iterations: 20,
        nprobe: 1,
        seed: 42,
    };
    let root = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(root.path(), dim, Distance::Cosine).unwrap();
    let corpus = floats(dir.join("corpus.f32"));
    assert_eq!(corpus.len(), count * dim);
    let start = Instant::now();
    for (i, row) in corpus.chunks_exact(dim).enumerate() {
        plain
            .upsert_point(
                i as u64 + 1,
                (i as u64).into(),
                only_default_vector(row),
                &HardwareCounterCell::new(),
            )
            .unwrap();
    }
    let ingest_seconds = start.elapsed().as_secs_f64();
    drop(corpus);
    eprintln!("Phase F: corpus loaded: {count} x {dim}");
    let (lmi, lmi_build) = build(&plain, root.path(), Indexes::LmiTrained(config));
    let stage = *super::training::STAGE_SECONDS.lock().unwrap();
    let classify = *super::build::POSTING_SECONDS.lock().unwrap();
    let lmi_size = size(&lmi);
    let (lmi, lmi_reopen) = reopened(lmi);
    let state_path = {
        let index = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow();
        let VectorIndexEnum::Lmi(index) = &*index else {
            panic!("not LMI")
        };
        assert!(index.routing_state.is_some());
        index.state_path().unwrap().to_owned()
    };
    let disk: Value = serde_json::from_slice(&std::fs::read(&state_path).unwrap()).unwrap();
    let router: MlpRouter = serde_json::from_value(disk["router"].clone()).unwrap();
    let mlp_postings: Vec<Vec<PointOffsetType>> =
        serde_json::from_value(disk["postings"].clone()).unwrap();
    let mut sample = Vec::new();
    {
        let storage = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
        for id in disk["sample_offsets"].as_array().unwrap() {
            let row = storage
                .get_vector_opt::<Random>(id.as_u64().unwrap() as u32)
                .unwrap();
            let crate::data_types::named_vectors::CowVector::Dense(row) = row else {
                panic!()
            };
            sample.extend_from_slice(&row);
        }
    }
    let start = Instant::now();
    let (_, centers) =
        super::training::cluster_with_centers(&sample, dim, &config, &AtomicBool::new(false))
            .unwrap();
    let control_cluster = start.elapsed().as_secs_f64();
    let affine = affine_parameters(&centers);
    let start = Instant::now();
    let mut centroid_postings = vec![Vec::new(); config.n_buckets];
    {
        let storage = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
        for id in 0..count as u32 {
            let row = storage.get_vector_opt::<Random>(id).unwrap();
            let crate::data_types::named_vectors::CowVector::Dense(row) = row else {
                panic!()
            };
            centroid_postings[order(&row, &centers, false)[0]].push(id);
        }
    }
    let control_posting = start.elapsed().as_secs_f64();
    let controls = json!({"centers":centers,"postings":centroid_postings});
    std::fs::write(
        dir.join("centroid_state.json"),
        serde_json::to_vec(&controls).unwrap(),
    )
    .unwrap();
    eprintln!("Phase F: LMI and controls built; building HNSW");
    let hc = HnswConfig {
        m: 16,
        ef_construct: 100,
        full_scan_threshold: 0,
        max_indexing_threads: 1,
        ..Default::default()
    };
    let (hnsw, hnsw_build) = build(&plain, root.path(), Indexes::Hnsw(hc));
    let hnsw_size = size(&hnsw);
    let (hnsw, hnsw_reopen) = reopened(hnsw);
    assert!(matches!(
        &*hnsw.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow(),
        VectorIndexEnum::Hnsw(_)
    ));
    let query_data = floats(dir.join("queries.f32"));
    let queries: Vec<_> = query_data
        .chunks_exact(dim)
        .map(|r| QueryVector::from(r.to_vec()))
        .collect();
    let normalized: Vec<_> = query_data
        .chunks_exact(dim)
        .map(|r| Distance::Cosine.preprocess_vector::<f32>(r.to_vec()))
        .collect();
    let exact = SearchParams {
        exact: true,
        ..Default::default()
    };
    let ground: Vec<_> = queries
        .iter()
        .map(|q| ids(&plain, &search(&plain, q, k, Some(&exact))))
        .collect();
    assert!(matches!(
        &*plain.vector_data[DEFAULT_VECTOR_NAME].vector_index.borrow(),
        VectorIndexEnum::Plain(_)
    ));
    let mut disagreement = 0;
    let mut teacher_hits = vec![0; config.n_buckets];
    for q in &normalized {
        let direct = order(q, &centers, false);
        assert_eq!(direct, order(q, &affine, true));
        let mlp = router.top_buckets(q, config.n_buckets).unwrap();
        disagreement += usize::from(mlp[0] != direct[0]);
        for p in 1..=config.n_buckets {
            teacher_hits[p - 1] += usize::from(mlp[..p].contains(&direct[0]));
        }
    }
    let mut output =
        std::io::BufWriter::new(std::fs::File::create(dir.join("queries.jsonl")).unwrap());
    let mut operating: Vec<(&str, usize)> = vec![("plain", 0)];
    for p in [1, 2, 4, 8, 16, 32, 64] {
        for method in ["centroid", "affine", "mlp"] {
            operating.push((method, p));
        }
    }
    for ef in [16, 32, 64, 128, 256, 512] {
        operating.push(("hnsw", ef));
    }
    for trial in 0..trials {
        // Rotate deterministic configuration order to reduce fixed warm-cache/order bias.
        let mut schedule = operating.clone();
        let len = schedule.len();
        schedule.rotate_left((trial * 9) % len);
        for (method, effort) in schedule {
            if method == "mlp" {
                let mut index = lmi.vector_data[DEFAULT_VECTOR_NAME]
                    .vector_index
                    .borrow_mut();
                let VectorIndexEnum::Lmi(index) = &mut *index else {
                    panic!()
                };
                index.routing_state = Some(
                    LmiRoutingState::new(router.clone(), mlp_postings.clone(), effort).unwrap(),
                );
            }
            let telemetry_before = hnsw.vector_data[DEFAULT_VECTOR_NAME]
                .vector_index
                .borrow()
                .get_telemetry_data(TelemetryDetail::default())
                .unfiltered_hnsw
                .count;
            for (qi, q) in queries.iter().enumerate() {
                let mut router_ns = 0u128;
                let mut preparation_ns = 0u128;
                let mut scoring_ns = 0u128;
                let candidates_count: Option<usize>;
                let result;
                let total_ns;
                if method == "plain" || method == "hnsw" {
                    let s = if method == "plain" { &plain } else { &hnsw };
                    let params = if method == "plain" {
                        exact.clone()
                    } else {
                        SearchParams {
                            hnsw_ef: Some(effort),
                            ..Default::default()
                        }
                    };
                    let start = Instant::now();
                    result = search(s, q, k, Some(&params));
                    total_ns = start.elapsed().as_nanos();
                    candidates_count = if method == "plain" { Some(count) } else { None };
                } else {
                    let full_start = Instant::now();
                    let route_input = Distance::Cosine
                        .preprocess_vector::<f32>(query_data[qi * dim..(qi + 1) * dim].to_vec());
                    let start = Instant::now();
                    let buckets = if method == "mlp" {
                        router.top_buckets(&route_input, effort).unwrap()
                    } else {
                        order(
                            &route_input,
                            if method == "affine" {
                                &affine
                            } else {
                                &centers
                            },
                            method == "affine",
                        )[..effort]
                            .to_vec()
                    };
                    router_ns = start.elapsed().as_nanos();
                    let start = Instant::now();
                    let postings = if method == "mlp" {
                        &mlp_postings
                    } else {
                        &centroid_postings
                    };
                    let mut candidates: Vec<u32> = buckets
                        .iter()
                        .flat_map(|&b| postings[b].iter().copied())
                        .collect();
                    candidates.sort_unstable();
                    candidates.dedup();
                    let tracker = lmi.id_tracker.borrow();
                    let storage = lmi.vector_data[DEFAULT_VECTOR_NAME].vector_storage.borrow();
                    candidates.retain(|&id| {
                        (id as usize) < count
                            && !tracker.deleted_point_bitslice()[id as usize]
                            && !storage.is_deleted_vector(id)
                    });
                    candidates_count = Some(candidates.len());
                    preparation_ns = start.elapsed().as_nanos();
                    let start = Instant::now();
                    let hw = HardwareCounterCell::new();
                    let scorer = BatchFilteredSearcher::new(
                        &[q],
                        &*storage,
                        None::<&QuantizedVectors>,
                        None,
                        k,
                        tracker.deleted_point_bitslice(),
                        hw,
                    )
                    .unwrap();
                    result = scorer
                        .peek_top_iter(candidates.into_iter(), &AtomicBool::new(false))
                        .unwrap()
                        .remove(0);
                    scoring_ns = start.elapsed().as_nanos();
                    total_ns = full_start.elapsed().as_nanos();
                }
                let result_segment = if method == "plain" {
                    &plain
                } else if method == "hnsw" {
                    &hnsw
                } else {
                    &lmi
                };
                let found = ids(result_segment, &result);
                let recall = found.intersection(&ground[qi]).count() as f64 / k as f64;
                let mut native_ns = None;
                if method == "mlp" {
                    let start = Instant::now();
                    let native = search(&lmi, q, k, None);
                    native_ns = Some(start.elapsed().as_nanos());
                    assert_eq!(
                        result, native,
                        "component harness must match actual learned path"
                    );
                }
                if qi >= warmup {
                    writeln!(output,"{}",json!({"method":method,"effort":effort,"trial":trial,"query":qi-warmup,"k":k,
                    "recall":recall,"candidate_count":candidates_count,"candidate_fraction":candidates_count.map(|n|n as f64/count as f64),
                    "router_ns":(method!="plain" && method!="hnsw").then_some(router_ns),"preparation_ns":(method!="plain" && method!="hnsw").then_some(preparation_ns),"scoring_ns":(method!="plain" && method!="hnsw").then_some(scoring_ns),"total_ns":total_ns,
                    "native_mlp_index_ns":native_ns})).unwrap();
                }
            }
            if method == "hnsw" {
                let after = hnsw.vector_data[DEFAULT_VECTOR_NAME]
                    .vector_index
                    .borrow()
                    .get_telemetry_data(TelemetryDetail::default())
                    .unfiltered_hnsw
                    .count;
                assert_eq!(
                    after - telemetry_before,
                    queries.len(),
                    "HNSW must execute, not Plain"
                );
            }
            output.flush().unwrap();
            eprintln!("Phase F: trial={trial} {method} effort={effort} done");
        }
    }
    let summary = json!({"dataset":meta,"lmi_config":config,"hnsw_config":hc,"ingest_seconds":ingest_seconds,
        "lmi_build_seconds":lmi_build,"clustering_seconds":stage[0],"mlp_training_export_seconds":stage[1],"mlp_posting_seconds":classify,
        "control_clustering_seconds":control_cluster,"centroid_posting_seconds":control_posting,"hnsw_build_seconds":hnsw_build,
        "lmi_index_bytes":lmi_size,"hnsw_index_bytes":hnsw_size,"centroid_index_bytes":dir.join("centroid_state.json").metadata().unwrap().len(),
        "lmi_reopen_seconds":lmi_reopen,"hnsw_reopen_seconds":hnsw_reopen,"centroid_affine_order_mismatches":0,
        "mlp_teacher_agreement":1.0-disagreement as f64/queries.len() as f64,
        "teacher_top_p_coverage":teacher_hits.iter().map(|&n|n as f64/queries.len() as f64).collect::<Vec<_>>(),
        "mlp_bucket_sizes":mlp_postings.iter().map(Vec::len).collect::<Vec<_>>(),"centroid_bucket_sizes":centroid_postings.iter().map(Vec::len).collect::<Vec<_>>(),
        "process_status":std::fs::read_to_string("/proc/self/status").unwrap()});
    std::fs::write(
        dir.join("build.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .unwrap();
    std::fs::copy(&state_path, dir.join("mlp_state.json")).unwrap();
}
