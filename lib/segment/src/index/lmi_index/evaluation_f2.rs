//! F.2 only: fixed teacher, frozen split, native router and common Qdrant scorer.
use super::*;
use std::fs::{File, OpenOptions};

fn save(path: &Path, value: &Value) {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(&mut f, value).unwrap();
}
fn write_new(path: &Path, bytes: &[u8]) {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    f.write_all(bytes).unwrap();
}
fn offsets(postings: &[Vec<u32>], n: usize) -> Vec<usize> {
    let mut labels = vec![usize::MAX; n];
    for (b, p) in postings.iter().enumerate() {
        for &id in p {
            assert_eq!(labels[id as usize], usize::MAX);
            labels[id as usize] = b;
        }
    }
    assert!(labels.iter().all(|&b| b < 64));
    labels
}

#[test]
#[ignore = "Phase F.2 fixed-teacher diagnosis; explicit separate output directory"]
fn phase_f2_study() {
    assert!(
        std::process::Command::new("taskset")
            .args(["-apc", "0", &std::process::id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let base = PathBuf::from(std::env::var("LMI_PHASE_F_DIR").unwrap());
    let out = PathBuf::from(std::env::var("LMI_PHASE_F2_DIR").unwrap());
    assert_ne!(base.canonicalize().unwrap(), out.canonicalize().unwrap());
    let mode = std::env::var("LMI_PHASE_F2_MODE").unwrap_or("baseline".into());
    let protocol: Value =
        serde_json::from_slice(&std::fs::read(out.join("protocol.json")).unwrap()).unwrap();
    let disk: Value =
        serde_json::from_slice(&std::fs::read(base.join("mlp_state.json")).unwrap()).unwrap();
    let controls: Value =
        serde_json::from_slice(&std::fs::read(base.join("centroid_state.json")).unwrap()).unwrap();
    let centers: Vec<Vec<f64>> = serde_json::from_value(controls["centers"].clone()).unwrap();
    let centroid_postings: Vec<Vec<u32>> =
        serde_json::from_value(controls["postings"].clone()).unwrap();
    let baseline_postings: Vec<Vec<u32>> =
        serde_json::from_value(disk["postings"].clone()).unwrap();
    let baseline_router: MlpRouter = serde_json::from_value(disk["router"].clone()).unwrap();
    let teacher = offsets(&centroid_postings, 99780);
    let dim = 768;
    let count = 99780;
    let tmp = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(tmp.path(), dim, Distance::Cosine).unwrap();
    for (i, row) in floats(base.join("corpus.f32"))
        .chunks_exact(dim)
        .enumerate()
    {
        plain
            .upsert_point(
                i as u64 + 1,
                (i as u64).into(),
                only_default_vector(row),
                &HardwareCounterCell::new(),
            )
            .unwrap();
    }
    let mut corpus = Vec::with_capacity(count * dim);
    {
        let storage = plain.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow();
        for id in 0..count as u32 {
            let crate::data_types::named_vectors::CowVector::Dense(row) =
                storage.get_vector_opt::<Random>(id).unwrap()
            else {
                panic!()
            };
            corpus.extend_from_slice(&row);
        }
    }
    let query_data = floats(base.join("queries.f32"));
    let queries: Vec<_> = query_data
        .chunks_exact(dim)
        .map(|r| QueryVector::from(r.to_vec()))
        .collect();
    let normalized: Vec<_> = query_data
        .chunks_exact(dim)
        .map(|r| Distance::Cosine.preprocess_vector::<f32>(r.to_vec()))
        .collect();
    let teacher_query: Vec<_> = normalized
        .iter()
        .map(|q| order(q, &centers, false))
        .collect();
    let exact = SearchParams {
        exact: true,
        ..Default::default()
    };
    let ground: Vec<_> = queries
        .iter()
        .map(|q| search(&plain, q, 10, Some(&exact)))
        .collect();
    if mode == "baseline" {
        // Establish source-offset equivalence, not merely aggregate equality.
        for (id, row) in corpus.chunks_exact(dim).enumerate() {
            assert_eq!(order(row, &centers, false)[0], teacher[id]);
        }
        let affine = affine_parameters(&centers);
        for q in &normalized {
            assert_eq!(order(q, &centers, false), order(q, &affine, true));
        }
        save(
            &out.join("ground_truth.json"),
            &json!({"teacher_query_order":teacher_query,"neighbors":ground.iter().map(|g|g.iter().map(|p|json!({"offset":p.idx,"external_id":p.idx,"score":p.score,"teacher_bucket":teacher[p.idx as usize]})).collect::<Vec<_>>()).collect::<Vec<_>>() }),
        );
        write_new(
            &out.join("teacher_labels.u8"),
            &teacher.iter().map(|&x| x as u8).collect::<Vec<_>>(),
        );
    }
    let jobs: Vec<Value> = if mode == "baseline" {
        vec![json!({"id":"baseline","sample_size":2048,"epochs":30,"seed":42})]
    } else {
        serde_json::from_slice(
            &std::fs::read(out.join(if mode == "test" {
                "selected_jobs.json"
            } else {
                "jobs.json"
            }))
            .unwrap(),
        )
        .unwrap()
    };
    for job in jobs {
        let name = job["id"].as_str().unwrap();
        let model_dir = out.join(name);
        std::fs::create_dir_all(&model_dir).unwrap();
        let done = model_dir.join(if mode == "test" {
            "test_done.json"
        } else {
            "done.json"
        });
        if done.exists() {
            eprintln!("F2 preserving completed {name} {mode}");
            continue;
        }
        let n = job["sample_size"].as_u64().unwrap() as usize;
        let mut sample: Vec<usize> = protocol["sample_order"].as_array().unwrap()[..n]
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        sample.sort_unstable();
        let mut train_data = Vec::with_capacity(n * dim);
        let mut labels = Vec::with_capacity(n);
        for &id in &sample {
            train_data.extend_from_slice(&corpus[id * dim..(id + 1) * dim]);
            labels.push(teacher[id] as i64);
        }
        let mut curve = vec![];
        let start = Instant::now();
        let router = if mode == "baseline" {
            baseline_router.clone()
        } else if mode == "test" {
            serde_json::from_slice(&std::fs::read(model_dir.join("router.json")).unwrap()).unwrap()
        } else {
            let cfg = LmiConfig {
                sample_size: n,
                n_buckets: 64,
                hidden_dim: 64,
                epochs: job["epochs"].as_u64().unwrap() as usize,
                seed: job["seed"].as_u64().unwrap(),
                batch_size: 256,
                nprobe: 1,
                kmeans_iterations: 20,
            };
            let mut observer = |epoch, loss, accuracy, optimizer_seconds| {
                curve.push(json!({"epoch":epoch,"loss":loss,"training_accuracy":accuracy,"optimizer_seconds":optimizer_seconds}));
            };
            super::super::training::train_labeled(
                &train_data,
                &labels,
                dim,
                &cfg,
                &AtomicBool::new(false),
                Some(&mut observer),
            )
            .unwrap()
        };
        let training_seconds = start.elapsed().as_secs_f64();
        if mode != "test" {
            if n == 2048 && job["epochs"] == 30 && job["seed"] == 42 {
                assert_eq!(
                    router, baseline_router,
                    "fixed-label trainer must reproduce saved model exactly"
                );
            }
            save(
                &model_dir.join("router.json"),
                &serde_json::to_value(&router).unwrap(),
            );
            save(
                &model_dir.join("training.json"),
                &json!({"job":job,"sample_offsets":sample,"curve":curve,"training_instrumented_seconds":training_seconds}),
            );
        }
        let start = Instant::now();
        let mut postings = vec![vec![]; 64];
        let mut ranks = Vec::with_capacity(count * 8);
        for (id, row) in corpus.chunks_exact(dim).enumerate() {
            let rank = router.top_buckets(row, 8).unwrap();
            postings[rank[0]].push(id as u32);
            ranks.extend(rank.iter().map(|&b| b as u8));
        }
        let posting_seconds = start.elapsed().as_secs_f64();
        if mode == "baseline" {
            assert_eq!(
                postings, baseline_postings,
                "saved postings must match exact original offset mapping"
            );
        }
        let mlp_labels = offsets(&postings, count);
        let query_order: Vec<_> = normalized
            .iter()
            .map(|q| router.top_buckets(q, 64).unwrap())
            .collect();
        if mode != "test" {
            write_new(&model_dir.join("corpus_top8.u8"), &ranks);
            save(&model_dir.join("query_order.json"), &json!(query_order));
            save(
                &model_dir.join("build.json"),
                &json!({"diagnostic_classification_posting_seconds":posting_seconds,"training_instrumented_seconds":training_seconds,"evaluation_build_seconds":training_seconds+posting_seconds,"bucket_sizes":postings.iter().map(Vec::len).collect::<Vec<_>>() }),
            );
        }
        let indices: Vec<usize> = if mode == "baseline" {
            (0..220).collect()
        } else {
            protocol["warmup"]
                .as_array()
                .unwrap()
                .iter()
                .chain(
                    protocol[if mode == "test" { "test" } else { "validation" }]
                        .as_array()
                        .unwrap(),
                )
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        };
        let filename = if mode == "test" {
            "test_queries.jsonl"
        } else {
            "queries.jsonl"
        };
        let mut output = std::io::BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(model_dir.join(filename))
                .unwrap(),
        );
        let mut trace = if mode == "baseline" {
            Some(std::io::BufWriter::new(
                File::create(model_dir.join("trace.jsonl")).unwrap(),
            ))
        } else {
            None
        };
        for trial in 0..2 {
            let mut probes = vec![1, 2, 4, 8, 16, 32, 64];
            if trial == 1 {
                probes.rotate_left(3);
            }
            for p in probes {
                for &qi in &indices {
                    let full = Instant::now();
                    let input = Distance::Cosine
                        .preprocess_vector::<f32>(query_data[qi * dim..(qi + 1) * dim].to_vec());
                    let start = Instant::now();
                    let buckets = router.top_buckets(&input, p).unwrap();
                    let router_ns = start.elapsed().as_nanos();
                    let start = Instant::now();
                    let mut candidates: Vec<u32> = buckets
                        .iter()
                        .flat_map(|&b| postings[b].iter().copied())
                        .collect();
                    candidates.sort_unstable();
                    candidates.dedup();
                    let tracker = plain.id_tracker.borrow();
                    let storage = plain.vector_data[DEFAULT_VECTOR_NAME]
                        .vector_storage
                        .borrow();
                    candidates.retain(|&id| {
                        (id as usize) < count
                            && !tracker.deleted_point_bitslice()[id as usize]
                            && !storage.is_deleted_vector(id)
                    });
                    let count_c = candidates.len();
                    let preparation_ns = start.elapsed().as_nanos();
                    let start = Instant::now();
                    let scorer = BatchFilteredSearcher::new(
                        &[&queries[qi]],
                        &*storage,
                        None::<&QuantizedVectors>,
                        None,
                        10,
                        tracker.deleted_point_bitslice(),
                        HardwareCounterCell::new(),
                    )
                    .unwrap();
                    let result = scorer
                        .peek_top_iter(candidates.into_iter(), &AtomicBool::new(false))
                        .unwrap()
                        .remove(0);
                    let scoring_ns = start.elapsed().as_nanos();
                    let total_ns = full.elapsed().as_nanos();
                    let hits = ground[qi]
                        .iter()
                        .filter(|g| result.iter().any(|x| x.idx == g.idx))
                        .count();
                    if p == 64 {
                        assert_eq!(result, ground[qi]);
                    }
                    if qi >= 20 {
                        let subset = if protocol["validation"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|x| x.as_u64().unwrap() as usize == qi)
                        {
                            "validation"
                        } else {
                            "test"
                        };
                        writeln!(output,"{}",json!({"model":name,"query":qi,"subset":subset,"trial":trial,"nprobe":p,"recall":hits as f64/10.0,"candidate_count":count_c,"candidate_fraction":count_c as f64/count as f64,"router_ns":router_ns,"preparation_ns":preparation_ns,"scoring_ns":scoring_ns,"total_ns":total_ns})).unwrap();
                    }
                    if let Some(trace) = trace.as_mut() {
                        if trial == 0 && qi >= 20 {
                            let t = &teacher_query[qi][..p];
                            let m = &query_order[qi][..p];
                            let mut counts = [0usize; 4];
                            let mut gains = [0usize; 2];
                            let mut losses = [0usize; 2];
                            for id in 0..count {
                                let states = [
                                    t.contains(&teacher[id]),
                                    m.contains(&teacher[id]),
                                    t.contains(&mlp_labels[id]),
                                    m.contains(&mlp_labels[id]),
                                ];
                                for i in 0..4 {
                                    counts[i] += usize::from(states[i]);
                                }
                                gains[0] += usize::from(!states[0] && states[1]);
                                losses[0] += usize::from(states[0] && !states[1]);
                                gains[1] += usize::from(!states[1] && states[3]);
                                losses[1] += usize::from(states[1] && !states[3]);
                            }
                            let neighbors:Vec<_>=ground[qi].iter().map(|g|json!({"offset":g.idx,"score":g.score,"teacher_bucket":teacher[g.idx as usize],"mlp_bucket":mlp_labels[g.idx as usize],"tt":t.contains(&teacher[g.idx as usize]),"mt":m.contains(&teacher[g.idx as usize]),"tm":t.contains(&mlp_labels[g.idx as usize]),"mm":m.contains(&mlp_labels[g.idx as usize])})).collect();
                            writeln!(trace,"{}",json!({"query":qi,"nprobe":p,"teacher_selected":t,"mlp_selected":m,"neighbors":neighbors,"counts_tt_mt_tm_mm":counts,"query_route_added_removed":[gains[0],losses[0]],"corpus_reassignment_added_removed":[gains[1],losses[1]]})).unwrap();
                        }
                    }
                }
            }
        }
        output.flush().unwrap();
        if let Some(mut t) = trace {
            t.flush().unwrap();
        }
        save(&done, &json!({"complete":true,"mode":mode,"job":job}));
        eprintln!("F2 complete {name} {mode}");
    }
}
