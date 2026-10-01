//! F.4: fixed learned corpus partition; oracle headroom and retrieval-target query routers.
use super::*;
use crate::entry::entry_point::NonAppendableSegmentEntry;
use std::fs::OpenOptions;

fn canonical_top10(rows: &mut Vec<ScoredPointOffset>) {
    rows.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.idx.cmp(&b.idx)));
    rows.truncate(10);
}

fn save(path: &Path, value: &Value) {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer_pretty(file, value).unwrap();
}

fn bucket_target(ground: &[ScoredPointOffset], labels: &[usize]) -> Vec<u8> {
    let mut result = vec![0u8; 64];
    for neighbor in ground {
        result[labels[neighbor.idx as usize]] += 1;
    }
    assert_eq!(result.iter().map(|&n| usize::from(n)).sum::<usize>(), 10);
    result
}

#[test]
#[ignore = "Explicit F.4 oracle-headroom and retrieval-target control"]
fn phase_f4_oracle_and_objective_control() {
    assert!(
        std::process::Command::new("taskset")
            .args(["-apc", "0", &std::process::id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let out = PathBuf::from(std::env::var("LMI_PHASE_F4_DIR").unwrap());
    let protocol: Value =
        serde_json::from_slice(&std::fs::read(out.join("protocol.json")).unwrap()).unwrap();
    let base = PathBuf::from(protocol["base"].as_str().unwrap());
    let f2 = PathBuf::from(protocol["f2"].as_str().unwrap());
    let dim = protocol["dimension"].as_u64().unwrap() as usize;
    let count = protocol["original_count"].as_u64().unwrap() as usize;
    let live = protocol["live_count"].as_u64().unwrap() as usize;
    let warm = protocol["warmup"].as_u64().unwrap() as usize;
    let train_end = warm + protocol["train"].as_u64().unwrap() as usize;
    let validation_end = train_end + protocol["validation"].as_u64().unwrap() as usize;
    let query_offsets: Vec<usize> =
        serde_json::from_value(protocol["query_offsets"].clone()).unwrap();
    assert_eq!(
        query_offsets.len(),
        validation_end + protocol["test"].as_u64().unwrap() as usize
    );
    let removed: HashSet<u32> =
        serde_json::from_value::<Vec<u32>>(protocol["removed_offsets"].clone())
            .unwrap()
            .into_iter()
            .collect();
    let partition_name = protocol["partition_model"].as_str().unwrap();
    let raw = floats(base.join("corpus.f32"));
    let root = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(root.path(), dim, Distance::Cosine).unwrap();
    for (id, row) in raw.chunks_exact(dim).enumerate() {
        plain
            .upsert_point(
                id as u64 + 1,
                (id as u64).into(),
                only_default_vector(row),
                &HardwareCounterCell::new(),
            )
            .unwrap();
    }
    for &id in &removed {
        assert!(
            plain
                .delete_point(
                    count as u64 + id as u64 + 1,
                    (id as u64).into(),
                    &HardwareCounterCell::new(),
                )
                .unwrap()
        );
    }
    let queries: Vec<_> = query_offsets
        .iter()
        .map(|&id| QueryVector::from(raw[id * dim..(id + 1) * dim].to_vec()))
        .collect();
    let normalized: Vec<_> = query_offsets
        .iter()
        .map(|&id| {
            Distance::Cosine.preprocess_vector::<f32>(raw[id * dim..(id + 1) * dim].to_vec())
        })
        .collect();
    let source = f2.join(partition_name);
    let existing: MlpRouter =
        serde_json::from_slice(&std::fs::read(source.join("router.json")).unwrap()).unwrap();
    let rank_bytes = std::fs::read(source.join("corpus_top8.u8")).unwrap();
    assert_eq!(rank_bytes.len(), count * 8);
    let mut postings = vec![Vec::new(); 64];
    for id in 0..count {
        postings[rank_bytes[id * 8] as usize].push(id as u32);
    }
    for bucket in &mut postings {
        bucket.retain(|id| !removed.contains(id));
    }
    assert_eq!(postings.iter().map(Vec::len).sum::<usize>(), live);
    let unique: HashSet<_> = postings.iter().flatten().copied().collect();
    assert_eq!(unique.len(), live);
    let mut labels = vec![usize::MAX; count];
    for (bucket, ids) in postings.iter().enumerate() {
        for &id in ids {
            labels[id as usize] = bucket;
        }
    }
    let exact = SearchParams {
        exact: true,
        ..Default::default()
    };
    let tie_expansions = std::cell::Cell::new(0usize);
    let ground: Vec<_> = queries
        .iter()
        .map(|query| {
            let mut limit = 11;
            loop {
                let mut rows = search(&plain, query, limit, Some(&exact));
                if rows.len() > 10 && rows[9].score == rows.last().unwrap().score && limit < live {
                    tie_expansions.set(tie_expansions.get() + 1);
                    limit = (limit * 2).min(live);
                    continue;
                }
                canonical_top10(&mut rows);
                break rows;
            }
        })
        .collect();
    assert!(ground.iter().all(|rows| rows.len() == 10));
    save(
        &out.join("ground_truth.json"),
        &json!(ground.iter().enumerate().map(|(query, rows)| json!({"query":query,"original_offset":query_offsets[query],"neighbors":rows.iter().map(|x|json!({"offset":x.idx,"score":x.score,"bucket":labels[x.idx as usize]})).collect::<Vec<_>>()})).collect::<Vec<_>>()),
    );

    // Each query contributes ten labeled copies: exact empirical CE over r_b(q)/10.
    let mut train_data = Vec::with_capacity((train_end - warm) * 10 * dim);
    let mut train_labels = Vec::with_capacity((train_end - warm) * 10);
    let mut target_rows = Vec::new();
    for query in warm..train_end {
        let target = bucket_target(&ground[query], &labels);
        target_rows.push(
            json!({"query":query,"original_offset":query_offsets[query],"bucket_counts":target}),
        );
        for (bucket, &multiplicity) in target.iter().enumerate() {
            for _ in 0..multiplicity {
                train_data.extend_from_slice(&normalized[query]);
                train_labels.push(bucket as i64);
            }
        }
    }
    assert_eq!(train_labels.len(), (train_end - warm) * 10);
    save(&out.join("train_targets.json"), &json!(target_rows));
    let seeds: Vec<u64> = serde_json::from_value(protocol["router_seeds"].clone()).unwrap();
    let cfg_base = LmiConfig {
        sample_size: train_labels.len(),
        n_buckets: 64,
        hidden_dim: protocol["router_config"]["hidden_dim"].as_u64().unwrap() as usize,
        epochs: protocol["router_config"]["epochs"].as_u64().unwrap() as usize,
        batch_size: protocol["router_config"]["batch_size"].as_u64().unwrap() as usize,
        nprobe: 1,
        kmeans_iterations: 20,
        seed: 0,
    };
    let mut names = vec!["existing_mlp_s42".to_owned()];
    let mut routers = vec![existing];
    let mut training = Vec::new();
    for seed in seeds {
        let cfg = LmiConfig {
            seed,
            ..cfg_base.clone()
        };
        let started = Instant::now();
        let router = super::super::training::train_labeled(
            &train_data,
            &train_labels,
            dim,
            &cfg,
            &AtomicBool::new(false),
            None,
        )
        .unwrap();
        let name = format!("retrieval_target_mlp_s{seed}");
        save(
            &out.join(format!("{name}.json")),
            &serde_json::to_value(&router).unwrap(),
        );
        training.push(json!({"model":name,"seed":seed,"seconds":started.elapsed().as_secs_f64(),"input_rows":train_labels.len(),"unique_queries":train_end-warm,"architecture":"768 -> 64 ReLU -> 64","objective":"empirical cross entropy over exact-neighbour fixed-bucket labels"}));
        names.push(name);
        routers.push(router);
    }
    save(&out.join("training.json"), &json!(training));

    let rank = |router: &MlpRouter, input: &[f32]| router.top_buckets(input, 64).unwrap();
    let oracle_rank = |query: usize| {
        let target = bucket_target(&ground[query], &labels);
        let mut rank: Vec<_> = (0..64).collect();
        rank.sort_by(|&a, &b| target[b].cmp(&target[a]).then(a.cmp(&b)));
        (rank, target)
    };
    let mut router_out = std::io::BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("router_queries.jsonl"))
            .unwrap(),
    );
    let mut oracle_out = std::io::BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("oracle_queries.jsonl"))
            .unwrap(),
    );
    let subsets = [
        ("validation", train_end, validation_end),
        ("test", validation_end, queries.len()),
    ];
    let mut choices = vec![vec![]; names.len()];
    for (subset, start, end) in subsets {
        for query in start..end {
            let (oracle, target) = oracle_rank(query);
            let mut hits = 0usize;
            let mut candidates = 0usize;
            for (index, &bucket) in oracle.iter().enumerate() {
                hits += usize::from(target[bucket]);
                candidates += postings[bucket].len();
                writeln!(oracle_out, "{}", json!({"query":query,"subset":subset,"nprobe":index+1,"oracle_recall":hits as f64/10.0,"candidate_count":candidates,"candidate_fraction":candidates as f64/live as f64,"r":target[bucket]})).unwrap();
            }
            assert_eq!(hits, 10);
            for (model, router) in names.iter().zip(&routers) {
                let ranking = rank(router, &normalized[query]);
                let mut router_hits = 0usize;
                let mut router_candidates = 0usize;
                for (index, &bucket) in ranking.iter().enumerate() {
                    router_hits += usize::from(target[bucket]);
                    router_candidates += postings[bucket].len();
                    writeln!(router_out, "{}", json!({"model":model,"query":query,"subset":subset,"nprobe":index+1,"recall":router_hits as f64/10.0,"candidate_count":router_candidates,"candidate_fraction":router_candidates as f64/live as f64})).unwrap();
                }
                assert_eq!(router_hits, 10);
            }
        }
        if subset == "validation" {
            for (model, router) in names.iter().zip(&routers) {
                let mut totals = vec![0usize; 64];
                for query in start..end {
                    let (_, target) = oracle_rank(query);
                    let mut cumulative = 0usize;
                    for (index, bucket) in rank(router, &normalized[query]).iter().enumerate() {
                        cumulative += usize::from(target[*bucket]);
                        totals[index] += cumulative;
                    }
                }
                for wanted in [0.90, 0.95] {
                    choices[names.iter().position(|x| x == model).unwrap()].push(
                        totals
                            .iter()
                            .position(|&n| n as f64 / (10 * (end - start)) as f64 >= wanted)
                            .unwrap()
                            + 1,
                    );
                }
            }
            save(
                &out.join("selection.json"),
                &json!({"methods":names,"targets":[0.90,0.95],"nprobe":choices,"selected_before_test":true}),
            );
        }
    }
    router_out.flush().unwrap();
    oracle_out.flush().unwrap();

    // Independent scoring checks: p=4 on 100 deterministic test queries plus full-probe warmups.
    let tracker = plain.id_tracker.borrow();
    let storage = plain.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let check = |router: &MlpRouter, query: usize, p: usize| {
        let ranking = rank(router, &normalized[query]);
        let mut candidates: Vec<u32> = ranking[..p]
            .iter()
            .flat_map(|&b| postings[b].iter().copied())
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        candidates.retain(|&id| {
            !tracker.deleted_point_bitslice()[id as usize] && !storage.is_deleted_vector(id)
        });
        let predicted = ground[query]
            .iter()
            .filter(|x| ranking[..p].contains(&labels[x.idx as usize]))
            .count();
        let mut limit = 11;
        let result = loop {
            let scorer = BatchFilteredSearcher::new(
                &[&queries[query]],
                &*storage,
                None::<&QuantizedVectors>,
                None,
                limit,
                tracker.deleted_point_bitslice(),
                HardwareCounterCell::new(),
            )
            .unwrap();
            let mut rows = scorer
                .peek_top_iter(candidates.iter().copied(), &AtomicBool::new(false))
                .unwrap()
                .remove(0);
            if rows.len() > 10
                && rows[9].score == rows.last().unwrap().score
                && limit < candidates.len()
            {
                tie_expansions.set(tie_expansions.get() + 1);
                limit = (limit * 2).min(candidates.len());
                continue;
            }
            canonical_top10(&mut rows);
            break rows;
        };
        let actual = ground[query]
            .iter()
            .filter(|x| result.iter().any(|r| r.idx == x.idx))
            .count();
        assert_eq!(actual, predicted);
        if p == 64 {
            assert_eq!(result, ground[query]);
        }
    };
    for router in &routers {
        for query in 0..warm {
            check(router, query, 64);
        }
        for query in (validation_end..validation_end + 100).step_by(1) {
            check(router, query, 4);
        }
    }
    save(
        &out.join("done.json"),
        &json!({"complete":true,"partition_model":partition_name,"live_count":live,"query_count":queries.len(),"train_queries":train_end-warm,"no_qdrant_serving_change":true,"no_corpus_reassignment":true,"trained_query_routers":routers.len()-1,"native_checks":routers.len()*(warm+100),"tie_expansion_calls":tie_expansions.get(),"process_status":std::fs::read_to_string("/proc/self/status").unwrap()}),
    );
}
