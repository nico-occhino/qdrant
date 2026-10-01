//! F.3: frozen routers, fresh internal queries, dense budgets and paired timing.
use super::*;
use crate::entry::entry_point::NonAppendableSegmentEntry;
use std::fs::OpenOptions;

fn canonical_top10(rows: &mut Vec<ScoredPointOffset>) {
    rows.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.idx.cmp(&b.idx)));
    rows.truncate(10);
}

fn save(path: &Path, value: &Value) {
    let f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer_pretty(f, value).unwrap();
}

#[test]
#[ignore = "Explicit bounded F.3 frozen-model evaluation"]
fn phase_f3_study() {
    assert!(
        std::process::Command::new("taskset")
            .args(["-apc", "0", &std::process::id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    let out = PathBuf::from(std::env::var("LMI_PHASE_F3_DIR").unwrap());
    let p: Value =
        serde_json::from_slice(&std::fs::read(out.join("protocol.json")).unwrap()).unwrap();
    let base = PathBuf::from(p["base"].as_str().unwrap());
    let f2 = PathBuf::from(p["f2"].as_str().unwrap());
    let count = p["original_count"].as_u64().unwrap() as usize;
    let live = p["live_count"].as_u64().unwrap() as usize;
    let dim = 768;
    let qids: Vec<usize> = serde_json::from_value(p["query_offsets"].clone()).unwrap();
    let removed: HashSet<u32> = serde_json::from_value::<Vec<u32>>(p["removed_offsets"].clone())
        .unwrap()
        .into_iter()
        .collect();
    let raw = floats(base.join("corpus.f32"));
    let tmp = tempfile::tempdir().unwrap();
    let mut plain = build_simple_segment(tmp.path(), dim, Distance::Cosine).unwrap();
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
                    &HardwareCounterCell::new()
                )
                .unwrap()
        );
    }
    let queries: Vec<_> = qids
        .iter()
        .map(|&id| QueryVector::from(raw[id * dim..(id + 1) * dim].to_vec()))
        .collect();
    let normalized: Vec<_> = qids
        .iter()
        .map(|&id| {
            Distance::Cosine.preprocess_vector::<f32>(raw[id * dim..(id + 1) * dim].to_vec())
        })
        .collect();
    let controls: Value =
        serde_json::from_slice(&std::fs::read(base.join("centroid_state.json")).unwrap()).unwrap();
    let centers: Vec<Vec<f64>> = serde_json::from_value(controls["centers"].clone()).unwrap();
    let affine = affine_parameters(&centers);
    let teacher: Vec<Vec<u32>> = serde_json::from_value(controls["postings"].clone()).unwrap();
    let mut names = vec!["centroid".to_owned(), "affine".to_owned()];
    let models: Vec<String> = serde_json::from_value(p["models"].clone()).unwrap();
    names.extend(models.clone());
    let routers: Vec<MlpRouter> = models
        .iter()
        .map(|m| {
            serde_json::from_slice(&std::fs::read(f2.join(m).join("router.json")).unwrap()).unwrap()
        })
        .collect();
    let mut postings = vec![teacher.clone(), teacher];
    for m in &models {
        let bytes = std::fs::read(f2.join(m).join("corpus_top8.u8")).unwrap();
        assert_eq!(bytes.len(), count * 8);
        let mut buckets = vec![vec![]; 64];
        for id in 0..count {
            buckets[bytes[id * 8] as usize].push(id as u32);
        }
        postings.push(buckets);
    }
    // Frozen partition membership; remove query rows without relabeling any survivor.
    for buckets in &mut postings {
        for bucket in buckets.iter_mut() {
            bucket.retain(|id| !removed.contains(id));
        }
        assert_eq!(buckets.iter().map(Vec::len).sum::<usize>(), live);
        let unique: HashSet<_> = buckets.iter().flatten().copied().collect();
        assert_eq!(unique.len(), live);
    }
    // Verify saved MLP assignments against native inference on all retained vectors.
    {
        let storage = plain.vector_data[DEFAULT_VECTOR_NAME]
            .vector_storage
            .borrow();
        for (m, router) in routers.iter().enumerate() {
            for (b, ids) in postings[m + 2].iter().enumerate() {
                for &id in ids {
                    let crate::data_types::named_vectors::CowVector::Dense(row) =
                        storage.get_vector_opt::<Random>(id).unwrap()
                    else {
                        panic!()
                    };
                    assert_eq!(router.top_buckets(&row, 1).unwrap()[0], b);
                }
            }
        }
    }
    let exact = SearchParams {
        exact: true,
        ..Default::default()
    };
    let gt_start = Instant::now();
    let tie_expansions = std::cell::Cell::new(0usize);
    let ground: Vec<_> = queries
        .iter()
        .map(|q| {
            let mut limit = 11;
            loop {
                let mut rows = search(&plain, q, limit, Some(&exact));
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
    for g in &ground {
        assert_eq!(g.len(), 10);
        assert!(g.iter().all(|x| !removed.contains(&x.idx)));
    }
    save(&out.join("ground_truth.json"), &json!(ground.iter().enumerate().map(|(qi,g)| json!({"query":qi,"original_offset":qids[qi],"neighbors":g.iter().map(|x|json!({"offset":x.idx,"score":x.score})).collect::<Vec<_>>()})).collect::<Vec<_>>()));
    eprintln!(
        "F3 ground truth: {} queries, {:.2}s",
        queries.len(),
        gt_start.elapsed().as_secs_f64()
    );
    let rank = |m: usize, input: &[f32]| -> Vec<usize> {
        match m {
            0 => order(input, &centers, false),
            1 => order(input, &affine, true),
            _ => routers[m - 2].top_buckets(input, 64).unwrap(),
        }
    };
    let mut labels = vec![vec![usize::MAX; count]; names.len()];
    for m in 0..names.len() {
        for b in 0..64 {
            for &id in &postings[m][b] {
                labels[m][id as usize] = b;
            }
        }
    }
    let warm = 20;
    let val_end = 320;
    let mut choices = vec![vec![]; names.len()];
    let mut curves = std::io::BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("dense_queries.jsonl"))
            .unwrap(),
    );
    // Validation pass is completed and choices persisted before any test ranking.
    for (subset, begin, end) in [
        ("validation", warm, val_end),
        ("test", val_end, queries.len()),
    ] {
        for m in 0..names.len() {
            let mut sums = vec![0usize; 64];
            for qi in begin..end {
                let r = rank(m, &normalized[qi]);
                if m == 0 {
                    assert_eq!(r, rank(1, &normalized[qi]));
                }
                let mut candidates = 0;
                let mut hits = 0;
                for (j, &b) in r.iter().enumerate() {
                    candidates += postings[m][b].len();
                    hits += ground[qi]
                        .iter()
                        .filter(|x| labels[m][x.idx as usize] == b)
                        .count();
                    sums[j] += hits;
                    writeln!(curves,"{}",json!({"model":names[m],"query":qi,"subset":subset,"nprobe":j+1,"hits":hits,"recall":hits as f64/10.0,"candidate_count":candidates,"candidate_fraction":candidates as f64/live as f64})).unwrap();
                }
                assert_eq!(hits, 10);
                assert_eq!(candidates, live);
            }
            if subset == "validation" {
                for target in [0.90, 0.95] {
                    choices[m].push(
                        sums.iter()
                            .position(|&h| h as f64 / (10 * (end - begin)) as f64 >= target)
                            .unwrap()
                            + 1,
                    );
                }
            }
        }
        if subset == "validation" {
            save(
                &out.join("selection.json"),
                &json!({"methods":names,"targets":[0.90,0.95],"nprobe":choices,"selected_before_test":true}),
            );
        }
    }
    curves.flush().unwrap();
    let mut timing = std::io::BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("timing_queries.jsonl"))
            .unwrap(),
    );
    let tracker = plain.id_tracker.borrow();
    let storage = plain.vector_data[DEFAULT_VECTOR_NAME]
        .vector_storage
        .borrow();
    let measure = |m: usize, qi: usize, probe: usize| {
        let total = Instant::now();
        let input = Distance::Cosine
            .preprocess_vector::<f32>(raw[qids[qi] * dim..(qids[qi] + 1) * dim].to_vec());
        let start = Instant::now();
        let ranks = rank(m, &input);
        let router_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let mut candidates: Vec<u32> = ranks[..probe]
            .iter()
            .flat_map(|&b| postings[m][b].iter().copied())
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        candidates.retain(|&id| {
            !tracker.deleted_point_bitslice()[id as usize] && !storage.is_deleted_vector(id)
        });
        let n = candidates.len();
        let preparation_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let mut limit = 11;
        let result = loop {
            let scorer = BatchFilteredSearcher::new(
                &[&queries[qi]],
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
            if rows.len() > 10 && rows[9].score == rows.last().unwrap().score && limit < n {
                tie_expansions.set(tie_expansions.get() + 1);
                limit = (limit * 2).min(n);
                continue;
            }
            canonical_top10(&mut rows);
            break rows;
        };
        let scoring_ns = start.elapsed().as_nanos();
        let total_ns = total.elapsed().as_nanos();
        assert!(result.iter().all(|x| !removed.contains(&x.idx)));
        let hits = ground[qi]
            .iter()
            .filter(|g| result.iter().any(|x| x.idx == g.idx))
            .count();
        let predicted = ground[qi]
            .iter()
            .filter(|g| ranks[..probe].contains(&labels[m][g.idx as usize]))
            .count();
        assert_eq!(
            hits, predicted,
            "membership recall must agree with native scorer"
        );
        if probe == 64 {
            assert_eq!(result, ground[qi]);
        }
        json!({"model":names[m],"query":qi,"nprobe":probe,"recall":hits as f64/10.0,"candidate_count":n,"candidate_fraction":n as f64/live as f64,"router_ns":router_ns,"preparation_ns":preparation_ns,"scoring_ns":scoring_ns,"total_ns":total_ns})
    };
    // Full-probe guard for every query; no full-probe timings enter comparison.
    for qi in 0..queries.len() {
        measure(0, qi, 64);
    }
    // All methods also checked at full probe on separate warmup queries.
    for m in 1..names.len() {
        for qi in 0..warm {
            measure(m, qi, 64);
        }
    }
    for trial in 0..2 {
        for target in 0..2 {
            for qi in (0..warm).chain(val_end..queries.len()) {
                for step in 0..names.len() {
                    let m = (step + qi + trial * 2 + target) % names.len();
                    let mut row = measure(m, qi, choices[m][target]);
                    if qi >= val_end {
                        row["trial"] = json!(trial);
                        row["target"] = json!([0.90, 0.95][target]);
                        row["order_slot"] = json!(step);
                        writeln!(timing, "{row}").unwrap();
                    }
                }
            }
            timing.flush().unwrap();
            eprintln!("F3 paired timing trial={trial} target={target} complete");
        }
    }
    save(
        &out.join("done.json"),
        &json!({"complete":true,"live_count":live,"queries":queries.len(),"centroid_affine_rank_mismatches":0,"native_assignment_checks":live*3,"full_probe_native_checks":queries.len()+4*warm,"measured_timing_rows":2*2*5*(queries.len()-val_end),"no_training":true,"tie_expansion_calls":tie_expansions.get(),"tie_rule":"score descending then segment offset ascending; retrieve k+1 and expand through the complete boundary tie before truncating to k=10","process_status":std::fs::read_to_string("/proc/self/status").unwrap()}),
    );
}
