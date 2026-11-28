#![cfg(all(feature = "lmi-training", feature = "cli-research", feature = "testing"))]

//! CLI Stage A (research/dli/cli/CLI_QDRANT_DESIGN.md §6b). Two tests:
//! * `cli_core_synthetic_loop`: a small synthetic stream through build, inserts, deletes and maintenance, for MLP and
//!   centroid routing. G2 runs inside every maintenance (`check_membership`), and exhaustive search must equal brute
//!   force.
//! * `cli_stage_a_plan` (ignored): replays a Phase 3 plan and records, after every batch, recall and vectors scored
//!   on one validation panel, plus the maintenance record. It is driven by environment variables:
//!   `CLI_PLAN`, `CLI_CORPUS` (Float16, 768-d, row = id − 1), `CLI_SET`, `CLI_OUT` (must not exist),
//!   `CLI_ROUTING` (mlp | centroid), `CLI_SIGNAL` (entropy | imbalance), `CLI_BMAX` (default 1000),
//!   `CLI_MAX_BATCHES` (optional) and `CLI_THREADS` (default 8).

use std::io::Write;
use std::path::{Path, PathBuf};

use common::ambient;
use half::f16;
use segment::index::lmi_index::cli::DriftSignal;
use segment::index::lmi_index::cli_core::{CliCore, CliParams, RoutingKind};

const DIM: usize = 768;
const K: usize = 30;
const BUDGETS: [usize; 17] =
    [1000, 2000, 3500, 5000, 7000, 10000, 14000, 20000, 28000, 40000, 50000, 56000, 80000, 113000, 160000, 226000, 320000];

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

fn u32s(path: &Path) -> Vec<u32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
}

#[test]
fn cli_core_synthetic_loop() {
    let _scope = ambient::test_guard();
    let dim = 8;
    let n = 6_000usize;
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut vectors = vec![0f32; (n + 1) * dim];
    for id in 1..=n {
        // A slow drift: later ids lean toward a second direction.
        let drift = id as f32 / n as f32;
        for j in 0..dim {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let noise = (x % 2001) as f32 / 1000.0 - 1.0;
            vectors[id * dim + j] = noise + if j == 0 { 2.0 * (1.0 - drift) } else if j == 1 { 2.0 * drift } else { 0.0 };
        }
        normalize(&mut vectors[id * dim..(id + 1) * dim]);
    }
    for routing in [RoutingKind::Mlp, RoutingKind::Centroid] {
        let mut params = CliParams::thesis(200, routing, DriftSignal::Entropy);
        params.build_epochs = 5;
        params.finetune_epochs = 3;
        params.threads = 2;
        let init: Vec<u32> = (1..=1_000).collect();
        let (mut core, rec) = CliCore::build(params, dim, &vectors, n + 1, &init).unwrap();
        assert_eq!(rec.buckets_after, 10, "eq. 2.1: 2 * 1000 / 200");
        let mut live: Vec<u32> = init.clone();
        let (mut splits, mut created, mut removed) = (0, 0, 0);
        for batch in 0..5u32 {
            let ids: Vec<u32> = (1_001 + batch * 1_000..=2_000 + batch * 1_000).collect();
            core.insert(&ids);
            live.extend(&ids);
            if batch == 2 {
                let gone: Vec<u32> = live.iter().copied().filter(|id| id % 7 == 0).collect();
                core.delete(&gone);
                live.retain(|id| id % 7 != 0);
            }
            let rec = core.maintain().unwrap(); // G2 runs inside
            println!("{routing:?} batch {batch}: {}", serde_json::to_string(&rec).unwrap());
            assert_eq!(rec.live, live.len());
            splits += rec.split_buckets + rec.reclustered_buckets;
            created += rec.created;
            removed += rec.removed;
            core.check_membership().unwrap();
        }
        assert!(splits > 0 && created > 0, "{routing:?}: maintenance never restructured ({splits}, {created}, {removed})");
        // Exhaustive search equals brute force over live points.
        let q: Vec<f32> = vectors[17 * dim..18 * dim].to_vec();
        let got = core.search(&q, &[usize::MAX], K).pop().unwrap();
        assert_eq!(got.0, live.len(), "exhaustive search scores every live point once");
        let mut brute: Vec<(f32, u32)> =
            live.iter().map(|&id| (q.iter().zip(&vectors[id as usize * dim..]).map(|(a, b)| a * b).sum(), id)).collect();
        brute.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let want: Vec<u32> = brute.iter().take(K).map(|p| p.1).collect();
        assert_eq!(got.1, want, "{routing:?}");
    }
}

#[test]
#[ignore = "Stage A plan replay; needs CLI_PLAN, CLI_CORPUS, CLI_SET, CLI_OUT"]
fn cli_stage_a_plan() {
    let _scope = ambient::test_guard();
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is required"));
    let plan_dir = PathBuf::from(var("CLI_PLAN"));
    let out = PathBuf::from(var("CLI_OUT"));
    std::fs::create_dir(&out).expect("CLI_OUT must not exist");
    let set = var("CLI_SET");
    let routing = match var("CLI_ROUTING").as_str() {
        "mlp" => RoutingKind::Mlp,
        "centroid" => RoutingKind::Centroid,
        r => panic!("CLI_ROUTING {r}"),
    };
    let signal = match std::env::var("CLI_SIGNAL").as_deref().unwrap_or("entropy") {
        "entropy" => DriftSignal::Entropy,
        "imbalance" => DriftSignal::Imbalance,
        s => panic!("CLI_SIGNAL {s}"),
    };
    let b_max: usize = std::env::var("CLI_BMAX").map_or(1000, |v| v.parse().unwrap());
    let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(plan_dir.join("plan.json")).unwrap()).unwrap();
    let batches = plan["batches"].as_array().unwrap();
    let max_batches: usize = std::env::var("CLI_MAX_BATCHES").map_or(batches.len(), |v| v.parse().unwrap());
    // Every id the plan can insert, then the corpus rows (row = id - 1), widened, normalized; row 0 is unused.
    let streams: Vec<(Vec<u32>, Vec<u32>)> = batches
        .iter()
        .take(max_batches)
        .map(|b| {
            let d = plan_dir.join(format!("batch_{:02}", b["index"].as_u64().unwrap()));
            (u32s(&d.join("insert.u32")), u32s(&d.join("delete.u32")))
        })
        .collect();
    let max_id = streams.iter().flat_map(|(i, _)| i.iter()).copied().max().unwrap() as usize;
    let bytes = std::fs::read(var("CLI_CORPUS")).unwrap();
    assert!(bytes.len() >= max_id * DIM * 2, "corpus shorter than the plan's ids");
    let mut vectors = vec![0f32; (max_id + 1) * DIM];
    for id in 1..=max_id {
        let src = &bytes[(id - 1) * DIM * 2..id * DIM * 2];
        let dst = &mut vectors[id * DIM..(id + 1) * DIM];
        for (d, b) in dst.iter_mut().zip(src.chunks_exact(2)) {
            *d = f16::from_le_bytes([b[0], b[1]]).to_f32();
        }
        normalize(dst);
    }
    drop(bytes);
    let mut params = CliParams::thesis(b_max, routing, signal);
    params.threads = std::env::var("CLI_THREADS").map_or(8, |v| v.parse().unwrap());
    let mut rows = std::fs::File::create(out.join("rows.jsonl")).unwrap();
    let mut log = std::fs::File::create(out.join("batches.jsonl")).unwrap();
    let mut core: Option<CliCore> = None;
    for (t, (ins, del)) in streams.iter().enumerate() {
        let started = std::time::Instant::now();
        let rec = match core.as_mut() {
            None => {
                assert!(del.is_empty(), "batch 0 has no deletes");
                let (c, rec) = CliCore::build(params.clone(), DIM, &vectors, max_id + 1, ins).unwrap();
                core = Some(c);
                rec
            }
            Some(c) => {
                c.insert(ins);
                c.delete(del);
                c.maintain().unwrap()
            }
        };
        let c = core.as_ref().unwrap();
        let spec = &batches[t]["sets"][&set];
        let qn = spec["n"].as_u64().unwrap() as usize;
        let qbytes = std::fs::read(plan_dir.join(spec["queries"].as_str().unwrap())).unwrap();
        let gold: Vec<u64> = std::fs::read(plan_dir.join(spec["gold"].as_str().unwrap()))
            .unwrap()
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let eval_started = std::time::Instant::now();
        for q in 0..qn {
            let mut query: Vec<f32> =
                qbytes[q * DIM * 4..(q + 1) * DIM * 4].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            normalize(&mut query);
            let g: std::collections::HashSet<u64> = gold[q * K..(q + 1) * K].iter().copied().collect();
            for ((scored, top), budget) in c.search(&query, &BUDGETS, K).into_iter().zip(BUDGETS) {
                let hit = top.iter().filter(|&&id| g.contains(&(id as u64))).count();
                writeln!(rows, "{}", serde_json::json!({"batch": t, "set": set, "query": q, "budget": budget,
                    "scored": scored, "recall30": hit})).unwrap();
            }
        }
        let record = serde_json::json!({"batch": t, "routing": format!("{routing:?}"), "signal": format!("{signal:?}"),
            "b_max": b_max, "maintenance": rec, "buckets": c.buckets(), "batch_seconds": started.elapsed().as_secs_f64(),
            "eval_seconds": eval_started.elapsed().as_secs_f64()});
        writeln!(log, "{record}").unwrap();
        println!("CLI_BATCH {record}");
    }
}
