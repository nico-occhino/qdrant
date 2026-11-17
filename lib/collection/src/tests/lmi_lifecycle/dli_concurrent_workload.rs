//! Tier A concurrent DLI workload: correctness under concurrent readers and a
//! writer that drives automatic native publications, plus contention metrics.
//!
//! Vectors are unit axes in 16 dimensions, so every cosine score is exactly 0
//! or 1 and a hit's score identifies whether the matching point version had
//! the query axis. The candidate budget covers all buckets, so learned routing
//! is exhaustive and any wrong hit is a visibility defect, not ANN recall loss.
//!
//! Checks:
//! * interval linearizability: each concurrent hit must match some version of
//!   that point whose lifetime overlaps the read's [start, end] interval;
//! * at quiescence and after graceful reopen, every axis query returns exactly
//!   the live points of that axis according to the operation log;
//! * no duplicate external IDs in any result.
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use super::*;
use crate::operations::types::Datatype;

const DIM: usize = 16;

async fn shard(c: &Collection) -> Arc<crate::shards::replica_set::ShardReplicaSet> {
    c.shards_holder().read().await.get_shard(0).unwrap().clone()
}

async fn apply(c: &Collection, op: PointOperations) {
    shard(c)
        .await
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(op)),
            WaitUntil::Visible,
            None,
            false,
        )
        .await
        .expect("mutation must apply")
        .expect("local replica must apply");
}

async fn axis_query(c: &Collection, axis: usize, limit: usize) -> Vec<(u64, f32)> {
    let mut q = vec![0.0; DIM];
    q[axis] = 1.0;
    c.search(
        CoreSearchRequest {
            query: QueryEnum::Nearest(NamedQuery::default_dense(q)),
            filter: None,
            params: None,
            limit,
            offset: 0,
            with_payload: None,
            with_vector: None,
            score_threshold: None,
        },
        None,
        None,
        &ShardSelectorInternal::All,
        None,
    )
    .await
    .unwrap()
    .into_iter()
    .map(|p| {
        let segment::types::PointIdType::NumId(id) = p.id else {
            panic!("numeric ids")
        };
        (id, p.score)
    })
    .collect()
}

fn config(buffer_capacity: usize) -> CollectionConfigInternal {
    let mut vp = VectorParamsBuilder::new(DIM as u64, Distance::Cosine).build();
    vp.datatype = Some(Datatype::Float16);
    let mut opt = OptimizersConfig::fixture();
    opt.indexing_threshold = None;
    opt.max_optimization_threads = Some(0);
    opt.default_segment_number = 1;
    opt.max_segment_size = Some(1_000_000);
    CollectionConfigInternal {
        params: CollectionParams {
            vectors: VectorsConfig::Single(vp),
            shard_number: NonZeroU32::new(1).unwrap(),
            ..CollectionParams::empty()
        },
        optimizer_config: opt,
        wal_config: Default::default(),
        hnsw_config: Default::default(),
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
        created_at: None,
        dli_experimental: Some(crate::config::DliExperimentalConfig {
            materialize_topk: true,
            quantized_shortlist: None,
            routing_policy: Default::default(),
            maintenance_policy: Default::default(),
            buffer_capacity,
            arity: 2,
            // Exhaustive learned routing: isolates visibility from ANN recall.
            candidate_budget: 1_000_000,
            model: LmiConfig {
                n_buckets: 4,
                sample_size: 64,
                hidden_dim: 8,
                epochs: 1,
                batch_size: 16,
                routing_batch_size: 16,
                kmeans_iterations: 2,
                nprobe: 4,
                seed: 11,
            },
        }),
    }
}

/// One committed version of a point: `axis = None` is a deletion. The version
/// may become visible as early as its op start and remains possibly visible
/// until the superseding op ends.
#[derive(Clone, Debug)]
struct Version {
    axis: Option<usize>,
    from: Instant,
    until: Option<Instant>,
}

/// Deterministic xorshift for reproducible op logs and query mixes.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return f64::NAN;
    }
    v[((v.len() - 1) as f64 * p).round() as usize]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
#[ignore = "Tier A concurrent DLI workload; set DLI_WORKLOAD_OUTPUT (optional DLI_WORKLOAD_READERS, DLI_WORKLOAD_OPS)"]
async fn dli_concurrent_readers_and_writer_preserve_visibility() {
    let _scope = common::ambient::test_guard();
    let output = std::env::var_os("DLI_WORKLOAD_OUTPUT").map(PathBuf::from);
    let readers: usize = std::env::var("DLI_WORKLOAD_READERS").map_or(4, |v| v.parse().unwrap());
    let ops: usize = std::env::var("DLI_WORKLOAD_OPS").map_or(1200, |v| v.parse().unwrap());
    let buffer_capacity = 96;
    let root = Builder::new().prefix("dli_workload_").tempdir().unwrap();
    let path = root.path().join("collection");
    let snapshots = root.path().join("snapshots");
    fs_err::create_dir(&path).unwrap();
    fs_err::create_dir(&snapshots).unwrap();
    let c = Arc::new(
        Collection::new(
            "dli_workload".into(),
            1,
            &path,
            &snapshots,
            &config(buffer_capacity),
            Arc::new(SharedStorageConfig::default()),
            CollectionShardDistribution {
                shards: AHashMap::from([(0, HashSet::from([1]))]),
            },
            None,
            ChannelService::default(),
            dummy_on_replica_failure(),
            dummy_request_shard_transfer(),
            dummy_abort_shard_transfer(),
            None,
            None,
            ResourceBudget::default(),
            None,
        )
        .await
        .unwrap(),
    );

    // Seed enough points to train the first learned level.
    let mut history: BTreeMap<u64, Vec<Version>> = BTreeMap::new();
    let mut rng = Rng(0x5eed_2026_1117);
    let mut next_id = 1u64;
    for chunk in 0..2 {
        let seed_start = Instant::now();
        let points: Vec<_> = (0..buffer_capacity)
            .map(|i| point(next_id + i as u64, (rng.below(DIM as u64)) as usize))
            .collect();
        for p in &points {
            let segment::types::PointIdType::NumId(id) = p.id else { unreachable!() };
            let VectorStructPersisted::Single(v) = &p.vector else { unreachable!() };
            let axis = v.iter().position(|x| *x == 1.0).unwrap();
            history.entry(id).or_default().push(Version { axis: Some(axis), from: seed_start, until: None });
        }
        next_id += buffer_capacity as u64;
        apply(&c, PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(points))).await;
        println!("WORKLOAD_SEED chunk={chunk}");
    }
    // Wait for the first publication so concurrent reads use learned routing.
    let replica = shard(&c).await;
    tokio::time::timeout(Duration::from_secs(120), async {
        while replica.local_dli_counts_for_test().await.is_none_or(|(p, _, _)| p == 0) {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("initial publication");
    let (published_before, learned_before, fallback_before) =
        replica.local_dli_counts_for_test().await.unwrap();

    // Reader tasks: query random axes until the writer finishes.
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut reader_tasks = Vec::new();
    for r in 0..readers {
        let (c, done) = (c.clone(), done.clone());
        reader_tasks.push(tokio::spawn(async move {
            let mut rng = Rng(0xabcd_0000 + r as u64 * 7919);
            let mut reads = Vec::new();
            while !done.load(std::sync::atomic::Ordering::Acquire) {
                let axis = rng.below(DIM as u64) as usize;
                let start = Instant::now();
                let hits = axis_query(&c, axis, 10).await;
                let end = Instant::now();
                reads.push((axis, start, end, hits));
            }
            reads
        }));
    }

    // Writer: deterministic mix of inserts, re-axis upserts and deletes.
    let mut write_latencies = Vec::new();
    let mut live: Vec<u64> = history.keys().copied().collect();
    let mut deleted_ids: Vec<u64> = Vec::new();
    for _ in 0..ops {
        let kind = rng.below(10);
        let start = Instant::now();
        let (id, axis) = if kind < 5 || live.len() < 32 {
            let id = next_id;
            next_id += 1;
            let axis = rng.below(DIM as u64) as usize;
            apply(&c, PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(vec![point(id, axis)]))).await;
            live.push(id);
            (id, Some(axis))
        } else if kind < 8 {
            let id = live[rng.below(live.len() as u64) as usize];
            let axis = rng.below(DIM as u64) as usize;
            apply(&c, PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(vec![point(id, axis)]))).await;
            (id, Some(axis))
        } else {
            let idx = rng.below(live.len() as u64) as usize;
            let id = live.swap_remove(idx);
            apply(&c, PointOperations::DeletePoints { ids: vec![id.into()] }).await;
            deleted_ids.push(id);
            (id, None)
        };
        let end = Instant::now();
        write_latencies.push((end - start).as_secs_f64() * 1e3);
        let versions = history.entry(id).or_default();
        if let Some(prev) = versions.last_mut() {
            prev.until = Some(end);
        }
        versions.push(Version { axis, from: start, until: None });
    }
    done.store(true, std::sync::atomic::Ordering::Release);
    let mut reads = Vec::new();
    for t in reader_tasks {
        reads.extend(t.await.unwrap());
    }

    // Interval-linearizability check for every concurrent hit.
    let mut violations = Vec::new();
    let mut read_latencies = Vec::new();
    for (axis, start, end, hits) in &reads {
        read_latencies.push((*end - *start).as_secs_f64() * 1e3);
        let ids: HashSet<u64> = hits.iter().map(|h| h.0).collect();
        assert_eq!(ids.len(), hits.len(), "duplicate external id in result");
        for (id, score) in hits {
            let matches_axis = *score > 0.5;
            let justified = history.get(id).is_some_and(|vs| {
                vs.iter().any(|v| {
                    v.axis.is_some_and(|a| (a == *axis) == matches_axis)
                        && v.from <= *end
                        && v.until.is_none_or(|u| u >= *start)
                })
            });
            if !justified {
                violations.push(format!("id={id} axis={axis} score={score}"));
            }
        }
    }

    // Quiescent exact state check (and after graceful reopen).
    let expected = |axis: usize| -> HashSet<u64> {
        history
            .iter()
            .filter(|(_, vs)| vs.last().unwrap().axis == Some(axis))
            .map(|(id, _)| *id)
            .collect()
    };
    let check_state = |label: &'static str, c: Arc<Collection>| {
        let expected = &expected;
        async move {
            for axis in 0..DIM {
                let hits = axis_query(&c, axis, 100_000).await;
                let ones: HashSet<u64> = hits.iter().filter(|h| h.1 > 0.5).map(|h| h.0).collect();
                assert_eq!(ones, expected(axis), "{label}: axis {axis} live set");
                let all: HashSet<u64> = hits.iter().map(|h| h.0).collect();
                assert_eq!(all.len(), hits.len(), "{label}: duplicate ids");
            }
        }
    };
    check_state("quiescent", c.clone()).await;
    let (published_after, learned_after, fallback_after) =
        replica.local_dli_counts_for_test().await.unwrap();
    let profile = replica.local_dli_named_profile_for_test().await.unwrap();
    let resurrected: Vec<u64> = deleted_ids
        .iter()
        .filter(|id| history[id].last().unwrap().axis.is_none())
        .copied()
        .collect();
    drop(replica);
    let c = Arc::into_inner(c).expect("all readers joined");
    c.stop_gracefully().await;
    drop(c);
    let reopened = Arc::new(
        Collection::load(
            "dli_workload".into(),
            1,
            &path,
            &snapshots,
            Arc::new(SharedStorageConfig::default()),
            ChannelService::default(),
            dummy_on_replica_failure(),
            dummy_request_shard_transfer(),
            dummy_abort_shard_transfer(),
            None,
            None,
            ResourceBudget::default(),
            None,
        )
        .await,
    );
    reopened
        .set_shard_replica_state(0, 1, ReplicaState::Active, None)
        .await
        .unwrap();
    check_state("reopened", reopened.clone()).await;
    let reopened_counts = shard(&reopened).await.local_dli_counts_for_test().await;

    let mut wl = write_latencies.clone();
    let mut rl = read_latencies.clone();
    let summary = serde_json::json!({
        "readers": readers, "ops": ops, "buffer_capacity": buffer_capacity,
        "reads": reads.len(), "violations": violations.len(), "violation_examples": violations.iter().take(5).collect::<Vec<_>>(),
        "deleted_ids": deleted_ids.len(), "resurrection_checked_ids": resurrected.len(),
        "read_ms": {"p50": percentile(&mut rl, 0.5), "p95": percentile(&mut rl, 0.95), "p99": percentile(&mut rl, 0.99)},
        "write_ms": {"p50": percentile(&mut wl, 0.5), "p95": percentile(&mut wl, 0.95), "p99": percentile(&mut wl, 0.99)},
        "publications_during_workload": published_after - published_before,
        "learned_queries": learned_after - learned_before,
        "exact_fallbacks": fallback_after - fallback_before,
        "fallback_reasons": profile["fallback_reasons"],
        "read_guard_nanos": profile["read_guard_nanos"],
        "reopened_counts": reopened_counts,
    });
    println!("DLI_WORKLOAD_SUMMARY {summary}");
    if let Some(output) = output {
        fs_err::create_dir_all(&output).unwrap();
        let file = output.join(format!("summary-readers{readers}-ops{ops}.json"));
        assert!(!file.exists(), "refusing to overwrite");
        fs_err::write(file, serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
    }
    assert!(violations.is_empty(), "visibility violations: {violations:?}");
    assert!(published_after > published_before, "workload must drive publications");
}
