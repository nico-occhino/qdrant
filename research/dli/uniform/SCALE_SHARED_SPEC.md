# Shared protocol for Q10 (memory budget) and Q11 (distributed scale)

**Status: DRAFT for cross-engine agreement.** No data has been collected under this spec. Each engine freezes its own
preregistration only after both sides accept this text, or an amended version pinned by commit and SHA-256. Every
threshold below is a proposal until that freeze. Once frozen, no threshold moves after data is seen.

Sources: the PostgreSQL proposal (Codex, 2026-11-27, `docs/research/STATUS.md` at `5f86dda`) and the Qdrant proposal
(this branch). Where they differ, §5 records the choice and the reason.

## 0. Shared conventions (unchanged from Q1–Q9)
- **Work.** Cross-engine comparisons use vectors actually scored.
- **The shared metrics.** W@0.90 is the first measured crossing, interpolated and never extrapolated; `upper_bound`
  and `not_attained` are explicit. Mean Recall@30 is also reported at the nominal 50K.
- **Uncertainty.** Every estimand uses a paired query bootstrap with 10,000 resamples.
- **Absolute latency is never compared across engines.** Each engine reports its own latency, and only within-engine
  ratios are put side by side (§1.6).
- **Gold.** Exact cosine over each engine's own stored representation (Float16 rows). Ties at the rank-30 boundary
  are classified, as in Q7.

## P0. Infrastructure preflight (both engines; not data)
One short job per engine on an ordinary compute node, labelled `preflight` and never analysed as results. It must
establish:
1. **Limit enforcement.** The PBS `mem=` request becomes a cgroup `memory.max` the job can read. Page cache from
   files the job reads is charged to that cgroup and reclaimed under pressure, and it does not cause OOM. Test this by
   reading a file larger than the limit.
2. **Readable counters.** From inside the job, these can be read: `memory.current`, `memory.peak` (or sampled at
   ≥ 1 Hz), `memory.stat` (anon, file, file_mapped), `memory.pressure` (PSI), `io.stat` (rbytes, wbytes), and
   per-process major faults.
3. **Per-file cache eviction without root.** `posix_fadvise(POSIX_FADV_DONTNEED)` on the engine's own files, with
   `memory.stat` file bytes dropping as a result.
4. **Q11 only.** In a multi-node job (`select=2`), the nodes reach each other on the engine's ports, and per-interface
   byte counters can be read.

If P0.1 fails, Q10 is redesigned: a design change goes back to the author.

## Q10. Memory budget

### 1.1 Question
How much memory does each method need to sustain matched recall at an acceptable tail latency? How does serving
degrade as the memory limit tightens below the method's footprint?

### 1.2 Data
- **The plan.** Phase 3 Insert-only, `I1-insert_only-20261121`, already verified on both sides. Its final snapshot is
  1M live vectors, built by replaying the plan's batches.
- **Queries and gold.** The plan's test panels, with exact gold per §0.

### 1.3 Arms
- **The first wave:** E1, IVF and HNSW at each engine's default settings.
  - **Qdrant:** E1 is Sommerová DLI; IVF is the final-benchmark IVF arm; HNSW is m = 16, ef_construct = 100.
  - **PostgreSQL:** E1, IVFFlat, and pgvector HNSW at its defaults.
- **DLI-best** joins only after its validation passes on that engine.

### 1.4 Storage configuration (declared per engine; fixed across limits)
- **Qdrant.** Vectors on disk (mmap), HNSW graph on disk, payload index on disk. Without this the vectors are
  anonymous memory and a limit below the footprint is an OOM, not a degradation. LMI postings and routers are
  deserialized into memory; P0.2 reports their anonymous share.
- **PostgreSQL.** `shared_buffers`, `work_mem`, `maintenance_work_mem` and `effective_cache_size` are set
  explicitly, to the same absolute values at every limit, and recorded.

### 1.5 Phases and limits
Each (arm, limit) is one job. One arm runs at a time on its node, with fixed CPUs (8), scratch SSD, index settings
and query concurrency (1).
- **Phase B (build and maintenance).** Replay the plan to the final snapshot, then flush. Report wall time, peak
  memory, anon/file peak, PSI and bytes read and written.
- **Phase S (serving).** Serve on the built index.
- **The unconstrained calibration U.** One run at 96 GiB per (engine, arm). It fixes:
  - **F.** The serving footprint, as resident file bytes plus anon after one warm pass.
  - **The knob k\*.** The smallest knob whose mean Recall@30 on the validation panel is ≥ 0.90, frozen before any
    limited run.
- **Limits, two grids, both reported:**
  - **Absolute grid:** 96, 64 and 32 GiB (PostgreSQL's proposal). It is directly comparable across engines.
  - **Relative grid:** max(4 GiB, A + r·(F − A)) for r ∈ {0.5, 0.25}, where A is the anonymous floor measured in U.
    This grid is what creates pressure for an engine whose 1M footprint is far below 32 GiB. For Qdrant at 1M, F is
    expected to be a few GiB, so the absolute grid alone would show no effect.
- **Cache states.**
  - **Warm (primary).** One discarded pass of the panel, then 5 measured repetitions.
  - **Cold (descriptive).** Evict per P0.3 before each repetition.

### 1.6 Metrics
For every (arm, limit, phase) we record:
- **Search:** Recall@30 and vectors scored at k\*, plus the full knob curve in U only;
- **Latency:** p50, p95 and p99 per query, single-threaded, as the median over repetitions;
- **Memory:** `memory.current` peak, anon and file, and PSI some/full;
- **I/O:** rbytes and wbytes, and major faults;
- **Build:** time;
- **OOM:** any OOM event, which is reported as an outcome, not retried.

### 1.7 Gates
- **Invariance (known answer).** At the same knob, the returned ids and scores are identical across limits, ties
  aside. Recall and work therefore cannot change with the limit; a difference is a defect, not a result.
- **Enforcement.** The measured peak never exceeds the limit, and `memory.max` equals the requested limit.
- **Gold.** The exact search on each limited index reproduces the gold.

### 1.8 Estimands
- **Degradation.** D(L) = p95(L) / p95(U) at k\*, from a paired bootstrap over queries. The same is reported for
  p99.
- **The minimum sustaining limit:** the smallest tested L with D(L) ≤ 1.5 and no OOM. The 1.5 is a proposal, to be
  frozen.
- **Cross-engine.** F, A, the minimum sustaining limit in absolute bytes and as a fraction of F, and D(L) per grid
  point. Absolute latency is not compared.

### 1.9 Engine-specific companions (descriptive; not in the primary estimand)
- **PostgreSQL HNSW build calibration.** At 1M and one larger prefix, with two session-level `maintenance_work_mem`
  values. Report the tuple count at which the graph stops fitting, and the build rate before and after. This tests
  the 42.90 h projection for 10.12M.
- **Qdrant twin.** HNSW build time and peak memory at the same prefixes, with 8 indexing threads. Qdrant has no
  maintenance-memory cliff, so this is descriptive.
- **Learned-router training memory (Qdrant M2; PostgreSQL to say whether an equivalent exists).** Sweep the router's
  training-sample cap, `QDRANT_LMI_SAMPLE_BUDGET_BYTES` ∈ {2 GB, 500 MB, 125 MB, 31 MB}, for S-LMI and E1. Report
  W@0.90, build time and training cost per level.

## Q11. Distributed scale

### 2.1 Question
Does a single-node advantage survive splitting the index into shards and merging their results? The costs of
splitting are separated from the costs of the network.

### 2.2 Data and partition
- **The snapshot.** The same I1 final 1M snapshot, served static. There are no writes, no replication and no
  resharding.
- **The shard function.** `shard(id) = splitmix64(id) mod 4`, applied to the plan's global point id, where
  `splitmix64(x)`:
  ```
  z = x + 0x9E3779B97F4A7C15;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
  return z ^ (z >> 31);
  ```
  Arithmetic is wrapping u64.
- **The shard-membership gate.** Each engine publishes the SHA-256 of each shard's sorted id list (u64 LE). The two
  engines must agree.
- **Building each shard.** Each shard is built by replaying, in plan order, only its own ids. Each shard therefore
  has its own learned hierarchy, buffer and routers. The trace that results is reported per shard.
- **Qdrant mapping.** `sharding_method: custom` with shard keys s0–s3, and a search that names no shard key (all
  shards).
- **PostgreSQL mapping.** Four instances with the DLI extension, behind an explicit coordinator that merges the
  top-30.

### 2.3 Configurations
- **D0, the reference.** One shard, one node: the Q10 U index.
- **D1, splitting only.** Four shards on one node.
  - **Knob.** The same per-shard knob on every shard, swept.
  - **Metrics.** Merged Recall@30, total work Σ over shards, the maximum per-shard work, and W@0.90 on merged
    recall.
- **D2, placement.** Four shards across 1, 2 and 4 nodes, with the same total CPUs (8) and the same total memory.
  - **Knob.** The per-shard knob fixed at D1's crossing point.
  - **Metrics.** End-to-end p50, p95 and p99 at the client, network bytes, the time of the slowest shard, and the
    coordinator's merge time.

### 2.4 Gates
- **Exact merge (known answer).** Shard-local exact top-30 lists, merged, equal the global exact top-30, ties
  aside.
- **Work accounting.** On D1, the merged-search work counter equals the sum of per-shard isolated runs, query by
  query.
- **Membership.** Per-shard id-set hashes match the other engine (§2.2).

### 2.5 Estimands
- **The cost of splitting.** W@0.90(D1) / W@0.90(D0) per arm, and the recall at 50K total work.
- **The cost of placement.** p95 at 2 and at 4 nodes, each relative to 1 node, within the engine.
- **Cross-engine.** Work ratios and recall only.

## 5. Decisions where the proposals differed
| Topic | PostgreSQL proposal | Qdrant proposal | Draft decision |
|---|---|---|---|
| Limit grid | 32 / 64 / 96 GiB | below the footprint | both; absolute is comparable, relative creates pressure (§1.5) |
| Multi-node | 1 / 2 / 4 nodes | in-process shards only | both: D1 counts work, D2 adds placement and network |
| Corpus-size scale | — | 10M static | already covered by Q1 (Qdrant) and the 10.12M E1 run (PostgreSQL); not repeated |
| Bucket-contiguous vector storage | — | possible original idea | out of scope here; Qdrant LMI postings are offsets into the shared vector store (`postings.rs`) |
| Partition | unspecified | hash | splitmix64 mod 4, cross-checked by hash |

## 6. Order
1. P0 on both engines.
2. Both sides accept this spec.
3. Each engine freezes its Q10 preregistration, then runs U, the limited runs and the companions.
4. Q11 freezes after Q10's U runs, because D0 reuses them.
