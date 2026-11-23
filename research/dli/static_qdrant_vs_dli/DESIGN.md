# Design: does a temporal learned hierarchy beat Qdrant's ordinary static-LMI lifecycle?

**Question.** Under dynamic workloads, does Sommerová's DLI (E1) give a measurable advantage over Qdrant's existing
segment lifecycle, which already buffers inserts in appendable segments and periodically rebuilds static LMI indexes?

The study compares two lifecycles on the same logical update stream, through real collection operations. It does not
compare two independently built final indexes. Negative results are kept.

## 1. Arms

- **D (E1).**
  - The existing research DLI: b = 5,000, a = 10, aⁱ buckets.
  - Bentley–Saxe compaction, router reuse on occupied targets, fresh training on empty ones.
  - Routing `mdn_code`, built with 16 CPUs (`QDRANT_DLI_BUILD_CPUS`).
  - Generic Qdrant optimizers are disabled, as in every earlier DLI study.
  - E1C, E1V, E1G and E1L are excluded.
- **S (static).**
  - Qdrant's ordinary merge, indexing, vacuum and config-mismatch optimizers, with one `LmiTrained` model config. The
    model rule is E1's (hidden 128, 10 epochs, 25 k-means iterations, sample max(B, min(n, max(100K, 40·B), 1M))).
  - One optimization job at a time, with every job CPU available to the build. Nothing is forced or triggered.
  - Two pre-specified configurations, both keeping a nominal bucket size of about 1,000 at 1M points. That is the
    static granularity that was best at 10M (`emma_path/MULTILEVEL_DIAGNOSIS.md`):
    - **S1**: `default_segment_number = 1`, 1,000 buckets;
    - **S4**: `default_segment_number = 4`, 250 buckets.
  - Both use `indexing_threshold = 20,000 KB`, the harness default.
  - Which S configuration is primary is chosen **on validation queries only** (§5). S as a family is not tuned on test.
    The pilot chose **S\* = S1**.
- **SG (granularity-matched static).** Added by Amendment A4 of the preregistration.
  - S1's topology, with B = round(L_max / 5,000) buckets: 200 for insert-only, 183 for shift, 20 for constant_r and
    constant_o at 1M.
  - A full segment therefore has D's bucket size of about b points.
  - SG isolates the lifecycle from partition granularity.

## 2. Workloads (plans from `emma_path/phase3_plan.py`, one plan per workload shared by both arms)

| ID | Scenario | Batches | Queries (test / validation) |
|---|---|---|---|
| W1 | insert-only: initial 10% of N, then 20 insert batches | 1 + 20 | test 9000–9999 / val 900–931 + calib 1000–2999 |
| W2 | shift: 10 spherical-k-means clusters in nearest-chain order; 1% holdout per cluster; last cluster withheld (OOD) | 20 | mq, ids, ood (1,000) / their `_val` (32) |
| W3 | constant_r: 1:1 insert:delete, seeded random deletes from the live set | 1 + 20 | test / val |
| W4 | constant_o: 1:1, oldest-first deletes | 1 + 20 | test / val |

**Ledger.** After every batch the driver checks the collection's live-ID count and SHA-256 against the plan, and fails
on any mismatch. Gold is the exact top-30 over the live set: float64 cosine of the stored Float16 rows against the
float32 queries, ties broken by the smaller ID. It differs from the official SISAP gold, which comes from the float32
originals.

## 3. Evaluation

After each settled batch, the shard is flushed and copied, and the copy is replayed:
- **D** with `run_frontier` over a global budget grid;
- **S** with `run_static_frontier` over an nprobe grid. That replay mirrors native `LmiIndex::search`: the same
  nprobe on every learned segment, doubling until filled, Plain segments scanned fully.

Both record recall and **actual vectors scored**, so they are compared on work, not on nominal knobs. Both also
record dead postings, and S records widening events.

## 4. Primary metric

For each batch t, W_X(t) is the work to reach mean test R@30 = 0.90, by linear interpolation between bracketing grid
points, never extrapolating; "not attained" otherwise. Per workload the summary is
R = mean_t W_D(t) / mean_t W_S(t) over batches 1–20 where both arms attain, with the attainment share reported. Batch
0, the initial load, is descriptive: neither arm has a learned index yet.

## 5. Selection without leakage

- **The pilot** (seed 20261301) replays validation panels only (val, calib, `*_val`) and never the test panels. It
  chooses the S configuration (S1 or S4) by the lower mean validation W(0.90) on W1, using the `calib` panel.
- **The primary run** uses a different seed (20261121, as in Phase 3) and the test panels.

## 6. Concurrent-service phase (implemented 2026-11-23; `dli_service_workload`)

A separate run per arm, on the same plan and arm configuration (`p3_setup`), through the live `Collection::search`
path:

- **Writer.** It applies each batch (inserts, then deletes) at a fixed rate (`DLI_SVC_WRITE_RATE` points/s, in chunks
  of `DLI_SVC_CHUNK`). There is **no synchronous drain**: DLI maintenance and Qdrant's optimizers run in the
  background, as in service.
- **Readers.** Open-loop arrivals at `DLI_SVC_QPS`, at most `DLI_SVC_INFLIGHT` in flight. Arrivals beyond the cap are
  counted as dropped. Each request has a timeout of `DLI_SVC_TIMEOUT_MS`. They run from the writer's start until
  maintenance has settled.
- **Recorded per batch:**
  - achieved QPS, latency percentiles, timeouts and errors;
  - maintenance busy seconds and the DLI job backlog;
  - DLI learned against exact-fallback request counts;
  - process CPU, RSS, I/O bytes and cgroup memory (`/proc`);
  - a settled sequential pass, with recall against the batch's gold.
- **Correctness.** A request must never return a point whose deletion completed before the request started. Any
  violation fails the run. The ledger is checked after every batch.
- **Overhead.** No request-path tracing (`DLI_QUERY_EVENTS` must be unset). The counters are read once per batch.
- **Operating points** are locked on validation from pilot2 (preregistration §5) and passed as
  `SVC_D_BUDGET` / `SVC_S1_NPROBE` / `SVC_SG_NPROBE`.
- **Pilot load:** 20 QPS, 8 in flight, a 10 s timeout, 2,000 points/s in chunks of 1,000 (`MODE=service` defaults).
- **Summary:** `summarize_service.py`.

**Local validation** (synthetic 20K, constant_r, 4 batches, 40 QPS):
- both arms issued about 1,550 requests with 0 timeouts, 0 errors, 0 drops and 0 deleted-point violations;
- the ledger matched after every batch;
- D's exact-fallback counts rose during maintenance windows. The summariser reports them.

Still missing: a router, selection and scoring time split for S in the replay.

## 7. Write amplification

A_write = points written by optimizer outputs ÷ logical points inserted. It is computed from
`QDRANT_OPTIMIZATION_EVENTS` (`destination_points`, which both arms emit through the native optimizer), excluding the
initial bulk load from the numerator and the denominator. Physical bytes are measured separately (§6.2).
Model-only changes do not exist in either arm, because every rebuild rewrites points.
