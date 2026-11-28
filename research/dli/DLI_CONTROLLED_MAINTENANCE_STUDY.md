# Native maintenance policy comparison — continuation 24

Source revision: `2e175df23dda5634d020197c7aa72e3781d9b45b` (build code). Starting revision: `06bce2a1d31bef3ed6b792230b2b1017fd20145d`. Raw artifacts: `<admin>/experiments/2026-11-12-dli-architecture/continuation-24/`. Previous first-run/post-update reports are preserved.

## Main finding

On six ordered 50K batches of canonical 300K SISAP23 LAION CLIP-768 data, geometric DLI rewrote fewer native destination points than native static LMI, **but rewrote more than the size-tiered DLI alternative**. Geometric scheduling itself therefore has no demonstrated maintenance advantage over that alternative in this workload. Native job wall sums for the two DLI policies were nearly equal. This is one run per policy, not statistical evidence of equal speed or a general result.

| Metric | S (geometric pair) | D-G | S (size-tiered pair) | D-N |
|---|---:|---:|---:|---:|
| Successful native jobs | 11 | 6 | 11 | 6 |
| Native destination points cumulatively produced | 1,300,000 | 750,000 | 1,300,000 | 600,000 |
| Rewrite amplification (output / 300K inserted) | 4.333 | 2.500 | 4.333 | 2.000 |
| Sum of native job wall seconds | 678.706 | 365.509 | 670.471 | 367.186 |
| Recall@10 | .7962 | .7625 | .7962 | .7556 |
| Recall@30 | .7628 | .7253 | .7628 | .7164 |
| Query p50 ms | 2.700 | 5.872 | 2.727 | 6.706 |
| Query p95 ms | 4.424 | 8.067 | 4.043 | 9.092 |
| Successful vector evaluations / 1000 queries | 3,087,797 | 2,489,314 | 3,087,797 | 2,234,807 |
| Native batch scorer calls | 48,747 | 39,405 | 48,747 | 35,907 |
| Auxiliary LMI files, bytes | 3,904,256 | 10,312,768 | 3,904,256 | 9,712,768 |
| Post-update learned/fallback queries | not separately attributed | 100 / 0 | not separately attributed | 100 / 0 |

D-G produces 42.31% fewer destination points than S; D-N produces 53.85% fewer. D-G produces **25% more** than D-N (denominator 600K). D-G job wall sum is about 0.46% lower than D-N, well within unmeasured run-order/system variability. D-G scores 11.39% more vectors than D-N and has higher recall; latency differences are not recall-matched. Compared with S, both DLI variants have lower recall, slower sequential query latency and larger auxiliary index files. Do not infer a search win from unequal scoring budgets.

## Policies and causal scope

S retains Qdrant's normal generic optimizer (one worker). D-G/D-N disable generic optimizer workers; their maintenance invokes native execute_optimization directly. This ownership difference is intentional: generic optimizer replacement cannot safely assign temporal logical membership. Native data/WAL/proxy/publication code is shared. This is a practical native-system comparison, not a pure scheduler-only comparison against S.

D-G uses existing Bentley–Saxe capacities b*a^i with b=50K,a=2 and buffer separate. D-N flushes independent runs while fewer than arity learned runs exist; thereafter it merges the buffer with the smallest learned run. It reuses the lowest compatible empty/selected logical slot. Slot capacities remain the existing geometric constraints. This alternative is bounded and selective, not a repeated full rebuild. D-N outputs 50K,50K,100K,100K,150K,150K vs D-G 50K,100K,150K,50K,100K,300K. Both policies use identical models, candidate validation, native optimizer/build and publication, global routing and descriptor recovery. Policy is stored in derived metadata and mismatches reject activation.

The difference in selected populations/rewrite counts between D-G and D-N is a direct consequence of scheduler choice in this deterministic stream. Query behavior also depends on resulting physical models and logical-slot priority multipliers; it is not an isolated routing ablation. No geometric advantage over D-N is supported by the measured rewriting or single-run time.

## Measurements and verification

`QDRANT_OPTIMIZATION_EVENTS=/absolute/path.jsonl` emits shared native events for all static/DLI optimizer attempts, including source UUID/live-point observations, destination UUID/points, native start timestamp, protection elapsed, build-plus-first-propagation, final-propagation-plus-publication, outcome and job-wall duration. Failed/unwound attempts are retained as such; no failures occurred in the benchmark jobs. Source counts are pre-protection observations, not exact deduplicated copy-input counts. Destination counts are native points; this fixture has one dense vector per live point, so they count vectors produced. A job can contain non-training work; native job count is not automatically model-training count.

`QDRANT_MEASURE_SCORING=1` enables relaxed atomic counters at actual successful/attempted native score_points batch boundaries, including Plain scan chunks and repeated evaluations. Counts are process-wide, with before/after deltas over sequential query phases. The isolated fixture initially counted7 rather than expected8, exposing omitted Plain scanning; after correction it counts exactly8 attempted/successful vectors (scorer-fixture-fixed.log). These counters measure the instrumented batch scorer paths, not arbitrary HNSW/quantization modes.

Native wall sums use identical boundaries for S/D-G/D-N, including protection/build/propagation/publication. They are **not CPU time**, training-only cost, or elapsed ingestion time. Event start-to-last-completion spans are in comparison.csv but include waits while the paired system ingests/optimizes; do not compare them as isolated time-to-quiescence. Job-wall sums may overlap; no sum is labeled elapsed time. Queuing time, model/clustering/posting splits, exact copy input, bytes rewritten, logical activation time and per-job temporary disk peaks are unmeasured.

All runs use optimized builds, seed42, Float16 cosine CLIP768, 548 buckets, sample32768, hidden512, epochs30, nominal DLI budget2000, same IDs and batches. Canonical hashes are unchanged from continuation-22/benchmark-300k/manifest.json. Queries: first1000 before updates and first100 after 128 same-vector upserts/128 distinct deletes. This stream preserves vector values; deleting <=128 IDs cannot exhaust official top1000 for top30, assuming official list validity. Float16 storage/numerical differences and ties can still affect oracle agreement. Vector-changing workload/fresh updated oracle is NOT RUN.

100K pilot passed with newly generated blocked normalized-cosine top1000 for the queried first1000 public query rows. It produced static200K vs DLI150K destination points, job sums187.139 vs119.982s, static993,796 vs DLI2,139,297 successful vector evaluations. Pilot gold manifest/hashes are in pilot-inputs/manifest.json; unused rows are zero and must never be queried.

Tests: instrumented DLI suite18 passed,0 failed,1 ignored; exact scorer fixture1 passed; pilot and both full300K test executions each1 passed. Initial compile failure while dependencies were changing and initial scorer-counter failure are preserved. No deleted-slot regression was found or fallback policy weakened.

## Artifacts and commands

Repository runner: `research/dli/controlled_study/run-study.sh`; analyzer: `research/dli/controlled_study/analyze.py`. Runner contains local paths/environment and executes pilot, full-size-tiered, full-geometric sequentially. The source hash is captured before runs. Each run directory contains environment.txt, run.log, native-events.jsonl, summary.json. Root comparison.csv/json and work_curves.csv are generated by analyzer; cumulative_destination_vectors.svg and cumulative_job_wall_s.svg plot completed-native-job axes, not checkpoint populations or CPU time. Reproduce analysis with the lmi-starterpack Python environment (matplotlib available).

## Unverified requirements

The full study still lacks isolated per-system RSS (summaries contain combined process VmHWM), randomized static-first order, replication, training/clustering/posting/CPU split, precise bytes and temporary storage, acknowledged write latency distributions (only six batch calls and individual mixed calls exist), frozen-topology global/per-level routing comparison, visibility-only timing ablation, candidate/recall budget sweeps and held-out recall matching, and vector-changing update ground truth. Consequently there are no recall-latency/scored-budget curves or statistical significance claims. Remaining work is experimental/instrumentation work, not a demonstrated hard blocker. Overall classification remains FUNCTIONAL EXPERIMENTAL DLI with a measured, bounded policy comparison.
