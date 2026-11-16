# Completed size-tiered 10M run: diagnostic

**Executed source:** `3393b4a3c205c5cf3f7fc5f957f9a16e3f737599`. **Result:** 1 test passed, 6,785.09 s. This analysis launched no benchmark or build and changed no index code. Later uncommitted profiling edits in the worktree were excluded from the evidence.

The principal finding is a **storage-mode control failure**: the two active learned segments contain 5M points each, but one uses `InRamChunkedMmap` and the other `ChunkedMmap`. All six latency outliers touch the latter in the validated offline route reconstruction. Low recall has a separate explanation: nearly every query selects only one bucket from one half of the corpus, strongly favoring the higher-numbered logical slot.

## Evidence and counter reconciliation

Let **R** be `/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-25/size-tiered-10120191/` and **D** its sibling `size-tiered-diagnosis/`. Inputs are `R/{summary.json,native-events.jsonl,build-events.jsonl,query-events.jsonl,run.log}` and the retained `R/collection/`. Derived checks, hashes, statistics and reconstruction are in `D/{diagnostic.json,query_reconstruction.csv,native_publications.csv,active_members.json,provenance-check.json,comparisons.json,input-artifact-hashes.json}`. Reproduce the bounded, non-scoring analysis with `OPENBLAS_NUM_THREADS=2 /home/nicoo/miniconda3/envs/lmi-rust-inspect/bin/python D/analyze.py` after expanding D.

`local_dli_search_metrics_for_test` in `lib/collection/src/shards/replica_set/mod.rs` at the executed SHA defines this exact tuple:

| Index | Counter | Value | Unit/domain |
|---|---|---:|---|
| 0 | `learned_queries` | 100 | successful learned requests |
| 1 | `exact_fallbacks` | 0 | fallback requests |
| 2 | `selected_postings` | 1,398,301 | posting entries before offset deduplication; excludes Plain estimate |
| 3 | `unique_posting_offsets` | 1,398,301 | unique offsets within selected physical members; not global external-ID cardinality |
| 4 | `scoring_nanos` | 23,718,737,115 | **nanoseconds**, broad post-plan scoring phase |
| 5 | `resolution_nanos` | 6,326,334,430 | **nanoseconds**, candidate validation, membership/version lookups and final native aggregation |

The last two counters total **30.045071545 s**, explaining **98.8426%** of the **30.396878400 s** sum of per-query latencies. Scoring is 78.0302%, resolution 20.8124%; the **0.351806855 s residual** includes routing/planning, scheduling, capture/locks and request overhead. These counters are accumulated times, not billions of candidates or CPU cycles. Their per-request means are 237.187 ms and 63.263 ms, versus 303.969 ms total.

`route_pinned` starts its scoring timer before grouping selected buckets and stops after `process_search_result`: it includes posting extraction/sorting, vector scoring/top-k, and **materializing all candidate hits with native IDs/versions**. It does not isolate the dot-product kernel or vector I/O. Resolution then revisits selected identities across all three captured members and aggregates top-30. The residual cannot be assigned entirely to inference. Query JSON emission and oracle evaluation occur outside the per-query latency timer.

All 100 logged queries return 30 hits. Their successful vector evaluations sum exactly to 1,398,301; 21,897 scorer calls are **batch calls**, not individual-vector counts. Source counters increment around native `score_points` after visibility filtering. There is no discarded-candidate counter or per-query phase timing in this run.

## Tail latency and actual routing scope

The percentile implementation selects sorted index `ceil((n-1)*p)`: p50 is index50 and p95 index95. Six queries exceed one second and consume **69.607%** of total query time. The largest ten consume 72.451%. Pearson correlation between vector count and latency is only **0.0836**; work volume alone does not explain the tail.

| Query row (zero-based) | Latency ms | Scored vectors | Reconstructed physical selection |
|---|---:|---:|---|
| 36 | 5,977.697 | 21,341 | Cold-mode level5, bucket44 |
| 99 | 4,623.932 | 17,002 | Cold-mode level5, bucket104 |
| 18 | 4,442.826 | 15,742 | Cold-mode level5, bucket120 |
| 8 | 2,823.239 | 8,676 | Cold-mode level5, bucket485 |
| 50 | 1,837.703 | 7,426 | Cached level6 bucket431 (1,473) + Cold level5 bucket483 (5,953) |
| 12 | 1,452.954 | 4,934 | Cold-mode level5, bucket318 |
| 1, control | 51.520 | 8,976 | Cached level6 only |
| 86, control | 226.230 | 31,691 | Cached level6 only |

The 94 queries not touching level5 consume 9.239 s, median91.173 ms. The six touching it consume 21.158 s despite only75,121 evaluations (5.37% of work). Their distinct bucket IDs and occurrence through query99 argue against a single startup-only outlier; they do not prove a particular cache/fault mechanism.

**Reconstruction versus observation:** original per-query logs lack bucket IDs/model IDs. The offline analysis decodes only persisted router parameters/posting boundaries, reproduces ordered Float32 forward passes and the source planner, and matches **every query's vector count and scorer-call count (100/100 each)**. No corpus vectors were scored and reconstruction runtime is not used as benchmark latency. Selected identities below are therefore validated reconstructions, not newly discovered trace fields. Floating-point implementation differences remain a limitation of reconstruction.

Source traversal evaluates both routers on every successful query: **2 models/query, 200 forward calls total** (source-derived). They produce1,096 learned bucket predictions plus the Plain prediction. Reconstruction selects **one bucket/one scoring member for99 queries**, and two buckets/two members for query50. Level6 is selected95 times; level5 six times. The Plain buffer is never selected in reconstruction; no query's total evaluation count reaches its120,191 points.

Each model has548 buckets and5M points (mean9,124.1 points/bucket). Level6 bucket sizes: min2, median8,314.5, max61,073; level5: min1, median8,428, max35,449. Each has88 nonempty buckets smaller than2,000. The nominal budget is a **minimum whole-bucket occupancy target**, not a ceiling: actual average13,983.01 evaluations/query is6.992 times2,000.

Why Recall10=.330: two equally populated models occupy slots6 and5, receiving probability multipliers64 and32. These are uncalibrated scores; the slot alone gives one model a2x advantage. The insert-only native source lineage reconstructs corpus-ID membership and matches all40 destination populations. Combined with reconstructed selected models, it yields an **upper bound of .503 Recall10 (.4987 Recall30) even if each selected model were searched exhaustively**. Thus497/1000 gold top-10 IDs are outside the selected model scope; observed hits are330. The remaining173-hit gap is attributable jointly to bucket/candidate selection and final ranking/numerics, not demonstrably neural misclassification alone. This is a model-scope ceiling, not measured candidate recall. An isolated counterfactual is required to quantify the multiplier's causal contribution.

## Storage-mode finding and measurement limitations

| Active physical member | Logical role | Points | Persisted vector storage |
|---|---|---:|---|
| `3d7c0a2c-a3d8-44ec-88fa-a25fd831dcc0` | level6 | 5,000,000 | `InRamChunkedMmap` (Cached placement) |
| `da38dfc6-5b9b-4ef7-97fb-b42b91a9cfda` | level5 | 5,000,000 | `ChunkedMmap` (Cold placement) |
| `3adb7860-c387-4165-869f-000af5c43ade` | Plain | 120,191 | `InRamChunkedMmap` |

The collection config says `on_disk:true`, but physical `segment.json` files show mixed placement. At the executed SHA, `dli_maintenance` clones `sources.first().config()`; `DliOptimizationStrategy::create_segment_builder` changes the index to LMI without enforcing destination storage mode. `create_temp_segment` explicitly forces in-RAM chunked COW storage. **Input-source configuration can therefore propagate into learned destinations.** This is a concrete experimental-control defect, not a proven data-correctness failure. Cold/Cached placement, residency, page faults, ID materialization and OS scheduling have not been timed separately; “disk stalls caused exactly X seconds” would be unsupported.

Also, `auxiliary_index_bytes` recursively sums **files**, including retained generations. The reported67,112,774 B comprises45,408,516 B in the two active indexes plus21,704,258 B in inactive retained source `eed171bb-638a-4d30-b31f-c0cea38d97f7` (4.75M points). It is not resident index memory or a third active router. Retention is consistent with native recovery obligations; no leak is established. Peak `VmHWM` is28,437,204 KiB (27.12 GiB); no private-memory, fault, swap or per-stage peak counters identify its cause.

## All40 publications and build cost

All40 recorded attempts are `published`; no recorded native failures. Both source-live snapshots and destination counts sum to **105,000,000** points. One dense vector/point makes destination count a vector-rewrite proxy here; actual copy input/bytes are not independently counted. Output sequence is `250K,250K,500K,500K,...,5M,5M`. Maximum output5M; average job163.751 s, maximum329.598 s. Rewrite amplification is **10.3753x** against10,120,191 inserted points, or10.5x against the10M points drained into learned runs.

| Checkpoint | Publications | Learned populations after publication | Cumulative output |
|---|---:|---|---:|
| .25M | 1 | .25M | .25M |
| .50M | 2 | .25M + .25M | .50M |
| 1M | 4 | .50M + .50M | 1.50M |
| 3M within full run | 12 | 1.50M + 1.50M | 10.50M |
| 10M | 40 | 5M + 5M | 105M |
| 10,120,191 | unchanged | above +120,191 Plain | unchanged |

Historical populations follow source/destination lineage; historical logical slot numbers were not logged. Final descriptor generation41 explicitly lists slots5/6 plus Plain. `D/native_publications.csv` preserves all source/output UUIDs and populations.

| Wall-time counter/stage | Seconds | Interpretation |
|---|---:|---|
| Native job sum | 6,550.039 | same execution boundary as geometric |
| Validation/COW/proxy prefix | 3.040 | cumulative entry-to-protected elapsed |
| Build + first propagation | 6,355.453 | broad native build phase |
| Final propagation/publication | 191.546 | native finish phase; logical activation excluded |
| Ingestion elapsed | 6,750.779 | includes acknowledged writes/waits; not sum of phase CPU costs |
| Teacher + MLP/export | 1,943.513 | nested inside build phase |
| Clustering | 228.524 | nested inside preceding row |
| MLP initialization / epochs / export-validation | 6.859 / 1,707.166 / .746 | nested component wall timers |
| Corpus routing/postings | 1,699.089 | separate from training |
| Sample/enumeration/read / coverage / serialization | 8.733 / .578 / 4.135 | wall timers |

There are40 completed training/build events, each with32,768 samples,548 outputs and30 epochs;153,600 reported training steps are configuration-derived on successful completion. Each native job window contains exactly one matching completed build with physical=eligible=destination count. Native jobs do not overlap (minimum inter-job gap3.690 s); first-start to last-completion span6,743.341 s. **CPU time, queue time, copy-only duration, bytes and logical activation duration are absent.** Roughly2,699.404 s of the broad build phase remain outside its exported substages; do not label that residual “copy time.” Nested training timers must not be added twice. Training remains about29.7% of total job wall time, unlike the dominant fixed training fraction in small jobs.

## Comparison with completed artifacts

All work below is per query; wall times are shared native job sums. 300K used50K buffers and1,000 queries; 3M/10.120M used250K buffers and100 queries. All are single, unrepeated operating points with unequal recall.

| Population/policy | Output M | Job wall s | Recall10/30 | p50/p95 ms | Scored/query |
|---|---:|---:|---|---|---:|
| 300K geometric | .75 | 365.509 | .7625/.7253 | 5.872/8.067 | 2,489 |
| 300K size-tiered | .60 | 367.186 | .7556/.7164 | 6.706/9.092 | 2,235 |
| 3M geometric | 9.0 | 991.428 | .595/.5703 | 1324/4742 | 8,904 |
| 3M size-tiered | 10.5 | 1080.054 | .373/.3433 | 23.954/1301 | 4,431 |
| 10.120M geometric | 31.75 | 3540.055 | .402/.391 | 3116/8779 | 17,586 |
| 10.120M size-tiered | 105.0 | 6550.039 | .330/.3153 | 94.308/1838 | 13,983 |

Geometric's rewrite ordering reverses between the short300K study and larger streams; at full scale size-tiered produces3.307x its destination population and takes1.850x its observed job-wall sum. Both train40 models: geometric teacher/MLP time1,936.178 s is close to size-tiered1,943.513 s; geometric routing/postings536.877 s versus1,699.089 s explains part of the growing difference.

**Wall-time and latency differences are not clean scheduler-only causal effects:** geometric3M's sole learned segment is Cold; geometric10M has a Cold6M segment plus Cached3M/.75M/.25M members. Size-tiered3M and10M have one Cold and one Cached learned member. The backend drift invalidates a claim that router count alone explains policy latency. Active geometric10M auxiliary files total50,817,028 B, versus size-tiered's active45,408,516 B (recursive67MB includes retention). Geometric peak RSS21,271,168 KiB versus size-tiered28,437,204 KiB is observational. No completed10M static-LMI result was found; no static10M numbers are inferred from300K.

Raw comparisons: `continuation-24/{full-geometric,full-size-tiered}/`; `continuation-25/{geometric-3000000,size-tiered-3000000,geometric-10120191}/`. Normalized extracts are in `D/comparisons.json`.

## Provenance verification

The official2023 corpus is10,120,191×768 Float16; the run uses public query rows0–99, not a2024 query set. Corpus point IDs are row+1; query-log indices remain zero-based. Query/gold MD5s were independently rechecked as `257b9eb3f7f25776e0d33b22451b7b32` and `45b05e4d60b8a66088b378ae7e0d278f`. All100 used raw query/gold rows match the official HDF5 files. The recorded successful corpus MD5 is `c05e4b1d2b2a0c7663ac9767753e25e1`; this diagnosis sampled raw/source equality at corpus/batch/end boundaries rather than rehashing15GB.

Persisted config and source specify Float16/Cosine, and learned routing normalizes queries with native cosine preprocessing; corpus insertion uses native cosine preprocessing. This matches the intended normalized-cosine task, subject to Float32/Float16 rounding and tie differences from official gold. No evidence identifies oracle mismatch as the cause of .330 recall. DLI budget2000 is global; `nprobe:4` in model config is not used by its direct router/postings path. Historical static-LMI probing is therefore not an equal-work or equal-recall service comparison. This scaling run is insert-only, not the earlier128-upsert/delete workload.

## Three bottlenecks and smallest discriminating checks (proposed only)

1. **Segment storage-mode drift and access/materialization tail.** Audit destination backend against explicit collection policy, including source-order permutations in a tiny builder fixture. On the frozen collection, replay only a Cold query such as36 and a comparable Cached query such as78 twice, recording actual selected UUID, backend, thread CPU, page faults and phase times. This discriminates residency/access from computation without another ingestion run.
2. **Candidate expansion and visibility/materialization cost.** Split the existing broad timer into posting preparation, `score_candidates`/Plain scoring, `process_search_result`, batched ID/version lookup and aggregation. Trace counts for those same few queries: about14K candidates become native hits to return30, and resolution already costs6.326 s. Do not truncate before version-safe resolution merely to improve timing.
3. **Budget/routing quality bottleneck.** On saved logits/postings, first compare slot-weighted selection with equal-population weighting or reserved per-model allocation at matched actual posting work. Record per-bucket/member candidate gold coverage for a few queries. This tests whether the2x slot multiplier and one-member scope, rather than within-model training alone, dominate misses; actual recall changes require a later frozen query test. No routing rule was changed here.

This diagnosis committed reports only. No benchmark/build, process monitoring, or algorithm modification was performed; existing uncommitted profiling work remains untouched.
