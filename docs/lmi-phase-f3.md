# Phase F.3: frozen-model consolidation

## Observed result

At validation target 0.90, centroid reaches test recall 0.9165 with 12575.5 candidates and p50 4.215 ms. Frozen MLPs reach recall 0.9097–0.9139, using 11315.5–11366.9 candidates and p50 3.778–3.850 ms. Candidate savings relative to centroid range from 9.61% to 10.02% (negative means more candidates). 1/3 models jointly pass the exploratory recall-margin and candidate-reduction gates.

At validation target 0.95, centroid reaches test recall 0.9494 with 17420.3 candidates and p50 5.783 ms. Frozen MLPs reach recall 0.9468–0.9509, using 16635.9–16680.6 candidates and p50 5.388–5.423 ms. Candidate savings relative to centroid range from 4.25% to 4.50% (negative means more candidates). 3/3 models jointly pass the exploratory recall-margin and candidate-reduction gates.

These are validation-selected operating points, not exactly equal-recall pairs. Read recall differences and paired intervals alongside speed/work differences. F.3 does not establish universal MLP superiority or a fresh HNSW comparison.

## Outcome and scope

F.3 evaluates the already-selected 4096/60 MLP models (seeds 42/43/44) without any training, teacher refit or HNSW rebuild. This is a new internal evaluation on a reduced corpus, not a rerun or replacement of Phase F/F.2 and not an external-dataset validation.

The original corpus had 99,780 vectors. We selected 20 warmup, 300 validation and 1,000 test queries from rows excluded from every F.2 training sample and the teacher sample. Removing queries and four additional exact normalized duplicates leaves 98,456 searchable vectors. These rows were previously observed as corpus during F/F.2; they are fresh as query evaluations, not historically untouched data. No near duplicate to training/old queries at cosine >=0.9999 was found. Semantic near duplicates are not ruled out.

## Fixed validation policy

For targets 0.90 and 0.95, choose the smallest integer nprobe reaching that mean recall on the 300 validation queries. Save the choices before evaluating test rankings. All p=1..64 are evaluated by exact-neighbor membership for recall/candidate curves. Native scoring and timing run only at the two fixed points per method, plus full-probe correctness checks. No interpolation, test-time selection or architecture/hyperparameter tuning is used.

| Method | Target | nprobe | Validation recall | Test recall | Mean candidates | Total p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| centroid | 0.90 | 5 | 0.9223 | 0.9165 | 12575.5 | 4.215 | 5.464 | 6.805 |
| affine | 0.90 | 5 | 0.9223 | 0.9165 | 12575.5 | 4.203 | 5.483 | 6.700 |
| n4096_e60_s42 | 0.90 | 5 | 0.9073 | 0.9097 | 11366.9 | 3.850 | 5.468 | 6.853 |
| n4096_e60_s43 | 0.90 | 5 | 0.9117 | 0.9118 | 11320.4 | 3.778 | 5.489 | 6.942 |
| n4096_e60_s44 | 0.90 | 5 | 0.9113 | 0.9139 | 11315.5 | 3.778 | 5.496 | 7.142 |
| centroid | 0.95 | 7 | 0.9520 | 0.9494 | 17420.3 | 5.783 | 7.408 | 9.083 |
| affine | 0.95 | 7 | 0.9520 | 0.9494 | 17420.3 | 5.759 | 7.486 | 9.138 |
| n4096_e60_s42 | 0.95 | 8 | 0.9553 | 0.9468 | 16673.6 | 5.423 | 7.777 | 9.292 |
| n4096_e60_s43 | 0.95 | 8 | 0.9563 | 0.9487 | 16635.9 | 5.420 | 7.847 | 9.590 |
| n4096_e60_s44 | 0.95 | 8 | 0.9550 | 0.9509 | 16680.6 | 5.388 | 7.857 | 9.827 |

## Paired uncertainty and decision gates

Intervals use 2,000 paired query resamples, preserving both trials within each query. Recall non-inferiority is exploratory with a predeclared 0.01 margin; interval containment is not a formal multiplicity-adjusted test. Training/teacher/sample uncertainty is not covered. Lower candidates at lower recall must not be reported as unconditional superiority.

| Model | Target | Recall delta [95% CI] | Candidate saving | Candidate delta [95% CI] | Recall margin met? |
|---|---:|---|---:|---|---|
| n4096_e60_s42 | 0.90 | -0.0068 [-0.0144, +0.0010] | 9.61% | -1208.5 [-1341.5, -1078.3] | False |
| n4096_e60_s43 | 0.90 | -0.0047 [-0.0117, +0.0029] | 9.98% | -1255.0 [-1387.9, -1125.4] | False |
| n4096_e60_s44 | 0.90 | -0.0026 [-0.0094, +0.0049] | 10.02% | -1259.9 [-1387.6, -1131.3] | True |
| n4096_e60_s42 | 0.95 | -0.0026 [-0.0087, +0.0035] | 4.29% | -746.7 [-932.4, -564.4] | True |
| n4096_e60_s43 | 0.95 | -0.0007 [-0.0065, +0.0057] | 4.50% | -784.4 [-978.2, -601.4] | True |
| n4096_e60_s44 | 0.95 | +0.0015 [-0.0043, +0.0075] | 4.25% | -739.7 [-926.0, -551.5] | True |

## Timing methodology

Same process, segment, Qdrant vector storage, native BatchFilteredSearcher, candidate preparation and deletion checks for all methods. CPU0 affinity; OMP/OpenBLAS/Rayon one thread. Method order rotates within each query, target and trial. Twenty warmup queries precede each trial/target block. The common harness computes a full 64-bucket ranking for every router, then takes its prefix. This is CPU in-process warm-cache latency, not HTTP latency, cold-cache performance, a production LmiIndex call or HNSW timing. Two trials are intentionally bounded and do not establish long-run system variance. Components and p50/p95/p99 appear in paired_summary.csv/json. Percentiles pool query/trial observations; paired mean-latency intervals are separate.

No current HNSW comparison is made on the changed corpus. Its historical F measurement is not copied into F.3 tables. Candidate visits for HNSW remain unavailable. No build-time improvement can be inferred: the models are loaded, and source partitions are pruned rather than rebuilt.

## Verification and preservation

- All 432 protected F/F.2 source files retained their SHA-256 hashes.
- 295,368 native MLP assignment checks reproduce saved postings on retained vectors.
- 1,400 full-probe native scoring checks equal Plain exact IDs/scores.
- Zero centroid/affine query-order mismatches; shared postings guarantee identical deterministic retrieval.
- 416,000 dense per-query operating-point rows and 20,000 per-query/per-trial timing rows are preserved.
- Native recall and candidate counts match dense membership predictions at every timed point; trials have identical retrieval results.
- Removed query rows never appear in exact or approximate results. Training is never invoked.

## Remaining limits and next step

Interpret F.3 as a controlled replication on fresh internal queries with frozen models and inherited partitions. The corpus is smaller and queries were previously corpus observations. F.2 model-selection history cannot be undone; an external query dataset, independent teacher/sample seeds and broader datasets remain necessary for generalization claims. The denser curve exposes high-recall tradeoffs but no 0.99 policy is selected for timing. No memory/build/reopen or lifecycle performance claims are added. Existing F.2 lifecycle results are historical evidence, not rerun here because production behavior is unchanged.

The next research increment, if justified, should compare retrieval-supervised affine routing and a simple query-prototype memory over fixed centroid postings. Preserve this study as a frozen baseline; do not tune the current models in response to its test results.

## Artifacts and reproduction

- protocol.json: full row mapping, split, policy, environment assumptions and hashes.
- ground_truth.json: exact top-10 offsets/scores per query.
- dense_queries.jsonl and dense_summary.csv/json: complete p=1..64 observations.
- timing_queries.jsonl and paired_summary.csv/json: native timing components and retrieval results.
- paired_bootstrap.json: paired query uncertainty.
- source_preservation.json, query_audit.json, done.json: executed checks.
- recall_vs_candidates.png and recall_vs_latency.png: scientific plots.
- environment.json and verification/native-run.log: machine/compiler and execution evidence.

Use a fresh output directory for reproduction; preparation and artifact writers refuse overwrite. Run the saved run_f3.sh with the existing Rust/tch CPU environment, then lmi_phase_f3_analyze.py. The protocol freezes source file hashes and the exact seed. No commit or push was performed.

## Boundary-tie correction and interpretation

The initial native run stopped before measured timings: two valid top-10 results disagreed by an ID at an equal-score boundary. Eleven queries had equal scores in the last two positions of the initial exact top-10. The failed run and its original outputs remain in lmi-phase-f3-attempt1-tie-failure. This was a harness correctness issue, not a demonstrated production search bug.

Both exact and candidate paths now request k+1 and expand when the score at rank 10 equals the last retrieved score, until the entire boundary tie is covered. Both sort by score descending then internal offset ascending and truncate to k=10. Recall remains canonical ID Recall@10, not a post-hoc score-tolerance metric. Full-probe results must match exact IDs AND scores. This benchmark-only canonicalization, including repeated scoring for ties, is included in measured latency. It is not identical to an ordinary production k=10 call.

Tie-related additional scoring calls across ground truth, correctness checks and timing: 272. The same split, frozen models, validation selection rule and two-trial schedule are retained. No performance-based tuning followed the failure.

Existing centroid/affine controls compute with f64 parameters; native MLP inference uses the existing f32 implementation. Contemporary paired timing compares these implementations, not a proof against a maximally optimized centroid kernel.

## Final source changes and checks

Only the test-only evaluation module registration and new F.3 files are added for this study. The existing training.rs and F.2 changes predate F.3 and are preserved. No operational LMI functionality, configuration, dependency or persistence format was changed.

The initial compile failed on a missing trait import; it was fixed. The first native run then stopped on the equal-score tie assertion. The retained final run passed all native F.3 assertions, analysis cross-checks and source-preservation checks. Git diff --check passes.

Tracked diff stat (includes pre-existing F.2 changes; untracked additions are listed below):

```text
 lib/segment/src/index/lmi_index/evaluation.rs |  6 +++++
 lib/segment/src/index/lmi_index/training.rs   | 39 ++++++++++++++++++++++++---
 2 files changed, 41 insertions(+), 4 deletions(-)
```

Final status:

```text
 M lib/segment/src/index/lmi_index/evaluation.rs
 M lib/segment/src/index/lmi_index/training.rs
?? docs/lmi-phase-f2.md
?? docs/lmi-phase-f3.md
?? docs/lmi_supervisor_call_brief.md
?? docs/lmi_supervisor_call_brief.pdf
?? docs/thesis-progress-call-notes.md
?? docs/thesis-progress-full-report.md
?? lib/segment/src/index/lmi_index/evaluation_f2.rs
?? lib/segment/src/index/lmi_index/evaluation_f3.rs
?? tests/lmi_phase_f2_analyze.py
?? tests/lmi_phase_f2_prepare.py
?? tests/lmi_phase_f3_analyze.py
?? tests/lmi_phase_f3_prepare.py
?? tests/run_lmi_phase_f3.sh
```
