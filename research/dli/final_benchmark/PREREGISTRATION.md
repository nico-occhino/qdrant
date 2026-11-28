# Final benchmark: preregistration

**Status: FROZEN, 2026-11-26, by the author's go-ahead.**
- The frozen revision is the commit that introduces this status line. Its hash is recorded in
  [FREEZE.md](FREEZE.md) by the next commit.
- From here on, any change is a dated amendment that says whether it came before or after data.
- **Nothing has run:** no selection-pilot or primary job exists. The only data so far are local smoke tests on
  20K–30K-row plans, used to test the implementation (§10). They are not results.
- **CLI (§3.6) is still a placeholder.** It enters by amendment before any primary data, or later on the same plans
  under its own amendment (§3.6).

**Author decisions, 2026-11-26.** The recommendations below were accepted before any data:
- panels from corpus rows ≥ 1M, with 64-query validation panels (§2.2);
- gold option G1 (§2.3);
- a replication seed (§5.4);
- only the three workloads of §2.1;
- the HNSW m candidates {16, 32} (§3.3).

## 1. Question and claims

**Question.** In native Qdrant, under dynamic workloads (growth, deletes, distribution shift), how does a learned
dynamic index compare with standard vector indexes? The learned index is DLI-best, Sommerová's DLI (E1) plus measured
maintenance and allocation fixes. The standard indexes are HNSW, IVF and Qdrant's own static-LMI lifecycle.

**Claims this design can support:**
- per workload: work, latency and maintenance cost at a matched recall target, with paired uncertainty;
- whether DLI-best improves on E1 by a preregistered amount;
- for each baseline: better, equivalent, worse or inconclusive, with a fixed margin.

**What it cannot claim:**
- results for datasets, scales or hardware not run;
- any production performance;
- anything about CLI until §3.6 is filled in.

**Known prior evidence.**
- In static-vs-DLI pilot 2 (validation panels, 6 batches), E1 needed 2.3–4.3× **more** work than Qdrant's static-LMI
  lifecycle. Partition granularity explained most of the gap.
- The DLI-best pilot gave DLI-best / E1 = 0.61–0.91.
- So a fair prior is that DLI-best still needs more work than the static lifecycle, and is cheaper to maintain under
  change. The design must be able to show that outcome as clearly as the opposite.

## 2. Data, panels and gold

### 2.1 Corpus and workloads
- **Corpus:** the SISAP23 LAION2B CLIP-768 Float16 corpus, the same file as Phase 3. The 1M studies use its first 1M
  rows (sha256 of `head -c 1536000000`: `6e1c7237…`).
- **Workloads:** from `emma_path/phase3_plan.py`, with **a new plan seed, 20261201**. It is shared by every arm and
  used by no earlier study:

  | ID | Scenario | Batches | Why |
  |---|---|---|---|
  | W1 | insert-only | 1 + 20 | growth |
  | W2 | constant-R (1:1 inserts to random deletes) | 1 + 20 | soft deletes; vacuum |
  | W3 | shift (10 k-means clusters in chain order; the last cluster withheld) | 20 | drift; out-of-distribution queries |

  The other Phase 3 delete scenarios are left out to bound cost. Phase 3 already decided their questions.
- **Selection seed.** The selection pilot (§8) uses **20261202**, a different plan, on validation panels only.

### 2.2 Query panels: fresh rows (decided)
- **Why the official queries cannot be used.** All 10,000 official queries have been evaluated in some earlier
  campaign. DLI-best's components were selected on rows 8000–9999 (Phase 3 test and `partb`). Reusing them would
  leak selection into the test.
- **Uniform workloads (W1, W2).** Queries are corpus rows **from 1,000,000 upward**. No 1M study ever inserted or
  queried them, and the 10M studies inserted them only as data.
  - Sample with seed 20261203, without replacement: **test 1,000; validation 64; calibration 2,000** (the last is
    kept for CLI or any arm that needs one).
  - **The same uniform panels serve every plan seed** (primary, selection, replication). Only the stream changes.
  - **Caveat:** corpus rows include exact duplicates of other rows. If a query has a duplicate in the live set, the
    duplicate is its own nearest neighbour. This is reported as the share of test queries with a score-1.0
    neighbour, and the tie rule (§2.3) handles it.
- **Shift (W3).** As in Phase 3, but under the new seed: `mq`, `ids` and `ood`, each 1,000 test + 64 validation,
  drawn from holdouts that are never inserted.
- **Validation panels grow from 32 to 64 queries.** In the pilot, a 32-query panel pointed the opposite way from a
  1,000-query one (`dli_best/RESULTS_INTEGRATION_PILOT.md` §5).
- **Alternative:** keep official rows 9000–9999 and disclose the selection leakage. *Not recommended.*

### 2.3 Gold and the recall rule (decided: G1)
The DLI-best pilot showed that gold computed on raw Float16 rows in Float64 disagrees with what any Qdrant arm can
return. Qdrant stores each Cosine vector normalised in f32 and cast to f16. That causes 96% of full-search misses.

- **G1 (recommended), stored-representation gold, tolerance-aware.**
  - The planner models Qdrant's Cosine preprocessing bit for bit (x86-64 AVX+FMA path: four 8-lane FMA accumulators,
    a fixed reduction order, multiplication by f32(1/√length), and rows within 10⁻⁶ of unit length left unchanged).
    It casts each row to f16 and scores it in Float64 against the query preprocessed the same way.
  - Per (batch, query) it emits the top-30 ids, the rank-30 score s₃₀, and the **tie set**: every live id with score
    ≥ s₃₀ − ε.
  - ε = 768 · 2⁻²⁴ ≈ 4.6·10⁻⁵. This bounds the f32 accumulation error of a 768-dimensional dot product of unit
    vectors, which is how Qdrant scores.
  - Recall@30 = |returned ∩ tie set| / 30, capped at 1. Ties and accumulation-order swaps then count as correct.
    Nothing farther than ε below rank 30 does.
  - **Oracle check:** in every job, 1,000 live points, evenly spaced over the sorted live IDs, are read back from the
    collection's vector storage after the first and the last batch. Their bytes must equal the planner's model, or
    the job fails.
    - The model follows Qdrant's source code, so the read-back is what makes it independent evidence.
    - On a node without AVX and FMA, Qdrant uses a different path, and the gate fails instead of passing silently.
- **G2: source gold with a tie set.** Gold stays on raw Float16 rows, plus the same ε tie set. It is simpler, but it
  leaves the rounding misses, whose margins reach 3.4·10⁻⁴ and so exceed ε. *Not recommended.*
- **Either way, recall against the Phase 3 source gold is also reported for every row.** That keeps the results
  comparable with Phase 3 and quantifies the precision ceiling.

## 3. Arms

All arms run in one Qdrant build, on the same plan, corpus storage (Float16, Cosine) and hardware class. Each arm's
configuration is fixed before the primary run, by §8 or as stated.

| Arm | What it is | Lifecycle | Knob swept at evaluation |
|---|---|---|---|
| **E1** | Sommerová's DLI as in Phase 3 (b = 5,000, a = 10, aⁱ buckets, router reuse, `mdn_code`) | DLI maintenance | global budget |
| **DLI-best** | E1 + vacuum > 20% dead + consolidation ρ = 0.25 + `PopulationProportional` (frozen, `emma_path/DLI_BEST_RULE.md`) | DLI maintenance | global budget |
| **S-LMI** | Qdrant's static-LMI lifecycle: ordinary optimizers, one `LmiTrained` model (arm S of the static-vs-DLI study) | Qdrant optimizers | nprobe |
| **IVF** | S-LMI with the exact centroid router (`LMI_RESEARCH_ROUTER=centroid`), i.e. IVF-Flat with spherical k-means | Qdrant optimizers | nprobe |
| **HNSW** | Qdrant's native HNSW | Qdrant optimizers | `ef` |
| **CLI** | placeholder, §3.6 | — | — |

### 3.1 E1 and DLI-best
- Exactly as frozen; no knob is tuned.
- Policies replayed on each snapshot:
  - DLI-best: `proportional_native`, the served policy, and `mdn_code`, a secondary descriptive replay the author
    accepted on 2026-11-26;
  - E1: `mdn_code`.
- **Budget grid:** the 17-point policy-replay grid plus full search. Every DLI arm therefore attains every recall
  target, because full search is exact under G1.

### 3.2 S-LMI and IVF
- Model rule as in the static-vs-DLI study: hidden 128, 10 epochs, 25 k-means iterations, and the same sample rule.
  `indexing_threshold` 20,000 KB; one optimization job at a time.
- **Bucket count chosen in the pilot (§8)** from {250, 1,000, 3,162, 10,000 (A1.2)}, separately for S-LMI and IVF, with
  `default_segment_number = 1`.
- **nprobe grid:** doubling from 1 up to all buckets, so both arms also attain every target.

### 3.3 HNSW
- Qdrant defaults: m = 16, ef_construct = 100, full-scan threshold at its default, the same optimizer settings as
  S-LMI.
- **m chosen in the pilot** from {16, 32}.
- **`ef` grid:** 30, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1,024, 1,448, 2,048.
  - HNSW has no exact mode, so it can fail to attain a target.
  - A non-attaining batch is reported as "not attained above `ef` 2,048". It is not extrapolated.
- **Work counter:** every scored vector, counted through the per-task scoring guard (I3), graph traversal and entry
  points included. The rows also include full scans of plain segments, and of HNSW segments below Qdrant's
  `full_scan_threshold`.

### 3.4 Shared harness rules
- Inserts arrive in chunks of b, then deletes are applied. Maintenance or optimizers then settle to a fixed point, the
  ledger is checked, the shard is flushed and copied, and the copy is replayed.
- All arms are replayed with in-process segment search on the copied shard. Server RPC is not used, so timing excludes
  networking. The search route is logged per row: learned segment, plain scan or exact fallback.

### 3.5 What is not an arm
- **E1R** (always retrain) is an upper-bound reference, already measured in Phase 3.
- **Calibrated allocation** lost the tie-break.
- **SG** is a diagnostic arm; it stays in the static-vs-DLI study.

### 3.6 CLI **[PENDING the author's reading of Jakubík's thesis]**
- It enters by a dated amendment **before any primary data**, with its configuration, mutability design and at most one
  variant.
- **If CLI is not ready** when everything else is, the primary run goes ahead without it. CLI is then added later on
  the **same plan seed and panels**, under its own frozen amendment, and compared with the archived arms.

## 4. Metrics

**Primary.** W@0.90 (G1 recall): mean vectors scored to reach mean Recall@30 = 0.90.
- Linear interpolation between the two bracketing grid points, never extrapolated.
- Each interpolated value records its bracketing points and fraction. It is labelled an estimate, never a deployable
  setting.
- **Per workload and panel:** R(X, Y) = mean_t W_X(t) / mean_t W_Y(t), over batches 1–20 (shift: 0–19) where both
  arms attain. The attainment share is reported.
- **Batch 0 (W1, W2)** is descriptive.

**Secondary, all reported:**
- **Recall at fixed work:** mean Recall@30 at 50K vectors scored, the Phase 3 B\*, interpolated per arm.
- **Latency at the target recall:**
  - single-thread, in-process, warm (one untimed pass first);
  - p50 and p95 per (batch, panel) at the measured configuration nearest **above** 0.90 recall, so no setting is
    interpolated.
  - Timed runs are separate from the work-counting runs, on the same snapshot, with tracing off.
- **Maintenance cost per batch:**
  - ingest and settle wall seconds, maintenance CPU seconds, trainings and training seconds;
  - points written and write amplification;
  - for HNSW and the static arms, optimizer jobs and their seconds.
- **Space:** index bytes on disk per arm and snapshot (vectors, postings or graph, model), and peak RSS per job.
  Bytes, RSS and scheduler accounting are kept separate.
- **Counters, where the arm has them:** visited postings, distinct candidates, dead postings scored and exact
  fallbacks. Counters an arm lacks are marked *unknown*.
- **Recall against the Phase 3 source gold** for every row (§2.3).
- **Concurrent service (descriptive):** one run per arm and workload with `dli_service_workload`, at a fixed write
  rate and open-loop query rate. It records achieved QPS, latency quantiles, timeouts and the deletion-visibility
  check, and is separate from the timed runs above.

## 5. Statistics and decision rules

**Uncertainty.**
- Paired by (batch, query): every arm sees the same plan and queries.
- 95% CIs come from a batch-stratified query bootstrap with 10,000 resamples and seed 20261204, as in Phase 3.
- Bonferroni-adjusted intervals are reported for each family below.

### 5.1 Family A: DLI-best against E1 (hypothesis, 5 panel decisions)
- Panels: W1 test, W2 test, W3 `mq`, `ids` and `ood`.
- **A is supported** in a panel if R(DLI-best, E1) ≤ 0.90 and the 95% CI upper bound is < 1.
- The 0.90 bar matches H10's 10%. The pilot (validation, 32 queries) gave 0.61–0.91, so `ood` is a real test.

### 5.2 Family B: DLI-best against each baseline (no directional hypothesis)
- Baselines: S-LMI, IVF and HNSW (plus CLI if added). Each is classified per panel, on work and on latency
  separately:
  - **better:** the CI is entirely below 0.95;
  - **equivalent:** the CI is inside [0.95, 1.05];
  - **worse:** the CI is entirely above 1.05;
  - **inconclusive:** otherwise.
- **HNSW** is judged on latency as well as on work, because a scored vector costs it differently (graph traversal).
  **A work-only verdict against HNSW is never reported on its own.**
- Maintenance cost and space are reported per workload, with no verdict.

### 5.3 Attainment rule
- A ratio with an attainment share below 80% is *descriptive*.
- For HNSW, every panel and batch where it does not attain is listed.

### 5.4 Replication (decided: yes)
- Pilot vs tie-break showed router training differs slightly across hosts. So the whole primary run is repeated once
  on a second plan seed, 20261205. The uniform workloads keep the same query panels (§2.2); shift draws its holdout
  panels under that seed, as its plan requires.
- A family A or B verdict is **replicated** if seed 2 gives the same category. Otherwise it is reported as *not
  replicated*.
- This measures stream-, training- and build-level variability, which the query bootstrap cannot.

## 6. Gates (all required per job; a failed gate is reported, never silently rerun)

1. **Ledger:** the live count and SHA-256 match the plan after every batch.
2. **Completeness:** every (batch, panel, arm, knob) row exists.
3. **Oracle:** every sampled stored vector is bit-identical to the planner's model (§2.3, `oracle_check.py`).
4. **Known answer:**
   - every DLI arm at full search, and every static arm with all buckets probed, has G1 Recall@30 ≥ 0.999 in every
     (batch, panel);
   - HNSW is checked through Qdrant's exact search on its own snapshot.
5. **Native equals control:** DLI-best `proportional_native` equals the offline `proportional` control in work in
   every row, as in the pilot.
6. **Monotone work** in the knob, per arm and batch, with non-monotone points listed.

## 7. Reporting
- Per job: an immutable manifest (commit, binary sha256, source key, plan, panel and corpus hashes, hardware, threads,
  run order), raw rows, `SHA256SUMS` and the analysis command.
- All results go to C:, verified against storage.
- **Every result is reported, whatever its direction:** failed gates, non-replicated verdicts, non-attainment and
  negative findings (for example, DLI-best worse than S-LMI).

## 8. Selection pilot (before the primary; validation panels only)
- Plan seed 20261202 and the 64-query validation panels. No test panel is read, and no DLI component is re-selected.
- **Stage 1 (W1):** E1, DLI-best, S-LMI and IVF with each of {250, 1,000, 3,162} buckets, and HNSW with m ∈ {16, 32}.
  That is 10 arm jobs.
  - **Choice** (`analyze_final.selection`): per arm family, the configuration with the lowest mean validation
    W@0.90 over W1 batches 1–20. The mean is taken over the batches where every candidate of that family attains.
    Exact ties are broken by the lower mean p50 latency.
  - The choice is recorded as a dated amendment, derived from validation data only. That use is preregistered here.
- **Stage 2 (W2, W3):** E1, DLI-best and the chosen S-LMI, IVF and HNSW configurations. That is 5 arm jobs per
  workload.
- **Both stages check:**
  - every gate of §6;
  - that every arm's grid brackets 0.90 in every batch (no `upper_bound`, and non-attainment only where §3.3 allows
    it);
  - the runtime per job, which sets the primary walltimes.
- **If a stage-1 grid fails to bracket,** the grid is extended by a dated amendment before stage 2. No candidate is
  dropped for failing to bracket.

## 9. PostgreSQL replication subset (separate preregistration, in the PostgreSQL repository)
- It reuses this document's plans, panels and gold contract, with gold recomputed for PostgreSQL's own stored
  representation.
- Arms: pgvector HNSW, pgvector IVFFlat, and the native DLI access method.
- Workloads: W1 and W2.
- Its question is transfer: does the ordering of arms seen in Qdrant hold under an optimizer-driven DBMS?

## 10. Implementation work before the pilot (none needs data)

| # | Item | Test before use |
|---|---|---|
| I1 | Planner: new seeds, corpus-row query panels, G1 gold, tie sets (s₃₀, ε) | known-answer on a synthetic corpus with planted ties; regenerating the Phase 3 source gold reproduces its hashes |
| I2 | Driver: Recall@30 counted against the tie set; source-gold recall alongside | unit test with a planted tie |
| I3 | Driver: HNSW arm (static system, no LMI model) and its `ef`-sweep replay with the work counter | `ef` = large on a small plan reaches exact-search recall |
| I4 | Driver: IVF through the static system with the centroid router | IVF rows equal a direct centroid ranking on a small plan |
| I5 | Stored-vector read-back gate | fails on a deliberately perturbed vector |
| I6 | Timed replay mode (warm, single thread, no counters) | p50 is stable across two passes within 5% on a small plan |
| I7 | Analysis script: families A and B, attainment, replication | known-answer synthetic test, as in earlier studies |

**Status of I1 (2026-11-26): done.** [`fb_plan.py`](fb_plan.py) and [`test_fb_plan.py`](test_fb_plan.py).
- **Known-answer tests pass:**
  - the stored-gold tie sets equal a brute-force sort, with planted exact duplicates, one-ulp near-duplicates and
    three chunk sizes;
  - an end-to-end corpus-panel plan passes its checks: query rows above the floor and never inserted, tie sets
    containing the stored gold, CSR offsets, and live digests.
- **Regression.** With official panels and source gold only, it reproduces the Phase 3 preflight plans (job 25842074,
  100K) byte for byte: 18/18 constant-R files and 35/35 shift files, gold and queries included. The only extra files
  are the new query-ID provenance files.
- **Measured on 1,064 corpus-row queries over 200K rows:**
  - stored-gold and source-gold top-30 agree on 99.77% of ids;
  - the mean tie set is 30.76 ids, and 6% of queries have more than 30 (at most 334, from duplicates);
  - stored gold costs about 1.2× source gold, so a 1M insert-only plan takes about 25 minutes.

**Status of I2 (2026-11-26): done.**
- **Code.** `QuerySet::with_ties` loads the planner's CSR tie sets. The DLI frontier, the static nprobe frontier and
  the service driver's settled pass now write `recall30_g1` next to the unchanged source-gold `recall30` (null for
  plans without tie sets, so Phase 3 analyses are unaffected). The Phase 3 driver attaches tie sets whenever the plan
  has them.
- **Tests.** The unit test `g1_recall_counts_tied_copies_and_caps_at_top` passes, as do all 54 collection DLI/LMI
  tests. Clippy reports nothing on the changed lines.
- **Smoke test** (a 20K-row G1 plan, constant-R, 3 batches, 24 queries, debug build):
  - **DLI-best:** 1,152 rows. At full search, G1 recall is 30/30 for all 96 queries under all three policies. Source
    gold misses 2 per policy, which is the rounding effect G1 removes. Native and control agree in work and G1 recall
    on every row.
  - **Static S-LMI (16 buckets):** 480 rows. With all buckets probed, G1 is 30/30 for 96/96 queries; source gold is
    94/96.

**Status of I3 (2026-11-26): done.**
- **Driver.** `DLI_P3_SYSTEM=hnsw` runs the static arm's optimizer lifecycle with no LMI model, so optimized segments
  become Qdrant HNSW indexes. Settings: `DLI_P3_HNSW_M`, `DLI_P3_HNSW_EF_CONSTRUCT` (default 100), and the static
  arm's segment and indexing-threshold variables. `run_hnsw_frontier` replays every `ef` in `DLI_P3_HNSW_EFS` through
  Qdrant's own segment search and merges the per-segment top-30.
- **Work.** A thread-local `TaskScoringProfileGuard` counts every scored vector. HNSW graph-search scorer calls
  (`FilteredScorer::score_points_unfiltered`, `score_point`) now feed that per-task counter. It counts only inside an
  active guard, so builds and service queries are not counted.
- **Qdrant behaviour, kept as is:** an HNSW segment smaller than `full_scan_threshold` (default 10,000 KB, about
  6,600 vectors here) is scanned exhaustively by Qdrant itself, and that work is counted.
- **Tests.** All 412 segment HNSW tests and all 54 collection DLI/LMI tests pass. Clippy reports nothing on the
  changed lines.
- **Smoke test** (30K-row G1 insert-only plan, m = 16, 24 queries, debug build):
  - batch 0 (3,000 points, below the threshold): the scored count is exactly 3,000 at every `ef`, an independent check
    of the counter;
  - batches 1–3 (12K–30K points): mean work grows with `ef` (at 30K: 489 → 16,369 for `ef` 30 → 4,096), and G1
    recall rises from 0.885 to 1.000. Source gold reaches only 0.997–0.999 at the top `ef`.

**Status of I4 (2026-11-26): done.**
- **Arm.** IVF is `DLI_P3_SYSTEM=static` with `LMI_RESEARCH_ROUTER=centroid`, labelled `IVF/b{B}/seg{n}`. The
  variable is process-wide, so the driver refuses it for the DLI and HNSW systems; that refusal was checked.
- **Replay checks.** The static replay classifies every learned segment's router as centroid (one Linear layer, zero
  bias, unit-norm rows) or MLP, and asserts that the class matches the arm. For IVF, at every query and starting
  nprobe, it asserts that the router's chosen buckets are the best centroids by dot product with the routed query.
- **Tests.** Unit tests cover the classifier, a correct ranking, and rejection of a wrong pick. All 56 collection
  DLI/LMI tests pass. Clippy reports nothing on the changed lines.
- **Smoke test** (30K-row G1 insert-only plan, 32 buckets, nprobe 1–32, 24 queries):
  - IVF: all 8 optimizer builds logged `"router":"centroid"`, and the centroid-ranking assertion held on every query
    of every batch;
  - S-LMI: all 8 builds trained the MLP;
  - final batch (30K points, nprobe 1 → 32): IVF G1 recall 0.576 → 1.000 for 1,063 → 30,000 vectors scored; S-LMI
    0.563 → 1.000 for 1,047 → 30,000. Full probing is exact for both.

**Status of I5 (2026-11-26): done.**
- **Code.**
  - The driver calls `dump_stored_sample` on the flushed snapshot after the first and the last batch of every system.
    It writes 1,000 live points, evenly spaced over the sorted live IDs, as `(id, 768 × f16)` records, and asserts
    each is visible in exactly one segment.
  - `oracle_check.py` compares them bit for bit with `fb_plan.stored_rows`, the function that also produces the G1
    gold.
- **First result: the planner's model was wrong.**
  - The f32-normalise-then-cast model differed from storage by one f16 ulp in 2.1–3.0% of sampled vectors, the same
    on all three systems.
  - Cause: Qdrant's AVX path sums the squares in four FMA accumulators, reduces them in a fixed order, multiplies by
    the reciprocal square root, and leaves rows within 10⁻⁶ of unit length unchanged (52 of 50K corpus rows).
  - `fb_plan.qdrant_cosine_preprocess` now models that path exactly. Since then, **6,000 of 6,000** sampled vectors
    are bit-identical: DLI-best, S-LMI and HNSW, batches 0 and 3 of the 30K smoke plan.
- **Re-checks after the model change.** The planner tests pass, and the 200K real-data figures of I1 are unchanged
  (99.77% agreement, mean tie set 30.76, at most 334). The I2–I4 smoke tests used plans made with the old model;
  their checks do not depend on it.
- **Tests.** The oracle known-answer test passes: model vectors pass, a one-ulp change fails (with score effect below
  ε), and empty or truncated samples are rejected. All 56 collection DLI/LMI tests pass. Clippy reports nothing on
  the changed lines.

**Status of I6 (2026-11-26): done.**
- **Mode.** `DLI_P3_TIMED=1` runs `run_timed` on the same flushed copy after the work replay, through each arm's
  production search path with no tracing or profiling:
  - DLI: `route_pinned_traced` with no trace, then candidate resolution, per policy in `DLI_P3_TIMED_POLICIES`
    (native policies only);
  - S-LMI/IVF: route at nprobe, score with `score_candidates_generic` (the scorer native LMI search uses), and widen
    until 30 results are found;
  - HNSW: segment search at each `ef`.
- **Protocol.** One untimed pass over every (query, knob), then one timed pass per query end to end (all segments
  plus merge) on one thread. Recall is computed after the clock stops.
- **Check.** Timed rows reproduce the work replay's recall on **1,440 of 1,440** rows: DLI-best (`mdn_code`,
  `proportional_native`), S-LMI and HNSW on the 30K smoke plan. Smoke latencies come from a debug build and are not
  results; the jobs use the optimised build.
- **HNSW known-answer rows.** The HNSW replay also records Qdrant's exact search (`hnsw_exact`, budget 0) per query.
  On the smoke plan these rows give G1 30/30 on 96 of 96, and their scored count equals the live points.

**Status of I7 (2026-11-26): done.** [`analyze_final.py`](analyze_final.py) and
[`test_analyze_final.py`](test_analyze_final.py).
- **Implements:**
  - W@0.90 under G1;
  - the paired, batch-stratified bootstrap (10,000 resamples, seed 20261204);
  - Family A, Family B on work and latency, and the 80% attainment rule;
  - Bonferroni intervals (family A: 5 decisions; family B: 30);
  - the secondary `mdn_code` replay on the DLI-best topology;
  - replication across seeds;
  - the per-run gates of §6 except the oracle, which is `oracle_check.py`.
- **Two rules fixed here, before any data:**
  - **If a batch's first grid point already reaches 0.90,** W is that point's work, flagged `upper_bound`. The grids
    are designed so this should not happen, and every occurrence is reported.
  - **Latency per batch** is the median timed query latency at the first grid point whose full-panel mean G1 recall
    is ≥ 0.90. That point is held fixed within bootstrap resamples, and latency R = mean_t L_X / mean_t L_Y.
- **Tests.** The known-answer test with planted W values passes: interpolation, upper bound, non-attainment, A
  supported, B better, worse and equivalent, latency ratios, replication, gates, and a broken ledger plus a
  native/control mismatch being caught. Run on the real 30K smoke outputs, every gate passes. Those verdicts are not
  results.

## 11. Budget and timeline (estimate)
- **Selection pilot:** 3 plan jobs and 10 + 10 arm jobs.
- **Primary:** 6 plan jobs (3 workloads × 2 seeds) and 30 arm jobs (5 arms × 3 workloads × 2 seeds), plus CLI by
  amendment.
- **Time:** about 2–6 h per arm job on validation panels, judging by the DLI-best pilot. Test panels add replay
  time. Primary walltimes are set from the stage-1 runtimes, at 1.5× the longest of each arm, and recorded in the
  manifest before submission.
- **Target:** pilot in early December, primary complete by mid-January 2027 (submission 2027-03-10).

## 12. Execution
- **One script, [`pbs/fb.pbs`](pbs/fb.pbs):**
  - `MODE=plan` writes the immutable plan of a (workload, seed) to
    `final-benchmark-20261126/plans/<W>-<seed>/`. It hash-checks the plan after publishing and makes it read-only.
  - `MODE=pilot|primary` runs one (workload, seed, arm, configuration). It copies that plan, re-verifies every file
    hash and the plan's seed and scenario, and runs the driver with `DLI_P3_TIMED=1`.
  - Then come the oracle gate (`oracle_check.py`) and the per-run gates (`analyze_final.gates`). A failed gate fails
    the job, and the data are kept.
- **Script guards:**
  - the pilot only on seed 20261202, and the primary never on it;
  - only preregistered seeds and configurations;
  - refusal of a node without AVX/FMA;
  - result directories are never reused.
- **Staging** works as in the earlier campaigns: source bundle of the frozen commit, vendor tarball and
  `inputs.sha256`, in `native-<commit12>`. The binary comes from the source-keyed cache.
- **Each submission needs the author's explicit authorisation.** Job IDs, hosts and walltimes go into
  `PBS_MANIFEST.json`.
- **Analysis:** `analyze_final.py --manifest RUNS.json`, with `"mode": "pilot"` for the selection pilot. Results go
  to C: with verified checksums.

## 13. Amendments

### A1 (2026-11-26, after the stage-1 validation data; validation panels only, no test panel read)
Authorised by the author ("accept all recommendations, push and resubmit"). Stage-1 outcomes are in
`PBS_MANIFEST.json`. Nine arm jobs completed, and every §6 gate passed. The provisional choice by §8's rule was
S-LMI 3,162, IVF 3,162 and HNSW m = 16.

- **A1.1 DLI-best rerun.**
  - **What happened.** Job 26011819 failed in its run phase (exit 101) after batch 16 of 21. The driver panicked when
    `writeln!` to `batches.jsonl` on network storage returned EBADF (os error 9). This was not a gate failure, and the
    cause is not established.
  - **What we do.** It is resubmitted **once, unchanged**, on the same staged revision (`native-52347112c7c2`). No
    data from the failed job are used.
  - **If it fails again in the same way,** it is treated as a defect. The defect is diagnosed and fixed by a further
    amendment before stage 2.
- **A1.2 Wider bucket grid for S-LMI and IVF.**
  - **Why.** Both families chose the largest candidate, 3,162. The §8 rule only requires extension when a recall
    grid fails to bracket, so this step goes beyond the rule. It is taken so the baselines are not under-tuned.
  - **The added candidate.** 10,000 buckets, one more half-decade step: about 100 vectors per bucket at 1M rows. It
    is added to §3.2's set, with the nprobe grid doubling from 1 up to 10,000.
  - **The choice.** It is re-taken by §8's unchanged rule over {250, 1,000, 3,162, 10,000}, using the stage-1 jobs
    for the first three candidates.
  - **Decided before these data:** the grid is not extended further. If 10,000 is chosen, it is used, and the choice
    at the edge of the grid is reported as a limitation.
- **A1.3 HNSW is not amended here.**
  - **What failed.** §8's bracketing check fails: the first `ef`, 30, already yields `upper_bound` in batch 0 for m = 16 and in 10 of
    batches 1–20 for m = 32.
  - **Why the grid cannot simply go lower.** Qdrant sets the effective `ef` to max(`ef`, limit)
    (`lib/segment/src/index/hnsw_index/graph_layers.rs:552`), and the limit is 30. So any `ef` below 30 is identical
    to 30.
  - **What comes next.** The HNSW remedy is a separate amendment, and it comes before stage 2.
- **Script.** `pbs/fb.pbs` accepts CFG 10000 for S-LMI and IVF. Nothing else changes. The binary's source key is
  unchanged, because no file under `lib/`, `src/` or `Cargo.*` changes.

### A2 (2026-11-26, after the stage-1 validation data; validation panels only, no test panel read)
Authorised by the author ("accept, write A2 and push"). This resolves A1.3.

- **A2.1 HNSW keeps m = 16.**
  - **The rule's choice.** §8's rule picks m = 16: mean validation W 4,698 against 5,738 for m = 32, over W1 batches
    1–20.
  - **The m = 32 figure is an upper bound.** m = 32 is `upper_bound` in 10 of those batches, so its 5,738
    overstates it.
  - **Why the choice still holds.** In the 11 batches where both settings are interpolated (2, 4–9, 14, 16, 17
    and 19), m = 16 averages 2,954 against 5,738.
  - **Is the grid extended?** No. The ef grid cannot go lower (A1.3), and the m grid is not extended.
  - **Bracketing on W1.** m = 16 is `upper_bound` only in batch 0, which W1's analysis window (batches 1–20)
    excludes.
- **A2.2 The upper-bound rule, for every arm and for both work and latency.** §8's requirement of no `upper_bound`
  is relaxed for HNSW only, where it cannot be met. Stage 2 still checks it for every other arm.
  - **What an upper bound means.** A W flagged `upper_bound` means that arm's true W is at most the reported value.
    So with R = W(DLI-best) / W(Y):
    - an upper bound on the baseline Y means the true R ≥ the reported R;
    - an upper bound on DLI-best means the true R ≤ the reported R.
  - **Which decisions survive.** Only decisions that moving R in the possible direction cannot change:
    - baseline bound only: Family B "worse" and Family A "not supported";
    - DLI-best bound only: "better" and "supported";
    - both: none.
  - **Every other decision** becomes "not decidable (upper bound)". The original decision and the bound batches are
    reported alongside it.
  - **Latency uses the same rule.** The latency grid point is the first one reaching 0.90, so it is grid point 0
    exactly when W is `upper_bound`.
  - **Implementation.** `analyze_final.bound_adjust`, with known-answer tests in `test_analyze_final.py`. Point
    estimates and intervals are unchanged; only the decision label changes.

### A3 (2026-11-27, after the A1 jobs' validation data; validation panels only, no test panel read)
Authorised by the author ("analyze, document and try again"). Details are in `RESULTS_SELECTION_W1.md`.

- **A3.1 What happened to the A1 jobs.**
  - **DLI-best rerun (26027409).** It completed and passed every gate in 46 min on another host. The EBADF in
    26011819 did not recur, so it is recorded as a one-off on that host (A1.1). No further amendment is needed.
  - **The 10,000-bucket jobs (26027410 S-LMI, 26027411 IVF)** failed only the monotone-work gate (§6.6). Every other
    gate and the oracle check passed. Their data were kept (§12).
- **A3.2 The cause is a production code path, not a counter defect.**
  - **The mechanism.** When the probed buckets hold fewer than `top` = 30 points, Qdrant's static LMI widens:
    `widen_until_filled` in `lib/segment/src/index/lmi_index/mod.rs` doubles nprobe and rescores the widened set
    from scratch. So a query that widens at nprobe n costs work(n) = s(n) + work(min(2n, B)), where the first pass
    s(n) < 30.
  - **What this predicts.** Every per-query drop in work from n to min(2n, B) is between 1 and 29 vectors.
  - **What the data show:**
    - 486 drops in the two jobs, all between 1 and 29 (maximum 29);
    - all at nprobe 1, 2 and 4, mostly in the early batches, where 10,000 buckets hold about 5–10 points each;
    - W@0.90 always falls between nprobe 8 and 128, where there is no drop;
    - the passing 3,162-bucket runs show 67 such per-query drops, too few to move a batch mean.
- **A3.3 Gate amendment (applies to every job, before and after this one).** A drop in batch-mean work between
  adjacent knobs is accepted as a widening restart only when all three hold:
  - the arm is S-LMI or IVF;
  - the step is n → min(2n, B);
  - every query whose work falls on that step falls by fewer than K = 30 vectors.

  Accepted drops are listed in `widening_restart_points`. Any other drop fails the gate and is listed in
  `non_monotone_points`, which §6.6 always required to be listed. This is implemented in
  `analyze_final.widening_restart`, with known-answer tests.
  - **The direction of this change.** It admits a stronger baseline configuration, so it cannot favour DLI-best.
  - **The rerun.** Re-evaluated on the kept data, all 12 W1 runs pass.
  - **Stage 2 and the primary** run this gate inside the job, from a new staging of this revision.
- **A3.4 The choice (§8, recorded here as §8 requires):**

  | Arm | Choice | Candidates, mean validation W@0.90 over W1 batches 1–20 |
  |---|---|---|
  | S-LMI | **10,000** buckets | 250: 15,357; 1,000: 9,051; 3,162: 6,333; 10,000: 3,307 |
  | IVF | **10,000** buckets | 250: 14,501; 1,000: 6,634; 3,162: 3,857; 10,000: 1,686 |
  | HNSW | **m = 16** (A2.1) | 16: 4,698; 32: 5,714 (upper bound) |

  - **Both static choices sit at the edge of the grid.** Per A1.2, decided before these data, the grid is not
    extended further, and this is reported as a limitation.
  - **Bracketing.** Every static candidate brackets 0.90 in every batch. HNSW is handled by A2.
- **A3.5 Walltimes.**
  - **What the pilot showed.** Static 10,000-bucket runs are dominated by Qdrant's per-batch index build: S-LMI took
    10.2 h of its 12 h walltime and IVF 6.6 h. Primary panels have 1,000 queries instead of 64.
  - **What we set.** Stage 2 and the primary request 24 h for the static arms, and 12 h for the others. For the W3
    primary, A4.2 replaces this with 48 h.

### A4 (2026-11-28, after the stage-2 validation data; validation panels only, no test panel read)
- **A4.1 What stage 2 showed** ([RESULTS_SELECTION_STAGE2.md](RESULTS_SELECTION_STAGE2.md)).
  - Every gate passes.
  - The pilot evaluated 64 queries per validation panel; the primary evaluates the 1,000-query test panels.
  - Per batch, the records separate ingest (`ingest_seconds`) from query evaluation (`frontier_seconds` +
    `timed_seconds`). Scaling only query evaluation by 1,000/64 projects W3 to 12.6 h (E1), 21.0 h (DLI-best), 25.2 h
    (S-LMI 10,000), 24.0 h (IVF 10,000) and 3.0 h (HNSW 16).
  - At A3.5's walltimes (12 h, and 24 h for the static arms), W3 E1, DLI-best and S-LMI would exceed them, and IVF
    would reach its limit. W1 and W2 project to ≤ 4 h per arm.
- **A4.2 Walltimes (replaces A3.5 for W3 only).**
  - **W3:** E1, DLI-best, S-LMI and IVF request **48 h**; HNSW keeps 12 h.
  - **W1 and W2:** unchanged (24 h static, 12 h otherwise).
  - **The basis.** §11's rule (1.5× the longest runtime) applied to the projection gives about 19 h, 32 h, 38 h and
    36 h. 48 h covers each, with margin for host variation: the W3 static arms ran on different node types.
- **A4.3 Nothing else changes.** Jobs, plans, panels, grids, gates and analysis are as before; only the requested
  walltime differs. A job that still exceeds 48 h is an infrastructure failure under §6, reported and not silently
  rerun. Windowed evaluation was considered and rejected: every window replays the stream from batch 0, so the
  static arms' ingest (12 h for S-LMI) would be paid per window.
- **A4.4 Submission.** Arm jobs depend on their plan job (`-W depend=afterok:`). Stage 2 met a PBS dependency system
  hold, so a held arm may be cancelled and resubmitted once its plan job has completed. Each such resubmission is
  recorded in `PBS_MANIFEST.json`, and every primary submission still needs the author's explicit authorisation
  (§12).

### A5 (2026-11-28, before any primary data): primary without CLI; CLI design decisions
- **A5.1 The primary starts without CLI**, as §3.6 allows. The author authorised it on 2026-11-28. Plan jobs
  26079249–26079254 cover W1–W3 × seeds 20261201 and 20261205. Each plan's five arms are submitted when that plan
  job has completed (A4.4); the arm job IDs go in `PBS_MANIFEST.json`.
- **A5.2 CLI enters later**, on the same plan seeds and panels, under its own frozen amendment, and is compared with
  the archived arms (§3.6). The author's decisions, recorded now:
  - **Mutability: in-segment.** An appendable learned segment whose posting lists and router change in place
    (Jakubík's design), not copy-on-write publication. It needs its own visibility, recovery and concurrency gates
    in Qdrant's segment layer, designed before code.
  - **The one variant: an imbalance trigger.** CLI's entropy trigger is replaced by the frozen bucket-imbalance
    rule (max/mean ≥ 8.0 over allocated buckets, as Q5's `ImbalanceRetrain::POSTGRES_FROZEN`). The growth trigger
    and DEDRIFT are unchanged. This links CLI to Q5.
  - **Still open** (fixed in the CLI amendment, before CLI data): b_max and soft budget, chosen by a selection pilot
    on seed 20261202's validation panels, as in §8; the replay-buffer size; the fine-tune cadence; and the
    hyperparameters, from thesis Table 3.2, following the thesis where `cli_ablation` disagrees (A4–A7 of the
    audit).
