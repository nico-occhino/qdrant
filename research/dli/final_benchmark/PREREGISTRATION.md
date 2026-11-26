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
- **Bucket count chosen in the pilot (§8)** from {250, 1,000, 3,162}, separately for S-LMI and IVF, with
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
