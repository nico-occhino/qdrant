# Final benchmark: preregistration

**Status: DRAFT, 2026-11-26.**
- **Not frozen.** Sections marked **[DECIDE]** need the author.
- **CLI (§3.6) is a placeholder** until its design note exists.
- **Freezing.** Freezing means a commit hash recorded here, before the selection pilot (§8) runs. Changes after that
  point are dated amendments, each stating whether it came before or after data.

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

### 2.2 Query panels: fresh rows **[DECIDE]**
- **Why the official queries cannot be used.** All 10,000 official queries have been evaluated in some earlier
  campaign. DLI-best's components were selected on rows 8000–9999 (Phase 3 test and `partb`). Reusing them would
  leak selection into the test.
- **Uniform workloads (W1, W2).** Queries are corpus rows **from 1,000,000 upward**. No 1M study ever inserted or
  queried them, and the 10M studies inserted them only as data.
  - Sample with seed 20261203, without replacement: **test 1,000; validation 64; calibration 2,000** (the last is
    kept for CLI or any arm that needs one).
  - **Caveat:** corpus rows include exact duplicates of other rows. If a query has a duplicate in the live set, the
    duplicate is its own nearest neighbour. This is reported as the share of test queries with a score-1.0
    neighbour, and the tie rule (§2.3) handles it.
- **Shift (W3).** As in Phase 3, but under the new seed: `mq`, `ids` and `ood`, each 1,000 test + 64 validation,
  drawn from holdouts that are never inserted.
- **Validation panels grow from 32 to 64 queries.** In the pilot, a 32-query panel pointed the opposite way from a
  1,000-query one (`dli_best/RESULTS_INTEGRATION_PILOT.md` §5).
- **Alternative:** keep official rows 9000–9999 and disclose the selection leakage. *Not recommended.*

### 2.3 Gold and the recall rule **[DECIDE: recommended option G1]**
The DLI-best pilot showed that gold computed on raw Float16 rows in Float64 disagrees with what any Qdrant arm can
return. Qdrant stores each Cosine vector normalised in f32 and cast to f16. That causes 96% of full-search misses.

- **G1 (recommended), stored-representation gold, tolerance-aware.**
  - The planner normalises each corpus row in f32, casts it to f16, and scores it in Float64 against the f32-normalised
    query.
  - Per (batch, query) it emits the top-30 ids, the rank-30 score s₃₀, and the **tie set**: every live id with score
    ≥ s₃₀ − ε.
  - ε = 768 · 2⁻²⁴ ≈ 4.6·10⁻⁵. This bounds the f32 accumulation error of a 768-dimensional dot product of unit
    vectors, which is how Qdrant scores.
  - Recall@30 = |returned ∩ tie set| / 30, capped at 1. Ties and accumulation-order swaps then count as correct.
    Nothing farther than ε below rank 30 does.
  - **Independent oracle check:** in every job, 1,000 random live points are read back from the collection's vector
    storage after batch 0 and the last batch. Their bytes must equal the planner's simulated stored vectors, or the
    job fails.
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
- **Work counter:** Qdrant's own count of vectors scored (`scoring_work_snapshot`) for every scored vector, graph
  traversal included. The rows also include full scans of plain segments.

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

### 5.4 Replication **[DECIDE: recommended yes]**
- Pilot vs tie-break showed router training differs slightly across hosts. So the whole primary run is repeated once
  on a second plan seed, 20261205, with the same panels drawn under that seed.
- A family A or B verdict is **replicated** if seed 2 gives the same category. Otherwise it is reported as *not
  replicated*.
- This measures stream-, training- and build-level variability, which the query bootstrap cannot.

## 6. Gates (all required per job; a failed gate is reported, never silently rerun)

1. **Ledger:** the live count and SHA-256 match the plan after every batch.
2. **Completeness:** every (batch, panel, arm, knob) row exists.
3. **Oracle:** the stored-vector read-back (§2.3) matches.
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
- Plan seed 20261202, all three workloads, the 64-query validation panels.
- **It chooses:**
  - S-LMI and IVF bucket counts and HNSW m, each by the lower mean validation W@0.90 on W1, ties broken by p50 latency;
  - and it confirms that the grids bracket 0.90 for every arm.
- **It checks:** every gate in §6, and the runtime per job, which sets walltimes.
- **It does not read** any test panel, and it re-selects no DLI component.

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

## 11. Budget and timeline (estimate)
- **Primary:** 5 arms × 3 workloads × 2 seeds. Jobs combine arms per workload where memory allows.
- **Time:** about 2–6 h per (workload, arm) on a perian node, judging by the pilots, so on the order of 30
  node-jobs. The selection pilot is about a third of that.
- **Target:** frozen in early December, primary complete by mid-January 2027. That leaves the rest for writing
  (submission 2027-03-10).
