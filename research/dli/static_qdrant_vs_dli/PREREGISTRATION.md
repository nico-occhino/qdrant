# Preregistration: Qdrant static-LMI lifecycle (S) vs Sommerová-style DLI (D)

**Status: DRAFT, 2026-11-22.** It is frozen only after the pilot (§7) and the author's approval, before any primary
(test-panel) data exists. Changes after freezing are dated amendments, and each one says whether it was made before or
after data. This preregistration is separate from the frozen Phase 3 preregistration and does not change it.

Design and rationale: [DESIGN.md](DESIGN.md). Audit of both lifecycles: [BASELINE_AUDIT.md](BASELINE_AUDIT.md).

## 1. Hypotheses (per workload W1–W4)

Primary contrast: R_w = mean_t W_D(t) / mean_t W_S*(t), where S* is the S configuration selected in the pilot (§7).
- **H1 (search efficiency).**
  - D needs at least 20% less work than S* at mean test R@30 = 0.90: R_w ≤ 0.80, with the 95% CI upper bound < 1.
  - S* needs at least 20% less work than D: R_w ≥ 1.25, with the 95% CI lower bound > 1.
  - Otherwise the result is *no practically relevant difference*. It is *inconclusive* if the CI includes both 0.80
    and 1.25.

  A 20% difference is the smallest effect of interest: 1 pp of R@30 costs about 8% work at the operating point
  (`emma_path/PHASE3_PREREGISTRATION.md` §5), so 20% is about 2.5 pp of recall at equal work.
- **H2 (attainment).** The per-batch attainment of R@30 = 0.90 within the grid is reported for each arm. An arm that
  attains in fewer than 80% of batches 1–20 has its R_w reported as descriptive only.
- **H3 (maintenance cost, descriptive with CIs where defined), for both arms:**
  - total training seconds, optimizer wall seconds and optimizer publications;
  - full retrainings and router reuses;
  - logical write amplification and physical bytes written;
  - settle time per batch.
- **H4 (concurrent service, descriptive).** At each arm's validation-locked operating point (§5): achieved QPS,
  p50/p95/p99 latency, timeouts and maximum stall under a fixed writer rate. This is defined in the concurrent phase
  (DESIGN.md §6), which must exist before freezing.

**Multiplicity.** There are four workload-level primary decisions (H1 × W1–W4). A Bonferroni sensitivity check (98.75%
CIs) is reported; the decisions use 95%.

## 2. Data, seeds and scale

- SISAP23 LAION2B CLIP-768 Float16, corpus SHA-256 `f5868d6e…` (MetaCentrum `results/sisap23-matched-final-20261028-01/corpus.f16le`).
- Primary N = 1,000,000 (the first 1M rows), 20 batches per workload, plan seed 20261121.
- Pilot seed 20261301, used by no other study.
- 10M confirmation of any supported H1 only, after a separate approval.

## 3. Queries and gold

- Test and validation sets as in DESIGN.md §2.
- Gold is the exact top-30 over the live set after each batch, over the stored Float16 rows (`phase3_plan.py`). It
  is not the official SISAP gold.
- **Known-answer gate:** at the maximum grid point, both arms reach R@30 ≥ 0.995 on every batch. Misses there are
  near-ties from f32 versus float64 scoring.

## 4. Grids

- **D** global budget: 1K, 2K, 3.5K, 5K, 7K, 10K, 14K, 20K, 28K, 40K, 50K, 56K, 80K.
- **S** nprobe per segment: 1, 2, 4, 8, 16, 32, 64, 128, 256, capped at each segment's bucket count.
- The pilot must show that both grids bracket R@30 = 0.90 on validation. If they do not, the grids are extended
  before freezing.

## 5. Operating points (for H4 only; H1 uses whole frontiers)

The smallest grid point whose mean **validation** R@30 ≥ 0.90 at the batch-10 snapshot of the pilot is locked per arm
and workload. It is never chosen on test.

## 6. Statistics

- Paired query bootstrap, 10,000 resamples, seed 20261123.
- Resampling is stratified by batch: the same resampled query indices for both arms within a batch.
- The time average is one trajectory statistic, a ratio of batch means. The repeated snapshots share queries and one
  ingestion stream, so they are not treated as independent.
- **Limitation, stated up front:** the CIs cover query sampling only, not ingestion-order or model-training variance.
  If resources allow, a second stream seed for W1 is run and its R_w reported next to the primary.

## 7. Pilot and gates (before freezing)

**Pilot:** 1M rows, seed 20261301, the first 6 plan batches of each workload, validation panels only, arms D, S1 and
S4 (`pbs/static_vs_dli.pbs`, `MODE=pilot`).

**Gates, all required.** A PBS exit 0 alone is not acceptance.
1. Every batch's live set matches the ledger (the driver asserts it).
2. S ran its ordinary optimizer: at least one optimizer-built LMI segment per batch after batch 0. D published through
   its maintenance.
3. Known-answer gate (§3) on validation.
4. Both grids bracket 0.90 on validation in at least 4 of the 5 evaluated post-initial batches.
5. Measured resources project a primary walltime within the queue limit (24 h per job).

**Selection.** S* = whichever of S1 and S4 has the lower validation W(0.90) on W1 (calib panel). In a tie within 5%,
S4 is chosen, because it is closer to Qdrant's default multi-segment layout.

## 8. Stopping rules

- A primary job that fails a correctness gate (ledger, known-answer) is not analysed for H1. It is fixed, amended and
  rerun.
- A primary result is never re-run in the hope of a different outcome.

## Amendments before freezing (2026-11-23, after the pilot, before any test-panel data)

The pilot (jobs 25863307–25863310, validation panels only, seed 20261301; [PILOT_RESULTS.md](PILOT_RESULTS.md)) ran
the pipeline correctly. It exposed two flawed gate definitions and an uncontrolled confound. Each amendment below is
a **pre-data correction**.

- **A1. First-point rule.** When an arm's curve already meets mean R@30 = 0.90 at its first grid point, W is the work
  at that point, flagged as an *upper bound*, instead of "not attained". The pilot case: S4's unindexed Plain
  segments at shift batch 0 were exact search. Implemented in `analyze_static_vs_dli.py` (`interp_or_first`) and
  tested.
- **A2. Known-answer gate at full search.** Gate 3 is evaluated at an explicit full-search grid point: D budget
  2,000,000 (≥ any live set at 1M), S nprobe 100,000 (= all buckets). The criterion is mean R@30 ≥ 0.995 in every
  batch, reported as `known_answer_min` / `known_answer_pass`. The largest *tuning* grid point was never full search.
- **A3. Extended D grid.** It adds 113K, 160K, 226K and 320K, because D did not reach 0.90 at 80K in shift batch 5.
- **A4. Granularity-matched arm SG.**
  - **Why.** The pilot could not separate *lifecycle* from *granularity*. S's fixed bucket count gave 100–325 points
    per bucket, against D's about 5,000.
  - **What SG is.** S\*'s topology (one target segment, ordinary optimizers) with B = round(L_max / b) buckets, where
    L_max is the largest live count in the workload's plan and b = 5,000. A full segment then holds about b points
    per bucket, D's invariant for a full level, and its buckets grow as it fills, like a DLI level.
  - **Values at 1M:** insert-only 200, shift 183, constant_r and constant_o 20.
  - **New hypothesis H1g**, with the same thresholds and CI as H1: R_g = mean_t W_D / mean_t W_SG.
  - **How to read H1 and H1g together:**
    - H1 holds but H1g does not: the S advantage comes from granularity.
    - Both hold: the advantage comes from the lifecycle.
    - H1g reversed (D better than SG): the hierarchy helps at matched granularity.
  - **Multiplicity.** The primary family grows to 8 decisions (H1, H1g × 4 workloads). The Bonferroni sensitivity
    check uses 99.375% CIs.
- **S\* = S1**, by the preregistered rule (§7). S4 is dropped from later runs.
- **A second pilot** (`MODE=pilot2`, same seed 20261301, validation panels only, arms D, S1 and SG) checks gates 1–5
  with A1–A4 before freezing.
- **A5. Concurrent-service phase (H4) specified** (DESIGN.md §6; `dli_service_workload`, `summarize_service.py`).
  - **Load:** 20 QPS open-loop, 8 in flight, a 10 s timeout, writer 2,000 points/s in chunks of 1,000.
  - **Operating points:** locked on pilot2 validation results (§5) before any service run.
  - **H4 outcomes, descriptive per arm and workload:**
    - median and worst per-batch p50/p95/p99/max latency;
    - achieved QPS, timeouts and drops;
    - DLI exact-fallback share;
    - settled recall;
    - CPU seconds and peak RSS.
  - **Hard gate:** zero deleted-point violations.
