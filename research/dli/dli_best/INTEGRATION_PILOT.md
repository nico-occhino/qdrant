# DLI-best integration pilot (validation panels only)

**Status: RUN, 2026-11-25 (jobs 25942585–25942587).** The criteria below were fixed before any run and are not
edited. Results: [RESULTS_INTEGRATION_PILOT.md](RESULTS_INTEGRATION_PILOT.md). Criteria 1, 2 and 4 pass on all three
workloads. Criterion 3 fails on insert-only and shift, through storage rounding shared by every policy.

**What DLI-best is** (`research/dli/emma_path/DLI_BEST_RULE.md`, final): E1 + vacuum above 20% dead + consolidation
ρ = 0.25 + `PopulationProportional` allocation.

**Purpose.** The components were tested one at a time, and allocation only through an offline replay control. This
pilot answers three questions before the final benchmark:
1. Does the combined configuration run correctly on all three workloads?
2. Does the **native** proportional policy (commit `c8aa74023`) reproduce the offline control on real snapshots?
3. Is DLI-best not worse than E1, descriptively?

It tests no hypothesis, and **no test panel is read**.

## Design

| Workload (Phase 3 primary plan, seed 20261121) | Panels (validation) | Comparison run (Phase 3, same plan and queries) |
|---|---|---|
| insert-only | `val` (32) | E1, job 25859945 (`insert_only-E1`) |
| constant-R (random deletes) | `val` (32) | E1, job 25859940 (`constant_r-E1`) |
| shift | `mq_val`, `ids_val`, `ood_val` (32 each) | E1, job 25859944 (`shift-E1`) |

- **One job per workload.** `DLI_P3_VARIANT=dlibest`.
- **Replayed policies:** `proportional_native` (DLI-best as served), `proportional` (the offline control) and
  `mdn_code`.
- **Budgets:** the policy-replay grid, plus 2,000,000 (full search, the known-answer check).
- **Resources:** 8 CPUs, 96 GB, 100 GB scratch, 6 h walltime. Validation panels are small, so ingestion dominates.

## Pass criteria (all required)

1. **Ledger.** The live-set SHA matches the plan after every batch, and all batches are evaluated.
2. **Native equals control.** For every (batch, panel, query, budget), `proportional_native` and `proportional` score
   the same number of vectors (`scored_total`). Every row where Recall@30 differs must be explained by an exact score
   tie at rank 30 in the Float64 gold. The tie check is done offline with
   `research/dli/static_qdrant_vs_dli/known_answer_diag.py`.
   - *Amended 2026-11-25, before any pilot run, after the local smoke test.* The draft also required recall to be
     identical in ≥ 99.9% of rows. On the 60K constant-R smoke plan, one query had 11 byte-identical rows tied at
     the rank-30 score. There the native resolution, which `mdn_code` uses too, returned other tied copies than gold's
     smaller-ID rule: 6 of 896 rows, 0.67%. The work was identical in all 896 rows. A fixed floor would fail on one
     duplicate-heavy query, so the tie-explanation rule alone decides.
3. **Known answer.** At the full-search budget, mean R@30 ≥ 0.995 per panel and batch. This is reported raw and with
   exact rank-30 ties excluded, as in the static-vs-DLI pilot 2 addendum.
4. **Mechanisms are active:**
   - in constant-R, at least one vacuum publication (no sources, buffer untouched), and no learned level above 20%
     dead after settling;
   - in insert-only or shift, the final topology differs from E1's in the way consolidation predicts: the deepest
     level holds more, and the levels above it hold at most ρ of it.

**Descriptive, with no gate:**
- W@0.90 for DLI-best (`proportional_native`) over W@0.90 for E1 (`mdn_code`, Phase 3), paired by query, per
  workload and panel, with batch-stratified bootstrap CIs. With 32-query panels these are rough.
- **If any point ratio exceeds 1.05, the pilot stops for discussion before the final benchmark is designed.**

## What it does not decide

Nothing about DLI-best's superiority. That is a final-benchmark question, on test panels, under its own
preregistration.
