# Preregistration: `mdn_code` versus `proportional` budget allocation in the DLI (policy replay)

**Status: DRAFT, 2026-11-23.** It freezes on the author's approval, before any job of this study runs. Later
changes are dated amendments that say whether they came before or after data. This study does not change the frozen
Phase 3 preregistration (`../emma_path/PHASE3_PREREGISTRATION.md`) or the static-vs-DLI draft.

**Origin.** Sommerová suggested comparing her `mdn_code` priority with population-proportional allocation on
different datasets and under distribution shift, and choosing the better one. This is **stage 1**: SISAP23
CLIP-768 under the Phase 3 workloads. Stage 2 (other datasets) needs a `DIM` parameter and a data converter, and gets
its own amendment.

## 1. What is compared

Both policies select whole buckets from one frozen E1 snapshot until a global candidate budget is met. The buckets,
routers, queries and gold are identical.

| Policy | Allocation across levels | Order inside a level | Buffer |
|---|---|---|---|
| `mdn_code` (Sommerová's released code) | global: bucket score P(bucket \| q) · B_level / a | descending P | score 1 (competes) |
| `proportional` (`dli_selection_frontier.rs::run_proportional`) | quota: budget · live(level) / live(all learned levels), at least 1 | descending P | always fully scored |

`mdn_code` multiplies every bucket of a level by the same constant, so the two policies order buckets *within* a level
identically. **The contrast isolates cross-level allocation.** The buffer rule also differs, but E1's buffer holds at
most b = 5,000 points, so its share of work is small; it is reported. This is not the reserved-per-level policy.

## 2. Hypotheses

The primary estimand per (workload, panel) is

R = mean_t W_prop(t) / mean_t W_mdn(t),

where W(t) is the work (vectors actually scored) needed to reach mean test R@30 = 0.90 at snapshot t. It is found by
linear interpolation over the budget grid. The mean runs over snapshots where both policies attain the target. If a
curve meets the target at its first grid point, W is that point's work, flagged as an upper bound (as in amendment A1
of the static study).

**Smallest effect of interest: 5% of work.** At the operating point, 1 pp of R@30 costs about 8% of work
(`PHASE3_PREREGISTRATION.md` §5), so 5% is about 0.6 pp of recall at equal work. The prior point estimate at 10M was
8.3% (`RESULTS_10M.md`: 74,819 against 81,580; a single replay with no CI).

| Outcome | Rule (95% CI of R) |
|---|---|
| `proportional` better | R ≤ 0.95 and CI upper bound < 1 |
| `mdn_code` better | R ≥ 1.0526 (= 1/0.95) and CI lower bound > 1 |
| equivalent | CI inside [0.95, 1.0526] |
| difference below the smallest effect of interest | CI excludes 1, R inside (0.95, 1.0526), CI not inside the margin |
| inconclusive | anything else |

**The primary family has 5 decisions:**

| Workload | Panel | What it probes |
|---|---|---|
| insert-only | `test` | uniform growth |
| constant_r | `test` | random deletes, dead postings |
| shift | `mq` | the drifting cluster (stale routers) |
| shift | `ids` | in-distribution queries during drift |
| shift | `ood` | a withheld cluster, out-of-distribution queries |

There is no multiplicity correction. A Bonferroni sensitivity analysis (99% CIs) is reported next to the decisions.

**Choosing a policy (Sommerová's "choose the best"):**
- A policy is **recommended** if it is better or equivalent in all 5 decisions, and better in at least one.
- Otherwise the per-workload winners are reported, and no single policy is recommended.
- "Equivalent everywhere" recommends `mdn_code`, the reference design.

**Secondary analyses, descriptive with CIs where they are defined:**
- **S1.** Δ mean R@30 (proportional − mdn_code) at B\* = 50K, averaged over batches; the Phase 3 Part A estimand.
- **S2.** Per-batch R(t), to see whether any advantage is concentrated mid-stream in shift, where routers are stalest.
- **S3. Level diagnostics at B\*:** mean buckets selected per level, and the share of missed top-30 gold points by
  level (from `unselected_level_rank`). This tests the mechanism: under stale routers, does `mdn_code` under-spend on
  the level that holds the drifting data?
- **S4.** The buffer's share of `proportional` work at the target.
- **S5. Reproduction check.** This study's `mdn_code` rows are compared with the Phase 3 primary E1 rows of the same
  seed and plan, at the 13 shared budgets (jobs 25859945 insert-only, 25859940 constant_r, 25859944 shift). It reports
  the share of identical per-query R@30 and the mean difference. Router training uses several CPUs and may not be
  bit-reproducible, so this check is descriptive, not a gate. A mean-recall difference above 1 pp at B\* is
  investigated before the H decisions are reported.

## 3. Data, plan, queries

- **Corpus:** SISAP23 LAION2B CLIP-768 Float16, sha256 `f5868d6e…`, the first 1,000,000 rows.
- **Plan:** `../emma_path/phase3_plan.py`, seed 20261121, 1 + 20 batches for the uniform workloads and 20 batches for
  shift. These are the Phase 3 primary plans.
- **Queries:** the Phase 3 test panels, 1,000 each.
  - Uniform workloads: official queries 9000–9999, with validation 900–931.
  - Shift: `mq`, `ids` and `ood`, each with a 32-query validation panel.
- **Panel reuse, stated up front.** Phase 3 analysed `mdn_code` on these panels. `proportional` has never been run
  on them, and no choice in this study was made on them.
- **Gold:** exact top-30 over the live set, float64 cosine of the stored Float16 rows, ties broken by smaller ID.
  - Known tie artefact (`../static_qdrant_vs_dli/PILOT2_RESULTS.md`, addendum): exact-duplicate rows at rank 30 can
    cost recall for a correct answer. It affects both policies.
  - The `proportional` merge breaks ties by smaller ID, as gold does; `mdn_code`'s resolution may not.
  - The queries that differ between the policies at the largest budget are checked offline for rank-30 ties, with
    `../static_qdrant_vs_dli/known_answer_diag.py`.

## 4. Grid

Budgets: 1K, 2K, 3.5K, 5K, 7K, 10K, 14K, 20K, 28K, 40K, 50K, 56K, 80K (the Phase 3 grid), plus 113K, 160K, 226K and
320K. The extension is there because E1 did not reach 0.90 at 80K in shift (static-vs-DLI pilot, batch 5). Both
policies use the same grid.

## 5. Statistics

- **Bootstrap:** paired query bootstrap, 10,000 resamples, seed 20261124. Resampling is stratified by batch: the same
  resampled query indices for both policies within a batch.
- **What the CI covers.** The batch-mean is one trajectory statistic. The CIs cover query sampling only, not
  ingestion order or training variance. Both policies read the same snapshot, so training variance does not enter
  the contrast itself, but its generality is limited to this one training draw.

## 6. Gates, all required

A PBS exit 0 is not acceptance.
1. **Ledger.** The live set matches the plan after every batch; the driver asserts it.
2. **Pairing.** In every analysed (batch, budget), both policies answered the same queries.
3. **Monotone work.** Mean work is non-decreasing in budget for each policy and batch.
4. **Attainment.** Both policies reach 0.90 within the grid in at least 80% of batches. Otherwise R is reported as
   descriptive.
5. **Completeness.** All batches are evaluated within walltime.

## 7. Execution

- **Jobs:** `pbs/policy_replay.pbs`, `MODE=primary`, one job per (workload, panel): `insert_only`, `constant_r`,
  `shift` with `PANEL=mq`, `PANEL=ids` and `PANEL=ood`.
- **What a job runs:** one E1 ingestion of the plan, with each frozen snapshot replayed under both policies
  (`DLI_FRONTIER_POLICIES=mdn_code,proportional`).
- **Resources and walltime.** The projection comes from Phase 3 E1 replay times (shift: 19,596 s for three policies,
  six panels and 13 budgets), scaled by the extended grid: about 2–3 h per job. Requested: 8 h, 8 CPUs, 96 GB.
- **Stopping rule.** A job that fails a gate is fixed and rerun, and the failure is recorded. A result is never rerun
  in the hope of a different outcome.
