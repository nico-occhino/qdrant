# Preregistration: DLI-best allocation tie-break (calibrated against proportional)

**Status: DRAFT, 2026-11-24.** It freezes on the author's approval, before any job of this study runs. Later changes
are dated amendments that say whether they came before or after data.

**Why this exists.** The frozen DLI-best rule (`research/dli/emma_path/DLI_BEST_RULE.md`, on the Emma branch) says:
if the policy replay recommends `proportional`, a short, separately preregistered calibrated-vs-proportional replay
on the insert-only snapshots decides DLI-best's allocation.
- The policy replay recommended `proportional` (`research/dli/policy_replay/RESULTS_POLICY_REPLAY.md`, `21b58e1dc`).
- The author chose to run this replay on **DLI-best's own topology** (2026-11-24).

## 1. Design

- **Topology.** DLI-best on the insert-only workload: E1 plus consolidation ρ = 0.25 (`DLI_P3_VARIANT=consolidate250`,
  the Phase 3 E1C arm). Vacuum is part of DLI-best but never fires without deletes, so the insert-only topology is
  exactly E1C.
- **Plan.** The Phase 3 insert-only primary plan: 1M SISAP23 rows, seed 20261121, 1 + 20 batches. The job checks file
  by file that its regenerated plan is identical to the archived Phase 3 plan.
- **Policies,** replayed on identical frozen snapshots, with the same routers and budgets:
  - `calibrated`: per-level isotonic calibration, fitted on rows 1000–2999 (the `calib` panel) at every batch, as in
    Part B;
  - `proportional`: live-population quota per level;
  - `mdn_code`: the reproduction check only.
- **Panel.** `partb`, official queries 8000–8999 (1,000 queries), Part B's confirmatory panel for allocation.
  `val` (900–931) is descriptive.
- **Budgets.** 1K, 2K, 3.5K, 5K, 7K, 10K, 14K, 20K, 28K, 40K, 50K, 56K, 80K, 113K, 160K, 226K, 320K (the policy-replay
  grid).

## 2. What has already been seen (disclosure)

- **Calibrated is not new on this topology.** Part B (job 25859945) replayed it on E1C, on `partb`. Two numbers from
  it are known:
  - E1CL / E1 = 0.799 (descriptive);
  - E1CL / E1C (calibrated against `mdn_code`) = 0.950.
- **Proportional has never run on this topology,** and the calibrated-vs-proportional contrast has never been
  computed on any panel.

## 3. Estimand and decision rule

R = mean_t W_cal(t) / mean_t W_prop(t), where W(t) is the work (vectors scored) needed to reach mean R@30 = 0.90 on
`partb` at batch t. Within a batch it is linearly interpolated; the first-point rule applies.
- **CIs:** paired query bootstrap stratified by batch, 10,000 resamples, seed 20261125.
- **Labels and margin:** the same as the policy replay, with a 5% margin: *calibrated better*, *proportional better*,
  *equivalent*, *difference below the smallest effect of interest*, or *inconclusive*.
- **Choice of DLI-best's allocation (proposed, for the author to freeze):**
  - **`calibrated` only if the decision is *calibrated better*,** meaning R ≤ 0.95 with the CI upper bound < 1;
  - **otherwise `proportional`.**

  Reasons for the asymmetry:
  - `proportional` is the recommended policy across all five workloads of the policy replay, whereas calibrated
    allocation has evidence on insert-only only;
  - `proportional` needs no calibration panel or per-batch fit, so calibrated has to earn its extra machinery with a
    practically relevant gain.

  This rule is applied mechanically.
- **Descriptive:**
  - Δ R@30 at B\* = 50K;
  - per-batch R;
  - per-level selection and misses;
  - the `val` panel;
  - S5: `mdn_code` and `calibrated` against Phase 3 E1C at the 13 shared budgets. Investigate if the mean difference
    at B\* exceeds 1 pp.

## 4. Gates (all required; PBS exit 0 is not acceptance)

1. Ledger (live-set SHA after every batch, asserted by the driver).
2. Pairing (same queries for both policies in every batch and budget).
3. Monotone work.
4. Both policies reach 0.90 in at least 80% of batches; otherwise the result is descriptive, and the choice falls to
   `proportional` under §3.
5. All 21 batches evaluated.

## 5. Execution

- **Jobs.** `pbs/tiebreak.pbs`: first `MODE=check` (2 batches, a dry run that is not analysed), then `MODE=primary`.
- **Resources.** 8 CPUs, 96 GB, 100 GB scratch.
- **Walltime: 16 h.** Part B's E1C arm, with four policies on three panels, ran about 6.4 h of driver time on
  haldan11. Zenon nodes run about 2× slower.
- **Reuse.** The binary is reused from the cache (source key `2c16e4913ae0`); no Rust changes.
- **Stopping rule.** A gate failure is fixed and rerun, and the failure is recorded. A result is never rerun for a
  different outcome.
