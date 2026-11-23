# Phase 3 Part B results: consolidation and calibrated allocation (insert-only, 1M)

**Verdict.** The frozen rules are in [PHASE3_PREREGISTRATION.md](PHASE3_PREREGISTRATION.md) §10.4, frozen at
`6538a4483`.
- **H9 not supported.** Consolidation (E1C, ρ = 0.25) needs **15.9% less work** than E1 to reach mean R@30 = 0.90,
  ratio **0.841** [0.831, 0.852]. The gain is real (the CI upper bound is < 1), but it is smaller than the
  preregistered 20% (ratio ≤ 0.80).
- **H10 not supported.** Calibrated level-aware allocation (E1L) needs **7.1% less work** than `mdn_code` on the same
  E1 topologies, ratio **0.929** [0.921, 0.936]. That is again real, but below the preregistered 10%.
- **H5 on insert-only (reported with H5): +0.54 pp** [0.47, 0.60] for always-retrain over router reuse at B\* = 50K,
  on the test panel. Under uniform data, router reuse is almost free. Under shift it cost 11.4 pp
  ([RESULTS_PHASE3_SHIFT.md](RESULTS_PHASE3_SHIFT.md)).
- **The 10M confirmation of E1C (§10.5) is not triggered,** because it was conditional on H9 being supported.

## Provenance

| Item | Value |
|---|---|
| Job | `25859945.pbs-m1.metacentrum.cz`, node haldan11, source `f370a0ef097e`, binary sha256 `43a4d6293756…` (cache key `4ae0671e1ae0`) |
| Run | E1, E1R and E1C complete: 21 batch records each (initial batch plus 20) and `summary.json`. **E1C50 (ρ = 0.5, descriptive) incomplete:** it started at about 19:50 of the 24 h walltime, and the author cancelled the job after E1C completed, because E1C50 could not finish. It is not reported |
| Consequence of cancelling | No in-job `analysis.json`, `COMPLETE.json` or `SHA256SUMS`. The arm directories were written directly to the archive and are complete |
| Analysis | Offline, with the committed `analyze_phase3.py` (`c42fde1ee`, unchanged since the freeze), on the `partb` and `test` rows extracted from the archive. Completeness check: every (batch, set, policy) has 13,000 rows (1,000 queries × 13 budgets) in all 21 batches. Output: [derived/phase3_partb_insert_only_25859945.json](derived/phase3_partb_insert_only_25859945.json) |
| Queries | Part B: rows 8000–8999 (unused by the diagnosis and by Part A). H5: test rows 9000–9999 |
| Archive | MetaCentrum `dli-emma-path-20261120/results/phase3-primary-insert_only-25859945.pbs-m1.metacentrum.cz/`. Local copy on C: `Research-Results\2026-11-20-dli-emma-path\phase3-primary-insert_only-25859945\`, verified against a sha256 list computed on storage |

## Results

| | E1 | E1L (calibrated) | E1C (consolidated) | E1CL (both, descriptive) |
|---|---|---|---|---|
| Mean W@0.90 over 21 batches | 22,725 | 21,106 | 19,115 | 18,155 |
| W@0.90 at the last batch (1M) | 27,597 | 28,785 | 22,532 | 21,061 |
| Ratio to E1 [95% CI] | — | **0.929** [0.921, 0.936] (H10) | **0.841** [0.831, 0.852] (H9) | 0.799 [0.789, 0.810] |
| Final levels | L1 10K, L2 440K, L3 550K | same topology as E1 | L1 50K, L2 55K, L3 895K | same topology as E1C |
| Points written / write amplification | 9.97M / 9.97 | — | 9.68M / 9.68 (0.97 × E1) | — |
| Fresh trainings / training seconds | 26 / 207 | — | 30 / 220 | — |
| Ingestion seconds | 237 | — | 255 | — |

E1R (always retrain): 204 trainings, 743 s (3.6× E1), the same 9.97M points written, 783 s of ingestion.

**Per-batch E1C / E1 ratio** (H9's snapshot dependence): 0.57, 0.46, 0.67, 0.74, **1.68**, 0.88, 0.86, 0.86, 0.87,
**1.35, 1.73**, 0.85, 1.00, 0.65, 0.69, **1.12**, 0.79, 0.71, 0.56, 0.54, 0.82 (batches 0–20).

**Per-batch E1L / E1 ratio:** 0.78–1.06, below 1 in 14 of 21 batches.

## Interpretation

1. **Consolidation works, but by less than the planner simulation suggested.**
   - It cuts work by 16% on average and writes 3% fewer points: consolidation replaces rewrites that Bentley–Saxe would
     do anyway, as `consolidation_sim.py` predicted.
   - The gain depends strongly on the snapshot, as MULTILEVEL_DIAGNOSIS.md predicted. It ranges from a 54% saving to a
     73% loss, with E1C worse in 4 of 21 batches (4, 9, 10, 15). *[Inferred, not yet checked against the router-hash
     history]* those are probably snapshots just after a merge, when a newly consolidated level holds data under an
     older router. A single-snapshot comparison would have shown anything
     from "consolidation halves the work" to "consolidation costs 73%".
   - The preregistered bar of 20% was not met, so H9 is not supported. The thesis reports a significant 16% gain that
     falls short of the effect size we committed to.
2. **Calibrated allocation gives a modest gain, as predicted.**
   - The diagnosis said selection could close at most the selection factor, and E1 already selects as efficiently as
     the static index. The 7% gain matches "modest".
   - It needs per-level isotonic calibration fitted on a calibration panel, so its engineering cost is real.
3. **The combination reaches about 20%, descriptively.** E1CL / E1 = 0.799, with a CI of [0.789, 0.810] that spans
   0.80. It was preregistered as descriptive (H11), so it is not a confirmatory result.
4. **Router reuse is free on uniform data and expensive under drift.** The cost is +0.5 pp here against +11.4 pp in
   Shift, for the same 3.6–3.9× training. Together with
   [SHIFT_MECHANISM.md](SHIFT_MECHANISM.md) (reuse fails through bucket imbalance), this is the case for retraining
   *only when drift is detected*.

## What this means for "DLI-best" (final-benchmark arm)

These are the candidate components, chosen by preregistered results only:
- **vacuum** (H6 supported, deletes);
- **consolidation** (H9: a real 16% gain, not supported at the 20% bar);
- **calibrated allocation** (H10: 7%, not supported at the 10% bar);
- the policy-replay winner (pending: jobs 25888669–25888673);
- an imbalance-triggered retraining rule (not yet tested).

The rule for assembling DLI-best must be written down **before** the policy-replay results are read. A proposal for
the author: include a component if its contrast has a 95% CI excluding no effect in the beneficial direction and
it does not hurt in any other decided workload. By that rule, consolidation and vacuum qualify. Calibrated allocation
also qualifies statistically, but it brings the calibration-panel machinery with it.

## Limits

- **One stream, one training draw.** The CIs cover query sampling only.
- **E1C50 is missing,** so the ρ sensitivity is unknown.
- **The oracle-cost decomposition per arm (H11)** is not computed here. Its inputs (`reference_costs.jsonl`) are in
  the archive, so it can be done offline.
