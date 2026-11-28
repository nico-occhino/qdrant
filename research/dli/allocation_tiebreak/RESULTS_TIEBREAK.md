# DLI-best allocation tie-break: results

**Status: FINAL, 2026-11-25.** The rules are in [PREREGISTRATION.md](PREREGISTRATION.md), frozen at `e8f168ed0`.

**Decision: equivalent, so the frozen rule chooses `proportional` allocation for DLI-best.**
- On DLI-best's insert-only topology (E1 + consolidation ρ = 0.25), calibrated allocation needs **3.75% more** work
  than proportional to reach mean R@30 = 0.90 on `partb`.
- R = W_cal / W_prop = **1.0375** [1.027, 1.047], Bonferroni 99% [1.024, 1.049].
- The CI lies inside the ±5% equivalence margin, so the decision is *equivalent*. The frozen rule chooses `calibrated`
  only if it is *better*.

## Provenance and gates

| Item | Value |
|---|---|
| Job | `25931720.pbs-m1.metacentrum.cz`, alfrid2, exit 0, 4:54:53 of 20:00, 37 GB peak memory; `COMPLETE.json` present; analysis files verify against `SHA256SUMS` |
| Source | commit `e8f168ed0` (frozen preregistration), cached binary `b1a7ad70…` (source key `2c16e4913ae0`) |
| Plan | identical to the Phase 3 insert-only primary plan (130/130 files); 21/21 batches; ledger asserted per batch |
| Gates | pairing, monotone work, attainment (21/21 for both policies), completeness: all pass; no bootstrap resample failed to reach the target |
| S5 reproduction | `mdn_code` **and** `calibrated` rows are 100% identical to the Phase 3 E1C run (job 25859945) at the 13 shared budgets |
| Check job | 25914476 (2 batches, identity checks only), passed; see [PBS_MANIFEST.json](PBS_MANIFEST.json) |
| Extracts | [derived/](derived/) |
| Archive | MetaCentrum `dli-emma-path-20261120/results/alloc-tiebreak-primary-insert_only-25931720.pbs-m1.metacentrum.cz/`; local copy `Research-Results/2026-11-24-allocation-tiebreak/` |

## Results

| Panel | R (cal / prop) | 95% CI | Decision | Mean W@0.90, cal / prop | Δ R@30 at 50K (cal − prop) |
|---|---|---|---|---|---|
| **`partb` (primary, 1,000 queries)** | **1.0375** | [1.027, 1.047] | **equivalent** | 18,155 / 17,498 | −0.34 pp [−0.38, −0.30] |
| `val` (descriptive, 32 queries) | 0.914 | [0.860, 0.969] | calibrated better (descriptive only) | 9,771 / 10,696 | +0.14 pp [−0.04, +0.33] |

- **Per-batch R on `partb`:** 0.97–1.11; calibrated is cheaper in only 5 of 21 batches.
- **The two panels point in opposite directions.** The preregistered decision uses `partb`, which has 1,000 queries,
  a CI width of 0.02, and is disjoint from both the calibration panel and every earlier analysis.
  - `val` has 32 queries, a CI width of 0.11, and its per-batch ratios swing from 0.71 to 1.10.
  - A small panel's direction is not evidence against the large one. This is reported, not interpreted.
- **Level diagnostics at B\* (S3):** calibrated buys twice as many level-1 buckets (2.8 against 1.4) and more level-2
  buckets (14.3 against 12.5), with about the same number of level-3 buckets.
  - Its misses fall in level 1 (0.08 against 0.18) but rise in levels 2 and 3 (0.42 / 0.38 against 0.34 / 0.27).
  - Under consolidation most data sit in level 3, so a fitted value-per-cost ranking that favours the small levels
    does not pay off. This is descriptive.

## What this means

1. **DLI-best is now fixed:** E1 + vacuum (> 0.2 dead) + consolidation ρ = 0.25 + **proportional** allocation. It is
   recorded in `research/dli/emma_path/DLI_BEST_RULE.md`.
2. **Calibrated allocation's earlier gain was over `mdn_code`, not over proportional.** Part B H10 measured 7% less
   work for calibrated than for `mdn_code` on E1.
   - The policy replay found proportional 9% cheaper than `mdn_code` on insert-only.
   - Here, on DLI-best's topology, calibrated and proportional are equivalent, with proportional slightly cheaper.
   - Consistent with this, the calibration machinery recovers about what simple population allocation already
     gives.
3. **A simpler DLI-best.** Proportional needs no calibration panel and no per-batch isotonic fit.

## Limits

- **One stream and one training draw.** The CIs cover query sampling only.
- **Insert-only only.** The policy replay already established proportional's behaviour in the delete and shift
  workloads.
- **Section 3 is descriptive interpretation.**
