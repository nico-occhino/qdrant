# Policy replay results: `mdn_code` against `proportional` allocation (interim, 4 of 5 decisions)

**Status: INTERIM, 2026-11-24.** Four of the five preregistered decisions are made. The fifth (Shift `ood`) failed
on a harness race and is being rerun from the same frozen inputs (job 25908996). **The overall recommendation waits
for it**, as the frozen rule requires all five. The rules are in [PREREGISTRATION.md](PREREGISTRATION.md), frozen
at `3c0ae9b9a`.

## Decisions

R is mean work for `proportional` over mean work for `mdn_code`, where work is the number of vectors scored to reach
mean R@30 = 0.90. Per batch it is interpolated, then averaged over batches. CIs come from a paired query bootstrap
stratified by batch.

| Workload / panel | R (prop / mdn) | 95% CI | 99% CI (Bonferroni) | Decision | Bonferroni | Δ R@30 at B\* = 50K (prop − mdn) |
|---|---|---|---|---|---|---|
| Insert-only / `test` | **0.908** | [0.900, 0.915] | [0.898, 0.917] | **proportional better** | same | +0.41 pp [0.36, 0.46] |
| Constant-R / `test` | **0.784** | [0.776, 0.792] | [0.773, 0.795] | **proportional better** | same | +0.70 pp [0.65, 0.75] |
| Shift / `mq` (drifting cluster) | **1.023** | [1.015, 1.031] | [1.012, 1.033] | **equivalent** (inside ±5%) | same | +1.30 pp [1.13, 1.47] |
| Shift / `ids` (in-distribution) | **0.887** | [0.874, 0.901] | [0.871, 0.906] | **proportional better** | same | +1.92 pp [1.77, 2.06] |
| Shift / `ood` (withheld cluster) | pending | | | rerun 25908996 | | |

**What the recommendation rule gives:**
- proportional is better or equivalent in all four decided workloads, and better in three of them;
- if `ood` comes out better or equivalent, **proportional is recommended**;
- if `ood` favours `mdn_code` or is inconclusive, no single policy is recommended, and winners are reported per
  workload.

Validation panels (32 queries, descriptive) point the same way:
- insert-only 1.00 (inconclusive);
- constant-R 0.80;
- `mq_val` 0.98 (inconclusive);
- `ids_val` 0.94.

## Gates and provenance

| Job | Node | Walltime | Gates (ledger, pairing, monotone work, attainment, completeness) | S5: `mdn_code` against Phase 3 E1, 13 shared budgets |
|---|---|---|---|---|
| 25888669 shift `mq` | alfrid4 | 3:20 | all pass, 20/20 attained | **100% identical** (260 cells) |
| 25888670 shift `ids` | tyra8 | 3:18 | all pass, 20/20 | 99.93% identical, mean Δ 0.0008 pp, max cell 0.09 pp |
| 25888672 insert-only | zenon32 | 9:53 (extended to 16 h) | all pass, 21/21 | 99.96%, mean Δ 0.001 pp, max cell 0.06 pp |
| 25888673 constant-R | zenon43 | 3:59 of 4:00 | all pass, 21/21 | 99.97%, mean Δ 0.00004 pp, max cell 0.19 pp |
| 25888671 shift `ood` | zenon32 | 7:44, **exit 101** | failed at batch 18: `copy_dir` hit a file that vanished during the frozen-copy step (`NotFound`, `dli_sisap_scaling_benchmark.rs:2533`). Batches 0–17 are archived and not used | — |

- **Shared provenance.** All jobs ran commit `3c0ae9b9a`, binary `b1a7ad70…` (cached), with plans identical to the
  Phase 3 primary plans (205/130/86 files). The analysis files verify against each job's `SHA256SUMS`. Extracts are
  in [derived/](derived/).
- **S5 (descriptive).** Its preregistered trigger was "investigate if the mean recall difference at B\* exceeds 1 pp".
  The largest mean difference is 0.001 pp, so the trigger is not reached. The exact reproduction on alfrid4 against
  tiny per-cell differences elsewhere suggests node-dependent floating point in builds **[I]**. It does not affect
  the contrast, because both policies read the same snapshot.
- **The `ood` failure is a pre-existing harness race,** not a measurement problem. The same copy ran about 100 times
  without error across the other jobs and Phase 3. Following the stopping rule ("fixed and rerun"), the job was
  resubmitted unchanged, with a 16 h walltime.

## Mechanism: where the budget goes (S3, at B\* = 50K, mean per query over batches)

| Workload | Policy | Buckets selected per level (L1 / L2 / L3) | Missed top-30 gold per query, by level (L1 / L2 / L3) |
|---|---|---|---|
| Insert-only | mdn | 6.6 / 7.8 / 20.2 | 0.01 / 0.72 / 0.39 |
| | prop | 1.6 / 12.6 / 33.1 | 0.21 / 0.56 / 0.23 |
| Constant-R | mdn | 7.6 / 22.1 / – | 0.01 / 0.55 / – |
| | prop | 3.0 / 32.4 / – | 0.14 / 0.21 / – |
| Shift `ids` | mdn | 4.9 / 7.8 / 10.3 | 0.21 / 2.39 / 0.92 |
| | prop | 2.2 / 9.8 / 31.1 | 0.22 / 2.49 / 0.23 |
| Shift `mq` | mdn | 3.5 / 6.2 / 10.1 | 0.56 / 4.16 / 1.11 |
| | prop | 2.0 / 7.7 / 31.7 | 0.66 / 4.48 / 0.30 |

1. **`mdn_code` over-spends on the smallest, newest level.**
   - Its priority P(bucket) · B_level / a gives a small level's buckets high scores, so it buys 3–8 level-1 buckets.
     Level 1 holds at most 50K points.
   - Proportional buys about 2–3 level-1 buckets and three times as many level-3 buckets, so its misses in the
     largest level fall by 40–75% (insert-only 0.39 → 0.23, constant-R 0.55 → 0.21, `ids` 0.92 → 0.23, `mq`
     1.11 → 0.30).
   - That is the whole gain in the uniform and delete workloads.
2. **Under drift (`mq`), the problem is not allocation.** Most misses sit in level 2, whose reused router collapses
   the drifting cluster into a few oversized buckets ([SHIFT_MECHANISM.md](../emma_path/SHIFT_MECHANISM.md), on the
   Emma branch). Neither policy can buy its way out of that, so they come out equivalent.
3. **Proportional loses when the relevant data are in a small, recent level.** Its per-batch ratio on `mq` is worse at
   batches 18–19 (1.50, 1.79). That is when the queries' own cluster arrives and lives in the newest small levels,
   which proportional starves by design. On `mq` it wins in 12 of 20 batches and loses most at batches 18–19.

   The trade-off is that population-proportional allocation is right when relevance follows population, and
   population-blind priority is right when the newest data matter most. This is descriptive and post hoc; no
   hypothesis was preregistered on it.

## Per-batch ratios (S2)

| Workload | Range | Batches with R < 1 |
|---|---|---|
| Insert-only | 0.76 – 1.04 | 17 / 21 |
| Constant-R | 0.70 – 1.00 | 17 / 21; R = 1.00 exactly in 4 batches, probably those where only one learned level is occupied **[I]** |
| Shift `ids` | 0.62 – 1.02 | 17 / 20 |
| Shift `mq` | 0.59 – 1.79 | 12 / 20 |

## Consequence for DLI-best (frozen rule, `emma_path/DLI_BEST_RULE.md`)

- **If proportional is recommended once `ood` is in,** the frozen tie-break runs a short, separately preregistered
  replay of calibrated allocation against proportional on the insert-only snapshots, and that decides DLI-best's
  allocation.
- **Otherwise** calibrated allocation stays.

The 10M point estimate that motivated this study (8.3% less work for proportional) is consistent with the 1M
insert-only result (9.2%).

## Limits

- **One stream per workload and one training draw.** The CIs cover query sampling only.
- **The `mq` panel is the same 1,000 queries in every batch.** The batch-stratified bootstrap treats batches as
  strata of one trajectory.
- **Section 3's interpretation is exploratory.**
