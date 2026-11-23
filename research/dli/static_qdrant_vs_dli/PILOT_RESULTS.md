# Pilot results: static Qdrant LMI lifecycle (S1, S4) vs DLI E1 (D)

**Status: COMPLETED, not ACCEPTED.**
- The pipeline gates pass.
- Two gate definitions in the draft preregistration turned out to be flawed (§3). They must be amended before
  freezing.
- These are **validation-panel** numbers from an independent seed (20261301). They are used to select S* and to plan
  the primary run. They are not the study's result.

## 1. Provenance

| Workload | Job | Exit | Walltime | Peak mem |
|---|---|---|---|---|
| W1 insert-only | 25863307 | 0 | 1:54:56 | 40 GB |
| W2 shift | 25863308 | 0 | 1:26:45 | 49 GB |
| W3 constant_r | 25863309 | 0 | 0:38:47 | 24 GB |
| W4 constant_o | 25863310 | 0 | 0:37:04 | 39 GB |

- **Source.** Commit `301319ada` (branch `research/static-vs-dli-20261122`). Source bundle sha256 `9b845fe6…`,
  `inputs.sha256` sha256 `1bce6774…`.
- **Setup.** N = 1M, the first 6 plan batches. Validation panels only: `calib` (2,000 queries) for W1, `mq_val` (32)
  for W2, `val` (32) for W3 and W4. Arms D (E1, `mdn_code`), S1 (1 segment, 1,000 buckets) and S4 (4 segments,
  250 buckets), each with a 20,000 KB indexing threshold.
- **Evidence.** Remote `dli-emma-path-20261120/results/static-vs-dli-pilot-<workload>-<job>/`; local copy
  `C:\Users\nicoo\Workspace\Research-Results\2026-11-22-static-vs-dli\`.

## 2. Work at mean validation R@30 = 0.90 (`analyze_static_vs_dli.py`, in-job)

| Workload | W_D / W_S1 [95% CI] | W_D / W_S4 [95% CI] | Attained (both arms) |
|---|---|---|---|
| W1 insert-only | **4.25** [4.12, 4.39] | **2.52** [2.46, 2.58] | 6/6, 6/6 |
| W2 shift | 2.57 [2.06, 3.32] | 1.87 [1.54, 2.16] (descriptive) | 5/6, 4/6 |
| W3 constant_r | 6.13 [4.92, 7.60] | 2.76 [2.30, 3.21] | 6/6, 6/6 |
| W4 constant_o | 7.91 [6.40, 9.74] | 3.36 [2.76, 3.92] | 6/6, 6/6 |

Per-batch W, as (S1, D):
- **W1:** batch 0 (2,922, 17,445) up to batch 5 (6,124, 31,102).
- **W3:** S1 about 1.2–1.6K, D 5.5–12K.

**In every workload, the ordinary static lifecycle reaches the target with much less search work than the DLI.**

**Likely mechanism, [I] inferred and not yet tested:**
- **S partitions far more finely.** S keeps 1,000 (or 250) buckets per segment whatever the population: about 100
  points per bucket at W3/W4's 100K live points, and about 325 at W1's batch 5.
- **D's design keeps a full level's buckets near b = 5,000 points.** This is the same structural factor as in the 10M
  diagnosis (`emma_path/MULTILEVEL_DIAGNOSIS.md`).

The pilot does not separate *lifecycle* from *granularity*.

## 3. Gates

| Gate | Result |
|---|---|
| 1. Ledger matches after every batch | **pass** (asserted per batch; all jobs exit 0) |
| 2. S's ordinary optimizer runs; D publishes | **pass**: S1 ends with one 1,000-bucket LMI segment and S4 with three 250-bucket segments; D made 54–65 publications |
| 3. Known-answer: R@30 ≥ 0.995 at the maximum grid point | **not evaluable as defined.** The largest grid point is not full search: D's 80K budget is < 25% of 325K live points, and S1's nprobe 256 is 25% of its buckets. Observed maximum-grid R@30: D 0.975–0.998, S1 0.925–0.999, S4 0.996–1.000. **Amendment needed:** add an explicit full-search point (D budget ≥ live count; S nprobe = all buckets) and apply the gate there |
| 4. Grids bracket 0.90 on validation in ≥ 4/5 post-initial batches | **fails for W2.** D does not reach 0.90 at 80K in shift batch 5 (0.872). Separately, S4's "not attained" at shift batch 0 is an artefact: its ~44K points were still unindexed Plain segments (exact), so recall starts above the target at the first grid point. **Amendments needed:** extend D's grid (113K, 160K, 226K, 320K, full); report "attained at the first grid point" as an upper bound W ≤ work(first point) instead of not attained |
| 5. Projected primary walltime within 24 h per job | **pass**: 37 min – 1 h 55 min for 6 batches × 3 arms; 21 batches × 2 arms projects to about 2–5 h per workload |

## 4. Maintenance cost (6 batches, from `build-events.jsonl` and `native-events.jsonl`)

| Workload | Arm | Builds (fresh / reused) | Training s | Optimizer wall s | Points rewritten | Settle s |
|---|---|---|---|---|---|---|
| W1 | D | 65 (7 / 58) | 13 | 38 | 2,475,000 | 0 (synchronous) |
| W1 | S1 | 12 (12 / 0) | 218 | 236 | 1,365,000 | 239 |
| W1 | S4 | 13 (13 / 0) | 78 | 85 | 653,744 | 90 |
| W3 | D | 65 (7 / 58) | 11 | 26 | 2,148,400 | 0 |
| W3 | S1 | 12 | 187 | 195 | 666,973 | 199 |
| W3 | S4 | 23 | 76 | 81 | 651,973 | 88 |

**The trade-off is real and runs the other way for maintenance:**
- D trains 15–20× less, because router reuse means 58 of 65 builds train nothing.
- But D rewrites 2–4× more points.
- S spends its maintenance on full retrains.

## 5. S* selection (preregistered rule, §7: the lower validation W on W1)

S1's W (2.9–6.1K) is below S4's (5.1–10.6K), more than 5% apart, so **S\* = S1**.

Note for the decision: S1 is also the more expensive S in maintenance (218 s against 78 s of training). As configured
it would not scale to 10M: a single segment of up to 2.7M points, retrained about every 13K inserts.

## 6. What the pilot implies for the primary run

1. Amend the gates and analysis (§3) **before freezing**. This is still a draft and no test data exist.
2. **Decision for the author: separate lifecycle from granularity.** Add a *granularity-matched* S arm, with buckets
   sized so that a full segment has about b = 5,000 points per bucket, as D's levels do. Without it, a primary result
   "S beats D" cannot say whether the hierarchy or the bucket size is responsible.
3. The concurrent-service phase (H4) is still not implemented (DESIGN.md §6).

**Decision (author, 2026-11-23):** add the granularity-matched arm SG (PREREGISTRATION.md, Amendment A4), and amend the gates (A1–A3).
