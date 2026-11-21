# Emma path — 300K pilot results (job 25813227)

**Verdict: the pilot gate passes, so 10M is submitted.** The gate is engineering-only
(PREREGISTRATION §4). The numbers below are descriptive: 300K cannot test the 10M
hypotheses.

## Provenance

| Item | Value |
|---|---|
| Job | `25813227.pbs-m1.metacentrum.cz`, node `eluo1-3`, 16 CPU / 128 GB, exit 0, walltime 00:44:21 |
| Source | `eb20276f906f` (clean tree, verified in-job); binary sha256 `f9a4dd58…` |
| Data | first 300,000 rows of the verified 10,120,191-row corpus (`f5868d6e…`) |
| Gold | exact prefix gold from `prefix_gold.py`. In preflight 25801978 the same tool reproduced the official SISAP23 top-30 on the full corpus for **every** row (val 32/32, test 256/256). |
| Queries | validation rows 900–931 (32), test rows 9000–9255 (256) |
| Evidence | remote `dli-emma-path-20261120/results/pilot-25813227…/`; local copy `C:\Users\nicoo\Workspace\Research-Results\2026-11-20-dli-emma-path\pilot-25813227\`. All `SHA256SUMS` verify except `job.log`, the known case where `tee` appends the `COMPLETE` line after hashing (verified: hash of the file minus its last line matches). |

## Gate (§4, E1)

| Criterion | Result |
|---|---|
| 1. Every publication succeeds; one build per publication; outputs = plans | pass (60/60) |
| 2. Collection pass learned 100/100, 0 exact fallbacks | pass |
| 3. Reuse builds keep the target router with 0 training s; fresh builds have aⁱ buckets | pass (53 reuse, 7 fresh) |
| 4. Projected 10M E1 ingestion ≤ 40 h | pass: **1.23 h** (2,024 publications, 206 trainings) |

Fit behind criterion 4: c₀ = 0.011 s, c₁ = 9.17 µs per output point, training
2.86×10⁻¹¹ s per k-means/MLP FLOP. The projected final levels [0, 220K, 4.4M, 5.5M] equal
the planner simulation.

## Topologies

| Arm | Members (live / buckets / largest bucket) |
|---|---|
| G0 | L1 250,000 / 548 / 2,200; Plain 50,000 |
| E1 | L2 275,000 / 100 / 6,884; L1 25,000 / 10 / 5,553; Plain empty |
| E1R | L2 275,000 / 100 / 6,190; L1 25,000 / 10 / 4,223; Plain empty |

## Maintenance

| Arm | Publications | Trained / reused builds | Training s | Optimizer wall s | Ingestion s | Rewritten points / inserted |
|---|---|---|---|---|---|---|
| G0 | 1 | 1 / 0 | 37.0 | 42.0 | 46.9 | 0.83 |
| E1 | 60 | 7 / 53 | **11.0** | 32.5 | **42.3** | 7.58 |
| E1R | 60 | 60 / 0 | 173.3 | 197.6 | 206.0 | 7.58 |

Router reuse removes 94% of the training time (11 s vs 173 s) and makes ingestion 4.9×
faster than retraining with the same structure. It does **not** reduce copying. Qdrant
segments are immutable, so every publication still rewrites the whole target level
(7.58× at 300K for both E arms). Reuse saves training, not writes.

## Selection work (test, mean R@30 = 0.90, linear interpolation)

| Policy | G0 | E1 | E1R |
|---|---|---|---|
| slot | 62,426 | 24,248 | 21,484 |
| mdn (thesis) | 59,377 | 24,248 | 21,484 |
| **mdn_code** (her code) | 59,868 | **24,248** | 21,484 |
| density | 59,880 | 24,961 | 22,196 |
| tdensity2 | 49,079 | 23,141 | 22,141 |
| proportional | 55,771 | 21,522 | 20,753 |
| static_flat | **8,145** (548 buckets, G0 model) | 15,506 (110 buckets, E1 model) | 15,506 |

Plain share at the target is 80–90% for G0 and 0% for E1/E1R (empty final buffer).

Paired query bootstrap (10,000 resamples, seed 20261120):
- **E1 `mdn_code` vs G0 `slot`:** 61.2% less work [53.3, 67.0].
- **E1 `mdn_code` vs matched static (110 buckets):** E1 needs 56% *more* work (saving −0.564 [−0.795, −0.366]).

## Interpretation (descriptive, 300K)

1. **The Plain-buffer penalty disappears by construction.** The E1 buffer is at most
   b = 5K, empty in this snapshot, and 0.19K at 10M. G0's 50K buffer is 17% of this
   collection, and it alone explains most of G0's work. The 61% gap is therefore mostly
   the G0 buffer at this scale, not evidence for H1. H1 is decided only at 10M, where
   G0's buffer is 1.2%.
2. **Her design does not close the gap to a static index.** The matched-bucket static
   reference needs 36–80% less work than E1. The 548-bucket static LMI with the G0 model
   (512 hidden, 30 epochs) is the best index here (8.1K), below every DLI arm. Finer
   partitions and a larger model matter more than the hierarchy rule.
3. **Reuse costs about 13% work** (E1 24.2K vs E1R 21.5K at the target). In exchange,
   training drops 16× and ingestion 4.9×. This is her design's trade-off, now measured
   in native Qdrant.
4. **Normalisation choice does not matter in E1 at this snapshot.** `slot`, `mdn` and
   `mdn_code` give identical work. With an empty buffer and two levels, the three scores
   only rescale levels relative to each other, and on this budget grid they select the
   same buckets.

## Observations, not gated

- **G0 collection pass: 10/100 exact fallbacks.** These are not a defect introduced
  here. With one learned level, `slot` scores level-1 buckets P·2 against Plain's 1.0;
  whenever the best bucket has P < 0.5 the 50K Plain unit ranks first. It alone
  exceeds the 2,000 budget, so the plan holds no learned bucket and the dispatcher
  rejects it (`no_learned_bucket`). Each such query scored 50K + 300K = 350,000 vectors.
  At H-10M the largest level weighted P·32, so this never occurred. E1 and E1R had 0/100.
- **E1 first-query latency** (collection pass, not preregistered): the first ~10
  queries take 250–410 ms against about 40 ms later. This is cold, freshly written
  on-disk segments after ingestion; G0's first queries are slow too.

## Next

10M (arms G0 and E1) was first job `25814270`, submitted from `af9e8fdc1`; it failed on a 60 s search timeout (Amendment 2) and was resubmitted as `25820728` from `10c65e255`, arms E1 then G0. That commit differs from the pilot only
in documentation and the PBS binary cache; the compiled-source key `5e30b2b4…` is
identical.

Walltime is 12 h. The preregistered formula (1.5 × 1.23 h + 8 h = 9.8 h) covered only E1
and the replays; about 1.2 h (the H-10M G0 arm) and the build are added.
