# Q5. The imbalance trigger in Qdrant (results)

**Status: complete on all three plans.** The preregistration is [Q5_PREREGISTRATION.md](Q5_PREREGISTRATION.md),
frozen at `f5fb54c16`.

## Summary (mean Recall@30 at 50K on the primary panel; paired query bootstrap, 10,000 resamples)
| Plan | Role | Firings | E1T − E1 | E1R − E1 | Recovery | Trainer E1T / E1R | Verdict |
|---|---|---|---|---|---|---|---|
| S1 (Phase 3 Shift) | in-sample | 2 (batches 4, 15) | **+7.89 pp** [7.63, 8.15] | +11.44 pp [11.20, 11.69] | 0.690 | 0.277 | **met** |
| S2 (PostgreSQL's fresh Shift) | out-of-sample | 1 (batch 7) | **+0.24 pp** [0.10, 0.38] | +1.12 pp [0.92, 1.32] | 0.213 | 0.228 | **not met** (recovery < 0.5) |
| I1 (Insert-only) | descriptive | 0 | 0.00 pp [0.00, 0.00] | +0.54 pp [0.47, 0.60] | 0 | 0.207 | not met (trigger never fired) |

- **The threshold does not transfer to S2.** It met both criteria on the stream it was in-sample for (S1), but not
  on the independent S2 stream.
  - **S2's headroom is small.** Retraining every compaction gains only +1.12 pp there, against +11.44 pp on S1.
  - **The one retrain is late.** It comes at batch 7, after E1's level 2 had already reached 10.6× at batch 6, and
    recovers 21% of that headroom (48% on mq_val).
  - **Training cost.** Criterion 2 (training cost) holds everywhere.
- **I1** behaved as stated before data: the trigger never fired (maximum 2.57×), and E1T equals E1 on every row.
- **Gate 4.** On every plan, all 40 (S1, S2) or 42 (I1) full-search cells fall short of 0.999 in the raw gate, and
  every miss is explained as stored-representation rounding (`q5_rounding_check.py`), with the arms identical.

## S1: Phase 3 Shift, seed 20261121 (job 26053393; in-sample for the 8× threshold)
- **The run.** E1, E1R and E1T ran serially on one node (galdor2, 8 CPUs) from the frozen staging (binary
  `ae6121df…`, source key `6b8101ff5d02a637552c4bed`). The job started at 14:47 UTC. Building the binary plus E1 took
  2 h 18 min, E1R 1 h 46 min and E1T 1 h 34 min, 5 h 38 min in all. PBS reported 12.5 GB in use at 5 h 03 min;
  that is one reading, not a peak.
- **The archive.** `Research-Results/2026-11-27-uniform-q5-S1\` (223 files,
  `STORAGE_SHA256.txt` verified, job `SHA256SUMS` verified).

### Gates (§6.4)
| Gate | E1 | E1R | E1T |
|---|---|---|---|
| Plan and live-ID ledger | pass | pass | pass |
| Completeness | pass | pass | pass |
| Monotone work | pass | pass | pass |
| Decisions agree with builds (E1T) | — | — | pass (159 decisions, 2 retrains, 0 problems) |
| Full-search known answer ≥ 0.999 (raw) | 0.9942–0.9973, 39 of 40 cells short | same | same |
| … after the preregistered tie/rounding classification | **pass**: 0 unexplained | **pass** | **pass** |

- **The known answer.** The job's raw gate failed, so it wrote `FAILED.json` (`gates_failed`) and kept its data.
  §6.4 says a shortfall is "classified with the tie check before the gate is judged". The job script did not
  implement that step; `q5_rounding_check.py` does.
  - **The rule.** It applies `q4_rounding_check.py`'s stored-representation rule, using its functions unchanged.
  - **The result.** Every miss in all 40 (batch, panel) cells is explained: plan gold is Float64 over raw Float16
    rows, while Qdrant searches its normalized Float16 copy.
  - **Arm independence.** The full-search results are identical across E1, E1R and E1T, query by query.
  - **Output.** `rounding-check.json` (SHA-256 `3d1df904…`).
  - **A disclosure.** The classifier was written for Q4 before S1's data existed, but the Q5 preregistration names
    the method ("the DLI-best tie and rounding slack"), not this file.

### Primary estimands (mean Recall@30 at 50K, paired query bootstrap, 10,000 resamples)
| Panel | E1T − E1 | E1R − E1 | E1T − E1R | Recovery |
|---|---|---|---|---|
| mq (primary) | **+7.89 pp** [7.63, 8.15] | +11.44 pp [11.20, 11.69] | −3.55 pp [−3.72, −3.37] | **0.690** |
| mq_val | +7.00 pp [5.55, 8.48] | +11.06 pp [9.63, 12.54] | −4.06 pp [−4.96, −3.15] | 0.633 |

- **Criterion 1** (lower bound > 0 and recovery ≥ 0.5): met.
- **Criterion 2** (E1T trainer walltime ≤ 50% of E1R's): met.
  - **Trainer walltime.** E1T is 220.6 s against E1R's 797.2 s, a ratio of **0.277**.
  - **Trained builds.** 23 vs 180, of which 21 are first fits in both arms. E1 has 21 trained builds and 176.9 s.
  - **Census overhead.** 16.7 ms in total over 159 decisions.
- **The verdict on S1 is "met"**, in-sample.

### Firing and imbalance
- **Firing.** The trigger fired twice, both at level 2:
  - at batch 4, job 39: 8.06× (12,100 of 150,076 points in the largest of 100 buckets);
  - at batch 15, job 139: 9.84× (14,760 of 150,076).
- **Expectation check (§5, stated before data).** "First firing at the first level-2 compaction of batch 4" was
  met. "Again in the batch-14 range" was met: the census taken at the batch-15 compaction is the batch-14 9.84×
  population. "Level 3 does not fire" was met: level 3 stayed at 3.98×.
- **Largest live max/mean, any level, by batch:**

| Batch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 1.9 | 2.7 | 4.1 | 8.1 | 10.1 | 11.4 | 11.1 | 11.3 | 10.7 | 10.5 | 10.5 | 4.0 | 4.0 | 7.8 | 9.8 | 9.0 | 8.6 | 9.5 | 10.8 | 15.7 |
| E1T | 1.9 | 2.7 | 4.1 | 8.1 | 2.5 | 4.5 | 3.4 | 4.7 | 4.2 | 5.3 | 5.3 | 4.0 | 4.0 | 7.8 | 9.8 | 4.0 | 4.0 | 4.0 | 5.6 | 5.0 |
| E1R | 1.8 | 2.6 | 1.9 | 2.0 | 2.5 | 2.4 | 2.3 | 2.4 | 1.9 | 2.0 | 2.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |

- **E1's level-2 trajectory** sits at 10–11× from batch 4 to batch 10 and reaches 15.7× at batch 19. This is
  Qdrant's E1 imbalance trajectory requested by the brief.
- **E1T after each retrain.** Both retrains bring the ratio back to 2.5–5.6×.
- **The per-batch gain** (mq, E1T − E1) is 0 until batch 3, +12.3 to +19.6 pp in batches 4–10, 0 in batches 11–14
  (after the level-3 promotion, where E1T and E1 coincide until the second firing), and +5.2 to +11.9 pp in
  batches 15–19.

### W@0.90 per batch
It is in each job's `analysis-<panel>.json` (`per_batch`), for every arm.

## S2: PostgreSQL's fresh Shift stream, seed 20261127 (job 26075075; out-of-sample)
- **The run.** grogu2, started 21:09 UTC: the build was cached; E1 took 1 h 37 min, E1R 1 h 43 min and E1T 1 h 32
  min.
- **The archive.** `…\2026-11-27-uniform-q5-S2\` (222 files, verified).
- **Gates.** The ledger, completeness, monotone work and decisions gates pass (159 decisions, 1 retrain). The raw
  known answer has a minimum of 0.99375 (38 of 40 cells short). After classification there are 0 unexplained misses,
  with the arms identical (`rounding-check.json`, SHA-256 `bb049f9f…0c7e`).

| Panel | E1T − E1 | E1R − E1 | E1T − E1R | Recovery |
|---|---|---|---|---|
| mq (primary) | +0.24 pp [0.10, 0.38] | +1.12 pp [0.92, 1.32] | −0.88 pp [−1.06, −0.70] | 0.213 |
| mq_val | +0.91 pp [0.14, 1.68] | +1.91 pp [0.83, 2.99] | −1.00 pp [−2.01, −0.01] | 0.477 |

- **Criterion 1** fails on recovery: the lower bound is > 0, but 0.213 < 0.5.
- **Criterion 2** holds: 173.0 s against 757.8 s, a ratio of 0.228, from 22 vs 180 trained builds.
- **Firing.** One firing: batch 7, level 2, at 10.58×.
- **Per-batch E1T − E1 (mq).** 0 through batch 6, then +1.4, +3.7, −4.0 and +3.7 pp in batches 7–10, and 0 from
  batch 11. Batch 11's level-3 promotion makes E1T and E1 coincide again.
- **Largest max/mean by batch:**

| Batch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 1.8 | 2.0 | 2.2 | 4.4 | 5.0 | 7.5 | 10.6 | 10.1 | 10.6 | 11.2 | 11.2 | 3.4 | 3.4 | 3.4 | 5.4 | 6.7 | 6.8 | 5.9 | 6.6 | 5.8 |
| E1T | 1.8 | 2.0 | 2.2 | 4.4 | 5.0 | 7.5 | 10.6 | 2.2 | 2.3 | 3.0 | 3.0 | 3.4 | 3.4 | 3.4 | 5.4 | 6.7 | 6.8 | 5.9 | 6.6 | 5.8 |
| E1R | 1.7 | 2.0 | 2.1 | 2.1 | 1.9 | 2.2 | 2.0 | 2.2 | 1.9 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 |

### Cross-engine on S2 (identical inputs; recall and work only)
PostgreSQL's result: `DLI_IMBALANCE_TRIGGER_RESULTS.md` at `92e2452`, SHA-256 `81043497…b147`, read-only.

| | PostgreSQL | Qdrant |
|---|---|---|
| Firings | 1, level 2, 9.868× (publication 44) | 1, level 2, 10.58× (batch 7) |
| Trigger − E1 | +1.20 pp [+1.03, +1.39] | +0.24 pp [+0.10, +0.38] |
| E1R − E1 | +3.05 pp [+2.84, +3.27] | +1.12 pp [+0.92, +1.32] |
| Recovery | 0.394 | 0.213 |
| Trainer trigger / E1R (within engine) | 0.121 | 0.228 |
| Verdict | not met (recovery) | not met (recovery) |

- **The engines agree qualitatively.** Each fires exactly once, at level 2 near 10×. Each gives a small positive
  gain that misses the 50% recovery criterion, at a small fraction of E1R's training.
- **Qdrant's headroom is smaller,** both E1R − E1 and the trigger gain.
- **§5's expectation (Qdrant fires no later than PostgreSQL) is undetermined.** PostgreSQL's document gives the
  firing as publication 44 of 183, not as a batch. Its batch is requested from the PostgreSQL side; it is not
  inferred here.

## I1: Insert-only, seed 20261121 (job 26075094; descriptive)
- **The run.** turin9, started 21:15 UTC: E1 took 1 h 52 min, E1R 2 h 05 min and E1T 1 h 49 min.
- **The archive.** `…\2026-11-27-uniform-q5-I1\` (231 files, verified).
- **Gates.** As on S2: the raw known-answer minimum is 0.99375 (40 of 42 cells short). After classification there
  are 0 unexplained misses, with the arms identical (`rounding-check.json`, SHA-256 `fc83232a…63d0`). 178 decisions,
  0 retrains.
- **No firing.** The largest eligible ratio was 2.57×, so E1T = E1 exactly: test 0.00 pp [0.00, 0.00].
  - **E1R − E1.** +0.54 pp [0.47, 0.60] on test, and +0.02 pp [−0.24, 0.26] on val.
  - **The imbalance.** E1 stayed at 2.5× (level 2) through batch 9, and 3.6× (level 3) from batch 10.
  - **The §5 expectation** of zero firings was met.
