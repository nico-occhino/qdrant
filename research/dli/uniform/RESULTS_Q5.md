# Q5. The imbalance trigger in Qdrant (results)

**Status: S1 complete and validated (in-sample). S2 (out-of-sample) and I1 (descriptive) are running.** The
preregistration is [Q5_PREREGISTRATION.md](Q5_PREREGISTRATION.md), frozen at `f5fb54c16`.

## S1: Phase 3 Shift, seed 20261121 (job 26053393; in-sample for the 8× threshold)
- **The run.** E1, E1R and E1T ran serially on one node (galdor2, 8 CPUs) from the frozen staging (binary
  `ae6121df…`, source key `6b8101ff5d02a637552c4bed`). The job started at 14:47 UTC. Building the binary plus E1 took
  2 h 18 min, E1R 1 h 46 min and E1T 1 h 34 min, 5 h 38 min in all. PBS reported 12.5 GB in use at 5 h 03 min;
  that is one reading, not a peak.
- **The archive.** `C:\Users\nicoo\Workspace\Research-Results\2026-11-27-uniform-q5-S1\` (223 files,
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

### Not yet reported
- **The cross-engine comparison.** It is meaningful on S2, where the inputs are identical to PostgreSQL's run; it
  waits for S2.
- **W@0.90 per batch.** It is in `analysis-mq.json` (`per_batch`) and will be tabulated with S2 and I1.
