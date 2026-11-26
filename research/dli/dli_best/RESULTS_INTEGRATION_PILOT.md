# DLI-best integration pilot: results (validation panels only)

**Status: COMPLETED, NOT ACCEPTED as written (2026-11-26).** Criteria: [INTEGRATION_PILOT.md](INTEGRATION_PILOT.md),
fixed before any run.

| Criterion | insert-only | constant-R | shift |
|---|---|---|---|
| 1. Ledger | **pass** | **pass** | **pass** |
| 2. Native equals control | **pass** | **pass** | **pass** |
| 3. Known answer ≥ 0.995 | **fail** | **pass** | **fail** |
| 4. Mechanisms active | **pass** | **pass** | **pass** |

What fails is criterion 3. The cause is shared by every policy, E1's own `mdn_code` included: the gold scores raw
Float16 rows in Float64, while Qdrant stores each Cosine vector normalised and cast to f16. A few points near rank 30
swap order. No miss is specific to DLI-best or to the native proportional policy. The threshold is not moved, so the
pilot is not accepted as written. How to treat criterion 3 is the author's decision (see the last section).

The answer to the pilot's three questions:
1. The combined configuration runs correctly on all three workloads.
2. The native proportional policy reproduces the offline control exactly.
3. DLI-best is not worse than E1. It needs 9–39% less work, and no ratio comes near the 1.05 stop line.

## Provenance

| Item | Value |
|---|---|
| Jobs | `25942585` insert-only (turin9, 1:39), `25942586` constant-R (zenon21, 0:58), `25942587` shift (zenon34, 6:20; walltime extended to 8 h by the author). All exit 0 |
| Source | commit `e32fc6b90`, staged as `native-e32fc6b9092a` (bundle sha256 `49ee22e258dbd33f…`). Binary sha256 `de07a5a3e50d…`, built by 25942585 and reused from the cache, source key `8a0a28b3807f670608710792` |
| In-job checks | 5 Rust unit tests and 3 Python known-answer tests passed in each job. The plan is identical to the Phase 3 primary plan (130, 86 and 205 files) |
| Integrity | `SHA256SUMS` verifies every file except `job.log` in all three runs. The job writes `job.log`'s final `COMPLETE` line after computing the checksums; without that line, `job.log` matches its recorded hash in all three |
| Archive | MetaCentrum `dli-emma-path-20261120/results/dli-best-pilot-*`. C: `Research-Results\2026-11-25-dli-best-pilot\`, identical to storage on all 247 files (`STORAGE_SHA256.txt` there) |
| Comparison runs | Phase 3 E1 (`mdn_code`): 25859945 insert-only, 25859940 constant-R, 25859944 shift. Same plan, same validation queries |
| Derived | [`derived/`](derived/): the in-job `analysis-*.json`, the offline `tie_check-*.json` and `versus_detail.json` (per-batch work and attainment) |

## 1. Ledger

The live count and live-set SHA-256 match the plan after every batch: 21, 21 and 20 batches. Every batch has its
`rows.jsonl`: 12,096 control/native row pairs per uniform workload and 34,560 for shift.

## 2. Native equals control

- **Work.** `proportional_native` and the offline `proportional` scored exactly the same number of vectors in
  every row (0 mismatches on all three workloads).
- **Recall.** Recall@30 differs in 0 rows on insert-only, 11 on constant-R and 37 on shift. The differences come
  from 1, 1 and 3 queries. Every one is an exact Float64 score tie at rank 30, with more tied rows than the
  difference: tied (gold, other) rows were (2, 1); (2, 2), (1, 3) and (28, 45). With equal work and equal tied
  scores, the two policies returned different copies of the same score.
- **Method.** [`tie_check.py`](tie_check.py) rebuilds each batch's live set and regenerates the validation gold. Both
  are checked against the plan's SHA-256 in every batch and panel.
  - Uniform workloads: the planner's own rules are used.
  - Shift: the regenerated plan directory of PostgreSQL R3 job 25910367 is used, after all 205 of its files matched
    the Phase 3 `plan.json` hashes. The corpus's first 1M rows match storage (sha256 `6e1c7237…`).

## 3. Known answer at full search (budget 2,000,000)

| | insert-only | constant-R | shift `mq_val` | shift `ids_val` | shift `ood_val` |
|---|---|---|---|---|---|
| Lowest batch, raw | 0.9938 (b19) | 0.9958 (b10) | 0.9948 (b5) | 0.9948 (b14) | 0.9948 (b6) |
| Lowest batch, exact ties excluded | **0.9938** | 0.9958 | **0.9948** | **0.9948** | **0.9946** |
| Batch-panels below 0.995 | 2 of 21 (b16, b19) | 0 of 21 | 1 of 20 | 1 of 20 | 1 of 20 |

**Every miss, classified** (`tie_check-*.json`):

| | insert-only | constant-R | shift |
|---|---|---|---|
| Native rows missing points at full search | 47 (48 points) | 41 | 132 (134 points) |
| Exact rank-30 tie | 0 | 1 | 5 |
| Storage rounding (the Qdrant-like rescore loses the same point) | 46 | 40 | 126 |
| Neither | 1 | 0 | 1 |
| Same missed rows as `mdn_code` in the same run | yes, all | yes, all | yes, all |

- **Rounding misses** lose 1 or 2 points at rank 30. Their Float64 margin between ranks 30 and 31 is 1.2·10⁻⁶ to
  3.4·10⁻⁴.
- **The rescore** mimics Qdrant's Cosine storage: f32 normalise, cast to f16, f32 query. It is the same method as
  the static-vs-DLI pilot 2 addendum, which found 39 of 40 such misses.
- **The two "neither" rows** are single points with margins of 1.2·10⁻⁴ and 9·10⁻⁶. `mdn_code` misses them too. The
  rescore does not reproduce Qdrant's exact f32 summation order, so a near-boundary swap it misses is plausible
  *[inferred, not shown]*.
- **The failing batch-panels** each have one or two extra rounding misses on a 32-query panel. One point is 1/960 of
  a batch-panel's recall, so 0.9948 is 5 lost points in 960.

**Reading.** Criterion 3 checks that the full search is exact. The misses come from how the gold is defined, not
from search: every policy, E1's included, misses the same rows. As written, though, the criterion fails on insert-only
and shift.

## 4. Mechanisms

| | DLI-best final levels (level: live) | E1 final levels (Phase 3) | Check |
|---|---|---|---|
| insert-only | L1 50,000; L2 55,000; L3 895,000 | L1 10,000; L2 440,000; L3 550,000 | deepest larger, the levels above ≤ ρ·deepest: **pass** |
| shift | L1 25,000; L3 875,482 | L2 400,203; L3 500,279 | **pass** |
| constant-R | L1 10,360; L2 89,640 | L1 31,117; L2 68,883 | **30 vacuum publications**; worst learned-level dead fraction after settling 13.0% ≤ 20%: **pass** |

## 5. Descriptive: DLI-best against E1 (no gate)

W@0.90 is the vectors scored to reach mean R@30 = 0.90, interpolated. Ratios are DLI-best (`proportional_native`)
over Phase 3 E1 (`mdn_code`), with batch-stratified bootstrap 95% CIs, on 32-query panels.

| | R DLI-best / E1 [95% CI] | Batches where both reach 0.90 | Where only DLI-best reaches 0.90 |
|---|---|---|---|
| insert-only `val` | **0.761** [0.707, 0.830] | 21 / 21 | — |
| constant-R `val` | **0.632** [0.587, 0.668] | 21 / 21 | — |
| shift `mq_val` | 0.774 [0.722, 0.960] | 11 / 20 | 9 (b4, 6–10, 17–19) |
| shift `ids_val` | 0.614 [0.574, 0.762] | 15 / 20 | 5 (b6–10) |
| shift `ood_val` | 0.909 [0.867, 1.168] | 7 / 20 | 13 (b5–10, 13–19) |

- **The 1.05 stop rule is not triggered.** No point ratio is above 1.
- **On shift, the ratio is computed only where both reach the target.** E1 does not reach mean R@30 = 0.90 within
  the Phase 3 budget grid in 5–13 of 20 batches, while DLI-best always does. Those batches are left out, so the
  ratios understate DLI-best's advantage under shift. The `ood_val` CI includes 1.

### Allocation on the DLI-best topology: a small-panel signal

The same runs also replayed `mdn_code` on DLI-best's own topology. That separates the allocation policy from the
maintenance components:

| R proportional / `mdn_code`, DLI-best topology, `val` (32 queries) [95% CI] | Larger panel on a consolidated topology | E1 topology (policy replay, test panels) |
|---|---|---|
| insert-only 1.072 [1.006, 1.122] | **`partb`, 1,000 queries: 0.915 [0.908, 0.924]** (tie-break job 25931720) | 0.908 |
| constant-R 0.855 [0.813, 0.895] | — | 0.784 |
| shift `mq_val` 1.183 [1.060, 1.239] | not run | 1.023 (equivalent) |
| shift `ids_val` 0.939 [0.893, 0.994] | not run | 0.887 |
| shift `ood_val` 0.849 [0.835, 0.879] | not run | 0.693 |

- **On insert-only, the 32-query panel is contradicted by the 1,000-query one.**
  - The tie-break job ran the same configuration on the same plan, and on `partb` proportional is clearly better than
    `mdn_code`. Recomputed from the C: archive: `derived/tiebreak_E1C_prop_vs_mdn.json`.
  - That job's own `val` panel gives 1.072 [1.006, 1.122], the same as this pilot. The tie-break report had already
    noted that `val` and `partb` point in opposite directions there.
- **The pilot reproduces the tie-break run closely.** On insert-only, recall is identical in all but 2 of 22,848
  shared `val` rows. Work differs by at most 0.7% per row.
  - The router hashes differ, so the two runs trained slightly different routers. *[Inferred, not checked]*: the
    hosts differ (alfrid2 has 256 CPUs, turin9 has 128), and threaded k-means or training on them may not be
    bit-reproducible.
  - The S5 reproduction in the tie-break was exact, but on its own job.
- **Shift `mq` is untested on a large panel.** It is the one remaining sign that proportional may lose its edge after
  consolidation. On E1's topology, `mq` was already the one panel where proportional was only equivalent.
- **Status.** All of this is descriptive, on validation panels, and was not preregistered. The frozen DLI-best rule
  does not re-select components on pilot data, and the rule says interactions are measured in the final benchmark.

## What is open (author decisions)

1. **Criterion 3.** Choose between two options:
   - **(a)** Record the pilot as not accepted on criterion 3. The documented cause is gold definition, which E1
     shares, and the final benchmark fixes the gold in its own preregistration.
   - **(b)** Run nothing more and treat it as a pass. This would move a fixed criterion after seeing data, so it is
     not recommended.

   Recommendation: (a). The final benchmark's preregistration should define gold on the stored representation
   (Qdrant-like f16 after normalisation), or count a returned point as correct when its exact gold score equals the
   rank-30 score. This must be fixed before any final-benchmark data exist.
2. **Allocation on the DLI-best topology (§5).** The frozen rule keeps DLI-best as it is. Recommendation: the final
   benchmark preregisters `mdn_code` on the DLI-best topology as a secondary, descriptive replay in every workload.
   Shift `mq` is where it matters. The replay already runs both policies on the same snapshots, so it costs nothing
   extra, and the thesis can report the interaction on test panels.
