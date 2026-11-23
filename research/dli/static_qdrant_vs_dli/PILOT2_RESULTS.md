# Pilot 2 results: D vs S1 vs the granularity-matched SG

**Status: COMPLETED, not ACCEPTED.**
- Gates 1, 2, 4 and 5 pass. Gate 3 (known-answer) fails on Shift (§3).
- The SG arm did **not** achieve the matching it was designed for during the trajectory (§2).
- These are **validation-panel** numbers from seed 20261301; they plan the study and are not its result.

## 1. Provenance

| Workload | Job | Exit | Walltime | SG buckets |
|---|---|---|---|---|
| W1 insert-only | 25875195 | 0 | 1:26:41 | 200 |
| W2 shift | 25875196 | 0 | 0:30:37 | 183 |
| W3 constant_r | 25875197 | 0 | 0:16:16 | 20 |
| W4 constant_o | 25875198 | 0 | 0:20:23 | 20 |

- **Source.** Commit `b5a2b1cd6`, with amendments A1–A4. Binary `9588d7f9…`, a cache hit (compiled source unchanged
  since `301319ada`). Source bundle `c42659df…`.
- **Setup.** N = 1M, the first 6 batches, validation panels (`calib` for W1, `mq_val` for W2, `val` for W3/W4).
  Arms D (E1, `mdn_code`, grid up to 320K plus full search), S1 (1 segment, 1,000 buckets) and SG (1 segment,
  B = round(L_max / 5,000)).
- **In-job tests.** Both analysis known-answer tests passed in every job.
- **Evidence.** Remote `dli-emma-path-20261120/results/static-vs-dli-pilot2-<workload>-<job>/`.

## 2. Work at mean validation R@30 = 0.90 (in-job `analyze_static_vs_dli.py`)

| Workload | W_D / W_S1 [95% CI] | **W_D / W_SG [95% CI]** | Attained |
|---|---|---|---|
| W1 insert-only | 4.25 [4.11, 4.38] | **2.31 [2.27, 2.35]** | 6/6 |
| W2 shift | 2.88 [2.37, 3.44] | **2.33 [2.06, 2.67]** | 6/6 |
| W3 constant_r | 6.13 [4.92, 7.60] | **0.79 [0.66, 0.91]** | 6/6 |
| W4 constant_o | 7.91 [6.40, 9.74] | **0.96 [0.78, 1.08]** | 6/6 |

D/S1 reproduces pilot 1 exactly: same seed, same plan, same S1. The run is deterministic.

**SG matches D only at the plan's peak live count, not per snapshot.** Size-weighted points per learned bucket:

| | Batch 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| W1 D | 2,328 | 1,679 | 1,762 | 2,155 | 2,710 | 3,096 |
| W1 SG | 500 | 725 | 950 | 1,175 | 1,400 | 1,625 |
| W2 D | 4,500 | 2,046 | 1,653 | 1,767 | 2,084 | 2,501 |
| W2 SG | 250 | 501 | 751 | 1,001 | 1,252 | 1,502 |
| W3 D | 2,328 | 1,680 | 1,321 | 1,369 | 2,545 | 1,424 |
| W3 SG | 5,000 | 5,000 | 5,000 | 5,000 | 5,000 | 5,000 |

- **Interpretation (descriptive).** In every workload, the arm with the finer buckets needed less work.
  - In the growing workloads (W1, W2), SG is 2–18× finer than D and beats it by 2.3×.
  - In the constant workloads (W3, W4), SG is 2–4× coarser than D, and ties with it (W4) or loses to it (W3, where D
    needs 21% less work).
- These six snapshots are therefore consistent with **partition granularity explaining the whole S-versus-D
  difference**. They give no clean evidence of a lifecycle effect in either direction.
- A per-snapshot granularity-matched control is required to decide (§5).

## 3. Gates

| Gate | Result |
|---|---|
| 1. Ledger matches after every batch | pass |
| 2. S's ordinary optimizer runs; D publishes | pass: 12 static builds per S arm; D 54–65 publications |
| 3. Known-answer at full search: mean R@30 ≥ 0.995 every batch | **pass** for W1, W3, W4 (minimum 0.997–0.998). **Fail for W2:** D 0.9938, S 0.9969 |
| 4. Grids bracket 0.90 | pass (6/6 everywhere) |
| 5. Resources | pass (16 min – 1.5 h) |

**On the W2 failure.** Full search misses about 0.3–0.6% of gold on the 32-query `mq_val` panel, in both systems. The
likely explanation is ties: Float16-stored, near-duplicate vectors inside one LAION cluster, scored in f32 by Qdrant
against a float64 gold. This is **not verified**, because the replay does not store returned IDs. Before freezing:
record returned IDs and scores at the full-search point, and check whether every miss is a tie at rank 30.

## 4. Maintenance (6 batches)

| Workload | Arm | Builds (fresh / reused) | Training s | Points rewritten | Ingest + settle s |
|---|---|---|---|---|---|
| W1 | D | 65 (7 / 58) | 9 | 2,475,000 | 40 |
| W1 | S1 | 12 (12 / 0) | 164 | 1,365,000 | 186 |
| W1 | SG | 12 (12 / 0) | 59 | 1,365,000 | 77 |
| W3 | D | 65 (7 / 58) | 9 | 2,148,400 | 41 |
| W3 | SG | 12 | 37 | 666,973 | 52 |

D trains 4–18× less than the S arms and rewrites 1.8–3.2× more points.

## 5. What this implies

1. **The question has changed.** It is no longer "lifecycle versus hierarchy" but "how much of each is granularity".
   Answering it needs an S arm with **per-snapshot matched bucket size**: an adaptive static LMI whose bucket count is
   set at each build from the segment's population, B = round(n / g) for a fixed target size g. That requires a new,
   opt-in LMI config option, which is a code change.
2. **Alternatively**, accept that this study measures the two lifecycles *as configured*. The thesis claim then
   becomes: "Qdrant's ordinary lifecycle with a sensible fixed bucket count beats E1 on search work at 1M; at matched
   granularity the difference is within the granularity effect."
3. Resolve the W2 known-answer failure (§3) before freezing.

## Addendum (2026-11-23): the W2 known-answer failure is a tie artefact, not a search bug

**Method.** The pilot2 Shift plan was regenerated locally from the verified corpus (sha256 `f5868d6e…`, first 1M rows)
with the committed `phase3_plan.py`. All 205 file hashes match the job's `plan.json`. For every query that misses at
full search, [`known_answer_diag.py`](known_answer_diag.py) recomputes the exact float64 ranking over that batch's live
set and asserts it equals the stored gold. It then reports:
- the score margin between ranks 30 and 31;
- how many live rows are byte-identical to the rank-30 row;
- whether a Qdrant-like scoring drops a gold point. That scoring normalises in f32 and stores the vector as f16, as
  Qdrant does for Cosine.

Output: [`derived/known_answer_shift_pilot2.json`](derived/known_answer_shift_pilot2.json).

**Result.** There were 43 misses at full search, all on validation panels, batches 0–5.
- **40 are shared by D, S1 and SG.** Each one loses a single point at rank 30, with a float64 margin of 1.5·10⁻⁶ to
  2.7·10⁻⁴. The Qdrant-like re-scoring reproduces the swap in 39 of the 40. **Cause:** the stored vector is rounded
  (normalised, then cast to f16), while gold scores the raw f16 row in float64. It is the same for every arm.
- **3 are D-only.** All three are `mq_val` query 15, at batches 3, 4 and 5 (D loses 5, 1 and 2 points). Its rank-30
  score is shared by **88, 169 and 230 byte-identical corpus rows**: a heavily duplicated LAION image. The
  rank-30/31 margin is exactly 0.
  - Gold breaks the tie by smaller ID. D's resolution breaks it differently. S's order happens to match the smaller
    IDs.
  - Any 30 of the tied rows are a correct answer, so D is not wrong.

**Consequence for the gate.** Without query 15, D's minimum is 0.999, and both arms pass 0.995 in every batch.
- The threshold stays at 0.995.
- The gate's *scoring* must become tie-aware: a returned point counts as correct if its exact score equals the gold
  rank-30 score. This needs the tie set, so the planner would emit every live row whose score equals gold rank 30,
  and the driver would count against it.
- This is a pre-freeze amendment for the author to approve. No test-panel data exist. Until it is implemented, the
  gate is reported both ways: raw, and excluding exact ties at rank 30.
