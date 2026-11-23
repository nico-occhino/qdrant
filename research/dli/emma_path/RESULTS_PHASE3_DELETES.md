# Phase 3 results: delete scenarios (primary, 1M)

**Status.**
- **Decided:** the delete hypotheses H6, H6b and H7.
- **Pending:** H5 (shift) and Part B (insert-only: E1C, E1L), from jobs 25859944 and 25859945, still running at
  this writing.

Rules come from the frozen preregistration ([PHASE3_PREREGISTRATION.md](PHASE3_PREREGISTRATION.md) §5, §9b, §9c).
The outcomes are the in-job `analyze_phase3.py` decisions; nothing was re-analysed.

## Provenance

| Scenario | Job | Exit | Walltime | Peak mem |
|---|---|---|---|---|
| Heavy-R | 25859938 | 0 | 9:51:48 | 46 GB |
| Medium-R | 25859939 | 0 | 7:37:07 | 42 GB |
| Constant-R | 25859940 | 0 | 4:09:32 | 37 GB |
| Heavy-O | 25859941 | 0 | 10:51:30 | 36 GB |
| Medium-O | 25859942 | 0 | 8:13:39 | 35 GB |
| Constant-O | 25859943 | 0 | 3:21:02 | 37 GB |

- **Source and inputs.** Commit `f370a0ef0`, whose compiled source, PBS script, planner and analysis are identical
  to the freeze `6538a4483`. Inputs `native-f370a0ef097e`. N = 1M, seed 20261121, 1 + 20 batches. Arms E1, E1G and
  E1V. Test panel rows 9000–9999.
- **Integrity.** Every batch of every run matched the plan's live-set count and SHA-256; the driver asserts it, and
  every job exited 0.
- **Evidence.** Remote `dli-emma-path-20261120/results/phase3-primary-<scenario>-<job>/`; local copy
  `C:\Users\nicoo\Workspace\Research-Results\2026-11-20-dli-emma-path\phase3-primary\`.

## Primary hypotheses (test, B* = 50,000; Δ = mean over 21 batches of the mean-R@30 difference)

| | Scenario | Contrast | Δ (pp) | 95% CI | 90% CI | Outcome | Bonferroni |
|---|---|---|---|---|---|---|---|
| **H6** | Constant-R | E1V − E1 | **+1.20** | [1.15, 1.24] | [1.16, 1.24] | **supported** (≥ 1.0, lower bound > 0) | supported |
| **H6b** | Medium-R | E1G − E1 | +0.11 | [0.08, 0.15] | [0.08, 0.14] | **not supported** (< 1.0 pp) | not supported |
| **H7** | Medium-O | E1G − E1 | +0.04 | [0.03, 0.04] | [0.03, 0.04] | **supported: equivalent within ±1 pp** (TOST) | equivalent |

## Descriptive (all six scenarios; mean R@30 at B*, dead-posting share of selected postings at B*)

| Scenario | E1 R@30 | E1G | E1V | E1 dead share (mean / max) | E1V dead share | Publications E1 / E1G / E1V |
|---|---|---|---|---|---|---|
| Heavy-R | 0.963 | 0.965 | 0.963 | 0.017 / 0.03 | 0.017 | 200 / 204 / 200 |
| Medium-R | 0.964 | 0.965 | 0.964 | 0.078 / 0.16 | 0.078 | 200 / 204 / 200 |
| Constant-R | 0.980 | 0.980 | **0.992** | **0.308 / 0.37** | **0.070** | 200 / 201 / **241** |
| Heavy-O | 0.961 | 0.964 | 0.961 | 0.016 / 0.03 | 0.016 | 200 / 201 / 200 |
| Medium-O | 0.962 | 0.962 | 0.962 | 0.051 / 0.10 | 0.045 | 200 / 201 / 202 |
| Constant-O | 0.981 | 0.981 | **0.986** | 0.19 / 0.31 | 0.073 | 200 / 200 / 226 |

## Interpretation

1. **Soft deletes are the Qdrant-specific cost, and the vacuum rule fixes it.** In Constant-R, about 31% of the
   postings selected at B* are dead under E1. Vacuuming at a 20% dead fraction cuts that to 7% and gains 1.2 pp of
   R@30, at the price of 41 extra publications (241 vs 200). Where deletes are lighter (Heavy and Medium), dead shares
   stay at or below 8%, the vacuum seldom fires, and nothing changes.
2. **Sommerová's Greedy rebuild does almost nothing in Qdrant.** In Medium-R, where it fires 4 times, it gains
   0.11 pp. It triggers on *live* occupancy, which soft deletes do not lower much. That is the design flaw found in
   preflight 1, now confirmed at 1M.
3. **Oldest-first deletion makes rebuilding irrelevant (H7), as her temporal argument predicts.** The oldest data
   sits in the deepest level, so deletes empty it from the bottom, and a rebuild has nothing to fix.
4. **Caveat on sensitivity.** At B* = 10·b = 50K, recall is 0.96–0.99. In the Constant scenarios the live set is only
   100K, so B* is half the collection. Differences are compressed near the ceiling. The effects above are therefore
   conservative, and the per-budget curves in the job outputs are the place to read them at lower budgets.
5. **One replicate.** CIs cover query sampling only.

## Next

H5 and Part B follow when jobs 25859944 and 25859945 finish.
