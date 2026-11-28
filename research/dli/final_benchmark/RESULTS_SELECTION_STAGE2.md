# Selection pilot, stage 2 (W2, W3), result

Status: **COMPLETE. Every gate passes.** Plan seed 20261202, validation panels only (W2 `val`; W3 `mq_val`,
`ids_val` and `ood_val`, 64 queries each), G1 gold. No test panel was read. The configurations are A3's: S-LMI
10,000 buckets, IVF 10,000 buckets, HNSW m = 16.
- **The analysis.** `analyze_final.py` in pilot mode, over the manifest `derived/selection_stage2_manifest.json`.
  The output is `derived/selection_stage2.json`, SHA-256 `0ddab7d72e7b4edddca2…`.
- **The data.** Local copies verified against every job's `SHA256SUMS`.

## Jobs
| Job | Workload / arm | Host | Runtime | Gates |
|---|---|---|---|---|
| 26042042 | plan W2 | | | complete |
| 26042043 | W2 E1 | elmu1-17 | 0.25 h | pass |
| 26042044 | W2 DLI-best | eluo1-4 | 0.29 h | pass (native = control) |
| 26042045 | W2 S-LMI 10,000 | elmu1-13 | 2.52 h | pass; 20 widening restarts admitted (A3.3) |
| 26042046 | W2 IVF 10,000 | elmu1-21 | 2.30 h | pass; 21 widening restarts admitted |
| 26042047 | W2 HNSW 16 | adan21 | 0.13 h | pass |
| 26042048 | plan W3 | | | complete |
| 26042049 | W3 E1 | elmu1-24 | 0.93 h | pass |
| 26044025 | W3 DLI-best | eluo1-3 | 1.45 h | pass (native = control) |
| 26044026 | W3 S-LMI 10,000 | turin49 | 12.85 h | pass; 14 widening restarts admitted |
| 26044027 | W3 IVF 10,000 | tyra25 | 9.78 h | pass; 15 widening restarts admitted |
| 26044028 | W3 HNSW 16 | tyra25 | 0.32 h | pass |

- **The resubmissions.** The first W3 DLI-best, S-LMI, IVF and HNSW submissions were held by PBS (a dependency
  system hold). They were cancelled and resubmitted with the author's authorisation; no data came from the held
  jobs.
- **Widening restarts.** They are all nprobe n → 2n drops in batches 0–4, where live counts are smallest relative
  to 10,000 buckets.

## Bracketing (§8: "every arm's grid brackets 0.90 in every batch")
- **Every arm except HNSW** interpolates in every batch, on every validation panel.
- **HNSW m = 16 is `upper_bound`** in W2 `val`, in 14 of 21 batches (0, 1, 3–11, 13, 15, 19). In W3 it is
  `upper_bound` on `ids_val` in 7 of 20 batches (0, 1, 3, 4, 6, 7, 11); `mq_val` and `ood_val` bracket.
  - **Why the grid cannot be lowered.** Its first point is ef = 30, and Qdrant clamps ef to at least the limit
    (k = 30).
  - **How it is handled.** Amendment A2's decision rule (`bound_adjust`) treats it as written. No attainment failure
    occurred anywhere.

## Runtime and the primary walltimes
The pilot evaluated 3 × 64 validation queries per batch for W3, and 64 for W2. The primary evaluates the 1,000-query
test panels in their place. Per batch, the records separate ingest (writes plus index build) from query evaluation
(`frontier_seconds` + `timed_seconds`). Scaling only the query part by 1,000/64 gives:

| Arm | W2 projected | W3 ingest | W3 query (pilot) | **W3 projected** | Current walltime (A3.5) |
|---|---|---|---|---|---|
| E1 | 2.9 h | 0.07 h | 0.80 h | **12.6 h** | 12 h |
| DLI-best | 3.5 h | 0.08 h | 1.34 h | **21.0 h** | 12 h |
| S-LMI 10,000 | 3.3 h | 11.98 h | 0.85 h | **25.2 h** | 24 h |
| IVF 10,000 | 3.9 h | 8.74 h | 0.97 h | **24.0 h** | 24 h |
| HNSW 16 | 0.7 h | 0.07 h | 0.18 h | **3.0 h** | 12 h |

- **The projection is linear in query count.** It ignores host variation, and S-LMI and IVF ran on different node
  types.
- **At A3.5's walltimes, W3 E1, DLI-best and S-LMI would exceed them, and IVF would reach its limit.** §11's own
  rule ("1.5× the longest of each arm") gives about 19 h for E1, 32 h for DLI-best, 38 h for S-LMI and 36 h for IVF
  on W3.
- **W1 and W2 fit** their current walltimes with ≥ 3× margin.
- **This needs a dated amendment before any primary submission (author decision).** Two options:
  1. **Longer walltimes for W3** (48 h for E1, DLI-best, S-LMI and IVF; HNSW unchanged), with the jobs unchanged.
     The cost is a longer queue wait for 48 h jobs.
  2. **Query evaluation split into windows** (each W3 arm run as two jobs over disjoint batch ranges, as PostgreSQL
     does and as Q5's `EVAL` allows). Each job stays ≤ 24 h, but every window replays the stream up to its first
     batch. Static S-LMI's 12 h ingest is then paid twice, so S-LMI alone would still need about 18 h per window.
  - **Recommendation: option 1.** It changes nothing in what is measured, and the arithmetic is simplest to audit.
