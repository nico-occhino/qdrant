# Uniform questions: status

Updated 2026-11-27. Branch `research/uniform-questions-20261127` (worktree `~/work/qdrant-uniform`).

## Author actions
- Optional, for Q4(c): ask Codex to export PostgreSQL's level routers (LMI1) at Shift batches 10–12, for both
  schedules.
- Otherwise none. MetaCentrum Kerberos is valid; if it expires, the `kinit` is written here.

## Running jobs (this worktree; cap of 4)
- `26075075` Q5 S2 (PostgreSQL's fresh Shift, seed 20261127), 24 h, queued 2026-11-27. The binary is cached.
- Q5 I1 (Insert-only) is submitted automatically when `26075042` (P0 network) ends.
- Also counted against the cap: the final-benchmark W3 jobs `26044026` (S-LMI) and `26044027` (IVF).

## Finished jobs
- `26053393` **Q5 S1: validated.** See [RESULTS_Q5.md](RESULTS_Q5.md); it is archived on C:.
- `26075041` P0 memory, first attempt: a script defect (it read the v2 path on a v1-memory host); no counters.
- `26075063` P0 memory, the rerun: **pass**.
- `26075042` P0 network: **pass** (`p0net.json`). The job then hangs at its final `wait` until walltime, so its
  SHA256SUMS is computed on collection.

## Q5. Imbalance trigger
- **Done:**
  - the native rule (`ImbalanceRetrain::POSTGRES_FROZEN`) and arm `E1T`;
  - the per-batch decision log, with census time;
  - `DLI_P3_DRAIN_ALIGN=global` for Q4.
- **Gate 1 (unit tests): PASS.** It covers the rule boundaries, serialization, the trigger that never fires (equal
  to E1 in routers, populations and search bits), the trigger that always fires (exactly the eligible levels
  retrain, decision census = destination before compaction), the E1/E1R reuse regression, and 53 DLI tests.
- **Analysis.** `analyze_q5.py` and its known-answer test pass.
- **Gate 2, the same-node E1 equality: PASS.** S1 batches 0–2 on one machine: 1,632 rows, the ledger with routers,
  and 27 builds are all identical.
  - **The retry.** The first attempt hit a `copy_dir` race in the harness, which `ffae40bfc` fixes. This was one
    documented retry.
- **Gate 3, the E1T smoke: PASS.** 23 decisions, no retrains, a largest eligible ratio of 2.59×, and decisions that
  agree with the builds.
- **The preregistration is FROZEN** at `f5fb54c16a969d939b0c118711a956c7c8243980`.
- **S1 (in-sample): verdict met.**
  - **Estimates.** E1T − E1 = +7.89 pp [7.63, 8.15] at 50K (mq), recovery 0.690, at 0.277× E1R's trainer
    walltime.
  - **Firing.** It fired at batches 4 and 15, as expected before data.
  - **Gate 4.** The raw full-search gate failed (0.994–0.997); after the preregistered rounding classification there
    are 0 unexplained misses (`q5_rounding_check.py`).
  - **Runtime.** 5 h 38 min, so each plan fits in one 24 h job.
- **Next.** Analyse S2 and I1 the same way, including gate-4 classification, then the cross-engine comparison on
  S2.

## Q10 and Q11. Memory budget and distributed scale (shared with PostgreSQL)
- **The spec** [SCALE_SHARED_SPEC.md](SCALE_SHARED_SPEC.md) is a DRAFT. The author accepted the defaults; it waits
  for PostgreSQL's review.
- **Shard hashes for I1:** `evidence/q11/shard_hashes_I1.json`.
- **P0 passes on Qdrant's side:** cgroup v1 memory, page cache charged and reclaimed, swap inside the limit,
  fadvise eviction, inter-node ssh and ports. Four amendments are proposed from P0 (per-job PSI and blkio are not
  available).

## Q4. Snapshot parity
- **(a), (a′) and (b) are done:** [RESULTS_Q4.md](RESULTS_Q4.md). (b) reproduces PostgreSQL's 117-publication ledger
  and its snapshot topologies exactly.
  - **Batch 11.** The schedule explains the gap: S = −24.17 pp, with +1.67 pp remaining.
  - **Batch 10.** A topology-dependent residual remains (+10.42 pp at Qdrant's schedule).
- **(c) is blocked.**
  - **What it needs.** PostgreSQL's level-router weights (LMI1) at batches 10–12, for both schedules.
  - **Author action.** Ask the PostgreSQL side (Codex) to export them.

## Q7. Filtering
- **Fixture.** `q7_fixture.py` reproduces every value of PostgreSQL's spec (3ec4041).
- **Harness.** `lib/segment/tests/q7_filter_fixture.rs` runs S-LMI (250 buckets), the reconstructed E1 DLI members,
  and HNSW (m 16), counting per-query scored work.
- **Analysis.** `analyze_q7.py`, with its known-answer test, and the job script `pbs/q7.pbs`.
- **The local smoke (6,000 rows) passed the runner gates.** No bad ids; exact work = the qualifying count; the one
  exact shortfall is a gold tie; 24 widening restarts.
- **The preregistration is FROZEN** (2026-11-27, after Q5 S1 validated). Next: stage that commit and submit
  `pbs/q7.pbs` (4 h) when a slot frees. The cap is reached by W3 ×2, S2 and I1.

## Inputs prepared locally (`~/research-admin/experiments/2026-11-27-uniform/`)
- **The S1 plan** `plans/S1-shift-20261121/`: copied from PostgreSQL R3 job 25910367's plan. Its 205 file hashes
  and batch ledger equal the Qdrant primary's plan (job 25859944).
- **The corpus prefix** `vectors-1m.f16`: SHA-256 `6e1c7237…1b229b1df2`, verified.
