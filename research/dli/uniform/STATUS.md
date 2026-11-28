# Uniform questions: status

Updated 2026-11-27. Branch `research/uniform-questions-20261127` (worktree `~/work/qdrant-uniform`).

## Author actions
- Trigger V2 is skipped (author, 2026-11-28).
- Otherwise none. MetaCentrum Kerberos is valid; if it expires, the `kinit` is written here.

## Running jobs (this worktree; cap of 4)
- None.

## Local runs
- **Q4(c)/(c′)**: done. Archived as `Research-Results/2026-11-28-uniform-q4c/`.

## Finished jobs
- `26076922` **Q7: complete.** The frozen monotone gate failed (summed DLI member restarts); it passes under
  Amendment A2 (author, after the data). Results in [RESULTS_Q7.md](RESULTS_Q7.md); archived on C:.
- `26075075` **Q5 S2: validated.** Verdict not met (recovery 0.213); archived on C:.
- `26075094` **Q5 I1: validated.** No firing; E1T = E1; archived on C:.
- `26053393` **Q5 S1: validated.** See [RESULTS_Q5.md](RESULTS_Q5.md); it is archived on C:.
- `26075041` P0 memory, first attempt: a script defect (it read the v2 path on a v1-memory host); no counters.
- `26075063` P0 memory, the rerun: **pass**.
- `26075042` P0 network: **pass** (`p0net.json`). The job then hangs at its final `wait` until walltime, so it was
  killed (Exit_status −29). Its evidence is checksummed on collection, in `evidence/p0/`.

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
- **S2 (out-of-sample): not met.** E1T − E1 = +0.24 pp [0.10, 0.38], recovery 0.213, trainer 0.228×. One firing
  (batch 7, 10.58×). PostgreSQL on the same inputs: +1.20 pp, recovery 0.394, also not met.
- **I1: no firing** (maximum 2.57×), so E1T = E1, as expected.
- **Done.** See [RESULTS_Q5.md](RESULTS_Q5.md). One open item: PostgreSQL's firing batch (publication 44), for the
  §5 descriptive expectation.

## Q10 and Q11. Memory budget and distributed scale (shared with PostgreSQL)
- **The spec** [SCALE_SHARED_SPEC.md](SCALE_SHARED_SPEC.md) is a DRAFT. The author accepted the defaults; it waits
  for PostgreSQL's review.
- **Shard hashes for I1:** `evidence/q11/shard_hashes_I1.json`.
- **Q11 is cut** (author, 2026-11-28): future work. Q10 waits for PostgreSQL's review (`SCALE_SPEC_PG_REVIEW.md`)
  and the author's freeze.
- **P0 passes on Qdrant's side:** cgroup v1 memory, page cache charged and reclaimed, swap inside the limit,
  fadvise eviction, inter-node ssh and ports. Four amendments are proposed from P0 (per-job PSI and blkio are not
  available).

## Q4. Snapshot parity
- **(a), (a′) and (b) are done:** [RESULTS_Q4.md](RESULTS_Q4.md). (b) reproduces PostgreSQL's 117-publication ledger
  and its snapshot topologies exactly.
  - **Batch 11.** The schedule explains the gap: S = −24.17 pp, with +1.67 pp remaining.
  - **Batch 10.** A topology-dependent residual remains (+10.42 pp at Qdrant's schedule).
- **(c) and (c′): done** (Amendment A1). With PostgreSQL's routers on the same topology, Qdrant reproduces
  PostgreSQL's recall and work: engine residual 0.0–0.1 pp. The batch-10 +10.42 pp residual is router fit (F′). See
  [RESULTS_Q4.md](RESULTS_Q4.md).

## Q7. Filtering
- **Fixture.** `q7_fixture.py` reproduces every value of PostgreSQL's spec (3ec4041).
- **Harness.** `lib/segment/tests/q7_filter_fixture.rs` runs S-LMI (250 buckets), the reconstructed E1 DLI members,
  and HNSW (m 16), counting per-query scored work.
- **Analysis.** `analyze_q7.py`, with its known-answer test, and the job script `pbs/q7.pbs`.
- **The local smoke (6,000 rows) passed the runner gates.** No bad ids; exact work = the qualifying count; the one
  exact shortfall is a gold tie; 24 widening restarts.
- **The preregistration is FROZEN** at `e85c176f7`. Job `26076922` is running (staging `native-q7-e85c176f72a7`).
- **Addendum A1** (`a6b067075`, before any result) defines the descriptive exact-vs-routed switch point.
- **Result (A2 applied; gate passes).** Qdrant routed (pre-filtered `RoutedIntersect`) never scores more than
  exact search, so its switch is below 0.1%. PostgreSQL's (post-filtered) switch is ≈ 12,326 qualifying vectors
  (≈ 20.5%).

## Inputs prepared locally (`~/research-admin/experiments/2026-11-27-uniform/`)
- **The S1 plan** `plans/S1-shift-20261121/`: copied from PostgreSQL R3 job 25910367's plan. Its 205 file hashes
  and batch ledger equal the Qdrant primary's plan (job 25859944).
- **The corpus prefix** `vectors-1m.f16`: SHA-256 `6e1c7237…1b229b1df2`, verified.
