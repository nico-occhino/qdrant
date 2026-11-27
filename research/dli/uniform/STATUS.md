# Uniform questions: status

Updated 2026-11-27. Branch `research/uniform-questions-20261127` (worktree `~/work/qdrant-uniform`).

## Author actions
- None needed right now. MetaCentrum Kerberos is valid; if it expires, the `kinit` is written here.

## Running jobs (this worktree; cap of 4)
- `26053393` Q5 S1 (Phase 3 Shift, seed 20261121; E1, E1R and E1T), 24 h, submitted 2026-11-27. It builds the binary (source key `6b8101ff…`).

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
- **Next.** S2 and I1 are submitted after S1's gates pass and its runtime is known (§8).

## Q4. Snapshot parity
- (a) and (b) are planned as local replays.
- **(c) is blocked:** it needs PostgreSQL's level-router weights at batches 10–12.

## Q7. Filtering
- The fixture spec is frozen at PostgreSQL 3ec4041. Implementation has not started.

## Inputs prepared locally (`~/research-admin/experiments/2026-11-27-uniform/`)
- **The S1 plan** `plans/S1-shift-20261121/`: copied from PostgreSQL R3 job 25910367's plan. Its 205 file hashes
  and batch ledger equal the Qdrant primary's plan (job 25859944).
- **The corpus prefix** `vectors-1m.f16`: SHA-256 `6e1c7237…1b229b1df2`, verified.
