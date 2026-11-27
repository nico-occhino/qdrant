# Q4. Snapshot parity: why Qdrant and PostgreSQL snapshots differ (preregistration)

**Status: DRAFT.** It is frozen by a commit recorded in `STATUS.md` before arm (b) or (a′) runs.

The PostgreSQL counterpart is `DLI_SHIFT_TOPOLOGY_REPLAY.md` (pinned in `PINS.md`). Aligning PostgreSQL to Qdrant's
publication schedule cut the batch-11 MQ-validation gap from 22.50 pp to 0.21 pp. This study runs the reciprocal
experiment in Qdrant, and splits the cross-engine gap at the same snapshots into schedule, router-fit and
residual-engine parts.

## 1. Inputs (all pinned)
- **The plan.** S1, Phase 3 Shift, seed 20261121, 1M rows.
  - The local copy's 205 file hashes and batch ledger equal those of Qdrant job 25859944's plan.
  - Panel: MQ validation, 32 queries per batch (`mq_val`).
  - Snapshots: the ends of batches **10, 11 and 12**, covering level-3 publication. In PostgreSQL's schedule the
    publication falls in batch 12; in Qdrant's, in batch 11.
- **Gold.** Plan gold, as PostgreSQL used. Budget grid: the 17-point R3 grid (1K–320K). Metrics: Recall@30 at
  nominal 50K, and W@0.90.
- **PostgreSQL comparators.** `Research-Results/2026-11-26-pg-r3-followups/shift-topology/`, whose
  `ARTIFACTS.SHA256` verified on 2026-11-27.
  - **What they hold.** Per-query IDs, scored vectors and Recall@30 at 50K for both PostgreSQL schedules.
  - **The ledgers.** There are 117 publications each.
  - **The digests.** PostgreSQL's live-ID digests equal Qdrant's at every snapshot (e.g. batch 10, `b923ab3d…`).

## 2. Arms (Qdrant; one machine unless stated)
| Arm | What | How |
|---|---|---|
| (a) | Qdrant as run | the archived E1 rows of job 25859944 (MetaCentrum), unchanged |
| (a′) | Qdrant's own schedule, replayed locally | E1, per-batch chunking (5,025-point drains at batch starts), measured at 10–12 |
| (b) | PostgreSQL's schedule | E1, `DLI_P3_DRAIN_ALIGN=global`: every drain holds exactly 5,000 points across batch boundaries |
| (c) | PostgreSQL's routers on (b)'s topology | **blocked**: needs PostgreSQL's level routers (LMI1) at batches 10–12, which PostgreSQL archived only as hashes |

- **The known-answer check for (b).** Before data, a simulation of the Bentley–Saxe rule with exact 5,000-point
  drains reproduced all 117 PostgreSQL publications (target level and entries).
  - Arm (b) is accepted only if its publication ledger (target level, published points) equals
    `postgres_schedule-publications.jsonl` exactly.
  - Its live-ID digests and level populations must also equal PostgreSQL's captures at 10, 11 and 12.
- **The check for (a′).** Its ledger must equal job 25859944's batch ledger: the live digests, and the level
  populations at every batch.

## 3. Estimands
For each snapshot t ∈ {10, 11, 12}, mean Recall@30 at 50K and mean vectors scored at 50K on the same 32 queries.
- **Schedule effect in Qdrant:** S(t) = (b) − (a′).
- **Host and fit noise in Qdrant:** N(t) = (a′) − (a). Same schedule, different machine and fit.
- **Remaining cross-engine gap at matched schedule:** G_b(t) = PostgreSQL original − (b). This is router fit plus
  engine, and is split further by (c) once available.
  - **Router fit:** F(t) = (c) − (b).
  - **Residual engine:** E(t) = PostgreSQL original − (c).
- **Mirror check:** PostgreSQL aligned − (a′). This is the gap at Qdrant's schedule, reproducing PostgreSQL's
  0.21 pp at batch 11 from the Qdrant side.

**Statistics.**
- All contrasts are paired over the 32 queries, with a query bootstrap of 10,000 resamples (seed 20261121).
- Point estimates and 95% CIs are reported per snapshot, and batch 11 is the headline.
- W@0.90 is reported per arm and snapshot. It is descriptive, because PostgreSQL captured only 50K.

## 4. Interpretation rule (stated before data)
- **What we compare.** At batch 11, if |G_b| is within 2 pp (PostgreSQL's per-batch margin) while |S| is at least
  10 pp, the schedule explains the gap from the Qdrant side, as it did from the PostgreSQL side.
- **What remains.** Batch 10 is reported with the same decomposition. PostgreSQL saw a residual there, and that
  residual is attributed to router fit and engine only once (c) exists.
- **What this is.** It is a diagnostic on validation panels. It changes no frozen result, and it makes no latency or
  equivalence claim.

## 5. Gates
- (a′) and (b) pass their ledger known-answer checks, above.
- Full-budget recall is ≥ 0.999 on every snapshot, using plan gold; tie and rounding shortfalls are classified as in
  the DLI-best pilot.
- The runs are local, archived to C: with a `STORAGE_SHA256.txt`.

## 6. Execution
- **The runs.** Local and serial: (a′) then (b), with `DLI_P3_MAX_BATCHES=13` and `DLI_P3_EVAL=10,11,12`.
- **The code.** The same binary for both, the frozen revision's perf build.
- **The analysis.** `analyze_q4.py`, with a known-answer test, committed before the runs.
