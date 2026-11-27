# Q5. The imbalance trigger in Qdrant (preregistration)

**Status: FROZEN 2026-11-27**, by the commit that carries this line (recorded in `STATUS.md`), before any run on
the plans below. Gates 1–3 passed locally (`evidence/q5-local-gates/`). Later changes are dated amendments that say
whether they came before or after data.

This is the Qdrant port of PostgreSQL's frozen imbalance-trigger rule (`PREREGISTRATION_DLI_IMBALANCE_TRIGGER.md`,
pinned in `PINS.md`). The rule, arms, estimand and decision criteria are copied. Every departure is forced by engine
semantics or by the shared Q5 protocol, and is listed in §7.

## 1. Question
Can a selective retraining rule recover at least half of the recall benefit of always retraining (E1R) while
using at most half of E1R's trainer walltime? The comparison is with reuse (E1), on identical inserts, queries and
gold, in native Qdrant DLI.

## 2. Arms (one job runs all three on one node, serially)
- **E1.** Sommerová's naive insert: a compaction into an occupied learned level reuses that level's router. The
  details are buffer 5,000, arity 10, aⁱ buckets, a 40-sample-per-bucket floor and `ModelDrivenNormalizedCode`
  routing (`DLI_ARM=E1`).
- **E1R.** As E1, but every compaction retrains its destination (`DLI_ARM=E1R`).
- **E1T (the trigger).** As E1, except as follows.
  - **The census.** At a compaction into an occupied destination at level 2 or deeper, the census is read from that
    destination's serving generation, before compaction.
  - **The decision.** If the largest live bucket divided by the mean live bucket size over **all allocated buckets**
    is **≥ 8.0**, the destination is retrained exactly as E1R would. Otherwise it reuses the router.
  - **Other destinations.** Occupied level 1 always reuses, and empty destinations always train.
  - **Implementation.** `ImbalanceRetrain::POSTGRES_FROZEN` (min_level 2, ratio 8,000‰), in
    `lib/collection/src/collection_manager/dli_schedule.rs`, with `DLI_ARM=E1T`.
  - **What the rule cannot see.** It never consults queries, future inserts or gold.

## 3. Plans and panels
| Id | Plan | Role |
|---|---|---|
| **S1** | Phase 3 Shift, seed 20261121, 1M rows, 20 snapshots | **primary** (the shared Q5 protocol of `CROSS_ENGINE_SYNTHESIS.md`) |
| **S2** | PostgreSQL's fresh Shift stream: source rows 1,000,001–2,000,000, seed 20261127, the exact files of manifest `90f117f8…` | **out-of-sample replication**, same inputs as PostgreSQL's own trigger run |
| **I1** | Phase 3 Insert-only, seed 20261121 | secondary, descriptive |

- **Panels.**
  - S1 and S2 use MQ test (1,000 queries) for every estimand, and MQ validation (32) for reporting only.
  - I1 uses its test panel (1,000) and validation panel (32).
- **Gold.** Each plan's own exact gold (float64 cosine, Float16 rows against Float32 queries, ties by smaller id).
  This is the gold PostgreSQL used.
- **Budget grid.** The R3 grid of 17 settings: 1,000; 2,000; 3,500; 5,000; 7,000; 10,000; 14,000; 20,000; 28,000;
  40,000; 50,000; 56,000; 80,000; 113,000; 160,000; 226,000; 320,000.
  - **One gate-only setting is added.** 2,000,000 is a full search for gate 4's known answer. It is excluded from
    every estimand and curve (`analyze_q5.r3_grid`), so W@0.90 is never interpolated beyond 320K; a curve that
    does not reach 0.90 by 320K is reported as not attained, as in PostgreSQL's R3.
- **In-sample caution (S1).** The 8.0 threshold was motivated on this stream. PostgreSQL's E1 reached 14.15× in
  R3, and Qdrant's E1 reached 8.06× at batch 3 and 15.72× at batch 19 in job 25859944.
  - So S1 is not an independent test of the threshold.
  - S2 is: PostgreSQL chose its source slice before any outcome on it.
  - Both are reported, and a verdict that holds on S1 alone is labelled in-sample.

## 4. Estimands and decisions (as PostgreSQL's)
- **Primary estimand** (per plan, MQ test or test). The equal-batch mean of paired Recall@30 at the nominal 50,000
  budget, over all 20 snapshots (I1 has 21 snapshots, 0–20).
  - Three contrasts: E1T − E1, E1R − E1 and E1T − E1R.
  - Paired query bootstrap, stratified by batch: 10,000 resamples, seed 20261127.
  - A sensitivity interval resamples shared query indices jointly across batches.
- **Recovery fraction** = (E1T − E1) / (E1R − E1). It is undefined if E1R − E1 ≤ 0.
- **Criterion 1.** The lower bound of the 95% interval of E1T − E1 is above 0, and the recovery point estimate is
  ≥ 0.5.
- **Criterion 2.** E1T's total trainer walltime is ≤ 50% of E1R's in the same job.
  - "Trainer walltime" is the sum of `sample_read_seconds + teacher_mlp_seconds` over LMI builds that trained, that
    is, `build_completed` with `reused_router = null`. That includes every first-model fit.
  - First-model fits are also reported separately.
  - Census overhead (`census_micros`), decision counts and total maintenance time (`optimizer_micros`) are reported,
    so a training saving cannot hide a costly decision path.
- **The practical verdict** is "met" only if both criteria hold on a plan.
  - It is reported per plan: S1 (in-sample), S2 (out-of-sample) and I1 (descriptive).
  - A zero-firing outcome is valid. E1T then equals E1 by construction (§6), and the verdict is "not met (trigger
    never fired)".
- **Descriptive results:**
  - per-batch recall and vectors scored at every budget;
  - W@0.90 by the first measured crossing, interpolated and never extrapolated, with non-attainment explicit;
  - for every arm, level and batch, the level's live max/mean bucket ratio;
  - every trigger decision: its job, level, live count, buckets, maximum, ratio, decision, router and census time;
  - the firing batches.
- **Cross-engine.** The comparison with PostgreSQL is in vectors scored and recall only, and on S2 the inputs are
  identical. Latency and trainer time are compared within the job only.

## 5. Expectation stated before data
- **S1.**
  - **The basis.** Qdrant E1 on S1 crossed 8× at the end of batch 3 (8.06×). Its level-2 router is reused from the
    batch-1 fit and drifts under Shift.
  - **First firing.** We therefore expect E1T to fire first at the first level-2 compaction of batch 4.
  - **Later firing.** After batch 11's level-3 promotion, the new level-2 E1 trajectory crossed 8× again from batch
    14 (9.84×). E1T is expected to fire again in that range.
  - **Level 3.** E1's level 3 stayed at 3.98× and is not expected to fire.
- **S2.**
  - **No Qdrant trajectory exists yet.** PostgreSQL's trigger saw at most 2.85× in batches 0–2.
  - **Expectation.** Qdrant is expected to fire no later than PostgreSQL. This is an expectation, not a hypothesis
    test, and it is checked only descriptively.
  - **Why Qdrant may fire earlier.** Qdrant drains its buffer per batch (5,025-point drains at batch starts), unlike
    PostgreSQL's global 5,000-point drains. Its level-2 router is fitted on a different population, and its
    routing/assignment step differs. On S1, Qdrant's reuse imbalance crossed 8× by batch 3.
- **I1.**
  - **The basis.** Qdrant E1 stayed at 2.5–2.6× (level 2) and 3.6× (level 3).
  - **Expectation.** Zero firings, so E1T = E1 on every row.

## 6. Gates (all before any plan run; a failed gate is reported, never silently rerun)
1. **Unit tests:**
   - `imbalance_retrain_matches_the_frozen_postgres_rule` checks exact 8.0× (≥), 7.9×, the mean over empty
     buckets, level 1 and empty censuses;
   - `imbalance_retrain_off_keeps_the_serialized_policy` checks that E1's serialized policy keeps its bytes;
   - `imbalance_trigger_off_equals_e1_and_on_retrains_eligible_levels` checks that a never-firing trigger is
     identical to E1 in routers, populations and search results;
   - the same test also checks that an always-firing trigger retrains exactly the eligible occupied levels, and that
     every decision's census equals the destination before compaction;
   - the existing E1/E1R reuse test still passes.
2. **Same-node E1 equality.** On one machine, E1 from the base revision `34dbeffd7` and E1 from the frozen revision
   run the first 3 batches of S1 (validation panel, full grid).
   - `rows.jsonl` must be byte-identical.
   - So must the per-batch publications (levels, populations, router fingerprints).
3. **Local smoke.** E1T runs the first 3 batches of S1 (validation only).
   - Every decision must agree with its build event: reuse means `reused_router` equals the decision's router, and
     retrain means `reused_router = null` and aⁱ buckets.
   - Every census must equal the destination's population at the previous snapshot or publication.
   - A smoke with no firing is acceptable; the unit test covers the firing branch.
4. **Per job:**
   - plan and live-ID ledger hashes;
   - completeness;
   - monotone work in the budget;
   - full-search (the gate-only 2M setting) known answer ≥ 0.999 on every batch and panel, using plan gold. The tie and rounding slack
     from the DLI-best pilot applies, so a shortfall is classified with the tie check before the gate is judged;
   - decision/build agreement as in gate 3.

## 7. Declared departures from PostgreSQL's protocol
1. **Plans.**
   - **Why.** The shared Q5 protocol names S1. PostgreSQL's own trigger run uses S2.
   - **What we do.** We run both and label S1 in-sample. I1 is added because the Qdrant brief asks for Insert-only;
     PostgreSQL has no Insert-only trigger arm.
2. **No windows.**
   - **Why.** PostgreSQL split the stream into seven prefix-replaying windows for its walltime. Qdrant's three-arm
     Phase 3 Shift job took 16.3 h on one node.
   - **What we do.** One job per plan measures all snapshots. If S1's job exceeds 20 h, S2 and I1 use two windows
     (batches 0–9 and 10–19/20). Each window replays the full prefix with all three arms, as PostgreSQL does. Trainer
     totals are then taken from the last window's full prefix, never summed across windows. The equal-batch
     estimand is unchanged.
3. **Census source.**
   - **What it is.** Qdrant's destination is one segment. Its census is the live (non-deleted) points in each of
     its router-assigned postings, the serving generation that searches read.
   - **When it is taken.** The decision is taken when Qdrant's maintenance plans the compaction, not at a SQL
     publication boundary. The prediction/publication agreement check uses build events (gate 3).
4. **Exact comparison.** The rule is evaluated as max · buckets · 1,000 ≥ 8,000 · live, in integers.
5. **Drain schedule.** This is Qdrant's own Phase 3 per-batch chunking, unchanged; Q4 studies alignment.
6. **No operational follow-up.** PostgreSQL's paced-write gate measures its locked worker. Qdrant's concurrency
   semantics differ (Q6), so it is out of scope here.
7. **Gold.** This is plan gold, not the stored-representation G1 gold. Rounding effects (~0.5%, DLI-best pilot)
   are common to all arms, so contrasts are paired over identical gold.

## 8. Execution
- **Resources.** One PBS job per plan (S1, S2, I1): 8 CPUs, 96 GB RAM, 100 GB scratch and 24 h, submitted without
  `-q`.
- **Staging.** A new immutable stage from a pushed commit of this branch, with a source-keyed binary,
  `inputs.sha256`, `SHA256SUMS` and a unique result directory.
- **Order.** S1 first. S2 and I1 follow once S1's gates pass and its runtime is known.
- **Analysis.** `analyze_q5.py`, with a known-answer test, committed before the first job.
