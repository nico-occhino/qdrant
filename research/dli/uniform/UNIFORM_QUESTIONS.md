# Qdrant answers to the shared question set (Q1–Q11)

This is the Qdrant side of `CROSS_ENGINE_SYNTHESIS.md` (PostgreSQL, pinned in [PINS.md](PINS.md)).
- **Branch.** `research/uniform-questions-20261127` of nico-occhino/qdrant. Cite a file by commit and SHA-256.
- **Shared conventions:**
  - cross-engine comparisons are in vectors actually scored;
  - latency is compared only within an engine and a job;
  - the shared metrics are W@0.90 (work at mean Recall@30 = 0.90, first measured crossing, never extrapolated) and
    mean Recall@30 at nominal 50K;
  - every estimand uses a paired query bootstrap with 10,000 resamples.

| Q | Question | Qdrant status | Protocol / result |
|---|---|---|---|
| Q1 | Multi-level penalty at 10M | **done** (Emma path) | E1 scored 81,580 vs static LMI 34,023 at W@0.90; E1/static = 2.40 [2.03, 2.79]. `research/dli/emma_path/RESULTS_10M.md` |
| Q2 | Retraining under drift | **done** (Phase 3) | Shift MQ, E1R − E1 = +11.44 pp at 50K. Level-2 max/mean reached 10–16× with reuse, ≈2× with retraining. Job 25859944 |
| Q3 | Insert-only and Constant-R | **done** (Phase 3) | Insert-only E1R +0.54 pp; consolidation W ratio 0.841 [0.831, 0.852]; Constant-R E1V +1.20 pp [1.15, 1.24]. `research/dli/emma_path/` |
| Q4 | Snapshot parity | **(a)/(a′)/(b) done; (c) blocked** | Batch 11: Qdrant at PostgreSQL's schedule falls 91.35% → 67.19% (S = −24.17 pp [−33.65, −15.31]); the remaining gap to PostgreSQL is +1.67 pp [−8.44, +11.35]. [RESULTS_Q4.md](RESULTS_Q4.md) |
| Q5 | Imbalance trigger | **done** | S1 (in-sample) met: +7.89 pp, recovery 0.690. S2 (out-of-sample) not met: +0.24 pp [0.10, 0.38], recovery 0.213; PostgreSQL on S2: +1.20 pp, recovery 0.394, not met. I1: no firing. [RESULTS_Q5.md](RESULTS_Q5.md) |
| Q6 | Maintenance under load | **descriptive only** | Concurrency report at 2bbb962b: busy fraction 2–14.9% (16-dimensional exhaustive fixture). Not ported (different publication semantics) |
| Q7 | Filtering | **running** | Frozen `e85c176f7`; job 26076922. [Q7_PREREGISTRATION.md](Q7_PREREGISTRATION.md) |
| Q8 | DLI-best vs E1 | **done** (pilot; both engines) | **Qdrant:** ratios 0.614–0.909 on validation; criterion 3 fails on Float16 rounding. `research/dli/dli_best/RESULTS_INTEGRATION_PILOT.md`. **PostgreSQL** (native pilot, `DLI_BEST_NATIVE_PILOT_RESULTS.md` at `126fb63`, pinned): DLI-best / E1 work at 0.90 = 0.901 [0.852, 0.988] Insert-only val; Shift 0.882 [0.835, 0.930] MQ, 0.779 [0.730, 0.825] IDS, 0.707 [0.684, 0.770] OOD; known answer 1.000; final index 3.72 → 4.27 GiB (Insert-only), +4 fits. PostgreSQL's **Constant-R arm is pending a defect investigation** (exhaustive-work mismatch on both attempts, jobs 26074468 and 26074838) |
| Q9 | Final benchmark | **running** (another session) | `research/dli/final_benchmark/PREREGISTRATION.md`, amendments A1–A3 |
| Q10 | Memory budget | **spec draft** (shared) | [SCALE_SHARED_SPEC.md](SCALE_SHARED_SPEC.md) §Q10; awaiting PostgreSQL acceptance and P0 |
| Q11 | Distributed scale | **spec draft** (shared) | [SCALE_SHARED_SPEC.md](SCALE_SHARED_SPEC.md) §Q11; after Q10's U runs |

## Q4. Snapshot parity
- **The question.** At the same live IDs, how much of the Shift batch 10–12 cross-engine gap comes from the
  publication schedule, how much from router fitting, and how much is left for the engine?
- **The plan.** S1, Phase 3 Shift seed 20261121, MQ validation (32 queries), batches 10, 11 and 12. Three arms
  are planned:
  - **(a) as run.** The archived Qdrant E1 (job 25859944), plus a local replay of Qdrant's own per-batch drain
    schedule, which measures host-to-host router noise.
  - **(b) PostgreSQL's schedule.** Qdrant E1 is replayed with `DLI_P3_DRAIN_ALIGN=global`, so every drain holds
    exactly 5,000 points across batch boundaries, as PostgreSQL's drains do.
    - **The known-answer check.** All 117 publication targets and populations must equal
      `postgres_schedule-publications.jsonl`, and the live-ID digests must match PostgreSQL's captures.
  - **(c) PostgreSQL's routers.** Qdrant scores the topology from (b) with routers trained by PostgreSQL.
- **(c) is blocked.**
  - **What is missing.** PostgreSQL's topology replay archived router hashes (`model_sha256`), not weights. Its
    parity tool converts Qdrant routers *into* PostgreSQL.
  - **What we need.** PostgreSQL's level routers (LMI1) at batches 10–12 of both schedules, from the PostgreSQL side.
  - **What we will do.** Prepare an LMI1-to-Qdrant importer, with a logit known-answer check against PostgreSQL's
    `router_logits`.
- **The decomposition (preregistered before (b) runs):**
  - **schedule** = (b) − (a′);
  - **router fit** = (c) − (b);
  - **residual engine** = PostgreSQL (original schedule) − (c), at matched live IDs.

## Q7. Filtering
- **The fixture.** Implemented exactly as `FILTER_FIXTURE_SPEC.md`: rows [2,600,000, 2,660,000), the SplitMix64
  `filter_rank`, predicates ≤ 60 / 600 / 6,000 / 30,000, 32 queries, k = 30, and gold over Qdrant's stored
  representation.
- **The arms:**
  - static LMI and DLI, each with payload-index pre-filtering and an exact fallback below a selectivity threshold;
  - HNSW filtered, as the baseline.
- **What is recorded:** recall, underfill, scored work (approximate and fallback separately) and the payload
  enumeration.
- **The sequence.** It is prepared and tested locally first, and submitted after Q5's first job validates.
