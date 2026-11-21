# Preregistration — Phase 3: distribution shift and deletes in native Qdrant DLI

**Status: DRAFT, 2026-11-21.** Not frozen. It is frozen by a commit that changes this
line to "FROZEN" after (a) the author's review of the decisions in §9 and (b) the tooling
is implemented and preflighted, all before any Phase 3 result exists. After freezing,
changes are appended as amendments only, as in Phase 1–2.

Depends on: [PREREGISTRATION.md](PREREGISTRATION.md) (arms G0/E1/E1R, routing policies,
gold tooling, bootstrap). Phase 2 (10M job 25823237) must finish before Phase 3 runs, but
Phase 3 does not depend on its outcome.

## 1. Motivation

Phases 1–2 test Sommerová's DLI under her easiest workload: insert-only, data in ID order,
which is effectively uniform. Her thesis (§1, Fig. 1.1; §3.3.2–3.4; §4.4–4.5) argues that the
point of a *dynamic* learned index is how it behaves when
1. new data **shifts** across partition boundaries, so a reused router routes it badly; and
2. **deletes** leave levels underutilized, so routers and partitions describe data that no
   longer exists.

Two of her design choices only matter under these workloads:
- **Naive insert into a non-empty level:** a reused router with no retraining (E1).
- **Underutilization rebuild:** Greedy, with No Rebuild as the baseline.

Qdrant changes the delete question:
- **Her code hard-deletes:** the vector leaves its bucket.
- **Qdrant soft-deletes inside immutable segments:** the offset stays in the bucket's
  postings, still consumes routing budget, and is filtered only at scoring
  (`dli_native_route.rs`, test `dli_topk_underfill_uses_complete_native_fallback`;
  comment: "postings still consume routing budget, but deleted offsets are filtered before
  the scorer").
- **Consequence:** without a rebuild, dead postings dilute every learned bucket. This
  effect does not exist in her implementation, and it is the main Qdrant-specific
  question here.

Questions:
- **Q-S:** How much search quality does router reuse lose under gradual shift, and what
  does retraining buy back at what maintenance cost?
- **Q-D:** Under deletes, does Greedy rebuild recover the quality lost to dead postings, and
  what does it cost? Does oldest-first deletion make rebuilds unnecessary, as her temporal
  argument predicts?

## 2. Scenarios (her Table 4.1, adapted)

| Scenario | Workload (insert:delete per batch) | Data order | Queries | Delete method |
|---|---|---|---|---|
| Insert-only (control) | OI 1:0 | uniform (ID order) | ID | — |
| Heavy-R / Medium-R / Constant-R | HI 10:1 / MI 2:1 / CI 1:1 | uniform | ID | random (seeded) |
| Heavy-O / Medium-O / Constant-O | HI / MI / CI | uniform | ID | oldest first |
| Shift-IDS | OI | shifted | IDS | — |
| Shift-OOD | OI | shifted | OOD | — |
| **Shift-M** | OI | shifted | MQ | — |

The three Shift scenarios share one ingestion per arm; only the query sets differ.

**Batches.**
- Delete scenarios and Insert-only: an initial batch of 10% of N (no deletes), then 20
  equal batches. Each batch inserts x = 0.9·N/20 and deletes x/10, x/2 or x.
- Shift scenarios: 20 equal batches and no initial batch, as in her table.
- Search evaluation runs after every batch, once the batch's scheduled maintenance has
  published (synchronous drain, as in Phase 2).

**Delete selection.**
- Random: uniform over live IDs, PRNG `StdRng` seeded 20261121 + batch index.
- Oldest: smallest live insertion sequence numbers. Insertion order is ID order for
  uniform data.

**Shifted data**, following her §4.1:
- Spherical k-means with 10 clusters on the N-row corpus (seed 20261121, 25 iterations).
- Order the clusters as a chain: start at a seeded random cluster, then repeatedly append
  the unused cluster whose centroid is nearest (cosine) to the last one added.
- Insert cluster by cluster in chain order, by ID within each cluster.

**Query sets.**
- **ID** (uniform scenarios): the Phase 1–2 sets, validation rows 900–931 and test rows
  9000–9255. They are constant across batches.
- **IDS:**
  - Before insertion, hold out 1% of each inserted cluster (seeded); held-out rows are
    never inserted.
  - After batch t, sample 256 test and 32 validation queries from the held-out rows of
    the clusters inserted so far, proportional to their inserted counts, seeded per batch.
- **OOD:**
  - The cluster that would come *last* in the chain is withheld entirely, so only 9
    clusters are inserted; the chain is still built over all 10.
  - 256 + 32 queries are sampled from it once.
- **MQ:** 256 + 32 queries sampled once from the held-out 1% of the last *inserted*
  cluster. They start out-of-distribution and become in-distribution as the data reaches
  that cluster.

Validation queries are used only for the secondary "best policy on validation" choice,
as in Phase 2.

**Gold.**
- Exact top-30 over the live set after each batch, float64 cosine, ties by smaller ID
  (`prefix_gold.py` extended to a live-ID mask).
- The extended tool must reproduce Phase-2 prefix gold exactly on an insert-only batch.

## 3. Arms

| Arm | Compaction into non-empty target | Fresh level | Underutilization rebuild | Used in |
|---|---|---|---|---|
| **E1** | reuse target router (her code) | train from scratch | none (No Rebuild) | all scenarios |
| **E1G** | reuse target router | train from scratch | **Greedy** (her §3.4.2) | delete scenarios |
| **E1R** | retrain from scratch | train from scratch | none | Insert-only, Shift |

Everything else is E1 from Phase 1: b = 5,000, a = 10, aⁱ buckets, hidden 128, 10 epochs,
25 k-means iterations, sample min(n, max(100K, 40·B)), routing `mdn_code`.

Not replicated, and why:
- **Leveling compaction:** her best is Bentley–Saxe, already used.
- **Basic rebuild:** dominated by Greedy in her Tables 4.4–4.5.
- **Partial or Full retrain:** they need warm-start training, which is not implemented.
- **Her "No Retrain" of an emptied level:** Qdrant does not keep an emptied level's router.

**References**, recomputed at batches 5, 10, 15 and 20 only:
- **Static-final:** one LMI trained on the current live set, with the matched bucket total
  (Phase 1 rule).
- **Static-initial:** an LMI trained on the first batch, with every later point routed by
  its fixed router. This is her §2.6 baseline, showing what any dynamic index must beat.

## 4. Measurements (per batch, per scenario, per arm)

1. **Recall.** R@10 and R@30 at the candidate-budget grid (§6), by selection-frontier
   replay on the frozen topology after the batch (same production
   `route_pinned_traced` + `resolve_pinned_candidates` as Phase 2).
2. **Work, split into three parts per query:**
   - selected postings (budget consumed);
   - **dead postings** (selected but deleted);
   - scored vectors (live learned + Plain).
   The dead-postings share = dead / selected. The trace (`RouteTrace`,
   `LearnedScoringProfile`) does not yet record dead postings explicitly. M3 (§7) adds the
   counter and verifies it against an independent count.
3. **Topology:** live and physical points per level; buckets; largest bucket; levels below Cmin(i).
4. **Maintenance:** publications, rebuilds (triggering level, target, merged levels),
   trained or reused builds, training seconds, rewritten points, optimizer wall seconds,
   batch wall seconds.
5. **Collection pass:** 100 queries at the arm's budget, counting learned vs exact
   fallback (provenance, see the 16 Nov lesson).

## 5. Hypotheses and decision rules (scale N = 1M, test queries)

Primary budget B* = 0.5% of N = 5,000 candidates, the same 0.5% she used (50K at 10M).
"Mean recall" is the mean over batches 1–20 of mean R@30 at B*. CIs are 95% paired query
bootstraps:
- 10,000 resamples, seed 20261121;
- queries resampled within each batch, using the same resample for both arms;
- for IDS, the per-batch query sets are the resampling units.

- **H5, primary (shift).** In Shift-M, E1R − E1 mean recall ≥ 2.0 pp, with CI lower bound
  > 0. Her analogue is 8.4 pp (Table 4.6, No Retrain vs From Scratch). Reported with it:
  the same difference in Insert-only, and the ratio of training seconds E1R / E1.
- **H6, primary (random deletes).** In Constant-R, E1G − E1 mean recall ≥ 1.0 pp, with CI
  lower bound > 0. Her difference is 0.5 pp (Table 4.5); we expect more, because Qdrant's
  dead postings consume budget. Reported with it: E1's dead-postings share at batch 20, and
  rewritten points E1G / E1.
- **H7 (oldest deletes, temporal advantage).** In Constant-O, |E1G − E1| mean recall
  < 0.5 pp, with the whole CI inside ±0.5 pp. E1G's rebuild count is reported.
- **H8 (descriptive).** Every scenario and arm:
  - per-batch recall curves and work at mean R@30 = 0.90 (linear interpolation as
    Amendment 1, where reached);
  - maintenance totals;
  - the static-initial and static-final references;
  - Heavy and Medium delete variants;
  - Shift-IDS and Shift-OOD.

Negative results are reported as such. Thresholds, B*, query sets, seeds and the primary
contrasts do not change after Phase 3 data exist.

## 6. Scale, budgets and stages

- **Preflight**, N = 100K: Constant-R E1G and Shift-M E1 only, 3 batches, 8 queries,
  budgets 1K/4K/16K. It checks the pipeline end to end; its numbers are not interpreted.
- **Primary**, N = 1M: all scenarios and arms in §2–3. That is 13 ingestions: 1 + 6 + 6
  for Insert-only and the uniform delete scenarios across arms, plus 2 for Shift (E1,
  E1R). Budgets: 1K, 2K, 3.5K, 5K, 7K, 10K, 14K, 20K, 28K, 40K, 56K.
- **10M confirmation** of H5 and H6, under all of these conditions:
  - the 1M stage passes the engineering gate (§8);
  - E1R's projected 10M ingestion is ≤ 40 h, using the Phase 2 fit
    (`analyze.py::projection_10m`, with retraining on every non-empty target);
  - the per-publication wait exceeds 3× the longest projected publication (Amendment 3).

  If E1R exceeds 40 h, H5 is decided at 1M only and this is reported. At 10M,
  B* = 50,000, as in her runs. Budget grid: 10K, 20K, 35K, 50K, 70K, 100K, 140K, 200K.

## 7. Implementation required (to be done and tested before freezing)

| # | Component | Where | Test that proves it |
|---|---|---|---|
| M1 | Underutilization detection: live(i) < Cmin(i) = b·aⁱ⁻¹, evaluated after each batch's deletes | `dli_schedule.rs` (pure planner) | unit: thresholds at the boundary; buffer and empty levels never underutilized |
| M2 | Greedy target selection and merge set, her §3.4.2 and Algorithm 11, preserving temporal order | `dli_schedule.rs` | unit: each preference rule (lower level with room / self / i−1), merged levels contiguous, populations conserved (live counts) |
| M3 | Dead-posting counters in the resolution trace, checked independently | `dli_native_route.rs` | unit: an all-deleted bucket, a partially deleted bucket, after reopen |
| M4 | Rebuild publication through the native optimizer path (same build, reuse rule and fail-closed checks as compaction) | `dli_maintenance.rs` | collection test at b = 40, a = 4: random and oldest deletes; rebuild drops dead points physically; full-budget search = exact search over live points; 0 fallbacks |
| M5 | Workload driver: batches, insert:delete ratios, delete methods, per-batch drain and evaluation | `dli_sisap_scaling_benchmark.rs` (new arm/scenario env vars) | small run: live count = expected after each batch; deterministic delete set for a fixed seed |
| M6 | Shift ordering, holdouts and query pools | `research/dli/emma_path/shift_data.py` | chain order deterministic for the seed; holdouts never inserted; manifest hashes |
| M7 | Live-mask gold | `prefix_gold.py --live-ids` | equals Phase-2 prefix gold on an insert-only live set |
| M8 | Per-batch frontier replay, plus the static references at 4 checkpoints | `dli_selection_frontier.rs`, PBS | preflight |
| M9 | Analysis: per-batch curves, mean-recall estimands, batch-stratified paired bootstrap | `analyze_phase3.py` | synthetic input with a known answer |

Out of scope: crash recovery during a rebuild, filtered search, concurrent writers during
evaluation.

## 8. Engineering gate before 10M (from the 1M stage)

1. Every scheduled publication and rebuild succeeds; the hierarchy descriptor validates
   after every batch.
2. Live points per batch equal the workload ledger exactly.
3. The collection pass is learned for 100/100 queries with 0 exact fallbacks, at the last batch of every scenario.
4. Rebuild events match the M1/M2 planner replayed offline from the logged occupancies.
5. The E1R 10M projection and the per-publication wait rule (§6) hold.

## 9. Decisions for the author before freezing

1. **Thresholds:**
   - H5: 2.0 pp. Her effect is 8.4 pp, so this is conservative.
   - H6: 1.0 pp.
   - H7: equivalence margin ±0.5 pp.
   Are these the effect sizes you consider practically meaningful?
2. **Primary scale 1M with a conditional 10M confirmation.** The alternative is 300K
   primary, which is cheaper, but buckets are about 3K vectors and there are only 3 levels.
3. **Primary budget B* = 0.5% of N.** Her fixed-candidate convention, not matched recall.
   Work at R@30 = 0.90 stays as a secondary (H8), because under shift some curves may
   never reach 0.90.
4. Is it acceptable to omit Basic rebuild and Partial or Full retraining (§3)?
5. **Original-idea boundary.** A drift-triggered retrain ("reuse until drift") is
   deliberately not an arm here. Phase 3 measures the reuse penalty that such a method
   would have to recover, and the method gets its own preregistration later.
