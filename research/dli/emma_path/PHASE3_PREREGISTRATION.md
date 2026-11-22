# Preregistration — Phase 3: distribution shift and deletes in native Qdrant DLI

**Status: DRAFT, 2026-11-21. Author decisions recorded (§9).** Not frozen. It is frozen
by a commit that changes this line to "FROZEN" after the tooling (§7) is implemented and
preflighted and the precision check (§5) is applied, all before any Phase 3 result exists. After freezing,
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
- Random: uniform over live IDs, drawn by `phase3_plan.py` with NumPy PCG64 seeded 20261121 + batch index (the plan, not the driver, fixes every delete; the driver checks the live-set SHA-256 per batch).
- Oldest: smallest live insertion sequence numbers. Insertion order is ID order for
  uniform data.

**Shifted data**, following her §4.1:
- Spherical k-means with 10 clusters on the N-row corpus (seed 20261121, 25 iterations).
- Order the clusters as a chain: start at a seeded random cluster, then repeatedly append
  the unused cluster whose centroid is nearest (cosine) to the last one added.
- Insert cluster by cluster in chain order, by ID within each cluster.

**Query sets.**
- **ID** (uniform scenarios): validation rows 900–931 as in Phase 1–2, and **test rows
  9000–9999 (1,000 queries)**, extending Phase 2's 9000–9255 (decision §9.1). They are
  constant across batches.
- **IDS:**
  - Before insertion, hold out 1% of each inserted cluster (seeded); held-out rows are
    never inserted.
  - After batch t, sample 1,000 test and 32 validation queries from the held-out rows of
    the clusters inserted so far, proportional to their inserted counts, seeded per batch.
- **OOD:**
  - The cluster that would come *last* in the chain is withheld entirely, so only 9
    clusters are inserted; the chain is still built over all 10.
  - 1,000 + 32 queries are sampled from it once.
- **MQ:** 1,000 + 32 queries sampled once from the held-out 1% of the last *inserted*
  cluster. They start out-of-distribution and become in-distribution as the data reaches
  that cluster.
- At 1M the 1% holdout is ≈ 1,000 rows per cluster. If a cluster's holdout is smaller than
  the queries it must supply, the holdout fraction is raised for that cluster only, and the
  rows are taken from that cluster before insertion. This is recorded in the manifest.

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

**Why these thresholds (smallest effects of interest).** On the Phase 2 pilot E1 curve
(test, `mdn_code`, budgets 6K–16K), 1 pp of mean R@30 costs about 7–8% more scored
candidates. So 2 pp ≈ 15% more work, half of H1's 30% threshold, and 1 pp ≈ 8%. Effects
below about 0.5 pp (≈ 4% work) are not treated as practically meaningful.

**Outcomes.** Each primary hypothesis has exactly one of these outcomes, decided by the rule
stated with it:
- *supported*;
- *not supported*;
- *inconclusive*, when the CI is too wide to decide: its half-width is greater than the
  threshold (H5, H6) or than the margin (H7).

"Not supported" is used only when the experiment was precise enough to see the threshold
effect.

- **H5, primary (shift).** In Shift-M, consider Δ = E1R − E1 mean recall.
  - **Supported** if Δ ≥ 2.0 pp and the 95% CI lower bound > 0.
  - **Inconclusive** if the 95% CI half-width > 2.0 pp.
  - **Not supported** otherwise.

  Her analogue is 8.4 pp (Table 4.6, No Retrain vs From Scratch). Reported with it: the
  same Δ in Insert-only, and the ratio of training seconds E1R / E1.
- **H6, primary (random deletes).** In Constant-R, consider Δ = E1G − E1 mean recall.
  - **Supported** if Δ ≥ 1.0 pp and the 95% CI lower bound > 0.
  - **Inconclusive** if the half-width > 1.0 pp.
  - **Not supported** otherwise.

  Her difference is 0.5 pp (Table 4.5); we expect more, because Qdrant's dead postings
  consume budget. Reported with it: E1's dead-postings share at batch 20, and rewritten
  points E1G / E1.
- **H7 (oldest deletes, temporal advantage), equivalence by TOST.** In Constant-O, consider
  Δ = E1G − E1 mean recall, with margin M = 1.0 pp.
  - **Supported (equivalent)** if the whole **90%** CI of Δ lies inside ±1.0 pp. That is
    two one-sided tests at 5% each.
  - **Not supported** if the 90% CI lies entirely outside ±1.0 pp.
  - **Inconclusive** otherwise, including whenever the 90% CI half-width > 1.0 pp.

  A non-significant difference is never reported as equivalence. E1G's rebuild count is
  reported.

**Multiplicity.** H5, H6 and H7 address three distinct questions on different scenarios,
each with its own decision. No family-wise correction is applied, and none is claimed. As a
sensitivity check, the report also gives Bonferroni-adjusted intervals (98.33% for H5 and
H6; 96.67% two-sided, i.e. two one-sided 1.67% tests, for H7). It states whether any
outcome would change under them.

**Precision check before primary data.** The preflight-stage noise is estimated from the
1,000 test queries for each primary contrast (SD of per-query differences at B*). If the
projected 95% half-width at 1M exceeds a threshold, the test query count for that scenario
is raised to the number needed, up to 5,000, before any primary run. The number is fixed by
amendment. The pilot proxy (E1R − E1 at 300K, 256 queries) had a per-query SD of 13–24 pp
and a paired SE of 0.8–1.5 pp; with 1,000 queries the expected SE is 0.4–0.8 pp per snapshot.
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

- **Preflight**, N = 100K: the arm pairs of the three primary contrasts (Shift-M E1/E1R,
  Constant-R E1/E1G, Constant-O E1/E1G), 3 batches each, all 1,000 test queries, budgets
  1K/4K/16K. It checks the pipeline end to end and estimates per-query noise for the
  precision check (§5). Its recall values are not interpreted.
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

## 9. Decisions by the author (recorded 2026-11-21, before any Phase 3 data)

The author accepted all six proposals on 2026-11-21 after reviewing the Phase 2 pilot noise
analysis (per-query E1R − E1 differences at 300K):
1. **Queries:** 1,000 test queries per scenario (rows 9000–9999 for ID; 1,000 sampled for
   IDS/OOD/MQ), with a precision check after preflight that may raise the count (§5).
2. **H5 threshold:** 2.0 pp (≈ 15% more work at the operating point).
3. **H6 threshold:** 1.0 pp (≈ 8% more work).
4. **H7:** equivalence by TOST with a 90% CI and margin ±1.0 pp. The ±0.5 pp margin of the
   first draft was dropped: it would need ≈ 4,000+ queries at the observed noise.
5. **Outcomes:** each primary hypothesis is supported, not supported or *inconclusive*
   (CI wider than the threshold or margin).
6. **Multiplicity:** no family-wise correction. The three questions are distinct; a
   Bonferroni sensitivity check is reported.

Also accepted:
- **Scale:** 1M primary, with the conditional 10M confirmation (§6). At 300K, B* is about
  one bucket per level.
- **Budget:** fixed B* = 0.5% of N, with work at R@30 = 0.90 as secondary.
- **Omissions:** Basic rebuild (dominated in her Tables 4.4–4.5) and Partial or Full
  retraining (warm-start training is not implemented in Qdrant) are omitted.
- **Original-idea boundary:** drift-triggered retraining is not an arm; it gets its own
  preregistration after Phase 3.

Remaining before freezing: implement and test M1–M9 (§7) and M10–M12 (§10.6), run the preflight, apply the
precision check, then change the status line to FROZEN.

---

## 10. Part B — the multi-level penalty (added 2026-11-22, before any Phase 3 data)

Motivation: [MULTILEVEL_DIAGNOSIS.md](MULTILEVEL_DIAGNOSIS.md), which is exploratory and
post hoc on test rows 9000–9255 at 10M.
- At 10M, E1 needs 2.4× the work of a matched static LMI.
- Oracle costs show the gap is **structural**: E1's oracle is 2.44× the static oracle, while
  selection efficiency is equal.
- It is caused by the coarse buckets of a nearly full co-resident level (L3: 4,444 vectors
  per bucket against 912 in the static index).

The author asked to test two interventions:
- **consolidation of co-resident large levels**, which attacks structure;
- **level-aware budget allocation**, which attacks selection.

### 10.1 Arms

| Arm | Structure | Selection | Ingestions |
|---|---|---|---|
| E1 | Bentley–Saxe, aⁱ buckets, reuse (Part A) | `mdn_code` | shared with Part A |
| **E1C** | E1 + **consolidation ρ = 0.25** | `mdn_code` | new: Insert-only and Shift-M |
| **E1L** | E1 topology | **calibrated level-aware allocation** | none (replay on E1) |
| **E1CL** | E1C topology | calibrated level-aware allocation | none (replay on E1C) |
| E1C ρ = 0.5 (sensitivity) | consolidation ρ = 0.5 | `mdn_code` | new: Insert-only only, descriptive |

**Consolidation rule (E1C).**
- Let D be the deepest non-empty level.
- When a Bentley–Saxe compaction would leave a level i < D holding more than ρ·live(D)
  points, all levels 0..i are instead merged into D, provided the result fits within
  D's capacity b·a^D.
- The merge reuses D's router, so no training happens, under the same fail-closed reuse
  checks as Part A. Otherwise the compaction proceeds as in E1.
- ρ = 0.25 was chosen from the arithmetic simulation's bucket-cost proxy
  (`consolidation_sim.py`) before any E1C data exists. It is fixed from now on.

**Calibrated level-aware allocation (E1L).**
- *Model:* for each level ℓ, a monotone map g_ℓ from the router probability P of a bucket
  to the expected number of the query's true top-30 neighbours in that bucket.
  - It is fitted by isotonic regression (pool-adjacent-violators) on calibration
    queries, over all (query, bucket) pairs in the top 64 buckets of that level.
  - The Plain buffer gets value 30·|Plain| / N (its expected share under exchangeability)
    and cost |Plain|.
- *Selection:* buckets from all levels are taken in decreasing g_ℓ(P) / live size, which
  is the greedy solution of the fractional knapsack, whole buckets only, until the budget
  is met. This generalises `density` (P / size) by making P comparable across levels.
- *Refitting:* the maps are refitted at each evaluated snapshot, using that snapshot's
  calibration gold.
- *Calibration queries:* rows 1000–2999 (2,000), disjoint from every test and validation set.

### 10.2 Queries for Part B

Confirmatory test queries are **rows 8000–8999** (1,000). The diagnosis did not use them,
and neither does Part A. Using rows 9000–9255 again would test the idea on the data that
suggested it.

### 10.3 Estimand

Insert-only, 1M, uniform data; evaluation after each of the 20 batches; budget grid §6.
- For each batch t and arm, compute W_t, the work at mean R@30 = 0.90, by linear
  interpolation between bracketing budgets (Amendment 1).
- The estimand is the ratio of the means over batches of W_t, where every compared arm
  reaches the target in that batch. Batches where an arm does not reach it are counted
  and reported.
- CIs: paired query bootstrap within batches, 10,000 resamples, seed 20261122.

### 10.4 Hypotheses

- **H9 (consolidation, primary).** E1C vs E1 (`mdn_code`): mean-work ratio ≤ 0.80, i.e. at
  least 20% less work, with the 95% CI upper bound < 1.
  - *Inconclusive* if the CI width exceeds 0.20.
  - Reported with it: write amplification E1C / E1 (predicted ≈ 1.0), training seconds,
    and per-batch ratios, to show the snapshot dependence the simulation predicts.
- **H10 (level-aware allocation, primary).** E1L vs E1 on the same E1 topologies: ratio
  ≤ 0.90 (at least 10% less work), with the CI upper bound < 1. *Inconclusive* if the CI
  width exceeds 0.10. The diagnosis predicts a modest gain.
- **H11 (descriptive).** Each batch reports:
  - E1CL, and the static-final reference at batches 5, 10, 15 and 20;
  - the oracle-cost decomposition (structure × selection) for every arm;
  - E1C ρ = 0.5.
- **H12 (descriptive, the cost of adaptivity).** In Shift-M, E1C − E1 mean recall at B*
  (Part A estimand). Consolidation routes most data through one older router, so it is
  expected to lose under shift. The size of that loss is the trade-off reported against
  H9's gain.

### 10.5 10M confirmation

If H9 is supported at 1M, E1C is run at 10M insert-only, under the same gates as §6.
- One final snapshot, so it is reported as a single-snapshot result, with the snapshot
  caveat.
- Test rows 8000–8999 (official gold).
- Its W@0.90 is compared with E1 from job 25827532, which is evaluated afresh on rows
  8000–8999 by replaying its retained frontier inputs where available, or by rerunning E1.

### 10.6 Implementation required (in addition to §7)

| # | Component | Test that proves it |
|---|---|---|
| M10 | Consolidation rule in `dli_schedule.rs` (pure planner) | unit: the ρ boundary, the fit guard, temporal order, population conservation; replay of `consolidation_sim.py`'s final levels for N = 1M and 10.12M |
| M11 | Calibrated allocation: isotonic fit per level and fractional-knapsack selection (`dli_bucket_plan.rs`), fitted in the replay harness | unit: PAV against a hand-computed example; the selection equals brute force on small inputs; whole-bucket budget semantics; the Plain value rule |
| M12 | Calibration and Part-B query sets with live-set gold at every evaluated batch | M7 extension; disjointness check across all query sets |
