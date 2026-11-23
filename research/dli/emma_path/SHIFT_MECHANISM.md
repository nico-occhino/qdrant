# Why router reuse loses recall under shift, and why consolidation helped (exploratory)

**Status: exploratory, post hoc**, 2026-11-23.
- **Data:** job 25859944 only (Phase 3 Shift-M, 1M, seed 20261121). These are the test panels already analysed for H5
  and H12. The decisions in [RESULTS_PHASE3_SHIFT.md](RESULTS_PHASE3_SHIFT.md) stand; this note changes only their
  *explanation*.
- **Sources:**
  - each arm's `batches.jsonl`: per-level live counts and router hashes after every batch;
  - `batch_XX/topology.json`: largest bucket per level;
  - `batch_XX/rows.jsonl`: per-query R@30 and buckets selected per level, at B\* = 50K with `mdn_code`.

  Extract: [`derived/shift_mechanism_25859944.json`](derived/shift_mechanism_25859944.json).

## 1. Router provenance: which router routes which data, and when

A router hash changes only when a level is built with a freshly trained router.

| Batches | E1 (reuse) | E1R (retrain) | E1C (consolidate) |
|---|---|---|---|
| 1–10 | **one L2 router (`686640`), trained at batch 1 on the first 50K points**, routes L2 as it grows to 450K | a new L2 router every batch | the same `686640` routes L2 up to 475K |
| 11 | L3 (1,000 buckets) built fresh on 500K points: `5f4c16` | identical `5f4c16` | L3 built fresh on 525K: `3169a1` |
| 12–19 | a new L2 router `528f22`, trained at batch 12 on 50K points, routes L2 up to 400K | a new L2 router every batch | L2 is merged into L3 at batches 14 and 19, **routed by the batch-11 router** |

**Training is deterministic given the same data.** E1 and E1R train byte-identical routers whenever they train on
the same point set: L3 `5f4c16` at batch 11, L2 `528f22` at batch 12, and L1 `a5d883` / `28f10e` at batches 8 and 18.

## 2. The reuse penalty is bucket imbalance

The table shows the largest L2 bucket relative to the mean (live / 100), next to the MQ R@30 gap at B\*:

| Batch | 2 | 4 | 6 | 8 | 10 | 13 | 15 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 max / mean | 4.1 | 10.1 | 11.1 | 10.7 | 10.5 | 7.8 | 9.0 | 9.5 | 10.8 | **15.7** (62,921 points) |
| E1R max / mean | 1.9 | 2.5 | 2.3 | 1.9 | 2.0 | 2.2 | 2.8 | 2.0 | 2.2 | 2.1 (8,421) |
| E1R − E1, MQ pp | +0.7 | +14.1 | +22.1 | **+27.5** | +24.6 | +3.2 | +5.8 | +10.6 | +19.0 | +15.7 |

- **A reused router collapses new clusters into a few buckets.** Its centroids were fitted to one earlier cluster.
  Points from later clusters fall into whichever buckets are nearest, so a handful of buckets grow to 10–16× the
  mean. A fresh router keeps the largest bucket at about 2× the mean.
- **That collapse is what costs recall at a fixed budget.** At batch 9, B\* = 50K buys E1 4.1 L2 buckets on average,
  against 10.0 for E1R. At batch 19 it buys E1 6.5 buckets in total, against 70.5 for E1R. The budget is spent on
  oversized buckets, so fewer of the query's neighbourhoods are covered.
- **The gap follows the imbalance.** Across batches 1–19, excluding batch 11, which has no L2: Pearson r = 0.75 between
  E1's imbalance relative to E1R and the MQ gap (n = 18). The batches share one stream, so this is descriptive.
- **The penalty is periodic.** It grows while one reused L2 router absorbs new clusters (batches 2–10 and 13–19). It
  collapses at batch 11, when L2 is merged into a freshly trained L3, and E1 recovers to 0.916. Its size therefore
  depends on where the snapshots fall in the Bentley–Saxe cycle, just as the consolidation proxy does
  ([MULTILEVEL_DIAGNOSIS.md](MULTILEVEL_DIAGNOSIS.md)).
- **What this does not show** is whether a stale router also *ranks* buckets worse for a query. Imbalance and
  ranking are confounded in these rows. Separating them needs an oracle-cost replay on E1's frozen snapshots: if the
  oracle cost itself rises with imbalance, the problem is structural.

## 3. Consolidation helped through granularity, not router freshness

The earlier explanation, in RESULTS_PHASE3_SHIFT.md point 3, was that "merging into a fresh deepest level trains a new
router on a larger, more recent sample". The ledgers contradict it:
- **E1C does not train more often:** 155 reused builds against E1's 159.
- **E1C's dominant router is older, not newer.** At batch 19, 875K points (97%) sit under the batch-11 L3 router. In
  E1, the recent data sit under the batch-12 L2 router.
- **Before batch 11 the two arms are the same.** In batches 0–10, E1C − E1 contributes +0.17 pp of the +3.32 pp mean
  (MQ), because both carry the same stale L2 router. The remaining **+3.14 pp comes from batches 11–19.** IDS and OOD
  show the same split.
- **What changes after batch 11 is the destination granularity.** E1 accumulates the newest clusters in a
  100-bucket L2 (largest bucket 62,921). E1C merges them into the 1,000-bucket L3 (largest 18,087; mean 888). That
  router is also reused and also imbalanced (about 20× its mean, against 4× for E1's fresh L3), but over ten times
  as many buckets. At B\* it buys E1C 20.8 L3 buckets against E1's 6.5 buckets in total.

So consolidation helped because it put the drifting data into finer buckets. That is the same structural factor as
at 10M. Router age did not help.

## 4. Consequences

1. **A drift trigger is available without queries.** A level's bucket-size imbalance after a reuse publication (the
   largest bucket over the mean, or a Gini index) separates stale routers from fresh ones here: the stale/fresh ratio reaches 4–7× in the stale phases. It costs a
   pass over bucket sizes, which the topology already records. That makes it a concrete candidate trigger for
   "reuse until drift" (selective retraining):
   - retrain when imbalance crosses a threshold;
   - fix the threshold on validation;
   - compare the arm against E1 and E1R at matched training seconds.
2. **The level-aware allocation question (H10) and the `mdn_code` against `proportional` replay should be read in
   the light of this.** With a stale level holding a few giant buckets, the two policies may differ mostly in how
   much budget they sink into that level. The policy-replay level diagnostics (S3) record exactly that.
3. **H12 stays as decided:** a descriptive +3.32 pp. Its mechanism is now granularity. The cost-proxy logic of
   MULTILEVEL_DIAGNOSIS applies: the gain depends on the snapshot.

## Limits

- **One stream and one chain order.** The periodic pattern depends on N = 1M and b = 5,000.
- **Imbalance is measured at the maximum bucket only.** `topology.json` records `bucket_size_max`, not the whole
  distribution.
- **The correlation is between batches of one trajectory.** It is descriptive and not an independent sample.
