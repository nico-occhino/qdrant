# Open issues

## Scientific

1. **Grid granularity at the decision point.** The preregistered native grid steps from nprobe 24 to 32 exactly where
   both A0 and A4occ cross R@10 = 0.95. The gate decision (−1.1% candidates) is therefore partly a grid artifact. The
   post-hoc continuous diagnostic gives −10.3%. The decision stands as preregistered. A finer grid (24, 26, 28, 30, 32)
   would sharpen the native estimate, but it would be a new experiment, and no positive outcome is expected from it
   given finding 3.
2. **The bootstrap re-selects operating points.** The CI of the matched latency ratio is skewed and lumpy, because grid
   points flip in and out of feasibility across resamples. A continuous-frontier estimator (an interpolated latency
   at fixed recall) would give better-behaved intervals, but it was not preregistered.
3. **Distribution of the augmentation queries.** P comes from the corpus distribution. Baseline difficulty matches T
   and V, but distributional equivalence with the official queries was not tested. A different query workload (for
   example, text-to-image) might reward supervision more.
4. **One partition and one seed.** Every result is for one native IVF-8192 build at 1M (fingerprint `7c61bcbf…`). The
   size of the learnable headroom may differ at 10M, where buckets are 10× larger.
5. **Query exposure.** G (rows 4000–4999) and C (5000–5999) were part of earlier campaigns' test splits. They were
   never used for selection, but they are not pristine.
6. **MLP arm.** Only one A5 configuration per data source was preregistered, so its poor result is not an
   architecture study.

## Numerical and systems

7. **Native against offline ranking.** 11 of 15,400 top-nprobe prefix checks disagreed (1–4 per arm), consistent with
   SIMD-accumulation near-ties. Not individually traced.
8. **1M partition check.** One candidate-count mismatch out of 6,000 checks against the profile job built before
   `aaca97f24`. It is attributed to the router's SIMD change and not traced further.
9. **Training-time anomaly.** Some T-only configurations ran far slower than their step counts imply (up to 4,638 s).
   Not diagnosed; selected models unaffected.
10. **Short results.** 239 queries at small nprobe returned fewer than 10 hits, all explained by too few candidates. The
    decomposed path has no underfill widening (as in Tier 3). None occurred at matched operating points.
11. **No QPS.** This campaign measured single-thread latency only, as preregistered. Multi-thread throughput is not
    reported.
12. **No figures produced remotely.** The remote Python environment has no matplotlib, so figures were generated
    locally from the persisted raw records.
