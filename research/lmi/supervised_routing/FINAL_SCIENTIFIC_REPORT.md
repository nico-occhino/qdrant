# Final report: does nearest-neighbour supervision make learned routing beat IVF?

**Outcome: Negative.** By the preregistered gate, supervised routing did not meaningfully outperform native IVF.

A small routing-quality signal exists, about 10% fewer candidates at 95% coverage on held-out queries. It sits at the
edge of the promotion threshold and below what the latency criterion would require. The 10M confirmation was
therefore not run.

## The question

Earlier campaigns (Tiers 1–3) showed that the original LMI loses to centroid IVF at matched recall:
- its router is a 768 → 512 → B MLP;
- it is trained with cross-entropy on k-means labels.

That leaves two explanations open:
- **(a)** learned routing is weak as a concept;
- **(b)** the training objective was wrong, because imitating cluster labels does not reward ranking the buckets that
  hold a query's neighbours.

This experiment kept everything physical fixed: the native IVF-8192 partition, postings, storage, scorer, query
preprocessing and measurement path. Only the router changed. Each router was trained to predict the buckets containing
a query's exact top-10 neighbours.

## What was tested

| Arm | Router | Trainable parameters |
|---|---|---|
| A0 | native centroid | 0 |
| A1 | closed-form LDA head | 0 |
| A2 | per-bucket biases | 8,192 |
| A3 | low-rank correction | 295K |
| A4 | full linear correction to the centroids | 6.3 M |
| A5 | neighbour-loss MLP, same shape as the original LMI | 4.6 M |

- Each arm except A5 was also tested with an occupancy-adjusted ranking (expected neighbours per candidate).
- Every linear arm runs as one `Linear` layer at exactly the centroid router's cost.
- **Training data:** 5,744 official queries, plus 100,000 corpus points as auxiliary queries.
- **Selection:** a fixed grid, selected on 2,000 validation queries only.
- **Gate:** natively, on 1,000 held-out queries, against a preregistered 10% candidate-reduction threshold.

## Findings

1. **The supervised objective is the right direction, but the gain is small.**
   - The best router, A4occ (full linear correction with occupancy ranking), reaches higher recall than centroid IVF at
     every nprobe, with slightly fewer candidates.
   - On held-out queries it needs about 10% fewer candidates at 95% coverage (offline continuous-nprobe diagnostic):
     4,350 against 4,849.
   - On validation it needed 14% fewer. Selection optimism accounts for about 4 points.
2. **Native measurement did not confirm a 10% gain.** At the preregistered grid, both routers first reach R@10 ≥ 0.95
   at nprobe 32: 4,949 against 5,006 candidates (−1.1%), and 2.125 against 2.173 ms median (−2.2%).
   - The gate failed on its first criterion. All correctness, budget and integrity criteria passed.
   - The grid granularity compresses the difference at 0.95 and exaggerates it at 0.97. Neither direction changes the
     preregistered decision.
3. **Even a 10% candidate cut would not give a 10% speed-up.** Routing is a fixed cost: about 23% of the 1M query and
   about 8% at 10M. A 10% cut in scored candidates removes roughly 7% of latency at 1M, and an estimated 9% at 10M
   (unmeasured). The supervised router has the same routing cost as A0; it just cannot remove enough scoring work.
4. **The oracle headroom is large but mostly unlearnable here.**
   - A bucket choice that knows the answer reaches 95% coverage with about 1,000 candidates; centroid routing needs about 4,850.
   - The best learned router closes about 13% of that gap.
   - With 105K training queries, the full correction memorises: train C95 1,794 against validation 3,863.
   - Without the corpus-as-query augmentation, every arm beyond biases overfits badly: 5,744 sparse targets cannot
     support 0.3–6.3 M parameters.
5. **More learned capacity in the wrong form hurts.** The neighbour-loss MLP is 59% *worse* than centroid routing at
   95% coverage, so the right objective does not rescue the hidden-layer architecture at this data scale.
   - The geometry-plus-statistics LDA head is 20% worse than centroids.
   - Occupancy adjustment helps calibrated learned scores but hurts raw centroid scores (+10–16%).
6. **Both explanations hold in part.** The original LMI's weakness was largely its objective: the neighbour-trained
   correction is the only family that beats centroids at all. But even the right objective buys only about 10% fewer
   candidates and single-digit latency, on a CLIP workload where centroid IVF is already a strong router. On this
   evidence, learned *routing* over a fixed geometric partition is not where a learned index can earn a practical
   advantage over IVF.

## Answers to the campaign questions

1. **Commit and baseline.**
   - Code: branch `research/static-supervised-neighbor-routing-20261119`; the pilot ran at `d76aea7b2`, based on
     `912b9c6e3` (the Tier 3A/3B code).
   - Baseline: native IVF-8192 (S 524,288, 20 iterations, seed 42) over the first 1M SISAP23 vectors, verified against
     profile job 25718749.
2. **Was the 1M pilot accepted?** Yes. Exit 0, partition verified, all integrity and correctness checks passed.
3. **Did any model pass the 10% gate?** No. The promoted A4occ reached −1.1% at the native matched point.
4. **Was 10M executed?** No; the gate failed, so the stopping rule applied.
5. **Best learned router and why.** A4occ, the full linear correction trained on neighbour-bucket targets with
   corpus-as-query augmentation, ranked by s/τ − log n_b. It is the only family whose learned scores carry real
   information beyond the centroids, and occupancy ranking turns that into fewer candidates.
6. **Did it reduce total native latency?** Marginally: −2.2% at the matched 0.95 point, and slightly lower at every grid
   point. That is not practically meaningful.
7. **Did it beat IVF at 0.95 Recall@10?** Not by the preregistered criterion: 2.125 against 2.173 ms, 4,949 against
   5,006 candidates.
8. **Routing and scoring costs.**
   - Routing 0.50 ms for both A0 and A4occ (one 8,192 × 768 layer); union 0.05 ms; scoring 1.57–1.61 ms at nprobe 32.
   - Router 25.2 MB, the same as A0; training 94 s.
9. **Did it generalise?** Partly. The held-out gain (about 10% offline) is smaller than the validation gain (14%), and
   much smaller than the training fit.
10. **Strongest limitations.**
    - The coarse preregistered grid near 0.95 (24 → 32).
    - One partition and one seed.
    - 1M only.
    - The G and C queries were reported, never selected on, in earlier campaigns.
    - The augmentation queries come from the corpus distribution.
    - Single-node, warm-cache timing.
11. **PBS jobs.**
    - 25756828: preflight; failed on a training-script bug (fixed).
    - 25758335: preflight; failed on short results, leading to Amendment 2.
    - 25759595: preflight; accepted.
    - 25759596: pilot; accepted; gate failed.
12. **Artifacts.**
    - Code and reports: `research/lmi/supervised_routing/`.
    - Evidence: `research-admin/experiments/2026-11-19-static-supervised-routing/` (raw job copy, analysis, V query-level CSV).
    - Remote: `/storage/budejovice1/home/nico/lmi-metacentrum/static-supervised-routing-20261119/`.

## Scientific outcome

**Negative.** Supervised routing does not meaningfully outperform IVF on the preregistered native criterion. A modest
routing-quality improvement of about 10% fewer candidates on held-out queries is real but too small to matter for
latency. It also did not survive the preregistered gate.

The static learned-routing line closes here. The project moves to dynamic indexing, Qdrant integration and thesis writing.
