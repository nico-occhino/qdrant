# CLI shared algorithm specification (Qdrant and PostgreSQL)

**Status: DRAFT for cross-engine agreement.** No CLI code or data exists yet. Each engine freezes its own CLI
amendment (Qdrant: final-benchmark §3.6 and A5) only after both sides accept this text, or an amended version
pinned by commit and SHA-256.

- **The source.** Michal Jakubík, *Continually Learned Index for Dynamic ANN Search*, FI MU 2026 (the thesis). Where
  the thesis and its code (`cli_ablation`) disagree, the thesis is followed and the difference is named.
- **Known code defects.** `cli_final` is not a reference: its config keys disable every trigger (audit A1). The audit
  is `CLI_AUDIT_AND_READING_GUIDE.md` (author's notes, 2026-11-23).
- **Labels.** [T] = the thesis section; [C] = the `cli_ablation` location.

## 1. The index
- **Structure.** One flat learned index: an MLP router d → 128 (ReLU) → |B| [T §2.1], over |B| mutable buckets that
  hold point ids.
- **The metric.** Cosine. Both engines route their stored normalized Float16 vectors, widened to Float32.
- **Search** [T Alg. 3].
  - **Bucket order.** Rank buckets by router logit (ties by bucket id).
  - **Whole buckets.** Take *whole* buckets until the candidates reach the budget sb, a soft minimum.
  - **Scoring.** Score every live candidate exactly and return the top k.
  - **Same rule as DLI.** This is the whole-bucket rule of DLI `mdn_code`, so work = live candidates scored.

## 2. Build, on the first plan batch (batch 0 = D_init)
1. **Bucket count.** |B| = ⌊|D_init| / (0.5 · b_max)⌋, at least 2 [T eq. 2.1]. The code floors before dividing
   (audit A4); this spec follows the thesis.
2. **Training sample.** |S| = ⌊0.1 · |B| · b_max⌋, a uniform sample of D_init without replacement, capped at
   |D_init| [T eq. 2.2].
3. **Labels.** Spherical k-means on S with k = |B|, 20 iterations [C `utils.py:50–62`].
4. **Training.** Adam, lr 1e-3, batch 256, 30 epochs, cross-entropy [T Table 3.2].
5. **Insert D_init** by argmax routing.
6. **The entropy baseline.** H_init = the mean router entropy, in bits, over the vectors of D_init not in S [C
   `cli.py:137–146`].
7. **The reclustering baseline.** |B|_last = |B|.

## 3. Operations
- **Insert** [T Alg. 1]. Route by argmax, then append the id to that bucket.
- **Update.** An update of an existing id is a delete followed by an insert.
- **Delete** [T Alg. 2]. The point is removed from its bucket, or marked dead and excluded from every count and from
  search.
  - **Empty buckets.** A bucket left with no live points is removed, with its output neuron, at the next maintenance
    step. That is lazy removal, as in the code [C `cli.py:580–606`]. The thesis removes it at once; for search the
    two are equivalent.
- **All counts are live counts.** Bucket sizes used by any rule below count live points only.

## 4. Maintenance, after every plan batch [T §2.3, Alg. 4; cadence "after each batch", not the code's 1,000 inserts, audit A6]
**4.1 Trigger evaluation.**
- **Signals,** both evaluated on the batch just inserted:
  - **entropy:** the mean router entropy of the batch's inserted vectors, in bits, is > H_init + 0.3 [T §3.6,
    eq. 3.6]. This is the absolute term only; the code's hidden relative term is set to 0 (audit A7);
  - **growth:** |B| > 10 · |B|_last [T §3.7].
- **Paths.** If either fires, take the reclustering path (4.3); otherwise take the splitting path (4.2).

**4.2 Splitting path.**
1. **Split every overflowing bucket.** For each bucket with n > b_max live points, k = ⌈n / b_max⌉ (eq. 2.3).
   Spherical k-means with that k runs on a uniform sample of min(n, 100 · k) of its vectors (eq. 2.4); every vector
   of the bucket is then labelled by its nearest centroid.
   - **b_max is a maintenance threshold, not a storage invariant.** k-means does not balance sizes, and reinsertion
     is by argmax, so a child can still exceed b_max. Following the thesis, no extra split is made within the same
     step. The child is split at the next maintenance, and the largest bucket is reported (§7).
2. **Grow the output layer.** The parent keeps its id and output neuron. The k − 1 new neurons copy the parent's
   output weights and bias [T Fig. 2.3; C `models.py:62–94`].
3. **Fine-tune.**
   - **Data.** The split buckets' sampled vectors with their k-means labels, plus a replay buffer: R vectors drawn
     uniformly at random from the non-split buckets, split evenly across them.
   - **Replay labels.** Each replay vector is labelled with its **stored** bucket assignment, never with the
     current router's prediction. Regenerating labels from the router would teach it to repeat its own drift, as in
     Forgáč's self-labelling (audit F1).
   - **Optimizer.** A fresh Adam (lr 1e-3, batch 256), 20 epochs, starting from the current weights [C
     `cli.py:366`].
4. **Reinsert** every vector of the split buckets by argmax under the fine-tuned router. Vectors in other buckets
   keep their assignment, which is the routing-accuracy loss the thesis measures.
5. **Remove empty buckets** and their neurons.

**4.3 Reclustering path (DEDRIFT)** [T §3.5, Alg. 4 l. 7].
1. **Select the buckets to recluster.** The rule is exact, as in `DedriftReclusteringStrategy`, [C
   `reclustering.py:206–330`]. Sizes are live counts.
   - **The largest buckets.** k = round_half_even(0.5 · |B|), clamped to [1, |B|]. If at most k buckets are
     non-empty, nothing is reclustered. B1 = the k largest buckets (ties by lower bucket id).
   - **The median.** μ is the median of the non-empty sizes, taking the *lower* middle value for an even count (as
     `torch.median` does).
   - **The smallest buckets.** k₂ = clamp(⌈|B1 sizes| / μ⌉, k, |B|). B2 = the k₂ − k smallest non-empty buckets
     not in B1 (ties by higher bucket id first, as the code iterates the reversed sort).
   - **The selection.** B1 ∪ B2 ∪ {every bucket with size > b_max}.
2. **Recluster.** The selected vectors are pooled (N of them) and reclustered by spherical k-means into
   k_new = ⌈N / (0.5 · b_max)⌉ clusters (thesis Alg. 4 l. 7; the code's default fill ratio is 1.0, and this spec
   follows the thesis).
   - **The k-means sample.** min(N, ⌊0.1 · b_max · k_new⌋) vectors (eq. 2.2's ratio, as in the code), seeded.
   - **Labels.** Every pooled vector is labelled by its nearest centroid.
3. **Initialize the new neurons.** Each new bucket's neuron copies the weights of the bucket that its centroid routes
   to under the current router [C `reclustering.py:121–169`].
4. **Fine-tune, reinsert and remove empty buckets,** as in 4.2 steps 3–5.
5. **Reset the reclustering baseline:** |B|_last = |B|.

- **A note on cloned neurons.** Copying the parent's weights makes the siblings' logits equal at first, but it changes
  the softmax normalisation. It is an initialisation, not a function-preserving expansion, so the entropy of the
  next batch is measured after fine-tuning.

## 5. The single variant, CLI-I (author, 2026-11-28)
- **The change.** CLI-I replaces the entropy trigger of 4.1 with the frozen bucket-imbalance rule of the Q5 trigger:
  fire if max live bucket · |B| · 1,000 ≥ 8,000 · live, in exact integers, over all allocated buckets. The growth
  trigger and everything else are unchanged.
- **A structural expectation, stated before data.**
  - **When it is evaluated.** The trigger runs on the state after a batch's inserts and before that batch's
    splitting.
  - **Uniform streams.** Inserts spread over the buckets, the mean is near 0.5–1 × b_max, and max/mean ≥ 8 needs a
    bucket to receive about 4–8 × b_max in one batch. CLI-I is therefore expected to fire rarely on Insert-only.
  - **Shift streams.** A 45,000-point batch concentrated in a few buckets can exceed 8× before it is split. For
    example, with b_max = 1,000 and 200 buckets, one bucket receiving about 10,000 points gives about 14×. CLI-I is
    expected to fire under Shift.
  - **Deletes.** Constant-R can also raise the ratio through sparse buckets.
  - **A zero-firing CLI-I** equals CLI with its entropy trigger disabled (splitting plus growth only). That is a
    valid outcome and is reported as such.

## 6. Parameters still to fix (by each engine's CLI amendment, before CLI data)
- **b_max and the budget grid.** b_max is chosen by a selection pilot on seed 20261202's validation panels, from
  {500, 1,000, 2,000}, with the lowest mean validation W@0.90, as the final benchmark's §8. The budget grid is the
  final benchmark's.
- **R, the replay size.** The thesis uses 5,000 at 300K and the code 50,000. Proposal: R = 0.05 · |D_init|, which is
  5,000 at a 100K build.
- **Seeds.** k-means, sampling and initialisation use seed 42, as the thesis does.

## 7. What each engine reports per batch (the cross-engine record)
- **Shape:** |B|, the live max/mean and the bucket-size histogram.
- **Maintenance:** the path taken (split or recluster) and which trigger fired, with the entropy H_batch and
  H_init; the number of buckets split, created and removed; the points reassigned; the fine-tune seconds and
  epochs; and the training-sample size.
- **Correctness:** a known answer after maintenance. Every live point is in exactly one bucket, and the size totals
  equal the live count.
- **Routing-accuracy loss** [T §1.2]: L_route = 1 − (share of live points in untouched buckets whose argmax under the
  new router equals their stored bucket), computed after every fine-tune. This is algorithmic drift between the
  router and stored assignments, distinct from any publication defect.
- **Trigger diagnostics.** |B|, H_batch − H_init and the mean and maximum batch logit are recorded every batch,
  whether or not a trigger fires. Entropy's scale changes as |B| grows, so the +0.3-bit rule is a frozen
  experimental setting, not a calibrated constant.
- **Search:** recall and vectors scored over the final-benchmark grid (W@0.90, R@30 at 50K), from the frozen
  snapshot of each batch, as for the other arms.

## 8. Proposed diagnostic, not an arm (author decision)
- **The centroid-routing control.** The same splitting and DEDRIFT maintenance, but routing by the nearest bucket
  centroid instead of the MLP.
  - **The motivation.** The 1M IVF ablation found centroid routing often needed less work than the MLP.
  - **The scope.** An offline replay on one uniform and one Shift plan, on validation panels, reported descriptively.
    It answers whether the neural router adds value beyond the maintenance structure. §3.6 allows at most one
    benchmark variant, which is CLI-I.
