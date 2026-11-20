# Training report: supervised routers over the 1M IVF-8192 partition

**Source:** pilot job 25759596 (commit `d76aea7b2`), `train/training.json`. Selection used validation (V) only.
**Partition:** native IVF-8192 over the first 1,000,000 SISAP23 vectors, fingerprint `7c61bcbf…f463`.
- Bucket occupancy: mean 122, CV 0.64, maximum 1,560, 10 empty buckets.

## Data

| Source | Queries | Labels |
|---|---|---|
| T (official training rows) | 5,744 | native exact top-10 over the 1M segment |
| P (corpus-as-query) | 100,000 | exact top-10 under stored-vector semantics, self excluded |
| V (validation) | 2,000 | native exact top-10 |

**P details**
- 310 sampled points were excluded as near-duplicates (cosine ≥ 0.95) of a V, G or C query.
- 107 P points did not rank themselves in their own top-11. These are exact duplicate vectors that win ties.
  Their labels are still correct non-self neighbours.

**Baseline difficulty under A0 (C95, the mean candidates at 95% coverage)**

| | T | V | P (first 20K) |
|---|---|---|---|
| C95 | 4,542 | 4,501 | 4,366 |

The query sources are comparably hard for centroid routing. That does not prove their distributions match, but no
gross mismatch is visible.

## Grid outcome (V criterion C95; lower is better; A0 = 4,501)

| Arm | Best configuration | Trainable parameters | V C95 | Change vs A0 |
|---|---|---|---|---|
| A0 centroid | — | 0 | 4,501 | — |
| A0occ | τ 0.02 | 0 | 4,983 | +10.7% |
| A1 LDA | λ 1.0 | 0 | 5,502 | +22.2% |
| A1occ | λ 1.0, τ 0.05 | 0 | 5,501 | +22.2% |
| A2 biases | τ 0.02, T+P | 8,192 | 4,369 | −2.9% |
| A2occ | τ 0.02, T+P | 8,192 | 4,094 | −9.1% |
| A3 low-rank | r 32, λ 0.1, τ 0.02, T+P | 294,912 | 4,323 | −4.0% |
| A3occ | same | 294,912 | 4,152 | −7.8% |
| A4 full | λ 1.0, τ 0.02, T+P | 6,299,648 | 4,130 | −8.3% |
| **A4occ** (promoted) | λ 0.01, τ 0.05, T+P | 6,299,648 | **3,863** | **−14.2%** |
| A5 neighbour-loss MLP | T+P | 4,596,224 | 6,440 | +43.1% |

## Generalisation and overfitting

**Augmentation was decisive.**
- Trained on the 5,744 official queries alone, every arm with more than biases overfits badly.
  - A4 (T only) reaches train C95 ≈ 1,100 but V C95 between 5,901 and 37,069, depending on λ and τ.
  - Low-rank models trained on T only are worse than A0 on V (5,931–34,108).
- With P added, the same arms generalise to V. Every selected trainable configuration uses T+P.

**The remaining gap is still large.** A4occ: train C95 1,794 against V 3,863. The full correction memorises its
training queries, and the gain that survives is modest.

**τ matters.** τ = 0.02 dominates 0.05 for most arms. The higher temperature combined with low rank (A3, τ 0.05)
fails badly.

**Occupancy adjustment** (s/τ − log n_b):
- helps every *learned* arm: A2 −2.9% → −9.1%, A3 −4.0% → −7.8%, A4 −8.3% → −14.2%;
- hurts centroid routing (A0occ +10.7%);
- is neutral for LDA, whose log(n_b/N) prior it cancels.

Learned scores behave like calibrated log-probabilities of neighbour mass, so dividing by occupancy approximates
expected neighbours per candidate. Raw cosine scores are not calibrated, so the same correction misranks them.

**Validation optimism.** The held-out G gain of the selected A4occ, measured post hoc on the same offline criterion,
is −10.3% (RESULTS_1M.md §4), against −14.2% on V. Selection on V inflated the effect by about 4 percentage points.

## Cost

- **Training time.** The selected A4occ configuration trained in 94 s (16 CPUs). The whole grid, including the A1
  closed forms, took 10,256 s.
- **An anomaly.** Some T-only runs were far slower than their step counts imply: the T-only A3 rank-8 configuration
  took 4,638 s against 41–106 s for comparable runs. The cause was not diagnosed. It does not affect the selected models.
- **Model size.** Every linear arm is one 8,192 × 768 `Linear` layer, 25.2 MB of f32 parameters, the same size as A0's
  centroid router. A5 is 18.4 MB.

## What the training shows

1. A nearest-neighbour objective with a linear correction to the centroids learns something real. On held-out queries
   it is about 10% better on the offline coverage criterion.
2. Most of the oracle headroom is not learnable from 105K training queries.
   - The candidate-budget oracle reaches 95% coverage with about 1,000 candidates.
   - A0 needs 4,849 (offline, G). The best learned router needs 4,350, capturing roughly 13% of that gap.
3. The neighbour-trained MLP (A5) is far worse than every linear arm. The right loss does not rescue the hidden-layer
   architecture at this data scale. Only one configuration per data source was preregistered, so this is not an
   architecture search.
