# Draft thesis section: candidate-selection efficiency at 10 million vectors

*Draft prose for the results chapter. Numbers are from `TEN_MILLION_VECTOR_ANALYSIS.md`; check the figures and units before inclusion.*

## Setup

We compare two learned indexes over the same 10,120,191 SISAP 2023 LAION CLIP-768 vectors (Float16, cosine), implemented natively in Qdrant 1.20.

**Static LMI.** A single 548-bucket model trained once on the full collection.

**Geometric dynamic learned index (DLI).** Built by inserting the vectors in batches. Background maintenance merges segments geometrically (arity 2) and retrains a model of the same configuration for each merged level. The final structure has:
- four learned levels holding 6.0M, 3.0M, 0.75M and 0.25M vectors;
- an unpartitioned buffer of 120,191 recently inserted vectors.

**Measurement.** Queries are replayed through the production routing and resolution code, with the scorer held fixed. The cost of a query is the number of vectors it actually scores. Requested budgets are not used as the cost, because whole-bucket selection can overshoot them.

**DLI allocation policies**
- **Default slot-weighted:** priority `P·2^level`, with the buffer at priority 1.
- **Population-proportional control:** the learned budget is split by member population, and the buffer is always scored.
- **Temperature-2 population-density.**

**Queries.** 256 test queries (rows 9000–9255 of the official query set) that were not used for any DLI design decision.

**Statistics.** Uncertainty is a paired percentile bootstrap over queries (10,000 resamples). It describes query sampling for one trained index of each kind, not variation across training runs.

## Results

**Primary result** (Figure 1).
- The static LMI reaches a mean Recall@30 of 0.90 at **1.64%** of the collection scored per query: 166K vectors, 95% CI 143K–219K.
- The default DLI needs 305K (285K–325K): static needs **45%** less work (95% CI 32–51%).
- The proportional control needs 285K: static needs **42%** less (30–47%).

This supports the preregistered hypothesis of a saving of at least 30%.

**No crossing.** The DLI curves never rise above the static curve within the measured range. The gap is largest at intermediate work (+15.5 percentage points Recall@30 at about 100K vectors) and shrinks below one point above about 500K (Figure 3).

**Policy dependence.** The temperature-2 policy is only 23% (7–34%) less efficient than static, so the size of the penalty depends on the allocation policy.

**Decomposition** (Figure 2). Static and proportional are compared at the same requested budget:
- They return the same recall (all differences within ±0.012 except one).
- The proportional DLI's extra work equals the buffer size, 120K vectors.

Proportional allocation scores the same learned candidates as the static index, plus the whole buffer. If learned selection efficiency is equal, the expected saving is buffer / (static work + buffer) = 41.9%. The observed value is 41.7%.

**Mechanism under the default policy.**
- Slot-weighted selection takes the buffer as soon as no remaining bucket has `P·2^level > 1`, and the buffer then adds 120K vectors in one step. This produces a plateau in its recall–work curve between 80K and 180K scored vectors.
- The buffer contains about 1.3% of the true neighbours, matching its share of the data.
- The temperature-2 policy postpones selecting the buffer, which explains its smaller penalty. It therefore misses those neighbours at moderate budgets, a loss of about the size of its remaining deficit.

**Router quality is a shared limitation, not a DLI-specific one.** For both indexes, covering a query's true top-30 in router-rank order costs about 3–3.7 times the work of an oracle that scores only the buckets containing those neighbours. The overhead is driven by a minority of hard queries; the median query needs only 1.3–1.4 times the oracle.

**Scale** (Figure 4).
- The same analysis at 65,536 vectors gives a similar static advantage over the default DLI: 40% (33–46%) at Recall@30 0.90.
- It shows the same equal-budget equivalence between proportional DLI and static, offset by that topology's 1,536-vector buffer.
- At both scales, the fraction of the collection scored for a given recall falls with collection size. Two topologies do not establish a scaling law.

## Discussion

**What the penalty is.** Geometric maintenance does impose a measurable selection penalty, but in this experiment it comes from the unpartitioned insertion buffer, not from replacing one global model with several independently trained level models. At equal learned work, the level models together select candidates as well as the global model.

**Implications.** Improving dynamic learned indexes for search should therefore target how the buffer is represented and when it is selected:
- cost-aware buffer priorities;
- provisional partitioning of new vectors;
- smaller buffers.

It should not target cross-level probability calibration alone.

**Dependence on buffer fill.** The size of the penalty depends on how full the buffer is relative to the work needed for the target recall. This experiment observed a single fill state of 48%.

## Limitations

- **Replication.** One index construction per method; seed and ingestion-order variance are not measured.
- **Budget grid.** Budgets were spaced by factors of two, so the magnitude (not the direction) of the 0.90-recall saving depends on interpolation.
- **Search path.** Results are offline replays of candidate selection with a fixed exact scorer, not end-to-end latency.
- **Not compared.** HNSW and quantized scoring were outside the experiment's scope.
