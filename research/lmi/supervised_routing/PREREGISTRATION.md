# Preregistration: nearest-neighbour-supervised routing over a fixed IVF partition

**Status:** frozen at commit, 2026-11-19, before any training, label generation or native run.
Deviations require a dated amendment at the end of this file.

**Branch:** `research/static-supervised-neighbor-routing-20261119`, created from `912b9c6e3` on
`research/static-lmi-ivf-router-fix` (the code that produced Tier 3A/3B). The DLI branch, the filtering branch
and every accepted Tier 1/2/3 artifact are untouched.

**Scope:** the final static learned-routing experiment. A gate failure stops it. No new architectures, partitions,
teachers, quantization, replication or filtering.

## Hypotheses

- **H0.** Nearest-neighbour-supervised routing does not improve the native IVF recall–latency frontier on held-out
  queries, after routing overhead is counted.
- **H1.** A supervised router reaches the same Recall@10 with fewer scored candidates and lower complete native query latency.

The primary comparison is learned routing against native centroid IVF on the **same physical partition**. HNSW is
context only and is not re-measured.

## Fixed physical partition

| | 1M pilot | 10M confirmation |
|---|---|---|
| Corpus | first 1,000,000 rows of the SISAP23 corpus | all 10,120,191 rows |
| Corpus file | `corpus.f16le`, sha256 `f5868d6e…0616e5` (both scales) | same |
| Storage | Qdrant Float16, cosine | same |
| Partition | native IVF-8192: `LMI_RESEARCH_ROUTER=centroid`, `n_buckets` 8192, `sample_size` 524,288, 20 iterations, seed 42, 16-CPU permit | same |

- **Partition verification.** Each job rebuilds the partition natively and must reproduce a recorded postings
  fingerprint: the sha256 of the bucket-ordered posting lists of external IDs.
  - **1M:** the first build must also reproduce, query by query, the candidate counts of the accepted 1M profile job
    25718749 (IVF-8192, same configuration).
  - **10M:** the build must reproduce Tier 3B (25743351) per-query candidate counts and result IDs for the queries
    both runs share.
  - A mismatch aborts the job.
- **Constant across arms.** Every arm uses the same segment, postings, vector storage, query preprocessing (cosine
  normalisation), native scorer, query sets, nprobe grid, binary and host. Only the router object differs.

## Queries and leakage control

The official SISAP23 public queries (10,000 rows, `public-queries-10k-clip768v2`) are split as follows:

| Set | Rows | n | Use | Prior exposure |
|---|---|---|---|---|
| **T** (train) | 0–999, 3000–3999, 6000–8999, 9256–9999 | 5,744 | training only | evaluated in earlier campaigns; rows 100–963 tuned DLI |
| **V** (validation) | 1000–2999 | 2,000 | model, hyperparameter and policy selection | Tier 1–3 validation (operating-point locking) |
| **G** (gate) | 4000–4999 | 1,000 | 1M promotion gate only | part of the Tier 1–3 test split (reported, never used for selection) |
| **C** (confirm) | 5000–5999 | 1,000 | 10M confirmation only | part of the Tier 1–3 test split (same caveat) |
| excluded | 9000–9255 | 256 | not used | Tier 1 bridge / H-10M |

G and C were reported in earlier campaigns but never used for any selection there or here. They are therefore not
pristine, and this is recorded as a limitation.

**Corpus-as-query augmentation (P).**
- 100,000 corpus points of the indexed scale, sampled with seed 20261119.
- Each one's own ID is removed from its neighbour list.
- A point is excluded if its cosine similarity to any V, G or C query is ≥ 0.95 (near-duplicate guard; the count is recorded).
- Whether to use P is a validation-selected training option: each trainable configuration is trained with T and with T + P.
- The distribution and baseline difficulty of P against T and V are reported.

## Ground truth and recall

- **Labels:** k = 10 for training targets; gold top-30 is kept for reporting.
- **1M:** gold is native exact search (the Qdrant Plain exact path, over the same Float16 segment) for T, V, G and P.
  Recall@10 at 1M is native-exact recall. The official gold is defined only at 10M.
- **10M:**
  - for T, V and C, the official SISAP23 top-30 (1-based → 0-based) is primary;
  - for C, native exact top-10 (Plain) is secondary;
  - P labels come from native exact search.
- **Coverage.** The coverage of candidates against gold is reported alongside the returned recall. With exact native
  scoring, returned recall equals candidate coverage up to score ties at rank 10.

## Targets and models

v_b(q) = |G_10(q) ∩ P_b|, t_b(q) = v_b(q)/10, which sums to 1.

| Arm | Router (always one native `MlpRouter`) | Trainable parameters |
|---|---|---|
| **A0** | Native centroid IVF: the built router, unchanged | 0 |
| **A1** | Geometry-derived LDA head, closed form (DESIGN.md §3) | 0 (shrinkage selected on V) |
| **A2** | Biases only: W = C, b = β | 8,192 |
| **A3** | Low-rank: W = C + U Vᵀ, b = β, with r ∈ {8, 32} | r(8,192 + 768) + 8,192 |
| **A4** | Full correction: W = C + Δ, b = β | 8,192 × 769 |
| **A5** | 768 → 512 → 8,192 MLP trained on t (same architecture as the original LMI) | 4.6 M |

- Every linear arm is stored as **one** `Linear` layer. Its folded weights are W/τ and its bias is b/τ (scaling by
  τ > 0 does not change the ranking).
- **Occupancy-adjusted variants**, with no retraining: priority = s/τ − log n_b. These are again one `Linear` layer,
  with the bias shifted by −log n_b. They are evaluated for A0, A1, A2, A3 and A4.
  - For A0 and A1, τ is selected on V from {0.02, 0.05}.
  - For A2–A4, τ is the training temperature.

**Loss:** L = mean_q[ −Σ_b t_b(q) log softmax(s(q)/τ)_b ] + λ · ‖Δ‖²_F / B, with Δ = 0 for A2 and Δ = U Vᵀ for A3.
- Initialisation: Δ = 0 and β = 0, so every linear arm starts exactly at A0's ranking. For A3, U = 0 and V is drawn
  from N(0, 0.01²).
- A5 is initialised as in the native LMI, with Adam at 1e-3 and plain listwise cross-entropy.

**Fixed training grid** (selection on V only):

| Arm | Grid | Data | Configurations |
|---|---|---|---|
| A2 | τ ∈ {0.02, 0.05} | {T, T+P} | 4 |
| A3 | r ∈ {8, 32}, λ ∈ {0.01, 0.1}, τ ∈ {0.02, 0.05} | {T, T+P} | 16 |
| A4 | λ ∈ {0.01, 0.1, 1}, τ ∈ {0.02, 0.05} | {T, T+P} | 12 |
| A5 | (fixed) | {T, T+P} | 2 |

- **Optimiser:** Adam, learning rate 1e-3, batch 1,024, seed 7.
- **Step budget:** 100 epochs on T only, or 10 epochs on T+P.
- **Checkpoints:** every 10% of the budget. The kept checkpoint is the best on the V criterion.

**V criterion (offline, exact simulation).**
- **C95:** the mean candidates at the smallest integer nprobe ∈ [1, 512] whose mean V coverage of native-exact top-10 is ≥ 0.95.
- Ties are broken by fewer parameters.
- For each arm, and for each occupancy variant, the configuration (and τ) with the lowest C95 is selected.

**Promotion candidate.** The single learned arm or policy (A2–A5, including occupancy variants) with the lowest V C95.
It is fixed before any native gate measurement. A1 is a reference and is not promotable.

## Native measurement (both scales)

- **Measurement boundary.** The Tier 3 decomposed native LMI path: `router.top_buckets(normalised q, nprobe)`, then the
  sorted, de-duplicated posting union, then `candidate_scoring::score_candidates`. Routing (forward pass and bucket
  ranking), union and scoring are timed separately; total time is wall time around all three. There is no fallback
  path, and every query must return 10 hits.
- **nprobe grid.**
  - 1M: {1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256}.
  - 10M: {4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128}, a superset of the Tier 3B grid near R@10 = 0.95.
  - No extrapolation.
- **Protocol.**
  - Single thread, perf profile, warm cache.
  - One untimed warm pass per (arm, nprobe), then repetitions: 1 on V; 3 on G and on C.
  - In each repetition, the order of arms is a seeded permutation (counterbalanced blocks).
  - Within an (arm, nprobe) block, queries run in fixed order.
  - Per-query latency is the median over repetitions.
- **Recorded per query, arm and nprobe:** the returned IDs, candidates (unique offsets given to the scorer; static
  segment with no deletions, so all are scored), route_ns, union_ns, score_ns and total ns.

## Promotion gate (1M, set G, native)

"Matched recall" is the lowest observed value over grid points whose mean G Recall@10 is ≥ 0.95, computed per arm.
The promotion candidate P is promoted to 10M only if **all** of the following hold:

1. **Candidates:** P uses at least 10% fewer mean candidates than A0 at matched recall.
2. **Latency:** P's median latency at matched recall is below A0's (point estimate). The 95% paired query-bootstrap CI
   is reported.
3. **Correctness:**
   - every G query returns 10 valid hits at every grid point;
   - native and offline top-nprobe buckets agree on ≥ 99% of (query, nprobe) checks;
   - any disagreement must be explained by floating-point near-ties.
4. **Budget:**
   - router bytes ≤ 2 × A0;
   - training ≤ 2 h on 16 CPUs;
   - routing p50 ≤ A0 routing p50 + 0.5 ms.
5. **Generalisation:** the gain must hold on G, which is held out by construction. The T-versus-V gap is reported.
6. **Integrity:** hashes and partition fingerprints all verify.

**If the gate fails, the campaign stops and a negative report is written.** No further arms or tuning follow.

## 10M confirmation (only if the gate passes)

- **Partition.** The frozen IVF-8192 partition is rebuilt and verified against Tier 3B.
- **Model.** The promoted configuration is retrained once with identical hyperparameters, on 10M labels (T official
  gold, plus P native gold if T+P was selected).
- **Arms.** A0, A1 (with its V-selected shrinkage, refitted at 10M), and P.
- **Cohorts.** C, with 3 counterbalanced repetitions; V with 1 repetition, for consistency only.

**Primary endpoint:** the ratio of P's to A0's lowest observed median latency at mean official C Recall@10 ≥ 0.95.
Uncertainty comes from a paired query bootstrap: queries are resampled with all their repetitions, with 10,000
resamples, and the operating point is re-selected inside each resample.

| Outcome | Condition |
|---|---|
| Success, robust | ratio ≤ 0.90, the 95% CI upper bound ≤ 0.90, and p95 at that point no worse than A0 + 5% |
| Success, tentative | ratio ≤ 0.90 but the CI upper bound > 0.90 |
| Routing-only positive | candidates −10% or more, but the latency criterion fails |
| Negative | otherwise |

Recall targets 0.90, 0.97, 0.98 and 0.99 are reported descriptively wherever they are attained.

## Non-goals

HNSW re-measurement, DLI/CLI, e-means, teacher optimisation, partition learning, replication, filtering, quantization,
new scorers, architecture search, 100M, and any modification of accepted artifacts.

## Amendment 1 (2026-11-19, before any job was submitted)

**1M partition check tolerance.** The accepted 1M profile job 25718749 ran at `f2f6f27d8`, which predates the
router constant-factor fix `aaca97f24` (SIMD dot product in the router). Router logits, and therefore near-tie bucket
choices at query time and in build-time posting assignment, can differ in the last floating-point bits.

The 1M rebuild is therefore accepted if at most 1% of the (query, nprobe) checks differ in candidate count or IDs.
Exact counts are reported. The 10M check against Tier 3B, which was built after the fix, remains exact (0 mismatches).

**Partition fingerprint.** At each scale, the pilot job records the canonical postings fingerprint. Later jobs at the
same scale must reproduce it exactly.

## Amendment 2 (2026-11-19, after preflight 25758335 on the 20K smoke partition, before any pilot job)

**Short results.** The decomposed native path (Tier 3 boundary) has no underfill widening. When the probed buckets hold
fewer than 10 points (small nprobe, small buckets), it returns fewer than 10 hits. The smoke preflight showed 47 such
results at nprobe 1.

- **Invariant enforced by the harness:** every query returns exactly min(10, candidates) hits; any other count aborts
  the run.
- **Scoring:** short results are recorded and count as recall misses.
- **Gate criterion 3 (replaces "every G query returns 10 hits at every grid point"):** no short results at the operating
  points used in the matched comparison, plus ≥ 99% native/offline ranking agreement.
