# Design: supervised routing over a fixed native IVF partition

## 1. Where the router sits

The native LMI index holds `LmiRoutingState { router: MlpRouter, postings: CompactPostings, nprobe }`.

The Tier 3 harness (`ann_matched_v120.rs`) already measures the learned path decomposed. Per query:

1. Route: `router.top_buckets(normalised q, nprobe)`.
2. Union: the posting union, sorted and de-duplicated.
3. Score: `candidate_scoring::score_candidates`, Qdrant's native scorer with its visibility and deletion checks.

This experiment keeps that boundary unchanged. An arm is **only a different `MlpRouter` value** applied to the loaded
segment's own `state.postings()`. The persisted segment, its postings and its built router are never modified, so the
physical partition is identical across arms by construction.

- **No new index type.** No Qdrant core code changes.
- **No change to the persisted LMI state format.** Learned routers are versioned artifacts of the experiment, loaded
  by the harness.
- **Routing parity.** Every linear arm is one `RouterLayer::Linear` (8,192 × 768 weights and 8,192 biases). That is
  exactly the shape and code path of the native centroid router, so linear arms carry no extra routing cost by
  construction. It is still measured.
- **Ties.** The native `top_ids` breaks equal logits toward the smaller bucket ID. The offline simulation uses the
  same rule.

## 2. Artifacts and their checks

**Router artifact:** `router.json` with:
- `{"arm", "layers": [{"kind": "linear", "in", "out", "weights": "<file>", "bias": "<file>"} | {"kind": "relu"}], …}`;
- raw little-endian f32 files, with their sha256 recorded in the JSON.

Before use, the harness:
- rebuilds the `MlpRouter`;
- runs `MlpRouter::validate()`, which checks shapes and finiteness;
- requires input dimension 768 and output dimension equal to the segment's posting count (8,192);
- checks the file hashes.

A router that fails any check aborts the run; there is no fallback.

**Export of the fixed partition** (harness `export` phase):
- the built router's layers, in the same artifact format (this is A0);
- `bucket_of_id.u32`: bucket per external ID − 1;
- `occupancy.u32`: n_b per bucket;
- `postings.sha256`: sha256 over buckets in order, each bucket's external IDs sorted, little-endian u64. This is the
  partition fingerprint.

## 3. Arms as single linear layers

Notation: unit query q; native centroid weights C (B × d) and biases c₀ (zero for the spherical centroid router;
the exported values are used, whatever they are).

| Arm | Folded weights W | Folded bias b |
|---|---|---|
| A0 | C | c₀ |
| A1 (LDA) | Σ_λ⁻¹ μ_b | −½ μ_bᵀ Σ_λ⁻¹ μ_b + log(n_b/N) |
| A2 | C/τ | (c₀ + β)/τ |
| A3 | (C + U Vᵀ)/τ | (c₀ + β)/τ |
| A4 | (C + Δ)/τ | (c₀ + β)/τ |
| occupancy variant of any of the above | same W/τ | b/τ − log max(n_b, 1) |

**A1 (LDA head).**
- μ_b is the mean of the unit vectors in bucket b.
- Σ is the pooled within-bucket covariance.
- Shrinkage: Σ_λ = (1 − λ)Σ + λ (tr Σ / d) I, with λ ∈ {0.01, 0.1, 1} selected on V.
- It is fitted on the indexed corpus only, with no queries. It is the shared-covariance Gaussian posterior
  log p(b | q) up to a constant, i.e. a geometry-plus-statistics router with closed form.
- Note: its log(n_b/N) prior favours **large** buckets, the opposite of occupancy adjustment.

**Occupancy adjustment.** log softmax(s/τ)_b − log n_b = s_b/τ − log n_b − logsumexp(s/τ). The last term is the same
for every bucket, so the ranking is a bias shift: it remains one `Linear` layer, with no additional inference.

**A5.** Two `Linear` layers with a `ReLU` between them (768 → 512 → 8,192), the existing native MLP shape. That is
4.6 M multiply-adds against 6.3 M for the linear arms.

## 4. Training (`train_router.py`, PyTorch 2.5.1 CPU, deterministic)

- **Inputs:**
  - the exported A0 router;
  - `bucket_of_id`;
  - unit query matrices for T, V (and P);
  - gold top-10 IDs.
- **Targets:** t_b(q) = |{g ∈ gold_10(q) : bucket(g) = b}| / 10, stored sparsely.
- **Loss:** listwise cross-entropy over all 8,192 buckets, plus λ‖Δ‖²_F / B (DESIGN = PREREGISTRATION grid).
- **Model selection:** at each checkpoint, the V criterion C95 is computed by exact simulation. Bucket ranking uses
  float32 logits with the native tie rule; coverage is accumulated per nprobe from the per-bucket target counts;
  candidates are summed from n_b. The best checkpoint per configuration, and the best configuration per arm, are
  written to `selection.json`, together with the promotion candidate.
- **Outputs:** every selected router as a native artifact, plus training curves, train and V coverage, parameter
  counts and wall times (`training.json`).

## 5. Native evaluation (harness phases)

| Phase | Purpose |
|---|---|
| `source` | Ingest N rows into a Float16 Plain segment (ids 1..N), as in Tier 3. |
| `build` | Native IVF-8192 through `SegmentBuilder`, as in Tier 3. |
| `export` | Section 2. |
| `gold` | Native exact top-K (Plain exact path) for a query file; multi-threaded, deterministic. |
| `consistency` | Native top-64 buckets for the first 200 V queries per arm, for offline/native agreement. |
| `evaluate` | Arms × nprobe grid × query sets, with the PREREGISTRATION protocol. Per-query JSONL: ids, candidates, route_ns, union_ns, score_ns, ns, rep. |

## 6. Correctness invariants and their checks

| Invariant | Check |
|---|---|
| Every indexed live point is in exactly one posting | `export` asserts sum(n_b) = available points, and each ID appears once. |
| Routing cannot alter assignments | Postings come from the loaded segment only; the fingerprint is checked before and after evaluation. |
| Returned IDs map to correct offsets | IDs are resolved through the segment's ID tracker. Recall against gold IDs and A0's reproduction of the profile and Tier 3B IDs validate the mapping. |
| Native Float16 scoring unchanged | Only `score_candidates` is called. No deletions occur in this static experiment. |
| Deterministic bucket order | `top_ids` total order; the `consistency` phase checks it. |
| No hidden fallback | The decomposed path has none. Every result must have 10 hits; a short result is recorded and fails the gate. |
| Reopen reproducibility | A Rust test persists a non-centroid single-`Linear` router with biases, reopens it, and checks identical rankings and results (`lmi_static_v120`). The harness re-runs A0 on a reloaded segment. |
