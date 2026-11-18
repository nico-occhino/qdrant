# Tier 1 preregistration: static LMI versus IVF routing (1M pilot)

Written 2026-11-18, before any Tier 1 result was computed.

## Question

Does the learned MLP router reduce the candidate work needed for a given recall, compared with exact centroid routing over the same partition? What happens when partition granularity changes?

## Data

**Corpus**
- The first 1,000,000 rows of the verified SISAP23 10.12M Float16 corpus (`continuation-25/data/vectors.f16`, sha256 `f5868d6e…` for the full file).
- The SISAP23 source is Float16: the HDF5 `emb` dtype is float16, so there is no higher-precision corpus to compare against.

**Queries:** official public queries (Float32), split as follows.

| Split | Rows | Count | Use |
|---|---|---|---|
| validation | 1000–2999 | 2,000 | operating-point selection only |
| test | 4000–5999 | 2,000 | final evaluation only |
| bridge | 9000–9255 | 256 | continuity with H-10M (report only) |

**Provenance.** DLI tuning used rows 100–963. H-10M used 900–931 and 9000–9255. The Nov 3 static campaign *evaluated* rows 20–9999 but fitted no parameters on queries. No model in this study is trained on queries.

**Gold.** Exact top-30 over the 1M subset, by cosine (inner product of unit vectors), with Float32 accumulation of the stored Float16 vectors. Ties are broken by lower row ID. This oracle has the same semantics as the stored data, so Float16/Float64 near-tie disagreement does not arise in Tier 1.

## Systems (all offline, numpy)

**Teacher (`native`).** Faithful to `lmi_index/training.rs::cluster_with_centers`:
- Sample S points, initialise centres from S random points, run 5 Lloyd iterations.
- Spherical: unit centres, dot-product assignment, means re-normalised, empty clusters keep their centre (spherical_kmeans.rs, used for cosine at training.rs:335).

**Correction (before any Tier 1 result was analysed).** The first draft said "Euclidean assignment to unnormalised means". The native cosine path is spherical. One B=64 run made with the Euclidean teacher is kept in `superseded-euclidean-teacher/` and not used.

**Teacher (`converged`).** The same algorithm with 20 iterations and S = max(32,768, 64·B).

**Router (`mlp`).** 768 → H (ReLU) → B, uniform(±1/√fan_in) initialisation, Adam with learning rate 1e-3, batch 256, E epochs, cross-entropy on teacher labels over the S sample points. The defaults are the native ones: H = 512, E = 30.

**Router (`centroid`).** Ranks buckets by ‖q − c_b‖² over the teacher centres. This needs no learning.

**Partitions**
- `mlp`: every point is assigned to its argmax bucket, as in native LMI.
- `centroid`: every point is assigned to its nearest teacher centre, i.e. IVF-Flat.

**Oracle.** Buckets that contain gold neighbours. Diagnostic only.

## Factors

| Factor | Levels |
|---|---|
| B | 64, 256, 1024, 4096 (mean bucket size 15.6K → 244) |
| Teacher | native S = 32,768 at every B; converged at every B |
| Partition × router | the 2×2 above |
| H | 512 at every B. At B = 1024 only: 128, and 0 (linear) |
| Epochs | 30 at every B. At B = 1024 only: 10 |
| Stopping rule | nprobe prefix, and work budget (take buckets in ranked order until cumulative size ≥ W) |

## Metrics

- **Per query and budget:** realized work (scored vectors), Recall@10 and Recall@30 as gold coverage of the candidate set. With exact rescoring and this oracle, coverage equals returned recall.
- **Work grid:** 64 budgets, log-spaced from the mean bucket size to 25% of N. nprobe runs 1 … min(B, 512), log-spaced.
- **Routing cost:** MACs, d·H + H·B for the MLP and d·B for the centroid router, reported separately as scored-vector equivalents (MAC / 768).
- **Construction:** lab wall seconds for teacher, training and assignment, and analytic FLOPs. Model bytes and posting bytes (4·N).

## Primary estimand

**W_target** is the mean realized work at which the test mean recall curve reaches the target, interpolated between adjacent fine-grid budgets. Targets: R@10 = 0.95 and R@30 = 0.90.

**Uncertainty:** paired percentile bootstrap over the 2,000 test queries (2,000 resamples). It covers query sampling only; this is one construction per configuration.

**Validation-locked operating point:** the smallest grid budget whose validation mean reaches the target. Its test recall and work are reported unchanged.

## Decision rules (fixed now)

**D1, router.** At a fixed partition and teacher, compare W_target(centroid router) with W_target(mlp router).
- If |ΔW| ≤ 10% and the CI includes ±10%, the router brings no routing advantage.
- If the MLP needs ≥ 10% less work and the CI excludes 0, the router is beneficial.

Routing MACs are reported alongside in both cases.

**D2, partition.** At a fixed router, the same rule decides whether centroid-assigned or MLP-assigned postings are better.

**D3, granularity.** Find the B that minimises W_target + routing equivalent among configurations that pass both conditions:
- routing is at most 10% of the total;
- construction FLOPs are at most 10× those of the native baseline.

That B is promoted.

**D4, simplification.** A smaller H, a linear model, or fewer epochs is accepted as "simpler, equivalent" if W_target is within +5% of H = 512 / 30 epochs, with the CI upper bound ≤ +10%.

**D5, teacher.** Converged versus native teacher is judged at fixed B by the D1 rule applied to the teacher.

**Promotion to 10M (Tier 2).** At most 4 configurations:
- the native baseline;
- the best IVF;
- the best LMI;
- one simplification if D4 passes.

## Not in Tier 1

Latency (Tier 3), scalar quantization, filtering, new training objectives (considered only if D1–D5 leave a clear router deficit), and HNSW.
