# Tier 2 preregistration: 10M offline evaluation of Tier 1 finalists

Written 2026-11-18, after Tier 1 selection and before any Tier 2 run.

**Selection basis:** Tier 1 under `AMENDMENT_01_VALIDATION_SELECTION.md`, giving `analysis/finalists-val.json`.

**Selection correction disclosed.** The first selector run omitted the preregistered D3 routing cap; it is kept as `*.SUPERSEDED-missing-routing-cap.json`. The corrected run applies both D3 conditions. Validation and test pick the same finalists.

**Excluded by rule:** B4096-converged. It needs less total work at 1M (IVF 11,145 versus 18,781), but exact centroid routing is about 37% of its total, which breaks D3's ≤ 10% routing cap. That limit exists because exact centroid routing at large B is itself a full scan of the centroids. B4096 is reported in Tier 1 and not promoted.

## Configurations (lab.py, full 10,120,191-vector corpus, seed 42)

| Job | Purpose | lab.py arguments |
|---|---|---|
| T2-F1 | Native baseline (accepted 10M configuration), 2×2 | `--B 3162 --sample 250000 --iters 5 --hidden 512 --epochs 30` (both partitions and both routers) |
| T2-F23 | F2 best IVF (centroid\|centroid) and F3 best LMI (mlp\|mlp): same converged teacher, 2×2 | `--B 8192 --sample 524288 --iters 20 --hidden 512 --epochs 30` |
| T2-F4 | Linear router, a D4 simplification | `--B 8192 --sample 262144 --iters 5 --hidden 0 --epochs 30 --partition mlp --router mlp,centroid` |

**Mapping 1M to 10M (Amendment 01 rule 4).** Keep N/B and S/B, with B the nearest power of two. B1024 at 1M has 977 points per bucket, giving B = 10,358 at 10M, rounded to 8,192 (1,235 per bucket). S/B = 64 for the converged teacher and 32 for the native teacher.

## Queries and oracle

- **Splits:** validation rows 1000–2999 and test rows 4000–5999, as in Tier 1. Bridge rows 9000–9255 are reported only.
- **Oracle:** the official SISAP23 top-30 (Float64 reference semantics). Recall is reported as coverage of official IDs by the candidate set.
- **Agreement measured on a 256-query test subset:** exact search over the stored Float16 vectors covers 0.988 of official top-10 and 0.993 of official top-30. Those figures describe agreement under these numerical semantics, not a universal bound. Native returned recall (Tier 3) can differ from coverage by up to about that amount.

## Estimands (analyze_tier1.py, unchanged)

- W_target at R@10 = 0.95 and R@30 = 0.90 on validation (selection-free reporting) and on test (evaluation).
- W_total = W_target + routing vector-equivalents.
- Paired bootstrap over queries for the IVF/LMI and router/partition ratios within each job.
- Bucket statistics, construction time, peak RSS, model and posting bytes.

## Decision for Tier 3

- Promote the configuration with the lowest **test** W_total at R@10 = 0.95, plus the native baseline.
- If IVF (F2) ≤ LMI (F3) with a CI excluding 1, Tier 3 uses native IVF (`LMI_RESEARCH_ROUTER=centroid`) and the native baseline LMI, with HNSW and Plain as references.
- Native build cost is profiled first at the promoted B before large allocations.

## Resources (estimates from Tier 1 timings, to be checked in the preflight)

| Job | ncpus | mem | scratch | walltime | Dominant cost |
|---|---|---|---|---|---|
| T2-F1 | 16 | 48 GB | 40 GB | 6 h | MLP training ≈ 9·10¹³ FLOP; two 10M assignments |
| T2-F23 | 16 | 48 GB | 40 GB | 12 h | MLP training ≈ 4.3·10¹⁴ FLOP at numpy small-batch efficiency |
| T2-F4 | 16 | 48 GB | 40 GB | 6 h | Linear training plus assignment |

**Preflight:** N = 1M, B = 256, native teacher, 2 epochs, 8 CPUs, 16 GB, 45 min. It must reach `COMPLETE.json` before any T2 job is submitted.
