# Filtered-search probing policies on SISAP23 300K: report

**Status:** complete, 2026-11-19. This is the single report for the study preregistered in
`PREREGISTRATION_FILTER_PROBING.md` (with Amendment 1). Source revision `781409136` (clean tree). The study is closed:
no further optimisation follows from it.

## What was tested
- **Policies:**
  - **B0:** the production planner, nprobe 16, break-even plan choice.
  - **B1:** one fixed nprobe chosen on validation.
  - **C1:** a linear cost model.
  - **R1 / R2:** recall-calibrated per estimated-selectivity bin, choosing on the mean / on the bootstrap lower bound.
  - **EXACT:** exact filtered search, as a reference.
  - **Oracle:** the cheapest measured cell per query, using ground truth.
  - All policies ran at τ = 0.90 and τ = 0.95.
- **Setup:**
  - Routers: IVF (centroid) and MLP, with identical partitions, postings and segments.
  - Two filter families: independent (`ind`, uniform random payload) and correlated (`slab`, quantile slabs along a fixed direction). The `slab` family is exploratory and was not used for any tuning.
  - Selectivity grid: s ∈ {1 … 0.0001}, 12 values.
  - Probe grid: nprobe ∈ {1 … 256}, 13 values.
  - Policies were chosen on validation rows 1000–2999 using the independent family only, then frozen and evaluated on test rows 4000–5999.
  - Latency was measured natively on a 500-query test subsample: warm-up, 3 repetitions, median.
- **Recall@10:** uses k_eff = min(10, m), where m is the number of points that match the filter. There were no empty cases (m = 0) at the tested selectivities.

## Integrity checks
| Check | Result |
|---|---|
| Native policy decisions and results vs grid simulation (720,000 records) | **0 mismatches** |
| B0 reproduces `result-nprobe16.json` (plans, recall, fill, widenings) | **0 mismatching cells** |
| Baseline planner digests (`lmi_filtered_v120`) | unchanged |
| Production `search()` path | unchanged (Baseline only) |

## Results (independent filters, test, IVF; MLP in brackets)

| s | B0: R@10 / p50 ms | R2@0.95: nprobe, R@10 / p50 ms | EXACT p50 ms |
|---|---|---|---|
| 1 | 0.953 / 8.70 (0.934 / 7.35) | 16, 0.953 / 8.84 (24, 0.953 / 10.31) | 230.9 |
| 0.5 | 0.943 / 4.80 (0.920 / 4.33) | 20, 0.955 / 5.76 (32, 0.957 / 7.54) | 114.0 |
| 0.1 | 0.916 / 1.16 (0.886 / 1.10) | 32, 0.960 / 2.11 (48, 0.957 / 2.72) | 21.7 |
| 0.05 | 0.894 / 0.61 (0.857 / 0.61) | 48, 0.968 / 1.64 (64, 0.960 / 1.97) | 10.0 |
| 0.03 | 0.876 / 0.43 (0.835 / 0.44) | 48, 0.962 / 1.12 (128, 0.980 / 2.63) | 4.0 |
| 0.02 | exact, 1.000 / 1.03 | 48, 0.957 / 0.91 (exact, 1.000 / 0.96) | 1.03 |
| ≤ 0.01 | exact, 1.000 | exact, 1.000 | identical |

Full per-cell tables are in `analysis.md`, outside Git.

## Findings

**1. Recall calibration removes B0's recall cliff.**
- B0's recall falls steadily as the filter tightens, down to 0.876 for IVF and 0.835 for MLP at s = 0.03. Below that point it switches to the exact plan.
- R1 and R2 at τ = 0.95 are the only policies that keep mean R@10 ≥ 0.94 at every s, for both routers, on held-out test queries.
- The price is latency at s = 0.03–0.1: 1.8–2.7× B0's for IVF and 2.5–6.0× for MLP. Compared with exact search, it is still 3.6–10× faster for IVF, but only 1.4–7.5× faster for MLP (1.4× at s = 0.03).
- R1 and R2 almost always pick the same cell. The bootstrap lower bound (R2) only adds a step where validation was noisy.

**2. Cost-only planning is unsafe.**
- C1 decides on cost alone and reuses a single nprobe at every s, so its recall follows B0 downwards. At τ = 0.95 it falls to 0.853 at s = 0.02.
- At τ = 0.90 the cost model routes s = 0.01 queries at nprobe 8 because that is predicted cheaper. Recall there drops to 0.66 (IVF) and 0.69 (MLP), against 1.0 for exact search at 0.33 ms.
- Conclusion: a latency model must never choose the plan unless it is paired with a recall model.

**3. A target on mean recall is not a per-query guarantee.**
- Even where the mean is ≥ 0.95, 20–28% of test queries fall below 0.95 and the 5th-percentile recall is 0.8, for every routed policy.
- The oracle, which picks the cheapest cell per query using ground truth, reaches 1.0 at τ = 0.95 for about half of R2's p50. For example, IVF at s = 1 takes 4.0 ms against 8.8 ms.
- That gap is per-query adaptivity, which no cardinality-only policy can capture. The oracle is an upper bound, not something a policy can achieve.

**4. Pre-registered adoption criterion: not met. B0 stays the default.** The failures fall into three kinds:
- **Genuine recall failures:** C1@0.95 at s = 0.2–0.02, for both routers.
- **A genuine latency failure against B1:** R1/R2 at s = 0.5 run 5.8 ms against B1's 4.7 ms (IVF). They reach 0.955 recall where B1 reaches 0.943. The preregistered "matched recall" test counts both as matched because both are ≥ 0.94.
- **Timing-noise failures:** many cells where the policy and its comparator run the *identical* plan (exact vs exact, or the same nprobe) and differ by 0.001–0.06 ms. One example is MLP R1@0.95 at s = 0.02: 1.024 ms against 0.963 ms, both exact.

Most of the noise failures disappear if a tolerance band for identical plans is added after the fact, but the s = 0.5 trade-off would remain. That reading is not preregistered. The preregistered verdict stands, and R1/R2 remain research options.

**5. IVF beats MLP under filters as well.**
- At B0 (equal nprobe), IVF's recall is higher by 0.019 at s = 1 and by 0.042 at s = 0.03, with 95% bootstrap CIs excluding 0.
- At matched calibrated recall (R2@0.95), IVF is faster: by 1.26 ms at s = 1, 0.57 ms at s = 0.1 and 1.50 ms at s = 0.03 (CIs exclude 0). The MLP needs a 1.5–2.7× higher nprobe.
- This agrees with the static Tier 1–3 result.

**6. Correlated filters break calibration transfer (exploratory).**
- These are slab filters with no retuning. R2@0.95, calibrated on the independent family, undershoots its target on slabs: IVF reaches 0.943 / 0.947 / 0.928 / 0.911 at s = 0.1 / 0.05 / 0.03 / 0.02, and MLP reaches 0.934 / 0.931 at s = 0.1 / 0.05.
- B0 does worse on slabs than on independent filters at the same s: IVF 0.827 against 0.894 at s = 0.05, MLP 0.782 against 0.857.
- A plausible explanation, not measured here: under correlation, the true filtered neighbours sit in buckets that the unfiltered ranking orders lower. Either way, a single calibration table keyed only by selectivity is not enough.
- This is the case where filter-aware bucket priorities could matter, and it is the one open question this study leaves.

## Correction to the baseline record (finding 5 in `SISAP300K_FILTERING_BASELINE.md`)
- The baseline record attributed the ≈2.2 ms extra latency of a 100% filter to filter checks. The stage pass does not support that.
- At nprobe 16 and s = 1 (IVF), the median stage times are: routing 0.38 ms, filter checks 0.75 ms (11,889 candidates), scoring 7.81 ms.
- The fitted marginal cost is 15.6 ns per filter evaluation and 575 ns per scored candidate.
- Filter checks therefore explain at most about a third of the 2.2 ms. The rest was not attributed, because no unfiltered stage pass was run. Scoring dominates both paths.

## Limitations
- One dataset (300K, 768-d cosine) and one partition (B = 548). Latencies were measured warm-cache on one machine, and the timing used 500 test queries.
- Both filter families are synthetic. Only one correlated structure was tested.
- Calibration is per selectivity bin only: no per-query features.

## Evidence (outside Git)
Location: `research-admin/experiments/2026-11-19-lmi-adaptive-filtering/probing/full/`. Inputs are the same as for the baseline, with hashes checked against the manifest.

| File | sha256 |
|---|---|
| `analysis.json` | `f4d150308a71078ad28c07c3cdae9b63213212783c422ba5ce79e012b76eafa3` |
| `analysis.md` | `69d6812fa1082ef45cd6ab073907c9011bb3593868427246b5ab696f9f4bb6fc` |
| `policies.json` (frozen choices + cost fits) | `025c5e06fa2c66326d9024add4a08dac00e816ddc0f01083c5e0a13224526046` |
| `b0repro.json` | `371f1bec2e55aacf4dd39e6cfb5e94be6e83121cad337a3d9fc51bafb8d9c943` |
| `manifest.json` | `a469cf7aa056138b9751a281159ae0478d344e079955c15bec4eb45dc8d37a30` |
| `policies/summary.json` | `24e33b0e966478b723925f432ecba9e90540649ab28485c7d817921f1b007b41` |

Raw per-query records, ≈15 MB plus segments, are kept in the `grid/`, `timing/`, `stages/` and `policies/` subdirectories. The run log is `logs/chain.log`, which ran from 2026-11-19 15:08 to 20:52.

## Decision
- B0 remains the production planner.
- The recall-calibrated policy (R2) is the recommended *research* configuration whenever a mean-recall target matters more than latency at intermediate selectivities.
- The study is closed. Next priorities are the Tier 3B synthesis, thesis writing and the DLI replication.
