# Results: native 1M pilot and promotion gate

**Job:** 25759596 on elmu1-22 (AMD, 16 CPUs), commit `d76aea7b2`, perf profile. Exit 0 after 4 h 35 min, with
`COMPLETE.json` written. Checksums verify except `job.log` (the known appended-line pattern; its prefix hash matches).

**Evidence:**
- `research-admin/experiments/2026-11-19-static-supervised-routing/hpc-pilot/` (read-only copy);
- remote `static-supervised-routing-20261119/results/pilot-25759596…`.

**Analysis:** `analyze.py pilot` writes `QUERY_LEVEL_RESULTS/gate_analysis_1M.json`, `COMPARISON_TABLES.csv` and `FIGURES/`.

## 1. Validity checks

| Check | Result |
|---|---|
| Partition against profile job 25718749 (V, nprobe 8/32/128) | 6,000 checks; 1 candidate-count mismatch, 0 ID mismatches (tolerance 1%, Amendment 1) |
| Postings fingerprint before and after evaluation | identical (`7c61bcbf…`) |
| V queries against the accepted Tier 1–3 file | bitwise identical |
| Hits = min(10, candidates) | 0 violations. 239 short results, all at small nprobe and all explained by too few candidates |
| Native against offline top-nprobe bucket agreement | 99.93% of 15,400 checks. Disagreements were 1–4 per arm, consistent with floating-point near-ties |
| Router artifacts | hashes, shapes and `MlpRouter::validate` checked before use; all 11 accepted |

**Protocol:** single thread, warm cache. The arms are measured on the native Tier 3 path: route, union, native scorer.
- V: 2,000 queries, 1 repetition.
- G: 1,000 queries, 3 repetitions, with arm order permuted per repetition.
- Every arm runs on the same segment and the same postings.

## 2. Held-out G: native operating points

Recall@10 is measured against native exact gold. Latency is the per-query median over 3 repetitions; p50 and p95 are
taken across queries.

| nprobe | A0 R@10 | A0 cand. | A0 p50 (ms) | A4occ R@10 | A4occ cand. | A4occ p50 (ms) |
|---|---|---|---|---|---|---|
| 16 | 0.9131 | 2,550 | 1.325 | 0.9196 | 2,547 | 1.302 |
| 20 | 0.9280 | 3,164 | 1.558 | 0.9335 | 3,155 | 1.523 |
| 24 | 0.9377 | 3,772 | 1.759 | 0.9434 | 3,752 | 1.725 |
| **32** | **0.9526** | **5,006** | **2.173** | **0.9564** | **4,949** | **2.125** |
| 40 | 0.9620 | 6,230 | 2.554 | 0.9653 | 6,122 | 2.507 |
| 48 | 0.9685 | 7,438 | 2.918 | 0.9707 | 7,287 | 2.893 |
| 64 | 0.9760 | 9,856 | 3.590 | 0.9777 | 9,583 | 3.527 |

**Pattern.** At every nprobe, A4occ has higher recall with slightly fewer candidates. Both first reach 0.95 at
nprobe 32, because the preregistered grid steps from 24 to 32.

**Time at nprobe 32, A0:**

| Stage | p50 | Share |
|---|---|---|
| Routing | 0.51 ms | 23% |
| Union | 0.05 ms | — |
| Scoring | 1.61 ms | 74% |

Routing costs the same for all linear arms (0.48–0.55 ms; A4occ 0.497 ms against A0's 0.500 ms, as medians over
all G rows). A5's routing is cheaper (0.26–0.40 ms).

**Lowest observed value at mean G R@10 ≥ target**

| Arm | 0.90: candidates / p50 | 0.95: candidates / p50 | 0.97: candidates / p50 |
|---|---|---|---|
| A0 | 2,550 / 1.325 | 5,006 / 2.173 | 9,856 / 3.590 |
| A0occ | 2,629 / 1.351 | 6,132 / 2.528 | 12,102 / 4.220 |
| A1 | 2,472 / 1.239 | 5,984 / 2.441 | 14,016 / 4.647 |
| A2 | 2,156 / 1.107 | 5,613 / 2.353 | 8,369 / 3.151 |
| A2occ | 2,491 / 1.342 | 4,898 / 2.187 | 9,631 / 3.575 |
| A3 | 2,823 / 1.371 | 5,546 / 2.318 | 8,219 / 3.245 |
| A3occ | 2,527 / 1.267 | 4,945 / 2.234 | 9,679 / 3.547 |
| A4 | 2,134 / 1.202 | 5,514 / 2.315 | 8,170 / 3.088 |
| **A4occ** | 2,547 / 1.302 | **4,949 / 2.125** | 7,287 / 2.893 |
| A5 | 3,432 / 1.413 | 7,880 / 2.873 | 15,213 / 4.780 |

The full per-arm, per-nprobe table, including p95 and p99, is `COMPARISON_TABLES.csv`.

## 3. Promotion gate (preregistered; candidate A4occ, selected on V before any native measurement)

| Criterion | Value | Pass |
|---|---|---|
| 1. Candidate reduction at matched R@10 ≥ 0.95 is ≥ 10% | **1.1%** (4,949 vs 5,006); bootstrap ratio 95% CI [0.751, 0.993] | **No** |
| 2. Median latency lower at matched recall | ratio 0.978 (2.125 vs 2.173 ms); bootstrap 95% CI [0.795, 0.983] | Yes |
| 3. No short results at matched points; ≥ 99% ranking agreement | 0 short results; 99.93% | Yes |
| 4. Budget: router ≤ 2× A0; training ≤ 2 h; routing p50 ≤ A0 + 0.5 ms | 25.2 MB (1.0×); 94 s; 0.497 vs 0.500 ms | Yes |
| 5. Gain held out | measured on G | Yes |
| 6. Integrity | partition verified, hashes verified | Yes |

**Decision: the gate failed on criterion 1.** By the stopping rule the campaign stops here. The 10M confirmation was
not submitted, and no further arms or tuning were attempted.

**Note on the bootstrap intervals.** The operating point is re-selected inside every resample, and both routers sit
near the 0.95 threshold between grid points 24 and 32. The resampled ratios are therefore lumpy and skewed: the point
estimate lies at the upper edge of its interval. The intervals reflect the grid as much as the routers.

## 4. Post-hoc diagnostic (not part of the gate)

The gate compares lowest observed grid points. Both routers landing on nprobe 32 hides most of the difference at 0.95.
At 0.97 the A0 → A4occ step (64 against 48) exaggerates it (−26% candidates).

To see past the grid, `posthoc_offline_g.py` replays the frozen artifacts on G with the exact offline simulation used
for selection, at every integer nprobe. Native and offline rankings agree on 99.93% of checks.

| Arm | Change in candidates vs A0 at coverage 0.90 / 0.95 / 0.97 |
|---|---|
| **A4occ** | −6.8% / **−10.3%** / −9.6% |
| A3occ | −7.7% / −4.1% / −3.4% |
| A4 | −4.9% / −3.6% / −3.1% |
| A2occ | −2.4% / −2.1% / +3.4% |
| A1 | +10.1% / +20.4% / +31.2% |
| A5 | +52.9% / +59.2% / +71.0% |

A0 needs 2,245 / 4,849 / 7,744 candidates (nprobe 14 / 31 / 50).

**Reading.** The best supervised router selects about 10% fewer candidates than centroid routing on held-out queries
at 95% coverage. That is right at the gate threshold, and smaller than the 14% seen on the selection set.

This diagnostic is descriptive and does not change the gate decision. The gate was preregistered on native grid
measurements, and its outcome stands.

## 5. Latency implication (analytic, not measured)

At 1M, routing is about 0.5 ms of the 2.2 ms query at the 0.95 point, and scoring about 1.6 ms. A 10% cut in scored
candidates therefore removes about 0.16 ms, roughly 7% of median latency. That is below the preregistered 10%
latency criterion for final success.

At 10M, routing is a smaller share (Tier 3B: about 1.0 of 12.4 ms), so a 10% cut in candidates would give roughly
9%. That figure is an extrapolation, not a measurement, and the 10M stage did not run.
