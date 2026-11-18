# Does the learned router help? Static LMI versus spherical IVF-Flat (Tier 1, 1M)

**Status:** Tier 1 complete, 2026-11-18. The analysis is validated; selection followed Amendment 01 (validation set).

**Evidence:** `research-admin/experiments/2026-11-18-static-lmi-ivf/`
- `tier1/` holds 11 runs, 114 traces, and `/usr/bin/time` logs, all exit 0.
- `analysis/tier1-selection-val.{json,txt}` and `analysis/tier1-eval-test.{json,txt}`.
- `analysis/finalists-val.json` and `finalists-test-confirmation.json`.
- The superseded selector output is kept.

**Code:** `ivf_ablation/lab.py`, Tier 1 sha256 in `tier1-inputs.sha256`; `analyze_tier1.py`; `select_finalists.py`.

**Design:** `ivf_ablation/PREREGISTRATION_TIER1.md` plus `AMENDMENT_01_VALIDATION_SELECTION.md`.

## Setup in one paragraph

**Data and queries**
- First 1,000,000 SISAP23 CLIP-768 Float16 vectors (cosine).
- 2,000 validation queries (official rows 1000–2999) and 2,000 test queries (rows 4000–5999).
- Exact top-30 gold over the subset, with the same Float16 semantics as the stored data.

**Teacher.** Spherical k-means reproducing `spherical_kmeans.rs`:
- `native`: S = 32,768 samples, 5 iterations, as in the accepted configurations.
- `converged`: S = max(32,768, 64·B), 20 iterations.

**Router.** The native MLP 768 → 512 → B, trained with Adam at learning rate 1e-3 for 30 epochs, on cross-entropy against the teacher labels of the sample.

**What is compared.** Both partitions crossed with both routers:
- MLP-argmax postings (native LMI) or Voronoi postings (IVF);
- MLP ranking or exact centroid ranking.

**Work** is the live candidates scored. Routing is reported in scored-vector equivalents (MACs / 768).

**Estimand.** Mean work to reach R@10 = 0.95 (also R@30 = 0.90), on exact per-query traces over a 64-point budget grid. Uncertainty is a paired bootstrap over queries. There is one construction per configuration, so the intervals cover query sampling only.

## Results (test set; validation numbers within 7% and the same ordering)

Mean scored work to reach R@10 = 0.95 (lower is better), with routing vector-equivalents per query in brackets.

| B | Teacher | LMI (mlp \| mlp) | IVF (centroid \| centroid) | IVF / LMI [95% CI] |
|---|---|---|---|---|
| 64 | native | 96,299 [555] | 93,309 [64] | 0.969 [0.913, 1.006] |
| 64 | converged | 92,677 | 86,219 | 0.930 [0.890, 0.976] |
| 256 | native | 46,840 [683] | 41,022 [256] | 0.876 [0.827, 0.923] |
| 256 | converged | 43,284 | 37,235 | 0.860 [0.806, 0.903] |
| 1,024 | native | 27,845 [1,195] | 21,426 [1,024] | 0.769 [0.724, 0.815] |
| 1,024 | converged | 22,346 | 16,804 | 0.752 [0.715, 0.809] |
| 4,096 | native | 21,863 [3,243] | 13,753 [4,096] | 0.629 [0.585, 0.674] |
| 4,096 | converged | 10,688 | 6,918 | 0.647 [0.596, 0.699] |

The full 2×2 at every B, the R@30 estimands, the nprobe stopping rule and the oracle costs are in `analysis/tier1-eval-test.json`.

### Findings

1. **The native 512-unit MLP router never beats exact centroid routing.** At a fixed partition, the centroid/MLP work ratio is 0.555 to 0.995 for every B and teacher. Intervals exclude 1 at B ≥ 256 and include it at B = 64.
   - The one exception is the D4 linear router. On its own (worse) partition, it beats centroid routing on that same partition (ratio 1.106 [1.041, 1.153]).
   - It still needs more work than true IVF (24,967 against 21,426 at B = 1,024, native teacher).
   - **IVF needs 3–37% less work than the native LMI**, and the advantage grows with B. The same direction appeared in continuation28 (65K) and in the Phase F/F2/F3 100K studies; this is the first large controlled measurement.

2. **The MLP does not learn the centroid partition; it memorises the sample.**
   - Training accuracy on the S sample points is 0.997–1.000.
   - Agreement between MLP-argmax postings and Voronoi assignment over all 1M points is only 0.91 at B = 64, 0.85 at 256, 0.76–0.81 at 1,024, and 0.64–0.82 at 4,096.
   - With the native S = 32,768 at B = 4,096 there are 8 training points per class. The MLP then produces more imbalanced postings: bucket CV 1.34 against 1.26 for the native teacher's Voronoi cells, and 0.63 for the converged teacher.

3. **Partition granularity is the dominant lever.** Work falls about 7× from B = 64 to B = 1,024 for both families, and about 13× to B = 4,096 for IVF. The gold-bucket oracle cost for top-10 falls from 45K to 1.4K. Router rank overhead grows with B:

   | B (converged) | Rank/oracle, IVF | Rank/oracle, LMI |
   |---|---|---|
   | 64 | 1.6× | 1.7× |
   | 4,096 | 6.7× | 11.8× |

   Finer partitions put a premium on routing quality, and centroid ranking handles that better than the MLP.

4. **The teacher matters.** At B ≥ 1,024 the converged teacher (S/B = 64, 20 iterations) cuts IVF work by 22% at B = 1,024 and by 50% at B = 4,096, against the native S = 32,768 with 5 iterations. That is at least as large as the router effect. FAISS gives the same guidance: 30–256 training points per list.

5. **Simplifications (D4, B = 1,024, native teacher), test ratio against H = 512:**

   | Variant | Ratio [95% CI] |
   |---|---|
   | H = 128 | 1.040 [0.985, 1.089] |
   | 10 epochs | 0.979 [0.927, 1.028] |
   | Linear (no hidden layer) | 0.897 [0.841, 0.966] |

   A linear softmax router is *better* than the 512-unit MLP, and it is a learned centroid-like scorer. It still loses to exact centroid routing on the native IVF partition: 24,967 against 21,426.

6. **The stopping rule barely matters.** Fixed nprobe and work-budget stopping give the same W within about 5% everywhere, because the ranking order is identical.

7. **Construction.**
   - At B = 4,096 converged, MLP training took 3,243 s of lab time, against 178 s for the teacher and 16 s for centroid assignment. IVF needs no training at all.
   - Router memory: MLP 768·512 + 512·B + biases, which is 1.6 MB plus 2 KB·B; centroids take 3 KB·B. They are equal near B = 1,536.

### Selection for 10M (Tier 2)

The rule is Amendment 01 with both preregistered D3 caps: routing ≤ 10% of total work, and construction ≤ 10× native.

- **F2 (best IVF):** B1024-converged centroid|centroid. Validation W_total 18,781; test 17,828.
- **F3 (best LMI):** B1024-converged mlp|mlp. Validation 24,499; test 23,541.
- **F4 (simplification):** the B1024-native linear router.
- **Excluded by the routing cap:** B4096-converged. Its total work is lower (IVF 11,145), but exact centroid routing is 37% of it.

Validation and test select the same configurations.

**Disclosure.** An earlier selector run omitted the routing cap and chose B4096; it is kept as `*.SUPERSEDED-missing-routing-cap.json`.

## Interpretation

On this workload the neural router adds cost (training time, inference MACs below B ≈ 1.5K) and loses candidate efficiency against the centroids it was trained to imitate. The mechanism is that a classifier trained on a small sample of hard Voronoi labels generalises worse than the Voronoi function itself, both for corpus assignment and for ranking the second and further buckets.

The improvements that matter are conventional:
- finer partitions (√N to 4√N lists);
- an adequately sampled, converged spherical k-means;
- exact centroid routing.

Every effect points the same way. These are 1M offline coverage results. Tier 2 tests whether they hold at 10M with the official oracle, and Tier 3 measures latency natively.

## Limitations

- One seed and construction per configuration.
- Offline lab replicas: the algorithms are matched to native code, but the RNG streams are not.
- Coverage recall, not native returned recall.
- 1M prefix of the corpus.
- No latency in this tier.
- The routing cap in D3 is a preregistered design choice that favours moderate B. A coarse quantizer (for example HNSW over centroids) would relax it, but that is out of scope here.

## Tier 2: 10M offline confirmation on MetaCentrum

**Status:** complete, 2026-11-18.

**Jobs.** All exited 0, wrote `COMPLETE.json`, and verified their checksums. The `job.log` mismatch is the known pattern: its final line is appended after `SHA256SUMS`, and the prefix hash matches.

| Job | Node | Wall time | CPU-hours |
|---|---|---|---|
| 25718516 (F1) | alfrid3 | 12 min | 2.5 |
| 25718517 (F2+F3) | tyra17 | 1 h 26 min | 21.1 |
| 25718522 (F4) | adan4 | 1 h 04 min | 10.7 |

**Evidence**
- Local copy: `hpc-tier2/` (read-only).
- Remote: `static-lmi-ivf-20261118/results/T2-*`.
- Analysis: `analysis/tier2-{val,test}.{json,txt}`.
- Code: frozen at commit `1641abb22` (`code.tar`, sha256 in `code.tar.sha256`).
- Corpus: the full 10,120,191 Float16 vectors, staged to `$SCRATCHDIR` and re-hashed.
- Oracle: official SISAP23 top-30 (coverage recall). 2,000 validation and 2,000 test queries.

**Test results.** Work needed to reach R@10 = 0.95: scored vectors, routing vector-equivalents, and their total. Validation gives the same ordering; validation-locked points reach test R@10 of 0.948–0.954.

| Configuration | Scored [95% CI] | Routing | Total | R@30 = 0.90 scored |
|---|---|---|---|---|
| **IVF, B 8,192, converged teacher (F2)** | **29,332** [26,172, 32,326] | 8,192 | **37,524** | 16,571 |
| Linear router, B 8,192 (F4) | 42,231 [37,652, 47,535] | 8,192 | 50,423 | 22,848 |
| LMI, B 8,192, converged teacher (F3) | 49,236 [43,408, 55,281] | 5,973 | 55,209 | 25,567 |
| IVF, B 3,162, native teacher | 58,537 [52,459, 65,044] | 3,162 | 61,699 | 33,036 |
| **Native LMI (accepted configuration, B 3,162)** | 85,949 [75,422, 96,969] | 2,620 | 88,569 | 45,939 |

**Paired ratios, IVF over LMI with the same teacher and B**

| B | R@10 | R@30 |
|---|---|---|
| 3,162 | 0.681 [0.633, 0.741] | 0.719 [0.685, 0.756] |
| 8,192 | 0.596 [0.550, 0.642] | 0.648 [0.620, 0.681] |

The router effect at a fixed partition (centroid over MLP) is 0.83 at B = 3,162 and 0.74 at B = 8,192. The partition effect at a fixed centroid router is 0.82 and 0.80.

**MLP versus Voronoi agreement over all 10M points:** 0.773 at B = 3,162 and 0.764 at B = 8,192. The MLP fits its training sample at 0.994 and 0.997.

**Linear router (F4).** It beats the MLP. On its own partition, centroid routing is slightly worse than the linear router (ratio 1.06 [1.00, 1.12]). It still needs 35% more total work than F2.

**Construction (lab, 16 cores):**

| Configuration | Teacher | MLP training | Assignment over 10M |
|---|---|---|---|
| F2+F3 (B 8,192, S 524K) | 188 s | 4,446 s | 159 s centroid / 225 s MLP |
| F1 (B 3,162, S 250K) | 13 s | 471 s | — |

**Promotion (Tier 2 rule).** F2, IVF B 8,192 with the converged teacher (S 524,288, 20 iterations), goes to Tier 3 together with the native LMI baseline. On these offline coverage measurements it needs about **58% less total work** than the accepted native LMI at R@10 = 0.95.
