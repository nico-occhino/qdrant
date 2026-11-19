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

## Native build profile at 1M (gate for Tier 3)

**Status:** complete, 2026-11-18. Recorded in full in `ivf_ablation/tier3/TIER3_PREREGISTRATION.md`, Amendment A.

**Jobs**
- **25718667** failed at `build_ivf8192`: the native sample-budget cap rejected S = 524,288. Commit `f2f6f27d8` raised the opt-in limit to the `LmiConfig` maximum.
- **25718749** (retry) ran on zenon51 with 8 CPUs and 32 GB, against the 1M source at commit `f2f6f27d8`. It exited 0 after 5 h 35 min of a 6 h walltime, and wrote `COMPLETE.json`.
- **Local copy:** `hpc-native-profile/` (read-only).

| Index | Build at 1M | Index size | Validation (1M gold) |
|---|---|---|---|
| IVF-8192 (S 524,288, 20 iterations) | k-means 14,492 s single-threaded, plus postings 379 s | 29.3 MB | nprobe 32: R@10 0.949, p50 16.68 ms (routing 14.30 ms) |
| LMI-3162 | k-means 733 s, MLP 2,404 s, postings 120 s | 12.1 MB | nprobe 64: R@10 0.964, p50 14.96 ms (routing 5.08 ms) |
| HNSW m16/efc100 | 168 s on 16 threads | 40.3 MB | ef 128: R@10 0.966, p50 0.98 ms |

**Consequences (Amendment A)**
- Native routing ran at only about 0.44 GMAC/s, which led to the router constant-factor fix (`aaca97f24`).
- The single-threaded k-means teacher at B = 8,192 takes 4.0 h, so IVF-8192 was deferred to Tier 3B.
- The profile's build stages gave the Tier 3A walltime prediction (≈ 4.5 h). The actual run took 4 h 19 min.

## Tier 3A: native Qdrant v1.20 latency at 10M

**Status:** complete, 2026-11-19. Preregistration: `ivf_ablation/tier3/TIER3_PREREGISTRATION.md` and Amendment A.

**Job.** 25738234 ran on elmu1-8 (AMD EPYC 9474F, 16 of 192 cores allocated) at commit `ec557c427`, binary sha256 `44e30e84…`. It exited 0 after 4 h 19 min of wall time; the prediction was ≈ 4.5 h. It wrote `COMPLETE.json` and its checksums verify; the `job.log` mismatch is the known appended-line pattern, and the prefix hash matches. Every query returned 30 results, so the decomposed path never underfilled.

**Evidence**
- Local copy: `hpc-tier3a/` (read-only).
- Analysis: `analysis/tier3a.{json,txt}`, produced by `ivf_ablation/tier3/analyze_tier3.py`.
- Recall: returned IDs against the official SISAP23 top-30. The harness stores row *i* as point *i* + 1.
- Panels: 2,000 validation and 2,000 test queries; Plain used 256 of each.

**Test at the validation-locked R@10 ≥ 0.95 point**

| System | Setting | R@10 [95% CI] | Candidates | Routing p50 | p50 / p95 / p99 (ms) | QPS at 1 / 16 threads |
|---|---|---|---|---|---|---|
| HNSW m16/efc100 | ef 256 | 0.961 [0.955, 0.966] | — | — | 1.30 / 1.81 / 2.04 | 760 / 6,335 |
| **IVF B 3,162 (centroid router)** | nprobe 16 | 0.951 [0.946, 0.956] | 65,577 | 0.98 ms | 20.50 / 27.75 / 31.88 | 48 / 317 |
| LMI B 3,162 (accepted) | nprobe 30 | 0.958 [0.953, 0.962] | 119,018 | 1.57 ms | 34.86 / 44.85 / 50.16 | 29 / 218 |
| Plain (exact) | — | 0.994 [0.991, 0.996] (n 256) | 10.1M | — | 612 / 630 / 636 | 1.6 / 7.6 |

At R@30 ≥ 0.90 the locked points are HNSW ef 64 (0.43 ms), IVF nprobe 16 and LMI nprobe 16 (both 20.5 ms). At that point IVF has the higher recall, 0.944 against 0.920.

**Build and size (10M, index only)**

| System | Build | Threads | Index bytes | Breakdown |
|---|---|---|---|---|
| HNSW | 596 s | 16 | 454.5 MB | — |
| IVF-3162 | 766 s | 1 | 50.2 MB | k-means 193 s, postings 567 s |
| LMI-3162 | 1,601 s | 1 | 48.6 MB | k-means 191 s, MLP 910 s, postings 492 s |

Peak RSS was 16.9–18.9 GB for all systems, dominated by the 15.8 GB Float16 vector storage.

### Answers to the preregistered questions

- **Q1. Yes.** At the locked R@10 points, native IVF scores **0.551×** the candidates of LMI [0.548, 0.554] and has **0.588×** its p50 latency. Its test recall is 0.007 lower [−0.010, −0.004], and both clear 0.95.

  At equal nprobe, IVF recall is higher at every width: +0.026 at nprobe 8, falling to +0.005 at 128, and every CI excludes 0. Candidates and latency are within 6% of each other at equal nprobe.

  The native result is therefore the same as Tiers 1 and 2: the centroid router reaches the target with about half the routing width. The native candidate ratio (0.55) is lower than Tier 2's offline 0.68. The likely reason is the coarse nprobe grid, where LMI's lock jumps from 16 to 30.
- **Q2.** HNSW is about **16× faster** than the best learned-partition index at p50 (1.30 against 20.5 ms) and about **20× faster** at 16 threads (6,335 against 317 QPS), at slightly higher recall.

  It costs **9×** the index bytes (455 against 50 MB) and up to about 9,500 CPU-seconds to build (596 s × 16 threads), against 766 single-threaded seconds for IVF. Wall-clock build time is close: 596 s against 766 s.

  Both IVF and LMI are bound by candidate scoring: about 0.3 µs per scored Float16 vector, and routing is under 5% of latency.
- **Q3.** LMI is **not on the recall–latency–build front**. IVF has the same B, the same teacher and the same postings path, and at every recall level it is equal or faster, with 0.48× the build time.

  The only axis on which LMI is non-dominated is index bytes, where it is 1.6 MB (3%) smaller than IVF. That is immaterial next to the 15.8 GB of vectors.

  The overall front is HNSW for latency and IVF for index bytes and CPU build cost.

### Notes and deviations

- **Plain repetitions.** Plain ran 3 test repetitions, not the 1 that A3 states, because the job script applies the test repetition count to every kind. This adds data and changes no operating point.
- **Plain agreement with the gold.** Native exact search agrees with the official gold at R@10 = 0.994 (test-256), above the offline `oracle_check.py` value of 0.988 on the same queries. The offline check ranks by raw inner product over the stored Float16 rows, whose norms deviate from 1 by up to 6×10⁻⁴; native Cosine renormalises. Near-tie flips are the likely cause. This is not verified.
- **Thread scaling of IVF and LMI is poor and non-monotone.** For example, LMI at nprobe 16 reaches 194 QPS at 16 threads but 668 at nprobe 8. The likely cause is memory bandwidth or allocation contention on the shared node. Treat the 16-thread throughput figures for these two indexes as indicative only.
- **Settings grid (deviation).** The run used the grid in `tier3-configs.tsv`: nprobe {8, 16, 30, 64, 128} for IVF and LMI and ef {32, …, 1,024} for HNSW. The preregistration lists nprobe {4, …, 256}, and Amendment A does not record the narrower grid. Every system crossed both recall targets inside the grid, so no locked point is affected. The coarser steps do make the locked comparison less precise (see Q1).
- **IVF-8192.** Deferred to Tier 3B (Amendment A2).

## Files and locations (Tier 3)

Repository paths are relative to the repository root on branch `research/static-lmi-ivf-router-fix`.

| File | Role | Commit |
|---|---|---|
| `lib/segment/src/index/lmi_index/routing.rs` | Router constant-factor fix: shape-only per-query check, SIMD dot product, partial top-nprobe | `aaca97f24` |
| `lib/segment/src/index/lmi_index/spherical_kmeans.rs` | Parallel k-means assignment on the build CPU permit; bit-identity test; ignored 50K benchmark | `3e49dabcd` |
| `lib/segment/src/index/lmi_index/training.rs`, `build.rs` | Pass the CPU permit to the teacher | `3e49dabcd` |
| `lib/segment/tests/ann_matched_v120.rs` | Native index-level harness; optional top-level `"threads"` for LMI | `3e49dabcd` |
| `research/lmi/ivf_ablation/tier3/TIER3_PREREGISTRATION.md` | Preregistration plus Amendments A (3A) and B (3B) | `ec557c427`, `891fa7b87` |
| `research/lmi/ivf_ablation/tier3/tier3-configs.tsv` | Systems and settings grids; the 3A version is at `ec557c427`, the 3B version at `891fa7b87` | — |
| `research/lmi/ivf_ablation/tier3/ann-10m-native.pbs` | 10M PBS job script, shared by 3A and 3B | `ec557c427` |
| `research/lmi/ivf_ablation/tier3/profile-native.pbs` | 1M native profiling job script | `1473f76ce`, `f2f6f27d8` |
| `research/lmi/ivf_ablation/tier3/analyze_tier3.py` | Recall against the official gold, validation-locked points, paired CIs | `305a97223`, `891fa7b87` |

**Local evidence**, under `research-admin/experiments/2026-11-18-static-lmi-ivf/` (copies are read-only):
- `hpc-native-profile/`: profile job 25718749.
- `hpc-tier3a/`: Tier 3A job 25738234.
- `analysis/tier3a.{json,txt}`: Tier 3A analysis output.
- `native-staging-ec557c427/` and `native-staging-891fa7b87/`: the staged inputs for 3A and 3B.
- `hpc-submissions/submitted.log`: the submission log.

**Remote (MetaCentrum)**, under `/storage/budejovice1/home/nico/lmi-metacentrum/static-lmi-ivf-20261118/`:
- `native-<commit>/`: inputs for each revision.
- `results/ann-10m-<job>/`: outputs, with 3A in `ann-10m-25738234…` and 3B (job 25743351, pending) in `ann-10m-25743351…`.
