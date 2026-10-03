# Phase S.3B2a — tie-safe experimental Torch build routing

Starting point: `b337b0365` on `thesis/lmi-integration`. The native build and query paths, persisted format, training objective, and postings are unchanged. Torch build routing remains opt-in via `LMI_EXPERIMENTAL_TCH_BUILD_ROUTING=1`. Raw outputs are under untracked `work/phase_s3/s3b2a/`; S.3B2 baselines were preserved.

## Boundary and decision rule

At SIFT offset 802590, Torch ranked bucket 31 (19.119821548) over 30 (19.119817734), margin 3.8146973e-6. Native ranked 30 before 31, both scoring 19.119819641, margin zero. At offset 826019, Torch ranked 37 (12.785542488) over 17 (12.785541534), margin 9.5367432e-7. Native ranked 17 before 37, both scoring 12.785541534, margin zero. Native prefers the smaller bucket ID on exact ties. Torch sees small nonzero margins, so changing Torch's tie rule alone cannot fix these rows.

For finite Torch top scores `s1 >= s2`, the experimental rule is `s1 - s2 <= 4 * f32::EPSILON * (1 + max(abs(s1), abs(s2)))`. The scale is symmetric in the two dimensionless logits, stays nonzero near zero, and grows with score magnitude. Nonfinite logits are rejected; if subtraction of two finite extreme logits overflows, the row also falls back to native. Rows meeting the rule run the complete native `BuildRouter` over the original input; other rows accept Torch top-1. Both posting passes use this rule, retain bounded K-row input/logit buffers, and share a reusable native verifier. Cancellation checks remain active. No new persistence field is introduced.

The prior sweep at multipliers 0, 1, 2, 4 left respectively 2, 1, 0, 0 SIFT corpus disagreements. Multiplier 2 still missed 3 of 137 generated near-boundary disagreements; 4 was the smallest **tested** multiplier covering all of them and selected 3/1,000,000 original SIFT rows. The prior sweep used `1+abs(s1)`, equal to the symmetric expression for these nonnegative, ordered top scores. This empirical threshold is not a numerical guarantee for every future model or input.

The 544 test-only rows perturb the first 16 eligible coordinates of the two observed SIFT vectors by ±1–128 float32 ULPs plus zero. They produced 137 raw Torch/native disagreements; the implemented fallback covered all 137 and selected 532/544 near-boundary rows. The native winner was in Torch top-2, top-4, and top-8 for **2/2 real** and **137/137 generated** disagreements. This is diagnostic containment, not a global top-M bound. Option A, full native verification, was implemented and measured. Option B, native first layer plus only Torch-selected outputs, was not implemented because observed top-M containment is not a bound for future rows. Option C, reusing Torch hidden activations with native selected outputs, was not implemented because first-layer numerical differences can survive into those outputs. With only three SIFT fallback rows, optimizing the rare verifier is not justified by corpus evidence; the tie-heavy synthetic fixture instead exposes a model-dependent limitation.

## Corpus and persisted-index results

Runs used preserved seed-42 models, Qdrant preprocessing, CPU 0, one Torch intra-op thread, and K=256. The actual fallback method was tested against native assignment for every corpus row and every generated near-boundary row. The build harness compared persisted files and reopened a learned query.

| Corpus | Rows | Raw Torch disagreements | Unique fallback rows | Remaining disagreements | Count+fill fallback calls | Native / raw Torch / tie-safe build (s) | Tie-safe count / fill (s) | Reopen (s) | Peak RSS (KiB) |
|:--|--:|--:|--:|--:|--:|:--|:--|--:|--:|
| SIFT1M | 1,000,000 | 2 | 3 (0.0003%) | 0 | 6 | 7.681152 / 2.175770 / 3.031786 | 0.937196 / 0.922797 | 0.038398 | 1,903,520 |
| LAION | 99,780 | 0 | 0 | 0 | 0 | 4.882435 / 1.297080 / 1.408158 | 0.232719 / 0.231152 | 0.011443 | 1,077,904 |

SIFT tie-safe sampling, clustering, and training/export took 0.006372, 0.103655, and 0.088653 s; LAION took 0.000683, 0.459508, and 0.175073 s. The SIFT whole build was 2.53× faster than its native reference and 0.856 s slower than raw Torch. LAION was 3.47× faster than native and 0.111 s slower than raw Torch. Each figure is one run without a confidence interval. For both corpora, `lmi_state.json`, `lmi_router.bin`, and `lmi_postings.bin` hashes matched the native reference byte-for-byte. Reopen used `StaticLearned` without training, and the harness query returned the same IDs and scores; this verifies one query per corpus, not a retrieval distribution.

## B=10,000 performance gate: negative result

The preserved d=768, H=512, B=10,000 synthetic fixture has 2,048 rows. Its deterministic output-weight and bias formulas repeat every 3,811 buckets (for example buckets 3,322 and 7,133), yielding structural exact ties. All 2,048 rows triggered native fallback. This is not a trained large-B model, but it is the prescribed throughput gate.

| K | Prior raw Torch rows/s | Fresh raw Torch rows/s | Tie-safe rows/s | Tie-safe µs/vector | Fallbacks | Disagreements |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 7,029 | 6,841 | 325 | 3,077 | 2,048/2,048 | 0 |
| 128 | 7,688 | 7,396 | 331 | 3,023 | 2,048/2,048 | 0 |
| 256 | 6,575 | 7,801 | 333 | 3,004 | 2,048/2,048 | 0 |

Fresh scalar was 338 rows/s; prior scalar was 382 rows/s. These single runs vary with cache/frequency. On this fixture the fallback collapses to native throughput, so the desired order-of-magnitude B=10,000 acceleration is **not achieved**. Top-2 outputs add at least `24*K` bytes per batch (two f32 values and two i64 IDs per row), plus Torch scratch and a reusable native workspace. Actual peak RSS is not determined by that lower bound. The opt-in backend is native-compatible on the tested corpora, but the rule does not prove universal parity or throughput. A real trained large-B model must be measured before promotion.

## CPU permit and next gate

The collection optimization worker initially obtains an I/O-only `ResourcePermit`; `lib/shard/src/optimize.rs` converts it through `ResourceBudget::replace_with` to a CPU permit. `SegmentBuilder::build` passes it via `VectorIndexBuildArgs` to LMI. The budget can grant p>1, but LMI currently checks only `num_cpus > 0`. `train_with_tch` sets process-global Torch intra-op threads to 1 and once-only inter-op threads to 1, regardless of p. Setting intra-op threads to p for concurrent builds without coordinated admission could race over Torch's global pool or oversubscribe CPUs. A future p-CPU design needs process-wide thread policy and explicit build scheduling; this phase makes no resource-policy change.

Keep the environment-gated backend, not a public persistent enum or automatic default. The threshold is an internal numerical policy, not an ANN hyperparameter. The next major phase remains **S.3C: scalable native spherical KMeans**, with a realistic trained B=10,000 tie-rate/throughput gate before claiming production readiness for this Torch path.

## Reproduction and verification

Use `LIBTORCH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch`, `LD_LIBRARY_PATH=$LIBTORCH/lib`, `TORCH_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, and `taskset -c 0` for timed tests. Build the ignored optimized harness with `cargo test --profile perf -p segment --lib --features lmi-training --locked phase_s3b2a_ --no-run`. Run its `phase_s3b2a_corpus_boundary_diagnostic`, `phase_s3b2a_large_router_microbenchmark`, and `phase_s3_build_benchmark` tests with `--ignored --nocapture --test-threads=1` and new output paths. The diagnostic uses `LMI_S3B2A_DATA`, `LMI_S3B2A_ROUTER`, `LMI_S3B2A_OUTPUT`, plus `LMI_S3B2A_SEED_OFFSETS=802590,826019` for SIFT; the microbenchmark uses `LMI_S3B2A_MICRO_OUTPUT`. The build uses `LMI_S3_DATA`, `LMI_S3_CONFIG`, `LMI_S3_OUTPUT`, and the opt-in Torch variable. Compare native/build directories with `tests/lmi_phase_s3b2_compare.py`. Raw JSON, exact commands' logs, and time/RSS outputs are in `work/phase_s3/s3b2a/`.

Observed verification: focused boundary test 1/1; SIFT/LAION corpus diagnostics 2/2; large-B regression 1/1 (negative performance gate); SIFT/LAION build/reopen 2/2; persisted comparisons 2/2; filtered LMI unit tests 16 passed, 11 intentionally ignored (plus two matching integration tests); explicit LMI integration tests 39/39. Segment, collection-tests, edge, default server, and training-enabled server checks passed with `--locked`. Targeted Clippy completed with repository/test-harness warnings; strict `-D warnings` fails on unrelated existing HNSW formatting and routing/training lints. Stable rustfmt warns about nightly-only import grouping. Final format/diff checks and Git status follow in final review.
