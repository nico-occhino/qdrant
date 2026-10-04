# Phase S.3C: native spherical clustering checkpoint

## Scope and repository state

The integrated cosine LMI teacher now uses spherical KMeans. The existing Euclidean Lloyd implementation remains available for Euclidean, dot-product, and Manhattan fields. Qdrant still owns sampling, Torch training, posting construction, persistence, and query execution. The production cosine assignment reuses Qdrant's f32 dot-product kernel; a separate f64 scalar implementation remains a correctness oracle. There is no FAISS or Python production dependency. This checkpoint does not claim a 10M index build, a 100M test, or a 1B test.

At entry, HEAD was `f7b613262` on `thesis/lmi-integration` and the only pre-existing working-tree content was preserved raw output under `work/phase_s3/`. Those 169 raw files were unexpectedly staged and were unstaged without deletion. No previous commit was amended or pushed.

## Reference audit and target semantics

The preserved Coda reference snapshots under `work/phase_s/references/Coda-Research-Group_LearnedMetricIndex/` were inspected read-only. The SISAP Python `task1.py` calls `faiss.Kmeans(..., spherical=True)` and chooses `B=int(alpha*sqrt(N))`; the FAISS wrapper default is 25 iterations unless overridden. The enhanced 100M/1B script also uses spherical FAISS clustering; its GPU variant selects GPU FAISS. Those scripts sample large corpora and process corpus vectors in chunks. The preserved Rust wrapper instead calls a third-party Euclidean KMeans with random-sample initialization. The third-party crate implementation is absent from the snapshot, so its accumulation precision, empty-cluster policy, and SIMD/threading cannot be inferred from the wrapper. See the [Coda SISAP code](https://github.com/Coda-Research-Group/LearnedMetricIndex/blob/paper-sisap24-indexing-challenge/task1.py), [enhancement branch](https://github.com/Coda-Research-Group/LearnedMetricIndex/tree/enhancing-performance/experiments), [Rust wrapper](https://github.com/Coda-Research-Group/LearnedMetricIndex/blob/rust/final/rust_lmi/src/lib.rs), and [FAISS clustering interface](https://github.com/facebookresearch/faiss/blob/main/faiss/python/extra_wrappers.py).

For Qdrant-preprocessed cosine vectors, the new algorithm initializes B distinct sampled rows in seeded shuffled order, normalizes nonzero seeds, assigns each row to the maximum centroid dot product, accumulates each cluster in f64, divides by its count, and renormalizes nonzero means. Empty clusters and zero-mean clusters retain their previous centroid. Exact score ties choose the lowest bucket ID. Exact zero vectors remain zero in Qdrant cosine preprocessing and therefore tie to bucket zero. Assignment stops on stable labels or at the configured iteration cap. A final reassignment uses the final centroids. Nonzero samples are checked for unit norm within 1e-3; Qdrant's near-zero preprocessing exception is accepted.

The scalar oracle computes dot products and reductions in f64. Production casts centroids to f32 for Qdrant `DotProductMetric::similarity`, whose architecture-specific SIMD reduction may differ near a decision boundary. Therefore bitwise agreement on all inputs is not asserted. The temporary f32 centroid matrix is dropped before f64 accumulators are allocated. This keeps the peak within the existing build-plan shape. On a tiny independent right-angle, one-centroid fixture, FAISS spherical KMeans and the native scalar reference both produced approximately `[0.70710678, 0.70710678]`.

## Memory and ownership

Let S be sample rows, B buckets, d dimensions, and I iterations. Sample storage is `4Sd` bytes. f64 centroids are `8Bd`; f64 accumulators are another `8Bd` while updating. The SIMD assignment temporarily adds `4Bd` f32 centroids, but they do not overlap the accumulator matrix. Labels plus final labels are at most `16S` bytes at the final-assignment boundary; cluster counts add `8B` bytes, with bounded vector/header and statistics overhead. There is no `S×B` score matrix. Checked multiplication protects centroid planning; the existing `BuildPlan` still enforces its component memory ceiling. The synthetic 250k×1,000 benchmark can exceed a real builder's admission ceiling and must not be called an integrated build.

The path from shard optimizer through CPU resource permit, `VectorIndexBuildArgs`, and LMI builder was traced. Although a permit can carry more than one CPU, integrated clustering remains single-threaded. Torch training separately limits intra/inter-op threads. No unconstrained Rayon pool or per-worker `8Bd` accumulator replicas were introduced. At B=10,000 and d=768, one f64 accumulator matrix alone is 61,440,000 bytes, so parallel reduction requires its own explicit memory budget.

## Controlled synthetic scale measurements

Optimized Rust test profile, deterministic normalized synthetic d=768 rows, seed 42, three configured iterations, one worker. Times are seconds. Peak RSS is the test process high-water mark, which includes runtime/test-harness overhead. Raw per-iteration results and process logs are retained in `work/phase_s3/s3c/`. These measurements exclude corpus classification, Torch training, persistence, and queries.

| S | B | Scalar full KMeans | Qdrant-dot full KMeans | Label differences | Peak RSS (native-only where available) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 64 | 0.168 | 0.0219 | 0 | process value in raw JSON |
| 10,000 | 100 | 1.289 | 0.1797 | 0 | process value in raw JSON |
| 100,000 | 316 | 38.434 | 5.302 | 0 | process value in raw JSON |
| 250,000 | 1,000 | not run | 45.332 | no scalar comparison | 872,176 KiB |

For the three paired cases, centroid coordinate differences were exactly zero in the recorded runs. Final assignment alone was approximately eight times faster with the native dot kernel on the same final centroids, also with zero label differences. This is evidence for the tested random fixtures, not universal parity near floating-point ties. The 250k run spent about 11.1-11.4 seconds in each assignment pass and 11.0 seconds in final assignment; accumulation and normalization were minor. It had zero empty clusters, with min/median/p95/max sizes 211/250/271/295. These numbers are synthetic and cannot establish 10M end-to-end throughput.

The next 1M×3,162 sample would require roughly 12.6 times as many dot products per pass as 250k×1,000. This is a complexity estimate, not a measured runtime. It was deliberately not launched blindly.

## LAION ~100k semantic comparison

The Phase F corpus/query split and old output directory were preserved. A new full harness run in `work/phase_s3/s3c/laion100k-spherical/` used symlinks to the same 99,780-vector corpus and 20 warmup plus 200 held-out queries, two trials, fixed seed/configuration (S=2,048, B=64, H=64, 30 epochs). The existing control builder remains Euclidean and its postings/results were unchanged. Since the baseline was produced at a previous code checkpoint, absolute timing and persisted-size differences are not isolated effects of the spherical change; retrieval quality/candidate counts are the primary semantic comparison.

| Method, nprobe=4 | Old Recall@10 | New Recall@10 | Old mean candidates | New mean candidates | New median harness ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP LMI | 0.9015 | 0.9010 | 14,504 | 14,616 | 4.72 |
| Euclidean centroid control | 0.8975 | 0.8975 | 10,197 | 10,197 | 3.20 |

The new MLP has 23/64 empty corpus buckets (old: 24/64). Held-out query top-1 agreement against the unchanged Euclidean centroid control is 0.6045 (old: 0.6364). The new LMI clustering stage measured 0.0762 s; whole LMI build 4.8444 s. Those timings are from a different code checkpoint and run, so they do not isolate a speedup against the old baseline. The new harness also rebuilt HNSW, validated native LMI search against common candidate scoring, and passed. Spherical training corrected metric semantics but did not improve MLP candidate efficiency in this configuration.

The currently recorded teacher agreement compares the spherical-trained MLP against the unchanged Euclidean control, not against spherical teacher labels. A dedicated spherical-teacher confusion/balance diagnostic remains to be collected; do not interpret that agreement as spherical training accuracy. The exact MLP corpus bucket sizes, all per-query observations, build decomposition, and plots are in the new raw directory. No baseline artifact was overwritten.

## Scaling interpretation and open gates

`B≈sqrt(N)` gives average bucket population near `sqrt(N)` in an ideal balanced partition. At N=10M and B≈3,162, four probes would visit about 12,648 candidates (0.126%) only in that balance model. At N=100M and B=10,000, the analogous figure is 40,000 (0.04%). Imbalance, routing, and overlap change reality. Increasing B also grows the MLP output layer roughly as H×B and clustering work as I×S×B×d. With H=512, output weights alone are about 6.48 MB at B=3,162 and 20.48 MB at B=10,000 (f32).

A bounded real-data trained-router gate used the first 10,000 rows of the preserved LAION corpus as a training sample, the next 2,048 as validation, d=768, B=1,000, H=512, 30 epochs, and three spherical iterations. The teacher had 1,000 active buckets (sample sizes 1 to 60). Training-sample teacher accuracy was 0.995. On validation, raw Torch/native top-1 mismatches were 0/2,048; tie-safe fallbacks and residual mismatches were also 0. The minimum raw top-two margin was 0.0002804, p01 0.01616, and median 1.1211. Teacher time was 1.906 s; second teacher plus training 18.969 s. This supports a fast path for this trained B=1,000 fixture only, not for B=10,000 or an entire 10M corpus. The prior B=10,000 synthetic structural-tie fixture is not evidence for a trained router. The current `BuildPlan` component ceiling and measured clustering work need a deliberate admission/scale review before 10M. There is no authorized remote/HPC storage destination established in this task, so no 15 GB dataset was downloaded or 10M ingestion/build/retrieval attempted. A bounded HDF5-to-REST streaming importer was prepared and smoke-tested using 128 dry-run rows plus 125 rows against an isolated mock endpoint; it has not been validated against a live 10M Qdrant collection. The official source and storage-safe preparation are in `docs/lmi-laion10m-runbook.md`.

## Files and verification

- `spherical_kmeans.rs`: deterministic spherical trainer, scalar oracle, Qdrant native-dot path, cancellation, validation, tests.
- `training.rs`: explicit cosine-vs-existing-Lloyd teacher selection.
- `build.rs`: pass the vector metric into training.
- `mod.rs`: feature-gated module and microbenchmark registration.
- `evaluation_s3c.rs`: explicit ignored synthetic scale/parity and trained B=1,000 tie-gate benchmarks; never run by default.
- `tests/lmi_laion10m_stream.py`: bounded HDF5-to-REST importer for a later authorized remote run.

Four focused spherical tests passed. The filtered LMI unit run passed 20 tests, with 13 ignored explicit benchmarks. The five historical LMI integration binaries passed 2, 1, 10, 12, and 14 tests respectively (39 total). The ignored full LAION ~100k harness and the trained B=1,000 gate each passed once. The importer dry-run and isolated mock-endpoint test passed. `cargo check` passed for segment training, collection tests with segment training, edge, default server, and training-enabled server. Targeted `cargo clippy` passed in advisory mode. Strict `-D warnings` stopped on five pre-existing warnings in HNSW dispatch, old LMI training, and routing outside this change; no unrelated fixes were made. Targeted rustfmt and `git diff --check` passed.

Recommended next S.3C substep: measure spherical teacher-label distribution over the full LAION corpus, then revisit build admission and permit-aware parallelism only with measured CPU/RAM budgets. Extend the trained Torch/native ambiguity gate to B≈3,162 before promoting it as a scale result. After an explicitly authorized remote storage path is available, stage and checksum the official 10M source there and validate the streaming ingestion path before any full build.


## Reproduction commands and final state

Run from the repository root. The Torch environment path is the one used for this measurement; substitute the equivalent installed environment on another host.

```bash
export LIBTORCH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch
export LD_LIBRARY_PATH=$LIBTORCH/lib
cargo test -p segment --features lmi-training --locked --lib spherical_kmeans::tests
cargo test -p segment --features lmi-training --locked lmi
for t in lmi_candidate_scoring lmi_dummy lmi_phase_c lmi_phase_d lmi_phase_e; do
  cargo test -p segment --features lmi-training --locked --test $t
done
cargo check -p segment --features lmi-training --locked
cargo check -p collection --tests --features segment/lmi-training --locked
cargo check -p edge --locked
cargo check --bin qdrant --locked
cargo check --bin qdrant --features segment/lmi-training --locked
cargo clippy -p segment --features lmi-training --locked --lib
rustfmt --edition 2024 --config skip_children=true lib/segment/src/index/lmi_index/spherical_kmeans.rs lib/segment/src/index/lmi_index/evaluation_s3c.rs
git diff --check
```

The optimized benchmark is an ignored test. Set `LMI_S3C_SAMPLE`, `LMI_S3C_BUCKETS`, `LMI_S3C_ITERATIONS`, and a new non-existing `LMI_S3C_MICRO_OUTPUT`; set `LMI_S3C_NATIVE_ONLY=1` to avoid the scalar run at the larger scale. For the LAION harness, point `LMI_PHASE_F_DIR` to a new directory containing links to the preserved `dataset.json`, `corpus.f32`, and `queries.f32` inputs, then run `cargo test --profile perf -p segment --lib --features lmi-training --locked phase_f_benchmark -- --ignored --nocapture` and the existing `tests/lmi_phase_f_analyze.py` script. For the trained large-B gate, set `LMI_S3C_CORPUS` to the preserved corpus f32 path and `LMI_S3C_TIE_OUTPUT` to a new JSON path, then run the ignored `phase_s3c_trained_large_b_tie_gate` test. Raw result paths under `work/phase_s3/s3c/` must remain untracked and untouched by routine source commits.

Final HEAD remains `f7b613262`; this checkpoint is uncommitted. `git diff --stat` for tracked edits alone reports three files, 26 insertions, four deletions; new source, documentation, and importer files appear as untracked files and therefore are absent from that Git statistic. No push occurred.
