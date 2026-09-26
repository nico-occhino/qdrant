# Phase F — controlled LAION evaluation

Recorded 2026-09-26, branch `thesis/lmi-integration`, base `a1987b504`. Packaging preserved the completed baseline without rerunning or tuning.

This report records one controlled operating-point study, not a general claim of LMI superiority. The integrated lifecycle is unchanged. All retrieval ground truth and candidate scoring use Qdrant.

## Dataset and experimental scope

Source: `/home/nicoo/work/LearnedMetricIndex/laion2B-en-clip768v2-n=100K.h5`; SHA-256 `16328af59fc2157c713f55a1f4531aa6f614bedc5ab6141d0dab52f499456d50`. Original 100,000 × 768 float16 `emb` array, converted to little-endian float32 by dataset orchestration only. Indexed corpus: 99780; 20 separate warmup queries; 200 held-out measured queries; 2 trials per configuration. Seed 20260926, NumPy PCG64 permutation. Exact source row identities are retained in dataset.json. Queries and corpus source rows are disjoint; no point self-match is included. Cosine metric, k=10.


LMI: 64 buckets, 2,048 sampled training vectors, hidden dimension 64, 30 epochs, batch 256, 20 maximum Lloyd iterations, seed 42. Bucket count allows a complete powers-of-two probe sweep through exact full coverage. This is a deliberately bounded first training budget, not a tuned architecture. HNSW: m=16, ef_construct=100, full_scan_threshold=0, max_indexing_threads=1; builder RNG seed 42. Both are built through SegmentBuilder and the production index constructors, rather than installing a hand-written MLP. Build scope includes target copying/publication/reopen and is not an HTTP optimizer scheduling measurement.

## Machine and timing protocol

Intel Core Ultra 9 285H under WSL2, 16 exposed CPUs, approximately 15 GiB RAM. The benchmark pins all its threads to CPU 0 before building/measuring. Torch intra/inter-op settings are one; RAYON_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS are one. Cargo release profile uses opt-level 3, fat LTO and codegen-units=1. Compilation uses a separate two-job budget and finishes before timings. This is a CPU-only experiment. Hardware/OS details and process RSS/high-water status are in build.json; RSS describes the whole benchmark holding several indexes, not per-index memory.

Each configuration runs warmup then the same held-out queries. Configuration order rotates across trials to reduce fixed-order cache bias. Per-query wall observations are retained; summary p50/p95/p99 pool the two trials (400 observations, only 200 distinct queries). Per-trial medians are also retained. p99 and small timing differences need more repetitions before strong claims. Human tables round to milliseconds; nanosecond clock output is not a precision guarantee.

The primary timing scope is **in-process index/harness search**, not HTTP end-to-end request latency. Plain/HNSW call VectorIndexRead::search, including the lightweight context setup. Centroid/affine/MLP component rows use a common candidate/scoring harness including query preprocessing, routing, candidate union/validity preparation, scorer creation and exact top-k scoring. These are comparable candidate-pipeline experiments, with a residual wrapper/telemetry boundary against native HNSW/Plain. The actual MLP VectorIndexRead call is additionally timed and must return exactly the same scores/offsets; it runs after the component call and thus has a warm-cache advantage. Its separate native_mlp_index timings must not be presented as a fair independent latency winner over the controls. Probe sweeps replace only the in-memory routing budget in test code, outside timed loops, keeping the saved model and postings fixed. No production per-query nprobe API or persisted-state mutation is introduced. No network or full-server latency conclusion follows from these data.

Verbose existing LMI/HNSW stderr markers are compiled out only in the unit-test build. Normal server logging and query semantics remain unchanged. Stage clocks/capture exist only under cfg(test), with no production query clocks.

## Ground truth and execution guards

Exact Plain scoring over the identical current Cosine corpus supplies Recall@10 ground truth. Results are compared using external point IDs, avoiding assumptions about segment-local offsets. Plain's separate measured sweep supplies the latency baseline. The HNSW enum, nonzero m, zero scan threshold and an exact unfiltered_hnsw telemetry-count increase prove graph dispatch for every query. MLP loads a validated persisted trained state; nearest dense queries with no filter/params/quantization enter its learned branch, and the native result must match the explicitly routed candidate pipeline. Baselines never use classifier accuracy as recall.

The dataset is immutable and has no deletions/deferred points. Candidate preparation checks tracker/vector deletion and bounds, so reported candidate counts equal eligible scored postings. HNSW visited/scored candidate counts are not exposed by this seam: candidate work and component timings are null for HNSW. ef is a search control, not a candidate-count estimate.

## Centroid and affine controls

The production Lloyd implementation now has a center-returning helper; its existing labels-only wrapper remains behaviorally unchanged. Evaluation replays clustering on the exact saved LMI sample offsets and gets the teacher's final f64 centroids. Centroid corpus postings are assigned by those centroids, independently of MLP postings. Affine and direct centroid queries share those centroid postings.

For each center c, the affine coefficients are precomputed as weights 2c and bias -||c||². Direct -||q-c||² and affine logits differ only by query-constant -||q||², so their bucket order should agree. Deterministic index-based tie breaking is used. The equivalence regression includes duplicate centroids; all held-out queries also assert identical full bucket order. Controls use f64 teacher arithmetic; native MLP uses its normal f32 parameters/scoring. This is an evaluation-only control, with no production REST/index variant or persisted-format change.

## Construction, persistence and reopen

| Measurement | Observed |
| --- | ---: |
| ingest_seconds | 0.278212424 s |
| lmi_build_seconds | 5.268562424 s |
| clustering_seconds | 0.431814291 s |
| mlp_training_export_seconds | 0.334407810 s |
| mlp_posting_seconds | 4.214627400 s |
| control_clustering_seconds | 0.519048699 s |
| centroid_posting_seconds | 2.387684266 s |
| hnsw_build_seconds | 48.610999886 s |
| lmi_reopen_seconds | 0.030136251 s |
| hnsw_reopen_seconds | 0.010793794 s |
| lmi_index_bytes | 1221407 bytes |
| hnsw_index_bytes | 3857946 bytes |
| centroid_index_bytes | 1642890 bytes |

MLP stage includes optimizer setup, training, export and parity validation; it is not pure tensor-step time. Candidate construction includes classifying the eligible corpus. Replayed control clustering uses the same data/algorithm but is timed separately from LMI construction. Control construction excludes database target copying/publication overhead, so its build timing is not a directly equivalent production-index total. Index sizes exclude authoritative vector storage; centroid JSON is an evaluation artifact shared by direct/affine, not an integrated production format. Reopen is warm-cache wall time including the segment, not isolated cold index I/O. No reliable per-index measured RAM estimate is claimed.

# Phase F measured operating points

| Method | Effort | Recall@10 | Candidates mean | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| affine | 1 | 0.5920 | 2493 | 0.89 | 1.43 | 1.55 |
| affine | 2 | 0.7815 | 5050 | 1.93 | 2.83 | 3.16 |
| affine | 4 | 0.8975 | 10197 | 3.51 | 4.96 | 5.48 |
| affine | 8 | 0.9610 | 20317 | 6.36 | 7.74 | 9.06 |
| affine | 16 | 0.9925 | 39228 | 11.20 | 14.38 | 15.45 |
| affine | 32 | 0.9980 | 70615 | 17.17 | 18.92 | 19.68 |
| affine | 64 | 1.0000 | 99780 | 19.99 | 23.75 | 25.65 |
| centroid | 1 | 0.5920 | 2493 | 0.91 | 1.48 | 1.62 |
| centroid | 2 | 0.7815 | 5050 | 1.86 | 2.66 | 2.92 |
| centroid | 4 | 0.8975 | 10197 | 3.53 | 4.73 | 5.52 |
| centroid | 8 | 0.9610 | 20317 | 6.32 | 7.93 | 8.50 |
| centroid | 16 | 0.9925 | 39228 | 11.37 | 13.57 | 14.60 |
| centroid | 32 | 0.9980 | 70615 | 17.11 | 18.48 | 19.46 |
| centroid | 64 | 1.0000 | 99780 | 20.23 | 22.72 | 24.93 |
| hnsw | 16 | 0.9000 | unavailable | 0.16 | 0.28 | 0.41 |
| hnsw | 32 | 0.9620 | unavailable | 0.20 | 0.29 | 0.35 |
| hnsw | 64 | 0.9850 | unavailable | 0.32 | 0.45 | 0.54 |
| hnsw | 128 | 0.9925 | unavailable | 0.53 | 0.74 | 0.89 |
| hnsw | 256 | 0.9955 | unavailable | 0.93 | 1.30 | 1.53 |
| hnsw | 512 | 0.9970 | unavailable | 1.71 | 2.44 | 2.80 |
| mlp | 1 | 0.5990 | 3994 | 1.47 | 2.74 | 3.49 |
| mlp | 2 | 0.7690 | 7632 | 2.84 | 4.28 | 5.06 |
| mlp | 4 | 0.9015 | 14504 | 4.91 | 6.50 | 7.05 |
| mlp | 8 | 0.9650 | 26468 | 8.16 | 10.67 | 11.96 |
| mlp | 16 | 0.9910 | 44629 | 12.94 | 15.14 | 18.22 |
| mlp | 32 | 0.9995 | 70047 | 17.25 | 19.34 | 21.31 |
| mlp | 64 | 1.0000 | 99780 | 20.32 | 24.11 | 27.12 |
| plain | 0 | 1.0000 | 99780 | 21.35 | 24.47 | 25.88 |

## Approximately matched recall

Only measured operating points within ±0.02 absolute recall are admitted; no interpolation. Select closest recall, breaking ties by median latency. Target 1.0 requires measured recall exactly 1.0. Actual recall must still be compared; absence is not evidence of inability.

| Target | Method | Effort | Actual recall | p50 ms |
| --- | --- | ---: | ---: | ---: |
| 0.50 | plain | — | no point in band | — |
| 0.50 | hnsw | — | no point in band | — |
| 0.50 | centroid | — | no point in band | — |
| 0.50 | affine | — | no point in band | — |
| 0.50 | mlp | — | no point in band | — |
| 0.80 | plain | — | no point in band | — |
| 0.80 | hnsw | — | no point in band | — |
| 0.80 | centroid | 2 | 0.7815 | 1.86 |
| 0.80 | affine | 2 | 0.7815 | 1.93 |
| 0.80 | mlp | — | no point in band | — |
| 0.90 | plain | — | no point in band | — |
| 0.90 | hnsw | 16 | 0.9000 | 0.16 |
| 0.90 | centroid | 4 | 0.8975 | 3.53 |
| 0.90 | affine | 4 | 0.8975 | 3.51 |
| 0.90 | mlp | 4 | 0.9015 | 4.91 |
| 0.95 | plain | — | no point in band | — |
| 0.95 | hnsw | 32 | 0.9620 | 0.20 |
| 0.95 | centroid | 8 | 0.9610 | 6.32 |
| 0.95 | affine | 8 | 0.9610 | 6.36 |
| 0.95 | mlp | 8 | 0.9650 | 8.16 |
| 0.99 | plain | 0 | 1.0000 | 21.35 |
| 0.99 | hnsw | 128 | 0.9925 | 0.53 |
| 0.99 | centroid | 16 | 0.9925 | 11.37 |
| 0.99 | affine | 16 | 0.9925 | 11.20 |
| 0.99 | mlp | 16 | 0.9910 | 12.94 |
| 1.00 | plain | 0 | 1.0000 | 21.35 |
| 1.00 | hnsw | — | no point in band | — |
| 1.00 | centroid | 64 | 1.0000 | 20.23 |
| 1.00 | affine | 64 | 1.0000 | 19.99 |
| 1.00 | mlp | 64 | 1.0000 | 20.32 |

## Teacher diagnostics

Held-out query teacher top-1 agreement (including warmup): 0.6364.
Direct/affine query ordering mismatches: 0.
Teacher agreement and coverage are diagnostics, not retrieval recall.

mlp: 24 empty buckets; min/median/max sizes 0/372.0/7361; largest fraction 0.0738.
centroid: 0 empty buckets; min/median/max sizes 1/1433.0/4501; largest fraction 0.0451.


## Interpretation and confounders

For this split/configuration, MLP-LMI exhibits ANN behavior but does not establish added efficiency over its simpler teacher control. Around 90% recall, MLP nprobe=4 yields 0.9015 recall, 14,504 mean candidates and 4.91 ms median; centroid nprobe=4 yields 0.8975, 10,197 candidates and 3.53 ms (affine: identical recall/work, 3.51 ms). Around 96%, MLP uses 26,468 candidates at 0.9650 recall versus centroid's 20,317 at 0.9610. These near-matched points show more candidate work for the tested MLP, not a benefit from its nonlinear architecture. At almost-full recall the differences become small; this is not a claim of dominance at every possible operating point.

HNSW recorded lower latency in this experiment: ef=16 gives 0.9000 recall at 0.16 ms, and ef=32 gives 0.9620 at 0.20 ms. Corresponding MLP native index-call medians were 5.22 ms (nprobe=4) and 8.64 ms (nprobe=8), in addition to the component-harness medians above. The native call's warm-cache caveat still applies, and these are not HTTP results. LMI's observed tradeoff was construction/storage: one build took 5.27 s versus HNSW's 48.61 s, with 1,221,407 versus 3,857,946 persisted index bytes, excluding the common vector corpus. Repeated independent builds are needed to generalize those costs.

The MLP had 24 empty corpus buckets out of 64 and 63.64% top-1 teacher agreement on held-out/warmup queries; centroid routing had no empty buckets. This suggests assignment imbalance/teacher-fit quality deserves investigation, but agreement itself is not Recall@10 and is not a proven causal explanation. At nprobe=4 the MLP median router time was about 0.06 ms versus 4.64 ms for exact candidate scoring: candidate work, rather than just neural forward time, accounts for most observed search time. Median components do not add exactly to the median total.

The tables provide the MLP-vs-centroid/affine and HNSW-vs-LMI comparisons at actual achieved recall, rather than assuming fixed nprobe and ef have equivalent effort. Any absence from a matched-recall band is an unsampled region, not a proof that the method cannot reach that target. Full-probe LMI/centroid rows also serve as closure checks against exact ground truth.

Only one dataset split, one model-training seed, one bucket count/training budget, and one HNSW build are evaluated. Repeated query trials do not establish training-seed robustness. The scalar native router/control implementations are not separately kernel-tuned. Host scheduling, turbo/thermal behavior, cache order and the post-component native MLP measurement can affect latency. Teacher agreement, top-p teacher coverage and bucket imbalance are diagnostics for investigating behavior; they do not establish retrieval quality or causality. HNSW work is unobserved, so recall-vs-candidate-work plots intentionally omit HNSW. No automatic planner, GPU, DLI/CLI, continual learning, or architecture search was added.

## Reproduction

From repository root, using the same Torch environment as Phase E:

```bash
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH
export CXX=clang++ CXXFLAGS=-g0 CARGO_BUILD_JOBS=2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LMI_PHASE_F_DIR=/home/nicoo/work/lmi-phase-f-data
python tests/lmi_phase_f_prepare.py --output "$LMI_PHASE_F_DIR"
cargo test --release -p segment --features lmi-training --locked --lib phase_f_benchmark -- --ignored --nocapture --test-threads=1
cargo test -p segment --features lmi-training --locked --lib centroid_affine_equivalence
PYTHONPATH=/home/nicoo/work/lmi-phase-f-python python tests/lmi_phase_f_analyze.py /home/nicoo/work/lmi-phase-f-data
```

Preparation needs h5py/NumPy; analysis uses matplotlib 3.10.8 installed only in an isolated benchmark dependency directory. HDF5/Python are not operational Qdrant dependencies. The harness invokes taskset internally and currently requires Linux CPU 0 to be available; adapt the pin deliberately on another machine and record it. Use separate output directories when changing dataset/split settings.

## File changes and scope

- `lmi_index/evaluation.rs`: ignored, opt-in unit-test benchmark and centroid/affine regression.
- `lmi_index/training.rs`: shared final-centroid return helper and cfg(test) clustering/training stage capture.
- `lmi_index/build.rs`: cfg(test) corpus-classification stage capture.
- `lmi_index/mod.rs`: test/training-only evaluation module.
- `lmi_index/read.rs`, `hnsw/read_view/dispatch.rs`: omit legacy stderr diagnostics only in unit-test builds.
- `tests/lmi_phase_f_prepare.py`, `tests/lmi_phase_f_analyze.py`: dataset split/export, raw aggregation, plots and matched-recall tables.
- `docs/lmi-phase-f-audit.md`, `docs/lmi-phase-f.md`: measurement audit and this report.

No persisted-format, API, lockfile, operational HDF5 or training architecture change. No commit or push.

Tracked git diff --stat (new files listed above are not included by Git until staged):

```text
 .../index/hnsw_index/hnsw/read_view/dispatch.rs    | 12 ++++++++++
 lib/segment/src/index/lmi_index/build.rs           |  9 +++++++
 lib/segment/src/index/lmi_index/mod.rs             |  3 +++
 lib/segment/src/index/lmi_index/read.rs            |  3 +++
 lib/segment/src/index/lmi_index/training.rs        | 28 +++++++++++++++++++---
 5 files changed, 52 insertions(+), 3 deletions(-)
```

## Final environment and preservation record

Rust 1.98.0 (88d9e12ae, 2026-08-18); Ubuntu clang 21.1.8; Python 3.11.16; operational environment NumPy 2.2.4, h5py 3.13.0, Torch 2.5.1+cpu and tch 0.18.1. WSL kernel: 6.18.33.2-microsoft-standard-WSL2. Isolated plotting dependencies: matplotlib 3.10.8 and NumPy 2.4.6. Install only the analysis dependencies into a separate directory if reproducing:

```bash
/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin/python -m pip install --target /home/nicoo/work/lmi-phase-f-python matplotlib==3.10.8 numpy==2.4.6
```

The recorded benchmark command (see verification evidence) used an outer `taskset -c 0`; compilation affinity was relaxed to allow the two-job build. The harness independently pinned every benchmark thread to CPU 0 before measurement. Initial release compilation took 15m35s; benchmark execution took 190.72s. Neither compilation nor orchestration wall time is an index build metric.

The archive preserves all 11,200 per-query/per-trial observations, 28 operating-point aggregates, saved MLP and centroid states, source-row manifest, diagnostics and verification logs. `SHA256SUMS.json` permits archive integrity checks; `baseline_preservation.json` records identical before/after hashes for raw observations and original state/metadata. The large source dataset and exported float32 corpus are not duplicated in the bundle; reconstruct them using the source hash, recorded split and preparation script. Preparation/benchmark commands are for future reproduction in a separate directory: do not overwrite this baseline.

## Detailed diagnostic distributions

`candidate_distributions.csv/json` records min, p25, p50, p75, p95, p99, max, mean and mean corpus fraction per operating point. It pools the same 400 observations (200 distinct queries); HNSW entries remain null. Full per-query counts remain in queries.jsonl. `bucket_distributions.csv` and `router_diagnostics.json` retain all 64 bucket counts and distribution summaries.

| Router | Active / empty | Min | p25 | Median | p75 | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp | 40 / 24 | 0 | 0.00 | 372.00 | 3156.25 | 5675.75 | 7361 |
| centroid | 64 / 0 | 1 | 389.50 | 1433.00 | 2475.50 | 3702.95 | 4501 |

Teacher top-p coverage (220 held-out/warmup queries; diagnostic, not Recall@10):

| p | Coverage |
| --- | ---: |
| 1 | 0.636364 |
| 2 | 0.763636 |
| 4 | 0.850000 |
| 8 | 0.927273 |
| 16 | 0.977273 |
| 32 | 0.990909 |
| 64 | 1.000000 |

Whole-process memory at the captured snapshot: VmPeak=2,761,520 KiB, VmHWM=1,730,400 KiB, VmRSS=1,730,400 KiB. These Linux /proc kB values are binary KiB; they include multiple indexes, mappings and shared pages, so they support no per-index memory ranking.

LMI measured stages total 4.980849501 s; the remaining 0.287712923 s is uninstrumented construction/copy/publication/open overhead and timer-boundary residual, not another training phase. Decimal digits preserve the recorded clock values, not an assertion of equivalent physical accuracy.

## Figures

![Recall versus median latency](../../lmi-phase-f-data/recall_vs_latency.png)

![Recall versus candidate fraction](../../lmi-phase-f-data/recall_vs_candidate_fraction.png)

Centroid and affine candidate curves overlap exactly; HNSW is intentionally absent from the candidate plot. Lines connect sampled operating points for readability, not interpolated guarantees.

## Existing lifecycle verification

All checks below completed successfully in the existing Phase F run; packaging did not rerun the experiment or tests. The 36 existing integration tests passed (candidate scoring 2, dummy 1, Phase C 10, Phase D 12, Phase E/E2 11). Five selected prior unit tests passed (clustering/cancellation 1, collection configuration 3, gRPC 1). The new centroid/affine equivalence unit passed, and the release benchmark passed its correctness assertions. There were zero test failures in this final verification matrix.

The two isolated HTTP drivers also passed: conflicting PATCH rejection with unchanged configuration; restart without training; fresh-storage snapshot restore with identical state, IDs and scores, explicit StaticLearned open/search and documented exact/filter fallbacks; deferred promotion with global HNSW m=0, immutable/retired old state, rebuilt postings, updates/deletions, mixed named indexes and restart. `named_indexes_independent=false` in the single-index case is not a failure: that case has no mixed named indexes; the mixed case verifies true. Native/universal corruption rejection is included in the integration suite. JSON results and server logs are bundled.

| Check | Result | Recorded wall seconds |
| --- | --- | ---: |
| final-cluster-unit | PASS | 31.46 |
| final-collection-check | PASS | 12.98 |
| final-collection-unit | PASS | 33.63 |
| final-default-check | PASS | 11.85 |
| final-diff-check | PASS | 0.03 |
| final-edge-check | PASS | 4.62 |
| final-equivalence | PASS | 0.72 |
| final-fmt | PASS | 3.17 |
| final-grpc-unit | PASS | 12.41 |
| final-lifecycle | PASS | 5.45 |
| final-lmi-tests | PASS | 16.66 |
| final-segment-check | PASS | 8.38 |
| final-snapshot | PASS | 4.74 |
| final-training-build | PASS | 44.59 |
| release-benchmark | PASS | 1126.11 |

Exact recorded verification commands (run from repository root with the environment above):

```bash
# final-cluster-unit
cargo test -p segment --features lmi-training --locked --lib lloyd_separates_clusters_and_obeys_cancellation -- --nocapture

# final-collection-check
cargo check -p collection --tests --features segment/lmi-training --locked

# final-collection-unit
cargo test -p collection --features segment/lmi-training --locked --lib lmi_config_ -- --nocapture

# final-default-check
cargo check --bin qdrant --locked

# final-diff-check
git diff --check

# final-edge-check
cargo check -p edge --locked

# final-equivalence
cargo test -p segment --features lmi-training --locked --lib centroid_affine_equivalence -- --nocapture

# final-fmt
cargo fmt --all

# final-grpc-unit
cargo test -p api --locked --lib lmi_configuration_defaults_roundtrip_and_validation -- --nocapture

# final-lifecycle
python3 tests/lmi_phase_e2_lifecycle.py --port 16733 --output /mnt/c/Users/nicoo/Documents/Codex/2026-09-21/referenced-chatgpt-conversation-this-is-an/work/phase_f/final-lifecycle

# final-lmi-tests
cargo test -p segment --features lmi-training --locked --test lmi_candidate_scoring --test lmi_dummy --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e -- --nocapture

# final-segment-check
cargo check -p segment --tests --features lmi-training --locked

# final-snapshot
python3 tests/lmi_phase_e_http_smoke.py --port 16633 --output /mnt/c/Users/nicoo/Documents/Codex/2026-09-21/referenced-chatgpt-conversation-this-is-an/work/phase_f/final-snapshot

# final-training-build
cargo build --bin qdrant --features lmi-training --locked

# release-benchmark
taskset -c 0 cargo test --release -p segment --features lmi-training --locked --lib phase_f_benchmark -- --ignored --nocapture --test-threads=1
```

Stable rustfmt emitted the known nightly-option warnings. One unrelated pre-existing formatter change was restored; final git diff --check passed. No Cargo.lock, dependency, API or persistence-format changes are part of Phase F.

## Recommended Phase F.2 — explain MLP versus centroid behavior

Keep this baseline immutable. First trace corpus and query teacher/MLP assignments and top-p coverage, quantify which buckets contribute candidate inflation and missed exact neighbours, and compare occupancy/confusion on the sampled training set versus held-out vectors. Treat the 24/64 empty MLP buckets as an observation to explain, not established causality. Separate teacher approximation errors from effects of changing corpus partitions and query routing.

Then predefine a bounded, one-factor-at-a-time study of training adequacy (budget/sample coverage and multiple seeds), holding centroid teacher, corpus, queries, metric and scorer fixed where the question requires it. Retain the centroid/affine controls and record actual candidate work at matched recall. Use a separate validation split for tuning and reserve an untouched test split for final claims. Record training curves, assignment distributions, Recall@10 and latency together; do not select models only by teacher accuracy. Any architecture change should follow evidence from this diagnosis, not precede it. Repeat independent builds and query trials when estimating uncertainty; CPU/cache conditions and measurement wrappers must remain explicit.

Phase F.2 is a recommendation only: no tuning study, new model, parameter change or repeat baseline experiment was performed in this packaging pass. The tested MLP configuration does not demonstrate candidate-efficiency or latency advantage over centroid routing. HNSW's substantially lower measured in-process latency and LMI's observed build/storage tradeoff apply to this dataset/configuration only; none establishes universal superiority or inferiority.
