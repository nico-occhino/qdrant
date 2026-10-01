# Phase S.3A: native build routing and explicit resource planning

## Scope and starting checkpoint

Started from `f89189702` on `thesis/lmi-integration`, a clean working tree. Unlike the
previous handoff, the F.2-F.4 work is now committed. It includes historical `work/`
files; new S.3 raw outputs remain untracked. Both S.1 `6f7085b9c` and S.2 `02e8fda85`
are ancestors of this checkpoint. A live HTTPS `git ls-remote` confirmed GitHub also
pointed to `f8918970226fc1221181e7076012da707325d01b`. SSH authentication failed;
remote state was verified through HTTPS instead. No push is part of S.3A.

S.1/S.2 revalidation confirmed contiguous offset postings, v2 separate files, legacy
v1 reader, streamed reservoir sampling, count/prefix/fill, native validity semantics,
universal open and three-file snapshot enumeration. No corpus vector copies are
added to the physical index. This phase does not change loss, partition semantics,
database lifecycle, persistence format or production configuration defaults.

## Pre-optimization hot path

`build.rs` calls `top_buckets_with_stop(row, 1)` once for each eligible vector in each
of the two posting passes. `output_dim()` validates all layer shapes and finite
weights; `forward_with_stop()` validates again. Forward copies the input into a Vec,
allocates one output Vec per linear layer, and top-buckets allocates all B bucket IDs
then uses a stable full sort. For the usual two-layer MLP this means at least four
Vec allocations (input, hidden, logits, IDs), plus any stable-sort workspace.
Temporary values occupy O(d+H+B), model validation O(P), forward O(P), selection
O(B log B), where P=dH+HB+H+B. Both corpus passes repeat all this work.

There is no explicit SIMD, BLAS or parallel inference kernel in this router: it uses
ordered scalar f32 multiply/add loops, with cancellation checks per layer/output row.
Compiler vectorization was not established by assembly inspection. Qdrant's existing
distance kernels do have SIMD dispatch, but replacing the router's ordered sum with
those kernels could change rounding and partition assignments. They are not substituted
in a parity-preserving implementation timing comparison.

## Baseline protocol

The Windows project directory contained SIFT1M: 1,000,000 vectors, d=128, Euclidean
distance. Its existing fvecs file was streamed into little-endian f32 rows, preserving
order. Source SHA-256: `21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816`;
export SHA-256: `ac8e11ed111b09e882bcf0ef1a77ebdd36c89f495bce8fd596603f69c35148e6`.
WSL reported 879GiB disk available and about 12GiB RAM available before preparation.
No download or synthetic corpus was created. The user explicitly approved this
dataset and Qdrant's existing `perf` profile. The initial fat-LTO release compilation
was stopped before producing measurements; no release timings enter the comparison.

Configuration: B=64, sample=2,048, H=64, epochs=30, K=256, Lloyd iterations=20,
nprobe=4, seed=42. Two independent `perf` processes per implementation, CPU affinity
0, Torch intra/inter-op threads 1, same source/configuration and builder seed.
`perf` uses opt-level=3, no LTO and 256 codegen units. This is a build benchmark,
not an F.5/HNSW/retrieval-quality study. No statistical significance is claimed.
The preserved LAION corpus (99,780 vectors, d=768, cosine) is a supplementary
one-build-per-version parity check, not a substitute for the primary SIFT1M run.

The opt-in `evaluation_s3.rs` streams f32 input into a native Plain segment, then uses
normal SegmentBuilder update/build with LmiTrained. Whole-build timing includes native
segment copying, LMI creation, persistence and builder reopen; ingestion is separate.
Existing test-only training timers split clustering from MLP training/export. Existing
S.2 log timers record sampling, posting count/allocation/fill, persistence and native
open. An additional segment reopen is measured with warm OS caches. All three state
files are retained for bytewise comparisons; one learned query's IDs/scores are saved.
This correctness sample is not a Recall@10 estimate.

`/usr/bin/time -v` measures the test process (not Cargo compilation). Peak RSS includes
LibTorch, source/destination segment storage, ingestion and all phases; it is not
isolated LMI workspace RSS. Two observations are insufficient for strong statistical
claims about small differences. Raw logs and both trials must remain available.

## Dedicated top-1 inference

`BuildRouter` borrows the router immutably, validates once, and owns two reusable scratch
buffers sized to the largest layer/input width. Immutable borrowing prevents weight or
shape mutation while the validated view exists. Each row is checked for dimension and
finite input; intermediate non-finite values remain errors before ReLU. Dot-product
accumulation order and cancellation checks are preserved. Strict greater-than argmax
selects the first bucket on ties, including signed zero. Selection is O(B), no sorting
or per-row heap allocation. This view is reused across both posting passes.

The normal top-nprobe query API and transient helper remain unchanged. Focused tests
compare old/new top-1 across random vectors, four metric preprocessing modes, multiple
bucket counts, tied logits, invalid models/inputs, nonfinite arithmetic and cancellation.
Scratch capacities remain constant across repeated predictions. Full persisted model
and posting byte equality is checked by the build benchmark comparison.

## Native clustering audit and S.3B direction

Current `cluster_with_centers` is standard squared-Euclidean Lloyd for every metric.
It initializes from distinct sample positions after a seeded shuffle (not k-means++).
Input samples are f32; centroids, coordinate differences, distances and sums are f64.
Assignments use a scalar loop over all centroids/dimensions, with smaller cluster IDs
winning ties. Empty clusters retain their old centroid. Iteration stops when labels
stabilize or the configured limit is reached, then final labels are recomputed.

No part of assignment/reduction is explicitly parallelized. Storage is O(Sd) sample,
O(S) labels/shuffle order and O(Bd) centers/sums; there is no S-by-B distance matrix.
Rows already stream through centroid comparisons. Work is approximately O(I*S*B*d),
plus a final assignment. Chunking alone does not reduce this arithmetic count.

**Cosine clustering is not spherical KMeans.** Qdrant normalizes the input vectors,
but Lloyd does not normalize centroid means. Euclidean assignment then includes a
centroid-norm term; spherical assignment instead compares directions/unit centroids.
The cached Coda SISAP24 `task1.py` explicitly uses `spherical=True`. Normalized inputs
alone do not establish equivalence. No centroid normalization change is made here.

At S=1M, B=10,000, d=768 and I=20, assignment alone implies roughly
153.6 trillion coordinate contributions plus final assignment; CPU scalar execution,
sample copies, batch logits, training and full-corpus inference remain substantial.
An allocation guard change cannot make this computation practical.

Proposed S.3B: a separately controlled native spherical Lloyd reference for cosine,
with explicit normalization/zero-centroid/empty-cluster rules and deterministic tests.
Compare its partition quality to the unchanged Euclidean reference before claiming an
improvement. Then evaluate existing Rust SIMD dot kernels and bounded parallel
assignment with per-worker centroid sums and deterministic reduction. Budget worker
count by B*d memory as well as CPU permits. Compare a native mini-batch spherical
variant only if measured full-Lloyd cost remains prohibitive; improved initialization
is a separate ablation, not a simultaneous change. No FAISS or external clustering
service becomes a production dependency. Existing tch/LibTorch CPU MLP training is
unchanged; native routing/clustering does not mean the current trainer is pure Rust.

## Preserved SIFT1M observations

Both trials below use the identical frozen configuration. Seconds are elapsed wall time.

| Measurement | Before 1 | Before 2 | Top-1 1 | Top-1 2 |
|---|---:|---:|---:|---:|
| sampling_seconds | 0.006516 | 0.006091 | 0.006084 | 0.008946 |
| clustering_seconds | 0.092666 | 0.093125 | 0.093710 | 0.092358 |
| training_export_seconds | 0.244492 | 0.124480 | 0.184312 | 0.131274 |
| count_seconds | 8.764821 | 8.598576 | 3.305917 | 3.391863 |
| allocation_seconds | 0.000260 | 0.000349 | 0.000239 | 0.000241 |
| fill_seconds | 8.524151 | 8.415021 | 3.296540 | 3.265785 |
| persistence_seconds | 0.002034 | 0.001985 | 0.002147 | 0.001686 |
| native_open_first_seconds | 0.004330 | 0.006372 | 0.004536 | 0.004242 |
| total_build_seconds | 18.427061 | 18.091070 | 7.735628 | 7.656738 |
| reopen_seconds | 0.040521 | 0.036303 | 0.035342 | 0.037421 |
| peak_rss_kib | 1905568 | 1905292 | 1905076 | 1905980 |

Descriptive ratios of medians (two trials, no significance claim):

- count_seconds: 2.592x.
- fill_seconds: 2.581x.
- total_build_seconds: 2.372x.
- Combined count/fill: 2.587x.

After top-1, count/fill remain 86.1% of whole-build elapsed time.
Bounded native batching is deferred as an independently measured next step. Existing ndarray and Qdrant SIMD kernels were inspected; using them requires an explicit numerical-parity decision, bounded packing/logit buffers and separate measurements. No new linear algebra dependency is added.

All four runs have identical metadata/router/posting hashes and the same learned query IDs/scores. This proves partition preservation for this corpus/configuration, not retrieval-quality improvement.

| File | Bytes | SHA-256 |
|---|---:|---|
| lmi_state.json | 207 | `de3512c2af57010f7df38b5ee85fa38b7f607df186019ae5a2a3dbf4d85dd278` |
| lmi_router.bin | 57949 | `248acbc78357c01cb0ee454e0d8d8246d95f7d209037494b725d6b8d24e43ac1` |
| lmi_postings.bin | 4000536 | `4647b615e6d2f1797a52fb14d4fc85685b759e2bcd5dad23c8a4b3e7540d060a` |

Total auxiliary LMI size: **4,058,692 bytes**.

Focused top-1 parity tests: 2 passed, zero failures. The older F-series studies were not executed.

## LAION768 supplementary regression

One build of each implementation reused `/home/nicoo/work/lmi-phase-f-data`, N=99,780,
d=768, cosine, with the same B/S/H/epochs/seed and the same `perf` profile as each
other. No original Phase F artifact was overwritten. All metadata/router/posting
bytes and the learned query IDs/scores match exactly. Evidence is retained in
`work/phase_s3/laion-before`, `laion-top1` and `laion-parity.json`. This confirms
partition preservation at the thesis dimensionality; it is not a new retrieval study.

## Memory/work planner: policy C

`build_plan.rs` estimates components before allocating the reservoir/training matrix,
first using physical slots as a conservative bound, then using the observed eligible
count. It does not run at open/reopen. Every sum/product is checked; invalid dimensions
or unrepresentable plans fail clearly. The normal builder still validates configuration.
The implementation inspects `common::budget::ResourcePermit`, which provides CPU/IO
permits, not a RAM reservation. No suitable memory permit is passed into this builder.

**There is no new fixed 512 MiB aggregate limit.** Following the user's option C,
aggregate bytes and operation counts are reported, while the existing conservative
component rejection envelope is retained: configured sample matrix and model weights
are each limited to **128,000,000 bytes**, equivalent to the former 32M f32 elements.
These are inherited experimental ceilings, not inherent Qdrant/LMI limits. The old
inline guard is replaced by checked plan construction and component-budget checking;
its safety restriction has not been relaxed. Requested sample size is still checked
even when the actual segment is smaller, preserving prior admission behavior.

The planner does not reserve RAM, admit concurrent builders against host availability,
or reject an aggregate plan solely because it exceeds physical RAM. It is consequently
not a complete OOM prevention mechanism. Global/configurable memory admission remains
future work; this limitation is explicit instead of assigning an arbitrary total cap.

Let S=min(configured sample,N), W=(d+B)H, P=W+H+B and K=min(batch size,S).
All terms below are bytes; they are sizing estimates/allowances, not an allocation trace:

| Component | Formula |
|---|---:|
| Rust sample / Torch sample copy | 4Sd each |
| Reservoir and persisted sample offsets | 8S |
| Labels and shuffle/workspace allowance | 32S |
| f64 centroids / f64 accumulators | 8Bd each |
| Cluster sizes | 8B |
| Parameters, gradients, each Adam moment, native export, initialization workspace | 4P each (six terms) |
| Batch input plus gradient allowance | 8Kd |
| Hidden activations plus gradient allowance | 8KH |
| Logits/loss/gradient allowance | 16KB |
| Batch indices/labels | 16K |
| Posting counts, boundaries, cursors | 24B+8 |
| Final posting offsets | 4N |
| Encoded posting buffer allowance during open | 4N+8(B+1)+16 |
| Encoded model/sample allowance | 4P+4S+256 |
| Open validation bitmap | ceil(N/8), using physical slots in the pre-scan plan |
| Reusable inference scratch | 8 max(d,H,B) |

The reported explicit subtotal conservatively sums these components across phases,
although not all coexist. The separate **backend allowance = subtotal/4 + 32 MiB**
is a declared, uncalibrated engineering allowance for Torch/allocator workspace, not
a proven upper bound or acceptance threshold. Some activation terms themselves include
conservative gradient/loss allowances. Neither value includes Qdrant-owned corpus
storage, native segment copies, IDs/payloads or the process's baseline shared libraries.

Reported work: Lloyd upper iteration work (I+1)SBd; approximate training dense
forward/backward MACs 3*epochs*S*W; two corpus forward passes 2NW. These are operation
estimates, not wall-time predictions or runtime work quotas. The new path removes
sorting and repeated parameter validation, not the O(NW) network arithmetic.

## Reproduction

Host: WSL2 x86_64 on Intel Core Ultra 9 285H, 16 visible CPUs; benchmark pinned to CPU
0. Compiler details are in `work/phase_s3/rustc.txt` (rustc 1.98.0). Raw process RSS,
per-run logs/results, corpus hashes, prepared data and saved before/top-1 test binaries
are retained under `work/phase_s3/`, outside the commits. Compilation is excluded from
timings. The sample configuration is fixed; `LMI_S3_CONFIG` may specify a JSON config
only for separately named future experiments.

```bash
cd /home/nicoo/work/qdrant
export PATH="/home/nicoo/.cargo/bin:/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH"
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export CXX=clang++ CXXFLAGS=-g0 CARGO_BUILD_JOBS=2

# Existing prepared input: do not overwrite or prepare it again.
export LMI_S3_DATA="$PWD/work/phase_s3/sift1m"
# Choose a NEW output directory for any intentional future repeat.
export LMI_S3_RUN_ROOT="$PWD/work/phase_s3/reproduction-new"
bash tests/run_lmi_phase_s3.sh

# To reproduce the preserved BEFORE implementation without rebuilding:
# export LMI_S3_BINARY="$PWD/work/phase_s3/before-sift-test-binary"
# To reproduce TOP-1 only:
# export LMI_S3_BINARY="$PWD/work/phase_s3/top1-test-binary"
# Set a different NEW LMI_S3_RUN_ROOT and invoke the same runner.

# Preparation for a different existing fvecs input (outside the physical index):
# python3 tests/lmi_phase_s3_prepare.py SOURCE.fvecs work/NEW_DATA --metric Euclid
```

The measured runs used the same underlying invocation, twice per implementation:
`/usr/bin/time -v -o TIME_LOG taskset -c 0 SAVED_BINARY phase_s3_build_benchmark
--ignored --nocapture --test-threads=1`, with `LMI_S3_DATA`, `LMI_S3_OUTPUT` and
`RUST_LOG=segment::index::lmi_index::build=info`. Build command was
`cargo test -p segment --profile perf --features lmi-training --locked --lib
phase_s3_build_benchmark --no-run`. The original top-1 binary build instead selected
`build_router_tests` and ran those two tests before measuring; compilation profile
and resulting benchmark code/configuration are the same.

## Scale status and next steps

- **SIFT1M: EMPIRICALLY TESTED at N=1,000,000, d=128**, for this build configuration.
- **LAION: EMPIRICALLY TESTED at N=99,780, d=768**, supplementary parity regression.
- **10M: NOT YET TESTED.** Need ingestion/build/reopen/RSS measurements and a concurrent
  builder memory policy. No direct timing extrapolation from SIFT is warranted.
- **100M: NOT YET TESTED.** Larger training coverage, scalable native clustering,
  corpus inference throughput, mmap/bounded decode and hardware/IO validation remain.
- **1B: NOT YET TESTED.** All previous gates plus offset/segment/distributed lifecycle
  and multi-terabyte storage validation. Representable arithmetic is not feasibility.

The implementation is structurally scale-aware. It is not demonstrated at 100M/1B.
At this measured small-sample configuration, clustering is around 0.09 seconds while
optimized count/fill still consume 86% of whole-build time. The best next throughput
experiment is **bounded native batched corpus inference**, preserving this scalar
reference and checking partition parity separately. No batching is implemented here.
The native spherical-clustering design above remains a separate semantics-controlled
S.3B experiment for larger training samples/bucket counts. No FAISS, GPU, new loss,
Python runtime dependency, external service, new C++ dependency or production default
was introduced. Existing LibTorch CPU training remains unchanged.

## Observed planner estimates

These are arithmetic estimates only; the large sample/corpus cases were not built. K=256, epochs=30 and I=20 in every row. Large corpus rows assume S=1M and d=768.

| Case | N | d | S | B | H | Explicit subtotal, bytes | Backend allowance, bytes | Reported total, bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1000000 | 128 | 2048 | 64 | 64 | 11454312 | 36418010 | 47872322 |
| sample_1m | 1000000 | 768 | 1000000 | 64 | 64 | 6200385128 | 1583650714 | 7784035842 |
| corpus_100m | 100000000 | 768 | 1000000 | 10000 | 512 | 7322110208 | 1864081984 | 9186192192 |
| corpus_1b | 1000000000 | 768 | 1000000 | 31622 | 512 | 15300481320 | 3858674762 | 19159156082 |

SIFT1M planner subtotal: **10.92 MiB**; with the declared backend allowance: **45.65 MiB**. Observed whole-process peak RSS (two-run median): **1860.87 MiB**. These are not comparable scopes: RSS includes native corpus/segment storage and process libraries; the planner covers the listed LMI build components and deliberately sums some non-overlapping phases. The difference does not calibrate a trustworthy Torch overhead multiplier.

The 1M-by-768 sample and both large-corpus examples exceed the retained sample component ceiling. A hypothetical large corpus with a small permitted sample may still pass component checks even when aggregate RAM is impractical; option C deliberately reports this limitation. No configurable/global memory admission has been implemented.

## Final verification and files

All commands below ran with the environment given above. No ignored F-series study was rerun.

| Command | Result |
|---|---|
| `cargo test -p segment --profile perf --features lmi-training --locked --lib build_router_tests -- --nocapture` | 2 passed |
| `cargo test -p segment --profile perf --features lmi-training --locked --lib index::lmi_index -- --nocapture` | 12 passed, 0 failed, 5 ignored (F/F.2/F.3/F.4/S.3 opt-in benchmarks) |
| `cargo test -p segment --profile perf --features lmi-training --locked --test lmi_candidate_scoring --test lmi_dummy --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e -- --nocapture` | 39 passed, 0 failed |
| `cargo check --bin qdrant --locked` | PASS |
| `cargo check -p edge --locked` | PASS |
| `cargo clippy -p segment --lib --features lmi-training --locked` | PASS with existing warnings, not a -D warnings run |
| `rustfmt --check --edition 2024 --config skip_children=true lib/segment/src/index/lmi_index/{routing,build,build_plan,evaluation_s3,mod}.rs` | PASS |
| `git diff --check` | PASS |

Planner tests cover overflow, zero dimensions, tiny/default plans, both sides of the
previous sample/model boundary, a 1M x 768 sample, B=10,000 and B=31,622 estimates,
and explicit demonstration that aggregate reporting is not a host-memory admission
guarantee. Integration tests retain native/read-only reopen, v1/v2 corruption checks,
empty/tiny behavior, deletion validity and optimizer rebuild coverage.

The final planner was verified through these tests; the completed two-before/two-after
scale experiment was not repeated after adding reporting/admission refactoring. Its
timing result specifically isolates the top-1 commit. A planner integration typo caught
by compilation was corrected before these final tests; no benchmark state was affected.

Commit 1, `3cbffecb3` (`LMI: optimize native top-1 corpus routing`):
`routing.rs` adds the validated scratch-reusing view/tests; `build.rs` uses it in both
passes; `mod.rs` registers the opt-in benchmark; `evaluation_s3.rs` implements that
benchmark; `tests/lmi_phase_s3_prepare.py` and `tests/run_lmi_phase_s3.sh` provide
external orchestration; this document records methods and observations.

Commit 2, `LMI: plan bounded native training memory`: `build_plan.rs` contains checked
accounting/component ceilings/tests; `build.rs` logs/checks plans only during build;
`mod.rs` registers the planner; this document completes the engineering handoff.

No Cargo manifest, lockfile, training algorithm, F-series evaluation source or original
experimental artifact changed. Only new `work/phase_s3/` raw inputs/results/binaries
remain untracked. There is no push. Final commit SHAs and status are provided with
the delivery snapshot.

Workspace `cargo fmt --all -- --check`: FAIL; existing unrelated differences preserved: /home/nicoo/work/qdrant/lib/segment/src/index/field_index/full_text_index/inverted_index/on_disk_inverted_index/on_disk_postings.rs:4:.
