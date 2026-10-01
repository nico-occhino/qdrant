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
