# Phase S.3B1 — bounded native corpus routing

## Scope

This increment replaces the posting-build hot path's one-row-at-a-time call
site with a bounded, native Rust batch implementation. It does not change
KMeans, training, query-time routing, persistence format, or the two-pass
count/fill postings architecture.

`routing_batch_size` is an explicit experimental `LmiConfig` setting. Its
default is **1**, preserving the S.3A scalar operating point. It is separate
from the Torch training `batch_size`; query routing is unaffected.

## Native primitive audit and decision

The workspace already contains `ndarray` and Rayon. Rayon would add parallel
CPU work, so it is intentionally not used. The existing Qdrant SIMD kernels
implement distances and quantized-code operations; they are not interchangeable
with a dense MLP layer and may use a different floating-point reduction order.
No BLAS or matrix-multiplication dependency is currently present.

The selected implementation is `BuildBatchRouter`: contiguous reusable Rust
`Vec<f32>` buffers, row-major dense-layer loops, no Rayon and no external
runtime. It retains the scalar router's ordered `f32` multiply/add loop for
each output neuron. Consequently it is expected to be bit/partition preserving
for identical model rows, inputs and compiler settings. The scalar
`BuildRouter` remains in the module as the reference implementation.

## Build path

```
eligible VectorStorage scan
  -> collect <= K offsets + dense rows
  -> BuildBatchRouter::top_buckets
  -> immediately count/fill CompactPostings
  -> clear and reuse buffers
```

The same closure and order run for both count and fill. No `N × B` logits or
full-corpus predictions are retained.

## Bounded memory

For `K = routing_batch_size`, `d` input dimensions and
`W = max(d, hidden_dim, n_buckets)`, the implementation allocates:

* gathered input: `4 K d` bytes;
* two activation buffers: `8 K W` bytes;
* offsets plus bucket identifiers: approximately `12 K` bytes on 64-bit
  builds.

The build plan reports these three values independently; it makes no invented
host-RAM admission guarantee. K is validated in `1..=65536` and is explicit in
the experimental configuration.

## Numerical and threading policy

The batch path is single-threaded. A cancellation check is performed before a
batch, before each layer and before each input row. Top-1 uses strict `>` so
ties remain assigned to the smaller bucket ID. Focused tests compare scalar and
batched assignments over batch capacities 1, 2, 3 and 8, including a final
partial batch and empty input.

## Deferred measurements

The S.3A SIFT1M corpus and LAION parity artefacts stay under `work/phase_s3`.
S.3B1 measurements must be written to a new non-versioned directory and use
the existing `phase_s3_build_benchmark` harness with explicit
`routing_batch_size`. Large-router synthetic shapes are compute microbenchmarks
only; they are not corpus-scale evidence.

At a later phase, storing u16 labels would cost `2N` bytes: about 200 MB at
100M vectors and 2 GB at 1B vectors. It could save the second forward pass,
but is deliberately not implemented here.

## Status limits

SIFT1M is empirically tested end-to-end in S.3A at N=1M, d=128. LAION was
empirically tested near N=100K, d=768 in S.3A. 10M, 100M and 1B are not yet
tested end-to-end. A future B=10,000 or B=31,622 router microbenchmark is only
a dense-router compute measurement and does not establish corpus scalability.

## Initial SIFT1M end-to-end measurements

All runs used the existing SIFT1M corpus (N=1,000,000, d=128, Euclidean,
64 buckets), CPU affinity 0, the perf profile, and the S.3A training
configuration. They are one controlled trial per batch point, not a claim of
statistical significance.

| routing K | count s | fill s | count+fill s | build s | peak RSS KiB | batch router workspace |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3.394437 | 3.191226 | 6.585663 | 7.651291 | 1,902,548 | 16,512 B |
| 256 | 3.321129 | 3.356609 | 6.677738 | 7.681152 | 1,901,956 | 264,192 B |

For comparison, S.3A’s two scalar-reference trials had median-like count
3.34889 s, fill 3.28116 s and total build 7.69618 s. The K=16 router and
postings binary SHA-256 values match the scalar reference exactly; K=256 does
as well. `lmi_state.json` deliberately differs because it records the explicit
`routing_batch_size` configuration.

These results show that the ordered native batch implementation preserves the
persisted partition. They do **not** show a reliable throughput improvement at
B=64: dense arithmetic and row-wise scalar reduction still dominate, and the
extra gather/copy work can offset batching. No production default changes on
this evidence.

## Continuation: compute diagnosis and LAION-768 regression (2026-10-03)

The starting tree was `313d7de2a` with only preserved `work/phase_s3/` artefacts
untracked. This continuation adds opt-in benchmark and analysis tooling; the
production kernel, configuration default, clustering and persistence stay as
committed in `be40920e9`.

### Method

The synthetic benchmark creates deterministic `f32` inputs and weights for
the existing two-layer ReLU router. It times the actual `BuildBatchRouter`
top-1 call on CPU 0, with Torch/OMP/MKL/Rayon thread environment variables set
to one. It compares every assignment with `BuildRouter` on the same rows. Three
separate processes produced the medians below. `K` is the maximum rows per
call. The native router executes on the caller thread; /proc/self/status reported
two process threads for the test process. The workspace estimate adds
build-path gathered inputs and point offsets to the observed capacities of
the reusable activation buffers and bucket IDs. Large-output tests use only
256 generated rows, without a corpus scan or model training.

| Shape | d/H/B | K | Rows/run | Median rows/s | Median µs/vector | Estimated build workspace bytes | Mismatches / 3-run rows |
|:--|:--|--:|--:|--:|--:|--:|--:|
| A | 128/64/64 | 1 | 4096 | 325,056.9 | 3.08 | 1,548 | 0/12,288 |
| A | 128/64/64 | 16 | 4096 | 326,503.2 | 3.06 | 24,768 | 0/12,288 |
| A | 128/64/64 | 64 | 4096 | 325,003.6 | 3.08 | 99,072 | 0/12,288 |
| A | 128/64/64 | 256 | 4096 | 319,622.5 | 3.13 | 396,288 | 0/12,288 |
| B | 768/64/64 | 1 | 1024 | 49,835.4 | 20.07 | 9,228 | 0/3,072 |
| B | 768/64/64 | 16 | 1024 | 49,188.6 | 20.33 | 147,648 | 0/3,072 |
| B | 768/64/64 | 64 | 1024 | 46,359.2 | 21.57 | 590,592 | 0/3,072 |
| B | 768/64/64 | 256 | 1024 | 48,294.7 | 20.71 | 2,362,368 | 0/3,072 |
| C | 768/512/1,024 | 1 | 256 | 2,821.3 | 354.44 | 11,276 | 0/768 |
| C | 768/512/1,024 | 16 | 256 | 2,874.3 | 347.92 | 180,416 | 0/768 |
| C | 768/512/1,024 | 64 | 256 | 2,797.4 | 357.47 | 721,664 | 0/768 |
| C | 768/512/1,024 | 256 | 256 | 2,832.2 | 353.08 | 2,886,656 | 0/768 |
| D | 768/512/10,000 | 1 | 256 | 365.8 | 2,734.01 | 83,084 | 0/768 |
| D | 768/512/10,000 | 16 | 256 | 374.5 | 2,670.53 | 1,329,344 | 0/768 |
| D | 768/512/10,000 | 64 | 256 | 380.7 | 2,626.44 | 5,317,376 | 0/768 |
| D | 768/512/10,000 | 256 | 256 | 380.6 | 2,627.24 | 21,269,504 | 0/768 |
| E | 768/512/31,622 | 1 | 256 | 111.1 | 9,003.34 | 256,060 | 0/768 |
| E | 768/512/31,622 | 16 | 256 | 114.2 | 8,757.26 | 4,096,960 | 0/768 |
| E | 768/512/31,622 | 64 | 256 | 116.2 | 8,607.85 | 16,387,840 | 0/768 |
| E | 768/512/31,622 | 256 | 256 | 117.0 | 8,549.82 | 65,551,360 | 0/768 |

At B=64, K changes throughput little and sometimes lowers it. At B=1,024
the best median differs from K=1 by about 2%. At B=10,000 the best differs by
about 4%, and at B=31,622 by about 5%. The benchmark always measures K in the
order 1/16/64/256 and A/B timing windows are short, so those small advantages
cannot be isolated from cache warming and ordinary run variation. The prior
SIFT1M end-to-end K=16 and K=256 builds were also effectively neutral.

### Why the current batch changes little

For **each** dense layer, the code nests loops `for input row -> for output
neuron -> for input coordinate`, with one ordered `f32` accumulator per neuron.
K>1 only changes contiguous collection and buffer/call reuse. There is no
explicit reuse of a weight across multiple rows within one inner loop, no
matrix multiply, no Rayon, and no source-level SIMD. A narrow inspection of
the optimized `BuildBatchRouter::top_buckets` symbol showed scalar `mulss`
and `addss` in the dense accumulation, including scalar unrolling; packed
`mulps`/`addps` and fused multiply-add were not observed there. This is
evidence for the inspected build, not a compiler guarantee on other CPUs.

An isolated B=10,000 stage timing proxy used the same ordered dense loops on
128 generated rows. It measured input gather **0.000213 s**, first dense
**0.026072 s**, ReLU **0.000019 s**, second dense **0.425524 s**, and argmax
**0.000581 s**. The second dense layer was **94.1%** of those timed stages;
the first dense layer was **5.8%**. The proxy excludes production cancellation
checks and storage decoding, so it diagnoses the compute kernel rather than
decomposing a full Qdrant build. Its result points the next optimization at
the H×B output layer.

### LAION-768 parity and timing

One controlled run per K used the established `/home/nicoo/work/lmi-phase-f-data`
corpus (N=99,780, d=768, cosine) and the S.3A model configuration. All three
router binaries and posting binaries have the same SHA-256 values as the
preserved scalar baseline. The one measured learned query returned identical
IDs and scores. Metadata files differ only in the recorded routing batch
configuration.

| K | Count s | Fill s | Combined s | Whole build s | Router/postings/query parity |
|--:|--:|--:|--:|--:|:--|
| 1 | 2.082607 | 2.082583 | 4.165190 | 5.051211 | exact |
| 16 | 1.947209 | 1.960293 | 3.907502 | 4.713098 | exact |
| 256 | 2.037723 | 2.075651 | 4.113374 | 4.882435 | exact |

K=16 was about 6.2% faster than K=1 for combined count/fill in this single
trial; this is a dimensionality regression and parity check, not sufficient
evidence for a production default change. `routing_batch_size=1` remains the
default. No assignment mismatch or changed posting was found in any synthetic,
SIFT1M, or LAION comparison completed here.

### S.3B1b design to evaluate next

| Candidate | Throughput mechanism | Temporary memory | FP/partition parity | Threads and Qdrant resource integration |
|:--|:--|:--|:--|:--|
| A. Rayon row parallelism | Independent rows on more cores; no per-core arithmetic improvement | Per-worker activation buffers plus gathered batches; scales with worker count | Per-row ordered loop can remain exact | Multiple workers must be bounded by an actual Qdrant CPU permit; avoid the unconstrained global pool. Measure multi-core separately. |
| B. Single-threaded tiled dense layers | Reuse each weight over a small row tile and improve locality, especially H×B | O(Kd + KH + tile×B) if logits are tiled; can stream top-1 instead of retaining K×B | Choose increasing input-coordinate order per row; prohibit fused contraction when exact scalar parity is required | One compute thread, current permit accounting unchanged. |
| C. SIMD across rows | Broadcast a weight and update several independent row accumulators per input coordinate | Potential packed/transposed input O(Kd), plus a few vector accumulators and tile logits | Each lane can preserve scalar input order; exact bits still require tests because FMA/contraction and compiler codegen may differ | One compute thread; platform-specific SIMD dispatch requires portable fallback and CPU feature checks. |
| D. Existing pure-Rust matrix multiply | `ndarray` invokes `matrixmultiply::sgemm` for f32 matrix products, with tiled GEMM kernels | Input/activation/logit matrices up to O(Kd+KH+KB), plus implementation scratch | Reduction order likely differs; measure assignment mismatches and small-margin concentration before use | `ndarray` is presently a **dev dependency** of `segment`; `matrixmultiply-threading` is not enabled. Production adoption would need dependency and thread-policy review. |

B and C can share a design: for each output neuron, visit input coordinates
in the scalar order while updating several row accumulators in parallel. A
transposed/tiled input pack would add about `4Kd` bytes and one O(Kd) copy,
but gives contiguous row-lane loads. Exact scalar partition parity is
plausible because each lane sees the same sequence of additions; it is not
guaranteed without bit-level tests on the target compiler and CPU. Begin with
a single-threaded tile and the same end-to-end byte-parity gates before
considering Rayon or matrix multiplication.

### Temporary u16 labels: estimate only

If B≤65,535, storing one temporary u16 label per vector costs 2N bytes:
**20 MB** at 10M, **200 MB** at 100M and **2 GB** at 1B vectors (decimal).
Writing and rereading labels moves at least 4N bytes: 40 MB, 400 MB and 4 GB.
It would avoid one of the two neural corpus passes. At synthetic shape D's
best median 380.7 rows/s, that pass corresponds arithmetically to about
**7.3 h / 73 h / 730 h** for 10M / 100M / 1B rows; shape E's best 117.0
rows/s gives about **23.7 h / 237 h / 2,374 h**. These are compute-only
extrapolations from hundreds of synthetic rows, **not** measured corpus build
times or scalability claims. The IO term is `2N/write_bandwidth +
2N/read_bandwidth`; no controlled temporary-label IO bandwidth was measured,
so no net time saving is claimed. Temporary labels remain unimplemented.

### Reproduction and raw evidence

The benchmark source is `lib/segment/src/index/lmi_index/evaluation_s3b1.rs`.
With the existing LibTorch installation, compile the ignored tests using
`cargo test -p segment --profile perf --features lmi-training --locked --lib
phase_s3b1_ --no-run --message-format=json`, then run the resulting segment
test binary on CPU 0 with `--ignored --nocapture --test-threads=1`. Set
`LMI_S3B1_MICRO_OUTPUT` or `LMI_S3B1_PROFILE_OUTPUT` to a **new** file for the
two tests. Set Torch/OMP/MKL/Rayon thread environment variables to one. The
LAION build uses the existing `phase_s3_build_benchmark` ignored test with
`LMI_S3_DATA=/home/nicoo/work/lmi-phase-f-data` and a JSON config setting
`routing_batch_size` to 1, 16 or 256. Detailed launch scripts and the
non-overwriting analysis script are under `tests/`.

Raw outputs, logs, compiler metadata, per-run JSON, the aggregate CSV/JSON,
and both sets of persisted LMI files are preserved under
`work/phase_s3/s3b1-continuation/`; S.3A artefacts were not overwritten.
The large-router numbers are **compute microbenchmarks only**. SIFT1M was
tested end-to-end at N=1M, d=128 and LAION here at N=99,780, d=768. 10M,
100M and 1B remain untested end-to-end.

The exact saved-run command pattern is:

```bash
cd /home/nicoo/work/qdrant
export LIBTORCH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch
export LD_LIBRARY_PATH="$LIBTORCH/lib"
export LMI_S3B1_RUN_ROOT=work/phase_s3/s3b1-new-run
mkdir -p "$LMI_S3B1_RUN_ROOT"
cargo test -p segment --profile perf --features lmi-training --locked \
  --lib phase_s3b1_ --no-run --message-format=json \
  > "$LMI_S3B1_RUN_ROOT/compiler-e.jsonl" \
  2> "$LMI_S3B1_RUN_ROOT/compiler-e.log"
bash tests/run_lmi_phase_s3b1_micro.sh
bash tests/run_lmi_phase_s3b1_laion.sh
python3 tests/lmi_phase_s3b1_analyze.py
```

Use a new `LMI_S3B1_RUN_ROOT` each time. The scripts refuse to overwrite
existing raw outputs. The SIFT1M comparison is retained from the earlier
S.3B1 run under `work/phase_s3/s3b1-k16` and `s3b1-k256`.
