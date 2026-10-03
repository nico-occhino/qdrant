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
