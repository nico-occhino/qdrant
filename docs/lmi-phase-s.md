# Phase S: compact static LMI storage and streaming construction

## Scope and reference audit (2026-10-01)

Base: `6e3cfdb0f`, branch `thesis/lmi-integration`. Existing F.2-F.4 changes in
training/evaluation and untracked reports are preserved and excluded from S commits.
No serving loss changes, dynamic indexes, GPU, FAISS dependency or large dataset run.

Read the Coda LearnedMetricIndex branches `paper-sisap24-indexing-challenge`
(README/task1/utils), `enhancing-performance` (1B/alpha/gpu/quant/inmemory/compile,
utils and MetaCentrum jobs), and `rust` (final/rust_lmi, lmi-lib, benchmark).
Rust reference revision: `1ad2587dc6d26cac9fa769b97a0c521f8f2bf79e`;
enhancing-performance: `de96f0f5c5a02fc1edff0fc28ccd1b3ad2d18814`.
Read the historical SISAP23 README, data-insertion/vibe-ood branches, dynamic-learned-index,
DynamicLearnedIndexRust paper-version, BalancedTreeLearner, ProteinEmbeddingBenchmark,
and AlphaFind documentation. Read-only reference copies/manifests are in work/phase_s/references.
No upstream files were changed. The Rust branch has no root README (404); its final/lmi-lib
README and sources are present.

The SISAP24 reference uses spherical FAISS clustering, sqrt(dataset size) buckets,
512 hidden units, 15 epochs, a 1M sample/chunk at 100M. The Deep1B experiment uses
a 5M sample, 1M chunks and about 31,622 buckets. These are reference experimental
settings, not defaults for Qdrant segments. Python chunks classification but retains
an O(N) label array and duplicates vector payloads in buckets. Rust count_bucket_sizes
and create_buckets_scalable count then fill exact storage in separate passes.
We adopt only that construction principle and retain offset-only postings.
GPU, quantization, compilation, in-memory and alpha scripts are experimental variants;
their existence is not evidence of a portable performance win. No ranking of those
variants is inferred without matching result files/hardware.

## Confirmed local blockers

build.rs materializes eligible offsets; DiskState embeds nested postings and model in
JSON; validation allocates a HashSet; native routing repeatedly validates the model
and sorts all outputs; CPU Lloyd training keeps the sample resident. The 32M-element
guard prevents a 1M x 768 sample. PointOffsetType is u32; the existing exclusive end
conversion restricts total slots to u32::MAX (1B is representable, not runtime-proven).

## S.1 design

Runtime postings: one Vec<u32> plus B+1 u64 boundaries. Query bucket access borrows
ranges. Candidate sorting preserves native equal-score ordering; dedup is retained
for transient fixtures without a candidate HashSet.

Version 2: lmi_state.json contains configuration, metric, dimension and physical vector
count; lmi_router.bin holds the router plus sampled offsets; lmi_postings.bin holds
boundaries and offsets. Existing common::fs atomic_save_bin/json and universal IO
read_bin_via are reused. Binary arrays use bincode fixed-width little-endian encoding.
Metadata is written last in the unpublished SegmentBuilder directory; SegmentBuilder
publishes the completed directory by rename. In-place rewriting of published indexes
is not supported. All three files are enumerated for snapshots. Version 1 JSON remains
readable through both native and universal open; readers never rewrite old indexes.

Native and universal open retain full range, uniqueness, coverage, sample, model,
configuration and metric checks. A one-bit-per-physical-offset bitmap replaces the
HashSet; validation remains O(N) time and O(total slots/8) additional bytes. Boundary
checks reject absent, descending, nonzero-start and wrong-terminal ranges. Deleted
postings may remain, and native validity filtering stays authoritative.

This first format uses heap-resident decoded arrays, not true mmap-backed serving.
common::mmap typed slices were inspected, but adopting them only for native open
would split the existing universal serving implementation. Universal remote readers
may temporarily buffer the whole binary file while decoding. Mmap/streamed remote
decode is a next storage increment, explicitly not claimed here. bincode is the
existing common binary helper rather than a private storage layer.

Structural validation still cannot identify replacement by a different valid
same-shaped state. No corpus fingerprint or authenticated file binding is added.

## Planned S.2

Reservoir sampling scans visible, point-live and named-vector-live offsets without an
eligible Vec. After unchanged CPU training, count predictions then fill one exact
offset array in a second scan, under stable tracker/storage read borrows. Use checked
prefix arithmetic, fallible allocation, cancellation and deterministic offset order.
Native inference initially processes bounded rows (batch size one), not a new Torch
batch implementation. Record sampling/count/allocation/fill/persistence stage times.

## Remaining scale gates

Do not remove the 32M guard in S.1/S.2. S.3 needs a defined training/model/workspace
memory budget, checked arithmetic and a scalable clustering decision. Existing CPU
Lloyd is O(sample x buckets x dimension x iterations). FP16 samples plus f32 batches
could reduce sample storage, but current tch training copies the full sample tensor;
that requires a distinct, tested training change. FAISS offers optimized spherical
clustering but adds C++/BLAS packaging, platform and license-review obligations; no
dependency is introduced here. Compare a Rust mini-batch backend and FAISS externally
before selecting one. Keep B explicit and report segment cardinality rather than
applying collection-size sqrt(N). Multiple segments add routing work and their local
candidate counts must be summed; global recall is measured after native top-k merge.

Datasets stay in separate benchmark tooling. BATL documents HDF5, FBIN/IBIN, fvecs,
NPY and image/text datasets; protein sources provide embedding-specific dimensions
and metrics. No dataset reader belongs in the physical index. A generalized ingestion
harness and the 1M/10M runtime ladder remain separate deliverables.

DLI's buffers/static levels/compaction map conceptually to appendable segments and
optimizer rebuilds; identical search/budget semantics are not implied. CLI splitting,
replay and output expansion require a later design. No lifecycle policy is changed.

At 100M offsets need about 400MB, at 1B about 4GB, plus 8(B+1)+16 binary bytes;
validation adds 12.5MB/125MB respectively. Corpus vectors remain Qdrant-owned.
100M x 768 f32 alone is 307.2GB; 1B is 3.072TB, before copies/staging. These stages
require server storage and suitable MUNI/MetaCentrum resources, not the laptop.

## Verification

Results and exact commands are appended after each phase. No 100M/1B execution is intended.


### S.1 observed verification

`cargo test -p segment --features lmi-training --locked --test lmi_candidate_scoring
--test lmi_dummy --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e -- --nocapture`:
2 + 1 + 10 + 12 + 14 = 39 passed, zero failed. Includes native/universal corruption,
legacy reopen, binary file enumeration, four metrics, deletion and optimizer rebuild.
`cargo test -p segment --features lmi-training --locked --lib index::lmi_index`:
5 passed, 4 intentional large-experiment ignores; no experiment was rerun.
`cargo check --bin qdrant --locked`: pass (59.53s).
`cargo check -p edge --locked`: pass (8.55s).
Targeted rustfmt and `git diff --check`: pass; nightly-only formatting warnings expected.
Synthetic 100K/316-bucket posting persistence is exactly 402,552 bytes. This is a
storage-size regression, not a repeated LAION retrieval benchmark or a peak-RSS result.

Existing ignored Phase F native evaluator now reconstructs its diagnostics through
the v1/v2 reader; its pre-existing F.2-F.4 module registrations are not part of S.1.
The E2 Python fixture reader understands v2 binary postings and model-file identity.
Historical raw measurements and models are never rewritten. Training changes already
present before S.1 remain outside these commits.
