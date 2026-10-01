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

## Pre-S.1 blockers (historical)

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

## S.2 design (implemented; verification below)

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

## S.2 completed construction and review

S.1 commit: `6f7085b9c` (`LMI: persist compact contiguous postings`). S.2 is the
separate `LMI: build postings with streaming two-pass construction` commit containing
this handoff. No push, large experiment, S.3 redesign or training-default change.

The complete build now makes three eligible-offset scans: seeded reservoir sampling,
prediction/count, and prediction/fill. The last two constitute the posting builder.
Every scan uses the same visible/non-deleted point mapping and named-vector deletion
predicate. Tracker and vector-storage read borrows are held throughout. The reservoir
uses the same ascending-offset order, RNG seed, increment-before-draw algorithm and
final sample sorting as before. Holes are never renumbered. Qdrant still owns vectors,
offset identity, deletions, deferred visibility, optimizers, snapshots and exact scoring.

After unchanged CPU training, the full f32 training matrix is dropped. Pass 1 predicts
one bucket per eligible vector and accumulates checked bucket counts. Checked prefix
sums create B+1 boundaries. One fallible exact reservation creates the 4N-byte offset
array, which is initialized before filling. Pass 2 repeats native prediction and
writes each offset at its bucket cursor. Overfilled and underfilled buckets fail;
invalid predicted buckets fail. Cancellation is checked before and during each scan,
between stages and after filling. Allocation/zero-initialization itself is not
interruptible. Stable borrows are required: matching counts alone cannot establish
that an arbitrary external visitor returned identical IDs in both passes.

Inference processes one row at a time, retaining existing prediction arithmetic and
tie policy. This bounds temporary input memory but doubles corpus forward passes.
No batched neural inference was added. Bucket lists retain source offset order;
query unions still sort/deduplicate offsets before Qdrant-native candidate scoring.
Empty and undersized segments retain the existing persisted exact-fallback behavior.
Postings are saved and released before reopening, avoiding a second retained decoded
posting array at that boundary. Sampling, count, allocation, fill, persistence and
native-open durations are logged. No new database lifecycle policy is introduced.

Review: `seen` and cursor increments are bounded by the checked u32 exclusive segment
end and allocated ranges; prefix/count additions are checked. B is validated <=65,536
before the builder, so B+1 cannot overflow. The helper is private to the module.
Existing sample/model guards remain in force. Public configuration validation and
native/universal open validation continue to reject malformed state.

### Files in the two commits

All Rust paths below are relative to `lib/segment/src/index/lmi_index/`.

| Commit | File | Change |
|---|---|---|
| S.1 | `postings.rs` | Contiguous offsets and bucket ranges, validation and unit tests |
| S.1 | `build.rs` | v2 binary persistence, legacy reader, shared opener validation bitmap |
| S.1 | `mod.rs`, `lifecycle.rs` | Module/state wiring and snapshot file enumeration |
| S.1 | `routing.rs` | Compact bucket access, removal of candidate HashSet |
| S.1 | `read.rs`, `read_only.rs` | Compact point count and universal open compatibility |
| S.1 | `evaluation.rs` | v1/v2 diagnostic reader compatibility only |
| S.1 | `lib/segment/tests/lmi_phase_e.rs` | Persistence, corruption, read-only and rebuild regressions |
| S.1 | `tests/lmi_state.py`, `tests/lmi_phase_e2_lifecycle.py` | Small-fixture binary inspection |
| S.1/S.2 | `docs/lmi-phase-s.md` | Design, verification and readiness record |
| S.2 | `build.rs` | Streaming eligibility, stable reservoir, two-pass construction, stage logs |
| S.2 | `postings.rs` | Count/prefix/fill helper; cancellation, holes, count mismatch, disk round-trip tests |
| S.2 | `evaluation.rs` | Formatting of S.1 diagnostic-reader lines only |
| S.2 | `tests/lmi_state.py` | Hash all three state files; model-only identity excludes sampled offsets |
| S.2 | `tests/lmi_phase_e_http_smoke.py`, `tests/lmi_phase_e2_lifecycle.py` | Use complete persisted-state hash |

The pre-existing `training.rs` changes and F.2-F.4 module registrations remain
uncommitted. Existing F reports, experimental sources, drivers and `work/` remain
untracked. The original training file backup compares byte-identically. No dependency
or lockfile change was needed. Verification uses the actual working tree, including
the preserved prior training refactor; it is not a clean-checkout-only test claim.

## Before versus after and memory accounting

Before Phase S: full eligible-offset Vec, nested growable posting vectors, inline JSON
model/postings, hash-set validity and query deduplication, single corpus prediction
pass after sampling. After S.2: streamed eligibility; count/prefix/fill construction;
one contiguous posting array; metadata/router/postings persisted separately; bitmap
open validation and sorted candidate deduplication. The legacy v1 compatibility reader
still materializes old JSON; only new v2 builds gain the compact file layout.

Let N be eligible vectors, S the configured sample cap (actual sample <=min(S,N)), d
dimension, B buckets, H hidden width, K training batch size, and T physical offset
slots including holes. For a single segment:

- Training matrix: 4Sd bytes in Rust plus a full Torch f32 copy during training;
  sampled offsets, labels and shuffle order add O(S). The sample matrix is released
  before corpus classification. Allocator RSS may not fall immediately.
- Router parameters: P=dH+HB+H+B f32 values, about 4P bytes. Training adds gradients,
  Adam moments, tensor/export copies and O(K(d+H+B)) batch activations/workspace.
  The existing 32M-element guards are not a complete peak-memory budget.
- Lloyd clustering: O(Bd) f64 centers/sums and O(S) labels/order; approximately
  O(iterations * S * B * d) distance work, plus final assignment.
- Posting construction: 4N bytes plus O(B) counts/boundaries/cursors. There is no
  N-element eligible-offset or predicted-label side array. Native per-row inference
  requires O(d+H+B) temporary values/indices and repeatedly validates the model and
  sorts bucket outputs. Two complete prediction passes cost O(NP + NB log B) each.
- Persisted postings: exactly 4N + 8(B+1) + 16 bytes with the current bincode layout.
  Persisted model/sample file adds O(P+S), metadata O(1) for fixed config fields.
- Open: decoded heap postings/model plus about T/8 validation bytes. Universal remote
  IO may hold a whole encoded buffer alongside decoded arrays. This is not zero-copy
  mmap serving. Open validation still scans relevant offsets/postings.
- Query: O(C) candidate offsets for selected buckets and O(C log C) sorting, followed
  by native scoring of C candidates. Corpus vectors remain solely Qdrant-owned.

Synthetic 100,000 offsets / 316 buckets round-trip to disk in **402,552 bytes**.
This verifies serialization and equality, not ANN quality, peak RSS or throughput.

## Final verification (2026-10-01)

Commands ran in `/home/nicoo/work/qdrant` with the following local environment:

```bash
export PATH="/home/nicoo/.cargo/bin:/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH"
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export CXX=clang++ CXXFLAGS=-g0 CARGO_BUILD_JOBS=2
```

| Command | Observed result |
|---|---|
| `cargo test -p segment --features lmi-training --locked --lib index::lmi_index -- --nocapture` | PASS: 7 passed, 0 failed, 4 ignored |
| `cargo test -p segment --features lmi-training --locked --test lmi_candidate_scoring --test lmi_dummy --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e -- --nocapture` | PASS: 2+1+10+12+14=39 passed, 0 failed |
| `cargo build --bin qdrant --features segment/lmi-training --locked` | PASS: 2m22s |
| `python3 tests/lmi_phase_e_http_smoke.py --output work/phase_s/verification/s2-http --port 17533` | PASS: build, PATCH rejection, restart and fresh snapshot restore |
| `python3 tests/lmi_phase_e2_lifecycle.py --output work/phase_s/verification/s2-lifecycle --port 17543` | PASS: solo and mixed named-vector lifecycle cases |
| `cargo check --bin qdrant --locked` | PASS: 12.03s |
| `cargo check -p edge --locked` | PASS: 4.69s |
| `cargo clippy -p segment --lib --features lmi-training --locked` | PASS with 4 existing warnings: 3 HNSW formatting suggestions and 1 training needless borrow; no -D warnings claim |
| `rustfmt --check --edition 2024 --config skip_children=true lib/segment/src/index/lmi_index/{build,postings,evaluation}.rs` | PASS after local formatting correction |
| `cargo fmt --all -- --check` | FAIL on one unrelated existing full-text `on_disk_postings.rs` import-order difference; preserved intentionally |
| `git diff --check` | PASS |

The four ignored tests are the explicit Phase F, F.2, F.3 and F.4 studies; none was
rerun. Passing Phase E tests include universal read-only serving, binary corruption
rejection by both openers, legacy reopen, snapshot enumeration, optimizer rebuild,
deletion validity, seed repeatability and native scoring/fallback regressions.
Library tests cover empty ranges, sparse offsets, legacy equality, changed counts,
invalid bucket IDs and cancellation in both passes.

HTTP acceptance: 300 indexed vectors; each learned query returns 150 candidates;
exact and filtered requests return 300. Source server stopped before restore to a
fresh storage directory. Configuration, IDs, scores and the combined metadata/router/
postings SHA-256 remain identical:
`f1d803eeea4a28129e82b1e42fb8af46d11bda526410ae4c6ef43c32c151e3c3`.
Restored logs explicitly show StaticLearned open and search, with no build/training.
Snapshot size was 122,880 bytes. These are tiny fixture measurements, not scale results.
Example count/allocation/fill times: 0.000780 / 0.000014 / 0.000705 seconds.

Both lifecycle cases hide deferred points, leave old states immutable until retirement,
rebuild segment-local postings, exclude deleted IDs and score updates correctly;
570 final live points each. Mixed named indexes remain independent. Restart does not
train. A final repeat with stricter model-only hashes also PASSED both cases and restart; it is recorded separately in
`work/phase_s/verification/s2-lifecycle-final/`; earlier evidence is preserved.

All logs/results are retained under `work/phase_s/verification/`, outside Git.
No large retrieval study, peak-RSS measurement, 1M ingestion or clean-checkout-only
build was performed. The committed code is structurally scale-aware; it has not been
empirically validated at 1M, 10M, 100M or 1B.

## Focused S.3 readiness assessment

| Target | Remaining gate verified in current code |
|---|---|
| 1M | No corpus-size guard directly forbids 1M, but there is no 1M ingestion/RSS/build/reopen/snapshot benchmark. Training is CPU/one thread; whole sample is copied into Torch. At d=768 the configured sample cap must be <=41,666 under the 32M-element guard, even for a smaller actual segment. Establish memory/work budget and representative sample coverage before increasing it. |
| 10M | Same evidence gap; Lloyd cost and two scalar corpus inference passes become substantial. Inference rescans model validity and sorts all B outputs per vector. Add measured bounded inference batching only after parity tests. Explicit per-segment B policy and concurrent-optimizer memory accounting are missing. |
| 100M | No optimized clustering backend, large-sample training path, true mmap serving, bounded remote decode or scale ingestion harness. Heap postings alone are 400MB/segment at N=100M, plus sample/model/validation/concurrent builds. Float32 768-D corpus is 307.2GB before staging/metadata. Need resource and IO planning plus the earlier scale ladder. |
| 1B | All prior gates; 4GB postings and 3.072TB raw 768-D f32 corpus at single-segment arithmetic. u32 offsets can represent this count, but require per-segment limit/overflow and distributed lifecycle testing. Large B grows output weights, batch logits, per-query scoring and inference cost. The sample config hard cap is 1M and does not allow the reference 5M sample. No billion-scale capacity or performance claim is supported. |

**Single best next S.3 task:** implement a checked, explicit per-build training-memory
and work-budget planner, replacing the opaque 32M-element approximation only when
equivalent rejection safety is established. Account for sample copies, centroid
workspace, optimizer states, batch logits, model export and concurrent builds. Expose
the selected sample/B/batch plan and test arithmetic/limits before allocating. Keep
defaults and training mathematics fixed initially; use it to design the first bounded
1M benchmark rather than lifting the guard blindly. Clustering/backend selection and
batched corpus inference then become independently measurable follow-up increments.

Additional preserved boundaries: structural validation does not bind state to corpus
generation; valid same-shaped file substitution remains undetected. Heap-loaded
binary state is not mmap-backed. Tiny/empty segments may intentionally use exact
fallback; filtered/exact/unsupported query paths retain their existing semantics.
This work does not promise better neural retrieval quality or change prior F results.

Reference resource requests reinforce the need for staged measurements: the upstream
Rust benchmark requests 32 CPUs/400GB; a 1B MetaCentrum job requests 32 CPUs/1,000GB
and 800GB scratch (different dataset dimensions; not capacity proof for 768-D data).
[FAISS uses MIT licensing](https://github.com/facebookresearch/faiss/blob/main/LICENSE);
its [native build requirements](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
include C++/OpenMP/BLAS tooling. No FAISS, GPU, hierarchy, DLI, CLI, new loss, or mmap
redesign is implemented by these commits.
