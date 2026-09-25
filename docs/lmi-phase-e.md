# Phase E: audit review and database-owned LMI construction

Date: 25 September 2026. Repository: `/home/nicoo/work/qdrant`. Starting commit: `4f3bd3798016162fb47d454dacb4f881cd44b524`. The working tree was clean before this task. No commit or push was requested or performed.

## Audit verdict

The attached P0 audit is substantially correct and the proposed integration is feasible. Inspection confirmed the optimizer → SegmentBuilder → build_vector_index seam, access to Qdrant-owned storage and identifier tracking, temporary-segment publication, Rust-side clustering/training in the reference implementation, and the need to keep native query-time routing and Qdrant scoring.

Four qualifications matter:

1. **Stable Rust compatibility:** `kmeans 1.1.0` itself declares nightly features (`test` and `portable_simd`). Removing the standalone SIMD scorer is insufficient to make that dependency work on stable. Phase E therefore implements the same small seeded Lloyd/random-sample clustering approach locally, without importing the standalone crate or adding a nightly requirement.
2. **Persistence cannot be entirely postponed:** SegmentBuilder drops constructed indexes, publishes the completed directory, and loads the segment again. An in-memory trained model would disappear during a normal build. This phase includes the minimum durable model/postings/open contract; broader E2 work remains.
3. **The request path needs configuration:** merely replacing the old LMI constructor does not make normal collections choose it. This change connects per-vector configuration through the optimizer to the real trained index and accepts default SearchParams without bypassing learned routing.
4. **Native inference does not imply a LibTorch-free server:** query evaluation uses the existing native MlpRouter, but a server built with the optional trainer links LibTorch. The regular feature-disabled build remains available.

## Implemented architecture

```text
REST / gRPC collection creation: vectors.lmi_config
    ↓
collection and shard optimizer configuration
    ↓ indexing threshold, or promotion of deferred points
Indexes::LmiTrained(LmiConfig)
    ↓
SegmentBuilder → build_vector_index → LmiIndex::build_trained
    ↓
Qdrant IdTracker + VectorStorage → eligible internal offsets
    ↓
seeded reservoir sample → stored float32 rows
    ↓
seeded CPU Lloyd k-means → pseudo-labels
    ↓
Rust/tch: Linear(d,h) → ReLU → Linear(h,B)
Adam, learning rate 0.001, cross entropy, shuffled batches
    ↓
explicit Linear.ws / Linear.bs export
    ↓
validated native MlpRouter; Torch/native logits checked on up to 32 rows
    ↓
classify all eligible stored vectors → bucket → PointOffsetType postings
    ↓
atomic lmi_state.json inside Qdrant's temporary segment
    ↓
Qdrant publishes segment directory and reopens saved index

normal explicit dense query (no filter, default params, no quantization)
    ↓
metric preprocessing for routing (including cosine normalization)
    ↓
native router → top nprobe buckets → per-query candidate offsets
    ↓
Qdrant visibility, point deletion, vector deletion and context checks
    ↓
BatchFilteredSearcher / RawScorer → Qdrant top-k
```

No HDF5 loading, Python preparation, PyO3 wrapper, duplicate persistent vector store, standalone SIMD scorer, GPU selection, or search-time training was introduced. The Python smoke-test client only sends HTTP requests and supervises an isolated test process.

The old `Indexes::Lmi {}` transient fixtures remain available for Phase B/C/D regressions. `LmiTrained` identifies the new automatically constructed and persisted index. Manual model installation and transient candidate-mode changes are rejected on persisted trained indexes to avoid making memory and disk state disagree.

## Configuration and operational boundaries

`lmi_config` is set per dense vector at collection creation, including named vectors. Omitting it preserves existing behavior. Per-vector HNSW settings and LMI settings are mutually exclusive. Collection-wide HNSW defaults do not select HNSW for an LMI-configured vector. Initial appendable segments remain Plain; normal optimizer thresholds control when an immutable trained index is built.

| Setting | Default | Meaning |
|---|---:|---|
| n_buckets | 8 | Pseudo-label classes and output logits |
| sample_size | 2048 | Maximum reproducible training sample |
| hidden_dim | 64 | One hidden ReLU layer |
| epochs | 30 | Complete passes through the sampled rows |
| batch_size | 256 | Training batch size; final partial batch is included |
| kmeans_iterations | 20 | Maximum Lloyd iterations |
| nprobe | 2 | Number of buckets selected for a query |
| seed | 42 | Local Rust RNG seed for sampling, clustering, weights and shuffling |

Validation requires `1 <= nprobe <= n_buckets <= sample_size`, with separate limits on dimensions and settings. Temporary sample and model parameter element counts are each limited to 32 million. These are allocation guards, not a total process-memory limit: Torch activations, gradients, Adam moments, postings and validation structures also consume memory.

Training is CPU-only and checks the Qdrant CPU permit. Torch intra-op and inter-op policies are conservatively set to one thread; no global Torch RNG seed is used. Cancellation is checked during sampling, clustering, batches, export checks and posting generation. A running tensor operation is not interrupted mid-operation. Reproducibility is tested on this runtime/hardware, not promised bit-for-bit across arbitrary platforms or future library versions.

If fewer eligible vectors than buckets exist, an explicit full-scan fallback state is saved. No training runs when an index opens. Small segments below the optimizer threshold also remain ordinary Plain segments.

Filters, non-default SearchParams (including `exact=true`), quantization, and unsupported query kinds retain the Plain path. Default omitted parameters and `{}` keep trained routing active. Legacy transient fixtures retain their previous parameter fallback behavior. Query-by-ID can introduce an exclusion filter internally, so the request-path acceptance test uses explicit dense vectors.

## Saved state and lifecycle

Version 1 uses one JSON artifact containing configuration, distance, dimension, storage count, sampled internal offsets, native model and postings. Atomic file replacement happens inside the temporary segment; Qdrant owns publication of the complete segment. Model and postings are generated together, and construction returns through the same open validator used on restart.

Opening checks version/configuration agreement, model dimensions and architecture, sample bounds/order, posting dimensions, offset bounds, duplicate membership, and coverage of currently live vectors. Deleted vectors may remain in saved postings: Qdrant's query-time validity checks exclude them. The artifact is included in index file and immutable-file listings for snapshots.

Internal offsets belong to one segment generation. Rebuilding through SegmentBuilder constructs new postings for that generation. A raw artifact must not be copied between arbitrary segments. Version 1 has structural validation but no cryptographic corpus identity or model authentication.

Trained segments are non-appendable; direct insertion/replacement through the index is rejected. Deletion delegates to Qdrant. Collection-level updates are expected to use Qdrant's normal mutable-segment and rebuild lifecycle; broad update/deferred-point/concurrency acceptance is follow-up work rather than an online-learning claim.

## Files and ownership

| Area | Files / role |
|---|---|
| Configuration and training | `lib/segment/src/index/lmi_index/config.rs`, `training.rs` |
| Build, save, open | `lib/segment/src/index/lmi_index/build.rs` |
| Native state and search | `mod.rs`, `routing.rs`, `read.rs`, `lifecycle.rs` in the same directory |
| Constructor and physical index | `lib/segment/src/types.rs`; `segment_constructor/segment_constructor_base/vector_index.rs` |
| Selection | Collection VectorParams, optimizer builder; shard optimizer config, selection and mismatch checks |
| Public API | Collection validation/conversions; API protobuf definition, generated Rust and configuration conversions |
| Optional dependency | Root and segment Cargo manifests; Cargo.lock adds tch/torch-sys and their dependencies |
| Regressions | `lib/segment/tests/lmi_phase_e.rs`; training and gRPC configuration unit tests |
| Request-path acceptance | `tests/lmi_phase_e_http_smoke.py` |
| Unsupported separate backend | `lib/segment/src/index/read_only/mod.rs` returns an explicit error for trained LMI |

Qdrant owns corpus storage, IDs, deletion/visibility, scoring, segment publication and rebuild scheduling. LMI owns the auxiliary model/postings and training working memory. Training parameters are exported from retained layer handles, not inferred from generated VarStore names.

## Reproduction

The verified local dependency environment is CPU Torch 2.5.1 with tch 0.18.1, stable Rust 1.98 (the repository requires at least 1.97), and clang++ on WSL Ubuntu. Python is used by the build script to locate the installed PyTorch distribution; operational training is called from Rust. A standalone compatible LibTorch distribution is another deployment option, not validated here.

```bash
cd /home/nicoo/work/qdrant
export PATH=/home/nicoo/.cargo/bin:/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CXX=clang++
export CXXFLAGS=-g0
export CARGO_BUILD_JOBS=2

cargo fmt --all
cargo check -p collection --tests --features segment/lmi-training --locked
cargo check -p edge --locked
cargo check --bin qdrant --locked
cargo check --bin qdrant --features lmi-training --locked
cargo test -p segment --features lmi-training \
  --test lmi_candidate_scoring --test lmi_dummy \
  --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e -- --nocapture
cargo test -p segment --features lmi-training --lib \
  lloyd_separates_clusters_and_obeys_cancellation -- --nocapture
cargo test -p collection --features segment/lmi-training --locked --lib lmi_config_ -- --nocapture
cargo test -p api --locked --lib lmi_configuration_defaults_roundtrip_and_validation -- --nocapture
cargo build --bin qdrant --features lmi-training
python3 tests/lmi_phase_e_http_smoke.py --output /tmp/lmi-phase-e-evidence
git diff --check
```

The HTTP script reserves a local test port, creates a separate temporary storage directory, starts its own process, and stops only that process. It retains storage and logs for inspection. It does not use an existing Qdrant instance. Use another `--port` if 16333 is occupied; the next port is used for gRPC.

Example collection body (the smoke test uses this small diagnostic configuration):

```json
{
  "vectors": {
    "size": 2,
    "distance": "Dot",
    "lmi_config": {
      "n_buckets": 2, "sample_size": 32, "hidden_dim": 8,
      "epochs": 60, "batch_size": 8, "kmeans_iterations": 8,
      "nprobe": 1, "seed": 42
    }
  },
  "optimizers_config": {"indexing_threshold": 1, "default_segment_number": 1},
  "shard_number": 1
}
```

A server without the feature rejects creation of a trained LMI configuration rather than silently accepting an index it cannot construct. Configuration changes to an existing trained index through VectorParamsDiff, query-specific nprobe, and concurrent HNSW/LMI residency are not implemented.

## Evidence and remaining work

Validation results are appended after the final runs below.

The implementation demonstrates a database-owned training/build/native-search seam, not an ANN performance result. The scalar Lloyd implementation and JSON format are deliberately experimental. They have not been validated at thesis-scale dimensions/corpus sizes. No production readiness, robust crash-injection campaign, snapshot recovery equivalence, live gRPC request, universal read-only serving, or measured recall/latency advantage is claimed.

Finish E2 by supporting the universal read-only backend, testing snapshot restore and failure recovery, and expanding update/deferred/multi-vector-name/optimization lifecycle acceptance. Then Phase F should compare Plain, HNSW, direct centroid routing, a centroid-derived affine router and the trained MLP at matched recall. Report router time, valid candidates scored, scoring time, end-to-end latency, build/training costs, reopen time and memory. Classification agreement with k-means is a diagnostic; retrieval recall is the actual search measure.

The small MLP is a compatible first integration model, not a necessary architecture for Euclidean k-means boundaries: nearest-centroid assignment is already represented by affine logits `2 c_j^T x - ||c_j||^2`. For Dot/Manhattan/Cosine retrieval, Euclidean clustering of the stored representation is a heuristic whose retrieval quality still needs evaluation. Architecture ablations and metric-specific partitioning belong in that evaluation.

## Review blockers resolved

The final review exposed compilation gaps outside the original server check and an update-time configuration inconsistency. These have now been corrected without adding LMI functionality:

- Ten existing collection-test fixtures and one edge configuration consumer explicitly initialize `lmi_config: None`.
- Edge configuration adapters treat LMI variants as having no HNSW settings. This makes enum matches exhaustive; it does not add edge LMI serving. The universal read-only trained-LMI rejection remains in place.
- Collection vector updates preflight every affected vector before any mutation. Introducing per-vector HNSW configuration on an LMI vector is rejected, including an empty HNSW object. The regression checks that an earlier plain-vector update in the same request is also left unapplied. Creation validation continues to reject adding LMI alongside HNSW; VectorParamsDiff has no supported LMI-setting field, so no inverse LMI PATCH functionality was introduced.
- `LmiRoutingState` no longer derives Serialize/Deserialize; DiskState still deserializes the model/postings and constructs a validated routing state. `set_candidate_mode` returns an error for persisted indexes. Phase C/D fixtures only gained `.unwrap()` at their existing setter calls; their expectations remain unchanged. A new Phase E test proves rejected mode changes leave both results and saved bytes unchanged.
- All three unrelated heck resolutions were restored to their original versions. A normalized lockfile audit found no removed packages or altered existing dependency resolutions: the sole added edge among existing packages is segment → tch 0.18.1; 15 new packages comprise the trainer dependency closure.

## Final verification after blocker fixes

Every final command below completed with exit code 0. Cargo checks/tests/builds used `--locked`. The collection test target was compiled, but the complete collection or edge test suites were not executed.

| Verification | Final observed result |
|---|---|
| `cargo check -p collection --tests --features segment/lmi-training --locked` | PASS |
| `cargo check -p edge --locked` | PASS |
| `lmi_candidate_scoring` | 2 passed, 0 failed |
| `lmi_dummy` | 1 passed, 0 failed |
| `lmi_phase_c` | 10 passed, 0 failed |
| `lmi_phase_d` | 12 passed, 0 failed |
| `lmi_phase_e` | 10 passed, 0 failed |
| Clustering/cancellation unit test | 1 passed, 0 failed |
| Collection configuration regressions (`lmi_config_`) | 2 passed, 0 failed |
| gRPC configuration unit test | 1 passed, 0 failed |
| Default `cargo check --bin qdrant --locked` | PASS |
| `cargo build --bin qdrant --features lmi-training --locked` | PASS |
| Extended isolated HTTP smoke test | PASS |
| `git diff --check` | PASS |

That is 35 LMI integration tests and 4 selected unit tests. The first blocker-check attempt caught a wrong type path in a newly added regression; it was corrected before the successful final sequence. No failed checks remain in that sequence. `cargo fmt --all` completed with the existing nightly-only option warnings, and unrelated formatter churn was removed.

The extended HTTP acceptance test additionally proves both nonempty and empty conflicting HNSW PATCH requests return HTTP 400; complete collection configuration stays unchanged immediately and after restart. Invalid creation configuration still returns HTTP 422. Python is only the isolated HTTP test driver.

```json
{
  "indexed_vectors": 300,
  "positive_candidates": 150,
  "negative_candidates": 150,
  "exact_results": 300,
  "filtered_results": 300,
  "conflicting_patch_rejected": true,
  "rejected_patch_preserved_configuration": true,
  "restart_preserved_configuration": true,
  "restart_same_results": true,
  "restart_same_state": true,
  "restart_did_not_train": true
}
```

The learned result sets were disjoint; default params matched the learned path; candidate ordering agreed with the exact ranking restricted to those candidates. Snapshot creation succeeded. Snapshot restoration is not implied. The test process was stopped; isolated storage remains available for inspection at `/tmp/qdrant-lmi-phase-e-09yv1azu`.

Evidence logs and exit-code records are retained in the task workspace under `work/phase_e/blockers-*.log` and `blockers-*.json`; HTTP results and both server logs are under `work/phase_e/blockers-http/`. The lockfile audit is `work/phase_e/blockers-lock-audit.json`.

## Remaining known limitations

- Universal read-only and edge LMI serving remain unsupported. Edge compilation success is not serving support.
- Filtered, non-default-parameter, quantized and unsupported query forms retain the documented Plain fallback.
- A binary without lmi-training can load native saved state but cannot train/rebuild it; a training-enabled deployment needs compatible LibTorch.
- Snapshot restore, crash/fault recovery, broad update/deferred-point/mixed-named-vector lifecycle coverage, and concurrent or large-scale training acceptance remain future work.
- Some backend operations may panic rather than return structured errors, cancellation cannot interrupt a tensor operation, and the allocation guards do not bound all working memory or CPU time.
- The JSON format has structural validation but no corpus fingerprint; no per-query nprobe or model/config tuning API was added. Duplicate posting-loop logic was deliberately not refactored in this blocker-only pass.
- No matched-recall advantage over HNSW or centroid/affine routing, production scalability, or complete learned-path telemetry is claimed. Those remain evaluation work.

## Proposed commit split — not executed

1. `feat(lmi): integrate CPU training and persisted routing into Qdrant`
   Production/configuration/API/optimizer changes, Cargo dependency closure, shared-consumer/exhaustive-match repairs, update preflight, invariant guards, embedded unit tests, and the two legacy fixture call-site adaptations. Keep these coupled changes together so the commit compiles.
2. `test(lmi): verify trained index lifecycle and REST request routing`
   The Phase E integration test file and extended isolated HTTP smoke test.
3. `docs(lmi): record Phase E architecture and verification boundaries`
   This engineering record, including final validation outcomes and explicit limitations.

No files were staged and no commit or push was made. The regular-server Phase E milestone is verified; the broader E2 limitations above remain explicit.
