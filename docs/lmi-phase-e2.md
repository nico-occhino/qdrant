# Phase E2.1 engineering handoff — snapshot restore and structural corruption validation

Checkpoint: `7029bcd0f`, `thesis/lmi-integration`, clean working tree verified 2026-09-25. No commits or pushes authorized.

| Lifecycle | Source path and function | Finding / verification boundary |
| --- | --- | --- |
| Build / publication | `lib/shard/src/optimizers/segment_optimizer.rs` configured index selection; `lib/segment/src/segment_constructor/segment_builder.rs::update/build`; `segment_constructor_base/vector_index.rs::build_vector_index`; `index/lmi_index/build.rs::build_trained` | Builder copies/deduplicates source data into new offsets, trains against target storage, writes model and postings together atomically. Saves metadata/version then renames the complete directory and loads it. No old LMI postings reused. |
| Native reopen | `segment_constructor_base/segment.rs::load_segment`, `create_segment.rs`, `vector_index.rs::open_vector_index`, `LmiIndex::open_trained` | Reads native state and validates; errors propagate. No training in open. Tiny/empty exact mode is explicitly persisted, not a recovery from missing trained state. |
| Collection snapshot | `lib/collection/src/collection/snapshots.rs::create_snapshot/restore_snapshot`; `shards/local_shard/snapshot.rs::get_snapshot_creator/restore_snapshot`; `lib/segment/src/segment/snapshot.rs::snapshot_files`; `index/lmi_index/lifecycle.rs::files/immutable_files` | LMI state is included alongside owning vector storage and IDs. Collection configuration is included. Existing HTTP test proves creation only, not restoration. |
| Restore | `src/main.rs::recover_collections_from_snapshot_args`; `src/snapshots.rs::recover_snapshots/recover_snapshot_mappings`; `Collection::restore_snapshot`; `lib/shard/src/snapshots/snapshot_utils.rs::restore_unpacked_snapshot`; `Segment::restore_snapshot_in_place`; native open | CLI recovery is suitable for isolated, stopped-source acceptance. Fresh target required. No LMI-specific restore code appears necessary; must execute to verify. REST recovery/distributed/partial snapshots are not covered by this acceptance. |
| Updates / rebuild / retirement | `lib/shard/src/optimize.rs::execute_optimization/build_new_segment/finish_optimization`; `SegmentBuilder::update/build`; `LmiIndex::update_vector/update_vector_raw` | Persisted LMI rejects vector insertion. Qdrant proxies/copy-on-write handle mutable data; rebuild makes new offsets and trains afresh. Finish applies changes, swaps, syncs manifest, and defers source destruction until durable flush. This source trace is not a concurrent-update crash/recovery proof. |
| Deferred promotion | `CollectionParams::get_deferred_point_id`; `SegmentOptimizer` selection when `any_has_deferred`; `SegmentBuilder::update`; `create_segment.rs` appendable-only cutoff | Immutable output drops deferred cutoff. However collection cutoff eligibility still consults effective HNSW m/payload_m, even for LMI. Global m=0 with LMI needs an explicit policy/regression in the next increment. |
| Named / mixed | `SegmentOptimizer` per-name loop; `SegmentBuilder::build` per-name loop; `segment_constructor_base/paths.rs::get_vector_index_path` | Name-specific config/storage/index directories are wired. Enum exhaustiveness alone does not establish mixed LMI/HNSW/sparse runtime correctness. Needs lifecycle acceptance, especially concurrent schema changes. |
| Universal read-only | `lib/segment/src/index/read_only/mod.rs::VectorIndexReadEnum::{preopen,open}` | Both explicitly reject LmiTrained. Needs generic UniversalReadFs state loading, reusable validation over read-only tracker/vector storage, and native routing/candidate scoring adapter for VectorStorageReadEnum/ReadOnlyQuantizedVectors. Existing native LmiIndex holds different concrete Arc handles; not a safe one-line reuse. Keep rejection. |
| Missing / corrupt state | `LmiIndex::open_trained`, `DiskState`, `MlpRouter::validate`, `LmiRoutingState::new` | Missing/JSON errors propagate; version/config/distance/dimension/count, model shape, sample ordering/bounds, posting bounds/duplicates/live coverage checked. Expand negative tests for truncated/malformed JSON, valid-but-mismatched config/dims, duplicate offsets, bucket count. |

## Selected increment

This increment delivers fresh-storage collection snapshot restoration with explicit learned-path evidence and systematic structural corruption rejection tests. Preserve format and production behavior. Verify the existing A–E tests and locked builds/checks. This is an E2 subset, not completion of all lifecycle requirements.

## Integrity boundary requiring a separate design

Atomic single-file state prevents normal publication from pairing a separately written model and postings; whole-directory publication associates it with the target storage. It does **not** detect replacement with a different structurally valid same-shaped state or valid-shaped altered weights/bucket assignments. No generation UUID/content binding exists. Therefore the strongest arbitrary-corruption/generation invariant is not established. Address binding with an explicit format compatibility/migration decision; do not quietly claim structural validation authenticates content. Preserve Phase E files in this increment.


## Implemented and observed

No production LMI code, persistence format, dependencies, or configuration behavior changed. The audited native lifecycle already supported this bounded case; E2.1 adds executable acceptance and negative regression coverage rather than another implementation of that lifecycle.

- `tests/lmi_phase_e_http_smoke.py`: retains Phase E creation/config rejection/training/query/snapshot/restart assertions; adds stopped-source CLI snapshot recovery into a fresh temporary storage directory, state-hash multiplicity comparison, complete query-row equality (including IDs/scores), explicit native learned open/search log checks, and absence of any LMI build marker. Checks both HTTP and gRPC ports before launching isolated processes. All launched processes are stopped in cleanup.
- `lib/segment/tests/lmi_phase_e.rs`: extends the existing corruption test from 7 to 15 cases; each invokes the real native segment loader and requires an error. Prints the rejected case name. Reuses production validation unchanged.
- `docs/lmi-phase-e2.md`: source-level audit, observations, verification commands, boundaries, and next-step recommendation.

### Snapshot evidence

Run: 2026-09-25. Source storage `/tmp/qdrant-lmi-phase-e-yo8iyw1w`; restored storage `/tmp/qdrant-lmi-phase-e2-restored-0h6ejz3_`. The harness asserts the new storage directory does not exist before restoration. The source/restart process has exited before target launch. Only the snapshot is supplied to the target using `--snapshot PATH:phase_e`; the source storage is not copied.

300 indexed vectors; positive and negative learned queries each return 150 disjoint candidates. Default/empty params preserve learned routing. Exact and filtered requests return all 300, preserving IDs and scores across restore. Configuration, including LMI settings, is identical. Snapshot size: 121856 bytes.

The one state file has identical SHA-256 before and after restore:

`d04e1857439cc490e6a5c15f90d1849beb864097ea548f7e506b8eebf31874dd`

Target log `server-3.log` contains:

```text
LMI open: mode=StaticLearned; no training
[LMI-SCAFFOLD] candidate_source=StaticLearned candidate_count=150
```

There are three learned search markers (positive, negative, empty params), and no `LMI build:` marker in the entire target lifetime. Exact/filter requests follow the existing fallback branches in `lib/segment/src/index/lmi_index/read.rs`, emit no learned candidate marker, and return the full corpus. Thus the restore claim is supported by artifacts, request results, and actual native serving-path evidence, not result equality alone. This proves no training for this acceptance run, not a universal guarantee over untested recovery scenarios.

Local evidence: `work/phase_e2/http/result.json`, `http/server-{1,2,3}.log`, plus per-command `.json` status records and `.log` outputs. Temporary database directories are diagnostic artifacts and may disappear; the logs are retained in the task workspace.

### Corruption cases verified

All 15 rejected by native open: unsupported version; out-of-range posting; missing file; omitted live-vector coverage; invalid sample offset; malformed model weights/shape; absent model on a non-tiny segment; malformed JSON; truncated JSON; valid but mismatched config seed; dimension mismatch; distance mismatch; total-vector-count mismatch; duplicate posting across buckets while retaining coverage; extra empty bucket while retaining all offsets. The successful normal/restart/restore tests provide positive controls. These are 15 scenarios inside one Rust test, not 15 additional test functions.

## Exact verification results

Every command below exited 0. Existing LMI suites: candidate scoring 2, dummy 1, Phase C 10, Phase D 12, Phase E 10 = **35 passed, 0 failed**. Selected unit tests: Lloyd/cancellation 1, collection exclusivity 2, gRPC configuration 1 = **4 passed, 0 failed**. One extended isolated HTTP acceptance passed. No full-workspace test-suite claim is made.

| Command | Result | Seconds |
| --- | --- | ---: |
| `cargo fmt --all` | PASS (exit 0) | 3.11 |
| `cargo check -p collection --tests --features segment/lmi-training --locked` | PASS (exit 0) | 10.55 |
| `cargo check -p edge --locked` | PASS (exit 0) | 3.08 |
| `cargo check -p segment --tests --features lmi-training --locked` | PASS (exit 0) | 68.72 |
| `cargo test -p segment --features lmi-training --locked --test lmi_candidate_scoring --test lmi_dummy --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e -- --nocapture` | PASS (exit 0) | 14.77 |
| `cargo test -p segment --features lmi-training --locked --lib lloyd_separates_clusters_and_obeys_cancellation -- --nocapture` | PASS (exit 0) | 9.93 |
| `cargo test -p collection --features segment/lmi-training --locked --lib lmi_config_ -- --nocapture` | PASS (exit 0) | 15.26 |
| `cargo test -p api --locked --lib lmi_configuration_defaults_roundtrip_and_validation -- --nocapture` | PASS (exit 0) | 8.71 |
| `cargo check --bin qdrant --locked` | PASS (exit 0) | 12.5 |
| `cargo build --bin qdrant --locked` | PASS (exit 0) | 129.65 |
| `cargo build --bin qdrant --features lmi-training --locked` | PASS (exit 0) | 15.5 |
| `python3 tests/lmi_phase_e_http_smoke.py --output /mnt/c/Users/nicoo/Documents/Codex/2026-09-21/referenced-chatgpt-conversation-this-is-an/work/phase_e2/http` | PASS (exit 0) | 4.64 |
| `git diff --check` | PASS (exit 0) | 0.01 |

Stable formatter emitted the known nightly-only configuration warnings. Its one unrelated full-text import-formatting hunk was inspected and removed; no existing E2 work was discarded. Cargo.lock is unchanged. The default build and training-enabled build both linked successfully; live restore was run with the training-enabled binary. A default-only binary serving this snapshot was not separately exercised.

Reproduction environment (same as Phase E): stable Rust, CPU PyTorch 2.5.1, optional tch 0.18.1; `LIBTORCH_USE_PYTORCH=1`, Torch environment's bin directory on PATH, `LD_LIBRARY_PATH` pointing to its `torch/lib`, `CXX=clang++`, `CXXFLAGS=-g0`, `CARGO_BUILD_JOBS=2`. Run the commands above from the repository; choose an unused HTTP/gRPC port pair and a new output directory for the Python acceptance driver. Python only orchestrates HTTP/process assertions; it does not train the model.

## Unsupported and not established

- Universal read-only / edge trained-LMI serving remains explicitly rejected at both preopen and open. No Plain substitution was introduced. A generic I/O loader plus tracker/storage/scoring adapter is needed before native routing state can be reused safely there.
- Update/rebuild/deferred and mixed/named-vector lifecycles were source-audited only in E2.1. No live concurrent update/rebuild, old-generation retirement, crash-recovery, or mixed-index acceptance is claimed. Existing earlier tests remain passing.
- Deferred eligibility still depends on effective HNSW `m/payload_m` in `CollectionParams::get_deferred_point_id`, even for LMI; a global zero-HNSW configuration can exclude LMI from deferred eligibility. This increment does not change that policy.
- Structural validation is not generation/corpus binding. A different valid same-shaped state can escape these checks. Atomic model+postings publication prevents ordinary separate-file mixing but does not authenticate arbitrary content replacement. No generation IDs, fingerprints, or format migration were added.
- REST snapshot recovery, distributed/partial snapshot recovery, recovery under concurrent updates, and universal-backend restore have not been acceptance-tested here.
- Exact/filter/nondefault-parameter/unsupported-query/quantized paths retain Phase E fallback semantics. No recall, latency, ANN-quality, or benchmark claim follows from the two-cluster fixture.

## Recommended next E2 subphase

E2.2 should resolve the deferred eligibility policy independently of HNSW settings and add optimizer-owned rebuild acceptance: old immutable trained segment plus later inserts/updates/deletes, deferred visibility before/after promotion, new segment-local postings, publication/reopen and source retirement. Include global `m=0` and per-name LMI/HNSW configurations. Keep the corpus/generation-binding decision explicitly open with a version/migration design; do not confuse structural corruption rejection with content authentication. Universal read-only serving remains a separate adapter project.

## Final working-tree scope

Base remains `7029bcd0f` on `thesis/lmi-integration`. No commit, push, or staging performed. No unrelated changes remain.

Exact `git diff --stat` (Git excludes the new untracked handoff file):

```text
 lib/segment/tests/lmi_phase_e.rs | 53 +++++++++++++++++++++++++++++++++++++++-
 tests/lmi_phase_e_http_smoke.py  | 47 +++++++++++++++++++++++++++++++----
 2 files changed, 94 insertions(+), 6 deletions(-)
```

Additional new file: `docs/lmi-phase-e2.md` (this handoff). Total changed/new files: 3.
