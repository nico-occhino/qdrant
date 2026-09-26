# Phase E2 final engineering handoff

Date: 2026-09-25. Base: `a807381fb test(lmi): validate snapshot restore and corrupt state rejection`, branch `thesis/lmi-integration`; initial working tree was clean. E2.1 remains the historical snapshot/corruption checkpoint. No commits, staging, or pushes were performed.

## Verdict

**Phase E2 complete for the specified acceptance scope.** All ten requested criteria are demonstrated below. This does not mean arbitrary corruption authentication, distributed crash testing, performance qualification, or support for every universal filesystem deployment. Those remain explicit limitations.

## Lifecycle architecture and source map

| Stage | File / function | Responsibility |
| --- | --- | --- |
| Deferred eligibility | `lib/collection/src/config.rs::CollectionParams::get_deferred_point_id` | LMI configuration independently enables eligibility; existing HNSW m/payload_m semantics remain for other vectors. Threshold disabled still means no deferred cutoff. |
| Configured physical index | `lib/shard/src/optimizers/segment_optimizer.rs` index selection | Selects per-name LmiTrained or HNSW at indexing/deferred-promotion time. Mutable segments retain Qdrant's ordinary update path. |
| Merge / offsets | `lib/shard/src/optimize.rs::{execute_optimization,build_new_segment}` → `lib/segment/src/segment_constructor/segment_builder.rs::{update,build}` | Copies/deduplicates source vectors into target storage and target-local offsets. Trains on the target; no old router/postings installation. |
| Promotion | `segment_constructor_base/create_segment.rs::create_segment` | Cutoff retained only for appendable segments; immutable optimized output exposes promoted points. |
| Training / persistence | `segment_constructor_base/vector_index.rs::build_vector_index` → `index/lmi_index/build.rs::build_trained` | CPU trainer reads target Qdrant VectorStorage, exports native model, classifies target offsets, atomically writes one model/postings state. |
| Publication / retirement | `SegmentBuilder::build`; `lib/shard/src/optimize.rs::finish_optimization` | Saves metadata/version, renames, native opens; proxy changes applied; holder manifest synchronized; source retirement tied to durable flush. |
| Native open | `segment_constructor_base/segment.rs::load_segment` → `vector_index.rs::open_vector_index` → `LmiIndex::open_trained` | Parses persisted state and calls shared validation. No trainer call. |
| Snapshot | `lib/collection/src/collection/snapshots.rs`; `shards/local_shard/snapshot.rs`; `lib/segment/src/segment/snapshot.rs::snapshot_files`; `lmi_index/lifecycle.rs::{files,immutable_files}` | Saves collection config, native index state, vectors/IDs and segment metadata together. |
| Restore | `src/snapshots.rs::recover_snapshots/recover_snapshot_mappings` → `Collection::restore_snapshot` → `LocalShard::restore_snapshot` → `lib/shard/src/snapshots/snapshot_utils.rs::restore_unpacked_snapshot` → `lib/segment/src/segment/segment_ops.rs::restore_snapshot_in_place` → native open | Fresh isolated target consumes the snapshot; learned state is loaded, not regenerated. |
| Universal open | `lib/segment/src/segment/read_only/lifecycle.rs` → `index/read_only/mod.rs::VectorIndexReadEnum::{preopen,open}` → `lmi_index/read_only.rs::ReadOnlyLmiIndex::open` | Prefetches/reads JSON through CachedReadFs/UniversalReadFs. Shares tracker and vector storage; invokes the same validator as native open. |
| Universal serving | `ReadOnlyLmiIndex::search` → native `MlpRouter` / `LmiRoutingState::candidates_for_query` → `BatchFilteredSearcher` | Per-query postings, authoritative tracker visibility and point/context/vector deletion, Qdrant scoring/top-k. No duplicate corpus or independent distance engine. |

## Production changes by file

- `lib/collection/src/config.rs`: eligible when `lmi_config.is_some()` OR existing effective HNSW m/payload_m enables indexing. Includes a unit regression for LMI with both HNSW values zero, disabled threshold, HNSW payload-only indexing, and normal HNSW indexing.
- `lib/segment/src/index/lmi_index/build.rs`: extracts `DiskState::validate` over `IdTrackerRead` and `VectorStorageRead`. Native and universal open share version/config/distance/dimension/count, sample, model shape, posting bounds/duplicates/coverage, and tiny fallback checks. Persisted version and JSON format unchanged.
- `lib/segment/src/index/lmi_index/read_only.rs` (new): minimal adapter over `ReadOnlyPlainVectorIndex`'s shared storage view. Native routing with metric preprocessing, per-query candidates, BatchFilteredSearcher validity/scoring. Exact/filter/nondefault parameters/quantization and unsupported query shapes use existing Plain semantics. Only validated explicitly saved tiny-corpus state selects exact mode for absent model; missing/corrupt trained state errors before adapter construction.
- `lib/segment/src/index/lmi_index/mod.rs`: exposes read-only adapter module.
- `lib/segment/src/index/read_only/mod.rs`: adds actual Lmi enum variant, state prefetch/open, search/statistics dispatch and RAM-resident state cache semantics. Trained LMI is no longer mapped to an unsupported error. The legacy transient Lmi fixture branch is unchanged.
- `lib/segment/src/index/read_only/live_reload.rs`: trained LMI is immutable; deletion-only reload uses shared authoritative masks. Appended-point reload explicitly errors, requiring a new segment rather than silently using incomplete postings. Normal trained segment lifecycle does not append to that segment.

No changes to optimizer training architecture, sample generation, MLP design, lockfile, dependencies, persisted format, or operational Python training. Python remains only an HTTP/process test driver.

## Deferred-policy decision

LMI is an indexed physical vector configuration even when HNSW is disabled. It therefore participates in the existing byte-threshold calculation independently of HNSW enablement. The existing minimum cutoff across eligible vector dimensions and threshold enable/disable semantics are retained. A solo named LMI collection with global `m=0,payload_m=0` now demonstrably defers points while optimizers are paused and promotes them when optimization resumes. This closes the E2.1 HNSW dependency.

## Update / rebuild and mixed-vector acceptance

New `tests/lmi_phase_e2_lifecycle.py` uses ordinary REST operations and optimizer controls, with isolated temporary storage/ports. It runs two collections: `solo` (named LMI only) and `mixed` (named LMI `learned`, named HNSW `graph` with per-vector m=8). Both have global m=0,payload_m=0, prevent_unoptimized enabled, 1KB indexing threshold, and normal optimizer-owned builds.

Each starts with 300 vectors and one trained LMI. Optimization is paused; another 300 vectors are inserted with wait=false. Partial visibility confirms applied mutable writes and hidden deferred points. The test flips vector signs for 20 existing IDs and deletes 30 others, then resumes optimization. It waits for new segment state paths, disappearance of the old learned-state paths, green status, and exactly the expected 570 live IDs. Original persisted files are checked unchanged while paused. Rebuilt model weights and artifact hashes differ from the old generation; postings are unique and bounded by each target storage's total, and cover 570 rows. Two learned queries jointly cover exactly the expected live IDs, have disjoint candidates, and produce scores matching the current vectors (including replacements). Deleted values do not leak; newly inserted points are searchable.

For `mixed`, physical segment configuration independently says LmiTrained and HNSW; `graph.bin` exists for HNSW and there is no LMI state under its index directory. Per-vector API settings remain unchanged. The same two collections are reopened by a fresh server process; rebuilt state hashes and query IDs/scores are identical and the restart log contains native learned open/search markers with no `LMI build:` invocation. This is segment rebuild, not online learning.

Observed lifecycle evidence:

```json
{
  "storage": "/tmp/qdrant-lmi-phase-e2-lifecycle-b_5x_0mp",
  "cases": [
    {
      "name": "solo",
      "global_hnsw_m": 0,
      "deferred_points_hidden": true,
      "visible_before_promotion": 428,
      "old_state_immutable": true,
      "old_state_retired": true,
      "new_state_hashes": {
        "/tmp/qdrant-lmi-phase-e2-lifecycle-b_5x_0mp/storage/collections/solo/0/segments/cbd46bcc-865c-40a5-af0a-4d2ab30821a2/vector_index-learned/lmi_state.json": "a34c92bd28f9d22f0beabcf7e9192d003b236b916c7a8d876be7628f6715b61f"
      },
      "expected_live_points": 570,
      "new_models": true,
      "postings_unique_in_target_range": true,
      "updated_scores_correct": true,
      "deleted_ids_absent": true,
      "named_indexes_independent": false
    },
    {
      "name": "mixed",
      "global_hnsw_m": 0,
      "deferred_points_hidden": true,
      "visible_before_promotion": 428,
      "old_state_immutable": true,
      "old_state_retired": true,
      "new_state_hashes": {
        "/tmp/qdrant-lmi-phase-e2-lifecycle-b_5x_0mp/storage/collections/mixed/0/segments/dc43ae87-d25d-4206-80ad-852416c7b309/vector_index-learned/lmi_state.json": "a34c92bd28f9d22f0beabcf7e9192d003b236b916c7a8d876be7628f6715b61f"
      },
      "expected_live_points": 570,
      "new_models": true,
      "postings_unique_in_target_range": true,
      "updated_scores_correct": true,
      "deleted_ids_absent": true,
      "named_indexes_independent": true
    }
  ],
  "restart_no_training": true,
  "restart_same_state_ids_scores": true
}
```

## Universal read-only evidence

The new integration test opens the actual trained segment with `ReadOnlySegment<MmapFile>::open` through the universal filesystem abstraction. It asserts the physical enum is Lmi, not Plain, and compares independently routed two-query batches with native LMI. Logs contain `[LMI-READ-ONLY] candidate_source=StaticLearned`. Candidate pruning is required, preventing Plain equivalence alone from satisfying the test.

Dot, Cosine, Euclid, and Manhattan pass, after a persisted point deletion. Default params preserve learned routing; exact/filter and recommendation-query fallback match native behavior; top=0 returns empty results. State bytes remain unchanged. Existing tiny/empty persisted exact fallback fixtures also open and serve through the universal adapter. No training method exists on the adapter; it imports the native routing state and shared validation only.

All 15 E2.1 corruption scenarios now execute against both native and universal segment openers: version; posting offset; missing file; omitted live coverage; invalid sample offset; malformed model shape; removed model on non-tiny corpus; malformed JSON; truncated JSON; config seed mismatch; dimension; distance; total-vector count; duplicate posting across buckets; extra bucket. Every case errors. No second validation layer was introduced.

## Snapshot / restart regression

The unchanged E2.1 harness passed again on the new server build: optimizer training → learned queries → snapshot → source stopped → genuinely fresh target storage → native learned open/search. Complete configuration, artifact hashes and IDs/scores match. Exact/filter retain full-corpus behavior. Target log has no LMI build marker.

Observed source/target state hash: `d04e1857439cc490e6a5c15f90d1849beb864097ea548f7e506b8eebf31874dd`.
Source storage: `/tmp/qdrant-lmi-phase-e-7ls44tj_`; restored storage: `/tmp/qdrant-lmi-phase-e2-restored-kh1kb7mu`.
Learned positive/negative candidate counts: 150/150; exact/filter counts: 300/300. `restore_static_learned_open`, `restore_static_learned_search`, `restore_did_not_build_or_train`, and every other restore assertion are true.

## Verification

Final suite: **36 LMI integration tests passed, 0 failed** (2 candidate scoring + 1 dummy + 10 C + 12 D + 11 E including universal E2). **5 selected unit tests passed, 0 failed** (Lloyd/cancellation 1; collection config/deferred 3; gRPC 1). Corruption matrix: 15 scenarios × 2 openers, within one integration test. Both live acceptance drivers passed; lifecycle driver covers solo and mixed collections. Checks/builds are not counted as test functions.

| Exact command | Final result | Seconds |
| --- | --- | ---: |
| `cargo fmt --all` | PASS, exit 0 | 2.98 |
| `cargo check -p collection --tests --features segment/lmi-training --locked` | PASS, exit 0 | 17.41 |
| `cargo check -p edge --locked` | PASS, exit 0 | 6.97 |
| `cargo check -p segment --tests --features lmi-training --locked` | PASS, exit 0 | 10.7 |
| `cargo test -p segment --features lmi-training --locked --test lmi_candidate_scoring --test lmi_dummy --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e -- --nocapture` | PASS, exit 0 | 14.79 |
| `cargo test -p segment --features lmi-training --locked --lib lloyd_separates_clusters_and_obeys_cancellation -- --nocapture` | PASS, exit 0 | 41.32 |
| `cargo test -p collection --features segment/lmi-training --locked --lib lmi_config_ -- --nocapture` | PASS, exit 0 | 48.54 |
| `cargo test -p api --locked --lib lmi_configuration_defaults_roundtrip_and_validation -- --nocapture` | PASS, exit 0 | 21.35 |
| `cargo check --bin qdrant --locked` | PASS, exit 0 | 19.97 |
| `cargo build --bin qdrant --features lmi-training --locked` | PASS, exit 0 | 23.46 |
| `python3 tests/lmi_phase_e_http_smoke.py --port 16633 --output /mnt/c/Users/nicoo/Documents/Codex/2026-09-21/referenced-chatgpt-conversation-this-is-an/work/phase_e2/final-snapshot` | PASS, exit 0 | 6.02 |
| `python3 tests/lmi_phase_e2_lifecycle.py --port 16733 --output /mnt/c/Users/nicoo/Documents/Codex/2026-09-21/referenced-chatgpt-conversation-this-is-an/work/phase_e2/final-lifecycle` | PASS, exit 0 | 5.31 |
| `git diff --check` | PASS, exit 0 | 0.01 |

Stable rustfmt emits the known nightly-only option warnings. The unrelated full-text import-formatting hunk it generated was inspected and removed. Default server and edge compile without the training feature; training-enabled server linked and ran the live acceptance. No full workspace suite or default-only live restore is claimed.

During development: first shared-validator extraction compile exposed one leftover identifier, fixed before tests. First lifecycle test assumed external ID ordering; Qdrant offset assignment does not guarantee it, so the test was corrected to assert partial visibility by cardinality. A retry encountered an unavailable port and exited before launching; a fresh port pair was used. These are recorded in the workspace logs; final results above are all passing.

Reproduce from repository root with the commands above. Environment as Phase E: stable Rust; CPU Torch 2.5.1/tch 0.18.1; `LIBTORCH_USE_PYTORCH=1`, Torch Python environment bin on PATH, `LD_LIBRARY_PATH` pointing at torch/lib, `CXX=clang++`, `CXXFLAGS=-g0`, `CARGO_BUILD_JOBS=2`. Choose fresh test output directories and unused HTTP/gRPC port pairs. Logs/status JSON are retained in the task workspace `work/phase_e2/final-*`; temporary storage is diagnostic and may disappear.

## Acceptance checklist

- [x] Fresh snapshot restore opens and serves learned state without training.
- [x] Missing/corrupt state errors in native and universal opening.
- [x] LMI deferred eligibility independent of HNSW enablement.
- [x] Ordinary insert/update/delete drives optimizer-owned rebuild.
- [x] New target generation has newly trained model and target-local postings.
- [x] Rebuilt state survives restart, with learned serving and no training.
- [x] Named LMI/HNSW preserve independent physical configuration/artifacts.
- [x] Universal read-only backend opens and serves trained native LMI.
- [x] No trained-to-Plain lifecycle substitution; only documented query/tiny-state fallback.
- [x] Existing Phase A–E/E2.1 checks remain passing.

## Remaining limitations

Structural validation still does not authenticate a manually substituted, valid same-shaped state against its exact corpus/segment generation. Atomic state plus ordinary Qdrant publication/rebuild ownership handles the tested normal lifecycle; no cryptographic fingerprint, generation ID, or format migration was added.

Universal acceptance uses MmapFs through the generic interface; remote/object-store adapters and edge end-user workflows were not separately exercised. The index remains RAM-resident after load. Append-to-trained-segment live reload is explicitly unsupported; callers must follow normal new-segment publication. The new append rejection is defensive and not a substitute for lifecycle rebuilding. Distributed recovery, crash injection mid-publication/flush, concurrent schema changes, and exhaustive quantization configurations were not tested. Existing query fallbacks and approximate recall limitations remain. Telemetry continues using the Plain aggregates as in Phase E; marker logs are correctness evidence, not performance telemetry.

Next work can proceed to Phase F evaluation on fixed saved artifacts. Keep any remote-backend qualification or crash-stress campaign separate from model/ANN experiments. No benchmark or planner claims are made here.

## Final diff scope

Exact tracked `git diff --stat` (untracked files are excluded by Git):

```text
 lib/collection/src/config.rs                   |  33 +++++-
 lib/segment/src/index/lmi_index/build.rs       |  94 ++++++++++-------
 lib/segment/src/index/lmi_index/mod.rs         |   1 +
 lib/segment/src/index/read_only/live_reload.rs |   8 +-
 lib/segment/src/index/read_only/mod.rs         |  40 ++++++--
 lib/segment/tests/lmi_phase_e.rs               | 136 ++++++++++++++++++++++++-
 6 files changed, 261 insertions(+), 51 deletions(-)

```

New files additionally: `lib/segment/src/index/lmi_index/read_only.rs`, `tests/lmi_phase_e2_lifecycle.py`, `docs/lmi-phase-e2-final.md` (this document). No unrelated edits remain. No commit or push performed.
