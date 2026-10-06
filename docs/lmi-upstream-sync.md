# LMI forward-port onto Qdrant 1.19.2

Status: in progress. This branch is not validated for thesis development.

- Historical frozen LMI SHA: `4e0527388b98f16623ab3e5124baab4cd6022b87`
- Historical branch: `thesis/lmi-integration`
- Upstream baseline SHA: `016542aa5deb6c66380bb137badf73d54f742bde`
- Merge base: `74f3e85b9473c62560006c043e13737ce6b48412`
- Before freeze: upstream 438 commits ahead of merge base; thesis 22.
- After freeze: upstream 438; thesis 25.
- Strategy: port final LMI architecture into an isolated worktree; do not rebase the historical branch or replay superseded prototypes.

## Baseline validation

Rust 1.98.0 and Cargo 1.98.0. Pure upstream `cargo check -p segment -p collection --locked` passed. Pure upstream segment HNSW-filtered unit selection passed 73 tests with one ignored. The initial `plain` name filter found one passing unit test and is not a full Plain search regression.

## Initial conflict audit

A non-working-tree `git merge-tree --write-tree` found textual conflicts in `Cargo.lock`, `lib/segment/Cargo.toml`, HNSW `dispatch.rs`, collection config mismatch and merge optimizers, collection `optimizers_builder.rs`, edge vector config, and shard optimizer config and segment optimizer. The full 26-path shared-file list and exact commit provenance are retained outside this repository at `/home/nicoo/work/qdrant-sync-safety-2026-10-06/upstream-conflict-matrix.md`.

Clean textual merges do not validate configuration serialization, scoring, or segment lifecycle. These contracts require independent review.

## Current Qdrant contract review

| Contract | Initial comparison | Evidence and required action |
|---|---|---|
| `VectorIndexRead` / `VectorIndex` | Unchanged signatures | Current `lib/segment/src/index/vector_index_base.rs` still requires `search`, telemetry, indexed count, searchable bytes, IDF contribution, `is_index`, files and raw/decoded update methods. |
| `VectorIndexEnum` | Same single-index model | Current enum has Plain, HNSW and sparse variants; add LMI deliberately and cover every exhaustive match. |
| `VectorDataConfig` / `Indexes` | Review pending | Current `lib/segment/src/types.rs`; validate persisted representation before adding LMI. |
| Candidate scorer and `RawScorer` | Semantics changed upstream | Batched deletion checks, batched HNSW search and graph-inline storage commits changed scorer-related code. Re-derive the candidate seam before porting. |
| HNSW filter dispatch | Semantics changed upstream | Upstream commits `a764bfe6a`, `142227719`, `47386a410`; historical instrumentation overlaps textually. Keep current upstream planner as baseline. |
| Vector storage and constructor | Semantics changed upstream | Combined-storage commit `0d6cd45e3` changed vector-index construction. Do not transplant old constructor paths. |
| Optimizer and segment publication | Review pending | Several textual conflicts. Postings must be rebuilt for each target segment's offsets. |
| Memory reporting | Review pending | Historical LMI hook and upstream graph-inline storage accounting require separate review. |
| Snapshots, read-only open, telemetry | Review pending | Verify current file enumeration and loaded-index semantics. |
| Cargo manifests / lockfile | Textual conflict | Keep upstream lockfile and add only required LMI dependencies/features. |

## Validation still required

Layered LMI compile and test gates; Plain/HNSW regressions; persistence and optimizer lifecycle; Float16 and routing parity; Phase G semantics; SISAP300K real-data comparison; old persisted format compatibility; bounded 10.12M read-only confirmation if compatible. No synchronized branch push until these gates pass.

## Historical science

The accepted 300K, 10.12M and Phase G measurements remain historical results on the frozen branch. No result has yet been revalidated on this synchronized branch.
# Upstream / historical LMI conflict matrix

Merge base: `74f3e85b9473c62560006c043e13737ce6b48412`
Upstream: `016542aa5deb6c66380bb137badf73d54f742bde`
Frozen thesis: `4e0527388b98f16623ab3e5124baab4cd6022b87`

The paths below changed on both branches. Textual conflict means a non-working-tree `git merge-tree --write-tree` reported content conflict. A clean simulated merge is not proof of semantic compatibility.

| Path | Textual | Semantic risk |
|---|---|---|
| `Cargo.lock` | YES | HIGH: resolve against current upstream contract |
| `Cargo.toml` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/api/src/grpc/conversions.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/api/src/grpc/proto/collections.proto` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/api/src/grpc/qdrant.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/collection/src/collection/vector_name_schema.rs` | No in simulated merge | REVIEW: inspect merged behavior |
| `lib/collection/src/collection_manager/optimizers/config_mismatch_optimizer.rs` | YES | HIGH: resolve against current upstream contract |
| `lib/collection/src/collection_manager/optimizers/indexing_optimizer.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/collection/src/collection_manager/optimizers/merge_optimizer.rs` | YES | HIGH: resolve against current upstream contract |
| `lib/collection/src/collection_manager/segments_searcher.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/collection/src/config.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/collection/src/operations/conversions.rs` | No in simulated merge | REVIEW: inspect merged behavior |
| `lib/collection/src/operations/types.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/collection/src/optimizers_builder.rs` | YES | HIGH: resolve against current upstream contract |
| `lib/edge/src/config/shard.rs` | No in simulated merge | REVIEW: inspect merged behavior |
| `lib/edge/src/config/vectors.rs` | YES | HIGH: resolve against current upstream contract |
| `lib/segment/Cargo.toml` | YES | HIGH: resolve against current upstream contract |
| `lib/segment/src/compat.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/segment/src/index/hnsw_index/hnsw/read_view/dispatch.rs` | YES | HIGH: resolve against current upstream contract |
| `lib/segment/src/index/read_only/live_reload.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/segment/src/index/read_only/mod.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/segment/src/segment_constructor/segment_constructor_base/vector_index.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/segment/src/types.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/shard/src/optimizers/config.rs` | YES | HIGH: resolve against current upstream contract |
| `lib/shard/src/optimizers/config_mismatch_optimizer.rs` | No in simulated merge | HIGH: clean merge still needs API/lifecycle review |
| `lib/shard/src/optimizers/segment_optimizer.rs` | YES | HIGH: resolve against current upstream contract |

## Commit provenance for every shared path

### `Cargo.lock`

Upstream commits:

- 016542aa5 2026-10-05T12:03:16+02:00 Bump version to 1.19.2 (#10946)
- e150553d1 2026-10-05T10:16:42+02:00 Prevent write tear in Gridstore mappings flusher (#10837)
- 39f9d2274 2026-10-05T10:16:42+02:00 Fix clippy lints for Rust 1.99 and beta (#10883)
- e661fd4c8 2026-10-05T10:14:31+02:00 build(deps): bump smallvec from 1.16.1 to 1.16.2 (#10817)
- 90eab164e 2026-10-05T10:14:31+02:00 build(deps): bump count-min-sketch from 0.1.8 to 0.2.0 (#10816)
- e41fbeb68 2026-10-05T10:14:30+02:00 build(deps): bump memchr from 2.8.0 to 2.8.3 (#10819)
- 5ff6fc28c 2026-10-05T10:14:30+02:00 build(deps): bump thiserror from 2.0.20 to 2.0.21 (#10820)
- 2f5162c62 2026-10-05T10:14:30+02:00 build(deps): bump rand from 0.10.2 to 0.10.3 (#10823)
- 0b9df79e9 2026-10-05T10:14:30+02:00 build(deps): bump siphasher from 1.0.3 to 1.0.4 (#10824)
- 15cf5291f 2026-10-05T10:14:29+02:00 build(deps): bump zerocopy from 0.8.57 to 0.8.59 (#10825)
- e74a60d6f 2026-10-05T10:14:29+02:00 build(deps): bump uniffi from 0.32.1 to 0.32.2 (#10822)
- 266be528c 2026-10-05T10:14:29+02:00 build(deps): bump uuid from 1.26.0 to 1.26.1 (#10821)
- cacff7e0a 2026-10-05T10:14:28+02:00 Compact versions file for the disk id tracker in serverless mode (#10803)
- cbeebe3e3 2026-10-05T10:12:53+02:00 Add CompactedTracker: RAM-resident Logstore tracker with a compact file (#10737)
- 2020d186c 2026-10-05T10:12:52+02:00 Meter remote requests and expose IO statistics to edge shard users (#10769)
- 51be46845 2026-10-05T10:12:50+02:00 [BM25] Add a BM25-over-sparse baseline benchmark (#10677)
- bedab3c8e 2026-10-05T10:12:49+02:00 Add `match: { substring }` filter condition (#10711)
- d79359519 2026-10-05T10:12:48+02:00 build(deps): bump tinyvec from 1.13.2 to 1.13.3 (#10728)
- ce2e532c6 2026-10-05T10:12:48+02:00 build(deps): bump dial9 from 0.5.0 to 0.5.1 (#10731)
- 14f5692a7 2026-10-05T10:12:48+02:00 fix(deps): repair Cargo.lock after concurrent syn and serde_with bumps (#10734)
- 0a43b000f 2026-10-05T10:12:47+02:00 build(deps): bump serde_with from 3.22.0 to 3.23.0 (#10732)
- 637840f4c 2026-10-05T10:12:47+02:00 build(deps): bump object_store from 0.14.1 to 0.14.2 (#10727)
- 73ced0b21 2026-10-05T10:12:47+02:00 build(deps): bump actix-multipart from 0.8.1 to 0.8.2 (#10729)
- b693260fe 2026-10-05T10:08:53+02:00 build(deps): bump syn from 3.0.5 to 3.0.6 (#10730)
- b3dafb374 2026-10-05T10:08:53+02:00 build(deps): bump rustix from 1.1.4 to 1.1.5 (#10726)
- 4c5efed4c 2026-10-05T10:08:53+02:00 build(deps): bump rustls from 0.23.44 to 0.23.45 (#10723)
- 18fdc0287 2026-10-05T10:08:52+02:00 build(deps): bump cc from 1.4.5 to 1.4.7 (#10725)
- 3e634c19a 2026-10-05T10:08:49+02:00 Fix never-ending optimization loop with `vectors:{memory:cached}` (#10664)
- 697c0ae7b 2026-10-05T10:08:47+02:00 build(deps): bump io-uring from 0.7.14 to 0.7.15 (#10648)
- 356760bce 2026-10-05T10:08:47+02:00 build(deps): bump rstest from 0.26.1 to 0.27.0 (#10653)
- 7104d8111 2026-10-05T10:06:44+02:00 build(deps): bump rustls from 0.23.43 to 0.23.44 (#10651)
- 3e370c706 2026-10-05T10:06:44+02:00 build(deps): bump smallvec from 1.16.0 to 1.16.1 (#10649)
- 43fd888d7 2026-10-05T10:06:43+02:00 build(deps): bump uniffi from 0.32.0 to 0.32.1 (#10652)
- 666c828ed 2026-10-05T10:06:43+02:00 build(deps): bump syn from 3.0.4 to 3.0.5 (#10654)
- 3896d9265 2026-10-05T10:06:43+02:00 build(deps): bump foyer from 0.22.4 to 0.22.6 (#10656)
- efe2375c1 2026-10-05T10:06:43+02:00 build(deps): bump zerocopy from 0.8.56 to 0.8.57 (#10655)
- b08ecde8e 2026-10-05T10:06:43+02:00 build(deps): bump reqwest from 0.13.4 to 0.13.5 (#10657)
- ef0dc1ad5 2026-10-05T10:06:42+02:00 Send the queue proxy batch as a pre-encoded gRPC body (#10617)
- 763c038e4 2026-10-05T10:03:50+02:00 build(deps): bump constant_time_eq from 0.5.0 to 0.6.0 (#10514)
- 9ab9ce463 2026-10-05T10:03:49+02:00 build(deps): bump foyer from 0.22.3 to 0.22.4 (#10508)
- 19cad848d 2026-10-05T10:03:49+02:00 build(deps): bump smallvec from 1.15.2 to 1.16.0 (#10509)
- e6079a8ea 2026-10-05T10:03:49+02:00 build(deps): bump tinyvec from 1.12.0 to 1.13.2 (#10512)
- b98db356e 2026-10-05T10:03:49+02:00 build(deps): bump indexmap from 2.14.0 to 2.14.1 (#10510)
- d0f05679a 2026-10-05T10:03:48+02:00 build(deps): bump ecow from 0.3.0 to 0.3.1 (#10511)
- 6e590726f 2026-10-05T10:03:48+02:00 build(deps): bump cc from 1.4.4 to 1.4.5 (#10513)
- 4058be9b6 2026-10-05T10:03:47+02:00 Remove msgpack (#10505)
- e05946bcb 2026-10-05T10:03:47+02:00 Probe large memory reports with temporary workers (#10458)
- 6ab21cac1 2026-09-03T14:36:18+02:00 Bump version to 1.19.1 (#10463)
- 4d1570c03 2026-09-03T12:46:01+02:00 feat: optional dial9 Tokio telemetry behind a `dial9` feature (#10442)
- c2aeb9011 2026-09-03T12:45:59+02:00 Update-only writer: leave optimizing targets alone and create fresh appendable segments (#10416)
- 5c37004f6 2026-09-03T12:45:57+02:00 [UIO] Split async into extension traits, implement only where genuinely async (#10424)
- 9e4748ba0 2026-09-03T12:45:57+02:00 [edge] open and reload IO don't block search pool (#10366)
- d7e41f35e 2026-09-03T12:45:57+02:00 [UIO] Segment `live_preload` waits for all IO before returning (#10357)
- 7a26b7117 2026-09-03T12:45:56+02:00 build(deps): bump tracing-tracy from 0.11.4 to 0.12.0 (#10413)
- d2c46a34c 2026-09-03T12:45:55+02:00 build(deps): bump actix-cors from 0.7.1 to 0.7.2 (#10408)
- a080e1d20 2026-09-03T12:45:55+02:00 build(deps): bump prost-wkt-types from 0.7.1 to 0.7.2 (#10415)
- 88581adc1 2026-09-03T12:45:55+02:00 build(deps): bump log from 0.4.33 to 0.4.34 (#10409)
- 185260be3 2026-09-03T12:45:55+02:00 build(deps): bump uuid from 1.24.1 to 1.26.0 (#10407)
- 6c288ecf2 2026-09-03T12:45:54+02:00 build(deps): bump flate2 from 1.1.9 to 1.1.10 (#10410)
- 4f0fc9012 2026-09-03T12:45:54+02:00 build(deps): bump syn from 3.0.3 to 3.0.4 (#10412)
- a21f535fd 2026-09-03T12:45:54+02:00 build(deps): bump actix-multipart from 0.8.0 to 0.8.1 (#10411)
- 7bdfe42aa 2026-09-03T12:45:54+02:00 build(deps): bump actix-web from 4.14.1 to 4.15.0 (#10414)
- f6e0566f3 2026-09-03T12:45:54+02:00 [UIO] `UniversalReadFs::open_async` (#10352)
- 52d72d94d 2026-09-03T12:45:53+02:00 [UIO] impl `IoUringFile::read_bytes_async` (#10288)
- 334d18020 2026-09-03T12:45:50+02:00 build(deps): disable unused default features (#10331)
- 75fe4cb2b 2026-09-03T12:45:49+02:00 build(deps): bump tango-bench from 0.7.2 to 0.8.0 (#10317)
- 2e8d61812 2026-09-03T12:42:28+02:00 build(deps): bump serde_with to 3.22.0 without default features (#10329)
- ab2035514 2026-09-03T12:42:28+02:00 build(deps): bump ordered-float from 5.3.0 to 5.5.0 (#10316)
- e79e50e8f 2026-09-03T12:42:28+02:00 build(deps): bump io-uring from 0.7.13 to 0.7.14 (#10319)
- c98203f94 2026-09-03T12:42:28+02:00 build(deps): bump uuid from 1.24.0 to 1.24.1 (#10318)
- 6d13c124f 2026-09-03T12:42:27+02:00 build(deps): bump cc from 1.4.3 to 1.4.4 (#10321)
- 4b8071428 2026-09-03T12:42:27+02:00 build(deps): bump actix-files from 0.6.10 to 0.7.0 (#10322)
- 82d48cda7 2026-09-03T12:42:26+02:00 Replace cgroups-rs with direct cgroup memory file reads (#10295)
- 7e9ec5d2f 2026-09-03T12:41:04+02:00 build(deps): bump actix-web from 4.14.0 to 4.14.1 (#10248)
- e43e9a6fc 2026-09-03T12:41:04+02:00 bump syn 3.0.3
- 851b52de5 2026-09-03T12:41:04+02:00 build(deps): bump async-trait from 0.1.91 to 0.1.92 (#10251)
- c5da3bac0 2026-09-03T12:41:04+02:00 build(deps): bump stumpalo from 0.5.1 to 1.0.0 (#10245)
- 6eda2934e 2026-09-03T12:41:03+02:00 build(deps): bump charabia from 0.9.9 to 0.10.0 (#10247)
- 4b21eace8 2026-09-03T12:41:03+02:00 build(deps): bump cc from 1.4.2 to 1.4.3 (#10249)
- 343d97403 2026-09-03T12:41:03+02:00 build(deps): bump syn from 3.0.2 to 3.0.3 (#10250)
- d343121f2 2026-09-03T12:41:03+02:00 build(deps): bump thiserror from 2.0.19 to 2.0.20 (#10252)
- d45fd9078 2026-09-03T12:41:03+02:00 build(deps): bump roaring from 0.11.4 to 0.11.5 (#10253)
- 8354711c2 2026-09-03T12:41:02+02:00 build(deps): bump futures from 0.3.33 to 0.3.34 (#10254)
- 22e2b8af9 2026-09-03T12:41:00+02:00 Replace permutation_iterator with rand's index sampler (#10217)
- 9bdd88548 2026-09-03T12:39:04+02:00 Replace sys-info dependency with already-present sysinfo (#10216)
- bcb7a45cc 2026-09-03T12:39:04+02:00 Add CachedBlobFile: cached reads + write-through appends for object stores (#10206)
- c6cf72a7c 2026-09-03T12:39:03+02:00 transfer: send raw payloads, behind feature flags (#10066)
- c0e4c4168 2026-09-03T12:38:59+02:00 Batched HNSW reader (#10054)
- d2cdaf1cf 2026-09-03T12:38:59+02:00 Add edge-tool: CLI for creating, seeding, optimizing, and uploading local edge collections (#10159)
- eec1dbc08 2026-09-03T12:38:59+02:00 build(deps): bump cc from 1.4.0 to 1.4.1 (#10176)
- 21955f881 2026-09-03T12:36:36+02:00 build(deps): bump rustls-pki-types from 1.15.0 to 1.15.1 (#10172)
- c8087b48d 2026-09-03T12:36:35+02:00 build(deps): bump zerocopy from 0.8.55 to 0.8.56 (#10177)
- 85d365841 2026-09-03T12:36:35+02:00 build(deps): bump pyo3 from 0.29.0 to 0.29.2 (#10175)
- d0a211c49 2026-09-03T12:36:35+02:00 build(deps): bump num-derive from 0.4.2 to 0.5.1 (#10174)
- daea89e11 2026-09-03T12:36:35+02:00 build(deps): bump futures from 0.3.32 to 0.3.33 (#10173)
- 32c9454ac 2026-09-03T12:36:35+02:00 build(deps): bump rustls from 0.23.42 to 0.23.43 (#10171)
- 507c9fab3 2026-09-03T12:36:34+02:00 build(deps): bump serial_test from 3.5.0 to 4.0.1 (#10169)
- 65b8b51a2 2026-09-03T12:36:34+02:00 build(deps): bump data-encoding from 2.11.0 to 2.11.1 (#10167)
- 97eff2b02 2026-09-03T12:36:31+02:00 chore(deps): drop dead dependencies in edge-path crates (#10109)
- f39b1bc6f 2026-09-03T12:36:29+02:00 Bump edge packages (Python + Rust + FFI) to 0.8.0 (#10098)
- 33a64d411 2026-09-03T12:33:34+02:00 build(deps): bump jsonwebtoken from 10.4.0 to 11.0.0 (#10078)
- 8a746a493 2026-09-03T12:33:33+02:00 build(deps): bump sysinfo from 0.38.4 to 0.39.6 (#10077)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `Cargo.toml`

Upstream commits:

- 016542aa5 2026-10-05T12:03:16+02:00 Bump version to 1.19.2 (#10946)
- cbeebe3e3 2026-10-05T10:12:53+02:00 Add CompactedTracker: RAM-resident Logstore tracker with a compact file (#10737)
- bedab3c8e 2026-10-05T10:12:49+02:00 Add `match: { substring }` filter condition (#10711)
- 3e634c19a 2026-10-05T10:08:49+02:00 Fix never-ending optimization loop with `vectors:{memory:cached}` (#10664)
- 356760bce 2026-10-05T10:08:47+02:00 build(deps): bump rstest from 0.26.1 to 0.27.0 (#10653)
- e5c21ad0a 2026-10-05T10:06:39+02:00 Refactor `OnDiskMapIndex` (#10592)
- 763c038e4 2026-10-05T10:03:50+02:00 build(deps): bump constant_time_eq from 0.5.0 to 0.6.0 (#10514)
- 4058be9b6 2026-10-05T10:03:47+02:00 Remove msgpack (#10505)
- 6ab21cac1 2026-09-03T14:36:18+02:00 Bump version to 1.19.1 (#10463)
- 4d1570c03 2026-09-03T12:46:01+02:00 feat: optional dial9 Tokio telemetry behind a `dial9` feature (#10442)
- 7a26b7117 2026-09-03T12:45:56+02:00 build(deps): bump tracing-tracy from 0.11.4 to 0.12.0 (#10413)
- 334d18020 2026-09-03T12:45:50+02:00 build(deps): disable unused default features (#10331)
- 2e8d61812 2026-09-03T12:42:28+02:00 build(deps): bump serde_with to 3.22.0 without default features (#10329)
- 4b8071428 2026-09-03T12:42:27+02:00 build(deps): bump actix-files from 0.6.10 to 0.7.0 (#10322)
- c5da3bac0 2026-09-03T12:41:04+02:00 build(deps): bump stumpalo from 0.5.1 to 1.0.0 (#10245)
- 9bdd88548 2026-09-03T12:39:04+02:00 Replace sys-info dependency with already-present sysinfo (#10216)
- c0e4c4168 2026-09-03T12:38:59+02:00 Batched HNSW reader (#10054)
- d2cdaf1cf 2026-09-03T12:38:59+02:00 Add edge-tool: CLI for creating, seeding, optimizing, and uploading local edge collections (#10159)
- 33a64d411 2026-09-03T12:33:34+02:00 build(deps): bump jsonwebtoken from 10.4.0 to 11.0.0 (#10078)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/api/src/grpc/conversions.rs`

Upstream commits:

- bedab3c8e 2026-10-05T10:12:49+02:00 Add `match: { substring }` filter condition (#10711)
- f8acc7c66 2026-09-03T12:42:25+02:00 feat: add a dedicated min operator to score formulas (#10296)
- aba9516e5 2026-09-03T12:42:25+02:00 feat: add a dedicated max operator to score formulas (#10287)
- fe27f6a0a 2026-09-03T12:41:01+02:00 Add acosh expression to formula query (#10231)
- c6cf72a7c 2026-09-03T12:39:03+02:00 transfer: send raw payloads, behind feature flags (#10066)
- 78ad7bc4b 2026-09-03T12:36:31+02:00 [Raw payloads]: read payload as stored bytes in retrieve_raw (#10040)

Thesis commits:

- be40920e9 2026-10-03T14:18:55+02:00 LMI: batch native corpus routing
- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/api/src/grpc/proto/collections.proto`

Upstream commits:

- 333ca003a 2026-10-05T10:06:40+02:00 Support `cached` id tracker memory placement (#10598)
- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- bad18437a 2026-10-05T10:03:46+02:00 [AI] docs: fix typos and duplicated words across config help and rustdoc (#10484)

Thesis commits:

- be40920e9 2026-10-03T14:18:55+02:00 LMI: batch native corpus routing
- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/api/src/grpc/qdrant.rs`

Upstream commits:

- 7477c7c36 2026-10-05T10:14:28+02:00 Fix stale snapshot transfer breaking cluster (#10643)
- bedab3c8e 2026-10-05T10:12:49+02:00 Add `match: { substring }` filter condition (#10711)
- 333ca003a 2026-10-05T10:06:40+02:00 Support `cached` id tracker memory placement (#10598)
- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- bad18437a 2026-10-05T10:03:46+02:00 [AI] docs: fix typos and duplicated words across config help and rustdoc (#10484)
- f8acc7c66 2026-09-03T12:42:25+02:00 feat: add a dedicated min operator to score formulas (#10296)
- aba9516e5 2026-09-03T12:42:25+02:00 feat: add a dedicated max operator to score formulas (#10287)
- fe27f6a0a 2026-09-03T12:41:01+02:00 Add acosh expression to formula query (#10231)
- 278140c48 2026-09-03T12:36:31+02:00 Add /profiler/consensus_lag to measure apply lag between peers (#10090)

Thesis commits:

- be40920e9 2026-10-03T14:18:55+02:00 LMI: batch native corpus routing
- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/collection/src/collection/vector_name_schema.rs`

Upstream commits:

- ac2175055 2026-09-03T12:42:26+02:00 Implement simple consensus operations on `ConsensusStateMachine` (#10280)
- a0d98c752 2026-09-03T12:42:25+02:00 Implement `ConsensusStateMachine` prototype (#10220)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/collection/src/collection_manager/optimizers/config_mismatch_optimizer.rs`

Upstream commits:

- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/collection/src/collection_manager/optimizers/indexing_optimizer.rs`

Upstream commits:

- 3e634c19a 2026-10-05T10:08:49+02:00 Fix never-ending optimization loop with `vectors:{memory:cached}` (#10664)
- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- fbbe4a5f2 2026-10-05T10:06:37+02:00 [combined-storage] Integrate `VectorStorageType::GraphInline` reading (#10515)
- 016328a3c 2026-09-03T12:42:26+02:00 Stop growing appendable segments past `max_segment_size` (#10027)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/collection/src/collection_manager/optimizers/merge_optimizer.rs`

Upstream commits:

- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/collection/src/collection_manager/segments_searcher.rs`

Upstream commits:

- 8065b9efa 2026-09-03T12:45:50+02:00 Remove stale clippy allows and the obsolete large-error-threshold override (#10337)
- ca31b7efd 2026-09-03T12:42:25+02:00 docs: parameter names in doc comments that the signatures do not have (#10290)
- 78ad7bc4b 2026-09-03T12:36:31+02:00 [Raw payloads]: read payload as stored bytes in retrieve_raw (#10040)
- 6b76d4e71 2026-09-03T12:36:30+02:00 feat(edge): add query_batch for batched planned queries (#10100)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing
- 4f3bd3798 2026-09-23T12:10:00+02:00 feat: integrate learned metric index into Qdrant

### `lib/collection/src/config.rs`

Upstream commits:

- 6994dce72 2026-10-05T10:06:41+02:00 Reject snapshot upload without collection config, without exposing the temp path (#10556)
- 333ca003a 2026-10-05T10:06:40+02:00 Support `cached` id tracker memory placement (#10598)
- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)

Thesis commits:

- e360295aa 2026-10-06T20:49:11+02:00 LMI: finalize Float16 build and configuration support
- a1987b504 2026-09-26T10:54:03+02:00 feat(lmi): complete trained index lifecycle support
- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/collection/src/operations/conversions.rs`

Upstream commits:

- 8d1005e33 2026-10-05T10:14:27+02:00 Add failed shard transfer metric (#7714)
- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/collection/src/operations/types.rs`

Upstream commits:

- a4cce6628 2026-10-05T10:14:32+02:00 Report transferred bytes for snapshot transfer (#10813)
- 8d1005e33 2026-10-05T10:14:27+02:00 Add failed shard transfer metric (#7714)
- e09d4e63c 2026-10-05T10:08:50+02:00 Implement `CreateShardKey`/`RemoveShardKey` for consensus state machine (#10666)
- 3e634c19a 2026-10-05T10:08:49+02:00 Fix never-ending optimization loop with `vectors:{memory:cached}` (#10664)
- 0faf5bf88 2026-10-05T10:08:49+02:00 fix: expose ReshardingStage in telemetry (#10618)
- 470694ae9 2026-09-03T12:45:52+02:00 Implement more operations on `ConsensusStateMachine` [1/2] (#10309)
- 93f3aafe3 2026-09-03T12:45:50+02:00 docs(schema): declare enforced 1..=65536 bound on VectorParams.size (#10324)
- c6cf72a7c 2026-09-03T12:39:03+02:00 transfer: send raw payloads, behind feature flags (#10066)

Thesis commits:

- e360295aa 2026-10-06T20:49:11+02:00 LMI: finalize Float16 build and configuration support
- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/collection/src/optimizers_builder.rs`

Upstream commits:

- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- bad18437a 2026-10-05T10:03:46+02:00 [AI] docs: fix typos and duplicated words across config help and rustdoc (#10484)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)
- 016328a3c 2026-09-03T12:42:26+02:00 Stop growing appendable segments past `max_segment_size` (#10027)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/edge/src/config/shard.rs`

Upstream commits:

- b9e0d3ae8 2026-10-05T10:16:40+02:00 feat(edge): expose unified `memory` placement on all collection components (#10863)
- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- bad18437a 2026-10-05T10:03:46+02:00 [AI] docs: fix typos and duplicated words across config help and rustdoc (#10484)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/edge/src/config/vectors.rs`

Upstream commits:

- b9e0d3ae8 2026-10-05T10:16:40+02:00 feat(edge): expose unified `memory` placement on all collection components (#10863)
- fbbe4a5f2 2026-10-05T10:06:37+02:00 [combined-storage] Integrate `VectorStorageType::GraphInline` reading (#10515)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/segment/Cargo.toml`

Upstream commits:

- cacff7e0a 2026-10-05T10:14:28+02:00 Compact versions file for the disk id tracker in serverless mode (#10803)
- bedab3c8e 2026-10-05T10:12:49+02:00 Add `match: { substring }` filter condition (#10711)
- 3e634c19a 2026-10-05T10:08:49+02:00 Fix never-ending optimization loop with `vectors:{memory:cached}` (#10664)
- 86260b933 2026-10-05T10:06:38+02:00 Remove RocksDB references (#10561)
- 4058be9b6 2026-10-05T10:03:47+02:00 Remove msgpack (#10505)
- 0337d43c1 2026-09-03T12:45:59+02:00 Turbo4 batched scan (#10362)
- d7e41f35e 2026-09-03T12:45:57+02:00 [UIO] Segment `live_preload` waits for all IO before returning (#10357)
- f6e0566f3 2026-09-03T12:45:54+02:00 [UIO] `UniversalReadFs::open_async` (#10352)
- 334d18020 2026-09-03T12:45:50+02:00 build(deps): disable unused default features (#10331)
- 82d48cda7 2026-09-03T12:42:26+02:00 Replace cgroups-rs with direct cgroup memory file reads (#10295)
- 6eda2934e 2026-09-03T12:41:03+02:00 build(deps): bump charabia from 0.9.9 to 0.10.0 (#10247)
- 9bdd88548 2026-09-03T12:39:04+02:00 Replace sys-info dependency with already-present sysinfo (#10216)
- c0e4c4168 2026-09-03T12:38:59+02:00 Batched HNSW reader (#10054)
- d0a211c49 2026-09-03T12:36:35+02:00 build(deps): bump num-derive from 0.4.2 to 0.5.1 (#10174)
- 97eff2b02 2026-09-03T12:36:31+02:00 chore(deps): drop dead dependencies in edge-path crates (#10109)
- 8a746a493 2026-09-03T12:33:33+02:00 build(deps): bump sysinfo from 0.38.4 to 0.39.6 (#10077)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/segment/src/compat.rs`

Upstream commits:

- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- 86260b933 2026-10-05T10:06:38+02:00 Remove RocksDB references (#10561)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/segment/src/index/hnsw_index/hnsw/read_view/dispatch.rs`

Upstream commits:

- 47386a410 2026-10-05T10:16:40+02:00 Report which algorithm ran a filtered graph search (#10781)
- 142227719 2026-10-05T10:06:41+02:00 Resolve filter ids once per filtered search (#10624)
- a764bfe6a 2026-09-03T12:41:05+02:00 Integrate batched HNSW (#10194)

Thesis commits:

- 6e3cfdb0f 2026-09-26T17:29:40+02:00 test(lmi): add controlled ANN evaluation framework
- a628691f6 2026-09-23T12:09:28+02:00 research: instrument HNSW search dispatch

### `lib/segment/src/index/read_only/live_reload.rs`

Upstream commits:

- d7e41f35e 2026-09-03T12:45:57+02:00 [UIO] Segment `live_preload` waits for all IO before returning (#10357)
- 76d18c319 2026-09-03T12:42:24+02:00 [LiveReload] Preload indexes (#10229)
- 82ce673c5 2026-09-03T12:39:00+02:00 [LiveReload] Add `live_preload` (#10036)

Thesis commits:

- a1987b504 2026-09-26T10:54:03+02:00 feat(lmi): complete trained index lifecycle support

### `lib/segment/src/index/read_only/mod.rs`

Upstream commits:

- 9cfbe82b0 2026-09-03T12:45:54+02:00 `schedule_open` returns nothing (#10355)
- 0e6845187 2026-09-03T12:45:53+02:00 [UIO] renames + enforce `LiveReload::live_preload` (#10351)

Thesis commits:

- a1987b504 2026-09-26T10:54:03+02:00 feat(lmi): complete trained index lifecycle support
- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing
- 4f3bd3798 2026-09-23T12:10:00+02:00 feat: integrate learned metric index into Qdrant

### `lib/segment/src/segment_constructor/segment_constructor_base/vector_index.rs`

Upstream commits:

- 0d6cd45e3 2026-10-05T10:08:50+02:00 [combined-storage] Combined storage write (#10669)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing
- 4f3bd3798 2026-09-23T12:10:00+02:00 feat: integrate learned metric index into Qdrant

### `lib/segment/src/types.rs`

Upstream commits:

- 281d146a1 2026-10-05T10:16:39+02:00 Reject full-text index with min_token_len > max_token_len (#10862)
- bedab3c8e 2026-10-05T10:12:49+02:00 Add `match: { substring }` filter condition (#10711)
- 3e634c19a 2026-10-05T10:08:49+02:00 Fix never-ending optimization loop with `vectors:{memory:cached}` (#10664)
- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- 86260b933 2026-10-05T10:06:38+02:00 Remove RocksDB references (#10561)
- fbbe4a5f2 2026-10-05T10:06:37+02:00 [combined-storage] Integrate `VectorStorageType::GraphInline` reading (#10515)
- 4058be9b6 2026-10-05T10:03:47+02:00 Remove msgpack (#10505)
- b40cf6de9 2026-10-05T10:02:14+02:00 Remove unused (Sparse)VectorStorageType::Empty (#10431)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)
- bcb7a45cc 2026-09-03T12:39:04+02:00 Add CachedBlobFile: cached reads + write-through appends for object stores (#10206)
- c6cf72a7c 2026-09-03T12:39:03+02:00 transfer: send raw payloads, behind feature flags (#10066)
- 78ad7bc4b 2026-09-03T12:36:31+02:00 [Raw payloads]: read payload as stored bytes in retrieve_raw (#10040)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing
- 4f3bd3798 2026-09-23T12:10:00+02:00 feat: integrate learned metric index into Qdrant

### `lib/shard/src/optimizers/config.rs`

Upstream commits:

- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)
- 016328a3c 2026-09-03T12:42:26+02:00 Stop growing appendable segments past `max_segment_size` (#10027)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing

### `lib/shard/src/optimizers/config_mismatch_optimizer.rs`

Upstream commits:

- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- fbbe4a5f2 2026-10-05T10:06:37+02:00 [combined-storage] Integrate `VectorStorageType::GraphInline` reading (#10515)
- b40cf6de9 2026-10-05T10:02:14+02:00 Remove unused (Sparse)VectorStorageType::Empty (#10431)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing
- 4f3bd3798 2026-09-23T12:10:00+02:00 feat: integrate learned metric index into Qdrant

### `lib/shard/src/optimizers/segment_optimizer.rs`

Upstream commits:

- 0d6cd45e3 2026-10-05T10:08:50+02:00 [combined-storage] Combined storage write (#10669)
- 9614dc0b0 2026-10-05T10:06:40+02:00 Expose id tracker memory placement in collection config (#10597)
- fbbe4a5f2 2026-10-05T10:06:37+02:00 [combined-storage] Integrate `VectorStorageType::GraphInline` reading (#10515)
- 943b9c3a0 2026-09-03T12:46:01+02:00 [combined-storage] Derive inline-storage warnings from the optimizer's vector config (#10430)

Thesis commits:

- 7029bcd0f 2026-09-25T12:52:30+02:00 feat(lmi): integrate database-owned training and persisted routing
## Candidate scoring contract: historical vs Qdrant 1.19.2

The current call chain for a candidate list is `PointOffsetType` offsets → `IdTrackerRead::point_mappings().filter_deferred_and_deleted(..., VisibleOnly)` → `BatchFilteredSearcher::new` → `peek_top_iter` → `ScorerFilters::check_vector` → storage `RawScorer::score_points` → `common::top_k::TopK` → `ScoredPointOffset`. The relevant sources are `lib/segment/src/id_tracker/id_tracker_base/point_mappings_ref.rs`, `lib/segment/src/index/hnsw_index/point_scorer.rs`, `lib/segment/src/vector_storage/raw_scorer.rs`, and `lib/common/common/src/top_k.rs`.

| Concern | Historical Phase C | Qdrant 1.19.2 | Status | Required LMI adaptation | Evidence/test |
|---|---|---|---|---|---|
| Offset validity | LMI bounded offsets before scoring | `NotDeletedChecker` rejects absent point bits; raw scorer should never receive an invalid offset | Same boundary | Explicitly bound by vector count and point bitmap length | Candidate fixture pending |
| Point deletion | Point bitmap from context or tracker | `ScorerFilters` checks `point_deleted`; missing flag means deleted | Same | Pass context override or tracker bitmap | Deleted-point fixture pending |
| Vector deletion | Storage deletion bitmap | `NotDeletedChecker` independently checks `vec_deleted` | Same | Keep storage authoritative | Deleted-vector fixture pending |
| Deferred points | Mapping filter before scorer | `filter_deferred_and_deleted(VisibleOnly)` excludes cutoff and mapping tombstones; shadowed actives excluded when no cutoff | Same, stronger explicit shadowing | Apply mapping filter before `peek_top_iter` | Deferred fixture pending |
| Candidate duplicates | Historical postings union deduplicated | `TopK` accepts repeated scored offsets and can emit duplicate IDs | Changed exposure | Sort and deduplicate at LMI boundary | Duplicate fixture pending |
| Candidate ordering | Postings order was implementation-specific | Equal-score acceptance depends on traversal order | Changed queue | Sort unique offsets for reproducible ties | Order fixture pending |
| Top-k queue | `FixedLengthPriorityQueue` | `TopK`: sorted insertion for k<32; median-based buffer for k>=32; rejects scores at/below threshold | Changed | Use current `BatchFilteredSearcher`; never implement own queue | Tie fixture pending |
| Ties/equal scores | Ordered ID parity not assumed for Phase G | Equal-score ID membership at cutoff is order-sensitive; `into_vec` sorts by score only | Changed | Compare scores and untied sets; document tie ambiguity | Tie fixture pending |
| Cosine preprocessing | Qdrant metric path | Current vector preprocessing applies distance through `Distance::preprocess_vector`; scorer uses storage metric | Same principle | Do not normalize or score manually | Cosine/Float16 fixture pending |
| Float16 vectors | Decoded/scored through Qdrant storage | `DenseMemmapHalf` and other Half variants dispatch through `new_raw_scorer` | Same principle | Borrow authoritative half storage | Float16 fixture pending |
| Query preprocessing | Caller-provided `QueryVector` | `BatchFilteredSearcher::new` builds a storage-specific `RawScorer` per query; higher layers prepare query vectors | Same seam | Pass query unchanged | Candidate fixture pending |
| Filter semantics | LMI used Plain fallback | Current Plain search queries payload index; candidate seam can accept optimized filter but need not for this stage | Same conservative policy | Plain fallback for filtered requests | Runtime fixture pending |
| Cancellation | Query context stop flag passed to scorer | `peek_top_iter` checks stop flag before and during candidate collection | Same | Pass `query_context.is_stopped()` | Cancellation fixture pending |
| Quantized storage | LMI fell back to Plain | `BatchFilteredSearcher::new` can use quantized vectors, but rescore/oversampling occurs in Plain search | Same conservative policy | Plain fallback; no quantized learned path | Runtime fixture pending |
| Batched scoring | Old scorer processed chunks | Current `peek_top_iter` batches offsets through `RawScorer::score_points`; full-scan `peek_top_visible` is a separate bitmap path | Changed implementation | Candidate path uses iterator API; never call full-scan API with sparse candidates | Candidate fixture pending |

`TopK` does not enforce unique offsets. Sorting/deduplication must occur before scoring for the synthetic LMI seam. The current upstream scorer implementation changed in commits `9074435d9`, `e2080ec98`, and `f494cb6cd`; the storage scorer gained graph-inline variants in `4b43179e9` and an empty-multivector fix in `6c0379387`. The historical Phase C tests did not exercise this current combination.

## Vector storage contract for LMI on Qdrant 1.19.2

`lib/segment/src/segment_constructor/segment_constructor_base/vector_storage.rs::open_vector_storage` constructs authoritative storage before `vector_index.rs::open_vector_index` receives an `Arc<AtomicRefCell<VectorStorageEnum>>`. The index receives shared ownership of that storage, not a copied corpus. `SegmentBuilder` rebuilds storage and index in its target segment. This preserves the historical ownership rule: LMI should own auxiliary routing state only; Qdrant owns vector bytes.

`VectorDataConfig::check_inline_vectors` permits graph-inline storage only for HNSW with inline storage enabled and quantization configured. `VectorStorageType::GraphInline` opens vectors from the graph file. A first LMI shell must reject GraphInline rather than treating it as an ordinary independent vector store. The combined-storage build path in `segment_builder.rs` is HNSW-specific and must remain upstream-owned.

| Storage/capability | Initial LMI policy | Reason |
|---|---|---|
| Dense mmap / chunked mmap Float32 | Candidate seam supported; runtime gate pending | Shared `VectorStorageEnum` and current `RawScorer` |
| Dense mmap / chunked mmap Float16 | Candidate seam supported; runtime gate pending | Half variants use current scorer; test required |
| In-RAM mmap / chunked mmap | Candidate seam supported; runtime gate pending | Same storage ownership, different placement |
| `Memory` (legacy RocksDB) | Unsupported | Current `open_vector_storage` explicitly rejects it |
| GraphInline / combined graph storage | Unsupported for first LMI shell | Graph file belongs to HNSW and config checks are HNSW-specific |
| Quantized vectors | Plain fallback | Plain owns current oversampling and rescore semantics |
| Sparse or multivector | Unsupported for first LMI shell | Historical LMI is dense single-vector only |

Float16 later training must decode bounded samples into f32 compute buffers; this task only proves the serving scorer boundary. No corpus vectors are copied into LMI.
## Current-upstream candidate seam proof

The new `lib/segment/src/index/candidate_scoring.rs::score_candidates` receives segment-local offsets only. It bounds them by storage and point-bitmap length, sorts and deduplicates them, applies `PointMappingsRefEnum::filter_deferred_and_deleted(VisibleOnly)`, and calls current `BatchFilteredSearcher::peek_top_iter`. Qdrant's scorer applies the context point mask and vector deletion mask, builds the metric-specific `RawScorer`, checks cancellation, and uses current `TopK`. The helper owns no vectors and does not map external IDs.

Evidence: `cargo test -p segment --test candidate_scoring_119 --locked` passed six tests for subset top-k, candidate order/duplicates, independent point/vector deletion, ties, context deletion/cancellation, and real Float16 Cosine storage. `cargo test -p segment --lib filter_deferred_and_deleted_skips_shadowed_on_include_all --locked` passed the upstream mapping test covering shadowed/deferred behavior. The helper currently does not accept filters or quantized scoring; a future LMI index must use Plain fallback for those requests.

The tie test proves repeatability after sorting candidates for its fixture. It does not establish a universal stable order for equal-score points: `TopK` compares scores, not point IDs. Results at a tie cutoff must be evaluated by score and appropriate ID-set semantics.
## First implementation stage: minimal index shell

Qdrant 1.19.2 now recognizes the historical tagged `Indexes::Lmi {}` and `Indexes::LmiTrained(LmiConfig)` segment configuration shapes. The `LmiConfig` fields and defaults are reused from the frozen branch; no Torch or training dependency was added. The first-stage constructor builds `VectorIndexEnum::Lmi(LmiIndex)` only for `Lmi {}`. `LmiTrained` open/build returns an explicit error because state persistence has not been ported. Read-only LMI open likewise returns an explicit error. This prevents old persisted generations from being mistaken for a working learned index.

The runtime `LmiIndex` wraps current `PlainVectorIndex`. Its synthetic candidate source is compiled only with the segment `testing` feature and is never persisted. Supported dense nearest queries with default params and no filter or quantized vectors use the test candidate set, normalized and scored by `score_candidates`. With no synthetic set, production queries use Plain. Exact, filtered, nondefault-param, quantized, and unsupported query forms use Plain. No MLP routing, postings, training, or persisted state exists on this branch yet.

`VectorDataConfig::check_inline_vectors` remains HNSW-specific. The storage constructor now explicitly rejects `GraphInline` when the index is not HNSW, before attempting graph-owned storage. The LMI index constructor also rejects multivectors and GraphInline; the runtime test confirms a clear GraphInline error. Dense independent Float16 storage and scorer behavior are tested. The collection REST/gRPC `lmi_config` field has not been exposed in this stage; direct segment configuration is the supported test entry point. Edge Python reports LMI as unsupported, and the infallible Edge FFI projection cannot carry LMI type information. Neither edge binding is a supported LMI configuration surface.

### Historical Phase C invariants against Qdrant 1.19.2

| Historical invariant | Current implementation | Evidence |
|---|---|---|
| LMI generates segment-local `PointOffsetType` only | Test-only synthetic source supplies offsets to `score_candidates` | LMI dispatch and subset tests pass |
| Qdrant owns vector bytes | `LmiIndex` borrows shared `VectorStorageEnum`; no vector corpus copy | Float16 runtime and scorer tests pass |
| Qdrant applies visibility/deletions | Mapping filter handles deferred/mapping state; `BatchFilteredSearcher` handles context point and vector deletion | Six scorer tests and upstream deferred/shadowed test pass |
| Qdrant performs metric scoring | Current storage `RawScorer` is built by `BatchFilteredSearcher` | Dot and Float16 Cosine fixtures pass |
| Qdrant performs top-k | Current `TopK` runs inside `peek_top_iter` | Subset, order and tie tests pass |
| External IDs remain outside LMI | Candidate helper returns `ScoredPointOffset` only | API and runtime type inspection |
| Query context remains Qdrant-owned | Context mask, cancellation and hardware counter pass through | Context/cancellation test passes |

### First-stage validation

- `cargo check -p segment -p collection --locked`: passed with no LMI warnings after test-only gating.
- `cargo check --workspace --locked`: passed after explicit Edge binding matches.
- `cargo test -p segment --test candidate_scoring_119 --test lmi_shell_119 --locked`: six candidate tests and five shell tests passed (the shell rejection case was added after the combined run and passed separately).
- `cargo test -p segment --lib filter_deferred_and_deleted_skips_shadowed_on_include_all --locked`: passed.
- `cargo test -p segment --test integration exact_search_test --locked`: passed.
- `cargo test -p segment --test integration test_batch_and_single_request_equivalency --locked`: passed.
- `cargo test -p segment --test integration test_filterable_hnsw --locked`: four filtered HNSW cases passed (nearest, discover, recommend best score, recommend sum scores).
- Check-only whole-workspace formatting reports only two pre-existing upstream file differences after formatting touched files: `lib/common/common/src/universal_io/simple_disk_cache/placeholder.rs` and `lib/segment/src/index/field_index/full_text_index/inverted_index/on_disk_inverted_index/on_disk_postings.rs`.

The next task can begin native MLP routing and compact postings against this tested scoring boundary. Persisted LMI state, collection-level configuration, read-only serving, optimizer lifecycle and real-data results remain unvalidated; this shell must not be used as the canonical thesis development branch yet.
