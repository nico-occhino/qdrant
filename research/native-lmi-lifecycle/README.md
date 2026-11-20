# Native LMI lifecycle experiment — 2026-11-09

## Scope and source

Isolated worktree: `/home/nicoo/work/qdrant-native-lmi-lifecycle-20261109`.
Branch: `research/native-lmi-lifecycle-20261109`.
Base commit: `9b12d583b1bf9df802b2cc420244bd3c24732d86`.
Selected uncommitted source patches: **none**. Existing committed learned/fallback
counters and lifecycle helpers suffice. The accepted campaign's v12 overlay is
not used or changed. See `source-provenance.json`, original status and original
uncommitted patch snapshots. Original untracked files and build cache were not
modified. Cargo uses an independently copied target directory outside both
worktrees. No production Rust implementation is changed.

## Test design

The new `native_segment_delta` child module reuses committed `lmi_lifecycle`
collection setup types and point fixtures. It inserts 10,001 points (10,000
regular points plus a distinguished external ID), uses native optimization to
build a trained LMI, pauses native optimization, and verifies coexistence with
fresh appendable Plain writes through `Collection::search`.

Mutations use normal `ShardReplicaSet::update_local` with `WaitUntil::Visible`;
the helper requires both a local result and `UpdateStatus::Completed`. This
covers the native local-shard WAL/update/segment-routing path. It does not alone
exercise external HTTP extraction, distributed consensus or remote replicas.

A through G are assertions in ONE integrated regression, not seven independently
executed tests. Initial population 10,001; post-delete population 10,001;
post-optimizer-trigger population 10,065. The extra 64 points make the existing
1 KiB indexing threshold eligible; scientific production configurations and
accepted artifacts are not used.

All vectors are coordinate axes in 16 dimensions, persisted as Float16, cosine.
The two-bucket router probes both buckets (`nprobe=2`). This intentionally tests
complete candidate eligibility rather than approximate recall. Independent
checks read actual stored vectors and verify their coordinates and exact 0/1
scores for single-axis queries. Equal-score ID permutations are permitted.
Two-axis queries force the two distinguished highest-score points to come from
different segment owners. The test additionally requests all live points and
checks the complete unique external-ID set.

Assertions cover:
- native mixed LMI/Plain search;
- insertion into appendable Plain;
- stable external ID and copy-on-write movement of an LMI-owned upsert;
- deletion from both static and mutable segments;
- static router/posting/metadata hashes unchanged during mutations;
- synchronous persistence and clean collection reopen;
- controlled native optimizer replacement and regenerated ownership;
- native learned-search counter increase and zero fallback increments.

## Existing tests reused

`tests::lmi_lifecycle::lmi_collection_builds_and_searches_with_fresh_plain_writes`
covers existing collection mutations, repeated optimizer generations, snapshot
restore and reopen. It passed in 0.41 s.

`tests::wal_recovery_test::test_wal_replay_is_synchronous_without_prevent_unoptimized`
covers the existing synchronous WAL replay path. It passed in 0.32 s. This is
separate general WAL evidence, not a new crash-injection test of mixed LMI/Plain
state. The new mixed fixture explicitly flushes before its identity-preserving
restart, so it establishes clean persisted restart, not arbitrary crash recovery.

## Actual native call paths

Mutation: `LocalShard::submit_update` / WAL `lock_and_write` → update workers →
`shard::update::points::upsert_points_impl` →
`SegmentHolder::apply_points_with_conditional_move` → appendable destination →
`Segment::upsert_point` / `vector_index.update_vector`.

A trained LMI is non-appendable and still rejects in-place vector updates.
The updated external ID moves to an appendable segment. Point deletion uses ID
tracker visibility; copied/stale vector bytes and postings do not imply a live
point.

Search: `Collection::search` → shard search fan-out → segment read-view
`search_batch` → `VectorIndexEnum` → ordinary `LmiIndex::search` and Plain search
→ native candidate scoring/visibility → external-ID conversion and merging.
The fixture does not call the dedicated SISAP direct evaluator.

Optimization: shard optimizer builds a replacement using authoritative visible
points, propagates proxy changes and publishes through holder replacement.
The regression checks new physical ownership and unchanged logical results; it
does not prove all concurrent writer/publication interleavings.

## Preparation failures preserved

1. Initial full compilation found a test-only import error: `Datatype` belongs
   to `crate::operations::types`, not `segment::types`. Corrected in the test.
   Evidence: `compile.log`, `compile-corrected.log`.
2. Initial runtime waited 180 s with zero indexed vectors because it attempted
   to enable workers only through a load-time optimizer override. Inspection
   showed `LocalShard::new` takes worker thread limits from shared persisted
   collection config, while load-time overrides feed effective optimizer setup.
   The test now updates the real collection config through
   `update_optimizer_params_from_diff` and awaits existing
   `on_optimizer_config_update`. No production behavior was changed.
   Evidence: `native-lifecycle.log` (failed),
   `native-lifecycle-corrected.log` (passed in 1.82 s).

## REST smoke

See `api_smoke.py` for the disposable loopback service test. It captures exact
requests and responses and uses only synthetic points. A completed run, if
available, is recorded in `api-smoke/summary.json`; script existence does not
establish execution. REST does not expose the test-only LMI learned/fallback
counters. API path attribution must therefore distinguish eligible request plus
persisted configuration/source inspection from directly observed internal
counters. Never interpret `hnsw_ef` as LMI `nprobe`.

## Decision and limits

For the tested local collection lifecycle, reuse Qdrant's static learned segments
plus appendable segments. An internal delta is NOT required merely to provide
query eligibility for new/upserted points. This is not a complete implementation
of temporal DLI: no Bentley–Saxe level capacity policy, cross-level learned budget,
classifier adaptation or continual learning is introduced.

Remaining work: mixed-state crash injection, mutation during active optimization,
concurrent readers/writers, multi-replica semantics, low-nprobe retrieval quality,
filters/quantization fallback observability, and workload/resource measurement.
Do not infer production readiness or ANN quality from these complete-probe axis
fixtures.

## Reproduction and review

Source `env.sh`, then use the exact selectors in `commands.txt`. Builds are offline
with two Cargo jobs; torch is the existing compatible 2.5.1 CPU environment.
`test-only-source.patch` includes the module declaration and complete new test.
No commit, push, index rebuild, accepted artifact change or PBS operation occurs.

## Final verified results

Final new integrated Rust selector: **1 passed, 0 failed, 0 ignored**, 341
filtered out, 1.67 seconds runtime. The two separately selected existing Rust
regressions also passed. Full collection test compilation succeeded.

The actual pinned-source server built successfully in 9m04s with two build
workers. REST smoke **passed**, including collection creation, 10,001-point
ingestion, learned publication, mixed search, upsert, delete, and complete server
process restart. Exact request payloads and responses are retained in
`api-smoke/requests-responses.json`; server logs in `api-smoke/server-*.log`.
The smoke used loopback ports 16433/16434 and shut both server instances down.
Its configured nprobe was 2, probing both buckets, and search params were absent.
REST has no exposed learned/fallback counters; attribution is source-derived
from eligible requests and persisted LMI state. Internal collection execution
was independently observed through learned/fallback counters.

Observed stable external ID 1000003: static LMI offset 10000 → appendable Plain
offset 2 → new learned generation offset 9999. Full visible-ID sets and stored
coordinates remained correct. Static LMI artifact hashes were unchanged during
mutations and clean reopen. No internal delta or production code change was
needed. See `results.json` for the machine-readable outcome and service hash.

Original checkout status and tracked-diff fingerprints still match the initial
snapshot. No commit or push was made.
