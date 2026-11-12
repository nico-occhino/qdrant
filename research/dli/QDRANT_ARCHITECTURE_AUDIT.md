# Preliminary Qdrant audit — incomplete

Source: parent b5c195bd5248589f3c01b940fb8365b14256f838. This is an initial source inspection, not a lifecycle audit.

`lib/collection/src/collection_manager/segments_searcher.rs::SegmentsSearcher::search` collects holder segments under a read lock, releases that lock, searches segments concurrently, and merges results with a sampling/rerun path. Existing search is a useful fan-out baseline, but the inspected code does not implement cross-level router budgeting or a level-manifest snapshot.

`lib/shard/src/segment_holder/mod.rs::SegmentHolder::swap_new` assigns a new ID to the output and removes input IDs. Thus holder IDs cannot simply be assumed stable across optimization. `lib/shard/src/optimizers/segment_optimizer.rs::SegmentOptimizer::optimize` takes input IDs and an output UUID and delegates to `execute_optimization`. Whether a DLI controller can safely coordinate all optimizer choices, publication and recovery is not established. A generation registry must not assume ordinary replacement retains membership.

Still to trace: API-to-shard routing, WAL/replay, update and proxy semantics, `execute_optimization` publication and recovery, snapshot and reopen, filters, cancellation, shard topology, and concurrent search consistency. No contract or feasibility conclusion is claimed yet.

## Follow-up source findings (still incomplete)

* `lib/shard/src/segment_holder/locked.rs` wraps the holder in an `Arc<RwLock<_>>` and separately serializes updates with an updates mutex; its lock-order comment is part of the contract.
* `lib/shard/src/optimize.rs::execute_optimization` verifies original inputs, prepares proxies and possibly a COW segment, finalizes masks under the holder write lock and replaces original entries with proxies retaining holder IDs. Build and initial propagation occur outside that lock. `finish_optimization` takes the updates mutex, propagates remaining proxy changes, flushes and writes `SegmentVersion`, then swaps proxies for an output with a fresh holder ID. It pins old segment destruction to post-flush work and synchronizes the native segment manifest. Cancellation removes a cancelled output and unwraps proxies; non-cancellation and crash cases require distinct treatment. Source contains targeted post-swap and proxy tests; these have not been executed here.
* `lib/shard/src/segment_holder/flush.rs` documents why COW and flush ordering matters: an old copy deleted durably before its new copy is persisted may not be reconstructible by WAL replay. A DLI manifest cannot bypass these dependencies.
* `lib/collection/src/collection_manager/segments_searcher.rs::SegmentsSearcher::search` clones segment handles while holding the holder read lock, releases it, searches concurrently, and aggregates with sampling/reruns and point versions. That protects handle lifetime, but does not establish an atomic joint snapshot of a separate level registry or prove all concurrent point visibility properties. Those require native tests.
* `lib/collection/src/shards/local_shard/mod.rs` loads normalized segment directories and has WAL replay code. Full replay ordering, manifest recovery, snapshot and failure boundaries remain unaudited; see `optimize.rs::finish_optimization` and `segment_holder/proxy.rs` for existing safeguards and limitations.
* `lib/segment/src/index/lmi_index/mod.rs` combines routing, candidate retrieval and scoring in index search. The split API proposed in [comparison](ARCHITECTURE_COMPARISON.md) is not present as a validated cross-level public contract.

Verdict: native mechanisms are promising building blocks, but safe registry publication and global routing are **not demonstrated**. An independently maintained registry can be invalidated by normal optimizers and cannot be declared safe; no production implementation should proceed on that assumption.

On startup `LocalShard` normalizes directories, loads segments, checks consistency, and recovers persisted proxy changes before WAL replay (`lib/collection/src/shards/local_shard/mod.rs`). Reloaded segments are added to a newly built holder, reinforcing that its in-memory IDs are not durable level identities. WAL replay can reapply operations and relies on idempotent storage/version behavior. `segment_holder/proxy.rs::unproxy_segments` propagates changes in two passes and refuses to unwrap when propagation fails. These source observations are not a proof of a new registry's recoverability.

## Search call sites examined during continuation

The REST entry is `src/actix/api/search_api.rs::search_points`; collection batch routing is in `lib/collection/src/collection/search.rs::core_search_batch` and `do_core_search_batch`. Local shard search calls `SegmentsSearcher::search` in `lib/collection/src/shards/local_shard/search.rs`, which invokes `search_in_segment` and `ReadSegmentEntry::search_batch`; segment vector-index dispatch then reaches `LmiIndex::search`. This is a source navigation map, not an end-to-end execution trace: intermediate request conversion, read consistency and distributed replica selection remain to be verified.

Static LMI has public read access to `LmiRoutingState::router()` and `postings()`, but `LmiIndex::search` internally performs route, posting lookup and scoring together. Those accessors alone do not supply a supported, generation-bound selected-bucket scoring interface with all Qdrant search semantics.
