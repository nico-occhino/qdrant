# Preliminary Qdrant audit — incomplete

Source: parent b5c195bd5248589f3c01b940fb8365b14256f838. This is an initial source inspection, not a lifecycle audit.

`lib/collection/src/collection_manager/segments_searcher.rs::SegmentsSearcher::search` collects holder segments under a read lock, releases that lock, searches segments concurrently, and merges results with a sampling/rerun path. Existing search is a useful fan-out baseline, but the inspected code does not implement cross-level router budgeting or a level-manifest snapshot.

`lib/shard/src/segment_holder/mod.rs::SegmentHolder::swap_new` assigns a new ID to the output and removes input IDs. Thus holder IDs cannot simply be assumed stable across optimization. `lib/shard/src/optimizers/segment_optimizer.rs::SegmentOptimizer::optimize` takes input IDs and an output UUID and delegates to `execute_optimization`. Whether a DLI controller can safely coordinate all optimizer choices, publication and recovery is not established. A generation registry must not assume ordinary replacement retains membership.

Still to trace: API-to-shard routing, WAL/replay, update and proxy semantics, `execute_optimization` publication and recovery, snapshot and reopen, filters, cancellation, shard topology, and concurrent search consistency. No contract or feasibility conclusion is claimed yet.
