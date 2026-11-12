# ADR 001: temporal hierarchy placement — pending feasibility gate

Status: **provisional research direction; implementation blocked pending native evidence**. One local shard is the initial scope.

Continue evaluating logical level groups across Qdrant-owned segments (B), as requested. This preserves the intended storage ownership and allows multiple segments per level. It is not a finding that B is superior to the encapsulated control A.

Evidence: `finish_optimization` replaces proxies through `swap_new`, then syncs the segment manifest; updates are frozen during critical propagation and publication. A standalone registry keyed by old holder IDs can therefore become stale, and separate persistent registry publication can diverge from the native manifest. Search currently snapshots segment handles, not a DLI registry generation. These are unresolved integration contracts, not a demonstrated impossibility. See [audit](QDRANT_ARCHITECTURE_AUDIT.md).

Required before production integration: coordinate optimizer planning and holder replacement with level metadata, atomic in-memory view and crash-recoverable on-disk publication, preserve COW/WAL safety, and expose generation-bound routing and scoring. Native tests must establish concurrent update/query, failure boundaries and reopen. If this requires rewriting the optimizer/durability contract rather than an evidenced bounded integration, stop and revisit this ADR. No independent vector store, WAL, or CLI maintenance framework is approved.

## Continuation evidence

A small native segment-level fixture now builds two trained LMI segments and a Plain buffer and checks full-probe search against an independent oracle. A test-only coordinator also uses real router outputs and Qdrant candidate scoring. This supports the *mechanical possibility* of composing native segment indexes in a bounded fixture; it does not resolve the holder/registry transaction, query-update consistency or durable recovery. A has not received equivalent native evidence. The decision remains provisional and production integration remains gated.

Holder continuation confirms a cooperating test controller can capture logical membership and native handles coherently and retain old query handles across native swap. An uncoordinated swap is detected rather than repaired. This supports further B research but does not resolve optimizer ownership or durable membership. Production direction remains provisional; pending-write replacement is explicitly unsupported by the test helper. Next gate is native proxy/delta or validated source-version publication, before compaction.

## Native optimizer evidence and M2 publication gate

The real optimizer's proxy/delta replacement and cancellation now pass with source upsert/delete and fresh insertion while private construction is paused. This removes the old no-writes helper restriction for this tested native path, but not for the old registry controller. Native publication leaves source preimages on disk until safe cleanup; the manifest is intentionally a superset listing and LocalShard startup scans/deduplicates directories. No exclusive durable level-membership rule follows from those facts. B remains selected and provisional; persistent registry expansion stops until logical generation commit is separated from native physical recovery dependencies. No claim of impossibility or completed DLI is made.

Derived metadata is a promising alternative to coupled publication: reject stale metadata after native recovery and use a complete native fallback while rebuilding. The live-holder control supports only exact-search feasibility in a non-overlapping fixture. Native WAL replay timing, retained-preimage deduplication and concurrent invalidation remain open; the decision is not yet validated or adopted for production.

The clean-restart collection control now confirms that native optimizer replacement can make a persisted owner set stale, and normal exact collection search can match the post-reopen oracle. It does not validate an actual derived registry or automatic fallback. Coupled versus derived publication remains provisional until full coverage, ongoing invalidation and failure-boundary tests pass.
