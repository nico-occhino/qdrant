# ADR 001: temporal hierarchy placement — pending feasibility gate

Status: **provisional research direction; implementation blocked pending native evidence**. One local shard is the initial scope.

Continue evaluating logical level groups across Qdrant-owned segments (B), as requested. This preserves the intended storage ownership and allows multiple segments per level. It is not a finding that B is superior to the encapsulated control A.

Evidence: `finish_optimization` replaces proxies through `swap_new`, then syncs the segment manifest; updates are frozen during critical propagation and publication. A standalone registry keyed by old holder IDs can therefore become stale, and separate persistent registry publication can diverge from the native manifest. Search currently snapshots segment handles, not a DLI registry generation. These are unresolved integration contracts, not a demonstrated impossibility. See [audit](QDRANT_ARCHITECTURE_AUDIT.md).

Required before production integration: coordinate optimizer planning and holder replacement with level metadata, atomic in-memory view and crash-recoverable on-disk publication, preserve COW/WAL safety, and expose generation-bound routing and scoring. Native tests must establish concurrent update/query, failure boundaries and reopen. If this requires rewriting the optimizer/durability contract rather than an evidenced bounded integration, stop and revisit this ADR. No independent vector store, WAL, or CLI maintenance framework is approved.
