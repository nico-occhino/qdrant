# DLI system contract — draft, not implemented

Scope: one local shard. A logical level may contain multiple segment generations; holder IDs are transient and are not a durable level identity. External point IDs identify logical points; offsets are meaningful only within their owning physical segment; model, postings and offset space must be from the same physical generation. A search-view generation must bind membership and index generations coherently. No current test establishes this binding under optimizer replacement.

For a faithful Sommerová-style hierarchy (§3.1–3.5), buffer capacity b and arity a determine maximum level occupancy b a^i and bucket count a^i for one-based i; data moves from buffer through levels using a documented compaction policy. Destination models and postings must be built and validated privately for their physical generation before publication. Updates during construction must be preserved using Qdrant's native proxy/COW, update locking, flush and WAL contracts. Old generation data remains available for readers and recovery until Qdrant's publication and reclamation requirements are met.

The query policy must explicitly define per-level routing, score normalization, buffer priority and whole-bucket budget overshoot; raw router scores are not inherently calibrated. Native visibility and segment-local scoring remain authoritative. Filters, unsupported queries, cancellation and duplicate point versions need explicit contract tests. The test-only global selection is exploratory and does not establish this contract.

No durable DLI manifest, atomic registry publication, optimizer ownership, crash recovery or concurrent compaction implementation is currently provided. Do not interpret this draft as a passed specification gate.
