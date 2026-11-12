# Hierarchy placement comparison (preliminary; executable gate incomplete)

Scope: one local shard. B is the research direction, not an established correctness result. A is a research control; neither is implemented. A level in B is a logical group of zero or more immutable segment generations, not a segment ID.

| Concern | A: inside a segment | B: across segments |
|---|---|---|
| Ownership and writes | Nested buffer and levels within one index; must reconcile appendability and segment offsets. | Qdrant owns vectors, IDs, WAL and appendable segments; registry owns logical membership. |
| Compaction | Nested rebuild can conflict with outer optimizer; internal offset changes and publication require design. | Native optimizer offers proxy/COW mechanism, but unrestricted optimizer replacement can invalidate membership; planner and publication must participate. |
| Search and budgets | Single index can coordinate predictions and bucket budgets locally. | Current fan-out invokes self-contained segment searches; global bucket budget requires staged routing and scoring with a coherent generation view. |
| Visibility and update | Segment handles external visibility; internal generations must follow deletions and offsets. | Existing versioning, deduplication and COW are essential; registry must never mask new buffer writes or stale versions. |
| Durability | Must atomically publish internal index generation with segment lifecycle. | Must couple persistent registry with segment manifest, optimizer publication and recovery; independent manifest writes are unsafe. |
| Memory/build | May duplicate mutable state and nest compaction; potentially simpler budget coordination. | Reuses vectors but temporarily holds old/new models, postings and segment build data; several segments can increase router overhead. |
| Engineering/testability | Narrow search interface but substantial nested lifecycle changes. | Broad shard/search/optimizer changes, but native ownership tests can exercise existing mechanisms. |

For query latency, routing, postings, scoring, buffer and merge are useful accounting components, not necessarily additive wall times: segments execute concurrently. Global routing adds a coordination barrier and sorts B nonempty buckets (approximately O(B log B)); actual model cost depends on architecture and bucket count. Whole-bucket budget can overshoot by a bucket. Compaction writes and temporarily retains old/new data; measure actual bytes and peak memory, do not infer benefit from the policy.

Minimum exploratory API boundary: immutable per-generation route output containing level/segment membership, bucket identifiers, comparable policy scores and visible occupancy; coordinator selects whole buckets using an explicitly stated policy and budget; segment-local scoring receives validated generation-bound bucket selections and applies filtering, deletion, vector scoring and version handling. A pinned holder/registry view must cover both stages. Do not send offsets between segment generations. Existing `VectorIndex::search` combines selection and scoring and does not expose this contract. Independent per-level probing is a baseline, not a reproduction of global selection. Raw outputs across routers are not calibrated; Sommerová's normalization is a heuristic, not calibration.

Multi-shard: each local shard has independent point ownership, buffer, hierarchy and registry. A future cluster-wide budget would need an additional allocation/merge contract and consistent distributed semantics; this investigation does not establish one.

The holder experiment tests B's restricted in-memory membership boundary, using actual native segment handles. It adds an external serialization boundary that production optimizers do not observe. This is evidence about what a coordinated adapter can accomplish, not evidence that B's full optimizer integration is simpler than A. Pending writes during rebuild and durable publication remain decisive unresolved costs.
