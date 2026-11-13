# Global bucket policy — independent production component

`collection_manager::dli_bucket_plan` is behind `dli-research`. It accepts caller-supplied per-bucket probabilities, physical segment UUIDs, one-based learned level IDs and occupancy estimates. Buffer segments use level 0, no bucket ID and priority 1. Learned bucket priority is P(bucket|q) × a^i (Sommerová §3.5.1, printed p. 40), rather than the different zero-based exponent used by the Rust reference. This is a heuristic, not calibrated probability. Stable tie breaking is by level, physical UUID, then bucket ID. Empty buckets are skipped; whole buckets are selected until the requested minimum candidate count is reached, and overshoot is explicit. Actual scored and visible counts can be smaller than occupancy estimates.

The planner rejects malformed probabilities, duplicate physical bucket identities, inconsistent level ownership of a physical segment, invalid arity and budget, and numeric overflow. It does not validate that inputs enumerate the full active topology, that probabilities sum to one, that offsets match a physical generation, or that predictions refer to actual loaded LMI postings. Its output must not be used for production learned queries until those contracts and native result merging/visibility are integrated.

The `dli_recovery_control` test feeds a real persisted LMI's logits and postings to this planner after native collection reopen, then scores the selected bucket with Qdrant's candidate scorer and verifies each returned score against the fixture oracle. This one-level control does not establish a full DLI search path or compare global ranking across two independent real models.

## Continuation 09: native routing primitive
`dli_native_route::route_pinned` invokes real LMI routers, existing whole-bucket planning, native candidate scoring, and native per-segment result conversion using caller-pinned segments. Its output is deliberately not a collection result: it cannot establish complete holder coverage or version correctness for unselected newer point copies. It is not wired to LocalShard or automatic fallback.

## Continuation 10: conservative candidate resolution
A constructed overlap demonstrates that native candidate-only aggregation accepts an older selected copy when its newer Plain copy was not selected. `route_resolved_pinned` adds bounded batched native lookups and returns a fallback decision for ambiguous occurrences, any deleted/deferred segment state or invalid physical tokens. An explicit test uses the same captured handles for native exact search and verifies all nine fixture scores. This is not automatic collection dispatch and does not resolve proxy/filter or full-holder coverage semantics.

Continuation 11: test-enabled LocalShard can opt into process-local research dispatch with complete UUID-role coverage checks, nonblocking pinned segment reads, native global routing, conservative resolution and automatic exact search on captured handles. A manually installed two-level collection fixture exercised learned and duplicate-fallback routes. The setter is test-gated; no durable or production-ready configuration is provided.

Continuation 12: test-only channel hook pauses after all original segment read locks and holder identity validation. Native upsert interleaving is exercised; segment read guards still span routing, scoring and version resolution, potentially delaying writers. Optimizer/proxy interleavings and fairness are not established.

Continuation 13: deterministic native delete and real optimizer proxy/private-build controls supplement the upsert test. `read_guard_nanos` measures the guarded worker interval, including any deliberate test pause. No bounded-starvation or general snapshot guarantee is established.
