# Implementation report — research controls only

Starting source 960a35f37ab1cba3b49ed7404f9e4d22066f2dee. No production code or public APIs are changed. `lib/segment/tests/dli_native_topology.rs` builds two independently trained LMI segments and a Plain segment, searches each with all learned buckets, maps offsets through the owning tracker, checks a small exact cosine oracle and reloads both LMI segments. The native test passed; see journal and test matrix for conditions and limitations. The test is segment-level and does not implement a registry, optimizer publication, concurrent updates or temporal compaction.

The separate `prototypes/budget_control.rs` is a mock policy and numerical-convention control. It is not native scoring or database evidence.

A test-only global bucket planner ranks actual LMI logits and scores selected native postings; the observed budget 2 selected 3 candidates in one bucket. It is research control code, not a native shard coordinator. There is no durable or concurrent DLI generation.

Holder continuation: `lib/shard/tests/dli_view.rs` adds test-only registry/controller, pinned view, generation-bound route tokens, controlled replacement and native update-helper tests. Two tests passed with deterministic barriers and actual trained LMI/native scoring. No production module/API changed. See generation specification for strict no-writes-during-build replacement precondition and unsupported proxy/publication/recovery states.

Native optimizer continuation adds a test `OptimizationStrategy` pause and calls real `execute_optimization`, not the no-writes replacement helper. Source upsert/delete and new insert survive both publication and cancellation. All four integration tests pass. Native old/new file coexistence is asserted. Durable DLI registry, hierarchy reopen, compaction, configuration and dynamic evaluation remain unimplemented; production source is unchanged. Persistent-registry expansion is gated by a missing logical commit/recovery contract, documented in the recovery matrix.

Derived-view continuation adds a test-only live-holder stale-registry rejection and exact native segment-search control. No metadata persistence, LocalShard recovery or production fallback was implemented.

A test-only collection lifecycle adaptation now checks a persisted stale physical owner set after clean LocalShard reopen and runs an explicit exact collection query against an independent 193-point oracle. It adds no production registry, automatic fallback, or publication mechanism.

The collection research fixture now compares a versioned, full-holder JSON snapshot and uses a test-only conditional exact-search dispatch. It is neither a DLI descriptor implementation nor native automatic fallback; generation and artifact-presence validation are incomplete and validation/search are not atomic.

Continuation-07 adds a production native handle-capture/search-execution seam and, behind collection feature `dli-research`, an explicitly called dense-nearest exact fallback with UUID topology-hint diagnostics. It reuses the native result aggregator and refuses `indexed_only`. The method always executes exact fallback, including on a matching hint; it is not connected to DLI configuration or LocalShard routing. No durable descriptor, global learned DLI routing, temporal levels or compaction are implemented.

Continuation-08 adds feature-gated production arithmetic components `dli_schedule` and `dli_bucket_plan`, each with three passing deterministic tests. A native persisted LMI fixture feeds real logits/postings to the bucket planner and checks native scoring. Neither component is wired to LocalShard or the optimizer: no automatic compaction, learned DLI collection search or durable hierarchy is present.

Continuation 09 adds feature-gated `dli_native_route`, combining real LMI routing, the existing global planner, native selected-posting scoring and per-segment native external-ID/version conversion. It is intentionally not a query-serving coordinator. A native private multi-input LMI build and clean segment reload are exercised; automatic compaction, native version-safe global aggregation and durable level publication are absent.

Continuation 10 adds a feature-gated conservative candidate resolver and composed route wrapper, using batched native ID/version lookups and physical offset checks. The wrapper returns a fallback decision; it does not itself execute native search or integrate LocalShard. Deleted/deferred segments and duplicate occurrences deliberately require exact fallback. A scheduler-selected private multi-input build is a test control, not automatic compaction.

Continuation 11 adds an opt-in process-local LocalShard research dispatcher; activation is restricted to test builds pending concurrency and recovery evidence. It executes actual global learned routing or same-capture exact fallback with counters; it is not automatic temporal compaction or a recoverable DLI configuration.

Continuation 12 adds a test-only bounded guard hook and a native upsert/query interleaving test. Production DLI activation and compaction behavior are unchanged.

Continuation 13 adds native delete/query and real optimizer proxy/publication/cancellation controls, plus guarded-worker duration accounting. Test-only activation, automatic-compaction and hierarchy-recovery status are unchanged.

Continuation 14 adds a test-activated update-completion maintenance observer via UpdateTracker and a selective asynchronous native LMI private builder. No training runs under the native update mutex. Automatic publication and persistent hierarchy integration remain disabled.

Continuation 15 introduces a test-gated native `OptimizationStrategy` LMI builder adapter and exercises it through real proxy mutation, publication and cancellation. The insert-triggered coordinator still only builds private outputs; connecting it safely requires synchronized DLI source-plan revalidation and lifecycle ownership.

Continuation 16 adds an expected UUID-set check to the native optimizer LMI adapter. The automatic private builder remains disconnected from native optimizer publication.

Continuation 17 adds a default native optimizer source-validation callback and physical UUID validation in the research LMI strategy before proxy installation. The maintenance scheduler is not yet wired to native optimization.

Continuation 18 carries observed source UUIDs into the private LMI adapter and implements restricted Plain COW construction. It does not connect the maintenance job to `execute_optimization`.

Continuation 19 introduces test-only native optimization from the update-triggered maintenance worker and validates an original UUID/role map under the native holder's upgradable read. It produces one observed physical publication and COW; logical activation, mutable/recoverable hierarchy and safe lifecycle are not implemented.
Continuation 20 implements process-local logical activation after native physical publication and tests two insert-triggered trained generations with learned collection search after each. It adds a versioned derived UUID/level descriptor with in-process validation and corrupt-metadata exact fallback. The test fixture uses manually assembled temporary segments and test-only activation; full collection reopen, managed maintenance, concurrent automatic-job mutation, and 300K evaluation remain open.
Continuation 20 completes three physical publications in the fixture (two sequential, one after a pre-optimizer cancellation); the cancelled attempt publishes nothing. Learned LocalShard search passes after each completed hierarchy update, metadata corruption falls back exactly, and active segment reload into a fresh holder validates the derived descriptor. Full collection recovery and production maintenance lifecycle remain unimplemented.
