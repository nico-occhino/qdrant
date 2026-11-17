# Independent code audit — native DLI request accounting and lifecycle

**Auditor:** Claude Code (independent of the Codex implementation sessions).
**Audit baseline:** `be2e58f1ed8a7802a7004c93863634c024ace648`. Earlier read-only findings were taken at `f6d080031`.
**Commits reviewed:** `e043de079`, `fcfa65e3a`, `f6d080031`, `ceca71685`, `b8d4b9157`, `be2e58f1e`.
**Continuation40 benchmark source:** `ceca71685743b7c3c66c57114faa3cde171d5f28`.

Status vocabulary:
- FIXED AND VERIFIED
- FIXED BUT INSUFFICIENTLY TESTED
- CONFIRMED OPEN DEFECT
- REPRODUCIBLE DESIGN RISK
- UNVERIFIED HYPOTHESIS
- FALSE POSITIVE / NOT APPLICABLE

"Fixed here" means changed by this audit; the commits are listed in §3.

All eleven findings are resolved or classified. Five commits change code; F8 and F10 are documented design risks with no code change.

## 1. Findings matrix (ordered by severity)

### F3 — Query-outcome attribution
| | |
|---|---|
| Severity | High (measurement validity) |
| Status at `be2e58f1e` | **CONFIRMED OPEN DEFECT** |
| Status after audit | **FIXED AND VERIFIED** |

**Evidence.** `segments_searcher.rs::search_dli_research` wrote `candidate_resolution: "resolved"` immediately after `resolve_pinned_candidates`. That was before the `insufficient_valid_candidates` underfill check. The test added in `be2e58f1e` (`dli_topk_underfill_uses_complete_native_fallback`) asserted `events[0]["candidate_resolution"] == "resolved"` for a request whose final result came from exact fallback, so it locked the mislabel in. Two further gaps:
- Routing errors and early rejections produced no learned-attempt record.
- Errors in exact search produced no event at all.

**Root cause.** The label was derived from the intermediate resolver result, not the dispatcher's final decision. Events were emitted at two different points in the code, each with partial knowledge.

**Change.** Exactly one `dli_request` event per logical request, emitted after the final decision.
- `final_outcome` is one of `learned`, `exact_fallback` or `error`, with `fallback_reason` / `error`.
- `attempt.attempt_resolution` is one of `resolved`, `underfilled`, `fallback` or `error`, and is decided after the underfill check.
- `learned_scoring` holds attempt scorer work, including partial work before a routing error.
- `exact_fallback` holds `scored_vectors`, `segment_search_ns_sum` and `elapsed_ns`.
- Join errors, the `indexed_only` rejection and exact-search errors emit `final_outcome: "error"`.

**Regression.**
- `dli_topk_underfill_uses_complete_native_fallback` (rewritten).
- `dli_request_events_are_final_and_request_local_under_concurrency`.
- `dli_request_events_cover_rejection_and_unavailable_quantization`.

**Limit.** A request whose future is dropped (client cancellation) emits no event. Drop-time I/O was deliberately not added.

### F2 — Exact-fallback scorer attribution
| | |
|---|---|
| Severity | High (measurement validity) |
| Status at `be2e58f1e` | **FIXED BUT INSUFFICIENTLY TESTED** |
| Status after audit | **FIXED AND VERIFIED** |

**Evidence.**
- `be2e58f1e` added `NativeExactSearchProfile`, an `Arc` per request. `run_segment_search_profiled` wraps each per-segment blocking task, including sampling re-runs, in a `TaskScoringProfileGuard` and adds the task-local success count to that request's `Arc`. This is sound: each segment search runs synchronously on one blocking thread, and the counter is request-owned rather than process-global.
- The tests only asserted `> 0` and `second < first`. A double count, or a count that included deleted vectors, would have passed.

**Change.** Equality assertions with independently derived expectations:
- Unfiltered exact search scores every live vector of every captured segment exactly once.
- A HasId-filtered exact search scores exactly the matching live vectors.
- Underfill fallback scores `learned_live + plain_live`.

**Regression.**
- `native_exact_profiles_are_request_local_under_concurrency` (strengthened).
- `dli_topk_underfill_uses_complete_native_fallback`.
- `dli_request_events_cover_rejection_and_unavailable_quantization`.

### F4 — Diagnostic file I/O inside timed, guard-held sections
| | |
|---|---|
| Severity | Medium |
| Status at `be2e58f1e` | **CONFIRMED OPEN DEFECT** for profiled requests. **Not applicable to continuation40 latency** (§2). |
| Status after audit | **FIXED AND VERIFIED** |

**Evidence.**
- The JSONL append ran inside the `spawn_blocking` attempt closure while every segment `RwLockReadGuard` was held.
- It also ran before `resolution_nanos` was read, so the shard-aggregate resolution time included file open/append, and `read_guard_nanos` included I/O.
- The route function appended `DLI_ROUTING_EVENTS` under the same guards.

**Change.**
- The attempt closure only fills a request-local `DliAttemptRecord` (`Arc<Mutex<_>>`, uncontended).
- `resolution_nanos` is accumulated right after `resolve_pinned_candidates` returns.
- The routing plan is captured in `RouteTrace.routing_event`.
- Both lines are written by `persist_dli_events` after the closure has returned (guards dropped), on `tokio::task::spawn_blocking`, so blocking I/O stays off async executor threads.
- No queue is introduced: one bounded write per request, awaited.

**Regression.** `dli_request_events_absent_when_profiling_disabled`: no file is created and no profile is collected when profiling is off (the latency configuration).

**Remaining limits.**
- With profiling on, the event write still adds to the profiled request's end-to-end Collection latency, after all internal timers. Latency must therefore continue to be measured with profiling off, as the continuation40 harness does.
- The older opt-in `DLI_NATIVE_ACCESS_EVENTS` fault trace still writes inside the scoring loop by design (it measures per-member faults). It must never be enabled during latency runs; continuation40 did not set it.

### F6 — Cargo feature contamination of static LMI
| | |
|---|---|
| Severity | Medium (comparability) |
| Status at `be2e58f1e` | **CONFIRMED** |
| Status after audit | **FIXED AND VERIFIED** (feature graph and compile) |

**Evidence.**
- Since `3a419c11b`, `collection/dli-research` enabled `segment/lmi-research`.
- Under `lmi-research`, static `LmiIndex::search` (`lmi_index/mod.rs`) switches to `candidates_for_query_profiled` + `score_candidates_profiled`, which add per-stage `Instant` calls and a per-query `log::info!(target: "lmi_research")`.
- The Plain fallback branch also logs a reason.
- Any static-LMI baseline compiled with `dli-research` after `3a419c11b` was therefore not comparable to earlier static numbers.

**Change.**
- New segment feature `lmi-scorer-profile`. It gates only the request-local scorer APIs:
  - `score_candidates_profiled` / `score_candidates_quantized_profiled` / `CandidateScoreProfile`
  - `peek_top_iter_profiled`
  - the opt-in `DLI_QUANTIZATION_BUILD_EVENTS`
- `lmi-research = ["lmi-scorer-profile"]` and keeps static search profiling.
- `dli-research = ["lmi-training", "segment/lmi-scorer-profile"]`.

**Feature provenance of past runs.**
| Campaign | Source | Effective segment feature |
|---|---|---|
| continuation-22..24 300K static vs DLI | < `3a419c11b` | `dli-research` did not imply `lmi-research`; static search unprofiled |
| continuation40 | `ceca71685` | `lmi-research` was on, but no static LMI search ran in that benchmark, and the DLI route takes unprofiled scorers when profiling is off (§2) |

**Remaining limit.** Static-vs-DLI timing must still be compiled from one source with identical features.

### F7 — Ordinary Qdrant optimizers acting on DLI-managed segments
| | |
|---|---|
| Severity | High (liveness) |
| Status at `be2e58f1e` | **REPRODUCIBLE DESIGN RISK** |
| Status after audit | **FIXED AND VERIFIED** (activation-time). The runtime update path remains a **documented limit**. |

**Evidence.**
- `configure_dli_experimental` (`local_shard/search.rs`) validated shard, replica, vector and quantization scope, but never `optimizer_config`.
- Only fixtures set `max_optimization_threads = Some(0)` (`dli_sisap300k_benchmark.rs:63`, `dli_sisap_scaling_benchmark.rs:92`, `dli_recovery_control.rs:406`).
- Continuation 22 reproduced the interference twice: `new DLI member is not appendable` with generic indexing enabled, then again with one generic worker (`continuation-22/benchmark-300k/logs/run.log`, `run-corrected.log`).
- A native-created segment is absent from the DLI role map. Search then rejects with `topology` on every request, and `DliMaintenance::observe` returns early when the buffer UUID is unknown (`dli_maintenance.rs`, the `roles_snapshot.get(&guard.uuid)` branch). The result is permanent exact fallback with no further learned publication until a reopen rebuilds roles.

**Change.** Activation fails closed with an explicit error unless `max_optimization_threads == Some(0)`. This is the narrowest reliable coordination. DLI maintenance already runs its own `execute_optimization` outside the generic worker pool, so disabling the pool removes no DLI functionality. Non-DLI collections are unaffected.

**Regression.** `dli_recovery_control` asserts that the configured DLI fixture with threads `None` (auto) and `Some(1)` is rejected with the message.

**Limits.**
- A runtime `update_optimizer_params` on an active DLI collection is not intercepted. It is rejected at the next load.
- Intercepting it inside `on_optimizer_config_update` would run after the new config is persisted and while workers are stopping, which needs a separate design.
- Vacuum of deleted points in learned levels is now only performed by DLI recompaction.

### F9 — Quantization-artifact recovery and configuration inheritance
| | |
|---|---|
| Severity | Medium (liveness / resource) |
| Status at `be2e58f1e` | Both parts **CONFIRMED** |
| Status after audit | Both parts **FIXED AND VERIFIED** |

**(a) Missing or invalid derived artifact.**
- `create_segment.rs` drops a missing, corrupt or dimension-incompatible quantized artifact for `LmiTrained` with a warning. The route then returned `quantized generation unavailable`, so every query took complete exact fallback until that member was recompacted. The top level may never be.
- Classification of the cases:

| Case | Why | Behaviour after audit |
|---|---|---|
| Valid learned index, quantization unavailable | The artifact is only a pre-filter derived from the member's own original vectors. Router, postings and vector storage are validated independently at load. Original scoring of the same candidates is the A0 path. | Degrade **that member** to original scoring, counted as `quantization_degraded_members` (route), `quantization_degraded_queries` (shard), and in the request event's `attempt` |
| Present but foreign/incompatible object (storage path or dimension mismatch at query time) | Signals mis-wiring | Still **fails closed** to complete exact fallback (`quantization_generation_mismatch`) |
| Corrupt learned-index metadata | Unchanged: `LmiIndex::open_trained` validation | Load error / native fallback (no change) |
| Incompatible physical generation or unsafe visibility | Unchanged: resolver and descriptor validation | Exact fallback |

- Rebuilding a missing artifact without recompaction is **not implemented**. It would need a DLI-owned quantization job on an immutable member. Learned search now stays available meanwhile.

**(b) Configuration inheritance.**
- The destination config was cloned from `sources.first()`, and quantization was only *set* when the shortlist was on. So disabling the shortlist kept propagating quantization from a previously quantized first source.
- `derive_destination_quantization` now sets or clears it from the current configuration.

**Regression.**
- `destination_quantization_follows_current_shortlist_not_first_source`.
- Route-level missing-object and reopen-without-metadata cases now assert one degraded member and per-segment results identical to the original route.
- The foreign-generation swap still errors.
- Configured Collection recovery with missing, corrupt and incompatible metadata: `learned=1 exact=0 degraded=1` (previously `learned=0`).

### F10 — Native publication / descriptor durability
| | |
|---|---|
| Severity | Medium (recovery cost / liveness; not result correctness) |
| Status | **REPRODUCIBLE DESIGN RISK**. The "no fsync" sub-claim is a **FALSE POSITIVE**. |

**Trace.**
1. Sources are selected by the plan from the live role map.
2. `execute_optimization` builds a private destination.
3. Proxy/COW transfers concurrent mutations.
4. The native holder swap happens. Qdrant's segment directories are the authoritative physical state.
5. `save_descriptor` writes `dli_hierarchy.json`.
6. In-memory roles are updated.
7. On reopen, `load_validated_descriptor` clears learned membership first, then requires exact UUID equality with the recovered holder, no proxies, no deferred points, level 0 = appendable Plain and level ≥ 1 = LMI with valid routing state.

**Durability of single files.** `common::fs::atomic_save_json` uses `atomicwrites::AtomicFile`, which fsyncs the temp file and, after `renameat`, the parent directory (`atomicwrites-0.4.4/src/lib.rs:164,200-202`). `lmi_index/state.rs` uses the same primitive and writes metadata last. Per-file atomicity and durability hold.

**Non-transactional gap.** The holder swap and the descriptor are not one transaction.

| Scenario | Outcome |
|---|---|
| Graceful restart | Descriptor matches; learned routing restored (existing clean-reopen test) |
| Descriptor write/rename failure after swap | Error logged into `optimizer_errors`; memory is ahead of disk. Reopen sees the previous descriptor (UUID mismatch) and takes exact fallback, `initialize_native_fallback`, then a full rebuild on the next update. The next successful publication rewrites the descriptor, so the in-process state heals itself. |
| Abrupt termination between swap and save | Same as above (fail-closed, expensive) |
| Interrupted optimizer before swap | Native rollback; descriptor unchanged and valid |
| Power loss | Bounded by Qdrant's own segment/WAL guarantees plus the per-file fsync above. **Not demonstrated** by any test. |

A stale descriptor can never *activate* incorrect learned routing, because validation requires exact UUID coverage and every publication creates new UUIDs.

**Evidence of the stale-descriptor state.** `dli_recovery_control` writes a descriptor with one member removed and asserts rejection. The corrupt-descriptor reconstruction test covers the resulting fallback and rebuild. A direct fault injection of the save failure was **not added**.

**No change.** Making the descriptor transactional with the native swap would be a new persistence protocol.

**Limit.** Crash consistency is fail-closed but can cost a full hierarchy rebuild. Power-loss behaviour is not demonstrated.

### F8 — Segment read-lock lifetime
| | |
|---|---|
| Severity | Medium (performance under concurrent writes) |
| Status | **REPRODUCIBLE DESIGN RISK** (required invariant). Load impact **UNVERIFIED**. |

**Ownership trace.**
- `capture_segments` clones `LockedSegment` handles (`Arc<RwLock<Segment>>`). These keep segment *objects* alive across a holder swap; they do not freeze contents.
- `search_dli_research` then `try_read`s every segment (never waiting while holding others, so no lock-order deadlock), checks pointer identity against the holder, and keeps all guards through:
  - routing,
  - scoring (vector storage / id tracker / LMI postings, all behind `&Segment`),
  - per-segment materialization,
  - `resolve_pinned_candidates`.
- The resolver compares each hit's owner and version with `point_versions` across **all** members. This is sound only if no member changes between scoring and resolution, which requires simultaneous guards. Native Qdrant locks one segment at a time and does not need this cross-member check.

**Consequence.**
- Writers to any member (upserts to the Plain buffer, deletes in learned members, the optimizer's proxy installation) wait for the whole learned query.
- With a waiting writer, parking_lot's `try_read` fails and DLI rejects with `busy`, falling back to exact search. This is correct and visible in `fallback_reasons`, but not measured under write load.

**Change.** None to the scope; no shorter safe scope exists without per-segment MVCC or version snapshots. The F4 fix removed diagnostic I/O from this region. Existing deadlock and visibility controls: `dli_query_guard_and_native_upsert_complete_without_deadlock`, `..._delete_complete_without_resurrection`.

**Open.** Writer-latency and `busy`-fallback rates under a concurrent update stream.

### F1 — Plain-buffer scorer attribution
| | |
|---|---|
| Severity | Low |
| Status | **FIXED AND VERIFIED** (by `e043de079` plus this audit's tests) |

**Evidence.**
- `TaskScoringProfileGuard` sets a thread-local flag and counter. The whole route runs synchronously in one `spawn_blocking` closure, and the guard is dropped on every return path, including `?` and unwinding. Plain attribution is the before/after delta around the Plain `index.search`, so learned scorer work recorded on the same thread is excluded.
- Nested guards `assert!`, which would panic the attempt task. No nesting exists: exact-fallback workers are separate tasks.
- Elapsed time covers the Plain index search (scoring plus top-k), not materialization.

**Regression.**
- `dli_request_events_are_final_and_request_local_under_concurrency` (16 concurrent requests on 4 workers): every request reports `plain_scored == plain_live`, `original_scored == learned_live`, unique request ids, one event each.
- Consistent with continuation40's per-request assertion `plain_scored == 1536` (buffer size).

### F5 — Vacuous conservation assertion
| | |
|---|---|
| Severity | Low |
| Status | **FIXED AND VERIFIED** |

**Evidence.**
- `e043de079` passed `profile_request = true`, so the old `if let (Some, Some)` branch did execute. It would still have been silently skipped if profiling regressed.
- `two_native_routers_global_plan_and_per_segment_scoring` now `expect`s both profiles.
- Count assertions in the new and strengthened tests use fixture-derived values, not values derived from the counters themselves:
  - live counts via `available_point_count`,
  - matching ids via `has_point`,
  - "deleted postings consume budget but are not scored" (an underfilled attempt with `unique_offsets ≥ 1` and `original_scored == 0`).
- Quantized and original counts are covered by the two-router test (6/6/6), the degraded-member tests and the continuation40 harness assertions (`quantized_scored == unique_offsets`, `original_scored ≤ 2·128`).

### F11 — Environment variables and configuration semantics
| | |
|---|---|
| Severity | Low |
| Status at `be2e58f1e` | **CONFIRMED** (`DLI_QUANTIZED_SHORTLIST`) |
| Status after audit | **FIXED AND VERIFIED** |

**Evidence.**
- `DLI_QUANTIZED_SHORTLIST` was read on every request.
- `value.parse::<usize>().unwrap_or(0)` turned malformed or zero values into `Some(0)`. The route then rejected every query with `quantization_configuration`, so a typo silently disabled learned search.
- `DLI_REQUEST_SCORING_EVENTS` and `DLI_REQUEST_SCORING_PROFILE` had already moved to construction time in `e043de079`/`be2e58f1e`.
- `DLI_ROUTING_EVENTS` was still read per profiled request.

**Change.**
- `parse_env_quantized_shortlist` runs once in `DliResearchConfig::new`. Absent, zero or malformed values yield `None` (original scoring, the default) with a warning.
- `routing_output_path` is cached at construction.
- The benchmark switches modes only through the per-shard atomic overrides. No test mutates process environment.

**Regression.** `invalid_quantized_shortlist_environment_selects_original_scoring`.

## 2. Continuation40 validity (source `ceca71685`)

The latency pass in `dli_sisap_scaling_benchmark.rs::sisap23_65k_multilevel_collection_s0_vs_s2` at the executed SHA:
1. Calls `configure_local_dli_scoring_profile_for_test(false)` before the ABBA/BAAB loop.
2. Only switches the per-shard `quantized_shortlist` override between modes.
3. Times only `run_65k_query`. The oracle check and JSON row writes happen after the clock stops.

With profiling off at that SHA:
- `route_pinned_with_quantized_shortlist_profiled(..., false)` uses the unprofiled `score_candidates` / `score_candidates_with_quantization` paths.
- No `TaskScoringProfileGuard` is created.
- No routing or request event is written: both were gated on the profile flag.
- `DLI_NATIVE_ACCESS_EVENTS` was unset (`run-native-65k.sh`).

Therefore F3, F4 and F6 do **not** affect the reported continuation40 latencies, recalls or scorer counts. Its profile-pass assertions required `learned_delta == 1` and `fallback_delta == 0` per request, so the `"resolved"` label it checked coincided with the true final outcome for all 128 profiled requests.

Residual asymmetry: in S0 mode (override 0) the old code called `std::env::var("DLI_QUANTIZED_SHORTLIST")` per request, which takes a process-env lock. S2 did not. This is a negligible absolute cost (well under a microsecond against 5–8 ms) and slightly penalises S0. It is recorded, not corrected by rerun.

The continuation40 results stand within their stated limits: warm, 64 previously unreported but not proven untouched queries, no cold I/O, no incremental RSS, no matched build cost.

## 3. Commits (this audit)

| Commit | Change | Findings |
|---|---|---|
| `58ff5f9d7` | `lmi-scorer-profile` feature split; static LMI unprofiled under `dli-research` | F6 |
| `7a499ea92` | One final per-request event after guard release; `RouteTrace`; atomic JSONL lines; env parsing; strengthened tests | F2, F3, F4, F5, F11, plus the interleaved-JSONL defect |
| `593225584` | Destination quantization derived from current shortlist | F9b |
| `74bb36b63` | Per-member degradation to original scoring when derived quantization is absent | F9a |
| `277cf0d94` | Reject DLI activation with generic optimizer workers enabled | F7 |

**Additional defect found by the new concurrency test.** Concurrent JSONL appends interleaved (unparsable lines). The cause was `writeln!` through `Display`, which issues several `write(2)` calls. Each line is now one `write_all` on `O_APPEND`. Continuation40 was unaffected: its profile pass was sequential.

## 4. Test evidence

Environment: `--locked --offline`, PyTorch 2.5.1 environment, shared research target, features `testing,lmi-training,dli-research`. Logs are in `/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-41/`.

| Log | Scope | Result |
|---|---|---|
| `commit2-accounting-suite.log` | `dli_` + `native_exact_profiles` at the accounting commit | 32 passed / 0 failed / 7 ignored |
| `lifecycle-fixes-suite.log` | `dli_`, `native_exact`, `destination_quantization`, `lmi_lifecycle`, `worker_lifecycle` at `277cf0d94` | 35 passed / 0 failed / 7 ignored |
| `segment-feature-split.log` | segment `candidate_scoring_v120` + `lmi_static_v120` | `lmi-research`: 9+8; `lmi-scorer-profile` only: 7+8; all pass |
| `clippy-collection.log` | Clippy on `collection` lib+tests with DLI features | Diffed against lines changed by this audit: 5 warnings (`std::fs` instead of `fs_err`, one useless conversion), fixed in the lint commit. The many remaining warnings are pre-existing in research code. |
| `final-suite.log` | Same filter as `lifecycle-fixes-suite.log`, after the lint fixes | 35 passed / 0 failed / 7 ignored |

The intermediate commits `593225584` and `74bb36b63` were not separately compiled. The final tree containing them was tested.

Not run:
- the ignored release benchmarks,
- the workspace-wide suite,
- `cargo +nightly fmt`. Repository-wide formatting already fails on baseline research files, and reformatting them would obscure this diff.
