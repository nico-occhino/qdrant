# Emma path — implementation notes (Phase 1)

Branch `research/dli-emma-path-20261120`. Design rationale and hypotheses are in
[PREREGISTRATION.md](PREREGISTRATION.md). This file records what changed in the code, why,
and how it is tested. All changes are behind the existing `dli-research` /
`lmi-training` features. Every default reproduces the previous behaviour exactly.

## Reference used

Sommerová's thesis §3 and her released code (`Emma_DLI.zip`, LGPL; no code copied). Three
facts from her code that the thesis text does not state:
1. Compaction into a **non-empty** level never retrains. Vectors are routed by the level's
   existing model (`LearnedMetricIndex._train_procedure`). Retrain strategies apply only
   when the target level is empty.
2. The MDN normalisation multiplies by `arity**level_idx` with a **0-based** index, i.e.
   P·aⁱ⁻¹, not the thesis's P·aⁱ.
3. Model: hidden 128 (`N_NEURONS`), 10 epochs, Adam 1e-3, batch 256, sample
   `min(n, max(100K, 40·k))`, Faiss k-means default 25 iterations.

## Changes

### Segment: build with a reused router (`lib/segment/src/index/lmi_index/`)

| File | Change |
|---|---|
| `config.rs` | `LmiConfig.reuse_router: Option<u64>` (serde default, omitted when `None`, so persisted configs and golden digests are unchanged). |
| `routing.rs` | `MlpRouter::fingerprint()`: FNV-1a over layer kinds, shapes and exact f32 bits. It is a content identity, not a security property. |
| `training.rs` | `TrainedRouter::from_native` (previously test-only): a Torch mirror of a trained two-layer router. Reused routers then assign postings through the same tie-safe path as fresh ones. |
| `build.rs` | When `reuse_router` is set, `build_trained` takes the router with that fingerprint from the build's source indexes (`VectorIndexBuildArgs::old_indices`). It skips sampling, k-means and MLP training, checks the shape against the config, and routes every eligible point. If the router is absent, the build fails rather than retraining. Build events gain `n_buckets`, `hidden_dim`, `epochs`, `reused_router` and `router_fingerprint`. |

The destination segment's persisted config keeps `reuse_router = Some(fp)` as provenance.
It is consistent with the saved LMI metadata, so the segment reopens normally.

### Collection: per-level model and router reuse (`lib/collection/src/collection_manager/`)

| File | Change |
|---|---|
| `dli_schedule.rs` | `LevelBuckets {Fixed, ArityPower}` and `LevelModelPolicy {buckets, min_samples_per_bucket, reuse_target_router}`. `fresh_config` gives level *i* aⁱ buckets (capped at 65,536), a sample floor of `min_samples_per_bucket·B` (capped at 1M), and clamps `nprobe`. |
| `dli_maintenance.rs` | `DliMaintenance.level_model`. `destination_model()` picks the destination model per job. With reuse and a non-empty target, it uses the target's own LMI config naming the target's router fingerprint. It requires exactly one trained target member and otherwise fails closed, recording an optimizer error with nothing scheduled. In every other case it uses `fresh_config(model, arity, target_level)`. Both the native and the private build path use it. `DliPublicationEvent` is now `Serialize`. |
| `dli_bucket_plan.rs` | `RoutingPolicy::{ModelDrivenNormalized, ModelDrivenNormalizedCode}` and `select_model_normalized`. A learned bucket scores P·B_m (thesis) or P·B_m/a (code), where B_m is the member's bucket count. Plain scores 1. Whole buckets are taken until the budget is met. |
| `dli_native_route.rs` | Dispatches the two new policies. |
| `../config.rs` | `DliExperimentalConfig.level_model` (serde default = legacy). |
| `../shards/local_shard/search.rs` | Installs `level_model` on activation. |

Not implemented (Phase 3 only): delete-triggered underutilisation rebuild (C_min, greedy).
Qdrant keeps soft deletes.

### Harness (`lib/collection/src/tests/lmi_lifecycle/`)

- `dli_sisap_scaling_benchmark.rs`:
  - `DLI_ARM=G0|E1|E1R` selects the preregistered arm; unset means G0, the previous behaviour.
  - Ingests in chunks of `min(250K, b)` and waits for the publication after every chunk that fills the buffer.
  - Accepts populations ≥ 100K, but the caller must stage population-specific gold.
  - `DLI_BENCH_BUDGET` sets the collection-pass budget.
  - The summary records the arm, the model, the level rule and every publication event.
- `dli_selection_frontier.rs`: adds the `mdn` and `mdn_code` policies. `DLI_FRONTIER_STATIC_MODEL` (JSON `LmiConfig`) sets the static reference model. A template's `reuse_router` is always cleared.

## Tests added

| Test | What it proves |
|---|---|
| `segment/tests/dli_native_topology.rs::reused_router_keeps_target_assignments_and_routes_new_points` | Merged segment keeps the router byte-for-byte; old points keep their buckets; new points get the router's top-1 bucket; reopen is identical; unknown fingerprint fails. |
| `dli_schedule::tests::level_model_policy_sizes_buckets_and_samples_per_level` | Legacy policy is the identity; aⁱ buckets, 40·B floor, 1M cap, overflow rejection, reuse identity cleared on fresh configs. |
| `dli_bucket_plan::tests::model_normalized_scales_by_member_bucket_count_and_code_variant_divides_by_arity` | Thesis vs code ordering relative to the buffer; empty buckets skipped; budget is a minimum. |
| `dli_sisap_scaling_benchmark::emma_level_rule_and_router_reuse_in_configured_collection` | Configured collection, b = 40, a = 4, 22 publications for each of E1 and E1R. Every learned level has aⁱ buckets and the planned population. Non-empty-target compactions keep the router (E1, 15/15) or retrain it (E1R, 15/15). Full-budget search equals exact search; 6/6 learned, 0 fallbacks. |

Regression run before commit (local, WSL, torch 2.5.1):
- `segment`: all seven v120 suites listed in `research/BUILD_ENVIRONMENT.md` plus the `lmi` lib tests.
- `collection --lib -- lmi dli` (43 passed).
- `shard --test dli_view` with `segment/lmi-training` (5 passed).
- `segment --features lmi-research --tests` check.

## Phase 3 (in progress, 2026-11-22)

Tracked against [PHASE3_PREREGISTRATION.md](PHASE3_PREREGISTRATION.md) §7 and §10.6.

| # | Component | Status |
|---|---|---|
| M1 | Underutilisation: `HierarchyConfig::minimum_capacity`, Cmin(i) = b·a^(i−1) | done |
| M2 | Greedy rebuild: `HierarchyConfig::plan_rebuild` (her §3.4.2 target rules 1–3; Greedy upward accumulation stops at the first level that does not fit; deepest underutilised level first; level 1 is never moved into the buffer) | done (planner only; native execution is M4) |
| M10 | Consolidation: `MaintenancePolicy::GeometricConsolidating { rho_permille }` overrides a Bentley–Saxe plan into a level above the deepest non-empty level D when its output would exceed ρ·live(D) and buffer + all levels above D fit D | done (planner); runs through the existing native path, which reuses D's router |
| M3 | Dead postings: `NativeRouteResult.dead_posting_offsets` (soft-deleted point or vector, or out of range), in request events (`dead_offsets`) and frontier rows | done; test `dli_learned_resolution_survives_delete_and_reinsert` checks it against the scored hits, the all-deleted request-event test checks dead = unique |
| M4 | Native rebuild: `DliExperimentalConfig.greedy_rebuild`; `dli_maintenance::next_plan` plans a rebuild when no compaction is due; rebuild plans (`buffer_points = 0`) leave the buffer out and are re-derived by `plan_rebuild` in topology validation; `local_dli_observe_for_test` re-plans without an update | done; test `greedy_rebuild_after_deletes_in_configured_collection` (random and oldest 70% deletes, b = 40, a = 4: 2 rebuilds each, no level below Cmin, rebuilt destinations hold no deleted points, learned search = exact over live points, 0 fallbacks) |
| M5 | Workload driver `dli_phase3_workload` (ignored test): executes a plan, inserts in buffer-sized chunks, deletes, `settle_dli_maintenance` fixed point, **live-set SHA-256 against the plan ledger every batch**, flushed copy, per-batch replay | done; smoke runs on synthetic 20K plans (Constant-R + Greedy, Shift, Insert-only + consolidation) pass every ledger check |
| M6, M7 | `phase3_plan.py`: scenarios, shift chain and holdouts, query pools, tie-exact live-set gold | done; gold of an insert-only batch equals `prefix_gold.py` exactly (with ties); holdout and OOD rows asserted never inserted; deterministic for a seed |
| M8 | `dli_selection_frontier::run_frontier(FrontierRun)` callable per batch; the old env-driven test is a thin wrapper | done |
| M9 | `analyze_phase3.py` (Part A batch-stratified paired bootstrap, TOST, inconclusive rule, Bonferroni sensitivity; Part B work-ratio; `--precision` for preflight) | done; `test_analyze_phase3.py` known-answer test (Δ = 3.0 pp supported; equivalence detected and rejected; work ratio 0.6) |
| PBS | `pbs/phase3.pbs` (MODE preflight / primary with SCENARIO) | written, not yet run |
| M11 | Calibrated level-aware allocation: `dli_bucket_plan::{IsotonicMap, LevelCalibration, select_calibrated}` (pool-adjacent-violators fit; buckets by g_level(P) / occupancy, buffer by k / N; whole buckets to the budget); `dli_native_route::route_pinned_calibrated` on the shared routing/scoring/resolution path (`route_pinned_traced` now delegates to `route_pinned_selected` with an optional selector) | done; tests `isotonic_fit_matches_hand_computed_pav` (hand PAV, ties, monotone on random input), `calibrated_selection_is_greedy_value_per_cost_with_whole_buckets` (equals a reference sort, whole-bucket budget, empty buckets skipped, missing level rejected) |
| M12 | Per-snapshot calibration fit in the replay (`fit_calibration`: each member's top-64 buckets of every calibration query, y = gold top-30 in the bucket); `FrontierRun.calibration_set`, `DLI_P3_CALIB_SET`; policy `calibrated`; `calibration.json` per batch | done; PBS insert_only runs fit on rows 1000–2999 |

Tests in `dli_schedule::tests`:
- `consolidation_reproduces_the_planner_simulation`: final levels and write amplification equal `consolidation_sim.py` at 10.12M for ρ = 0.25 and 0.5; ρ = 1 equals Bentley–Saxe.
- `consolidation_boundary_fit_guard_and_shape`: the strict ρ boundary, the fit guard, plans already targeting D, an empty hierarchy.
- `greedy_rebuild_follows_the_three_target_rules`: each rule, empty intermediate levels, the Greedy stop, deepest-first, the level-1 exclusion, over-capacity rejection.

Collection `--lib -- dli_ lmi`: 47 passed (after M4).
