# Baseline audit: Qdrant's static-LMI lifecycle (arm S) against the DLI (arm D)

This is a read-only audit made before any change, on 2026-11-22, in `~/work/qdrant`.
- **Starting state:** branch `research/dli-emma-path-20261120` at `5e9107bdd`, with a clean worktree.
- **Study branch:** `research/static-vs-dli-20261122`, created from that commit. It is not pushed.
- **Upstream base:** Qdrant v1.20.2. The merge-base with tag `v1.20.2` is `23065a0bf685`.
- **Other branches:** `main` is at `c90ac6059`, and this branch contains it.

## 1. Ordinary static-LMI lifecycle (arm S)

**One model config for every optimizer.** `VectorParams.lmi_config` flows unchanged into `SegmentOptimizerConfig` for
every optimizer (`lib/collection/src/optimizers_builder.rs:195–230`). Merge, indexing, config-mismatch and vacuum
(`optimizers_builder.rs:316–`) therefore all build their output segment as `Indexes::LmiTrained(lmi_config)` through
`SegmentBuilder`. They check that training is available and refuse quantization and GraphInline
(`lib/shard/src/optimizers/segment_optimizer.rs:168–184`).

**Gating.** A segment is indexed when its vector bytes reach `indexing_threshold_kb` *and* it has at least
`n_buckets` live vectors (`lib/shard/src/optimizers/indexing_optimizer.rs:72–83`). The config-mismatch optimizer
applies the same bucket guard (`config_mismatch_optimizer.rs:101–112`).

**One bucket count for every segment.** The bucket count is fixed per collection. A small segment gets the same
`n_buckets` as a large one, so its buckets are smaller. Fixing the bucket count, segment count and indexing threshold
for S is therefore a design decision (§3 of DESIGN.md), not a free parameter.

**Merge target.** `default_segment_number` sets how many segments the merge optimizer aims for (auto = number of
CPUs when 0). The maximum segment size is threads × 256,000 KB (`lib/shard/src/optimizers/config.rs:17,226–233`):
about 2.7M Float16-768 vectors at 16 threads.

**Vacuum is part of S.** `OptimizersConfig::fixture()` sets `deleted_threshold = 0.1` and
`vacuum_min_vector_number = 1000` (`optimizers_builder.rs:123–135`). S therefore rewrites a segment that is more than
10% deleted, dropping its dead points, and retrains its LMI.

**Search.** Qdrant searches every segment. An LMI segment routes to its configured `nprobe` buckets and widens by
doubling until it has `min(top, live)` hits (`lib/segment/src/index/lmi_index/mod.rs:470–500`). A Plain segment is
scanned exhaustively. Results are merged across segments.

**No per-query `nprobe`.** Non-default search params send an LMI to Plain (`lmi_shell_v120.rs`,
`exact_filter_and_nondefault_params_use_plain`). The only research switch is `LMI_RESEARCH_ROUTER`
(`training.rs:346`). A recall-versus-work sweep for S therefore needs a replay over frozen segments, like D's.

**Existing harness.** `collection_config(false)` in `lib/collection/src/tests/lmi_lifecycle/dli_sisap_scaling_benchmark.rs`
sets `indexing_threshold = 20,000 KB`, `default_segment_number = 1` and `max_optimization_threads = 1`, and waits for
*full* indexing. Under a live lifecycle, the newest appendable segment legitimately stays Plain, so "settled" must mean
collection Green with optimizer status Ok, not "everything indexed".

## 2. DLI maintenance and search (arm D)

- **Maintenance.** It is triggered by `observe()` after every update (`collection_updater.rs:111`). It plans with
  live counts (`dli_maintenance.rs:676`, `next_plan` `:1081`) and builds through `execute_optimization` with a
  `DliOptimizationStrategy`, which re-derives the plan under the optimizer's locks.
- **Generic optimizers are disabled for D.** `collection_config(true)` sets `max_optimization_threads = Some(0)`
  (`dli_sisap_scaling_benchmark.rs:161`) and gives no `lmi_config`. The DLI owns every segment transition. This is the
  key asymmetry: **D has no Qdrant merge, indexing or vacuum optimizer**, and S has no DLI maintenance. Enabling both
  in one collection is untested and would let two owners rewrite the same segments.
- **Search.** `search_dli_research` (`segments_searcher/dli_research.rs:261`) ranks buckets globally across levels
  plus the buffer (`mdn_code`), scores them, and resolves newer copies. It falls back to native exact search for
  filters, named vectors, payload or vector materialization, or an unsafe topology (`:313–326`, `:700`).
- **Build CPUs.** `QDRANT_DLI_BUILD_CPUS` (default 1) sets the CPUs per DLI build (`dli_maintenance.rs:1067`). S builds
  take the optimizer's `ResourceBudget` (all CPUs available to the job).

## 3. Lifecycle differences that the comparison must expose, not hide

| Aspect | S (ordinary optimizer) | D (E1) |
|---|---|---|
| When indexes are rebuilt | merge, indexing and vacuum decisions (size thresholds, segment count, 10% deleted) | every buffer fill (b = 5,000) by Bentley–Saxe; never on deletes |
| Model per rebuild | always trained fresh (k-means + MLP) | fresh only for empty target levels; otherwise the target's router is reused |
| Bucket count | fixed per collection | aⁱ per level |
| Deleted points | dropped when a segment is rewritten (vacuum at 10%) | stay as dead postings until the level is rewritten |
| Search | per-segment nprobe; Plain segments scanned fully | one global candidate budget over levels and buffer |
| Writes during maintenance | concurrent (the optimizer runs in the background) | the harness drains synchronously after each buffer chunk (Phase 3 protocol) |

## 4. Reusable harness (extended, not duplicated)

- **`research/dli/emma_path/phase3_plan.py`** provides deterministic plans for insert-only, shift (10-cluster chain,
  holdouts, OOD), constant_r and constant_o. Each batch has an insert/delete list, a live-set SHA-256 and tie-exact
  gold. Its gold is verified equal to `prefix_gold.py`.
- **`dli_phase3_workload`** (driver) gains `DLI_P3_SYSTEM=static`. It uses the same plan, the same ledger check after
  every batch, the same flushed-copy replay and the same output schema.
- **`dli_selection_frontier::run_frontier`** is used for D, and the new **`run_static_frontier`** for S, which mirrors
  native per-segment search including widening.
- **`analyze_phase3.py`** supplies the interpolation, bootstrap and CI helpers, imported by `analyze_static_vs_dli.py`.

## 5. Missing instrumentation (gaps at this commit)

- **Live-request latency and QPS under concurrent writes and background optimization.** The driver evaluates by replay
  on settled snapshots. A concurrent reader/writer phase is required before the primary run (DESIGN.md §6).
- **Per-arm CPU-seconds, RSS, cgroup memory and I/O bytes.** Only job-level `/usr/bin/time -v` and optimizer event logs
  (`QDRANT_OPTIMIZATION_EVENTS`, `QDRANT_LMI_BUILD_EVENTS`) exist.
- **Maximum query stall during maintenance.** Not measured.
- **Router-inference, bucket-selection and scoring time split for S.** The replay records only total elapsed time and
  work. D's split exists in request events.

## 6. Expected resources (from measured runs)

- **D at 1M:** E1 ingestion of 1M took 166–951 s per scenario, and a 3-budget replay of 1,032 queries took 55–300 s
  (preflight 25857840, `PHASE3_PREREGISTRATION.md` §9c). Peak memory at 1M is far below 128 GB. At 10M, E1 peaked at
  167 GB (job 25827532).
- **S1 at 1M:** one segment, retrained about every 13K inserts (20,000 KB ≈ 13.3K Float16-768 vectors), so on the order
  of 70 full retrains per 1M inserted. This has not been measured yet; the pilot measures it.
