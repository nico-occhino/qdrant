# CLI selection pilot (preregistration)

**Status: FROZEN (2026-11-28; the author asked for the pilot after freezing spec v3)**, at the commit that adds this
line, before any pilot job runs.

- **The algorithm.** [CLI_SHARED_SPEC.md](CLI_SHARED_SPEC.md), frozen v3 (`a2261ddde`, SHA-256 `8ad6afc0…`).
- **The parent protocol.** The final benchmark's §8 selection pilot (`research/dli/final_benchmark/PREREGISTRATION.md`)
  and its amendment A5 (CLI enters later on the same seeds and panels).

## 1. Question
Which b_max ∈ {500, 1,000, 2,000} does CLI use in the final benchmark? The pilot also records bracketing and
runtime, which set the CLI primary walltimes.

## 2. Implementation used, and its declared limit
- **What runs.** The Stage A core (`lib/segment/src/index/lmi_index/cli_core.rs`, harness
  `tests/cli_stage_a.rs::cli_stage_a_plan`): the whole algorithm of spec v3, in memory, over the plan's stored
  representation (rows f32-normalized, then Float16, `CLI_STORED_F16=1`).
- **What it does not include.** Qdrant's segment lifecycle (Stage B).
- **Why Stage A may select b_max.** The choice depends only on the algorithm's routing and partition, not on
  persistence or concurrency.
- **The condition on the final arm.** Before any CLI primary data, the native Stage B arm must reproduce Stage A on
  the pilot seed, as a gate in the later CLI amendment: per-batch bucket counts, maintenance paths and G1 curves
  within a frozen tolerance. If it does not, the pilot is repeated on Stage B.

## 3. Runs (seed 20261202, validation panels only, no test panel read)
| Workload | Plan | Panels | b_max |
|---|---|---|---|
| W1 Insert-only | `W1-20261202` | `val` (64) | 500, 1,000, 2,000 |
| W2 Constant-R | `W2-20261202` | `val` (64) | 500, 1,000, 2,000 |
| W3 Shift | `W3-20261202` | `mq_val`, `ids_val`, `ood_val` (64 each) | 500, 1,000, 2,000 |

- **The arm.** CLI: MLP routing, the entropy trigger and spec v3 defaults (hidden 128; 30 build and 20 fine-tune
  epochs; replay ⌊0.02 · live⌋; seed 42; growth at 10×).
- **The jobs.** Nine, one per (workload, b_max), each with 8 CPUs and a 24 h walltime.
- **CLI-I is not piloted.** It takes the chosen b_max.
- **The grid.** Budgets are the final benchmark's grid plus 2,000,000 (gate only).

## 4. Choice (as final-benchmark §8)
- **The rule.** The b_max with the lowest mean validation W@0.90 (G1 recall, first measured crossing, interpolated,
  never extrapolated) on W1 batches 1–20, over the batches where every candidate attains.
- **Ties** go to the lower total fine-tune seconds on W1. Stage A has no latency measure, which is declared.
- **Recording.** The choice is recorded in a dated CLI amendment, from validation data only.

## 5. Gates per job
- **Inputs.** The plan's 363 or 729 file hashes, its seed and scenario, and the 1M corpus prefix SHA-256
  `6e1c7237…`.
- **Unit tests.** `cli_core_synthetic_loop` passes inside the job (the `cli` known answers are library tests, run before staging).
- **During the run.** G2 membership inside every maintenance (`check_membership`), which aborts the run on
  failure.
- **After the run.** Completeness (every batch, panel, query and budget); the G1 known answer (mean G1 Recall@30 ≥
  0.999 at the exhaustive budget on every batch and panel); and monotone mean work along the grid.
- **Analysis.** `analyze_pilot.py`, with its known-answer test, committed before the jobs.
- **Failures.** A failed gate is reported, never silently rerun.

## 6. Reported, not decided
Bracketing per (workload, b_max, panel), the per-batch maintenance record (spec §7), fine-tune and wall seconds,
and peak memory.

## 7. Addendum A1 (2026-11-28, before any centroid-CLI pilot data): the centroid-routed CLI variant
- **The author's decision.** A second, exploratory CLI variant, **C-CLI**: spec v3 maintenance (splits, DEDRIFT,
  stored-label membership) with routing by each bucket's spherical k-means centroid instead of the MLP.
  - **Maintenance and triggers.** Centroids are replaced at each split or recluster, with no training. The drift
    trigger is growth only, because entropy is undefined for this router (`cli_core.rs`, `RoutingKind::Centroid`).
  - **Its basis.** The Stage A diagnostic `RESULTS_CENTROID_CONTROL.md`, which is descriptive. This variant is
    labelled exploratory everywhere it is reported.
- **Its pilot.** It mirrors §3–§5. The same three workloads and three b_max candidates, the same choice rule and gates,
  nine jobs with `ROUTING=centroid`. Its b_max is chosen independently of CLI's.
- **What stays.** The MLP pilot (§3) is unchanged. Its jobs are already running from staging `21cf4eaa1`.
