# Agent handoff: where the full picture lives (updated 2026-11-25)

This index is for an agent, or a person, who has to **continue** the research on the same path, **audit** it, or only
**analyse** results. Read it first. It lists where things are. It does not restate results: the result documents
are the authority, and this index names them.

Thesis: *Integration and Evaluation of Learned Indexes in Vector Database Systems* (submission 2027-03-10). The
author writes the thesis; agents advise, build and run experiments. Never claim a job ran when it did not, never
invent a measurement, and never move a preregistered threshold after seeing data.

## 1. Read in this order

| # | File | Why |
|---|---|---|
| 1 | `research/DEVELOPMENT_JOURNAL.md` (this branch) | The project's whole history, Parts I–XIII. §15 lists negative results; §17 says where evidence lives |
| 2 | `research/dli/DEVELOPMENT_JOURNAL.md` (this branch) | Short dated DLI log. The newest entries are at the end |
| 3 | `research/dli/emma_path/PHASE3_PREREGISTRATION.md` | **Frozen** at `6538a4483`. Hypotheses H5–H12, thresholds and arms. Never edit; amendments only, dated |
| 4 | `research/dli/emma_path/RESULTS_*.md` | Decided results: pilot, 10M, Phase 3 deletes, Phase 3 shift, Phase 3 Part B |
| 4a | `research/dli/emma_path/SHIFT_MECHANISM.md` and `DLI_BEST_RULE.md` | Why reuse fails under shift (bucket imbalance); the **frozen** rule that assembles DLI-best |
| 4b | Branch `research/policy-replay-20261123`, `research/dli/policy_replay/` | `mdn_code` against `proportional`: frozen preregistration, final results, manifest |
| 4c | `C:\Users\nicoo\Workspace\Learning stuff\Learned\CLI_AUDIT_AND_READING_GUIDE.md` | Audit of Jakubík's CLI thesis and code, and of Forgáč's DLI (the three different "DLI"s) |
| 5 | `research/dli/emma_path/MULTILEVEL_DIAGNOSIS.md` | Why E1 needs 2.4× static work: a structural cause, shown by the oracle-cost method |
| 6 | Branch `research/static-vs-dli-20261122`, `research/dli/static_qdrant_vs_dli/` | Second study: ordinary Qdrant static-LMI lifecycle against the DLI (§3) |
| 7 | `C:\Users\nicoo\Workspace\Learning stuff\Progress_Ladder\PROJECT_EXPLAINER.md` | A first-principles explainer and audit of both repositories, with ranked findings and code/doc disagreements |

## 2. Repositories and branches

| What | Where | Branch / state |
|---|---|---|
| Qdrant fork (all DLI/LMI work) | `~/work/qdrant` (WSL), remote `github` = `nico-occhino/qdrant` | `main`; `research/dli-emma-path-20261120` (Emma path, journals); `research/static-vs-dli-20261122` (static-vs-DLI). Both research branches are pushed and **not merged**. History is in `archive/*` and `exp/*` tags |
| PostgreSQL extension (native `lmi`/`dli` AMs) | `C:\Users\nicoo\Workspace\Learning stuff\PostgreSQL-Learned-Indexes` (the author works here) and a WSL clone `~/work/postgresql-learned-indexes` | `feature/native-dli`. The C: checkout was 1 commit ahead of GitHub (`1a63b16`) on 2026-11-23 |
| Admin, staging, archives | `~/research-admin/` | `experiments/<date>-<study>/` holds staging trees and submit logs; `archive/` holds backup indexes |
| Thesis (LaTeX) | separate repository | Chapter 1 drafted, Chapters 2–9 skeletons |

**Two sessions may share `~/work/qdrant`.** Run `git branch --show-current` before every commit. The journals are
maintained on the Emma branch. The static-vs-DLI branch has older copies of them; do not edit those.

## 3. Studies: code, documents, jobs, evidence

### 3.1 Emma path: a Sommerová-faithful DLI (branch `research/dli-emma-path-20261120`)

- **Documents:** `research/dli/emma_path/`.
  - Specification and amendments: `PREREGISTRATION.md`, `IMPLEMENTATION.md`, `REPRODUCE.md`.
  - Phase 3: `PHASE3_PREFLIGHT.md`, `PHASE3_PREREGISTRATION.md`.
  - Results: `RESULTS_PILOT.md`, `RESULTS_10M.md`, `RESULTS_PHASE3_DELETES.md`, `RESULTS_PHASE3_SHIFT.md`.
  - Job register: `PBS_MANIFEST.json`. Small derived JSON: `derived/`.
- **Tools:**
  - `phase3_plan.py`: the deterministic workload planner and gold.
  - `analyze_phase3.py` (tested by `test_analyze_phase3.py`): interpolation, paired bootstrap and decisions.
  - `analyze.py` (pilot and 10M); `prefix_gold.py`; `consolidation_sim.py`.
  - `pbs/phase3.pbs` (modes preflight, preflight1m, primary, with `SCENARIO=`); `pbs/emma.pbs` (pilot and 10M).
- **Code (Rust):**
  - `lib/collection/src/collection_manager/`:
    - `dli_schedule.rs`: Bentley–Saxe plans, Greedy, vacuum, consolidation.
    - `dli_maintenance.rs`: planning, the build-CPU permit `QDRANT_DLI_BUILD_CPUS`, topology validation.
    - `dli_native_route.rs`: pinned routing and the dead-posting trace.
    - `dli_bucket_plan.rs`: routing policies, `mdn_code`, reserved selection, isotonic calibration.
  - Search: `lib/collection/src/collection_manager/segments_searcher/dli_research.rs`.
  - Experiment drivers, `lib/collection/src/tests/lmi_lifecycle/`:
    - `dli_sisap_scaling_benchmark.rs`, with `dli_phase3_workload`;
    - `dli_selection_frontier.rs`: `run_frontier`, the policy list, and the `proportional` control at `run_proportional`.
- **Remote evidence (MetaCentrum):** `/storage/budejovice1/home/nico/lmi-metacentrum/dli-emma-path-20261120/`.
  - `native-<commit12>/` holds staged source and inputs.
  - `results/<job-name>-<jobid>.pbs-m1.metacentrum.cz/` holds per-job outputs.
  - `bin-cache/` holds binaries keyed by a hash of the compiled source.
- **Local evidence:** `C:\Users\nicoo\Workspace\Research-Results\2026-11-20-dli-emma-path\`. Each job directory
  carries its own `SHA256SUMS`.
- **Staging:** `~/research-admin/experiments/2026-11-20-dli-emma-path/staging-<commit12>/`. `stage_and_submit.sh` is
  stale (old host and socket); staging is done by hand through perian.

### 3.2 Static Qdrant LMI lifecycle against the DLI (branch `research/static-vs-dli-20261122`)

- **Documents:** `research/dli/static_qdrant_vs_dli/`.
  - Study files: `BASELINE_AUDIT.md`, `DESIGN.md` (§6 is the concurrent-service phase), `METRICS_SCHEMA.json`,
    `REPRODUCE.md`.
  - `PREREGISTRATION.md`: a **DRAFT** with amendments A1–A5. It is not frozen.
  - Results: `PILOT_RESULTS.md`, `PILOT2_RESULTS.md`. Job register: `PBS_MANIFEST.json`.
- **Tools:**
  - `analyze_static_vs_dli.py` (tested by `test_analyze_static_vs_dli.py`); `summarize_service.py`.
  - `pbs/static_vs_dli.pbs`: modes pilot, pilot2, primary (prepared, **not approved**) and service.
- **Code:** in `dli_sisap_scaling_benchmark.rs`:
  - `DLI_P3_SYSTEM=static` and the `DLI_P3_STATIC_*` settings (`p3_setup`);
  - `dli_service_workload`, the concurrent readers and writer.

  `dli_selection_frontier.rs::run_static_frontier` replays S, mirroring native per-segment search.
- **Remote evidence:** the same `dli-emma-path-20261120/results/` directory, under `static-vs-dli-pilot-*` and
  `static-vs-dli-pilot2-*`.
- **Local evidence:** `C:\Users\nicoo\Workspace\Research-Results\2026-11-22-static-vs-dli\`.
- **Staging:** `~/research-admin/experiments/2026-11-22-static-vs-dli/`.

### 3.3 PostgreSQL replication

`docs/research/` in the PostgreSQL repository holds:
- `DLI_10M_QUALITY_RESULTS.md` (job 25859979: the 2.35× penalty reproduces);
- `PREREGISTRATION_TEMPORAL_QUALITY.md`;
- `AUDIT_2026-11-22.md`.

Remote: `/storage/budejovice1/home/nico/lmi-metacentrum/postgresql-native-temporal-dli-20261121/`.

### 3.4 Shared data

- **Corpus:** SISAP23 LAION2B CLIP-768, Float16, 10,120,191 rows, sha256 `f5868d6e…`.
  - Remote: `/storage/budejovice1/home/nico/lmi-metacentrum/results/sisap23-matched-final-20261028-01/corpus.f16le`.
  - Local: `~/work/pg-dli-env/inputs/corpus.f16le`. The first 1M rows (`head -c 1536000000`) are the 1M studies'
    corpus.
  - Queries: `dli-v120-20261117/inputs/` (remote).
- **Candidate datasets for the multi-dataset routing study,** f32 fbin under
  `/storage/brno12-cerit/home/prochazka/lvi/CANDOR-Bench-images/raw_data/`:

  | Dataset | Size × dim | Notes |
  |---|---|---|
  | openimages | 1M × 512 | |
  | glove | 1,192,514 × 100 | also `_reference/glove-100-angular.hdf5` |
  | coco | 100K × 768 | |
  | sift, msong, sun | — | L2 only, so unusable for this cosine study |

  The project folder `/storage/brno12-cerit/projects/fi-lmi-data` is permission-denied, and no VIBE copy exists
  there.

## 4. State on 2026-11-25, and what is open

| Item | State | Next action |
|---|---|---|
| Phase 3 deletes (H6, H6b, H7) | decided, `RESULTS_PHASE3_DELETES.md` | — |
| Phase 3 shift (H5, H12) | decided, `RESULTS_PHASE3_SHIFT.md`. Mechanism in `SHIFT_MECHANISM.md`: reuse fails through bucket imbalance, and consolidation helps through granularity | optional: an oracle-cost replay to separate ranking from imbalance |
| Phase 3 Part B (H9, H10, H5 insert-only) | decided, `RESULTS_PHASE3_PARTB.md`. H9 not supported (0.841), H10 not supported (0.929). Job 25859945 was cancelled by the author after E1C; E1C50 is incomplete | the H11 oracle-cost decomposition can be done offline |
| DLI-best | **final**, `DLI_BEST_RULE.md`: E1 + vacuum + consolidation ρ = 0.25 + **proportional** allocation (tie-break: calibrated against proportional equivalent, R 1.0375, `research/dli/allocation_tiebreak/RESULTS_TIEBREAK.md` on branch `research/allocation-tiebreak-20261124`) | integration pilot of the combined configuration (needs the driver to combine maintenance variants) |
| Policy replay, `mdn_code` against `proportional` | **final**, `policy_replay/RESULTS_POLICY_REPLAY.md` (policy-replay branch). **Proportional recommended**: better on insert-only 0.908, constant-R 0.784, Shift `ids` 0.887 and Shift `ood` 0.693; equivalent on Shift `mq` 1.023 | done: the tie-break chose proportional |
| Static-vs-DLI pilot2 | completed, not accepted. Granularity explains the S advantage. The known-answer failure is explained as exact-duplicate ties (`PILOT2_RESULTS.md` addendum) | **superseded by the final-benchmark plan.** S1 and the tie-aware gate fold into it |
| Final benchmark (author decision, 2026-11-23) | planned. Qdrant hosts IVF, HNSW, E1, DLI-best, CLI and at most one CLI variant. PostgreSQL is a replication subset | after the CLI design note (mutability options: an appendable learned segment, or copy-on-write); the author is reading Jakubík's thesis first |
| CLI implementation | not started. Audit and reading guide done | wait for the author |
| PostgreSQL R3 replication | run by a separate agent in the PostgreSQL repository. Prompts were given to it in chat (pilot done; freeze and primary prompt issued, with 10M conditional) | read `docs/research/R3_*.md` there |
| Multi-dataset study | folded into the final benchmark (one extra OOD dataset); not started | needs a `DIM` parameter, a data converter and query pools |

**Evidence on C: (all verified against sha256 lists computed on storage, in
`~/research-admin/experiments/2026-11-20-dli-emma-path/copy-to-C-20261123/`):**
- `Research-Results\2026-11-20-dli-emma-path\`: Phase 3 Shift and insert-only;
- `Research-Results\2026-11-22-static-vs-dli\`: pilot 2;
- `Research-Results\2026-11-23-policy-replay\`: the check job, the five primaries and the `ood` rerun 25908996;
- `Research-Results\2026-11-24-allocation-tiebreak\`: the tie-break check (25914476) and primary (25931720).

## 5. How to work here safely

- **MetaCentrum access** goes through perian.
  - The SSH ControlMaster socket is `~/.ssh/metacentrum-control`. A stale socket file silently disables multiplexing:
    delete it before starting a new master.
  - The master dies every few hours. Only the author can restore it, by password login (`kinit` is not installed in
    WSL):
    `rm -f ~/.ssh/metacentrum-control && ssh -M -S ~/.ssh/metacentrum-control -o ServerAliveInterval=60 -fN nico@perian.metacentrum.cz`.
  - `qextend <job> HH:MM:SS` extends a running job: 20 extensions per 30 days, by the author. The zenon nodes are
    about 2× slower, so budget walltime for them.
- **GitHub push** needs `SSH_AUTH_SOCK=~/.ssh/agent.sock`; if it is dead, the author restarts it with
  `eval "$(ssh-agent -a ~/.ssh/agent.sock)" && ssh-add`. Push only with the author's permission. Never force-push.
- **Commit by explicit paths, never `git commit -a`.** Parallel sessions share `~/work/qdrant`. The working tree shows
  only the checked-out branch's files, so check `git branch --show-current` first.
- **Jobs.**
  - Every job carries `COMPLETE.json` and `SHA256SUMS`. PBS exit 0 alone is not acceptance; the gates in each
    preregistration are.
  - Single-threaded replay dominates walltime: 16 CPUs were 3–7% used. Request fewer CPUs, or parallelise replay.
- **Data handling.** Never overwrite artifacts. No large raw data in git. Results are kept on C:, with checksums
  verified after copying.
- **Volatile:** the session scratchpad (`/tmp/claude-1000/...`) is cleared between days. Nothing there is evidence.
- **Out of bounds unless the author asks:**
  - modifying PostgreSQL from the Qdrant studies;
  - implementing the CLI;
  - rewriting the native LMI algorithm to improve a benchmark;
  - deleting branches, worktrees, datasets or evidence.
