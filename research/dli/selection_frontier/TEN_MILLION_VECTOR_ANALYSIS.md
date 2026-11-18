# H-10M: static LMI versus geometric DLI candidate selection at 10.12M vectors

**Job:** `25677858.pbs-m1.metacentrum.cz` (`dli-10m-frontier`), finished 2026-11-17 22:26 CET.
**Analysis:** written 2026-11-18, read-only, from the job's raw query-level records.
**Status of this document:** validated analysis. It supersedes the job's own `analysis-summary.txt`, whose matched-work section is biased at 10M (§5.2).

## Contents
1. [Summary](#1-summary)
2. [Experimental setup and provenance](#2-experimental-setup-and-provenance)
3. [Quality checks](#3-quality-checks)
4. [H-10M decision](#4-h-10m-decision)
5. [Recall versus work](#5-recall-versus-work)
6. [Where the work goes: per-level and Plain-buffer analysis](#6-where-the-work-goes-per-level-and-plain-buffer-analysis)
7. [Comparison with 65K, and correction of the 65K interpretation](#7-comparison-with-65k-and-correction-of-the-65k-interpretation)
8. [Runtime and resources](#8-runtime-and-resources)
9. [Scientific interpretation](#9-scientific-interpretation)
10. [Limitations](#10-limitations)
11. [Next research priorities](#11-next-research-priorities)
12. [Locations](#12-locations)

---

## 1. Summary

1. **The job completed every planned stage.** Stages: verified inputs, offline build, full 10,120,191-vector geometric DLI ingestion (40 native publications), in-job static LMI build, and replay of 288 queries × 4 policies × 7 budgets = 8,064 records. Every record is present, none are duplicated, and all checksums verify (§3).

2. **H-10M is supported** under its preregistered rule.
   - Static LMI reaches mean test Recall@30 = 0.90 at **166K** scored vectors per query (1.64% of the collection).
   - Geometric DLI needs **305K** with the default `slot` policy and **285K** with `proportional`.
   - Static therefore needs **45.4% less work than slot** (paired bootstrap 95% CI 32.3–50.9%) and **41.7% less than proportional** (30.4–47.4%).
   - Neither curve crosses static anywhere in the measured range.
   - The margin over the 30% threshold is small at the lower CI bound. It relies on interpolating between budgets spaced 2× apart (§4.3).
   - It is not robust to the routing policy: T=2 density DLI is only 23.3% (7.2–34.0%) behind.

3. **The gap comes from the unindexed Plain buffer, not from the independent per-level routers.**
   - At equal *learned* work, the four DLI levels together select gold neighbours at least as efficiently as the single static model.
   - With proportional allocation, recall equals static at every equal budget, and the work difference is exactly the Plain buffer's 120,191 vectors.
   - The Plain buffer is scored as one indivisible unit. `slot` gives it a fixed priority of 1.0, so it is selected from a 40K budget onward and its 120K vectors dominate mid-range work. The rule is in the code (§6.1); the cost is measured.
   - The 1.19% of the collection held in Plain costs 72% of static's work at the target.

4. **The 65K conclusion that "router ranking is the main DLI inefficiency" does not hold as a statement about the static–DLI gap.** Re-analysed with the same corrected method (§7), the 65K data shows the same Plain-dominated pattern. Router ranking costs about 3–5× over the gold-bucket oracle, but static's own router has the same overhead.

5. **The reported 7% CPU use is PBS `cpupercent` (118, where 100 = one core) divided by 16 allocated cores.** The two measurement phases are serial by design: single-thread training (`tch::set_num_threads(1)`), one native maintenance job at a time, and a serial replay loop. Only the compile phase was parallel (§8).

---

## 2. Experimental setup and provenance

### 2.1 Design (preregistered before submission)
`hpc-bundle-20261117/H10M_PREREGISTRATION.md`, sha256 `582a75b5…`.

**Hypothesis.** A static LMI with the same model configuration as each DLI member reaches mean test R@30 = 0.90 with ≥ 30% less scored work than geometric DLI with `slot`.
- Model configuration: 548 buckets, 32,768-sample teacher, 512 hidden, 30 epochs, seed 42.
- Falsified if DLI `slot` or `proportional` gets within 30% of static's work, or if the curves cross.

**Collection.** SISAP23 LAION CLIP-768, 10,120,191 vectors, Float16 storage, cosine.

**Systems**
- **Geometric DLI** built by normal ingestion: continuation25 harness, buffer capacity 250,000, arity 2.
- **Static LMI** built in the job by the native `SegmentBuilder` over the same vectors (270.6 s).

**Final DLI topology** (`dli_hierarchy.json`, generation 41)

| Member | Live vectors | Buckets | Mean bucket | Max bucket | Bucket CV | Top-10% buckets' share of points |
|---|---|---|---|---|---|---|
| Level 5 | 6,000,000 | 548 | 10,949 | 51,723 | 0.72 | 24.5% |
| Level 4 | 3,000,000 | 548 | 5,474 | 25,578 | 0.76 | 25.8% |
| Level 2 | 750,000 | 548 | 1,369 | 5,830 | 0.69 | 23.6% |
| Level 1 | 250,000 | 548 | 456 | 2,203 | 0.73 | 24.7% |
| Plain (level 0) | 120,191 | — (one unit) | — | — | — | — |
| **Static LMI** | 10,120,191 | 548 | 18,468 | 85,124 | 0.75 | 25.9% |

There is no level 3 at this snapshot. The Plain buffer is at 48% of capacity.

**Execution.**
- Native replay through the production `route_pinned_traced` + `resolve_pinned_candidates`, with the original Float16 scorer (A0). Only selection varies.
- **Work** = live vectors actually scored per query (learned + Plain). This is the realized work, not the requested budget.

**Policies**
- `slot`: the default, `P·2^level`, Plain priority 1.
- `proportional`: an offline control. The learned budget is split by member population and Plain is always scored.
- `tdensity2`: population-density priority at temperature 2.
- `static_flat`: one static LMI over the whole collection.

**Budgets:** 10K, 20K, 40K, 80K, 160K, 320K and 640K.

**Query sets**
- **Validation:** rows 900–931 (32 queries), used only to lock operating points. Earlier use of these rows in DLI work is not excluded.
- **Test:** rows 9000–9255 (256 queries). No DLI parameter was ever fitted on these rows.
- **Oracle:** official SISAP23 top-30 IDs.

**Excluded by design:** HNSW, scalar128 (A2) and production latency. This document infers nothing about them.

### 2.2 Source identity, verified from job artifacts (not from local HEAD)

| Check | Evidence | Result |
|---|---|---|
| Submitted script | remote `dli-10m-frontier.pbs` sha256 `d9169517…` = local `dli-10m-frontier.filled.pbs` = handoff record | match |
| Source bundle | `job.log`: `source.bundle: OK` against `769fab91…` | verified |
| Commit | script guard `test "$(git rev-parse HEAD)" = 072be2e0c9575d9bd8129873dc8f9c6e471b5591` plus `status --porcelain` empty, under `set -euo pipefail`. A failure writes `FAILED.json`; the job reached `COMPLETE.json` | **072be2e0c**, clean tree |
| `COMPLETE.json` | `"commit":"072be2e0c…"`, `"binary_sha256":"09224072…"` | consistent |
| `Cargo.lock` | `cargo-lock.sha256` `73731b8e…` = preregistered value | match |
| Vendored crates | `vendor.tar: OK` (`c35ad341…`) | verified |
| Inputs | corpus (staged copy re-hashed on scratch), queries, gold, and val/test slices: all `OK` in `job.log` | verified |
| Toolchain | rustc 1.98.0, cargo 1.98.0, torch 2.5.1+cpu, protoc 3.18.0, g++ 14.2.0 | as smoke |

`summary.json` also records `source_head: 072be2e0c…`. That field is passed in from the environment, so it is not independent evidence; the guard above is.

**Binary hash.** The job binary (`09224072…`) differs from the smoke binary (`4b3d5bf5…`), despite the same commit, `Cargo.lock`, features, profile and toolchain. The two builds ran under different absolute scratch paths (`/scratch.ssd/…` versus `/scratch/local.ssd/…`), and rustc embeds source paths, so the hashes are not expected to match. This is a likely explanation, not a verified one.

So the smoke job validated the build path (bundle checkout, vendored offline build, tests), not this exact binary. The binary used is identified by its own `binary.sha256`.

---

## 3. Quality checks

**PBS.** Job 25677858 reached state F with `Exit_status = 0`, walltime 01:11:21, on `alfrid1`. Smoke job 25677616 also reached F with exit 0 (00:09:15 on `elmu1-24`) and `SMOKE_COMPLETE`, 6/6 tests passing. PBS stdout (`dli-10m-frontier.o25677858`) is byte-identical to `job.log`.

**Checksums.** All 19 files in `SHA256SUMS` verify. The exception is `job.log`: its recorded hash equals the file with its final line (`COMPLETE …`) removed, because `tee` appends that line after `SHA256SUMS` is written. This is benign and reproducible, and the smoke job shows the same pattern. `COMPLETE.json` is written after verification, so it is not in `SHA256SUMS`.

**Per-stage exit status.** Every timed stage reports `Exit status: 0` (build, ingest, frontier). The frontier test reports `1 passed; 0 failed`.

**Copy-back.** The job writes outputs directly to persistent storage, except `frontier/` and the ingest artifacts. Those are copied from scratch and then covered by the `SHA256SUMS` pass, which the job re-verifies (`sha256sum -c --quiet`) before `COMPLETE.json`. There is no separate receipt file; the re-verification is the receipt.

**Record completeness** (`analyze_10m.py`, `integrity`)
- 8,064 rows = (32 + 256) queries × 4 policies × 7 budgets, with 0 duplicate keys.
- Query IDs are contiguous, and every (set, query, policy, budget) cell is present exactly once.
- 288 reference-cost rows, one per query.
- Topology has 5 members plus the static reference.

**Invariants (0 violations)**
- `scored_total = original + plain`.
- `unique_offsets ≤ posting_entries`.
- Recall within its range.
- DLI miss attribution sums to `30 − recall30`.
- `unselected_level_rank` length equals `miss_bucket_unselected`.
- Plain work is always 0 or 120,191: the whole buffer or nothing.

Two apparent violations from a first naive check were resolved against the harness source at `072be2e0c`:
- `static_flat` rows carry no miss attribution, by design (`if !flat`).
- `unselected_level_rank` lists learned-bucket misses only, not Plain misses.

**Monotonicity.** Realized work never decreases with budget for any query and policy. Recall decreases with budget in 4 of 6,912 per-query steps (2 `slot`, 2 `tdensity2`), consistent with Float16 near-tie reordering.

**Cross-check.** The ingestion harness's own 100-query probe at its default budget (R@10 = 0.403, about 17.6K scored per query) agrees with the replay's `slot` at a 10K budget (R@10 = 0.405, 19.0K scored).

**Coverage of the plan.** Both query sets, all 7 budgets, all 4 policies and the static reference completed. Nothing planned is missing.

---

## 4. H-10M decision

### 4.1 Primary estimand

`W(0.90)` is the realized mean work at which the mean test R@30 curve reaches 0.90. It is computed by linear interpolation between the two measured budgets that bracket it, and is never extrapolated. Uncertainty comes from a paired percentile bootstrap over the 256 test queries (10,000 resamples, seed 20261118): every policy is evaluated on the same resampled queries.

| Policy | W(0.90) [95% CI] | Measured bracket (budget: work → R@30) | Static saving [95% CI] | P(saving ≥ 30%) |
|---|---|---|---|---|
| **static_flat** | **166,354** [143,130, 218,622] | 80K: 95,718 → 0.837; 160K: 175,093 → 0.908 | — | — |
| slot | 304,884 [284,532, 325,403] | 160K: 178,831 → 0.781; 320K: 327,299 → 0.921 | **45.4%** [32.3, 50.9] | 0.987 |
| proportional | 285,358 [262,699, 333,228] | 80K: 215,033 → 0.837; 160K: 295,529 → 0.909 | **41.7%** [30.4, 47.4] | 0.977 |
| tdensity2 (not in the rule) | 216,881 [166,358, 280,078] | 160K: 177,357 → 0.889; 320K: 341,428 → 0.935 | 23.3% [7.2, 34.0] | 0.114 |

### 4.2 Crossing rule

The preregistration says the hypothesis is also falsified "if the curves cross". I take that to mean: a DLI policy (`slot` or `proportional`) is better than static at some matched work inside the measured range.

Matched-work differences are computed on mean curves, only where both policies were measured, with paired bootstrap bands (§5.2, Figure 3):
- **static − slot:** positive and significant at all 19 grid points from 33K to 565K. It peaks at +0.155 around 100K and shrinks to +0.008 at 565K.
- **static − proportional:** positive and significant from 160K to 413K, and not significant (+0.006, +0.005) at 483K and 565K.

**No crossing.** The curves converge at high work instead.

### 4.3 Decision: **SUPPORTED**, with these qualifications

1. **Direction does not depend on interpolation; size does.** Static's 0.90 crossing lies between measured points at 95.7K and 175.1K. Slot's lies between 178.8K and 327.3K, and proportional's between 215.0K and 295.5K. The brackets do not overlap, so static reaches 0.90 with less work under any monotone interpolation. The ≥ 30% size, however, rests on linear interpolation across a 2× budget grid: the extreme bracket combinations give 2% to 71% for slot.

2. **The margin is thin.** The lower CI bounds (32.3% and 30.4%) only just clear 30%.

3. **It depends on the policy.** A DLI policy that avoids paying for the Plain buffer early (`tdensity2`) gets within 23%.

4. **It depends on the target.** At R@30 = 0.95 the slot saving is 28.6% [4.1, 41.4] and the proportional saving 19.0% [−1.7, 34.8].

5. **One model, one topology.** All intervals are over queries for one trained DLI topology, one static model, one ingestion order and one hardware run. They do not cover variation across training seeds or index constructions (§10).

### 4.4 Validation-locked operating points (secondary)

The operating point is the smallest budget whose validation mean R@30 is ≥ 0.90.

| Policy | Locked budget | Val R@30 / work | Test R@30 [CI] | Test work |
|---|---|---|---|---|
| static_flat | 80K | 0.929 / 96,906 | 0.837 [0.810, 0.863] | 95,718 |
| slot | 320K | 0.951 / 325,524 | 0.921 [0.902, 0.939] | 327,299 |
| proportional | 80K | 0.918 / 216,504 | 0.837 [0.811, 0.862] | 215,033 |
| tdensity2 | 80K | 0.901 / 95,759 | 0.815 [0.789, 0.841] | 89,443 |

The 32 validation queries are much easier than the test set at 10M: static scores 0.929 versus 0.837 at 80K. So three of the four locked points miss 0.90 on test, and the locked points do not form a matched-recall comparison.

They still show the operational consequence. With validation-chosen settings, static and proportional deliver the same test recall, and proportional spends 2.25× the work. A 32-query validation set is too small and too shifted to choose a 10M operating point.

---

## 5. Recall versus work

### 5.1 Measured operating points (test, n = 256, means)

*Realized / requested* is mean work divided by budget. *Sel.* is mean work as a share of 10,120,191. *Plain sel.* is the share of queries that scored the Plain buffer. *R@30 p10/p50/p90* are query-level quantiles.

| Policy | Budget | R@10 | R@30 [95% CI] | R@30 p10/p50/p90 | Queries R@30 ≥ 0.9 | Work | Realized / requested | Sel. | Postings = unique | Plain sel. |
|---|---|---|---|---|---|---|---|---|---|---|
| static | 10K | 0.618 | 0.605 [.564,.646] | .12/.67/1.0 | 31% | 32,801 | 3.28 | 0.32% | 32,801 | — |
| static | 20K | 0.651 | 0.641 | .20/.70/1.0 | 33% | 38,394 | 1.92 | 0.38% | 38,394 | — |
| static | 40K | 0.729 | 0.724 | .27/.83/1.0 | 43% | 55,674 | 1.39 | 0.55% | 55,674 | — |
| static | 80K | 0.848 | 0.837 | .48/.93/1.0 | 60% | 95,718 | 1.20 | 0.95% | 95,718 | — |
| static | 160K | 0.914 | 0.908 | .73/.97/1.0 | 77% | 175,093 | 1.09 | 1.73% | 175,093 | — |
| static | 320K | 0.948 | 0.947 | .85/1.0/1.0 | 88% | 333,738 | 1.04 | 3.30% | 333,738 | — |
| static | 640K | 0.966 | 0.968 | .93/1.0/1.0 | 93% | 654,659 | 1.02 | 6.47% | 654,659 | — |
| slot | 10K | 0.405 | 0.396 | .10/.40/.70 | 3% | 19,481 | 1.95 | 0.19% | 19,011 | 0% |
| slot | 20K | 0.528 | 0.512 | .15/.50/.87 | 8% | 32,783 | 1.64 | 0.32% | 26,210 | 5% |
| slot | 40K | 0.684 | 0.671 | .30/.73/.97 | 27% | 87,856 | 2.20 | 0.87% | 40,437 | 39% |
| slot | 80K | 0.766 | 0.750 | .40/.83/.98 | 39% | 152,823 | 1.91 | 1.51% | 50,004 | 86% |
| slot | 160K | 0.802 | 0.781 | .43/.87/1.0 | 48% | 178,831 | 1.12 | 1.77% | 59,110 | 100% |
| slot | 320K | 0.923 | 0.921 | .78/.97/1.0 | 79% | 327,299 | 1.02 | 3.23% | 207,108 | 100% |
| slot | 640K | 0.962 | 0.965 | .90/1.0/1.0 | 91% | 646,795 | 1.01 | 6.39% | 526,604 | 100% |
| proportional | 10K | 0.627 | 0.605 | .23/.63/.97 | 23% | 150,126 | 15.0 | 1.48% | 29,935 | 100% |
| proportional | 40K | 0.767 | 0.754 | .37/.83/1.0 | 44% | 176,624 | 4.42 | 1.75% | 56,433 | 100% |
| proportional | 160K | 0.922 | 0.909 | .72/.97/1.0 | 77% | 295,529 | 1.85 | 2.92% | 175,338 | 100% |
| proportional | 640K | 0.972 | 0.974 | .93/1.0/1.0 | 94% | 774,511 | 1.21 | 7.65% | 654,320 | 100% |
| tdensity2 | 10K | 0.454 | 0.437 | .10/.43/.80 | 5% | 17,718 | 1.77 | 0.18% | 17,718 | 0% |
| tdensity2 | 40K | 0.732 | 0.712 | .37/.77/1.0 | 38% | 47,096 | 1.18 | 0.47% | 46,626 | 0% |
| tdensity2 | 80K | 0.831 | 0.815 | .50/.90/1.0 | 55% | 89,443 | 1.12 | 0.88% | 85,218 | 4% |
| tdensity2 | 160K | 0.896 | 0.889 | .67/.97/1.0 | 73% | 177,357 | 1.11 | 1.75% | 157,168 | 17% |
| tdensity2 | 320K | 0.936 | 0.935 | .83/.98/1.0 | 83% | 341,428 | 1.07 | 3.37% | 272,412 | 57% |
| tdensity2 | 640K | 0.960 | 0.962 | .90/1.0/1.0 | 91% | 649,970 | 1.02 | 6.42% | 536,821 | 94% |

Every budget, with CIs and all columns, is in `ten_million/derived/analysis_10m.json` → `eval.curves`. Selected posting entries equal unique offsets in every row (no duplicate postings).

**Requested versus realized budget.** The budget is a *minimum* target, satisfied in whole buckets (`dli_bucket_plan.rs`, `select`).
- At 10K, every policy overshoots: one static bucket averages 18.5K vectors, one level-5 bucket 10.9K.
- `slot` overshoots by up to 2.2× in the 40K–80K range, because the next unit it takes is the 120K Plain buffer.
- `proportional` overshoots by up to 15× because it always scores Plain.
- **Requested budgets are therefore not comparable across policies.** All comparisons here use realized work.

### 5.2 Matched-work differences (Figure 3)

These are differences of interpolated mean curves, evaluated only where both policies were measured, with a paired bootstrap.

| Work | static − slot | static − proportional | static − tdensity2 | tdensity2 − slot |
|---|---|---|---|---|
| 38,924 | **+0.114** [.078,.149] | (not measured) | −0.008 [−.043,.026] | **+0.122** |
| 100,084 | **+0.155** [.134,.175] | (not measured) | +0.016 [−.002,.035] | **+0.138** |
| 160,488 | **+0.136** [.117,.155] | **+0.223** [.198,.247] | **+0.020** [.007,.033] | **+0.116** |
| 219,866 | **+0.099** [.083,.114] | **+0.077** [.061,.094] | **+0.018** [.007,.029] | **+0.081** |
| 301,214 | **+0.042** [.031,.053] | **+0.028** [.017,.039] | **+0.015** [.006,.024] | **+0.027** |
| 412,661 | **+0.019** [.011,.028] | **+0.014** [.005,.022] | **+0.010** [.002,.019] | **+0.009** |
| 565,341 | **+0.008** [.001,.016] | +0.005 [−.002,.012] | +0.007 [−.000,.015] | +0.001 |

Bold means the CI excludes 0. The full 25-point grid is in `derived/analysis_10m.json` → `eval.matched_work`.

**Equal requested budget, per query, no interpolation** (`eval.equal_budget`). Static − proportional in R@30 is +0.001, −0.012, −0.031*, −0.000, −0.001, −0.002 and −0.005 from 10K to 640K; only 40K (*) is significant. The work difference is −117K to −121K at every budget. **At the same learned allocation, the two systems find the same neighbours, and DLI additionally scores the Plain buffer.**

**Audit of `analyze_frontier.py`** (the script the job ran). Its per-query interpolation clamps any work below a query's smallest measured point to that point's recall.
- At 10M, proportional's smallest work is about 150K, so the job's `analysis.json` gives proportional R@30 = 0.605 at 20K, 40K and 80K. That is recall measured at 150K work: 256 of 256 queries were clamped.
- Static and slot are clamped for 193 and 99 queries respectively at 20K.

The script has three further problems:
- It bootstraps only against `slot`.
- It drops queries above the measured range, so the sample changes between work levels.
- It averages coverage costs over covered queries only; 82–92 of 256 are uncovered at top-30.

None of the conclusions here use those outputs. The job's `analysis.json` stays in the raw evidence unchanged, and `analyze_10m.py` reproduces the old method side by side (`eval.old_method_per_query_interp`) to measure the bias. The validation-locked points and the raw mean curves in `analysis-summary.txt` are correct.

### 5.3 Figures (`ten_million/figures/`, SVG)
- `fig1_recall30_vs_work_10m.svg`: mean test R@30 against mean realized work (log scale), 95% CI whiskers, the 0.90 target, n = 256. `fig1_recall10_vs_work_10m.svg` is the same for R@10.
- `fig2_work_decomposition_10m.svg`: learned versus Plain work per requested budget and policy, with requested-budget ticks.
- `fig3_matched_work_difference_10m.svg`: static − DLI at matched work with paired bootstrap bands.
- `fig4_scale_comparison.svg`: R@30 against % of the collection scored, 65K and 10M panels.
- `fig5_miss_location_10m.svg`: missed top-30 gold per query by level and cause, for slot, tdensity2 and proportional.

The figures are generated by `make_figures.py` from the derived JSON. They use the reference categorical palette slots 1–4 with distinct marker shapes and direct legends, so identity does not depend on colour alone. The palette validator could not run (no Node.js here) and the SVGs were not rasterized. Coordinates were checked to lie inside the axes, but **open them once before using them in the thesis**.

---

## 6. Where the work goes: per-level and Plain-buffer analysis

### 6.1 The Plain buffer is an all-or-nothing cost (mechanism verified in code; costs measured)

`dli_bucket_plan.rs::select` ranks learned buckets by `P(bucket | q) · 2^level` and gives the Plain buffer a fixed score of 1.0. It adds whole units until the selected occupancy reaches the budget. Plain is one unit of 120,191 vectors.

A level-L bucket outranks Plain only if P > 2^−L: 3.1% at level 5, 50% at level 1. So after a query's few confident buckets, Plain comes next and adds 120K vectors in one step.

The effect on slot:

| Budget | Learned buckets per query by level (L1 / L2 / L4 / L5) | Learned work | Plain selected | Missed gold in Plain (per query) |
|---|---|---|---|---|
| 10K | 0.0 / 0.0 / 0.2 / 1.0 | 19,011 | 0% | 0.40 |
| 40K | 0.5 / 0.7 / 1.4 / 1.7 | 40,437 | 39% | 0.23 |
| 80K | 0.8 / 1.1 / 1.7 / 2.1 | 50,004 | 86% | 0.03 |
| 160K | 1.1 / 1.5 / 2.1 / 2.6 | 59,110 | 100% | 0 |
| 320K | 5.0 / 6.2 / 9.1 / 9.9 | 207,108 | 100% | 0 |

Between 40K and 160K requested, slot's realized work grows from 88K to 179K, but its learned work grows only from 40K to 59K. The rest is the Plain buffer, which holds 0.40 gold neighbours per query: about 1.2% of the corpus and 1.3% of top-30 gold, in line with its population share. That produces the plateau in slot's frontier (R@30 0.750 → 0.781) and the largest matched-work gap (+0.155 near 100K).

`proportional` scores Plain on every query by construction (`run_proportional`, `share = usize::MAX / 4` for level 0). `tdensity2` defers Plain: it is selected in 0% of queries at ≤ 40K and in 57% at 320K. It therefore loses 0.34–0.40 gold per query (1.1–1.3 pp R@30) through ≤ 160K. That is about the size of its remaining deficit against static (+0.015 to +0.020 from 117K to 301K). The match is a correlation, not a demonstrated cause.

### 6.2 Learned levels versus the static model at equal learned work

This table puts each DLI policy's recall next to the static curve interpolated at the DLI policy's *learned* work. It excludes the Plain cost, but the DLI recall still includes whatever gold the Plain buffer provided.

| Policy @ budget | Learned work | DLI R@30 | Static R@30 at the same work |
|---|---|---|---|
| slot @ 80K | 50,004 | 0.750 | 0.697 |
| slot @ 320K | 207,108 | 0.921 | 0.916 |
| tdensity2 @ 160K | 157,168 | 0.889 | 0.892 |
| proportional @ 80K | 94,842 | 0.837 | 0.834 |
| proportional @ 160K | 175,338 | 0.909 | 0.908 |
| proportional @ 640K | 654,320 | 0.974 | 0.968 |

When Plain is scored (proportional, and slot at ≥ 160K), DLI gets Plain's gold (about 0.013 R@30) without it counting as learned work. After subtracting that, the learned levels are within about 1 pp of static at equal learned work. Slot at low budgets looks better than static only because its 120K Plain cost is excluded here.

**Accounting check.** If learned selection efficiency were equal, the static saving against `proportional` at the 0.90 target would be Plain / (W_static + Plain) = 120,191 / (166,354 + 120,191) = **41.9%**. The measured value is **41.7%**.

**Interpretation.** At 10M, four independently trained per-level routers with population-proportional allocation select candidates about as efficiently as one router trained on the whole collection. The measured static advantage is almost entirely the price of exhaustively scoring the unpartitioned Plain buffer, plus slot's early selection of it. This decomposition is exact work accounting on measured quantities. Attributing recall to it is consistent with the data but is not an intervention experiment: no run indexed the Plain buffer.

### 6.3 Router ranking, partition granularity and miss depth

Reference costs are mean scored vectors per query needed to cover the gold neighbours. The oracle scores only gold-containing buckets; *router-ranked* scores every bucket ranked at or above the deepest gold bucket in each member, with perfect allocation across members.

| | DLI oracle | DLI router-ranked | Ranked / oracle (ratio of means; median query) | Static oracle | Static router-ranked | Static ranked / oracle |
|---|---|---|---|---|---|---|
| top-10 | 68,655 | 197,197 | 2.87; 1.16 | 84,174 | 268,078 | 3.18; 1.17 |
| top-30 | 134,103 | 425,941 | 3.18; 1.33 | 135,413 | 494,791 | 3.65; 1.38 |

These costs are means over all 256 queries, with no censoring.
- **Router ranking is far from the oracle in both systems, and slightly worse for static.** It is not a DLI-specific penalty.
- The ratio-of-means is driven by a heavy tail. The median query needs only 1.2–1.4× the oracle, but the 90th percentile needs over 1M vectors for top-30.
- DLI's finer effective partition (2,192 learned buckets, smaller at lower levels) gives a lower top-10 oracle cost than static's 548 larger buckets: 68.7K versus 84.2K.

**Unrecoverable misses.** Gold in buckets ranked > 50 within their member averages 0.37 per query at every budget and policy. The sweep never reaches those buckets: at 640K, slot selects 15–27 buckets per level.

**Selected but lost.** These are gold neighbours that were scored but not returned. They rise with budget from 0.08 to 0.24 per query (0.8 pp at the top). These are Float16 storage against the Float64-semantics oracle near-ties.
- At 65K the same quantity was 0.02–0.06.
- The harness computes no static miss attribution, so whether static loses the same amount is not measured.

**Bucket imbalance.** It is similar at every level and in static: coefficient of variation 0.69–0.76, and the top 10% of buckets hold 24–26% of points. Nothing here points to a level-specific imbalance problem. Absolute bucket sizes do matter for the low-budget overshoot (§5.1).

**Coverage costs** (first measured budget reaching full top-k coverage; uncovered queries are right-censored):

| Policy | Top-10 covered / 256 | Top-10 median | Top-30 covered / 256 | Top-30 median |
|---|---|---|---|---|
| static | 206 | 168K | 174 | 328K |
| slot | 205 | 324K | 164 | 642K |
| proportional | 216 | 224K | 174 | 452K |
| tdensity2 | 204 | 168K | 165 | 429K |

---

## 7. Comparison with 65K, and correction of the 65K interpretation

`analyze_10m.py` was rerun unchanged on `continuation-42/frontier`: the 65K testB set of 256 rows 9000–9255, with the same oracle semantics. Output is in `derived/analysis_65k.json`.

| | 65,536 vectors | 10,120,191 vectors |
|---|---|---|
| Topology | L2 48K, L1 16K, Plain 1,536 (2.3%) | L5 6M, L4 3M, L2 750K, L1 250K, Plain 120,191 (1.2%) |
| Static W(0.90) | 2,871 (4.4% of N) | 166,354 (1.6% of N) |
| Static saving vs slot at 0.90 | 39.5% [33.0, 46.0] | 45.4% [32.3, 50.9] |
| vs proportional at 0.90 | 32.7% [24.7, 41.1] | 41.7% [30.4, 47.4] |
| vs tdensity2 at 0.90 | 39.4% [32.3, 46.0] | 23.3% [7.2, 34.0] |
| Static saving vs slot at 0.95 | 28.5% [20.0, 35.7] | 28.6% [4.1, 41.4] |
| Plain / (W_static + Plain) at 0.90 (predicted saving vs proportional) | 34.9% | 41.9% |
| Static − proportional at equal budget | −0.008 to −0.001 R@30; work −1.53K | −0.031 to +0.001 R@30; work −120K |
| Rank / oracle cost, top-10, DLI / static (ratio of means) | 3.55 / 5.13 | 2.87 / 3.18 |

**Reproduced.** Static needs less work than slot and proportional at the 0.90 target, at both scales, by a similar relative amount.

**Changed at 10M**
- **The slot frontier has a mid-range plateau** from Plain selection (§6.1). At 65K, Plain (1,536) was smaller than one request at almost every budget; at 10M it is larger than most requests.
- **The T=2 density gap narrows** from 39% to 23%, because density priority postpones the 120K Plain cost.
- **The high-recall gap converges.** Differences against slot and proportional shrink to under 1 pp at ≥ 480K work.

**Correction to the 65K report** (`CANDIDATE_SELECTION_REPORT.md`, conclusions 2–3)
- The 65K report gave router ranking (3.5× over the oracle) as the main inefficiency, and a 1.8× cross-level allocation loss.
- With the static reference costs it also recorded, static's router has the same or a larger ranked/oracle overhead (5.1× at 65K).
- The 1.8× figure divides a covered-queries-only mean by an all-queries mean, so it is biased.
- At 65K as at 10M, proportional matches static at equal budget, and the work gap equals the Plain buffer.

Router ranking is the largest source of *absolute* work above the oracle for every LMI-style index here. It does not explain the *static–DLI* gap. An addendum to the 65K report records this; the original text is kept.

**Scale effect on selectivity.** Both systems scan a smaller *fraction* of the collection at 10M for the same recall. Static goes from 4.4% to 1.6% of N at R@30 = 0.90 (Figure 4). These are two single topologies, so this is not a fitted scaling law.

---

## 8. Runtime and resources

### 8.1 What "7% CPU" and "60% RAM" mean

From PBS accounting (`qstat -x -f`, `qstat-final-25677858.txt`):

- **CPU.** `resources_used.cpupercent = 118`, `cput = 01:26:34` (5,194 CPU-seconds), `walltime = 01:11:21` (4,281 s). 5,194 / 4,281 = 1.21 cores used on average, and 1.21 / 16 allocated ≈ 7.4%. In PBS, `cpupercent` is CPU use in percent of *one* core, so the UI's 7% is that number divided by the allocation. **It measures how many allocated cores were busy, not whether the job was CPU-bound.**
- **Memory.** `resources_used.mem = 80,282,864 kB` (76.6 GiB) against 128 GB requested, which is the 60%.
  - Peak process RSS was 41.1 GB during ingestion and 47.3 GB during the frontier phase (`/usr/bin/time -v`).
  - The PBS figure is higher. That is consistent with the job's memory cgroup counting file-backed and page-cache pages, which the previous campaign's memsw behaviour also indicated. It is not verified from PBS documentation here.
  - `diag_messages = '{}'`, there were no swaps and the job was not killed: **no memory-pressure event was recorded.**

### 8.2 Time and CPU by phase

| Phase | Wall | CPU (user + sys) | Mean cores | Notes |
|---|---|---|---|---|
| Guards, staging, corpus copy | ≈ 2 min | — | — | Corpus NFS → scratch SSD: 15.5 GB in 58.2 s (≈ 270 MB/s) |
| Offline build (`--profile perf`, `CARGO_BUILD_JOBS=16`) | 5:59 | 1,760 s | 4.9 | Parallel compile |
| Geometric DLI ingestion | 44:52 | 2,577 s | 0.96 | 40 native publications; peak RSS 41.1 GB |
| Static build + frontier replay | 17:59 (static 270.6 s) | 812 s | 0.75 | Peak RSS 47.3 GB |
| Analysis + checksums | < 1 min | — | — | |
| **Total** | **71:21** | **5,194 s** | **1.21** | The phase sum (5,149 s) matches `cput` |

### 8.3 Why the measurement phases were serial

The serial execution was deliberate, not a stall.

**Training is single-threaded by design.** `lib/segment/src/index/lmi_index/training.rs:416-417` sets `tch::set_num_interop_threads(1)` and `tch::set_num_threads(1)` so training is deterministic.
- MLP epochs took 1,451 s in total.
- Teacher plus MLP took 1,664 s across the 40 builds, an average of 37 s each for 3,840 steps.

**Native maintenance runs one job at a time.**
- The 40 jobs' wall times sum to 2,408 s, which is 89% of ingestion wall time.
- `cpu_permit_wait` was about 0 µs for every job, so they never waited for a CPU permit.

**The ingestion harness waits for each publication.**
- Each 250K-point batch takes about 3.7 s to write, then 30–90 s waiting on maintenance (`SISAP23_DLI_WAIT`).
- The largest merge (5 sources → 6M) took 180 s.
- Write-blocking protected windows were at most 56.6 ms.

**The frontier replay is a serial loop over queries × configurations,** so that work and miss attribution are per query and per configuration.

### 8.4 I/O

All indexes and vectors were read from `$SCRATCHDIR` (`/scratch.ssd`, md RAID). NFS was read only during staging and input hashing.

| Phase | Major page faults | Read | Written |
|---|---|---|---|
| Ingestion | 17.6M | ≈ 40 GB (78.2M × 512 B) | ≈ 91 GB (177.1M × 512 B), segment rewrites at each merge |
| Frontier | 7.6M | ≈ 31 GB | ≈ 16 GB (static segment) |

The frontier's 25% non-CPU wall time is plausibly page-fault I/O wait on memory-mapped vectors. I/O wait, PSI and lock-contention counters were not collected, so **the job is not shown to be I/O-bound.**

The node was a shared 2 × 64-core AMD EPYC 9554 with 1.5 TB RAM, with CPU frequency scaling reported at 56% at start.

### 8.5 Were the resources appropriate?

**Memory.** 128 GB was appropriate; 64 GB would have risked the cgroup limit given the 80 GB peak.

**Walltime.** 24 h was 20× the 71 min actually used. A 3–4 h request would have been accepted just as safely and queues sooner.

**CPUs.** 16 CPUs only paid off for the 6-minute compile. More useful uses of the same allocation:
- Run 3–4 independent replicates (seeds or ingestion orders) concurrently in one 16-core job. That directly addresses the replication gap (§10), within the 128 GB.
- Or parallelize the replay over queries with per-thread attribution.
- Or request 4–8 CPUs for a single serial chain.

**Do not reuse the instrumented `elapsed_ns` values as latency.** They include page faults: the median ns per scored vector is 140–680 and varies with policy.

---

## 9. Scientific interpretation

1. **Search efficiency, static versus geometric DLI.** On the same data, model configuration and scorer, a static LMI reaches R@30 = 0.90 with 42–45% less candidate work than geometric DLI under default or proportional allocation. At very high recall (≥ 0.96) the systems converge.

2. **A measurable selection penalty exists, and its mechanism is the Plain buffer.**
   - Geometric maintenance leaves up to `buffer_capacity` recent vectors unpartitioned. Here that is 120,191, 1.2% of N.
   - Search must score that buffer whole to find the ≈ 1.3% of gold neighbours inside it.
   - At 10M that fixed cost is 72% of static's work at the target.
   - The independently trained per-level models are not the main source of inefficiency. Their combined selection matches the static model at equal learned work, to within about 1 pp.

3. **Independent models, cross-level selection and buffer size.**
   - Independent models: no measurable penalty at equal learned work with population-proportional allocation.
   - Cross-level selection: `slot`'s uncalibrated priority `P·2^L` with Plain at 1.0 decides *when* the Plain cost is paid. That timing produces slot's mid-range deficit.
   - Buffer size: the penalty scales with the buffer's fill level relative to the work needed for the target. This run sampled one fill level (48%), so the gap will vary over an ingestion cycle. Its expected range is untested.

4. **Implications for routing calibration and model adaptation.** The data points to *buffer-aware, cost-aware selection* rather than router retraining. Options:
   - Partition or cheaply index the Plain buffer, for example with provisional centroids or nearest-bucket assignment through the existing level models.
   - Give Plain a priority per unit of work instead of a fixed 1.0.
   - Use a smaller buffer, traded against maintenance frequency.

   Temperature calibration (T=2) helps mainly because it defers Plain, not because it improves cross-level ranking. Improving the routers themselves (rank/oracle 3×) would help static and DLI alike.

5. **Unanswered.**
   - Latency.
   - HNSW at matched recall on v1.20 and the same hardware.
   - Scalar128 at 10M.
   - Concurrent updates at scale: Tier B contention was measured only on a 16-dimensional fixture.
   - Behaviour across buffer fill levels and seeds.

   None is inferred here. The accepted Nov 3 static-LMI-versus-HNSW measurements are v1.19 (overlay, deimos11, NFS) and are not comparable with this v1.20 native replay.

---

## 10. Limitations

- **One replicate.** One DLI topology (one ingestion order, one model seed per member), one static model, one hardware run. The CIs cover only query sampling. Seed and construction variance is unmeasured and could be comparable to the margin over 30%.
- **One buffer state.** Plain was at 120,191 of 250,000. The size of the static–DLI gap depends on this (§6.2, §9.3).
- **Coarse budget grid** (2× steps). The H-10M magnitude depends on interpolation; the direction does not (§4.3).
- **Validation set.** 32 rows (900–931) whose earlier DLI use is not excluded, and much easier than test at 10M. Validation-locked points are informative but not matched.
- **Oracle semantics.** The official top-30 is computed on Float32/Float64 originals against Float16 storage. Recall has an attainable ceiling below 1 (selected-but-lost 0.24 per query at 640K).
- **Replay, not Collection search.** No request path, no concurrency, no cold-versus-warm control. `elapsed_ns` is not latency.
- **Partial miss attribution.** None for static, and Plain misses are not ranked.
- **Not tested:** HNSW, A2 or quantized scoring, density, reserved, other temperatures, buffer capacities, and Collection-level end-to-end runs.

---

## 11. Next research priorities

These are proposals only; nothing was launched.

1. **Replication.** 3–4 DLI ingestions with different seeds and orders, plus static seeds, in one 16-core job. Report between-replicate spread next to the query CIs.
2. **Buffer-state sweep.** Replay frontiers at several Plain fill levels: snapshots during ingestion, or a controlled 65K topology rebuilt at several fill levels. Test the accounting prediction `saving ≈ Plain / (W_static + Plain)`.
3. **Plain-handling intervention (offline replay first, 65K then 10M).** Cost-aware Plain priority, provisional partitioning of the buffer through the existing level models, and a smaller buffer capacity. This is the experiment that would turn the §6.2 accounting into a causal claim.
4. **Finer budget grid near the target** (√2 steps from 100K to 400K) to remove the interpolation dependence of the H-10M magnitude.
5. **Larger, disjoint validation set** (≥ 128 rows) for operating-point selection at 10M.
6. **Latency and matched-recall HNSW on v1.20,** same hardware, warmed Collection search. Measured only after the selection question above is settled.
7. **Harness and tooling.** Static miss attribution, Plain miss ranking, and retiring the clamping interpolation in `analyze_frontier.py` in favour of `analyze_10m.py`.

---

## 12. Locations

**Immutable raw results** (local copy, read-only, manifest `hpc-results-25677858.SHA256SUMS`, 37 files):
`/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/hpc-results-25677858/`
- `frontier-10m-25677858.pbs-m1.metacentrum.cz/`: the full job output directory, copied byte-for-byte.
- `smoke-25677616.pbs-m1.metacentrum.cz/`.
- `pbs-stdout/`.
- `qstat-final-*.txt`, `remote-listing.txt`.

**Remote original:** `/storage/budejovice1/home/nico/lmi-metacentrum/dli-v120-20261117/results/frontier-10m-25677858.pbs-m1.metacentrum.cz/`. Left untouched.

**Derived analysis** (worktree, `research/dli/selection_frontier/ten_million/`)
- `analyze_10m.py`
- `make_figures.py`
- `derived/analysis_10m.json`, `derived/analysis_65k.json`
- `figures/*.svg`

**Thesis draft:** `research/dli/selection_frontier/ten_million/THESIS_RESULTS_10M.md`.

**Reproduce**
```
python3 -I research/dli/selection_frontier/ten_million/analyze_10m.py <raw>/frontier --out <dir>/analysis_10m.json
python3 -I research/dli/selection_frontier/ten_million/make_figures.py <dir>
```
Requires numpy; seed 20261118, 10,000 resamples, about 8 s.
