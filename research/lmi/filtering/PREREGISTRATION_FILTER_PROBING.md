# Preregistration: probing policies for filtered LMI/IVF search

**Status:** frozen at commit, 2026-11-19, before any implementation or run. Deviations require a
dated amendment in this file.

**Branch:** `research/lmi-adaptive-filtering` only. Not merged into the main research branch.
**Baseline:** commit `1b323f496` (planner) and `a89ab1fd3` (SISAP23 300K baseline). The Tier 3B
campaign (`research/static-lmi-ivf-router-fix`, job 25743351) is frozen and is not touched.
**Scope:** narrow. One concise report follows the evaluation. Further optimization stops there unless
the results establish a new, important finding. The next priorities are Tier 3B synthesis, thesis
writing and DLI replication.

## Motivation

The 300K baseline (`SISAP300K_FILTERING_BASELINE.md`) shows three things at a fixed `nprobe` = 16:

- filtered Recall@10 falls as filters become more selective (IVF 0.953 → 0.874, MLP 0.934 → 0.828);
- the break-even ignores the cost of checking the filter;
- exact filtered search has a cost cliff of its own.

This experiment measures the `nprobe` × selectivity surface, then compares candidate policies on
held-out queries. Scaling `nprobe` as 1/selectivity is **not** assumed or evaluated as a policy.

## Questions

- **F1 (descriptive):** for each selectivity *s* and probe count *p*, what are filtered Recall@10, latency
  and candidate work, for IVF and for the MLP router, against exact filtered search under a matched protocol?
- **F2:** does a recall-calibrated policy reach a target mean recall τ on test at lower latency than
  (a) exact filtered search and (b) the best fixed-`nprobe` policy that reaches τ?
- **F3:** does adding the filter-check cost to the cardinality break-even change plan choices, and does it
  lower latency at equal recall?
- **F4:** under each policy, how do IVF and the MLP router compare (paired)?
- **F5 (diagnostic):** what explains the exact-search cost cliff (payload-index iteration versus full scan)?

## Data, splits and gold

- **Corpus:** the canonical SISAP23 300K subset (hashes as in the baseline), Float16, cosine.
- **Indexes:** IVF and MLP, built natively and identically to the baseline: 548 buckets, S = 32,768,
  5 k-means iterations, seed 42; MLP hidden size 512, 30 epochs. One build per router, reused for every
  phase. The stored `nprobe` (16) is the B0 probe count; every other probe count is set per call.
- **Queries:**
  - validation (calibration and selection only): official rows 1000–2999 (2,000);
  - test (evaluated once, with frozen choices): rows 4000–5999 (2,000).
  - These are the same splits as Tier 1. Baseline rows 0..999 are used only for the B0 reproduction check.
- **Gold:** native exact filtered search over the same source segment, per (split, filter). The official
  top-1000 gold is used only as an unfiltered sanity check.

## Recall definition

For a query and filter with *m* live matching points, let k_eff = min(10, *m*). Recall@10 is
|returned ∩ gold| / k_eff.

- When *m* = 0, recall is undefined. Such cases are excluded from every recall statistic and reported
  separately: the count, and whether the result was correctly empty.
- Gold ties at the 10th position are resolved by the native exact search. Any tie-induced mismatch counts
  against the system and is not corrected.

## Filters

- **Independent family (primary; the only family used for calibration and selection):**
  `r = splitmix64(id) mod 10^6`, filter `r < s·10^6`, with
  *s* ∈ {100, 50, 20, 10, 5, 3, 2, 1, 0.5, 0.3, 0.1, 0.01}%.
- **Correlated slab family (exploratory):** `g` = the quantile rank (×10^6) of the stored vector's
  projection on a fixed random unit direction (seed 7), with filter `g ≥ (1 − s)·10^6` at the same *s* grid.
  - Membership correlates with geometry and therefore with routing.
  - The policies frozen on the independent family are applied **without retuning**, and the results are
    reported separately.

## Grid

- Probe counts *p* ∈ {1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 128, 256}. Probing all 548 buckets is
  equivalent to the exact plan, which is measured directly.
- Every (router, family, *s*, *p*) cell runs the **forced routed plan**: route at *p*, intersect with the
  filter, score, and widen only to fill `min(10, live)`, exactly as in the baseline.
- Every (router, family, *s*) cell runs the **forced exact plan** through the index's own Plain path.

## Instrumentation (per query, recorded as JSONL outside Git)

Counters are kept separately **for the final probe** and **cumulatively over all probes of the query**,
because each widening re-routes and re-evaluates from scratch:

- raw posting offsets visited (the sum of the selected buckets' posting lengths);
- unique candidate offsets (after de-duplication);
- filter evaluations;
- surviving candidates (passed the filter);
- vectors actually scored (visible and not deleted, as passed to the scorer);
- widenings, initial and final *p*.

Also recorded: split, router, family, *s*, *p*, plan, payload estimate (min / exp / max), whether sampling
decided, true matching count *m*, Recall@10, latency. For exact plans: the payload-iteration path
(primary-clause iteration or full scan) and the points scored.

## Measurement protocol

0. **Pilot** (before the full sweep): 100 validation queries, 3 selectivities, all *p*, both routers and
   both families. It validates runtime, memory and record completeness. Pilot records are kept but not analysed.
1. **Counter and recall pass.** All queries of both splits. Deterministic, run in parallel over queries.
   No latency is taken from it.
2. **Timing pass.** Single thread, `perf` profile.
   - Seeded subsamples of 500 validation and 500 test queries.
   - All *p* plus exact, for IVF, MLP and the exact baseline, under the same protocol.
   - One discarded warm-up pass per (router, family, *s*), then 3 repetitions; within each repetition the
     cell order is shuffled with a seeded generator.
   - Per-query latency is the median of the repetitions. p50 and p95 are taken over queries.
3. **Stage-cost pass.** Only with the `lmi-research` feature, on the validation timing subsample. It times
   routing, filter evaluation and scoring separately, and is used only for cost-model coefficients. Its
   latencies are never reported as headline numbers. The filter-check cost is also estimated by regressing
   timing-pass latency on the counters; both estimates are reported.

Runtime is an estimate, not a constraint. Correctness checks are never reduced to save time.

## Policies

Runtime policies may use **only** the payload-index cardinality estimate (and the existing sampling check)
and the frozen calibration tables. They never use a query's true matching count, its gold, or anything
derived from test queries. All parameters are selected on validation only and then frozen before test
evaluation. Each router is calibrated on its own validation records with the same procedure, targets and bins.

| ID | Policy | Plan decision | Probe count |
|---|---|---|---|
| **B0** | Committed planner, unchanged | Estimated cardinality ≤ probe volume at *p* = 16 → exact; otherwise routed | 16 |
| **B1** | Fixed-probe baseline | Same rule as B0 at its own *p* | Smallest grid *p* whose validation recall on the 100% filter ≥ τ |
| **C1** | Cardinality cost model | Exact if predicted T_exact(ĉ) ≤ predicted T_route(*p*); otherwise routed | As B1 |
| **R1** | Recall-calibrated, mean | Look up the bin of ŝ = ĉ_exp / N; exact if no grid *p* reaches τ in that bin, or if T_exact(ĉ) ≤ T_route(*p*) | Smallest *p* with validation mean recall ≥ τ in that bin |
| **R2** | Recall-calibrated, conservative | As R1 | Smallest *p* whose bootstrap 95% **lower** bound of validation mean recall ≥ τ |
| **O** | Oracle (not deployable, a bound only; uses gold) | Per query, the cheaper of exact or routed | Per query, the smallest *p* reaching τ |

- **Targets:** τ ∈ {0.90, 0.95}.
- **Cost models**, fitted on the validation timing pass:
  - T_exact(ĉ) = a_e + b_e·ĉ, fitted per exact-iteration path;
  - T_route(*p*, ĉ) = a_r + b_check·U_p + b_score·(ĉ/N)·U_p, where U_p is the mean unique candidates at
    probe *p* on validation.
- **Bins:** the 12 grid selectivities. ŝ is mapped to the nearest grid value **at or below** it, which gives
  the larger *p* and is the conservative side.
- **Not a guarantee.** R1 and R2 target **mean** recall over the validation query distribution, for filters
  independent of the vectors. They give no per-query guarantee. Tails are reported for every policy: the
  share of test queries below τ, and the 5th-percentile recall.

## Evaluation

1. **Simulation.** Policies are calibrated and their candidate work is analysed by simulation from the grid
   records. A forced (plan, *p*) is deterministic, so a policy's result for a query equals that grid record.
   Simulated latency is not treated as native runtime.
2. **Native measurement.** The frozen B0, B1, C1, R1 and R2 run natively through the research API on the test
   timing subsample, under the timing protocol, for both families.
   - Native results must equal the simulated ones; any mismatch is reported and blocks adoption.
   - Every reported policy latency comes from this run.
3. **B0 reproduction.** Run natively after the code changes, B0 must reproduce `result-nprobe16.json` exactly on
   rows 0..999: plans, recall, fill and widenings. Latency is allowed to differ.
4. **Reported** per router × policy × τ × family × *s*:
   - mean Recall@10 with a 95% query-bootstrap CI;
   - share of queries below τ, and 5th-percentile recall;
   - p50 / p95 latency;
   - exact-fallback frequency;
   - candidate counts (unique, filter evaluations, scored; final probe and cumulative);
   - the empty-result cases.
5. **Paired comparisons** use a 2,000-resample query bootstrap: R1 and R2 against B0 and B1, C1 against B0,
   and IVF against MLP within each policy.
6. **Adoption criterion, fixed in advance.** A policy can replace B0 as the default only if, on the
   independent-filter test split, for every *s*:
   - its mean recall is ≥ τ − 0.01 at τ = 0.95;
   - its p50 does not exceed exact filtered search;
   - across *s*, its p50 is no worse than B1's at matched recall.

   Otherwise B0 stays the default and the policies remain research options. Adoption itself is a separate
   decision and is not part of this branch's scope.

## Code changes (each in its own commit)

1. **Instrumentation:** the per-probe and cumulative counters above, the explicit initial probe, and the
   exact-path record. No change to Baseline outputs.
2. **Policy logic:** `FilterProbePolicy` {Baseline, CostModel, Calibrated}. It is reachable only through a
   research entry point under the `testing` / `lmi-research` features. The production `search()` stays on Baseline.
3. **Tests:**
   - Baseline equivalence: digests of outputs and plan decisions recorded at `a89ab1fd3`, before any change;
   - calibration-table behaviour and exact fallback;
   - counter consistency.
4. **Results:** benchmark harness, analysis script and the final report. Large artifacts stay outside Git.

## Threats to validity

- One build and one seed per router; one machine; one corpus and dimension.
- Timing is index-level and single-threaded.
- The payload estimate's error interacts with binning; it is recorded per query.
- The correlated family is a single synthetic construction.
- Calibration tables are tied to this partition and are not persisted with segments in this study.

## Amendment 1 (2026-11-19, after the pilot, before any full-run data)

**Pilot (protocol step 0).** 100 validation queries; *s* ∈ {100, 3, 0.1}%; all *p*; both routers and both
families. Pilot outputs are kept under `probing/pilot/` and are not analysed.

**What the pilot showed**
- Every grid record was complete, and every exact cell had recall 1.0.
- The native/simulation check found 0 mismatches over 36,000 policy calls.
- B0 reproduced `result-nprobe16.json` exactly on rows 0..999: 18/18 cells had identical plans, identical
  widenings and recall equal to within 1e-12. The research Baseline path equalled production `search()` on
  all 18,000 queries, and the native builds are deterministic.
- Peak RSS was 3.4 GB. The extrapolated full runtime is about 5 h.

**Changes, decided on pilot data only**

1. **Cost-model fitting.** The fits use relative-error least squares (weights 1/y²). A negative coefficient is
   fixed at 0 and the remaining terms are refitted. Reason: latencies span four orders of magnitude, and an
   unweighted fit is dominated by the largest cells.
2. **Model form.**
   - The pilot shows that exact-plan latency is not linear in matches: about 67, 487 and 764 ns per match at
     0.1%, 3% and 100%, all through payload-index iteration.
   - The linear routed model ignores widening work. Predicted/measured ratios ranged from 0.08 to 1.86.
   - **C1 keeps the preregistered linear form** and is evaluated as registered, with its prediction errors reported.
   - **R1/R2 use an empirical cost table:**
     - exact plans: mean validation latency against each filter's estimated matches;
     - routed plans: mean validation latency per *p* against estimated selectivity;
     - the tables are interpolated log-log, clamped at the ends, and evaluated at the runtime estimate only.
   - The tables are added as optional fields of `CostModel`.
3. **Bin keys.** Calibration bins are keyed by each validation filter's **estimated** selectivity
   (ĉ_exp / available), the quantity the runtime policy sees, rather than the nominal grid value.
   - In the pilot, nominal keys sent a 3% filter (ŝ = 2.9993%) into the 0.1% bin.
   - Rounding down remains the rule for any other estimate.
4. **Exact reference in the native pass.** The native policy pass includes an `EXACT` reference: a calibrated
   table with one empty bin, so it always runs the exact plan. The exact filtered baseline is therefore measured
   in the same pass and under the same protocol as the policies.
5. **Warm-up.** For each (router, family, *s*, split), every cell or policy runs over the first 50 subsample
   queries, and those runs are discarded.
6. **B1 fallback.** If no grid *p* reaches τ on the 100% filter, B1 uses *p* = 256. This did not occur in the pilot.
7. **Oracle O.** Per test timing-subsample query, O takes the cheapest measured cell (median latency) whose recall
   is ≥ τ. The exact cell always qualifies.
8. **Adoption, "no worse than B1 at matched recall".** This is checked at every *s* where both the policy and B1
   have mean recall ≥ τ − 0.01.
9. **Phase order.** prepare → grid → timing → stages → calibrate → policies → report → b0repro. Calibration needs
   the stage pass (in the pilot it was first run before it, and the stage estimate was missing).
10. **Pilot policy phase.** The pilot exercised the policy phase on validation rows, labelled as such.

**Observations to examine, not results**
- Every exact plan iterates a payload index. The per-match cost growth is consistent with the matching set
  staying cache-resident under repeated queries with the same filter, so the reported latencies are warm-cache.
- The stage pass puts filter evaluation at 13–18 ns per candidate. That is far below the ≈2.2 ms extra latency
  of the 100% filter in the baseline (8,760 candidates × ~15 ns ≈ 0.13 ms), so that extra latency has another source.
