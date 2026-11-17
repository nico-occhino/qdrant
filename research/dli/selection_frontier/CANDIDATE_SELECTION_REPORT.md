# DLI candidate-selection frontier (continuation42)

**Question.** Can learned routing reach competitive recall while examining fewer candidates? Where does candidate work go?

**Evidence:** `/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-42/`
- `frontier/rows.jsonl`, `reference_costs.jsonl`, `topology.json`, `analysis.json`.
- `frontier-run.log`, `source-head.txt`.
- `source.patch` (taken at launch) and `source-at-completion.patch`.
- Harness committed in `1e1a60d50`.

## Method

**Topology.** A byte-identical copy (tree sha256 `d9859d32…`) of the frozen continuation40 65,536-vector topology:
- L2: 48,000 vectors.
- L1: 16,000 vectors.
- Plain: 1,536 vectors.
- Each learned member: 548 buckets, 32,768-sample teacher, 512 hidden.

**Execution: native replay.** For every query, policy and budget, the harness calls production `route_pinned_traced` + `resolve_pinned_candidates`. The scorer is held fixed (original Float16 scoring), so only selection varies. Work is reported as **live scored vectors** (original + Plain), not as the nominal budget.

**Policies**
- `slot` (default)
- `density`
- `tdensity2` (T=2)
- `reserved`
- `proportional`: an offline control that splits the learned budget by member population and always scores Plain.
- `static_flat`: one static LMI built by the native `SegmentBuilder` over the same 65,536 vectors with the same model config (52.9 s).

**Budgets:** 500–24,000.

**Query sets and oracles**
- **val:** rows 900–931 (32). Not used in recorded continuations 26–38; earlier use not excluded.
- **testA:** rows 932–963 (32), same provenance.
- **testB:** rows 9000–9255 (256), **DLI-unseen**. DLI work used rows 100–963, and the continuation24 pilot records rows 1000–9999 unused. These rows were evaluated only by the static LMI/HNSW SISAP campaign, which fits no DLI parameter. New exact oracle `continuation-42/test-oracle`: Float64 cosine over Float16-stored vectors, ties broken by external ID.

**Protocol (fixed before inspecting results)**
- Operating points are locked on val only (smallest budget with mean R@30 ≥ 0.95).
- Matched-work comparisons interpolate each query's recall along its own sweep and are paired against `slot`, with a 10,000-resample bootstrap.

## Results (testB, n = 256)

| Recall@30 at matched scored work | 2,000 | 4,000 | 6,000 | 10,000 |
|---|---|---|---|---|
| slot (DLI default) | 0.628 | 0.884 | 0.925 | 0.963 |
| tdensity2 Δ | **+0.196** [.178,.213] | +0.003 [−.002,.008] | **−0.005** [−.008,−.001] | **−0.006** [−.009,−.002] |
| density Δ | +0.094 | −0.007 | −0.008 | −0.005 |
| proportional Δ | +0.070 | **+0.006** [.003,.010] | **+0.012** [.008,.017] | +0.008 [.005,.011] |
| **static_flat Δ** | **+0.239** [.221,.257] | **+0.044** [.034,.053] | **+0.026** [.019,.034] | **+0.011** [.006,.015] |

**Val-locked operating points, R@30 ≥ 0.95** (scored work, then testA / testB recall):

| Policy | Scored work | testA R@30 | testB R@30 |
|---|---|---|---|
| slot | 6,054 | 0.950 | 0.926 |
| proportional | 5,620 | 0.950 | 0.933 |
| **static_flat** | **4,090** | 0.953 | 0.929 |

`reserved` is identical to `slot` at these budgets.

**Decomposition** (testB, mean scored vectors needed to cover a query's gold):

| Gold | Gold-bucket oracle (partition bound) | Router-ranked, perfect allocation | slot (actual) | static: oracle / ranked / actual |
|---|---|---|---|---|
| top-10 | 874 | 3,102 | 5,785 | 802 / 4,116 / 3,805 |
| top-30 | 1,967 | 7,193 | 9,656 | 1,623 / 9,161 / 7,003 |

The actual-cost columns exclude queries not covered within the sweep: 14–41 of 256.

**Miss attribution** (testB, slot, budget 10,000):
- 1.05 gold misses per query are in unselected buckets, and those buckets are mostly ranked >50 within their member.
- 0.05 per query are selected but lost. These are Float16-versus-Float64 near-ties against the oracle.
- The Plain buffer is never the cause at this budget.
- At budget 2,000, slot loses about 10 neighbours per query: Plain (priority 1.0) consumes 1,536 of the budget before most learned buckets.

## Conclusions

**VERIFIED** (one 65K topology, one seed, Float16 cosine, original scorer):
1. **At equal scored work, a single static LMI over the same data beats every DLI allocation policy,** by +0.011 to +0.239 R@30, with all CIs excluding 0. At the val-locked R@30 ≥ 0.95 target it needs about 32% less work than slot. The geometric multi-level structure costs selection efficiency.
2. **The main inefficiency is router ranking.** The gold-bucket oracle needs about 3.5× less work than the router's own ranking. Partition granularity is cheap: about 900 vectors cover top-10.
3. **DLI adds a cross-level allocation loss** on top of ranking: about 1.8× over perfect allocation for slot at top-10. It also adds a fixed Plain cost.
4. **T=2 helps only at very low work** (≤ 3K) and is slightly harmful at ≥ 6K. It is not a general improvement, which confirms the earlier "inconclusive" status.
5. **Proportional per-member allocation** is a small, significant improvement at ≥ 4K (+0.6 to +1.2 pp) and worse at 2K.

**Implementation observations**
- Selected-but-lost misses are 0.02–0.05 per query: execution, resolution and visibility are not a measurable recall source here.

**Not promoted:**
- Proportional's gain is small compared with the structural gap to static, so it is not implemented natively. Native latency was not measured, as this is a replay study.

**Hypotheses for future work (not tested)**
- Calibrating member probabilities onto a common scale.
- Matching bucket size across levels. Each 548-bucket member has a different average bucket size: about 29 vectors in L1 and 88 in L2, versus 120 for static.
- Merging levels with a shared model at compaction.

## 10M validation of the harness

`continuation-42/frontier-10m-validation*` used the retained continuation25 10M geometric topology:
- Five members (levels 0, 1, 2, 4, 5), copy verified by tree hash.
- 4 val queries, slot and proportional, budgets 20K and 160K.

The run completed in 182 s with 11.9 GB RSS. Local timing is meaningless: 283K major page faults with 15.5 GB of vectors on a 23 GB machine. It established:
- The harness works at full scale.
- The 10M Plain buffer is about 125K vectors, which `proportional` scores on every query.

This motivated the preregistered H-10M HPC job: `hpc-bundle-20261117/dli-10m-frontier.pbs`.
