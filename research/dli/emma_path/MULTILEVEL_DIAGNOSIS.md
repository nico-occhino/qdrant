# The multi-level penalty at 10M: diagnosis (exploratory)

**Status: exploratory, post hoc**, 2026-11-22. It uses job 25827532's E1 and G0 frontier
outputs on the **test queries already analysed** (rows 9000–9255). Nothing here is a
confirmatory result. The interventions it motivates are preregistered in
[PHASE3_PREREGISTRATION.md](PHASE3_PREREGISTRATION.md) Part B, on queries this note did not use.

## Question

E1 needs 81.6K scored vectors to reach mean test R@30 = 0.90. A static LMI with the same
11,100 buckets and the same model rule needs 34.0K ([RESULTS_10M.md](RESULTS_10M.md), H4).
Does the 2.4× gap come from **selection** (routing across levels) or from **structure**
(the partitions themselves)?

## Method: oracle cost separates structure from selection

`reference_costs.jsonl` records, per query, the **oracle cost**: the smallest total size of
whole buckets that together contain the query's true top-30. It is computed for the DLI
topology and for the static reference. The oracle knows the answer, so it measures the
partition alone, with perfect routing. Comparing the two decomposes realized work:

realized work = oracle cost × (realized / oracle).

The first factor is **structural**. The second is **selection inefficiency**.

## Results (test, 256 queries)

| | DLI oracle | Static oracle | Ratio | Realized W@0.9 (DLI / static) | Realized / oracle |
|---|---|---|---|---|---|
| **E1** | 29,456 | 12,084 | **2.44×** | 81,580 / 34,023 | 2.77 / 2.82 |
| G0 | 134,103 | 135,413 | 0.99× | 295,626 / 165,191 | 2.20 / 1.22 |

1. **E1's penalty is structural.** Even a perfect selector over E1's buckets needs 2.44× the
   work of a perfect selector over the static buckets. That matches the realized gap
   (2.40×). E1 selects as efficiently as the static index (realized/oracle 2.77 vs 2.82).
2. **G0's penalty was selection.** Its partitions are as good as static (oracle 0.99×), but
   it selects 1.8× less efficiently. That is the Plain buffer, as diagnosed in H-10M. Two
   arms, two different mechanisms.

**Where the structural cost comes from:**

| Level | Live | Buckets | Mean bucket | Size-weighted mean (what a random neighbour costs) | Queries needing the level | Median rank depth |
|---|---|---|---|---|---|---|
| L4 | 5.5M | 10,000 | 551 | 1,004 | 256 / 256 | 14 |
| **L3** | 4.4M | 1,000 | **4,444** | **6,141** | 253 / 256 | 4 |
| L2 | 220K | 100 | 2,200 | 2,648 | 135 / 256 | 1 |
| static | 10.12M | 11,100 | 912 | — | — | — |

Sommerová's invariant gives a *full* level buckets of about b = 5,000. L3 is 88% full, so
its buckets are almost that size, about 5× coarser than the static index at this scale. L4
was created recently and is 11% full, so its buckets are small. L3 holds 43% of the data,
and almost every query needs it, so its coarse buckets dominate the cost. The routers are
not the problem: L3's needed buckets sit at median rank 4.

## What this predicts for the two interventions

**(a) Level-aware budget allocation** (selection) can at most close the selection factor.
E1 already selects like the static index, so the gain is bounded. It might still improve
both, since calibration helps any index, but it cannot remove the 2.44× structural factor.
Expectation: a modest gain.

**(b) Consolidating co-resident large levels** (structure) attacks the cause. If L3's
points were merged into L4 and routed by L4's router, they would spread over 10,000
buckets: 9.9M / 10,000 ≈ 990 per bucket, close to the static 912.

A planner simulation ([`consolidation_sim.py`](consolidation_sim.py), pure arithmetic) uses
this rule: level i above the deepest non-empty level D may hold at most ρ·live(D);
otherwise levels 0..i merge into D, reusing D's router. The **bucket-cost proxy** is
Σ_level share × mean bucket size, plus the Plain buffer.

| ρ | Final levels at 10.12M | Write amplification | Cost proxy vs E1 |
|---|---|---|---|
| off (E1) | L2 220K, L3 4.4M, L4 5.5M | 14.4 | 1.00 |
| 0.5 | L1 35K, L2 55K, L3 2.2M, L4 7.83M | 13.8 | 0.49 |
| **0.25** | L1 40K, L3 1.1M, L4 8.98M | 14.2 | **0.41** |
| 0.1 | … | 16.2 | 0.48 |
| 0.05 | … | 24.4 | 0.45 |

At ρ = 0.25 the proxy reaches the static granularity (912 / 2,260 = 0.40) **with no extra
write amplification**: merging into L4 replaces rewrites of L3 that Bentley–Saxe would have
done anyway. Below ρ ≈ 0.1 the extra merges start to cost writes.

**The gain depends on the snapshot.** Same rule, other populations (proxy vs E1, ρ = 0.25):

| N | 1M | 3M | 6M | 10.12M | 20M | 50M |
|---|---|---|---|---|---|---|
| Proxy | 0.48 | 1.06 | 0.71 | 0.41 | 0.96 | 0.91 |

Consolidation helps only while two large levels coexist. At 3M or 20M one level already
dominates. And a dominant level that is itself nearly full (20M: 16.5M in 10,000 buckets ≈
1,650 each) is coarse anyway. So **any single-snapshot comparison can be cherry-picked**.
The fair estimand averages over the ingestion cycle, which Phase 3's evaluation after every
batch provides (Part B).

## Caveats

- Post hoc, on analysed queries; one replicate.
- The proxy ignores router quality. It assumes L4's reused router spreads L3's points as
  evenly as its own, which needs checking: L4's router was trained on a 400K sample of
  the first 5.5M points. Under distribution shift this is exactly where consolidation
  should hurt: one stale router for 90% of the data. Part B therefore also runs the
  consolidated arm in Shift-M.
- A third structural option was not simulated: more buckets per level than aⁱ (a target
  bucket size s < b). It multiplies the cost of training fresh levels, and the one
  10,000-bucket build already took 1,337 s with 16 CPUs.
