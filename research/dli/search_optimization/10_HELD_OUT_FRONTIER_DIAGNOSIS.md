# Held-out frozen 10M frontier

Executed revision `dce9f1a54c85de06f7e19887b781003343d7b138`; release, 12 held-out official queries 600–611. Validation 400–499 selected budgets and T=2. Native logs and plots: `<admin>/experiments/2026-11-12-dli-architecture/continuation-28/native-frontier`. 492 learned queries, zero fallback; four exact controls are separate. Small sample and sequential cache/order effects limit inference.

|ID|policy|budget|scored/query|R@10|R@30|scope@30|bucket@30|p50 ms|p95 ms|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|2|SlotWeighted|200000|208525|0.8417|0.8500|1.0000|0.8500|93.2|241.8|
|5|SlotWeighted|280000|287522|0.9500|0.9417|1.0000|0.9417|134.4|163.7|
|8|SlotWeighted|500000|504371|0.9750|0.9722|1.0000|0.9778|236.0|273.4|
|11|ReservedPerLevel|200000|208525|0.8417|0.8500|1.0000|0.8500|87.6|92.0|
|14|ReservedPerLevel|280000|287522|0.9500|0.9417|1.0000|0.9417|135.5|149.6|
|17|ReservedPerLevel|500000|504371|0.9750|0.9722|1.0000|0.9778|227.0|242.7|
|20|PopulationDensity|90000|140354|0.8250|0.8361|1.0000|0.8361|70.4|91.4|
|23|PopulationDensity|240000|251117|0.9333|0.9306|1.0000|0.9306|115.3|132.8|
|26|PopulationDensity|400000|406454|0.9667|0.9583|1.0000|0.9611|181.3|195.1|
|29|TemperatureDensityT2|80000|92706|0.8250|0.8222|0.9917|0.8222|69.2|87.4|
|32|TemperatureDensityT2|180000|196225|0.9417|0.9389|0.9917|0.9417|123.1|139.2|
|35|TemperatureDensityT2|340000|371827|0.9667|0.9611|1.0000|0.9667|186.8|218.0|

For three-objective dominance (minimize mean_scored, maximize both recalls), efficient IDs: 5, 8, 14, 17, 20, 29, 32, 35; dominated IDs: 2, 11, 23, 26. Equal results are retained as ties. This is descriptive, not statistical significance.

For three-objective dominance (minimize p50_ms, maximize both recalls), efficient IDs: 5, 11, 17, 20, 23, 26, 29, 32, 35; dominated IDs: 2, 8, 14. Equal results are retained as ties. This is descriptive, not statistical significance.

## Interpretation

All listed configurations use top-k materialization and the same held-out rows. ID2 is slot/T1/200K, 11 reserved/T1/200K, 26 density/T1/400K, 32 density/T2/180K, 35 density/T2/340K. IDs2 and11 have identical selected work and quality. At comparable work, ID32 scores 196,225 versus 208,525 vectors for ID2 (6.27% relative spread), Recall@10 .9417 versus .8417 and Recall@30 .9389 versus .8500. Both evaluate four models. Model scope@30 is .9917 versus 1.0; bucket coverage@30 is .9417 versus .8500: the gain comes from bucket selection, not better model coverage. Actual latency p50 is 123.1 versus 93.2 ms, so the work advantage is not a demonstrated latency advantage. The same twelve query IDs are paired. All studied policies and settings are exploratory at this sample size.

The 36 priority pairs compare three later passes to the first of four ABBA control passes, on 12 identical queries: equal scored counts and ID/score/version sets. This proves observed result/work equivalence, not identical event ordering or an end-to-end speedup. Mean planner time uncached 1.39/1.29 ms versus cached .85/.83 ms; warm p50 uncached 27.38/25.94 ms versus cached 30.82/30.22 ms. Noise/order outweighs the isolated planner gain here.

Exact control query603 exchanges official ID550836 (raw Float16 cosine .78939184) for ID9934145 (.78922763); native score of the latter .78946722. Raw-cosine margin is .00016421; observed native numerical score differs from raw normalized cosine by up to .00035550 across exact results. Float16 preprocessing/normalization and ranking precision plausibly reverse this near-boundary pair, but the stored native score for the missing ID was not measured: causation remains unproven. Four queries are insufficient to characterize exact recall. Official gold and raw vector source semantics must be retained.

First-touch is not uniformly cold: case1 incurs 415,419 major faults and p50 8765ms, while its warm repeat incurs zero and 93ms. Later first-touch cases benefit from earlier access; no clean cold comparison is justified. For case32 warm, mean scorer108.1ms, router4.89ms, planning .82ms, resolution1.06ms and materialization .15ms; scorer dominates. These phase counters are nested and should not be summed as independent wall times.

The private 65,536-point teacher control gives training-point assignment agreement .99945 and held-out pool agreement .80310; query teacher top bucket appears in model top4 for .96875 of 32 queries. This is a different model and dataset subset, not the original 6M teacher. It motivates a controlled generalization study, not full-scale retraining. Global allocation is demonstrably improvable, while classifier generalization may also limit bucket quality; neither mechanism is established as the sole dominant limitation.

Plots `recall-work.svg`, `recall-latency.svg`, `recall-p95.svg` and `offline-frontier.svg` are in the raw run directory. Native detailed rows: `oracle-analysis/cases.csv` and `per-query.csv`; teacher factorial: `teacher-control/teacher-frontier.csv`. No static 10M quality baseline is asserted.
