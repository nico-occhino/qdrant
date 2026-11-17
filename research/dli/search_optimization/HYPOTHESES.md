# Search hypotheses — continuation 26

| Hypothesis | State | Discriminating evidence / next measurement |
|---|---|---|
| Source-order inheritance changes vector placement | Supported and corrected | Twelve destination builds/reopens; explicit collection memory precedence now overrides inherited backend. Existing files untouched. |
| Geometric routing repeats size-tiered's exact model selection behavior | Rejected | Four geometric routers, 6M Cold selected100/100; baseline scope10 .600, not .503. |
| Slot multipliers correct probability-scale differences in this implementation | Unsupported assumption | Sommerová §3.5.1 derives multiplication from growing output counts; our output count548 is constant. Uniform scores become2^i/548. |
| Reserving each member improves coverage efficiently | Supported offline; bounded native execution measured | .643 bucket10 at28,750 mean offsets versus slot .567 at32,340. Whole-bucket floor is expensive. No final-recall/latency improvement claimed yet. |
| Population-density priority is universally inferior | Rejected | Poor at same nominal2000, useful at matched work: .633 bucket10 at29,007 offsets. Demonstrates nominal-budget comparisons can mislead. |
| Global selection is the only recall bottleneck | Rejected | Reserved scope10 .987 but bucket10 .643. Substantial within-member ranking misses remain. |
| Returning every scored hit is necessary for visibility correctness | Supported in bounded native controls, opt-in | Per-member top-k can contain all possible global winners. Every retained ID still checks complete pinned membership; ambiguity invokes exact fallback. Mutation/recovery oracle passes; identical-work native warm full/top-k totals1.260/.405s in debug, not a release claim. |
| Cold vector access explains seconds-scale latency | Supported major-fault access mechanism | All baseline G queries select Cold; this gives no within-G storage control. New task-local trace:75,702 major faults and18.924s access versuszero faults/.306s on repeat; detailed counts in04. |
| Larger training sample fixes the recall gap | Proposed, untested | Existing final models have no persisted teacher centers; diagnose gold-neighbor bucket rank first. No new training or ingestion justified yet. |

## Continuation27 release evidence

|Hypothesis|Result|Evidence|
|---|---|---|
|Top-k saves time only in debug|Rejected|Same-work/same-quality release p50 129–138→25–26ms; all72 ID/score/version pairs match modulo equal-score order.|
|Reserved/density gives a routing Pareto win|Not supported|At actual25.6–27.9K work, recall improves but latency/CPU costs also increase; selected points are not recall matched.|
|Cold outliers are mostly neural inference|Rejected|Cold-member clearing produces356K–442K major faults and seconds per query; four-router inference remains≈5ms.|
|Warm tails are residual major faults|Rejected on observed outliers|Held-out warm major faults0; tails are in CPU/native scoring. Arithmetic/access/core-placement contributions not isolated.|
|Density sort must re-evaluate priorities|Rejected/corrected|Caching preserves1200 ordering/count cases; isolated optimized frozen-frame planner median1.204→.470ms. End-to-end post-fix speed unmeasured.|
|Temperature calibration is an established gain|Inconclusive|Validation T2 lowers neighbor-bucket NLL; held-out inclusion+.007 with bootstrap95%[-.015,.028], no native deployment.|
|Top-k must preserve deterministic ID order at equal score|False contract|Native comparator is score-only. Four paired lists reorder ties, with identical IDs/scores/versions and score sequences.|

Full results and sampling/measurement limitations: [07_RELEASE_RESULTS.md](07_RELEASE_RESULTS.md).
