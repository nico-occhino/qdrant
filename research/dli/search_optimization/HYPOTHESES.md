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
