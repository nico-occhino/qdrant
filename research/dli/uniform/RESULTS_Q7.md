# Q7. Filtered search on the shared fixture (results)

**Status: COMPLETE; every gate passes under Amendment A2** (applied by the author's decision, after the data). The
frozen monotone-work gate failed in the job on DLI at 1% (§1, §5). A2 applies the restart bound per DLI member. Under
it the gate passes, with 20 admitted restarts, and no reported number changes (`analysis-A2.json`, SHA-256
`9802335f…337c`). No rerun was made.

- **The protocol.** [Q7_PREREGISTRATION.md](Q7_PREREGISTRATION.md), frozen at `e85c176f7`, with descriptive
  Addendum A1 (`a6b067075`, before results).
- **The job.** 26076922 (8 CPUs): binary `c357b0b5…`, source key `eeed84f487969debf1a053eb`. Search took 8.2 s,
  build plus search under 1 min, and peak RSS was 1.4 GB.
- **The archive.** `C:\Users\nicoo\Workspace\Research-Results\2026-11-28-uniform-q7\` (21 files,
  `STORAGE_SHA256.txt` verified). `rows.jsonl` SHA-256 `1b011c6d…97bf`, `analysis.json` `949e92dd…7306`.
- **The PostgreSQL side.** `FILTER_FIXTURE_PG_RESULTS.md` at `04863df` (pinned).

## 1. Gates
| Gate | Result |
|---|---|
| Fixture: every spec hash and the 60 / 600 / 6,000 / 30,000 matching counts | pass |
| No duplicate or non-qualifying id (4,224 records) | pass (0) |
| ExactMatches gives Recall@30 = 1.0, with work = qualifying count | pass (S-LMI and DLI, all predicates) |
| Underfill | 0 rows underfilled |
| Monotone work, widening restarts admitted (n → 2n, every per-query drop < 30) | frozen rule: **fail** (DLI nprobe 2→4 and 4→8 at 1%, per-query drops 39 and 52). Under A2 (per-member bound): **pass**, 20 restarts admitted |

- **Why the monotonicity gate fails.** The DLI arm is two members: level 2 (ids 1–55,000, 100 buckets) and level 1
  (55,001–60,000, 10 buckets). Each runs its own routed search, and each can widen and restart.
  - **Per member, every drop is a restart within the rule.** Over all 267 query-level drops on DLI curves, the
    largest per-member drop is 28 (level 2) and 27 (level 1), always < 30.
  - **The failing drops are two member restarts summed.** For example, at 1%, query 1, nprobe 4→8: level 2 drops
    83→60 and level 1 drops 74→50, a total of 157→110.
  - **Why the rule misfires.** The frozen rule was written, as in final-benchmark A3, for a single routed search.
- **Two declared omissions (§3–§4 of the preregistration).**
  - The harness did not record the per-query payload-enumeration count (`payload_matches`).
  - It recorded one `elapsed_ns` per row, not the median of 5 repetitions. Latency is therefore not reported.
  - Vectors scored, which is the cross-engine measure, is recorded in full.

## 2. Exact versus routed, in vectors scored (mean over 32 queries)
| Predicate (qualifying) | Exact | S-LMI routed W@0.90 | DLI routed W@0.90 | HNSW W@0.90 | PostgreSQL routed W@0.90 |
|---|---|---|---|---|---|
| 0.1% (60) | 60 | 55.8 | 52.1 | 60 (upper bound; exact path) | 50,088 |
| 1% (600) | 600 | 273.8 | 288.4 | 600 (upper bound; exact path) | 26,558 |
| 10% (6,000) | 6,000 | 713.4 | 1,119.3 | 6,000 (upper bound; exact path) | 12,701 |
| 50% (30,000) | 30,000 | 1,853.8 | 3,624.8 | 753.1 (upper bound; graph) | 11,277 |

- **Recall at a nominal 50K is unreachable for every Qdrant curve.** The work is bounded by the qualifying set.
- **Restarts make the low-nprobe points expensive.** At 0.1% and 1%, the routed S-LMI and DLI curves score more
  than the qualifying count at small nprobe (for example 83 vectors for 60 qualifying, S-LMI at 0.1%). Widening
  restarts re-score from scratch, and recall only rises at large nprobe.
- **HNSW at ≤ 10% runs Qdrant's exact path** (cardinality below its full-scan threshold). Its "curve" is therefore
  flat at the qualifying count, with recall 1.0.

## 3. The switch point (Addendum A1)
D(p) = routed W@0.90(p) − qualifying count.

| Arm | D at 0.1 / 1 / 10 / 50% | Switch point m\* |
|---|---|---|
| PostgreSQL routed | +50,028 / +25,958 / +6,701 / −18,723 | **≈ 12,326 qualifying vectors (≈ 20.5%)** |
| Qdrant S-LMI | −4.2 / −326 / −5,287 / −28,146 | **below 0.1%** (routed never scores more) |
| Qdrant DLI | −7.9 / −312 / −4,881 / −26,375 | **below 0.1%** |
| Qdrant HNSW | 0 / 0 / 0 / −29,247 | set by Qdrant's planner, not measured: exact at ≤ 10%, graph at 50% |

- **The reading.**
  - **The opposite results have one cause: where the filter is applied.**
    - **PostgreSQL's routed arm post-filters.** It materializes up to the budget of candidates, then applies the
      predicate, with no predicate index. Its work grows as the selectivity falls, so exact search wins up to about
      20% selectivity.
    - **Qdrant's routed plan pre-filters.** `RoutedIntersect` intersects the probed buckets with the payload-index
      match set and scores only qualifying points. Its work is a fraction of the qualifying count, so in vectors
      scored it never loses to exact search.
  - **At 0.1% the margin is 4–8 vectors,** a practical tie, and exact search gives recall 1.0 rather than 0.90.
  - **What vectors scored leaves out.** It excludes payload-index enumeration, heap and segment traversal, and
    PostgreSQL's candidate materialization. This is a work comparison, not a latency or production verdict.

## 4. Qdrant's native operating points (the planner's own choice)
| Predicate | S-LMI (nprobe 16): plan, recall, work | DLI (E1, nprobe 1 per member): plans L2+L1, recall, work | HNSW (ef 128) |
|---|---|---|---|
| 0.1% | ExactMatches, 1.000, 60 | Exact+Exact, 1.000, 60 | exact path, 1.000, 60 |
| 1% | ExactMatches, 1.000, 600 | Routed+Exact, 0.566, 136 | exact path, 1.000, 600 |
| 10% | RoutedIntersect, 0.825, 398 | Routed+Exact, 0.508, 553 | exact path, 1.000, 6,000 |
| 50% | RoutedIntersect, 0.913, 2,014 | Routed+Routed, 0.594, 566 | graph, 0.996, 2,216 |

- **S-LMI.** The planner picks exact search where the routed alternative saves only a few hundred vectors, and
  routes at 10% and 50%. At 10%, nprobe 16 misses 0.90 (0.825); 0.90 needs about 713 vectors (nprobe ≈ 32).
- **DLI.** Its native nprobe 1 per member (the E1 configuration) is far below 0.90 whenever the large member
  routes (0.51–0.59). DLI's own search path rejects filters, so this fallback is not tuned for filtered search.
- **HNSW** reaches ≥ 0.996 at every predicate, at exact cost up to 10%.

## 5. Amendment A2 (applied 2026-11-28 by the author's decision; recorded in the preregistration §9)
- **The proposed rule.** Apply the widening-restart exception per member. A drop in a query's total DLI work from n
  to 2n is admitted when, for every member, that member's work drop is < 30, from the per-member `plans[].work`
  already in the rows. Single-index arms are unchanged.
- **Its effect on these data.** All DLI drops are admitted (largest per-member drop 28), and the gate passes. No
  number in §2–§4 changes: the analysis does not depend on the gate's outcome.
- **The rejected alternative** was to keep the failure, and report Q7's DLI curves as non-monotone at the query
  level. That is still true of the raw curves.
