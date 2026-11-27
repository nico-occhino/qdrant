# Q7. Filtered search on the shared fixture (preregistration)

**Status: DRAFT.** It is frozen before any measured run. Per the author brief, it is submitted only after Q5's first
job validates.

This is the Qdrant side of PostgreSQL's frozen shared fixture `FILTER_FIXTURE_SPEC.md` (3ec4041, pinned in
`PINS.md`). The fixture is implemented exactly as specified, by `q7_fixture.py`. That script checks every published
value: the slice SHA-256 `c1f24a8d…`, the `filter_rank` SHA-256 `3f0c875c…`, the first five ranked ids, the query
SHA-256 `fb72bd7f…`, and the 60 / 600 / 6,000 / 30,000 matching counts. All of them passed locally on 2026-11-27.

## 1. Data and representation
- **The data.** 60,000 vectors (local ids 1–60,000) and the 32 queries. The predicates are
  `filter_rank ≤ 60 / 600 / 6,000 / 30,000` (0.1%, 1%, 10% and 50%); k = 30.
- **Stored representation.** Cosine, Float16 vector storage. Qdrant normalizes on insert, by its AVX+FMA path.
  Queries are widened from binary16 to Float32 losslessly, then normalized by Qdrant.
- **The payload.** `filter_rank` is set before index construction, as an indexed integer payload with range
  support. Matching ids are enumerated from the payload index; this is recorded per query as `payload_matches`.
- **Gold.** Exact cosine over Qdrant's stored vectors for every qualifying live id, by Qdrant's exact scorer: a
  full-scan filtered search with `exact = true`.
  - **Ordering.** Ranked by distance, then ascending local id.
  - **What is recorded.** The rank-30 and rank-31 boundary and the exact score ties.
  - **Gold overlap with PostgreSQL** is descriptive only.

## 2. Arms (all Qdrant-native)
| Arm | Index | Filtered strategy |
|---|---|---|
| **S-LMI** | static MLP LMI over all 60,000, 250 buckets, Qdrant's default LMI training rule | Qdrant's adaptive planner (`choose_filtered_plan`): `ExactMatches` below its selectivity threshold, otherwise `RoutedIntersect` |
| **DLI** | E1 hierarchy from inserting ids 1–60,000 in order (buffer 5,000, arity 10, aⁱ buckets): level 2 holds 55,000 (100 buckets), level 1 holds 5,000 (10 buckets) | Qdrant-native fallback, because DLI's own search path rejects filtered requests (`dli_research.rs`). Each member runs the same adaptive filtered search, and the plain buffer runs exact |
| **HNSW** | m = 16, ef_construct = 100, the default full-scan threshold, a payload index on `filter_rank` | Qdrant's filtered HNSW, including its own exact path for small cardinalities |

- **Native operating point.** The plan each arm chooses at its configured settings: LMI `nprobe` from the model
  configuration, HNSW `ef` = 128. It is reported per predicate, together with the chosen plan.
- **Curves:**
  - **LMI and DLI.** `FilteredSearchMode::ForceRouted { nprobe }` sweeps n ∈ {1, 2, 4, …, buckets} (for DLI, the
    same n is applied to each member, capped at its bucket count). Separately, `ExactMatches` is recorded.
  - **HNSW.** The `ef` sweep is {30, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1,024}.

## 3. Per query, strategy and predicate (spec §"Search, ground truth and reporting")
For each, we record:
- the qualifying count, returned ids and gold ids;
- Recall@30, and underfill (30 minus the distinct qualifying ids returned);
- whether any id is a duplicate or non-qualifying, which is a hard failure;
- **the actual vectors scored**, from Qdrant's per-task scoring counter (`TaskScoringProfileGuard`), which counts
  every distance evaluation including widening restarts and rescoring. Approximate and exact work are separated by
  plan: routed vs `ExactMatches` / full scan;
- the payload enumeration count (separately from vector work), and the named plan.

## 4. Summaries
- **Per predicate and strategy.** Mean Recall@30, mean underfill and mean scored work at the native point.
- **Per curve.** W@0.90 by the first measured crossing (never extrapolated; `upper_bound` and unattained are
  explicit), and Recall@30 at the curve point nearest to and at least 50,000 scored. For the 0.1% and 1% predicates
  that point is unreachable: the work is bounded by the qualifying count, which is reported instead.
- **Latency** is reported only within this job and arm, single-threaded, as the median of 5 repetitions per query.
- **Cross-engine.** Recall, underfill and scored work are compared with PostgreSQL's fixture run once it exists.
  The gold overlap is descriptive.

## 5. Gates
- **Fixture checks.** All spec hashes, and the gold computed by two independent paths (Qdrant exact search, and a
  NumPy model of the stored vectors) agree on ids up to exact ties.
- **No returned id may be duplicate or non-qualifying.**
- **ExactMatches must give Recall@30 = 1.0 (ties aside),** with scored work equal to the qualifying count.
- **Work counters must be monotone** in `nprobe` and `ef`, with one exception, as in final-benchmark A3. A drop from
  n to 2n is accepted when the routed plan widened to fill k and the per-query drop is below the restarted first
  pass. All drops are listed.
  - **Why the exception is needed.** The local smoke showed S-LMI at the 1% predicate scoring 98 vectors at
    nprobe 1 but 84 at nprobe 16.
- **Exact ties.** Gold ties at the rank-30 boundary are classified per query (`s30`, `s31`), and a shortfall of the
  exact arms is admitted only as such a tie. In the smoke, DLI exact at 50% gave 0.999; it will be classified in the
  full run.

## 6. Local smoke (2026-11-27; runner check, not data)
- **What ran.** `Q7_ROWS=6000`, which gives DLI = level 1 (ids 1–5,000) plus a 1,000-row buffer. It produced 4,224
  records, with no duplicate or non-qualifying ids.
- **What it confirmed.** S-LMI's exact path scored exactly the qualifying count. On 6,000 rows, HNSW sits below its
  full-scan threshold, so it searched exactly; the full fixture exercises the graph.
- **A runtime setting.** Builds need `QDRANT_LMI_SAMPLE_BUDGET_BYTES=2000000000`, as the Phase 3 jobs set.

## 7. Execution
- **Where.** One local smoke on a 6,000-row prefix of the fixture, with separate labels. Then one MetaCentrum job (8
  CPUs, ≤ 4 h) on the full fixture, staged from a pushed commit.
- **Implementation.** `lib/segment/tests/` plus the collection harness, adapted from `lmi_filter_probing_sisap300k.rs`.
