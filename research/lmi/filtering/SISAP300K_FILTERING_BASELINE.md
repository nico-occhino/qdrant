# Adaptive filtered search: SISAP23 300K baseline

**Status:** complete, 2026-11-19. This is the reference measurement for the
filtering planner as committed in `1b323f496`. It is the baseline (policy B0)
for the follow-up probing experiment in `PREREGISTRATION_FILTER_PROBING.md`.

**Evidence (outside Git, read-only):**
`research-admin/experiments/2026-11-19-lmi-adaptive-filtering/`
- `result-nprobe16.json` (sha256 `9042bacc…2b68`) and `run-nprobe16.log` (`1600854d…8c68`).
- `PROVENANCE.md`: source commit, test-file and binary hashes, toolchain, host, command.
- `inputs.sha256`: matches `continuation-22/benchmark-300k/manifest.json`.

**Code:** planner `lib/segment/src/index/lmi_index/filtered.rs` (commit `1b323f496`);
benchmark `lib/segment/tests/lmi_filtered_sisap300k.rs` (sha256 `311aae3a…ca60`,
identical to the file that produced the results).

## Setup

- **Data:** canonical SISAP23 LAION CLIP-768 300K subset (rows 0..299999), Float16 storage, cosine.
- **Queries:** official public queries, rows 0..999. They were not used to tune anything here.
- **Payload:** `r = splitmix64(id) mod 10^6`, integer-indexed. The filter `r < s·10^6` matches a
  fraction *s* of the points and is independent of the vectors by construction.
- **Indexes:** both are built natively by Qdrant from the same source segment. They share 548 buckets,
  the spherical k-means teacher (S = 32,768, 5 iterations, seed 42) and `nprobe` = 16.
  - IVF: `LMI_RESEARCH_ROUTER=centroid`.
  - LMI: MLP with hidden size 512, 30 epochs.
- **Planner:** if the payload-index cardinality estimate is at most the probe volume
  (`nprobe·N/B` ≈ 8,759), it scores the matching points exactly. Otherwise it routes, intersects the
  probed postings with the filter, and widens the probe only to fill `min(top, live)`.
- **Gold:** native exact filtered search over the same source segment.
- **Timing:** wall time of one `VectorIndexRead::search` call, one query per call, one thread,
  warm cache, `perf` profile, Intel Core Ultra 9 285H.

## Results (test queries 0..999, top 10)

| Selectivity | Matching | Plan | IVF R@10 | MLP R@10 | IVF p50 | MLP p50 | Exact filtered p50 |
|---|---|---|---|---|---|---|---|
| 100% | 300,000 | routed | 0.953 | 0.934 | 7.72 ms | 6.07 ms | 229.2 ms |
| 50% | 150,002 | routed | 0.945 | 0.923 | 4.70 ms | 3.44 ms | 104.0 ms |
| 10% | 29,960 | routed | 0.914 | 0.884 | 1.14 ms | 1.05 ms | 20.2 ms |
| 5% | 14,895 | routed | 0.896 | 0.857 | 0.61 ms | 0.61 ms | 9.67 ms |
| 3% | 8,995 | routed (sampled) | 0.874 | 0.828 | 0.43 ms | 0.42 ms | 4.99 ms |
| 1% | 2,973 | exact | 1.000 | 1.000 | 0.32 ms | 0.32 ms | 0.32 ms |
| 0.3% | 879 | exact | 1.000 | 1.000 | 0.05 ms | 0.05 ms | 0.05 ms |
| 0.1% | 298 | exact | 1.000 | 1.000 | 0.02 ms | 0.02 ms | 0.02 ms |
| 0.01% | 30 | exact | 1.000 | 1.000 | 0.004 ms | 0.004 ms | 0.004 ms |

**Unfiltered reference**

| | IVF | MLP |
|---|---|---|
| Official R@10 | 0.956 | 0.937 |
| p50 | 5.51 ms | 5.16 ms |
| Native build | 6.7 s | 51.2 s |

p95 values are in the JSON. The run took 8 min 04 s, with peak RSS 2.3 GB.

## Findings

1. **Correctness.** All 18,000 filtered queries returned only matching points and filled `min(10, matching)`.
   Exact plans equal exact search by construction and in measurement. No query needed widening at `nprobe` 16.
2. **Plan choice.** The plan follows the estimate: exact up to 2,973 matches, routed from 8,995. The 3% case
   straddled the threshold and was settled by sampling. Routed plans are 12–38× faster than exact filtered search.
3. **Filtered recall falls with selectivity at fixed `nprobe`.** It goes from 0.953 to 0.874 (IVF) and from 0.934 to 0.828 (MLP).
   The filtered top 10 lies deeper in the routing order. The break-even counts only vectors scored, so near the
   threshold it trades recall for speed: at 3%, exact search would return recall 1.0 in about 5 ms.
4. **IVF beats the MLP router under filters too.** Latency is equal, but the recall gap grows from 0.019 to 0.046 as
   selectivity drops. This is consistent with the static Tier 1–3 result.
5. **Filter checks are not free.** At 100% selectivity the same probe costs about 2.2 ms more than unfiltered search
   (IVF). The planner's break-even ignores this cost.
   *Erratum:* the later stage pass shows that filter checks account for only ≈0.75 ms of this (15.6 ns per
   evaluation). See `FILTER_PROBING_REPORT.md`.
6. **Exact filtered search has its own cost cliff.** It costs about 0.11 µs per matching point at 1% and about
   0.55–0.76 µs from 3% up to 100%. This is probably Plain switching from payload-index iteration to a full scan with
   per-point checks. Not verified.

## Limitations

- One run, one machine, one seed and one build per router. Intervals cover none of these.
- One `nprobe` value, independent filters only, a single integer range predicate.
- Query rows 0..999 have appeared in earlier campaigns' evaluations; no validation/test split was used here.
- Timing is index-level (no REST, no collection fan-out) and is not paired or interleaved between systems.
