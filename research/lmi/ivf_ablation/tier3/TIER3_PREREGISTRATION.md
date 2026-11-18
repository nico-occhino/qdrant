# Tier 3 preregistration: native Qdrant v1.20 latency for static LMI, native IVF, HNSW and Plain

Written 2026-11-18. At that time Tier 2 F1 was complete and F2+F3 and F4 were still running. The native profiling job (25718667) was queued or running. **No Tier 3 benchmark has been submitted.**

## Systems

All are built natively by `SegmentBuilder` from one 10,120,191-vector Float16 cosine source segment, in the same job, on the same node.

| Name | Index | Notes |
|---|---|---|
| plain | Plain (exact) | 256-query test subset only (about 0.5 s per query) |
| hnsw | HNSW m = 16, ef_construct = 100, 16 build threads | Qdrant default graph; built new on v1.20 |
| lmi-native | LmiTrained B = 3,162, S = 250,000, H = 512, 30 epochs, 5 iterations, seed 42 | The accepted static LMI configuration |
| ivf-3162 | Same configuration with `LMI_RESEARCH_ROUTER=centroid` | Native counterpart of Tier 2 F1 centroid\|centroid (same teacher algorithm) |
| ivf-promoted | The Tier 2 configuration with the lowest **test** W_total at R@10 = 0.95, built with `LMI_RESEARCH_ROUTER=centroid` | Included only if it differs from ivf-3162. Built only if the native profile predicts it fits the walltime. |

**Historical v1.19 numbers** (Nov 3 campaign, deimos11) may appear in reports only under a "historical, not matched" label.

## Protocol

- **Boundary.** Index-level search through `ann_matched_v120` (the same boundary as the v1.19 matched harness):
  - LMI/IVF: `router.top_buckets(nprobe)`, then the posting union, then Qdrant `score_candidates`;
  - HNSW: `VectorIndex::search` with `hnsw_ef`;
  - Plain: `PlainVectorIndex::search`.

  The decomposed LMI path omits underfill widening, so result length is checked and reported. Top = 30.
- **Settings.** LMI/IVF nprobe {4, 8, 16, 30, 64, 128, 256} (IVF-promoted scaled by B/3,162). HNSW ef {32, 64, 128, 256, 512, 1024}.
- **Warm protocol.** One untimed pass of every query at every setting, then timed single-thread passes. Validation (2,000 queries) gets 1 repetition; test (2,000 queries) gets 3. Throughput is measured at 1, 4, 8 and 16 threads on test.
- **Operating point.** For each system, the smallest setting whose **validation** returned R@10 ≥ 0.95 against the official top-10 (and separately R@30 ≥ 0.90). Test latency and recall are reported at that setting. Full curves are reported too.
- **Recall.** Returned IDs are compared with the official SISAP23 gold. Agreement between exact stored-Float16 search and the official gold is reported from the Plain run on the same panel.

## Metrics

- R@10 and R@30 with paired bootstrap CIs.
- Latency p50/p95/p99 per repetition; routing ns (LMI/IVF); candidates per query; result length.
- Throughput (QPS) by thread count.
- Index-only bytes and segment bytes.
- Build seconds by stage, from `QDRANT_LMI_BUILD_EVENTS`: teacher, MLP, posting assignment.
- Peak RSS per phase.

## Decision questions (answered from test at validation-locked points)

1. **Q1.** Does native IVF need fewer candidates and less latency than native LMI at the same recall? This is the native confirmation of Tiers 1 and 2.
2. **Q2.** How far is the best learned-partition index (LMI or IVF) from HNSW in latency at R@10 = 0.95, and what does each cost in index bytes and build time?
3. **Q3.** Is any part of LMI on the recall–latency–memory–build Pareto front?

## Gate

Resources come from the native profile (25718667):
- teacher and MLP seconds equal their 10M cost;
- posting assignment scales ×10.12;
- the HNSW 1M build rate gives an indicative 10M estimate.

The 10M job is submitted only if the predicted walltime, with 50% headroom, fits within 24 h. Otherwise ivf-promoted is dropped first.
