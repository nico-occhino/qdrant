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

## Amendment A (2026-11-18 23:30, before any Tier 3 submission)

**Profiling evidence.** Job 25718749 (zenon51, 1M source, commit f2f6f27d8) reached COMPLETE.

| Index | Build at 1M | Index size |
|---|---|---|
| IVF-8192 (S = 524,288, 20 iterations) | k-means 14,492 s single-threaded, plus postings 379 s | 29.3 MB |
| LMI-3162 | k-means 733 s, MLP 2,404 s, postings 120 s | 12.1 MB |
| HNSW m16/efc100 | 168 s on 16 threads | 40.3 MB |

Native search on 2,000 validation queries (returned recall against the 1M gold):

| Index | Setting | R@10 | p50 | Routing p50 |
|---|---|---|---|---|
| LMI-3162 | nprobe 64 | 0.964 | 14.96 ms | 5.08 ms |
| IVF-8192 | nprobe 32 | 0.949 | 16.68 ms | 14.30 ms |
| HNSW | ef 128 | 0.966 | 0.98 ms | — |

**A1. Router constant-factor fix (implementation, not algorithm).** Native routing ran at about 0.44 GMAC/s for both the MLP and the centroid router. The causes:
- every query re-validated every parameter;
- the matrix-vector product was a scalar loop;
- all B logits were fully sorted.

Commit `research/static-lmi-ivf-router-fix` changes three things, without changing the ranking rule (descending logit, smaller ID on ties):
- a shape-only check per query (the full finite check still runs at load, and non-finite parameters still fail through the per-output check);
- Qdrant's SIMD dot product per row;
- a partial top-nprobe selection.

Tests:
- `top_ids` equals the full sort, including ties and signed zeros;
- the optimized routing equals the scalar reference except for near-ties within float summation order;
- the error behaviour is unchanged.

The router fix applies to both lmi-native and ivf-3162. Tier 3 therefore measures the index design with a reasonable router implementation; the profile above is the pre-fix reference.

**A2. Systems.** Tier 3A runs plain, hnsw (ef up to 1,024), lmi-native and ivf-3162.

IVF-8192 is deferred to Tier 3B. Its native single-threaded teacher alone takes 4.0 h, so the 10M build would idle 15 of 16 cores for about 5 h. Tier 3B runs after the assignment step of spherical k-means is parallelised, with hnsw and ivf-3162 rebuilt on the same node for matched comparison. IVF-8192's offline advantage (Tier 2) and its native build cost are reported as findings.

**A3. Plain** uses the first 256 queries of each split (`{val,test}.q256.f32`): 1 repetition, with throughput at 1 and 16 threads.

**A4. Resources.** 16 CPUs, 96 GB, 200 GB scratch. Walltime 10 h, against a predicted ≈ 4.5 h:
- source ingestion ≈ 2 min;
- HNSW 10M build ≈ 20–40 min;
- lmi-native ≈ 70 min (733 s k-means, 2,404 s MLP, about 1,200 s postings);
- ivf-3162 ≈ 35 min;
- searches ≈ 25 min per learned index.

Each configuration's results are copied to persistent storage as soon as it finishes, so a walltime kill loses only the configuration in progress.

## Amendment B (2026-11-19, after Tier 3A results, before any Tier 3B submission)

**Context.** Tier 3A (job 25738234) is complete and reported in `LMI_VS_IVF_ROUTING_ABLATION.md`. Its systems, operating points and answers are final and are not re-run here. Tier 3B was defined in A2. Its precondition, parallelising the spherical k-means assignment step, is met by commit `3e49dabcd`:
- Per-row assignment, in every iteration and in the final pass, runs on a rayon pool sized by the build CPU permit.
- Labels and centers are bit-identical for any thread count. A unit test checks 1, 2, 3 and 8 threads.
- A bounded benchmark (50K × 768, B 2,048) gives 6.0 s → 0.69 s per assignment pass at 16 threads.
- MLP training and posting assignment are unchanged and stay single-threaded.

**B1. Systems.** All three are built in one job on one node:
- **hnsw:** m16/efc100 with 16 threads, unchanged from 3A.
- **ivf-3162:** the 3A configuration, now with a 16-CPU permit. Only its k-means wall time can change.
- **ivf-8192:** the Tier 2 F2 configuration, built with `LMI_RESEARCH_ROUTER=centroid` and a 16-CPU permit: B 8,192, S 524,288, 20 iterations.

Plain and lmi-native are not rebuilt. Their 3A results stand, and the Plain–gold agreement comes from 3A.

**B2. Settings grids.** These are finer than 3A near the R@10 = 0.95 operating points, because the coarse 3A grid made the locked comparison lumpy.
- hnsw: ef {64, 128, 160, 192, 256, 512}.
- ivf-3162: nprobe {8, 12, 16, 20, 24, 30, 64}.
- ivf-8192: nprobe {12, 16, 20, 24, 32, 40, 48, 64, 128}. Tier 2 predicts R@10 = 0.95 at about 24 buckets (29.3K scored ÷ 1,235 per bucket).

The grids are in `tier3-configs.tsv`.

**B3. Protocol and analysis.** These are unchanged from the preregistration and 3A:
- warm protocol, 1 validation repetition, and 3 test repetitions at 1, 4, 8 and 16 threads;
- validation-locked operating points at R@10 ≥ 0.95 and at R@30 ≥ 0.90;
- `analyze_tier3.py`, extended only to accept the 3B system list.

**Decision question Q4.** Does the converged, finer IVF-8192 beat IVF-3162 natively at the locked R@10 = 0.95 point in p50 latency and in candidates? What do the extra routing cost and build time amount to? Q2 is re-answered with the best IVF of 3B against HNSW on the same node.

**B4. Resources.** 16 CPUs, 96 GB and 200 GB scratch. Walltime is 8 h, against a predicted ≈ 3.7 h:
- **Staging and binary:** ≈ 10 min.
- **hnsw:** build ≈ 10 min; searches ≈ 10 min.
- **ivf-3162:** build ≈ 10 min (postings 567 s); searches ≈ 55 min.
- **ivf-8192:** build ≈ 55 min (teacher 14,492 s ÷ about 9, plus postings ≈ 567 s × 8,192/3,162); searches ≈ 50–70 min.

As in 3A, each system's results are persisted as soon as it finishes.
