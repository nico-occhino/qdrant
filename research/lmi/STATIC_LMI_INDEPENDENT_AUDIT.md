# Static LMI independent audit (Qdrant 1.20 fork)

**Auditor:** Claude Code, continuation43, 2026-11-17.
**Source:** branch `research/dli-cross-segment-20261112`.
- Audit start: `b6182abad`.
- Fixes: `15727cc2e`, `8c74ac1b0`.

**Scope:** the per-segment static LMI that DLI members also use. The DLI-specific audit is in `research/dli/unified_architecture/INDEPENDENT_CODE_AUDIT.md`.

**Labels:**
- **VERIFIED**: by test or log.
- **CODE**: source observation.
- **DEFECT (fixed)**
- **LIMITATION**: deliberate or unsupported.
- **RISK**: unverified.

## 1. Lifecycle trace

| Stage | Implementation | Status |
|---|---|---|
| Configuration and admission | `LmiConfig::check` (`lmi_index/config.rs`): nprobe ≤ n_buckets ≤ sample_size ≤ 65,536 buckets. `build_trained` accepts Cosine, dense single-vector, Float32/Float16 only (`build.rs:144`). Indexing optimizer requires ≥ n_buckets live vectors. | CODE; tests in `lmi_training_v120.rs` |
| Memory admission | `BuildPlan::estimate`/`check_budget` before and after counting eligible vectors (`build_plan.rs`, env `QDRANT_LMI_SAMPLE_BUDGET_BYTES`). Every large allocation uses `try_reserve_exact` and returns an `OperationError`, not an abort. | CODE |
| Training sample | Seeded reservoir sampling (Algorithm R) over *eligible* offsets: live, not deferred, vector not deleted (`build.rs:48-91`). Uniform over the build-time population, without stratification. | CODE. The model reflects the build-time distribution only; drift is handled solely by rebuild. |
| Teacher | Spherical k-means for Cosine (`training.rs`, `spherical_kmeans.rs`), Lloyd otherwise; seeded. | CODE |
| Router | Linear–ReLU–Linear MLP trained with tch on 1 thread (deterministic for a fixed seed; the continuation36 548-bucket replicate was SHA-identical). Export to a pure-Rust `MlpRouter` (`routing.rs`). | VERIFIED (C36 replicate) |
| Assignment and postings | Every eligible offset is routed to its tie-safe top-1 bucket. `CompactPostings::build_cached_u16` is a two-pass CSR build. `validate_coverage` proves each eligible offset appears exactly once (bitmap) (`build.rs:112-141`). | CODE plus coverage proof at every build |
| Persistence | Router, postings, then `lmi_state.json` last, each via `atomic_save_*` (`atomicwrites`: file fsync, rename, directory fsync). Metadata validated before saving (`state.rs:234`). | CODE |
| Load and validation | `load_state` checks version, config, dimension, distance, datatype, total vectors, preprocessing, MLP shape, posting boundaries, live count and offset range. Any mismatch or corruption makes **segment load fail** (fail closed), matching upstream semantics for a corrupt derived index. | VERIFIED: 9 corruption cases in `corrupt_persisted_state_rejects_segment_load` (renamed; it was misnamed `..._fails_open`) |
| Query routing | Cosine query re-normalized for routing. `top_buckets_with_stop` does a full sort of B logits (ties → lower bucket id), takes the union of the top-`nprobe` postings, then sorts and deduplicates (`routing.rs`). | CODE |
| Candidate scoring | `score_candidates*` (`candidate_scoring.rs`): bounds check, visibility filter (`filter_deferred_and_deleted`, `VisibleOnly`), deleted-vector bitmap, native `BatchFilteredSearcher`. Offsets in ascending order keep equal-score tie handling identical to the Plain scan. | VERIFIED (`deleted_static_candidate_is_filtered_by_current_scorer`) |
| Fallbacks | Filters, non-default `SearchParams`, quantized vectors, or non-dense/non-nearest queries take a **full Plain scan** of the segment. | LIMITATION: correct but O(n); see §3 |
| Updates and deletes | A trained index is immutable: `update_vector` with a vector errors. Qdrant moves updated points to appendable segments. Deletes mark id-tracker/vector bits; postings keep deleted offsets until a rebuild (vacuum/config-mismatch optimizers). | CODE |
| Restart / WAL | The segment reopens from persisted state without retraining. WAL replay goes to appendable segments. | VERIFIED (`persisted_static_search_restarts_without_training`) |

## 2. Defect found and fixed

### S1 — Underfilled learned search (DEFECT, fixed `15727cc2e`)

**Invariant:** an unfiltered segment search returns `min(top, live points)` hits. HNSW guarantees this (`ef ≥ top`), and so does the Plain scan.

**Violation:** `LmiIndex::search` and `ReadOnlyLmiIndex::search` scored only the fixed top-`nprobe` bucket union. Empty buckets count toward `nprobe`, and deleted postings stay in buckets. When the union held fewer than `top` live points, the segment returned fewer hits than it has. The pre-existing test asserted an **empty** result after one deletion in a 4-point segment.

**Reachability:** admission only requires eligible ≥ n_buckets. For example, a 600-point segment with 548 buckets and nprobe 4 averages about 4–5 candidates, so a `limit=10` search underfills. A single-segment collection then returns fewer results than requested. In multi-segment collections it silently drops true neighbours from this segment.

**Fix:** a shared generic `widen_until_filled` doubles the probe count until the result fills.
- Probing every bucket covers all vectors eligible at build. A trained segment is immutable apart from deletions, so the last step is exact search over live points.
- The configured path is unchanged whenever it fills.
- The full-scan alternative was rejected as O(n) per underfilled query.
- DLI routing does not use this path; DLI already had an `insufficient_valid_candidates` exact fallback.

**Regression:** `underfilled_learned_probe_widens_to_fill_top`.
- `top=3` equals exact search, with exactly one widening.
- A deleted offset stays excluded.
- Results are capped at the live count.
- The read-only index widens identically.

## 3. Limitations and risks (not changed)

- **L1 — Fallback cliff.** Any filter, quantization or non-default search parameter scans the whole segment. Correct, but at 10M this is a full scan per segment and per query. Filtered learned search is unsupported.
- **L2 — Fixed-nprobe granularity.** `nprobe` counts buckets, not candidates. With uneven bucket sizes, work varies widely per query: at 10M, accepted nprobe=30 averaged 117,212 candidates. Empty buckets waste probes.
- **L3 — Static model under drift.** The router reflects the build-time sample only. New data lives in other segments until a rebuild. No incremental adaptation.
- **L4 — Deleted postings.** They consume probe budget until vacuum/rebuild. Widening restores result count, not efficiency.
- **R1 — Router inference cost (CODE).** The forward pass is scalar Rust with no SIMD/BLAS. The accepted 10M campaign measured 3.14 ms routing per query (768→512→3,162). This is about 8.5% of the nprobe=30 median and secondary to candidate scoring (§4).
- **R2 — Feature coupling.** Fixed in continuation41 (`58ff5f9d7`): `dli-research` no longer switches static search to the profiled path with per-query logging. Static and DLI timings must still come from one source with identical features.

## 4. Why static LMI was ~27× slower than HNSW (accepted Nov 3 campaign)

**Source:** `Progress_Ladder/STATIC_LMI_RESEARCH_REPORT.md` §6–7.
- Remote raw data: `results/sisap23-matched-final-20261028-01`.
- Analysis: `final-scientific-analysis-20261103/`.
- Workload: 10,120,191 × 768 Float16, cosine; 1,000-query matched panel × 3 repetitions; source `9b12d583b`; host `deimos11` (EPYC 9555), data on NFS; segment-boundary measurement, not REST.

At recall ≥ 0.95:

| Method | Median latency | Recall@10 |
|---|---|---|
| LMI nprobe=30 | 36.879 ms | 0.9600 |
| HNSW-A ef256 | 1.379 ms | 0.9578 |
| HNSW-B ef128 | 1.339 ms | 0.9570 |

**Measured decomposition at nprobe=30:**
- Mean candidates: 117,212 (1.158% of corpus).
- Mean routing time: 3.138 ms.
- Mean candidate-scoring time: 33.412 ms.

Candidate scoring is therefore about 91% of the median. **The gap is explained mainly by the volume of vectors that bucket partitioning must score to reach high recall, not by neural inference** (about 8.5%). HNSW reaches similar recall while visiting a tiny fraction of that volume.

LMI's measured advantage is index-only size: 48.57 MB versus 515.8 / 702.2 MB.

These measurements came from the v1.19 overlay and were not re-run on v1.20. The v1.20 port was validated for result identity (9,980/9,980 identical rankings, `experiments/2026-11-11-v120-parity`), not latency.

The 65K static reference in continuation42 is consistent with this picture: candidate work, not inference, dominates (see the DLI candidate-selection report).

## 5. Test evidence

- `continuation-43/segment-static-suite.log`: `lmi_static_v120` 9, `lmi_shell_v120` 3, `lmi_training_v120` 5, `candidate_scoring_v120` 9 (+1 ignored), `dli_native_topology` 1. All pass with `lmi-research`.
- `segment-static-noresearch.log`: `lmi_static_v120` 9 pass without it.
