# Synchronized SISAP10.12M LMI baseline before G2

## Provenance and scope

The implementation baseline is commit `977ef25150f71e3bc0cbedc4ea37b7539e15b2bb` on `thesis/lmi-qdrant-sync`. The annotated tag `thesis-lmi-qdrant-1.19.2-pre-g2-2026-10-07` peels to that commit. No production Rust code was changed for this experiment. Only opt-in collection test fixtures and this report were added. The historical repository `/home/nicoo/work/qdrant` and historical storage `/home/nicoo/work/lmi-sisap2023-storage/10m-f16` were kept read-only. No multi-probe G2 sweep was run.

The initial fresh artifact in `/home/nicoo/work/lmi-sisap2023-storage/10m-f16-qdrant-119-sync` completed ingestion and then its WSL virtual machine shut down while the optimizer was running. It was preserved. A separate clone at `/home/nicoo/work/lmi-sisap2023-storage/10m-f16-qdrant-119-sync-attempt2` was recovered through `Collection::load`, and Qdrant's ordinary background optimizer built the LMI target there. Results and complete logs are in `/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap10m/pre_g2_119_attempt2`; the initial attempt's outputs remain in `pre_g2_119`.

## Dataset and environment

| Input | Path | HDF5 dataset and shape | MD5 |
|---|---|---|---|
| Corpus | `/mnt/c/datasets/sisap2023/laion2B-en-clip768v2-n=10M.h5` | `emb`, Float16, 10,120,191 × 768 | `c05e4b1d2b2a0c7663ac9767753e25e1` |
| Queries | `/mnt/c/datasets/sisap2023/public-queries-10k-clip768v2.h5` | `emb`, Float32, 10,000 × 768 | `257b9eb3f7f25776e0d33b22451b7b32` |
| Gold | `/mnt/c/datasets/sisap2023/laion2B-en-public-gold-standard-v2-10M-F64-IEEE754.h5` | `knns`, int32, 10,000 × 1,000 | `45b05e4d60b8a66088b378ae7e0d278f` |

Source shapes and hashes were checked before the run, as recorded in `pre_g2_119/preflight.json` and `pre_g2_119_attempt2/source.json`. The environment used rustc and cargo 1.98.0, Python 3.12.14, and PyTorch 2.7.1+cpu through the known libtorch environment. Initial disk availability was about 801 GB on `/dev/sdd`; memory was 23 GiB with 8 GiB swap. The command set `QDRANT_LMI_SAMPLE_BUDGET_BYTES=768000000`, `LIBTORCH`, `LD_LIBRARY_PATH`, `LIBTORCH_BYPASS_VERSION_CHECK=1`, `CXX=clang++`, and `CXXFLAGS=-g0`.

## Fixed configuration and lifecycle

The collection contains 10,120,191 vectors of dimension 768, with Cosine distance, Float16 cold vector storage, one shard, and a 32,000,000 KB segment size ceiling. LMI settings were B=3,162 buckets, H=512, sample=250,000, five spherical KMeans iterations, 30 MLP epochs, training and routing batches of 256, nprobe=4, seed=42. Optimization was paused during ingestion. Ingestion through Qdrant's collection/shard update path took 2,110.026 s. The collection reported 10,120,191 points and zero indexed vectors before optimization.

The recovered clone reported the same point and indexed counts before optimizer resumption. The ordinary collection optimizer published one LMI target after 2,762.393 s measured from optimizer activation. Its state is `/home/nicoo/work/lmi-sisap2023-storage/10m-f16-qdrant-119-sync-attempt2/collection/0/segments/168c6d29-9024-404e-aff5-184913f584e7/vector_index/lmi_state.json`.

| Build stage | Synchronized elapsed time |
|---|---:|
| Sample offset selection | 0.072251 s |
| Spherical teacher, including final assignment | 263.869755 s |
| MLP training and export | 1,505.511547 s |
| Teacher plus MLP | 1,769.381366 s |
| Full corpus routing and posting construction | 635.230946 s |
| Exact coverage gate | 0.052348 s |
| Persistence | 0.207170 s |
| Optimizer activation to published indexed target | 2,762.393150 s |

The physical and eligible-live resource plans each covered 10,120,191 vectors. The plan estimated 2,243,253,649 incremental bytes (1,767,759,374 explicitly budgeted plus 475,494,275 backend allowance), with 768,000,000 byte Rust and Torch sample allocations, and was admitted. The corpus routing stage recorded 15 tie fallbacks. The resource monitor recorded a maximum VmHWM of 15.979 GiB, maximum anonymous RSS of 3.490 GiB, and zero process swap across the build and full query run; these figures include resident mapped files and should not be mistaken for incremental builder allocation. The two 31 GB attempt storages were separate, preserved artifacts.

## Persisted v3 state and restart

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `lmi_state.json` | 362 | `09c88623def82515b50ddf4f1c85d82a1188e5a29bafbc4fb00ca729dbf8d81f` |
| `lmi_router.bin` | 8,063,420 | `6789b1aace656c3c475675041433ccdb97bd13190d15edc8ff2b7c0bf8708a94` |
| `lmi_postings.bin` | 40,506,084 | `adcf9f104ef60899e9b922e0bbb155061af3a61532fdcf11705ed55340a7eade` |

The target metadata recorded 10,120,191 indexed live vectors. An independent parser found 3,163 posting boundaries and 10,120,191 posting offsets, spanning exactly 0 through 10,120,190 with zero missing, duplicate, or out-of-range offsets. There were zero empty buckets, median occupancy 2,977, and largest occupancy 19,642. After clean close and collection reopen, indexed vector count remained 10,120,191; the persisted `StaticLearned` target opened and no teacher/MLP rebuild occurred. After all queries, a separate collection reopen with optimization disabled again reported 10,120,191 points and indexed vectors and returned 10 hits from a collection-level search. The post-query run log contains zero training events. The three state hashes stayed identical after both the queries and final reopen.

## Query protocol and result

Rows 0–19 of the official query file were warmup. Rows 20–9,999 were measured (9,980 queries), with official one-based external IDs and conventional top-10 overlap against the official gold. The fixed nprobe was 4. The test uses native `VectorIndexRead::search` on the optimizer-built segment after collection restart. It also counts the actual candidates selected by the persisted router. This native segment path is **not comparable** to historical HTTP end-to-end latency. Full-run native latency includes cold reads; a separately labelled rows 1,000–9,999 suffix describes warmed behavior.

The full release run passed. Its actual returned mean Recall@10 was **0.8159318637**. The p5, median, and p95 were **0.3, 0.9, and 1.0**. Counts for recall bins 0.0 through 1.0 were `[93,154,225,284,380,430,553,701,956,1609,4595]`. The mean candidate count was **17,639.65982**, or **0.1743016492%** of the 10,120,191 corpus. Candidate min/p5/median/p95/p99/max were **2,625 / 9,653 / 17,114 / 27,165 / 33,037 / 49,065**.

Native segment search latency across all 9,980 measured rows had mean/p50/p95/p99/max **56.520 / 10.852 / 166.967 / 1,397.333 / 5,789.837 ms**. The explicit warmed suffix, rows 1,000–9,999 (9,000 rows), had mean/p50/p95/p99/max **11.218 / 10.572 / 15.711 / 32.108 / 209.457 ms**. These suffix percentiles were independently recomputed with NumPy interpolation; the full-run p50/p95/p99 here also use the independently recomputed method, which differs slightly from the test fixture's nearest-rank values in `summary.json`.

Independent validation recomputed every returned top-10 overlap from the official gold, confirmed 9,980 consecutive measured rows and unique one-based IDs, checked candidate and recall summary values, decoded the persisted postings for exact coverage, and confirmed the state/router/postings hashes were unchanged after querying. The machine-readable result is `independent_validation.json`.

## Historical comparison and decision

The frozen historical 10M reference at nprobe=4 has returned mean Recall@10 0.8152505010, mean 17,667.078 candidates, and mean candidate fraction 0.1745726%. Historical recall histogram for bins 0.0 through 1.0 is `[86,151,228,285,388,467,549,669,931,1672,4554]`. Its HTTP end-to-end full-run mean/p50/p95/p99/max were 71.461/13.2985/170.9925/1888.683/6882.29 ms. Its warmed HTTP suffix mean/p50/p95/p99 were 13.512/13.026/17.962/29.245 ms. The synchronized native segment latency must not be treated as a speed comparison. The historical state/router/postings sizes were 237/9,063,429/40,506,084 bytes; v3 router bytes are intentionally different and byte identity is not an acceptance criterion.

Historical import, consolidation, teacher, MLP, route/cache, fill, first native open, and admission-to-open timings were 3,756.28, 180.15, 238.084, 1,533.601, 517.888, 0.075, 0.100, and 2,625.308 s respectively. Stage boundaries, code paths, cache state, and conditions differ, so these figures do not establish a speed regression or improvement.

The synchronized recall exceeds the historical value by **0.00068136** (0.0681 percentage points). Its mean candidate count is **27.418** lower (about 0.155%). These are small, non-material differences consistent with retraining variation; the full recall histogram remains similar. The benchmark does not establish latency equivalence because the synchronized measurement uses native segment search and the historical measurement was HTTP end-to-end. The synchronized v3 baseline is **accepted for retrieval compatibility**: the complete collection and query protocol ran, exact coverage and persistence passed, clean and post-query restarts served queries without retraining, and no material recall or candidate regression appeared. G2 is safe to begin while preserving this exact v3 router and postings.

## G2 handoff, after acceptance only

Use this exact persisted v3 target and the same 10.12M corpus, query file, gold, ID mapping, and 20/9,980 protocol. Next experiment: query-time `nprobe ∈ {1,2,4,8,16,32}`. For every point, measure actual returned Recall@10, candidate count and fraction, native latency, HTTP end-to-end latency if practical, and both full cold and warmed suffix distributions. Reuse the same router and postings; do not retrain or rebuild per probe. Do not select an optimum until the full returned-recall and latency frontier has been measured. No G2 probe other than the baseline value 4 was executed in this task.

The next task should add a query-only opt-in harness that opens the segment above, applies each nprobe as a query-time control, and refuses to write index state. It should confirm the three SHA-256 hashes above before and after every probe. The current test fixture has a fixed `nprobe=4`; there is no existing safe, executable six-probe command to run unchanged. Once that query-only harness exists, run exactly the ordered set `1 2 4 8 16 32` against the same state and save a separate full 9,980-row result for each value. This is a handoff plan, not an executed sweep.

## Reproduction and evidence

The opt-in fixtures are `lib/collection/src/tests/lmi_sisap10m.rs` (fresh ingest) and `lib/collection/src/tests/lmi_sisap10m_resume.rs` (recovery, optimizer, restart, native query evaluation), registered in `lib/collection/src/tests/mod.rs`. The release test command is:

```bash
QDRANT_LMI_SAMPLE_BUDGET_BYTES=768000000 RUST_LOG=info \
LIBTORCH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch \
LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch/lib \
LIBTORCH_BYPASS_VERSION_CHECK=1 CXX=clang++ CXXFLAGS=-g0 \
cargo test -p collection --release --features lmi-training \
  sisap10m_resume_collection_lifecycle_and_queries --locked -- --ignored --nocapture
```

Do not rerun this fixture on the existing attempt paths without first changing them to a new, deliberate artifact path. The independent check can be rerun with `/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tools/lmi/validate_sisap10m_pre_g2.py`. Evidence files include `preflight.json`, `source.json`, `ingest.json`, `interruption_and_recovery.json`, `recovery.json`, `build.json`, `hashes_after_build.json`, `restart.json`, `run.log`, `monitor.jsonl`, `queries.full.jsonl`, `summary.json`, and `independent_validation.json`. A separate post-query health test records `postquery.json` and `postquery_run.log`.
