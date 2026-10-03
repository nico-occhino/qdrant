# Phase S.3B2 — bounded tch corpus routing experiment

## Decision and scope

Implementation commit: `6927a51f6` on `thesis/lmi-integration` (base `fbe8e4ed3`). This phase tests one change to the **build-only** corpus classifier: feed bounded `[K,d]` tensor batches through the MLP that was just trained by tch, and immediately reduce each `[K,B]` logit tensor to one bucket ID per vector. Query routing, compact postings, Qdrant vector ownership, KMeans, training objective, persisted format v2, and reopen/search remain native and unchanged. The backend is deliberately opt-in with `LMI_EXPERIMENTAL_TCH_BUILD_ROUTING=1`; unset or any other value keeps the S.3B1 native builder. `routing_batch_size` supplies K independently of the storage scan and the training mini-batch size.

This is a controlled 1M/100K and synthetic-shape experiment. It is not a 10M or 100M end-to-end build.

## Read-only reference audit

The inspected revisions were:

| Reference | Revision and exact code path | What corpus classification actually does |
|:--|:--|:--|
| [SISAP 2024 Python](https://github.com/Coda-Research-Group/LearnedMetricIndex/blob/paper-sisap24-indexing-challenge/task1.py) | `1163d64f9c0fd3e2cfb333267175ee4b55aa2ff1`, `task1.py`: `_predict`, `_label_data`, `_create_buckets`, `create` | `_label_data` loads one HDF5 chunk, converts it to float32 and passes the entire `[chunk_rows,d]` tensor to `_predict`. The two-layer `Linear(d,512)→ReLU→Linear(512,B)` model executes one batched forward; `topk(1)` returns one label per row. It retains a full `N`-element int32 class array, then rereads/sorts corpus chunks into vector-containing buckets. The CLI default is `chunk_size=1,000,000`; training uses DataLoader mini-batches of 256. |
| [1B and enhancement experiments](https://github.com/Coda-Research-Group/LearnedMetricIndex/tree/enhancing-performance/experiments) | `de96f0f5c5a02fc1edff0fc28ccd1b3ad2d18814`: `task1-1B.py`, `task1-gpu.py`, `task1-compile.py`, `task1-quant.py`, `task1-inmemory.py` | The 1B script changes input loading to fbin, raises the default training sample to 5M, and retains the same chunk-to-batched-model `_predict` pattern and default 1M chunk. GPU variant moves model/input to CUDA; compile variant times compiled versus ordinary inference; quant variant times dynamic int8 inference. The in-memory variant loads the entire HDF5 corpus into one tensor, then takes tensor slices for each batched forward; it is incompatible with the intended bounded Qdrant vector scan. These are experiments, not evidence that a 1M×B logit tensor fits in Qdrant's build budget. |
| [MUNI Rust LMI](https://github.com/Coda-Research-Group/LearnedMetricIndex/blob/rust/final/rust_lmi/src/lib.rs) | `1ad2587dc6d26cac9fa769b97a0c521f8f2bf79e`, `final/rust_lmi/src/lib.rs`: `predict`, `count_bucket_sizes`, `create_buckets_scalable` | Each HDF5 chunk becomes a tensor; both count and fill call `predict(&chunk,1)` once per chunk under `no_grad`. `predict` performs `model.forward(X).softmax(...).topk(...)`; this is tensor-batched across rows, not a row-wise Rust MLP loop. It retains/buckets vectors, unlike Qdrant's offset-only postings. |
| [DynamicLearnedIndexRust](https://github.com/Coda-Research-Group/DynamicLearnedIndexRust/tree/paper-version/dynamic_learned_index/src/model) | `b7c5f8005e350ab5ec48e898f6aba1fffea7a08a`: `tch_model.rs`, `candle_model.rs`, `mix_model.rs`, `mod.rs`, `Cargo.toml` | `tch::predict_many` views a flattened slice as `[rows,d]`, runs one forward and softmax, then materializes all class scores. Candle's `predict_many` internally chunks the input by configured batch size and also materializes per-class probabilities. Mixed mode trains in tch, exports/synchronizes weights to Candle, and predicts through Candle. Optional Candle `mkl` feature exists. Its bounded Candle batching is useful design evidence; full probability materialization is unnecessary for Qdrant top-1 corpus placement. |

The original Python default would require **40 GB of float32 logits alone** for one million rows and B=10,000 (`4×1,000,000×10,000` bytes), before inputs, hidden activations, temporary tensors, or vectors. That is why this prototype uses independent small K. The references demonstrate the batching principle; they do not establish that their default chunks are suitable for this Qdrant host.

## Qdrant mapping and lifecycle

```text
Qdrant VectorStorage + ID tracker
  → eligible dense rows, at most K gathered at once
  → tch input tensor [K,d] on CPU
  → retained trained tch Linear→ReLU→Linear
  → finite [K,B] logits → immediate argmax [K]
  → existing two-pass CompactPostings count/fill, storing offsets only
  → persist native MlpRouter + CompactPostings (v2 files)
  → drop tch training model

open/restart → validate persisted native router/postings → native query top-nprobe
             → Qdrant candidate validity checks and exact scorer
```

The training tensor/model previously died when `training::train` returned the native export. The new `TrainedRouter` owns both the live tch layers/VarStore and the exported `MlpRouter` until the two posting passes finish. The default native branch still drops tch immediately after export. No Python service, FAISS routing, LibTorch model serialization, or query-time tch dependency is introduced. This uses the already-present `lmi-training` feature and its one-CPU thread policy. The persisted router and state payloads are unchanged.

The same eligible-vector iterator drives count and fill. Each flush classifies only its gathered K rows; no N×B tensor or N labels are retained. `check_process_stopped` brackets each batch, while Qdrant's existing posting validator and native opener still validate persisted output. A changed numerical assignment can change a posting without changing the exported native weights, so parity is reported explicitly.

Files changed:

- `lib/segment/src/index/lmi_index/training.rs`: retain build-local tch layers with the validated native export; bounded batched top-1 and finite-output check; test-only native-weight rehydration for exact model comparisons.
- `lib/segment/src/index/lmi_index/build.rs`: explicit experimental backend switch within the existing two-pass builder; preserve native fallback and zero native batch workspace when tch is selected.
- `lib/segment/src/index/lmi_index/evaluation_s3b1.rs`: ignored deterministic synthetic microbenchmark and full-corpus parity/margin test.
- `tests/lmi_phase_s3b2_compare.py`: non-overwriting persisted-posting, router hash and query-result comparison.
- This engineering record.

## Bounded memory model

For one float32 batch, lower-bound tensor payloads are input `4Kd`, hidden `4KH`, logits `4KB`, and int64 top-1 `8K` bytes. The finite-logit check adds approximately `KB` bytes of bool storage. Torch/GEMM may allocate further scratch and retain allocator capacity; these formulas are **not** peak RSS. The Qdrant scan buffer and K offsets add `4Kd+4K` bytes. A model with d=768, H=512, B=10,000 has 5,523,728 float32 parameters including biases, about 22.09 MB per weight copy; build time briefly holds both tch and native copies. Whole-process peak RSS includes Qdrant-owned vector storage, training state, and allocator effects.

| K | input MB | hidden MB | logits MB | finite mask MB | top-1 KB |
|--:|--:|--:|--:|--:|--:|
| 32 | 0.098 | 0.066 | 1.28 | 0.32 | 0.256 |
| 64 | 0.197 | 0.131 | 2.56 | 0.64 | 0.512 |
| 128 | 0.393 | 0.262 | 5.12 | 1.28 | 1.024 |
| 256 | 0.786 | 0.524 | 10.24 | 2.56 | 2.048 |
| 512 | 1.573 | 1.049 | 20.48 | 5.12 | 4.096 |
| 1024 | 3.146 | 2.097 | 40.96 | 10.24 | 8.192 |

MB and KB in this table are decimal. `routing_batch_size` remains default 1. K=2048/4096 was not needed to find a useful operating point; their B=10,000 logits alone would be 81.92/163.84 MB. This phase does not change Qdrant's build admission budget, which remains a separate limit to inspect before larger corpora.

## Results

### Synthetic compute, one CPU thread

The perf-profile benchmark used deterministic f32 weights and rows, actual `BuildRouter`, `BuildBatchRouter`, and tch `Linear→ReLU→Linear` on the same model, one process pinned to CPU 0. Torch, OMP, MKL and Rayon thread controls were set to one. The table is **one run** per point, not a confidence interval. Every tested K/shape had zero assignment mismatches against scalar native; the same result held for a separately pinned four-thread run. Torch setup and weight materialization are outside the timed per-batch windows; Qdrant build timings below include training/export and posting construction.

| Shape d/H/B | Rows | Scalar rows/s | Native batch K=256 rows/s | tch K=256 rows/s | tch µs/vector | tch/scalar |
|:--|--:|--:|--:|--:|--:|--:|
| 128/64/64 | 4,096 | 302,195.94 | 284,858.16 | 1,955,621.34 | 0.51 | 6.47× |
| 768/64/64 | 2,048 | 46,328.77 | 44,540.23 | 660,917.20 | 1.51 | 14.27× |
| 768/512/1,024 | 1,024 | 2,686.95 | 2,571.58 | 47,494.59 | 21.06 | 17.68× |
| 768/512/10,000 | 2,048 | 382.06 | 379.34 | 6,575.13 | 152.09 | 17.21× |
| 768/512/31,622 | 1,024 | 114.03 | 118.20 | 2,437.38 | 410.28 | 21.37× |

For the important B=10,000 shape, bounded K changed tch throughput as follows:

| K | Single-thread rows/s | Single-thread µs/vector | Four-thread rows/s (separate run) | Logits MB |
|--:|--:|--:|--:|--:|
| 32 | 6,120.24 | 163.39 | 26,444.32 | 1.28 |
| 64 | 7,028.62 | 142.28 | 24,634.48 | 2.56 |
| 128 | **7,687.95** | **130.07** | **29,035.12** | 5.12 |
| 256 | 6,575.13 | 152.09 | 23,916.43 | 10.24 |
| 512 | 6,488.00 | 154.13 | 23,102.22 | 20.48 |
| 1024 | 6,074.13 | 164.63 | 23,086.93 | 40.96 |

K=128 is the reasonable **measured** single-thread throughput/memory choice for B=10,000 on this host. The run order and one timing process permit cache and frequency variation, so this is a first operating point, not a universal optimum. Four Torch compute threads were allowed CPU affinity 0–3 and observed as eight total process threads; they are **not** used by the integrated builder because its CPU permit is one. The single-thread process reported two total process threads. Full synthetic-run peak RSS was 544,436 KiB for one thread and 567,696 KiB for four; these are process peaks across *all* shapes/K, not per-K allocations. No GPU was used.

### SIFT1M and LAION end-to-end builds

Both builds used their preserved S.3B1 K=256 configuration, CPU 0, the same perf-profile harness, deterministic seed 42 and unchanged source datasets. Rows were gathered from Qdrant VectorStorage for each of the existing count/fill passes. `result.json` records build and reopen; logged `pass1_seconds`/`pass2_seconds` exclude the rest of builder work. These are one matched run per backend. The baseline files were read, not regenerated or overwritten.

| Dataset/backend | Count s | Fill s | Whole build s | Reopen s | Peak RSS KiB |
|:--|--:|--:|--:|--:|--:|
| SIFT1M, native K=256 | 3.321129 | 3.356609 | 7.681152 | 0.035234 | 1,901,956 |
| SIFT1M, tch K=256 | 0.556220 | 0.574086 | 2.175770 | 0.032581 | 1,903,076 |
| LAION 99,780, native K=256 | 2.037723 | 2.075651 | 4.882435 | 0.012707 | 1,077,956 |
| LAION 99,780, tch K=256 | 0.186445 | 0.187169 | 1.297080 | 0.010945 | 1,077,812 |

SIFT whole-build time in this one comparison is 3.53× lower and count+fill is about 5.91× lower. LAION whole-build time is 3.76× lower and count+fill is about 11.01× lower. In both runs the saved `lmi_state.json` and `lmi_router.bin` SHA-256 hashes match their native counterparts. Reopening without training succeeded, and the existing learned query returned identical IDs and scores. This checks one query result recorded by the harness, not a full query-distribution recall analysis.

A separate full-corpus parity pass loaded the **persisted native weights**, compared scalar-native top-1 with tch batched top-1 after the dataset's Qdrant distance preprocessing, and independently compared the actual posting binaries:

| Corpus | Compared vectors | Assignment/posting mismatches | Rate | Native top-1/top-2 margin on mismatches | Router file | Posting file |
|:--|--:|--:|--:|:--|:--|:--|
| SIFT1M | 1,000,000 | 2 | 0.0002% | 0.0, 0.0 (exact native ties) | byte-identical | differs at offsets 802590 (30→31), 826019 (17→37) |
| LAION | 99,780 | 0 | 0 | n/a | byte-identical | byte-identical |

The two SIFT assignments are numerical tie-boundary differences; the observed native margin is exactly zero, so we should not characterize them as broad partition drift. The experiment does not prove that no query could be affected by those two postings. Native construction remains selectable by leaving the experimental variable unset.

### Verification and exact raw outputs

- `cargo check -p segment --features lmi-training --locked`: **pass**.
- `cargo test -p segment --features lmi-training --locked lmi`: **14 unit tests passed**, 9 benchmark/diagnostic tests intentionally ignored; the name filter also ran two matching integration tests. This is not presented as the full integration suite.
- `cargo test -p segment --features lmi-training --locked --test lmi_candidate_scoring --test lmi_dummy --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e`: **39/39 passed** (2, 1, 10, 12, 14).
- Final optimized ignored benchmarks: **single-thread synthetic 1/1**, separate four-thread synthetic **1/1**, SIFT1M build/reopen **1/1**, LAION build/reopen **1/1**, SIFT1M full-corpus parity **1/1**, LAION full-corpus parity **1/1**. The two posting comparisons completed successfully and saved JSON.
- `cargo check -p collection --tests --features segment/lmi-training --locked`: **pass**.
- `cargo check -p edge --locked`: **pass**.
- `cargo check --bin qdrant --locked`: **pass**.
- `cargo check --bin qdrant --features segment/lmi-training --locked`: **pass**.
- `rustfmt --edition 2024 --check --config skip_children=true` on all three changed Rust files and `git diff --check`: **pass**. Repository-wide `cargo fmt --all --check`: **fails on seven pre-existing formatting diffs in three untouched files** (`on_disk_postings.rs`, `build_plan.rs`, `routing.rs`); stable rustfmt also warns that this repository's import-grouping settings require nightly. No unrelated files were reformatted.


The untouched S.3A/S.3B1 baselines remain under `work/phase_s3/`. Compact machine-readable final observations are committed under `docs/lmi-phase-s3b2-data/` (single/four-thread microbenchmarks, SIFT/LAION parity and posting comparisons, and build-result JSON). This phase's raw logs, compiler metadata, all per-shape/per-K records, full-corpus margins, posting comparisons, new output files and `/usr/bin/time -v` reports are under `work/phase_s3/s3b2/`. In particular: `micro-final-single.json`, `micro-final-four.json`, `sift-final-parity.json`, `laion-final-parity.json`, `sift-final-compare.json`, `laion-final-compare.json`, the two `*-final-tch-k256/` directories, and their logs. Pilot results remain separate and are not used in final tables.

Reproduce on this host with the current branch and existing source corpora:

```bash
cd /home/nicoo/work/qdrant
export LIBTORCH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch
export LD_LIBRARY_PATH="$LIBTORCH/lib"
export TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
cargo test --profile perf -p segment --lib --features lmi-training --locked phase_s3b2_ --no-run
# This is the test executable used for the recorded run; re-resolve if Cargo changes its hash.
export BIN=/home/nicoo/work/qdrant/target/perf/deps/segment-92315b0b0fce32c5
mkdir -p work/phase_s3/s3b2-reproduction
export LMI_S3B2_TORCH_THREADS=1
export LMI_S3B2_MICRO_OUTPUT=$PWD/work/phase_s3/s3b2-reproduction/micro.json
taskset -c 0 "$BIN" phase_s3b2_tch_microbenchmark --ignored --nocapture --test-threads=1

export LMI_EXPERIMENTAL_TCH_BUILD_ROUTING=1
export RUST_LOG=segment::index::lmi_index::build=info
export LMI_S3_DATA=/home/nicoo/work/qdrant/work/phase_s3/sift1m
export LMI_S3_CONFIG=/home/nicoo/work/qdrant/work/phase_s3/s3b1-k256-config.json
export LMI_S3_OUTPUT=$PWD/work/phase_s3/s3b2-reproduction/sift-tch
taskset -c 0 "$BIN" phase_s3_build_benchmark --ignored --nocapture --test-threads=1
# Substitute the LAION corpus/config paths for the LAION run.

python3 tests/lmi_phase_s3b2_compare.py \
  work/phase_s3/s3b1-k256 $PWD/work/phase_s3/s3b2-reproduction/sift-tch \
  $PWD/work/phase_s3/s3b2-reproduction/sift-compare.json

export LMI_S3B2_DATA=/home/nicoo/work/qdrant/work/phase_s3/sift1m
export LMI_S3B2_ROUTER=$PWD/work/phase_s3/s3b2-reproduction/sift-tch/lmi_router.bin
export LMI_S3B2_PARITY_OUTPUT=$PWD/work/phase_s3/s3b2-reproduction/sift-parity.json
taskset -c 0 "$BIN" phase_s3b2_corpus_parity --ignored --nocapture --test-threads=1
```

The harness and comparison tool refuse to overwrite output paths. For the separate multithread diagnostic, set `LMI_S3B2_TORCH_THREADS=4`, `TORCH_NUM_THREADS=4`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and use `taskset -c 0-3` with a new output file. That diagnostic is **not** the integrated Qdrant CPU-permit policy.



## Interpretation and next gate

| Candidate | Implementation/dependency cost | Expected performance and memory | Portability/parity | Fit now |
|:--|:--|:--|:--|:--|
| Existing tch build inference | Small reuse of the already-linked trainer; no new crate or model format | Measured strong single-thread GEMM gain; bounded K×B logits, plus Torch allocator scratch | Requires LibTorch at build time; two SIFT tie-boundary differences in 1M assignments | Best experimental choice |
| Candle build inference | New Candle model/weight-transfer path and optional MKL integration | Batched GEMM is available in the reference, but Qdrant-host throughput and peak memory are unmeasured | Could avoid LibTorch for builds, though MKL still adds native dependencies; numerical parity must be remeasured | Reserve for a concrete portability need |
| Native tiled/SIMD kernel | Highest code, dispatch, maintenance and test cost; potentially no new runtime dependency | Could avoid full K×B logits with streamed argmax, but no measured throughput yet | Could target native reduction order, subject to compiler/FMA checks | Defer unless tch fails a later gate |

The recommendation is **A: retain opt-in tch build batching for further controlled testing**, because the existing training dependency can execute bounded matrix operations without a new kernel and the measured single-thread B=10,000 gain is material. Do not make it the production default yet. Native remains the correctness oracle and safe fallback, especially around exact or near ties.

Candle is a plausible later alternative if eliminating the LibTorch build dependency becomes a priority. The reference Candle backend supports bounded `predict_many` and optional MKL, but moving this project would require weight transfer/format and numerical parity validation, a new dependency surface, and fresh performance tests. A custom tiled kernel currently has the largest engineering cost and no demonstrated need: it should be reconsidered only if tch fails memory, portability, or controlled 10M/100M throughput gates. Neither the synthetic B=10,000 test nor SIFT1M establishes 100M scalability.

The next gate is an end-to-end 10M build with a realistic B and explicit storage-scan, sampling/KMeans, training, count/fill, persistence, peak-RSS and cancellation budgets. The existing sample/clustering budget, two posting passes, vector-storage footprint, and Qdrant CPU permit/thread policy may dominate or block that build. A later 100M run is required before any 100M claim. Query-time latency and recall must be measured separately; the build acceleration does not predict them.
