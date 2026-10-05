# Phase S.3E — cached corpus bucket labels

Starting branch: `thesis/lmi-integration` at `7110beb66`. The verified S.3D patch was already staged (21 files) at entry; its untracked raw results under `work/phase_s3/` and unrelated `pherical clustering` file were left untouched. No 10M run was started in this phase.

## Question and change

S.3D routed the same 10M corpus vectors twice: 1,374.330446 s for count and 1,375.874314 s for fill. S.3E retains first-pass bounded Torch or native routing, including the S.3B2a tie-safe native verification, and stores each **final** bucket ID as `u16`. After prefix-sum allocation, the builder replays only eligible point offsets in the same order and fills postings from these labels. It does not fetch vectors or invoke either router on this replay. The previous two-route-pass builder remains as a reference and explicit experimental fallback (`LMI_EXPERIMENTAL_TWO_PASS_POSTINGS=1`).

The representable IDs are 0 through 65535, with no sentinel. Thus 65,536 buckets fit exactly. Larger bucket counts select the old builder, although current `LmiConfig` validation itself caps bucket count at 65,536. Conversion to `u16` is checked. The cache is build-lifetime memory only and is never serialized; failure or cancellation drops it automatically.

`eligible()` walks ascending physical point offsets through the tracker visibility/deletion filter and vector-storage deletion check. The builder holds immutable tracker and storage borrows from the initial eligible scan through routing and replay, so the underlying mapping, deletion bits and vector deletion state cannot change during this generation. The replay checks its count against the initial scan and cached labels. Both paths use the same bucket-grouped insertion order. The exact file format and open path are unchanged. These immutable-borrow and deterministic-iterator conditions are the necessary ordering proof; a same-length mutated source could otherwise defeat a count-only check, but cannot be produced through these held borrows.

The build plan now includes `cached_u16_labels = 2 × eligible_count` bytes when the bucket count fits; the initial physical-slot plan uses physical slots as a conservative upper bound. At 10M this is 20,000,000 bytes (19.07 MiB); at 100M it would be 200,000,000 bytes (190.73 MiB). The forced two-pass oracle still receives that conservative plan estimate. The build log reports selected mode, eligible count, cached count/bytes/type, first route/count/cache time, allocation time, fill time and tie-safe fallback rows for the actual number of routing passes.

## Verification

- Focused sparse/deletion fixture: eligible offsets `[2,5,9,10,17,23,29]`, leaving beginning, middle, end and consecutive holes. Bucket counts 1, 4, 64 and 65,536 include empty and one-point buckets and exercise bucket ID 65,535. Old and cached compact postings and bincode bytes match exactly; cached routing is called once per eligible ID.
- Count-change, cancellation and unsupported-bucket error cases pass.
- Isolated full-index deterministic A/B gate, using the same Phase E training fixture and experimental two-pass oracle, proves byte identity of `lmi_state.json`, `lmi_router.bin` and `lmi_postings.bin`. The router and state save code was not changed.
- LMI unit suite: 24 passed, 15 explicitly ignored (large/optional gates, including the opt-in benchmark). Historical integration files: 2 + 1 + 10 + 12 + 14 = 39 passed. The isolated byte-parity gate adds one passing ignored test. Five locked checks passed: segment with training, collection tests with training, edge, default Qdrant server, and training-enabled server. Known stable-rustfmt warnings concern nightly-only import configuration.

The explicit 100K synthetic posting-builder benchmark (one run per mode, debug test profile, 64 buckets, deterministic CPU classifier) measured:

| Mode | First route/count or route/cache | Allocation | Second route or cached fill | Whole builder | Peak test-process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old | 0.013913 s | 0.000480 s | 0.013909 s | 0.028411 s | 148,364 KiB |
| Cached | 0.014763 s | 0.000392 s | 0.001695 s | 0.016916 s | 148,304 KiB |

The isolated test-process RSS values include the Rust test harness and shared libraries, and are not a measured cache allocation delta. This synthetic classifier and in-memory fixture do not predict the 10M neural-routing speedup. The byte-parity gate and historical integration tests provide correctness evidence; a full LAION100K/SIFT1M old/new build benchmark was not completed here. No 10M S.3E time, RSS, or retrieval result is claimed.

## Reproduction of bounded gates

```bash
cd /home/nicoo/work/qdrant
export LIBTORCH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch
export LD_LIBRARY_PATH="$LIBTORCH/lib"
cargo test -p segment --features lmi-training --lib index::lmi_index --locked
cargo test -p segment --features lmi-training --test lmi_phase_e --locked cached_labels_match_two_pass_full_index_bytes -- --ignored --exact --nocapture
LMI_S3E_BENCH_MODE=old /usr/bin/time -f 'peak_rss_kib=%M' cargo test -p segment --features lmi-training --lib cached_labels_100k_synthetic_benchmark --locked -- --ignored --nocapture
LMI_S3E_BENCH_MODE=cached /usr/bin/time -f 'peak_rss_kib=%M' cargo test -p segment --features lmi-training --lib cached_labels_100k_synthetic_benchmark --locked -- --ignored --nocapture
```

## Manual LAION10M A/B rerun — prepared, not executed

Run only after reviewing the new paths and ensuring enough disk and memory. This rebuild uses the same Lance dataset, N=10M, d=768, Cosine, B=3162, H=512, S=250K, five spherical KMeans iterations, 30 epochs, training/routing batches 256, nprobe 4 and seed 42 as S.3D. Use the same machine, 16 visible CPUs and 23.47 GiB WSL memory. No SISAP data is involved. Fresh storage and results keep S.3D intact; copying its already-trained collection would invalidate the build-time A/B comparison. The old S.3D timings remain the reference; do not enable `LMI_EXPERIMENTAL_TWO_PASS_POSTINGS` for this new run.

```bash
cd /home/nicoo/work/qdrant
export PY=/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python
export PYTHONPATH=work/phase_s3/s3d/deps
export LIBTORCH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch
export LD_LIBRARY_PATH="$LIBTORCH/lib"
export QDRANT_LMI_SAMPLE_BUDGET_BYTES=1000000000
export LMI_EXPERIMENTAL_TCH_BUILD_ROUTING=1
unset LMI_EXPERIMENTAL_TWO_PASS_POSTINGS
export S3E_ROOT=/home/nicoo/work/lmi-s3e-storage/full-10m-u16
export S3E_OUT=work/phase_s3/s3e/full-10m-u16
test ! -e "$S3E_ROOT" && test ! -e "$S3E_OUT"
cargo build --release --bin qdrant --features segment/lmi-training --locked
$PY tests/lmi_s3d_full_controller.py start --root "$S3E_ROOT" --output "$S3E_OUT" --port 17034
```

In a second terminal, with the same `PY`, `S3E_ROOT` and `S3E_OUT`, ingest from the beginning. The importer writes its checkpoint for safe resume; do not change collection configuration or start the learned build until the required gates below.

```bash
cd /home/nicoo/work/qdrant
$PY tests/lmi_laion10m_lance_stream.py --source /home/nicoo/datasets/laion10m-lance/base.lance --url http://127.0.0.1:17034 --collection laion10m_lmi --scan-batch 512 --upsert-batch 128 --start 0 --end 10000000 --checkpoint "$S3E_OUT/ingest-checkpoint.json" --server-pid "$(cat "$S3E_OUT/server.pid")" --disk-path /mnt/c
```

As in S.3D, around 5,278,528 ingested points pause optimizer work by applying the max-segment-size configuration **while ingestion continues**. Then, after all 10M points are acknowledged, allow one consolidation, wait until one green 10M-point segment exists, and only then enable the single LMI build. Record every transition and collection status in the new output directory. The transition commands are:

```bash
curl -fsS -X PATCH -H 'Content-Type: application/json' -d '{"optimizers_config":{"max_segment_size":67108864,"max_optimization_threads":0}}' http://127.0.0.1:17034/collections/laion10m_lmi
# AFTER ingestion reaches exactly 10M:
curl -fsS -X PATCH -H 'Content-Type: application/json' -d '{"optimizers_config":{"max_optimization_threads":1}}' http://127.0.0.1:17034/collections/laion10m_lmi
# AFTER the one 10M-point segment is green (inspect collection/segment state):
curl -fsS -X PATCH -H 'Content-Type: application/json' -d '{"optimizers_config":{"indexing_threshold":1,"max_optimization_threads":1}}' http://127.0.0.1:17034/collections/laion10m_lmi
```

In a separate terminal, immediately after enabling the build, run the existing safety watcher and preserve its output. Do not start a second optimizer.

```bash
cd /home/nicoo/work/qdrant
$PY tests/lmi_s3d_watch_server.py --pid "$(cat "$S3E_OUT/server.pid")" --storage "$S3E_ROOT" --output "$S3E_OUT/build-watch.jsonl" --host-disk /mnt/c
```

After publication, locate the new segment dynamically (its UUID will differ), inspect files, stop and restart the server through the controller, verify its `StaticLearned; no training` open marker, then optionally repeat the fixed 199-query HTTP sample. Do not reuse the S.3D segment UUID or output names.

```bash
cd /home/nicoo/work/qdrant
mapfile -t S3E_POSTING_FILES < <(find "$S3E_ROOT/storage/collections/laion10m_lmi/0/segments" -type f -name lmi_postings.bin)
test "${#S3E_POSTING_FILES[@]}" -eq 1
S3E_INDEX=$(dirname "${S3E_POSTING_FILES[0]}")
$PY tests/lmi_s3d_postings_inspect.py --segment-index "$S3E_INDEX" --output "$S3E_OUT/postings-summary.json" --expected-points 10000000 --expected-buckets 3162
$PY tests/lmi_s3d_full_controller.py stop --root "$S3E_ROOT" --output "$S3E_OUT" --port 17034
$PY tests/lmi_s3d_full_controller.py start --root "$S3E_ROOT" --output "$S3E_OUT" --port 17034
grep 'LMI open: mode=StaticLearned; no training' "$S3E_OUT/server.log"
$PY tests/lmi_s3d_eval_http.py --queries /home/nicoo/datasets/laion10m-lance/queries.lance --url http://127.0.0.1:17034 --server-log "$S3E_OUT/server.log" --output "$S3E_OUT/http-nprobe4-sampled-raw.jsonl" --start 35 --end 4992 --stride 25 --warmup 20
```

Compare the new build log's route/count/cache, allocation, cached fill and persistence times to the S.3D values above, accounting for source-copy/optimizer wall time separately. Compare actual peak RSS, MemAvailable and swap under the same watcher settings. The expected second-pass saving is **not measured at 10M**. Segment layout, OS cache, concurrent work, Torch/CPU settings, build binaries, source collection state or corpus ingestion differences could invalidate a direct timing attribution; record them explicitly.
