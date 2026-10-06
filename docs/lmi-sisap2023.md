# SISAP 2023 Float16: accepted 300K and 10.12M results

Status as of 2026-10-05: 300K is the accepted bounded compatibility/retrieval gate; 10.12M is the accepted full official build, persistence/restart and retrieval benchmark. SISAP100M is not downloaded or run. The latest explicitly authorized task evaluated the existing generation without rebuilding, re-ingesting, reconsolidating or tuning it. The old 59-query partial remains preserved and is not a benchmark.

## Dataset and preserved state

The source base belongs at `/mnt/c/datasets/sisap2023`, outside the WSL VHDX.
The 300K base audit found HDF5 dataset `emb`, shape `(300000, 768)`, dtype
`float16`, contiguous and uncompressed. The file named `n=10M` has `emb` shape
`(10120191, 768)`, also contiguous `float16`. This is the complete official
hash-verified challenge subset; do not truncate it to 10,000,000 rows for an
official-gold comparison. The 10M public gold contains IDs through 10,120,191;
1,205 entries among the ten nearest neighbors and 1,149 of the 10,000 queries
reference IDs above 10,000,000. This is why the exact cardinality gate
uses 10,120,191 while keeping the challenge's canonical “10M subset” name and
the requested `B=3162` configuration. The official [SISAP 2023 dataset
description](https://sisap-challenges.github.io/2023/datasets/) lists this
asset and describes the vectors as Float16 with one-based gold IDs. The public
queries use `emb`, shape
`(10000, 768)`, dtype `float32`. Each public gold file has `knns` and `dists`
with shape `(10000, 1000)`; IDs are one-based and distances are float32
`1 - cosine`. Structural audit output is saved at
`work/phase_s3/sisap2023/dataset-audit.json`.

Published MD5 values verified:

| Asset | MD5 | Result |
|---|---|---|
| `laion2B-en-clip768v2-n=300K.h5` | `d238b4b037c32bae41e497f95dffa895` | Pass |
| `public-queries-10k-clip768v2.h5` | `257b9eb3f7f25776e0d33b22451b7b32` | Pass |
| `laion2B-en-public-gold-standard-v2-300K-F64-IEEE754.h5` | `5d635f26630cced971358fd76f37c32e` | Pass |
| `laion2B-en-public-gold-standard-v2-10M-F64-IEEE754.h5` | `45b05e4d60b8a66088b378ae7e0d278f` | Pass |
| `laion2B-en-clip768v2-n=10M.h5` | `c05e4b1d2b2a0c7663ac9767753e25e1` | Pass |

`MD5SUMS.txt` in the dataset directory records all five verified files. The
10M source is downloaded and audited; the accepted full-run artifact is
recorded below. Never fetch the 100M file in this migration.

## Legacy Lance retry and cleanup audit

The explicitly named S.3E retry is no longer running. Its stale PID file records
PID `419`; the prior `full-10m-u16` run records PID `417`; neither PID was live
at this audit. The retry checkpoint records `next_source_row=3898560`, meaning
3,898,560 sequential source rows were acknowledged at the last saved checkpoint.
The exact Qdrant point count cannot now be re-read: its live storage directory
`/home/nicoo/work/lmi-s3e-storage/full-10m-u16-retry1` is absent. The checkpoint,
initial config, ingestion log, and server log remain preserved under
`work/phase_s3/s3e/full-10m-u16-retry1/`; the server log tail is in that file.
The old Lance base and `full-10m-u16` storage directories are also absent. No
additional destructive cleanup was performed during this continuation.

The previous S.3D cleanup removed its exact live storage directory after the
300K compatibility gate passed. Its reports and compact evidence remain under
`work/phase_s3/s3d/`. The previously observed logical WSL space reclaimed by
that removal was approximately 29 GB; this did not compact the Windows
`ext4.vhdx`. Current filesystem readings are recorded in the final handoff.

## Float16 integration boundary

The LMI builder now accepts dense Float16 as well as Float32 storage. Qdrant's
vector reader supplies decoded Float32 rows to the existing spherical KMeans,
training, routing and scoring code; each LMI sample or routing input is gathered
from storage in bounded batches. No corpus-sized Float32 matrix is allocated.
The corpus label cache remains `u16` when the bucket count is at most 65,536.
The builder logs the vector datatype, estimated persistent backing payload,
Float32 compute representation and label-cache representation separately.
Its build plan counts explicit sample/training/posting/workspace allocations,
but does not count mmap/file-backed vector payload as a new anonymous RAM
allocation. The storage payload estimates are 460,800,000 bytes for 300K and
15,544,613,376 bytes for the complete official “10M” HDF5 matrix at 768
dimensions. The latter is the raw payload; the HDF5 file is 797 bytes larger
for metadata. The corresponding u16 labels occupy 20,240,382 bytes. These are
decimal byte counts.

The focused regression `float16_storage_trains_persists_reopens_and_routes`
passed: the built segment reports `Float16`, LMI and Plain scores agree within
the declared `0.002` half-precision tolerance, and a reopened trained index
returns the same results and unchanged postings bytes. This proves the segment
builder/storage seam, not the HTTP ingestion or collection-optimizer lifecycle.

The SISAP importer streams `emb` rows, converts only each bounded block to
Float32 JSON values, and writes one-based IDs `row + 1`. It writes the next-row
checkpoint only after the whole source block is acknowledged; retrying a
partially acknowledged block safely upserts the same IDs. After import, it
retrieves deterministic source samples and checks the normalized vectors
against Qdrant with a `0.002` maximum absolute error bound.

## Bounded 300K gate

The isolated collection is `sisap2023_300k_f16_lmi`, with 768 dimensions,
Cosine distance, Float16 on-disk vectors, HNSW `m=0`, one shard, indexing
threshold zero, one intended segment and one optimizer thread. The scaled
training configuration is:

```json
{"n_buckets":548,"sample_size":32768,"hidden_dim":512,"epochs":30,
 "batch_size":256,"routing_batch_size":256,"kmeans_iterations":5,
 "nprobe":4,"seed":42}
```

Only after exact ingestion and source verification should collection
consolidation finish. The `lmi_sisap_enable_300k.py` admission gate refuses the
build unless the collection is green, has exactly 300,000 points, zero indexed
vectors, exactly one segment, idle optimizer, disabled threshold, the expected
Float16/Cosine LMI config, and no prior postings. Its evidence file is exclusive
and durable before its one PATCH. After the LMI index is published, restart the
server, query again, and verify that the log after restart contains a
`StaticLearned` candidate marker but no training/sampling marker. Then run the
official 10K query/gold evaluator. HTTP query time is reported separately from
in-process build logs; this evaluator does not claim to isolate route and
score component latency.

The release gate completed using the corrected optimized server in fresh paths
`/home/nicoo/work/lmi-sisap2023-storage/300k-f16-release` and
`work/phase_s3/sisap2023/300k-f16-release/`. Exactly 300,000 points were
imported and source-verified (18 samples; maximum absolute error
0.000107259 <= 0.002). The controlled build used 32,768 samples, 548 buckets,
and 30 epochs. KMeans had zero empty teacher buckets (sizes 1-289; about
3.99 s); MLP training plus export took 33.83 s. The build published native
postings. The LMI state/router/postings files were respectively 233,
2,830,573, and 1,204,408 bytes. Peak Qdrant RSS/HWM was 1,932,742,656 bytes;
this is a process high-water mark, not a complete machine-memory measurement.
After restart, `StaticLearned` opened without sampling/training; the captured
query returned identical IDs and scores, persisted file hashes were unchanged,
and the learned-path marker reported 1,548 candidates.

The official query evaluation used 20 warmups and 9,980 measured queries.
Mean Recall@10 was 0.7978858. HTTP latency p50/p95/p99 was
3.0395/3.9409/10.2577 ms. Candidate count p50/p95/p99 was
2,923/4,938/6,047.4; mean candidate fraction was 0.0101374 (1.0137%). These
are end-to-end HTTP measurements, not route/scoring component timings.
The query JSONL preserves each measured query, returned IDs/scores, gold IDs,
recall, candidate diagnostics, and latency. The summary and all import/build,
restart, log, and resource evidence remain under
`work/phase_s3/sisap2023/300k-f16-release/`. Do not extrapolate these results
to 10M.

A separate debug-profile 300K attempt was stopped before model/postings
publication because its full-corpus routing pass was too slow. Its outputs are
preserved under `work/phase_s3/sisap2023/300k-f16/`; they are not the accepted
gate result. The successful S.3D live storage was removed after the 300K gate;
its reports, logs, and compact results remain preserved under `work/phase_s3/`.
The obsolete Lance storage paths listed in the migration request are absent; no
additional deletion was performed during this continuation.

## Reproduction: bounded 300K

Run from the Qdrant repository in WSL. Build with the same feature enabled for
the server and segment tests:

```bash
export LIBTORCH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch
export LD_LIBRARY_PATH="$LIBTORCH/lib"
cargo test -p segment --features lmi-training --test lmi_phase_e
cargo build --release --bin qdrant --features segment/lmi-training --locked
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tests/lmi_sisap_hdf5_audit.py \
  /mnt/c/datasets/sisap2023/laion2B-en-clip768v2-n=300K.h5 \
  /mnt/c/datasets/sisap2023/public-queries-10k-clip768v2.h5 \
  /mnt/c/datasets/sisap2023/laion2B-en-public-gold-standard-v2-300K-F64-IEEE754.h5 \
  --output work/phase_s3/sisap2023/dataset-audit.json
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tests/lmi_sisap_controller.py start \
  --binary target/debug/qdrant --mode 300k --port 17036 \
  --root /home/nicoo/work/lmi-sisap2023-storage/300k-f16 \
  --output work/phase_s3/sisap2023/300k-f16
$PY tests/lmi_sisap_hdf5_stream.py \
  --source /mnt/c/datasets/sisap2023/laion2B-en-clip768v2-n=300K.h5 \
  --url http://127.0.0.1:17036 --collection sisap2023_300k_f16_lmi \
  --checkpoint work/phase_s3/sisap2023/300k-f16/import-checkpoint.json \
  --start 0 --end 300000 --expected-total 300000 \
  --scan-batch 256 --upsert-batch 64 --verify-samples 18
# Wait until Qdrant reports green, one segment, and optimizer_status=ok.
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tests/lmi_sisap_enable_300k.py \
  --url http://127.0.0.1:17036 --collection sisap2023_300k_f16_lmi \
  --segments-dir /home/nicoo/work/lmi-sisap2023-storage/300k-f16/storage/collections \
  --checkpoint work/phase_s3/sisap2023/300k-f16/import-checkpoint.json \
  --evidence work/phase_s3/sisap2023/300k-f16/build-admission.json
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tests/lmi_sisap_monitor.py \
  --pid-file work/phase_s3/sisap2023/300k-f16/server.pid \
  --url http://127.0.0.1:17036 --collection sisap2023_300k_f16_lmi \
  --segments-dir /home/nicoo/work/lmi-sisap2023-storage/300k-f16/storage/collections \
  --output work/phase_s3/sisap2023/300k-f16/resources.jsonl
# Once published, stop and restart with the same controller root/output/port.
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tests/lmi_sisap_restart_verify.py capture \
  --url http://127.0.0.1:17036 --collection sisap2023_300k_f16_lmi \
  --queries /mnt/c/datasets/sisap2023/public-queries-10k-clip768v2.h5 \
  --query-key emb --expected-points 300000 \
  --segments-dir /home/nicoo/work/lmi-sisap2023-storage/300k-f16/storage/collections \
  --server-log work/phase_s3/sisap2023/300k-f16/server.log \
  --checkpoint work/phase_s3/sisap2023/300k-f16/restart-baseline.json
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tests/lmi_sisap_controller.py stop \
  --mode 300k --port 17036 --root /home/nicoo/work/lmi-sisap2023-storage/300k-f16 \
  --output work/phase_s3/sisap2023/300k-f16
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tests/lmi_sisap_controller.py start \
  --binary target/release/qdrant --mode 300k --port 17036 \
  --root /home/nicoo/work/lmi-sisap2023-storage/300k-f16 \
  --output work/phase_s3/sisap2023/300k-f16
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tests/lmi_sisap_restart_verify.py verify \
  --url http://127.0.0.1:17036 --collection sisap2023_300k_f16_lmi \
  --queries /mnt/c/datasets/sisap2023/public-queries-10k-clip768v2.h5 \
  --query-key emb --expected-points 300000 \
  --segments-dir /home/nicoo/work/lmi-sisap2023-storage/300k-f16/storage/collections \
  --server-log work/phase_s3/sisap2023/300k-f16/server.log \
  --checkpoint work/phase_s3/sisap2023/300k-f16/restart-baseline.json \
  --evidence work/phase_s3/sisap2023/300k-f16/restart-verification.json
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python tests/lmi_sisap_hdf5_eval.py \
  --queries /mnt/c/datasets/sisap2023/public-queries-10k-clip768v2.h5 --query-key emb \
  --gold /mnt/c/datasets/sisap2023/laion2B-en-public-gold-standard-v2-300K-F64-IEEE754.h5 \
  --knns-key knns --dists-key dists --url http://127.0.0.1:17036 \
  --collection sisap2023_300k_f16_lmi \
  --server-log work/phase_s3/sisap2023/300k-f16/server.log \
  --output work/phase_s3/sisap2023/300k-f16/queries.jsonl
```

The paths above describe the earlier debug attempt. For exact reproduction of
the accepted release gate, use `target/release/qdrant` and replace each
`300k-f16` output/storage path with `300k-f16-release`; the actual release
configuration, admission evidence, restart evidence, and 9,980-query outputs
are preserved there. The accepted release gate completed successfully.

## Completed SISAP 10.12M build (historical, not repeated)

The corpus contains exactly 10,120,191 vectors of dimension 768, Cosine metric,
Float16 on disk. Fixed settings: B=3162, H=512, sample=250000, epochs=30,
KMeans iterations=5, training/routing batch=256, nprobe=4, seed=42.
All eligible vectors have postings. The u16 corpus-label cache was 20,240,382 bytes;
14 tie-verification rows were checked in one routing pass.

| Stage | Observed seconds |
|---|---:|
| Ingestion | 3756.28 |
| Consolidation | 180.15 |
| Teacher generation | 238.084101 |
| MLP training/export | 1533.601499 |
| Corpus route/count/cache | 517.887848 |
| Posting allocation | 0.022025 |
| Posting fill | 0.075437 |
| Persistence | 0.065238 |
| First native open | 0.099868 |
| Build admission to first native open | 2625.308 |

These component timers do not sum to the admission-to-open interval; remaining
orchestration/preparation time is not isolated. Ingestion verified 32 source
samples, maximum absolute error 0.00014221668243408203 against tolerance 0.002.
Consolidation used the previously recorded 32,000,000 KB merge ceiling.
State/router/postings sizes are 237 / 9,063,429 / 40,506,084 bytes.
The prior stopped storage-tree measurement was 15,980,223,973 bytes.

## Restart and static admission (passed again)

The unchanged release binary was used, without a Rust rebuild. Before evaluation,
the collection was green with optimizer_status=ok, points_count=10120191,
indexed_vectors_count=10120191, and two segments. Detail-4 telemetry established
one nonappendable Float16Mmap `lmi_trained` segment with all 10,120,191 points and
zero deletions, plus one appendable plain Float16ChunkedMmap placeholder with
zero points/vectors. No optimizer work was triggered to remove the placeholder.
Post-sweep collection and detail-4 telemetry were saved separately.

Startup evidence at 2026-10-05T20:39:40.980215Z:
`LMI open: mode=StaticLearned; no training; seconds=0.057938`.
The representative restart query returned the same IDs and exact scores as the
historical checkpoint, with 23,204 candidates and StaticLearned routing.
No sampling, teacher generation, training or posting reconstruction occurred.

SHA-256 values below match both the historical restart evidence and pre/post-sweep
files. The equality establishes unchanged learned-state bytes during evaluation.

| File | SHA-256 before and after |
|---|---|
| target/release/qdrant | `93a4f7eda82fb4a7146ca607585f48b4e0e95ef19175aa6e733b2c95d180baac` |
| lmi_state.json | `61cb61886af0ccbda56f013ac4fe5496ad688b7e5831deb67436f53036e4e459` |
| lmi_router.bin | `35df7ec2613687950ea422b708c7fe2ad18ebc61776724e5a68dc7773c0ac340` |
| lmi_postings.bin | `c82b1cd646d4b349fdc4d5b6a3b5b34154ddcc710cda4af058cc1cc035742bf1` |

The learned files are under storage/storage/collections/sisap2023_10m_f16_lmi/0/
segments/8f67bcc1-6bca-49c8-bef5-4ae8f4565ab1/vector_index relative to
`/home/nicoo/work/lmi-sisap2023-storage/10m-f16`.

## Full official retrieval benchmark (accepted)

The fresh output contains 20 warmups followed by 9,980 measured queries (official
rows 20 through 9999), all validated with ten distinct valid result IDs and finite
scores. Exactly 10,000 StaticLearned query markers cover warmups plus measurements.
Evaluator exit code is zero, stderr is empty, there were no server errors/crashes
or training/optimizer events, and learned hashes remained unchanged.
Completion: 2026-10-05T20:56:41.393839+00:00.

Mean Recall@10: **0.815250501002004**. Recall p5/p50/p95: **0.3 / 0.9 / 1.0**;
minimum/maximum: **0 / 1**.

| Metric | Mean | p50 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Candidate count | 17667.078056 | 17106 | 27325.30 | 33665.57 | 45266 |
| Candidate fraction (%) | 0.1745725753 | 0.1690284304 | 0.2700077499 | 0.3326574568 | 0.4472840483 |
| HTTP end-to-end latency (ms) | 71.461039752 | 13.2984955 | 170.99248105 | 1888.68285909 | 6882.290202 |

Measured-sweep wall time: **715.189087486 seconds** (11 min 55.189 s).
Whole evaluator process wall time: **780.093964308 seconds** (13 min 0.094 s),
including startup, HDF5 loading, warmups and summary writing. The sweep timer
includes per-query diagnostics/output overhead, not just HTTP durations.
Quantiles use NumPy's default linear interpolation, so candidate quantiles may
be nonintegers. Fraction denominator is the actual N=10,120,191.

Recall is conventional top-10 ID overlap against official gold, with no tie
expansion and unchanged one-based IDs. The exact official query/gold MD5s were
verified again. These are sequential HTTP end-to-end measurements, not router,
scorer or neural-inference component timings. There was no controlled cache
flush or full preload beyond the fixed 20 warmups. The run's slow early prefix
is retained in full; the sweep accelerated as it progressed. Cache/residency
and temporal effects may contribute, but this run does not isolate their causes.
Do not replace these accepted statistics with a warmed suffix.

## Comparison with the preserved 300K result

| Metric | 300K (B=548) | 10.12M (B=3162) |
|---|---:|---:|
| Mean Recall@10 | 0.7978857715 | 0.8152505010 |
| Mean candidates | 3041.215631 | 17667.078056 |
| Candidate p50 | 2923 | 17106 |
| Candidate p95 | 4938 | 27325.30 |
| Candidate p99 | 6047.40 | 33665.57 |
| Mean candidate fraction (%) | 1.0137385438 | 0.1745725753 |
| HTTP p50 (ms) | 3.039506 | 13.2984955 |
| HTTP p95 (ms) | 3.9408773 | 170.99248105 |
| HTTP p99 (ms) | 10.25765196 | 1888.68285909 |

Absolute candidate counts and HTTP latency increased while the searched fraction
fell and mean recall rose slightly. N and bucket count both changed; these are
not isolated scaling effects. Cache state was not standardized across runs.
There is no HNSW comparison, parameter sweep or 100M performance claim here.
The 300K reference was recomputed from its preserved raw file without changing it.

## Memory-accounting correction

The historical 30,186,135,552-byte value (30.186 GB / 28.113 GiB) came from
`VmHWM` in `/proc/1053/status`, converted from kB by multiplying by 1024 by
`tests/lmi_sisap_monitor.py`. It is not virtual address space, mapped-file size,
a storage estimate or a process-tree sum. The first peak was recorded during
consolidation at 21:14:19 local, before LMI build admission at 21:17:46;
it is not an isolated neural-training peak. The largest sampled VmRSS was
30,112,256,000 bytes.

No contemporaneous smaps/PSS, RssAnon/RssFile or cgroup snapshot was preserved
at that historical peak. The counter's provenance is established, but its
interpretation as unique physical resident usage cannot be reconstructed.
**The previous 30.2 GB value cannot be interpreted as physical resident memory
and is excluded from the RSS claim.** Historical raw evidence is unchanged.
The historical minimum MemAvailable was 21,533,507,584 bytes; SwapFree ranged
from 8,589,778,944 to 8,589,934,592 bytes. These remain separate system observations.
MemAvailable includes reclaimable cache and does not imply that all resident
file-backed pages are unavailable.

Linux documents [status RSS/HWM counters](https://man7.org/linux/man-pages/man5/proc_pid_status.5.html)
as potentially inaccurate; [proc documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html)
explains per-mapping smaps/PSS and available-memory accounting. Current maps show
two virtual mappings of the same matrix.dat inode; this is evidence of current
aliasing, not proof of the historical anomaly's cause. Mapping extent is not
resident memory.

The new serving-process post-sweep smaps_rollup reports Rss=15,979,816 kB,
Pss=15,976,562 kB, Pss_Anon=577,868 kB, Pss_File=15,398,694 kB and Swap=0.
These are contemporaneous process accounting snapshots, not reconstruction of
the historical build peak or a whole-machine peak. Before/after status, smaps,
smaps_rollup, maps and meminfo are retained; full.memory-audit.json stores precise
byte counts and source provenance.

## Evidence, commands and handoff

All new evidence is under `/home/nicoo/work/qdrant/work/phase_s3/sisap2023/10m-f16`:

- `queries.full.jsonl`: accepted 9,980 per-query observations.
- `queries.full.summary.json`: accepted metrics, hashes and status.
- `queries.full.evaluator-summary.json`: unchanged evaluator-produced summary.
- `full.acceptance.json`: independent row/count/hash/log acceptance checks.
- `full.admission.json`, `full.restart-verification.json`, `full.startup.log`:
  static-index admission and restart evidence.
- `full.execution.json`, `full.runtime.json`, `full.evaluator.py`,
  `full_eval_runner.py`: exact command, versions, environment and implementation snapshots.
- `full.monitor.jsonl`, `full.proc-*`, `full.memory-audit.json`: raw resource evidence.
- `full.reference-300k.json`: reference metrics and source hash.
- `full.collection-after.json`, `full.telemetry-after.json`: final live state.
- `queries.partial-59.jsonl`, `queries.partial-59.metadata.json`,
  `server.partial-59.log`: permanently preserved **partial / non-benchmark evidence**.
  The original `queries.jsonl` is unchanged. No aggregate thesis result is reported
  from those 59 measurements. The new run did not append to that file.

Accepted raw JSONL SHA-256:
`26df51cf59be8a70c08661124ab780c10a4e1138edaa058006a774940fcfdf38`.

The completed command below is recorded for reproducibility, not to overwrite
accepted artifacts. A future authorized repeat must choose new output/log files.
No training, ingestion or consolidation command is needed to reopen this index.

```bash
cd /home/nicoo/work/qdrant
export PY=/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python
export LIBTORCH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch
export LD_LIBRARY_PATH="$LIBTORCH/lib"
export QDRANT_LMI_SAMPLE_BUDGET_BYTES=800000000
export LMI_EXPERIMENTAL_TCH_BUILD_ROUTING=1
unset LMI_EXPERIMENTAL_TWO_PASS_POSTINGS
$PY tests/lmi_sisap_controller.py start --binary target/release/qdrant \
  --mode 10m --port 17038 --root /home/nicoo/work/lmi-sisap2023-storage/10m-f16 \
  --output work/phase_s3/sisap2023/10m-f16
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python /home/nicoo/work/qdrant/tests/lmi_sisap_hdf5_eval.py --queries /mnt/c/datasets/sisap2023/public-queries-10k-clip768v2.h5 --query-key emb --gold /mnt/c/datasets/sisap2023/laion2B-en-public-gold-standard-v2-10M-F64-IEEE754.h5 --knns-key knns --dists-key dists --url http://127.0.0.1:17038 --collection sisap2023_10m_f16_lmi --server-log /home/nicoo/work/qdrant/work/phase_s3/sisap2023/10m-f16/server.log --output /home/nicoo/work/qdrant/work/phase_s3/sisap2023/10m-f16/queries.full.jsonl --expected-points 10120191 --warmup 20 --limit-queries 10000
```

Runtime: WSL2 Ubuntu 26.04, kernel 6.18.33.2-microsoft-standard-WSL2,
16 visible CPUs (server affinity 0-15), Python 3.12.14, NumPy 2.2.6,
h5py 3.14.0 and HDF5 1.14.6. The allowlisted environment was checked in the
running server. Binary SHA-256 matched the completed-build checkpoint.

The only evaluator change in this continuation adds a monotonic measured-loop
wall timer and its summary field (three lines); query semantics and parameters
are unchanged. Its before-copy and exact diff are retained. Python compilation
and the complete execution passed. No Rust code was changed or rebuilt in this
continuation; the four existing tracked Rust modifications remain pre-existing.
No commits, staging or pushes were performed. Final Git status/stat/check are
saved with the result artifacts. The evaluation server was stopped cleanly after
final telemetry capture; all three hashes also matched after shutdown. This handoff's previous text is retained as
full.handoff-before.md.

## 100M boundary and disk

Do not download SISAP100M in this migration. A future planning note is
`100M × 768 × 2 = 153.6 GB` decimal raw Float16 payload and `100M × 2 = 200 MB`
of u16 labels, with initial `B≈10000`, `H=512`, sample 1M only if measured
server memory permits. Hardware and memory headroom must be established first.

Logical deletion of files inside WSL can increase free space shown by `df`, but
does not by itself shrink Windows `ext4.vhdx`. Never edit or compact that VHDX
from this task.

Final filesystem audit after stopping the SISAP10M server: `/` had 852 GB free
of 1007 GB; `/mnt/c` had 512 GB free of 931 GB. The dataset directory occupied
16 GB on `/mnt/c`; the retained stopped SISAP10M storage occupied 15 GB inside
WSL. The exact obsolete S.3D `full-10m` and S.3E `full-10m-u16*` storage paths
were absent; their parent directories measured 495 MB (`lmi-s3d-storage`) and
4 KB (`lmi-s3e-storage`). The old Lance source directory was absent. The
previously measured 29 GB logical WSL space reclaimed by deleting the S.3D
`full-10m` storage is historical; the later 15 GB SISAP storage accounts for
some of the lower current free-space figure. The Windows VHDX size was not
measured or changed. The old successful S.3D live storage has been removed; its scientific artifacts
remain preserved under `work/phase_s3/s3d/`.
