# Advanced search: bounded locality and projection controls

Executed against starting revision 6273a8770. Inputs: continuation-28/native-frontier and continuation-28/teacher-control; outputs: `/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-29/`. Existing 10M collection was not opened for mutation.

## Locality

`locality_replay.py CONTINUATION_28 OUTPUT_DIR` replays actual held-out case 2, 32 and 35 bucket selections for all 12 queries and four learned physical members (144 member-query traces). The native path in `dli_native_route.rs` already sorts and deduplicates physical offsets before scoring. Replaying the selected postings in bucket order yields mean member-query absolute offset jump 1,611.33, 2,432 backwards jumps and 6,301,396 logical page transitions. Native-sorted order yields 116.72, zero backwards jumps and 6,185,927 transitions, respectively. The logical 4K pages touched are identical: 6,186,071 member-query pages. The model assumes 1,536 bytes per vector and ignores mmap header/chunk boundaries and OS page cache; it does not measure physical faults, bytes, runtime, or prove a new speedup. It falsifies the premise that offset sorting is absent. It shows that sorting does not change page coverage. Duplicate candidate handling remains native sort/dedup. Prior first/repeated accesses recorded 75,702 versus zero major faults with 18.924 versus .306 seconds, but cannot attribute every fault to vector data.

## Projection feasibility control

`OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 projection_control.py TEACHER_CONTROL OUTPUT_DIR` uses the previously sampled 65,536 vectors and 32 queries (official 800–831). Float32-normalized pool/query vectors, seeded Gaussian random projections, Float16 projected storage, approximate dot-product shortlist and exact Float32 pool oracle are used. This is a feasibility screening experiment, not a native DLI scorer, PCA, an official 10M gold comparison, independent validation/test tuning or an end-to-end latency benchmark. The entire pool is scanned approximately for each query; no learned bucket or native visibility logic participates. Exact scores are precomputed for all points and then indexed for reranking, so the rerank timer is not a real original-vector access cost.

|Projection dimensions|Shortlist|Recall@10|Recall@30|Projected bytes/vector|
|---:|---:|---:|---:|---:|
|64|512|.6469|.5167|128|
|64|8192|.9750|.9448|128|
|128|2048|.9656|.9260|256|
|256|512|.9813|.9458|512|
|256|2048|1.0000|.9948|512|

The measured projected-array footprints are 8, 16 and 32 MiB respectively; 10,120,191 vectors at 256 dimensions would require approximately 5.18 GB for projected vectors alone, before indexing or duplicate generations. Batched 32-query approximate matrix multiplication measured .035/.076/.165 seconds for 64/128/256 dimensions with two BLAS threads; it includes conversion of Float16 projection to Float32, and is not per-query native serving latency. Random projections are a baseline, not a proposed accepted index. No native update, visibility, recovery, persistence or shortlist replenishment was tested. The results justify evaluating existing native quantization on a controlled segment before adding a separate persistent format, not deploying this prototype.

## Remaining evidence and decisions

The prior T=2 matched-work 10M pilot raises bucket coverage at approximately 196K scores, but 12 held-out queries and sequential warm-cache order do not establish a latency advantage. The private 65K model reproduces held-out point assignments at 80.31%; it does not reconstruct original 10M teachers. These findings support testing native quantization and model quality, but do not identify a single dominant recall limitation or justify full 10M retraining. A real compressed scorer must account for external-ID versions, stale compressed generations, concurrent proxy/COW and safe replenishment, and must measure compressed scoring, full-vector fetches, recall and build/storage costs independently.

Run commands:

```bash
python3 research/dli/advanced_search/locality_replay.py /home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-28 /home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-29
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 research/dli/advanced_search/projection_control.py /home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-28/teacher-control /home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-29
```
