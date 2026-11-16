# First complete SISAP23 10M scaling track

Starting HEAD9133a5cb0; executed Rust source3393b4a3c205c5cf3f7fc5f957f9a16e3f737599. The geometric track passed1M,3M and full10,120,191-vector native collection runs. This is the first completed track, not the complete S/D-G/D-N/M comparison.

## Provenance

All official2023 input MD5s passed before extraction: corpus c05e4b1d2b2a0c7663ac9767753e25e1, queries257b9eb3f7f25776e0d33b22451b7b32, full gold45b05e4d60b8a66088b378ae7e0d278f. Corpus HDF5 emb is(10120191,768) Float16; public emb(10000,768) Float32; knns(10000,1000) Int32, one-based IDs. Full-run gold is the official10M file, never the300K gold. [Official documentation](https://sisap-challenges.github.io/2023/datasets/). Intermediate1M/3M oracles were freshly computed with Float64 normalized cosine in8192-row blocks for the first100 queries; unused query rows are zero and never queried. Independent two-block/full-matrix validation matched all100 rows' top1000 scores within1e-14 and top30 sets; equal-score rank permutations caused a strict ID-order assertion to fail before tie-aware diagnosis. That failure is recorded, not hidden.

Sources already existed under/mnt/c/datasets/sisap2023; no network corpus transfer or HPC job occurred. One bounded raw extract supports Rust without loading the corpus into RAM. Raw extracts/manifests, source/binary hashes, hardware and Python/Torch metadata are under `continuation-25/`. Collections are retained for later frozen-topology study.

## Configuration and execution

One local shard, Float16 cosine, disk-backed vectors, b250K,a2,548 buckets,32768 samples, hidden512,30 epochs, seed42, candidate budget2000. This initial continuity/feasibility configuration was not recall-optimized or selected through a validation sweep. Each population runs in a separate optimized process; staged250K acknowledged writes await native publication, followed by100 ordered queries without explicit warmup. Generic optimizers are disabled for DLI ownership; native proxy/COW publication is unchanged. Final partial buffer120,191 is included in active membership. Counts are asserted at every checkpoint.

| Metric | 1M | 3M | 10,120,191 |
|---|---:|---:|---:|
| Native publications | 4 | 12 | 40 |
| Destination vectors | 1.75M | 9M | 31.75M |
| Rewrite amplification | 1.75x | 3.00x | 3.1373x |
| Native job-wall sum s | 264.398 | 991.428 | 3540.055 |
| Ingestion to last checkpoint s | 290.542 | 1065.144 | 3730.779 |
| Recall10/30 | .552/.5317 | .595/.5703 | .402/.391 |
| Query p50/p95 ms | 262.44/662.99 | 1324.23/4742.18 | 3116.07/8778.63 |
| Actual scored vectors /100 queries | 304124 | 890418 | 1758567 |
| Learned/fallback | 100/0 | 100/0 | 100/0 |
| Process peak RSS KiB | 6614704 | 15530824 | 21271168 |
| Auxiliary LMI bytes | 9408512 | 14704258 | 50817028 |

Full run:1 passed,0 failed,4070.80s test execution, with all active native points counted. Native source/destination UUIDs and each output population are saved. Observed sampled allocated-storage maximum23,699,152,896 bytes; sampling began mid-run and is a lower bound on transient peak, not exact peak. Native clean restart/corruption regressions pass at small scale; full10M restart is NOT RUN.

## Interpretation and remaining work

The fixed-sample model explains near-constant small-build training time, but measured large-transition routing/posting and other native work grows. Query phase is dominated by native scoring/vector access, not evidence of multiple-router overhead (see query report). Recall and seconds-scale latency are not competitive evidence and do not establish a maintenance advantage against a matched service.

Size-tiered progressive runs have started separately; static and original-MUNI scaling are not completed in this checkpoint. No ranking of10M systems, recall-matched budget sweep, frozen routing ablation, vector-changing-update workload, replicated significance or break-even QPI is claimed. The scripts retain native authority and visibility protections. See other reports in this directory and raw `continuation-25/`.

Reproduction: `prepare.py SOURCE OUTPUT`, then `run.sh geometric` (or size-tiered/static), using existing validated Rust/PyTorch2.5.1 environment and isolated Cargo target. `analyze.py` generates raw stage/scaling CSVs and SVG scatter plots. Existing output directories are preserved; use a fresh experiment root for repeat runs.
