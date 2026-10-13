# SISAP 2023 matched Qdrant benchmark

## Scope and audited paths

This benchmark starts from Qdrant 1.19.2 commit 97c20f5f0bd7d65239970285a848706f39458c84. The repository already had an opt-in 10M LMI build/reopen fixture and G2 query sweeps. The new fixture is `lib/collection/src/tests/lmi_sisap23_matched.rs`; it leaves the G2 files and outputs unchanged.

Plain calls Qdrant `PlainVectorIndex::search`, which scans visible stored vectors and supplies the exact Float16 storage reference. HNSW calls the persisted `VectorIndexEnum::Hnsw` directly with `SearchParams.hnsw_ef`, so graph traversal is explicit and the planner cannot substitute Plain. LMI verifies `VectorIndexEnum::Lmi`, selects router ranked postings at each nprobe, then calls Qdrant's existing `score_candidates` RawScorer path. LMI reports selected candidate offsets; HNSW reports ef as graph search effort. These counters have different meanings.

## Fixed conditions

All variants ingest the identical prepared corpus bytes and use 10,120,191 vectors, 768 dimensions, Cosine, Float16 cold vector storage, one shard, one segment, and IDs 1 through N. Measured queries are source rows 20 through 9,999 (9,980 queries); rows 0 through 19 warm each setting.

- Plain: exact native Qdrant full scan.
- LMI: B=3162, sample=250000, hidden=512, epochs=30, spherical KMeans iterations=5, batch sizes 256, seed=42; nprobe 1, 4, 8, 16, 30, 40, 64, 100, 160, 256.
- HNSW-A: m=16, ef_construct=100; query hnsw_ef 16, 32, 64, 128, 256, 512, 1024.
- HNSW-B: m=32, ef_construct=200; the same query sweep.

Plain's output is a separate exact reference for Qdrant's Float16 stored representation. Official SISAP gold is also retained and validated. The validator computes returned Recall@10 against both references, indexing the reference arrays with `query_row - 20`; this remains correct for nonconsecutive panel IDs.

### Primary memory-placement policy

Float16 vector storage is `Memory::Cold` for Plain, LMI, and HNSW. HNSW uses Qdrant `Memory::Cached` for its graph. The persisted LMI router and compact postings are opened through Qdrant's mmap-backed `MmapFs`; the operating system caches touched pages, while the small router and frequently used posting pages can remain resident. HNSW Cached gives graph traversal a warmable disk cache as well, rather than forcing graph page faults while LMI routing metadata benefits from the OS cache.

This policy compares both algorithms after warmup with their index structures eligible for page-cache residency, while keeping the much larger vector storage cold for every variant. It does not impose equal resident bytes: HNSW graph size and LMI router/postings size differ, and Cached pages can be evicted under pressure. Report peak RSS, index bytes, node memory, and filesystem/cache context. The three paired warm panels are the primary latency comparison; full sweeps preserve sequential cache order. Do not describe this as a fixed-memory-budget comparison.

Every condition runs one full sweep and three paired warm panels over the same deterministic 1,000 query IDs. Setting order rotates between panel repetitions; each setting starts with the same 20 warmup queries. Run all builds and evaluations on the same node and persistent filesystem, one benchmark process at a time. Avoid unrelated I/O during paired panels; note PBS node, CPU model, filesystem, and cache state.

## Inputs and preflight

Compute-node Python needs Python 3.12, h5py, and numpy. The script checks published SISAP file MD5s, shapes, and dtypes, then exports:

- `corpus.f16le`: row-major little-endian Float16 bytes (15,544,613,376 bytes).
- `queries_warmup.f32le`: 20 queries.
- `queries_20_9999.f32le`: 9,980 queries.
- `gold_20_9999_top10.i32le`: official top-10 IDs for measured rows.
- `panel_rows.u32le`: shared deterministic 1,000-query panel.
- `manifest.json`: source commit, source and prepared input hashes, and metadata.

Preparation writes only to a new output root and fails if it already exists. Each Rust build hashes the corpus while ingesting and compares it with the manifest. Keep these prepared files immutable for all four variants.

## Cluster resources and manual sequence

The matched suite requests 96 GB RAM, 16 CPUs, 200 GB local scratch, and up to 96 hours. Standalone build jobs request 200 GB scratch; standalone evaluation requests 80 GB scratch, 48 GB RAM, and 8 CPUs. Preflight requests 32 GB RAM. Confirm site limits and the exact protobuf module name before submission. These are initial resource requests, not measured requirements or a scheduling guarantee.

Rust 1.98.0, Python 3.12, and the PyTorch 2.5.1 CPU package are checked on compute nodes. Rustup and Cargo use `/storage/brno2/home/nico/.rustup` and `/storage/brno2/home/nico/.cargo`. The scripts set `LIBTORCH_USE_PYTORCH=1`, add the installed Torch library directory to `LD_LIBRARY_PATH`, and leave `LIBTORCH_BYPASS_VERSION_CHECK` unset. They require `protoc` from the loaded protobuf module and verify `g++` by compiling and running a small probe. Cargo's `CARGO_TARGET_DIR`, `TMPDIR`, and the source corpus copy use PBS local scratch; logs, manifests, result chunks, and finished collection generations stay on persistent storage. The suite records tool versions and compiler details below `provenance/`.

The scripts require the exact experimental branch and commit SHA. Fetch the branch and fast-forward an existing local branch, then record its reviewed HEAD once; pass the same SHA to every job. A job stops before compute if its checkout differs. If the local branch is new, create it to track the fetched branch instead.

```bash
WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
QDRANT_DIR=$WORKSPACE/qdrant
SISAP_SOURCE=/storage/brno12-cerit/home/prochazka/datasets/sisap23
SISAP23_BENCH_ROOT=$WORKSPACE/results/sisap23-matched-20261013-01
git -C "$QDRANT_DIR" fetch origin thesis/sisap23-matched-bench
git -C "$QDRANT_DIR" switch thesis/sisap23-matched-bench || git -C "$QDRANT_DIR" switch --track -c thesis/sisap23-matched-bench origin/thesis/sisap23-matched-bench
git -C "$QDRANT_DIR" merge --ff-only origin/thesis/sisap23-matched-bench
SISAP23_REVIEWED_COMMIT=$(git -C "$QDRANT_DIR" rev-parse HEAD)
test "$SISAP23_REVIEWED_COMMIT" = "$(git -C "$QDRANT_DIR" rev-parse origin/thesis/sisap23-matched-bench)"
export WORKSPACE QDRANT_DIR SISAP_SOURCE SISAP23_BENCH_ROOT SISAP23_REVIEWED_COMMIT
qsub -v WORKSPACE,QDRANT_DIR,SISAP_SOURCE,SISAP23_BENCH_ROOT,SISAP23_REVIEWED_COMMIT,PROTOBUF_MODULE=protobuf "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/00_preflight.pbs"
```

After preflight, an optional compilation smoke job checks the `collection` test targets with `lmi-training` using the same setup as the suite. It runs `cargo check`, so it does not link the final test binary or build a 10M index. Review its persistent log under `provenance/` before submitting the suite.

```bash
qsub -v WORKSPACE,QDRANT_DIR,SISAP23_BENCH_ROOT,SISAP23_REVIEWED_COMMIT,PROTOBUF_MODULE=protobuf "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/05_compile_smoke.pbs"
```

Submit one suite only after preflight succeeds. Preserve only after the suite has completed validation. PBS `afterok` ensures the preservation job cannot run after a failed or interrupted suite:

```bash
SUITE_JOB=$(qsub -v WORKSPACE,QDRANT_DIR,SISAP23_BENCH_ROOT,SISAP23_REVIEWED_COMMIT,PROTOBUF_MODULE=protobuf "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/40_suite.pbs")
qsub -W depend=afterok:$SUITE_JOB -v WORKSPACE,QDRANT_DIR,SISAP23_BENCH_ROOT,SISAP23_REVIEWED_COMMIT "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/30_preserve.pbs"
```

No job was submitted for this implementation. Do not run extraction, cargo builds, or dataset scans on a frontend. The individual `10_build.pbs` and `20_evaluate.pbs` scripts support isolated work; separate PBS jobs may land on different hosts, so use `40_suite.pbs` for the matched thesis comparison.

## Recovery and result layout

The suite writes `suite.state` before starting its first variant. If interrupted, resubmit `40_suite.pbs` with the same benchmark root, reviewed commit, prepared-input hashes, host, CPU model, and persistent filesystem. It checks those values before continuing. Completed immutable build markers are reused, completed query chunks are skipped after row and setting checks, and unfinished query conditions resume from their saved chunks. Cargo recompiles into the new job's local scratch as needed.

If a build stopped after creating a partial collection but before `build.complete.json`, the suite preserves that collection and fails closed rather than rebuilding over it. Keep the partial run for diagnosis and use a new root for a clean matched suite. A resume on a different host or filesystem is rejected so paired panels are never silently mixed across machines. If a chunk or assembly write was interrupted, its final file is absent and the evaluator removes only that incomplete temporary file before regenerating it. Completed chunks are immutable and are reused. A partial build directory without its completion marker remains fail-closed.

The suite runs the independent validator after all conditions complete and writes `suite.complete.json` only after validation passes. The preservation job requires that marker and checks the source and input-manifest hashes before creating `results.sha256`. Submit it with PBS `depend=afterok` as shown above.

- `runs/<variant>/build.complete.json` and attempt-specific build logs/resources files.
- `runs/<variant>/full/<setting>/chunks/` and `rows.jsonl`
- `runs/<variant>/panel/rep-1..3/<setting>/...`
- Attempt-specific evaluation logs and resource reports under each variant run.
- `suite.state`, `suite.complete.json`, `validation.json`, and `results.sha256`

`/usr/bin/time -v` records wall time, user/system CPU, CPU percentage, and peak RSS for build/evaluation processes. Build markers record ingest/build seconds plus collection, segment, and vector-index byte sizes. Raw query rows hold native latency and IDs; validation derives full-sweep p50/p95/p99 and sequential QPS.

## Acceptance criteria

From a compute node, run:

```bash
python tools/lmi/sisap23_matched/validate.py --root "$SISAP23_BENCH_ROOT"
sha256sum -c "$SISAP23_BENCH_ROOT/results.sha256"
```

Accept only if all four build markers identify the expected physical index and parameters and reviewed source commit; all 25 full settings have rows 20..9999 and 10 unique returned IDs per row; independent official-gold recall matches the raw return IDs; Float16-reference overlap is separately computed using global query-row alignment; all three paired panels per setting are present; and build, evaluation, hardware, toolchain, and input provenance files are preserved.

## Local validation boundary

Allowed local checks are source formatting, Python/Bash syntax, a synthetic validator test, and bounded Rust compilation. No full corpus ingest, 10M build, 9,980-query sweep, SSH, or PBS job submission is part of implementation.
