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

## Recovery of the interrupted 2026-10-13 Plain generation

Job `24633011.pbs-m1.metacentrum.cz` ingested all 10,120,191 vectors but failed while counting
`vector_index/` for Plain. Qdrant's Plain index opens directly from vector storage and has no
required `vector_index/` directory. The reported stack reaches the completion marker after
`stop_gracefully`, segment discovery, reopening, and index-kind verification. The collection
is **not accepted yet**: its point mappings, storage, inputs, and exact-search behavior still
require the independent recovery test on MetaCentrum.

The repair branch is `thesis/sisap23-matched-recovery`. It adds variant-aware index-size
accounting and the opt-in `sisap23_matched_recover_plain` test. Dry-run verifies manifest
hashes for every prepared input, one reopened Plain segment, Float16/Cosine/768 storage,
10,120,191 live vectors and points, every one-based ID and reverse mapping, persisted point
versions, no deleted vectors, and one native Plain top-10 search. Publication repeats all checks,
then creates `build.complete.json` with an atomic create-new link. The marker retains the
original `source_commit` and records `recovery_commit` separately; missing historical timing
is null, never invented. If validation fails, no marker is created. The recovered collection
generation is never rebuilt or rewritten by the test. Keep the recovery logs and original
build resources report alongside the marker.

The existing suite must subsequently run from frozen branch
`thesis/sisap23-matched-bench` at
`42a163701a51078ec24983b3a93f96ab7d661489`. Its manifest and `suite.state`
remain pinned to that generation commit. The suite reuses the recovered Plain marker, builds
the remaining variants with the original code, and validates all results. The suite's exact
hostname/CPU/filesystem fingerprint is unchanged; it will reject a new host before copying
the corpus or starting a build. A resume is scientifically matched only if PBS allocates
`adan12.grid.cesnet.cz` with the same fingerprint. If this is unavailable, keep the recovered
Plain collection for correctness evidence and start a new matched run root; do not edit
`suite.state` or relabel cross-host latency as paired.

### Commands for the owner to run on MetaCentrum

The commands below are instructions; they have **not** been executed here. Run them after
logging in yourself. The first block only reads small metadata on the frontend:

```bash
WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
QDRANT_DIR="$WORKSPACE/qdrant"
SISAP23_BENCH_ROOT="$WORKSPACE/results/sisap23-matched-20261013-02"
ORIGINAL=42a163701a51078ec24983b3a93f96ab7d661489
export WORKSPACE QDRANT_DIR SISAP23_BENCH_ROOT
git -C "$QDRANT_DIR" status --short --branch
git -C "$QDRANT_DIR" rev-parse HEAD
cat "$SISAP23_BENCH_ROOT/suite.state"
test -d "$SISAP23_BENCH_ROOT/runs/plain/collection"
test ! -e "$SISAP23_BENCH_ROOT/runs/plain/build.complete.json"
find "$SISAP23_BENCH_ROOT/runs/plain/collection" -name segment.json -print
test ! -e "$SISAP23_BENCH_ROOT/suite.complete.json"
qstat -x 24633011.pbs-m1.metacentrum.cz
qstat -x 24633013.pbs-m1.metacentrum.cz
```

The following block updates the **source checkout**, not the collection. Run only with a
clean checkout, and confirm the reviewed recovery SHA against the published GitHub branch.
Do not change the manifest, suite state, or result files:

```bash
test -z "$(git -C "$QDRANT_DIR" status --porcelain)"
git -C "$QDRANT_DIR" fetch origin thesis/sisap23-matched-recovery
git -C "$QDRANT_DIR" switch --track -c thesis/sisap23-matched-recovery origin/thesis/sisap23-matched-recovery
SISAP23_RECOVERY_COMMIT=$(git -C "$QDRANT_DIR" rev-parse HEAD)
test "$SISAP23_RECOVERY_COMMIT" = "$(git -C "$QDRANT_DIR" rev-parse origin/thesis/sisap23-matched-recovery)"
export SISAP23_RECOVERY_COMMIT
```

The first PBS job is a dry run. It compiles and checks the collection on a compute node,
hashes the prepared inputs, and runs one exact query, but creates **no** marker:

```bash
qsub -v WORKSPACE,QDRANT_DIR,SISAP23_BENCH_ROOT,SISAP23_RECOVERY_COMMIT,PROTOBUF_MODULE=protobuf   "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/15_recover_plain.pbs"
```

After it finishes successfully, inspect
`$SISAP23_BENCH_ROOT/runs/plain/recovery.dry-run.*.log` and the matching resources file.
Confirm the `sisap23_plain_recovery_verified_dry_run` line and absence of a marker.
The next submission repeats verification and creates the marker; it does **not** build an index:

```bash
test ! -e "$SISAP23_BENCH_ROOT/runs/plain/build.complete.json"
qsub -v WORKSPACE,QDRANT_DIR,SISAP23_BENCH_ROOT,SISAP23_RECOVERY_COMMIT,SISAP23_RECOVERY_PUBLISH=1,PROTOBUF_MODULE=protobuf   "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/15_recover_plain.pbs"
```

Inspect the publish log and marker. Preserve the original five-hour build log to
document ingest time. The following checks read only small result files:

```bash
ls -ltr "$SISAP23_BENCH_ROOT/runs/plain"/recovery.*.log
tail -n 80 "$SISAP23_BENCH_ROOT/runs/plain"/recovery.publish.*.log
python - "$SISAP23_BENCH_ROOT" "$ORIGINAL" "$SISAP23_RECOVERY_COMMIT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
m = json.loads((root / "runs/plain/build.complete.json").read_text())
assert m["source_commit"] == sys.argv[2]
assert m["recovery_commit"] == sys.argv[3]
assert m["point_count"] == 10120191
assert m["index_kind"] == "plain" and m["segment_count"] == 1
assert m["vector_index_bytes"] == 0
assert m["recovered_after_verified_reopen"] is True
print("recovered Plain marker verified")
PY
```

Only after these checks, return the source checkout to the original frozen commit. This
does not touch the benchmark collection:

```bash
test -z "$(git -C "$QDRANT_DIR" status --porcelain)"
git -C "$QDRANT_DIR" switch thesis/sisap23-matched-bench
test "$(git -C "$QDRANT_DIR" rev-parse HEAD)" = "$ORIGINAL"
SISAP23_REVIEWED_COMMIT="$ORIGINAL"
export SISAP23_REVIEWED_COMMIT
```

Submit the existing suite only if `adan12.grid.cesnet.cz` can be allocated again under
the existing PBS resource request. The suite itself checks the recorded fingerprint and
fails closed on a different host. It will reuse Plain and start LMI next. Submit
preservation only with an `afterok` dependency on the new suite job:

```bash
SUITE_JOB=$(qsub -v WORKSPACE,QDRANT_DIR,SISAP23_BENCH_ROOT,SISAP23_REVIEWED_COMMIT,PROTOBUF_MODULE=protobuf   "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/40_suite.pbs")
qsub -W depend=afterok:"$SUITE_JOB"   -v WORKSPACE,QDRANT_DIR,SISAP23_BENCH_ROOT,SISAP23_REVIEWED_COMMIT   "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/30_preserve.pbs"
```

After completion, check `suite.complete.json`, `validation.json`,
`results.sha256`, and `sha256sum -c "$SISAP23_BENCH_ROOT/results.sha256"` on a
compute node. Do not run the 15 GB input hash or Rust build on a frontend.
