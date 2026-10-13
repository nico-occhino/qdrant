# SISAP 2023 matched Qdrant benchmark

## Scope and audited paths

This benchmark starts from Qdrant 1.19.2 commit 97c20f5f0bd7d65239970285a848706f39458c84. The repository already had an opt-in 10M LMI build/reopen fixture and G2 query sweeps. Those fixtures established the persisted LMI configuration and native search calls, but they were bound to local paths and measured LMI only. The new fixture is lib/collection/src/tests/lmi_sisap23_matched.rs and leaves the G2 files and outputs unchanged.

Plain calls Qdrant PlainVectorIndex::search, which scans visible stored vectors and supplies the exact Float16 storage reference. HNSW calls the persisted VectorIndexEnum::Hnsw directly with SearchParams.hnsw_ef, so graph traversal is explicit and the planner cannot substitute Plain. LMI verifies VectorIndexEnum::Lmi, selects router ranked postings at each nprobe, then calls Qdrant's existing score_candidates RawScorer path. LMI reports selected candidate offsets; HNSW reports ef as graph search effort. These counters have different meanings.

## Fixed conditions

All variants ingest the identical prepared corpus bytes and use 10,120,191 vectors, 768 dimensions, Cosine, Float16 cold vector storage, one shard, one segment, and IDs 1 through N. Measured queries are source rows 20 through 9,999 (9,980 queries); rows 0 through 19 warm each setting.

- Plain: exact native Qdrant full scan.
- LMI: B=3162, sample=250000, hidden=512, epochs=30, spherical KMeans iterations=5, batch sizes 256, seed=42; nprobe 1, 4, 8, 16, 30, 40, 64, 100, 160, 256.
- HNSW-A: m=16, ef_construct=100; query hnsw_ef 16, 32, 64, 128, 256, 512, 1024.
- HNSW-B: m=32, ef_construct=200; the same query sweep.

Plain's output is a separate exact reference for Qdrant's Float16 stored representation. Official SISAP gold is also retained and validated. The validator computes returned Recall@10 against both references.

Every condition runs one full sweep and three paired warm panels over the same deterministic 1,000 query IDs. Setting order rotates between panel repetitions; each setting starts with the same 20 warmup queries. Full timings retain their sequential cache order. Run all builds and evaluations on the same node class and persistent storage, one benchmark process at a time. Avoid unrelated I/O during paired panels; note PBS node, CPU model, filesystem, and cache state.

## Inputs and preflight

Compute-node Python needs Python 3.12, h5py, and numpy. The script checks published SISAP file MD5s, shapes, and dtypes, then exports:

- corpus.f16le: row-major little-endian Float16 bytes (15,544,613,376 bytes).
- queries_warmup.f32le: 20 queries.
- queries_20_9999.f32le: 9,980 queries.
- gold_20_9999_top10.i32le: official top-10 IDs for measured rows.
- panel_rows.u32le: shared deterministic 1,000-query panel.
- manifest.json: source and prepared input hashes and metadata.

Preparation writes only to a new output root and fails if it already exists. It reads HDF5 datasets rather than assuming a fixed header offset. Each Rust build hashes the corpus while ingesting and compares it with the manifest. Keep these prepared files immutable for all four variants.

## Cluster resources and manual sequence

The matched suite requests 96 GB RAM/16 CPUs/200 GB local scratch for up to 96 hours. The standalone build jobs request the same resources, standalone evaluation requests 48 GB RAM/8 CPUs, and preflight requests 32 GB RAM. These are initial resource requests, not measured requirements or a scheduling guarantee. Confirm site limits and the exact protobuf module name before submission. Scripts default to module name protobuf; override PROTOBUF_MODULE for a versioned site module. Persistent workspace needs room for prepared data and four separate collection generations, plus optimizer temporary space; confirm quota before starting. Build jobs copy corpus.f16le into PBS $SCRATCHDIR, while collections/results remain on persistent storage.

The scripts expect the experimental branch checked out in $WORKSPACE/qdrant. Do not run extraction, cargo builds, or dataset scans on a frontend. No job was submitted for this implementation. Use a new run root. First update the existing clone from GitHub:

    git -C /storage/budejovice1/home/nico/lmi-metacentrum/qdrant fetch origin thesis/sisap23-matched-bench
    git -C /storage/budejovice1/home/nico/lmi-metacentrum/qdrant switch --track origin/thesis/sisap23-matched-bench

Then prepare the output root and submit preflight:

    WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
    QDRANT_DIR=$WORKSPACE/qdrant
    SISAP_SOURCE=/storage/brno12-cerit/home/prochazka/datasets/sisap23
    SISAP23_BENCH_ROOT=$WORKSPACE/results/sisap23-matched-20261013-01
    export WORKSPACE QDRANT_DIR SISAP_SOURCE SISAP23_BENCH_ROOT
    qsub -v WORKSPACE,QDRANT_DIR,SISAP_SOURCE,SISAP23_BENCH_ROOT,PROTOBUF_MODULE=protobuf $QDRANT_DIR/tools/lmi/sisap23_matched/pbs/00_preflight.pbs

After preflight succeeds, the recommended matched run uses one long PBS allocation. That keeps all four builds and all paired query panels on the same host and shared persistent filesystem. Check the site's maximum walltime; the suite requests 96 hours. Submit the suite only after preflight finishes successfully:

    qsub -v WORKSPACE,QDRANT_DIR,SISAP23_BENCH_ROOT,PROTOBUF_MODULE=protobuf $QDRANT_DIR/tools/lmi/sisap23_matched/pbs/40_suite.pbs

After the suite completes, preserve and independently validate its outputs:

    qsub -v WORKSPACE,QDRANT_DIR,SISAP23_BENCH_ROOT,PROTOBUF_MODULE=protobuf $QDRANT_DIR/tools/lmi/sisap23_matched/pbs/30_preserve.pbs

The individual 10_build.pbs and 20_evaluate.pbs scripts support isolated or recovered component runs, but separate PBS jobs may land on different hosts. Use 40_suite.pbs for the matched thesis comparison. The suite records hostname, CPU details, filesystem, and date in hardware.txt, then serially builds and evaluates Plain, LMI, HNSW-A, and HNSW-B. It submits no nested jobs.

## Recovery and result layout

Preflight and build outputs are create-once. A build refuses an existing collection and will not delete or replace an index generation. If it stops before build.complete.json, retain that partial directory for diagnosis and choose a new unique benchmark root for a clean build. Build restart is intentionally fail-closed because the fixture cannot prove that an interrupted optimizer target is safe to resume.

Query results use atomic 100-query JSONL chunks. On resume, the evaluator checks saved chunk rows and setting before skipping them, and checks any assembled result before marking it complete. It refuses an interrupted temporary file. All sweeps reopen the same persisted segment path from the build marker and do not start collection optimizers.

- runs/<variant>/build.complete.json, build.log, build.resources.txt
- runs/<variant>/full/<setting>/chunks/ and rows.jsonl
- runs/<variant>/panel/rep-1..3/<setting>/...
- runs/<variant>/evaluate.log and evaluate.resources.txt
- validation.json and results.sha256

/usr/bin/time -v records wall time, user/system CPU, CPU percentage, and peak RSS for build/evaluation processes. Build markers record ingest/build seconds plus collection, segment, and vector-index byte sizes. Raw query rows hold native latency and IDs; validation derives full-sweep p50/p95/p99 and sequential QPS.

## Acceptance criteria

From a compute node, run:

    python tools/lmi/sisap23_matched/validate.py --root "$SISAP23_BENCH_ROOT"
    sha256sum -c "$SISAP23_BENCH_ROOT/results.sha256"

Accept only if all four build markers identify the expected physical index and parameters; all 25 full settings have rows 20..9999 and 10 unique returned IDs per row; independent official-gold recall matches the raw return IDs; Float16-reference overlap is separately computed; all three paired panels per setting are present; and build, evaluation, hardware, and input provenance files are preserved.

## Local validation boundary

Allowed local checks are source formatting, Python/Bash syntax, and bounded Rust compilation. No full corpus ingest, 10M build, 9,980-query sweep, SSH, or PBS job submission was run as part of implementation.
