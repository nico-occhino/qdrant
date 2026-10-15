# SISAP23 10M recovered Plain preservation

## Provenance and acceptance

The frozen run used commit 42a163701a51078ec24983b3a93f96ab7d661489
on adan12. WAL replay from f654cfc96fa7b1819f5a5bfcfa0f5e965cb0ca13
succeeded on alfrid4 (PBS 24671125.pbs-m1.metacentrum.cz). Its scratch
copy was temporary. The original suite.state pins source commit, input
manifest hash and host/CPU/filesystem. Its Plain build marker is absent.
Do not edit suite.state or create a marker for the original run.

The opt-in preservation PBS job repeats recovery from two new scratch copies.
It verifies the original collection SHA-256 inventory before and after the
job. The scratch post-report must have accepted=true. A copy of the complete
Qdrant collection (metadata, WAL, segments) is made to a new persistent
staging directory. Relative file inventory, size and SHA-256 are compared
against the validated scratch generation. The Rust verifier performs two
normal Qdrant collection reopens with optimizers paused, scans every ID
mapping and vector slot, compares the 1,679 restored vectors against the
corpus, and checks an independent Float16 exact top-10 query. The pre-copy
and post-reopen inventories are both recorded, since a normal reopen can
update WAL bookkeeping. The staging directory is published under a unique
PBS job ID and source SHA only after all verification passes. Failed staging
directories are retained separately.

Expected persistent JSON fields: accepted=true,
first_reopen.complete_and_valid=true,
second_reopen.complete_and_valid=true,
first_reopen.total_vectors=10120191,
second_reopen.total_vectors=10120191,
recovered_vectors_compared=1679,
index_files_unchanged=true. generation.json records the original, WAL
and preservation commits, PBS job/host, report hashes, copy inventory hash,
post-reopen inventory and changed files. No old build or suite completion
marker is written.

## Manual Perian commands

Replace REVIEWED_SHA with the published commit from the delivery response.
Run these on Perian. Inspect preservation reports before submitting the suite.
No PBS job is submitted automatically.

    WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
    ORIGINAL=$WORKSPACE/results/sisap23-matched-20261013-02
    git -C "$WORKSPACE/qdrant" fetch origin thesis/sisap23-matched-preservation
    git -C "$WORKSPACE/qdrant" worktree add -b thesis/sisap23-matched-preservation       "$WORKSPACE/qdrant-sisap23-preservation" REVIEWED_SHA
    export QDRANT_DIR="$WORKSPACE/qdrant-sisap23-preservation"
    export SISAP23_REVIEWED_COMMIT=REVIEWED_SHA
    export SISAP23_BENCH_ROOT="$ORIGINAL"
    qsub -v QDRANT_DIR,SISAP23_REVIEWED_COMMIT,SISAP23_BENCH_ROOT       "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/18_preserve_recovered_plain.pbs"

The accepted generation path is printed by the successful preservation job.
Its generation.json and preservation report must be inspected first.

    export SISAP23_ORIGINAL_ROOT="$ORIGINAL"
    export SISAP23_IMPORTED_GENERATION="ACCEPTED_GENERATION_PATH"
    export SISAP23_MATCHED_ROOT="$WORKSPACE/results/sisap23-matched-recovered-NEW_RUN_ID"
    qsub -v QDRANT_DIR,SISAP23_REVIEWED_COMMIT,SISAP23_ORIGINAL_ROOT,SISAP23_IMPORTED_GENERATION,SISAP23_MATCHED_ROOT       "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/41_recovered_matched_suite.pbs"

If automating submission after a known preservation job, use
qsub -W depend=afterok:JOB_ID for the suite, but only after setting the
accepted generation path from its report.

## Matched comparison and recovery

The new suite uses a different run root and copies the original prepared
inputs with hash verification. It records the original manifest separately.
Plain is copied and hash-verified into the new persistent run root. Its
imported.generation.json record identifies the original data commit and
accepted preservation generation. Query execution uses this copy, leaving
the accepted generation untouched. It does not claim a new build.
LMI and HNSW-A/B are built under the reviewed commit with the original fixed
grids. All four query sweeps and paired panels run on the same node and
persistent filesystem. The suite checks the new source commit, manifest hash
and host/CPU/filesystem fingerprint on each resumption. A different host
requires a fresh matched root. Historical Plain build time is not matched
construction time; report it separately. Data correctness, construction
time and search latency are distinct claims.

If preservation fails, retain diagnostics and the .staging generation;
a fresh PBS attempt gets a new generation name. Never overwrite the
original collection or accepted generation. A partial LMI/HNSW build
without a completion marker remains preserved and requires a fresh
matched root. Completed build markers and query chunks may resume only
with the same reviewed commit and exact hardware fingerprint. Only after
all variants and independent validation pass does the new suite write
suite.complete.json.
