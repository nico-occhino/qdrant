# SISAP23 Plain: 1,679-slot diagnostic

This procedure diagnoses the interrupted Plain build. It never publishes a completion marker or opens the original collection with Qdrant. The PBS job copies the collection to node scratch, verifies the copy with `diff -qr`, and opens only the copy. It requires 200 GB scratch for the collection and Cargo build.

## Evidence and source trace

The first job (24633011) logged 10,120,191 ingested points, passed `collection.info().points_count == N`, and reopened one Plain segment before failing during byte accounting. Recovery job 24658172 reported `total_vector_count() = 10,118,512` and stopped before examining IDs. This is a **count discrepancy**, not proof that 1,679 vectors were lost.

`SegmentInfo.num_points` comes from `id_tracker.available_point_count()` (lib/segment/src/segment/read_view/info.rs, lines 63-84), whereas vector count comes from vector storage. Dense `total_vector_count()` returns the underlying vector length (lib/segment/src/vector_storage/dense/appendable_dense_vector_storage.rs:159 and dense/read_only/read_ops.rs:69). The appendable chunked vector length is loaded from `vectors/status.dat` (chunked_vectors/lifecycle.rs:25-60); `Status.len` is the stored-vector count (chunked_vectors/config.rs:18-24). Thus the pre-shutdown logical point count and reopened physical length count different objects and potentially different persistence states.

The 1,000-point application batches use `WaitUntil::Visible` (lib/collection/src/tests/lmi_sisap23_matched.rs:406-433). It waits for search visibility (lib/collection/src/shards/shard_trait.rs:23-36), not explicitly for a durable segment flush. The periodic worker flushes WAL and segments before acknowledging WAL (lib/collection/src/update_workers/flush_workers.rs:62-125). `stop_gracefully` signals and joins update workers (lib/collection/src/shards/local_shard/shard_ops.rs:587-610); the flush loop's stop branch returns without starting another pass (flush_workers.rs:141-150). This does not prove a final flush occurred, nor does it prove data loss. Full collection startup replays retained WAL (lib/collection/src/shards/local_shard/mod.rs:767-850); direct `load_segment` bypasses that replay.

Chunked vector insertion writes chunks, then raises `Status.len` (chunked_vectors/write_ops.rs:54-79). Its flusher flushes chunks before status (chunked_vectors/lifecycle.rs:63-79). ID metadata, deleted flags, and vector chunks are separate persisted objects. The segment flusher explicitly writes ID mappings first, vectors and payload second, and point versions last (lib/segment/src/segment/entry.rs:449-453, 529-544). Its source comments say this order allows WAL recovery after a partial flush: an ID mapping may be durable while its vector is not, but the missing version keeps the operation replayable. A partial persistence state is possible in principle, but the cause here remains unknown. Deletions can leave physical slots, and optimization can change segment layout; the report records both deleted counters and enforces one segment. The production chunk size constant is 32 MiB; 512 KiB is for test builds (vector_storage/common.rs:24-29). The actual chunk geometry is recorded from `config.json`. The difference 1,679 is not a multiple of the 1,000-point batch, and no chunk-boundary conclusion is justified yet.

Opening chunked vectors can create files and opens `status.dat` writable (chunked_vectors/lifecycle.rs:25-50); segment loading may migrate versions. The diagnostic checks version on the **copy** and never loads the original. It does not replay or truncate WAL. Newer upstream git commits touching live reload (`36572ea9e`, `1f076bd07`, `7397af254`, `ef79f22b4`) are not evidence that this persisted-count incident has a known fix.

## Manual submission from Perian

Zenith previously resolved HOME to a nonexistent path and produced PBS start failure -2. Submit from `perian.metacentrum.cz`. Set `DIAG_SHA` to the published diagnostic commit from the delivery message.

```bash
ssh nico@perian.metacentrum.cz
WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
ROOT=$WORKSPACE/results/sisap23-matched-20261013-02
DIAG_SHA=<published-diagnostic-commit>
cd "$WORKSPACE/qdrant"
git fetch origin thesis/sisap23-matched-diagnostics
git switch thesis/sisap23-matched-diagnostics || git switch --track origin/thesis/sisap23-matched-diagnostics
git merge --ff-only origin/thesis/sisap23-matched-diagnostics
test "$(git rev-parse HEAD)" = "$DIAG_SHA"
test -z "$(git status --porcelain)"
test -d "$ROOT/runs/plain/collection"
test ! -e "$ROOT/runs/plain/build.complete.json"
mkdir -p "$ROOT/diagnostics/pbs"
SUBMIT_DIR=$(mktemp -d "$ROOT/diagnostics/pbs/submit.XXXXXXXX")
JOB=$(qsub -o "$SUBMIT_DIR/stdout" -e "$SUBMIT_DIR/stderr"   -v "SISAP23_DIAG_MODE=1,SISAP23_BENCH_ROOT=$ROOT,SISAP23_DIAG_COMMIT=$DIAG_SHA"   tools/lmi/sisap23_matched/pbs/16_diagnose_plain.pbs)
printf '%s\n' "$JOB" | tee "$SUBMIT_DIR/job-id.txt"
qstat -fx "$JOB"
```

After completion, inspect the PBS exit status with `qstat -fx "$JOB"`, both files in `$SUBMIT_DIR`, and `$ROOT/diagnostics/plain.$JOB.{log,json,files.tsv,copy-diff.txt,resources.txt}`. A JSON report exists only if the test reached publication. Keep all failed logs. For retrieval from your workstation, replace `ROOT` and `JOB` with the actual absolute root and job ID:

```bash
scp "nico@perian.metacentrum.cz:$ROOT/diagnostics/plain.$JOB.json" .
scp "nico@perian.metacentrum.cz:$ROOT/diagnostics/plain.$JOB.log" .
```

## Report and decision

The JSON records physical total/available/deleted vectors, `status.dat` length, chunk config and file sizes; tracker total/available/deleted points; iterated mappings; ID and offset ranges; missing/duplicate/invalid IDs; missing vector slots; reverse mapping, version and deletion anomalies; up to 16 examples of each. It classifies a contiguous missing ID tail versus distributed gaps.

A sample of first, middle, physical-boundary and final IDs compares stored vectors against the source corpus after cosine normalization and Float16 rounding, with maximum absolute tolerance 0.002. It reports raw-input error separately. One native Plain query runs only if the inspected mappings are safe; it checks returned IDs, but does not prove full-corpus correctness, measured scan work or an independent top-10 reference.

- **Complete despite count mismatch:** require all expected IDs, valid undeleted vectors, verified values and versions, a full collection load on a second scratch copy, and a source-supported explanation for the count. Change an invariant only with evidence.
- **Incomplete but recoverable:** require exact missing IDs/vectors, WAL coverage, replay on a separate scratch copy, corpus comparison and independent exact-result checks. Repair and marker publication need separate authorization.
- **Incomplete without provable repair:** preserve original artifacts; plan a controlled build in a new root only after investigating the ingestion/persistence cause.

Do not fill IDs silently, relax a gate, or resume the benchmark based on one successful search.
