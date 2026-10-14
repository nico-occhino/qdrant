# SISAP23 Plain WAL inspection and controlled replay

The original 10M Plain collection is evidence. Do not load, repair, or modify it.
PBS 17 copies the entire original collection twice into compute-node scratch:
one copy for typed WAL and segment inspection, and one fresh copy for normal
Qdrant collection startup. It verifies both copies byte-for-byte, and hashes
every original collection file before and after the attempt. Reports and logs go
to the persistent diagnostics directory. No completion marker is created.

## Source semantics

The persisted segment contains 10,118,512 internally consistent live vectors
and IDs, with 1,679 dataset IDs missing. The prior successful pre-shutdown
collection-info check counted live IDs in memory, whereas the direct segment
reopen measured persisted vector length. WaitUntil::Visible ensures application
and search visibility, not a synchronous segment flush
(lib/collection/src/shards/shard_trait.rs:23-36).

SerdeWal::new reads first-index and clamps its ack_index against the physical
WAL range (lib/shard/src/wal.rs:72-99). first_index() is the first *logical*
record to replay; first_closed_index() is the first physically retained record;
len(false) and len(true) give their respective counts (wal.rs:173-204,
268-284). WAL files may retain acknowledged records because truncation happens
at whole-segment boundaries (wal.rs:187-204). Thus ack_index=10085 is a logical
boundary, not 10,085 points or a dataset ID. File names are not interpreted
as record counts. The job reports the actual typed record ranges.

One 1,000-point benchmark upsert batch is one collection update
(lib/collection/src/tests/lmi_sisap23_matched.rs:406-433). WAL append returns
an operation sequence number (lib/collection/src/wal_delta.rs:45-72), which
is passed to the update worker (local_shard/shard_ops.rs:123-151). The segment
stores that operation number as a point version (segment/segment_ops.rs:635-653).
The sequence can contain other operations; no arithmetic equivalence with
external IDs is assumed.

The periodic flush worker flushes WAL and segments before acknowledging the
WAL (lib/collection/src/update_workers/flush_workers.rs:62-125). Segment
flush writes mappings, then vectors and payload, then point versions so a
partial flush remains replayable (lib/segment/src/segment/entry.rs:449-453,
529-544). stop_gracefully signals and joins workers, but its stop path does
not start an extra periodic flush (local_shard/shard_ops.rs:587-610;
flush_workers.rs:141-150).

Normal Collection::load reads collection version, configuration and shard
metadata (lib/collection/src/collection/mod.rs:214-297); LocalShard::load
opens WAL and segments, performs consistency checks and possible repairs,
then calls load_from_wal (local_shard/mod.rs:367-581). It needs the entire
collection directory, not just a segment or WAL files. Loading may rewrite
collection version, segment files and WAL; it is permitted only on scratch.
For this fixture prevent_unoptimized must be false, so load_from_wal replays
all logical records synchronously and forces a synchronous segment flush
(local_shard/mod.rs:767-825, 895-982). Its own comments permit idempotent
reapplication of partly applied operations (lines 863-874).

## Two stages

The opt-in test first audits the inspection copy, reads WAL through SerdeWal,
and writes a pre report. It requires each of the 1,679 missing IDs in retained
logical upserts, including its full original Float16 input vector. Unexpected
record types, malformed records, missing IDs or vector mismatches make
safe_for_replay false. Inspection mode stops there.

Replay mode repeats inspection on a new inspection copy, checks the same
coverage gate, and loads the untouched replay copy with Collection::load.
Optimizers are disabled at runtime. It records visible counts, stops the
collection, audits persisted storage and all ID mappings, loads and stops
the copied collection a second time, and audits again. It compares **all
1,679 restored vectors** against the source corpus after cosine normalization
and Float16 rounding. One warmup query is compared with an independent
streaming exact reference over the prepared corpus. Near a score tie,
floating-point reduction order may differ; a mismatch is a failure for
manual review, never automatic acceptance.

The pre JSON contains initial segment counts and mapping audit, physical
and logical WAL ranges, raw first-index metadata, record types, parse errors,
and missing-ID vector coverage. The post JSON contains visible counts after
both loads, persisted audits after both shutdowns, all restored-vector
comparison counts/errors, one-query reference overlap, and accepted.
The PBS status, stdout/stderr, original hashes, copy diff, resource log and
Rust log are distinct persistent files. A panic before post-report publication
is a failed experiment, not recovery.

## Manual worktree and inspection job

Run from perian.metacentrum.cz. Never use the existing development checkout
as the PBS source. Replace the quoted SHA placeholder with the reviewed commit
in the delivery message; the script checks branch, full SHA and cleanliness.

~~~bash
ssh nico@perian.metacentrum.cz
set -euo pipefail
WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
ROOT=$WORKSPACE/results/sisap23-matched-20261013-02
REVIEWED_SHA='PASTE_PUBLISHED_WAL_COMMIT_SHA_HERE'
WT=$WORKSPACE/qdrant-sisap23-wal-replay
cd "$WORKSPACE/qdrant"
git fetch origin thesis/sisap23-matched-wal-replay
test ! -e "$WT"
git worktree add -b thesis/sisap23-matched-wal-replay "$WT" origin/thesis/sisap23-matched-wal-replay
test "$(git -C "$WT" rev-parse HEAD)" = "$REVIEWED_SHA"
test -z "$(git -C "$WT" status --porcelain)"
mkdir -p "$ROOT/diagnostics/wal-replay/pbs"
SUBMIT_DIR=$(mktemp -d "$ROOT/diagnostics/wal-replay/pbs/inspect.XXXXXXXX")
INSPECT_JOB=$(qsub -o "$SUBMIT_DIR/stdout" -e "$SUBMIT_DIR/stderr" \
  -v "SISAP23_WAL_MODE=inspect,SISAP23_BENCH_ROOT=$ROOT,SISAP23_WAL_COMMIT=$REVIEWED_SHA,QDRANT_DIR=$WT" \
  "$WT/tools/lmi/sisap23_matched/pbs/17_replay_plain_wal.pbs")
printf '%s\n' "$INSPECT_JOB" | tee "$SUBMIT_DIR/job-id.txt"
qstat -fx "$INSPECT_JOB"
~~~

Wait for exit status 0. Review both PBS logs, the per-job exit-status file,
the pre JSON and the original-before/after SHA256 files. Require
safe_for_replay=true, 1,679 expected missing IDs, 1,679 matching WAL
vectors, zero uncovered IDs, zero read errors and zero unexpected operations.
If any gate fails, stop; preserve all outputs.

Only after reviewing those facts, submit a fresh replay job:

~~~bash
SUBMIT_DIR=$(mktemp -d "$ROOT/diagnostics/wal-replay/pbs/replay.XXXXXXXX")
REPLAY_JOB=$(qsub -o "$SUBMIT_DIR/stdout" -e "$SUBMIT_DIR/stderr" \
  -v "SISAP23_WAL_MODE=replay,SISAP23_BENCH_ROOT=$ROOT,SISAP23_WAL_COMMIT=$REVIEWED_SHA,QDRANT_DIR=$WT" \
  "$WT/tools/lmi/sisap23_matched/pbs/17_replay_plain_wal.pbs")
printf '%s\n' "$REPLAY_JOB" | tee "$SUBMIT_DIR/job-id.txt"
qstat -fx "$REPLAY_JOB"
~~~

Each attempt has a unique PBS ID and refuses to replace existing reports.
Inspect $ROOT/diagnostics/wal-replay/plain.$JOB.$MODE.pre.json and, for replay,
the corresponding .post.json, .log, .exit-status.txt, .copy-diff.txt,
.original-before.sha256 and .original-after.sha256. Transfer reports from a
workstation with scp using the absolute ROOT and returned PBS job IDs.

## Outcome

accepted=true on the scratch copy requires N unique expected IDs, N available
points and vectors, no invalid/deleted references or missing versions, all
1,679 restored vectors matching the source, the same counts after restart, and
the independent exact reference match. The original collection remains
unmodified; successful scratch recovery is evidence for a separate decision
about whether and how to preserve a recovered generation. It does not publish
results or resume the benchmark. On failure, keep the logs, inspect the failing
gate and the retained WAL, and do not fill missing IDs from the corpus.
