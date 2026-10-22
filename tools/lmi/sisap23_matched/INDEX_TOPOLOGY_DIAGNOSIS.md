# SISAP23 index topology: failed job 24851723

Baseline: `thesis/sisap23-publication-repair` at
`4e9b33cd1f93efb6eb75e68c820b3359e2c7ed5e`.
This branch adds benchmark diagnostics and a stricter completion gate. It does not
change Qdrant's production optimizer, LMI training, fixed parameter grids, or
existing generations. No expensive MetaCentrum job has been run on this branch.

## Established from source and reported job output

* `LocalShard::local_shard_info` adds `SegmentInfo::num_indexed_vectors`
  over active segments. `Segment::info` gets that count from each index.
  This is an aggregate count; it carries no single-segment guarantee.
* The old `wait_for_index` returned when the aggregate indexed count equaled
  10,120,191 and optimizer status had no error. It did not require Green status,
  exact point count, or a topology check.
* The job reached that return, then `locate_segment` found two nonempty
  persisted segment directories and rejected them. The source data generation
  had previously passed its full Float16 mapping and fingerprint checks.
  No index generation was accepted or published.
* `stop_gracefully` stops the update/optimizer workers, then asks outstanding
  optimization handles to stop and awaits them. A count-only return could
  therefore stop an in-progress merge or indexing operation. This is a
  possible race, not an established explanation for this job.
* The merge optimizer is a target-based planner. It uses a strict combined
  vector-size limit and only schedules merges that reduce the number of
  segments after accounting for an appendable replacement. A stable pair of
  segments can remain even with `default_segment_number=1`; the exact
  planner state on adan11 is unavailable.
* The failed build's native Rust test ran on a scratch copy. The Python stage
  copies that collection into persistent staging only after the native test
  succeeds. The retained `.staging/build-lmi-24851723.pbs-m1.metacentrum.cz-7921254f10c7`
  therefore contains logs, not a verified persistent copy of that failed
  index. Check contents before drawing any conclusion about scratch survival.

## Configuration semantics

`default_segment_number=1` is a desired segment count, not an invariant.
`max_segment_size=32000000` is **KiB** of vector data per segment, not a point
count (32,768,000,000 bytes). `indexing_threshold=1` is **KiB** of vector
data before indexing is considered, not one point. LMI additionally requires
at least one live vector per bucket. `max_optimization_threads=1` limits
concurrent optimization jobs per shard; it does not serialize all optimizer
phases into a single segment. HNSW-A and HNSW-B use the same aggregate
completion and single-segment verifier, so they have the same control-flow
weakness, though their actual topology has not been measured.

## Repair in this branch

The benchmark now waits for the expected indexed count, exact point count,
Green collection status, and no optimizer error. It rejects counts exceeding
the corpus. At completion or every 30 seconds it logs status, optimizer
status, indexed count, point count, segment count and elapsed time. After
shutdown it still requires exactly one nonempty persisted search segment.
Failure now prints every UUID with live, total and deleted point counts,
index kind and indexed vector count. An opt-in diagnostic test prints the
same topology from a **separate scratch copy**. It does not build an index.
No segment is silently ignored, and the existing full mapping/fingerprint,
persisted index, and restart checks remain in force.

Green means Qdrant's active planner sees no pending optimization at that
instant. It does not prove one segment; the disk topology check remains
mandatory. This patch prevents the known count-only early return and makes
a repeat actionable. It does not claim to force Qdrant into one segment.

## Read-only inspection before any new job

Inspect only the retained reports and logs on MetaCentrum:

~~~bash
ROOT=/storage/budejovice1/home/nico/lmi-metacentrum/results/sisap23-independent-recovery-20261019-01
STAGE="$ROOT/.staging/build-lmi-24851723.pbs-m1.metacentrum.cz-7921254f10c7"
find "$STAGE" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
grep -E 'SISAP23_|sisap23_|optimizer|expected exactly one|panicked' "$STAGE/scratch.log" | tail -n 100
cat "$STAGE/scratch.exit.json"
find "$STAGE" -maxdepth 1 -type d -printf '%f\n'
~~~

If the scheduler preserved the exact scratch collection, inspect a **verified
copy** on a compute node with the opt-in diagnostic test. Supply the copied
collection path only after verifying its source and checksum; the test may let
Qdrant reconcile metadata on the copy. Never point it at the original or
accepted Plain collection.

~~~bash
# Run only inside a separately approved compute allocation and new checkout.
export SISAP23_BENCH_ROOT=/storage/budejovice1/home/nico/lmi-metacentrum/results/sisap23-matched-recovered-20261016-01
export SISAP23_RECOVERY_ROOT=/storage/budejovice1/home/nico/lmi-metacentrum/results/sisap23-independent-recovery-20261019-01
export SISAP23_RECOVERY_COLLECTION="$SCRATCHDIR/verified-failed-index-copy/collection"
test -d "$SISAP23_RECOVERY_COLLECTION"
cargo test -p collection --release --features lmi-training --locked --lib \
  sisap23_recovery_inspect_topology -- --ignored --nocapture --test-threads=1
~~~

If scratch is gone, the saved log cannot establish each live segment's count,
index kind or optimizer state. A later, explicitly approved index-only
diagnostic run from the accepted data generation is then needed. Capture the
new optimizer-state and topology lines. Do not rerun the 10M build just to
test this patch locally.

## Resubmission gate

1. Review the retained failure log and confirm whether the failed scratch
   collection survived. If not, approve a new index-only diagnostic job
   separately. Never rerun the data recovery/ingestion stage by default.
2. Deploy this branch to a distinct MetaCentrum checkout. Pin
   `SISAP23_REVIEWED_COMMIT` to the exact reviewed SHA of this branch. The
   branch and default checkout directory in `recovery_stages.py` and the
   recovery PBS scripts have been updated for that distinct checkout.
3. Set `SISAP23_DATA_PRODUCER_COMMIT` to the producer SHA recorded by the
   existing accepted data generation (likely the earlier baseline; read its
   `generation.json`, do not infer it). Keep its data origin, inventory,
   fingerprint and manifest unchanged.
4. Use a fresh output generation and job ID. Do not reuse or overwrite
   `build-lmi-24851723...`, the accepted recovered data, or Plain outputs.
   Preserve the source commit, host fingerprint and run logs in generation
   metadata. Regenerate or update campaign manifests only after a generation
   passes complete persistence and topology validation.
5. If Green plus full count still yields two live segments, stop. Use their
   sizes, kinds and Qdrant optimizer logs to determine whether the merge
   planner intentionally leaves them or an optimizer stopped/failed. A
   one-segment construction policy or an explicitly multi-segment benchmark
   would require a separate scientific decision and implementation.

Acceptance for LMI and HNSW remains one nonempty segment containing all
10,120,191 uniquely mapped Float16 vectors, the expected persisted index
kind/config, unchanged vector fingerprint, and successful restart
verification. Query latency comparisons must use the same approved hardware
fingerprint and topology policy.
