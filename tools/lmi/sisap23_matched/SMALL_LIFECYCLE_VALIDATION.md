# SISAP23 small topology validation (proposed; no cluster job submitted)

Base: `6ceb94e19dc42a02e6775c799173a255b7d01159` on a new isolated
`thesis/sisap23-small-lifecycle` branch. The accepted 10,120,191-vector
generation, Plain measurements, failed-job diagnostics, and fixed benchmark
parameter grids are never opened or changed by these tests. The new LMI
configuration (2 buckets, 32 samples, one epoch) is **diagnostic only**;
it is not a benchmark candidate or performance measurement.

## Matrix

| Stage | Exact test | Distinguishes |
| --- | --- | --- |
| A | `collection_manager::optimizers::merge_optimizer::tests::two_live_segments_can_remain_at_one_segment_target` | One-segment target versus planner behavior with exactly two live segments. |
| A | `tests::lmi_sisap23_matched::sisap23_optimizer_error_fails_even_with_a_complete_count` | Aggregate count versus optimizer error and Green status. |
| A | `tests::lmi_sisap23_matched::sisap23_one_live_segment_and_empty_segment_is_accepted_after_reload` | Physical empty segment versus a second live search segment. |
| A | `tests::lmi_sisap23_matched::sisap23_two_live_segments_remain_rejected_after_reload` | Two live search segments versus the strict one-segment invariant. |
| A | `tests::lmi_sisap23_matched::memory_recovery::sisap23_reuse_vectors_without_reingestion` | Small Float16 HNSW index-only rebuild and normal restart without a second upsert. |
| B | `tests::lmi_sisap23_matched::small_lifecycle::sisap23_small_float16_holder_retirement` | Controlled two-active-segment state versus one active segment plus a retired source directory pending Qdrant's post-flush cleanup. This fixture calls the native holder's deferred-drop protocol directly; it does not claim to reproduce a full optimizer schedule. |
| B | `tests::lmi_sisap23_matched::small_lifecycle::sisap23_small_float16_hnsw_lifecycle` | First aggregate HNSW count versus optimizer quiescence, flush, shutdown, and restart, using the same 64 disposable Float16 vectors throughout. |
| B | `tests::lmi_sisap23_matched::small_lifecycle::sisap23_small_float16_lmi_lifecycle` | The same native lifecycle for a tiny learned index; distinguishes LMI-specific temporary/retired directory behavior from HNSW behavior. |

All Stage A/B test invocations use a single, fully qualified `--exact`
selector and require the harness to report exactly one passing test. Stage B
tests are ignored by default and run only with `--ignored`. The PBS jobs
compile the release collection test binary with `lmi-training`; they
never run the whole test suite.

## Observable phases and acceptance

The collection lifecycle records holder entries, proxy count, physical UUID
directories, per-active-original live/deleted counts, index kind and indexed
count, and collection/optimizer status at plain baseline, first aggregate
indexed count, three consecutive Green samples, synchronous flush, and clean
restart. It separately records physical directories after graceful shutdown.
These cross-API readings are observational snapshots, not an atomic view of
a running optimizer.

The test rejects an optimizer error or a 120-second small-index deadline. It
checks every external ID 1–64 exactly once, the reverse mappings, live
Float16 vectors, cosine configuration, and an order-independent ID/vector
fingerprint before and after indexing. The post-flush native index must be
present with 64 indexed vectors; its file hashes must survive restart.
The existing full-audit helper rechecks the persisted collection.

Stage B passes only if its three explicit tests pass and both collection
reports contain every phase, a 64-point fingerprint, and index files. A
`stage-b.complete.json` marker is written only then. Logs and successful
reports stay in new per-job persistent results directories; vector data and
Cargo build files stay in PBS scratch. Failed runs retain logs and never
create a success marker. The tests use private scratch temp directories.

## What this can establish

A one-segment benchmark can be **accepted** without reingesting vectors or
changing search semantics if an index-only build from a verified copy reaches
quiescence, receives a synchronous native flush, shuts down, restarts, and
then passes exact mapping/fingerprint/index checks with exactly one nonempty
search segment. An empty retained segment is separately checked; a retired
source must disappear or be identified before publication. No test changes
the strict `locate_segment()` rule or combines results across segments.

Small-scale success does **not** prove that the 10M LMI planner will choose
the same topology: its size thresholds, memory behavior, scheduling, and
on-disk cleanup timing differ. If the LMI small case leaves two active live
segments after quiescence/restart, stop; a flush cannot merge them without a
new, explicitly reviewed topology protocol. If it leaves only a retired
source directory, the flush/restart gate can be considered for a later
index-only attempt on a verified *copy* of the accepted data. No 10M run is
authorized by this document.

## Opt-in deployment and execution after review

Publish the branch only after reviewing this implementation and compiling it
against MetaCentrum's environment. Deploy it in a distinct checkout. The
scripts verify the branch, exact SHA, and clean tree. They call the established
Rust 1.98 / Python 3.12 / PyTorch 2.5.1 CPU / g++ / protobuf setup; Cargo
and temporary files are placed in PBS scratch. Resource requests are 8 CPUs,
32 GB memory, 80 GB scratch, and three hours per stage. Each selected test
has a separate 600-second (Stage A) or 900-second (Stage B) timeout.

The following are **manual commands, not executed by this patch**. Replace
`REVIEWED_SHA` with the eventual published implementation SHA:

```bash
WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
QDRANT_DIR="$WORKSPACE/qdrant-sisap23-small-lifecycle"
REVIEWED_SHA=REVIEWED_SHA
git clone --single-branch --branch thesis/sisap23-small-lifecycle \
  https://github.com/nico-occhino/qdrant.git "$QDRANT_DIR"
test "$(git -C "$QDRANT_DIR" rev-parse HEAD)" = "$REVIEWED_SHA"
test -z "$(git -C "$QDRANT_DIR" status --porcelain)"
A=$(qsub -v "WORKSPACE=$WORKSPACE,QDRANT_DIR=$QDRANT_DIR,SISAP23_REVIEWED_COMMIT=$REVIEWED_SHA" \
  "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/56_small_stage_a.pbs")
# Inspect Stage A's five logs and completion marker before authorizing Stage B.
A_MARKER="$WORKSPACE/results/sisap23-small-stage-a-$A/stage-a.complete.json"
qsub -W "depend=afterok:$A" \
  -v "WORKSPACE=$WORKSPACE,QDRANT_DIR=$QDRANT_DIR,SISAP23_REVIEWED_COMMIT=$REVIEWED_SHA,SISAP23_STAGE_A_RESULT=$A_MARKER" \
  "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/57_small_stage_b.pbs"
```

Stage B validates Stage A's marker and matching SHA again. Do not queue
Stage B until Stage A's actual logs have been reviewed; a successful marker
is a gate, not permission for a 10M build.
