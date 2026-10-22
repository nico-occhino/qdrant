# SISAP23 topology decision after job 24851723

Status: diagnostic hardening is ready for review; another 10M LMI build is
not yet justified. This report audits commit 4630620af on the isolated branch.
The failed-job baseline 4e9b33cd and all accepted and failed experiment
artifacts remain unchanged.

## Source-backed findings

1. Collection indexed-vector count is the sum of per-segment indexed counts
   (LocalShard::local_shard_info). The original wait_for_index accepted that
   sum alone. Commit 4630620af also requires exact point count and Green
   collection status.
2. Green is a snapshot: LocalShard::local_shard_status checks for proxy
   segments and whether the current optimizer planner proposes work. It does
   not wait for a durable flush, post-flush cleanup, or a stable result across
   repeated samples. The final persisted topology and full data checks remain
   authoritative.
3. LocalShard::stop_gracefully stops update, optimizer and flush workers,
   then asks outstanding optimization handles to stop and awaits them.
   With the former count-only gate, a running task could have been stopped.
   The new Green gate reduces this race but does not prove it impossible.
4. MergeOptimizer may leave two active segments with a one-segment target:
   it will not plan a two-segment batch unless another batch makes the
   replacement appendable segment yield a net reduction. The small planner
   test added in 4630620af passes. default_segment_number=1 is a target.
5. During optimizer swap, the old source segment's files are retained until
   a subsequent flush proves the new data durable (shard::optimize and
   SegmentHolder::register_segment_drop). LocalShard::load enumerates
   physical segment directories and performs reconciliation. Consequently,
   two nonempty directories could be an indexed replacement plus a retired
   source awaiting cleanup, as well as two active search segments. This is
   a source-supported hypothesis for this job, not a verified diagnosis.
6. The 10,120,191 Float16 x 768 input vectors occupy 15,544,613,376 raw
   bytes. max_segment_size=32,000,000 means KiB of vector data
   (32,768,000,000 bytes), not a point count. This nominal cap does not
   prove that any particular two segments are mergeable: the planner uses
   measured per-segment available vector sizes, a strict size inequality,
   and its own batch rules. indexing_threshold=1 is KiB; one optimization
   thread is a concurrency limit.

The reported job log establishes two nonempty physical directories after
the old count-only completion. It does not give their UUIDs, live counts,
index kinds, active-holder membership, optimizer task state, or whether a
post-flush cleanup was pending. The source data had one nonempty Plain
segment and passed full coverage checks. No post-build indexed collection
was copied to persistent staging because the scratch native test failed.
The saved scratch path alone does not prove the directory survived PBS
cleanup.

## Construction choices

| Choice | Preservation and comparability | Decision |
| --- | --- | --- |
| Native index-only rebuild from the accepted Plain data, then a documented synchronous flush and restart | Retains Float16 vectors and fixed LMI parameters; can be validated without reingestion. An explicit flush/cleanup gate needs a small native regression before a large run. | Preferred direction if the extra directory was retired source data. Not yet implemented or proven. |
| Pre-index consolidation on a scratch copy | Can keep one source segment, but the accepted Plain data already reports one nonempty segment. It may not address a split created during indexing. Requires full fingerprint and mapping audit. | Not justified from current evidence. |
| Search all segments and merge results | Requires every live segment to be indexed and searched. Separate LMI routers/postings per segment or a Plain/LMI mixture change the scientific configuration and scoring work. HNSW effort semantics also change. | Separate benchmark design, not an acceptance shortcut. |

Neither the patch nor existing optimizer settings guarantee one final segment.
A repeat 10M build now would mainly test the unresolved race/retirement
hypothesis. If a compile-only smoke passes, the next informative experiment
is a small native index-only construction plus forced durability/restart
test using disposable Float16 data. Only after that should an expensive
index-only run be considered. Do not reingest the accepted 10M data.

## Compile-only PBS smoke prepared

The new 55_recovery_compile_smoke.pbs script uses the established
common-env.sh: Python 3.12/PyTorch 2.5.1 CPU, LIBTORCH_USE_PYTORCH=1, no
version-check bypass, g++, Rust 1.98.0 and protobuf. It pins a clean
checkout of the isolated branch to an explicitly supplied reviewed SHA,
places Cargo target and temporary files in PBS scratch, and compiles and
links the collection library test binary in release mode with lmi-training.
It does not run tests or access the corpus. It writes logs and a
compile-only marker to a fresh persistent directory created exclusively
for that job. The script has been syntax checked; no PBS job was submitted.

Local tests at commit 4630620af: one Qdrant merge-planner regression and
13 Python recovery workflow tests passed, as did Bash syntax and diff
checks. Feature-gated Rust compilation remains unverified: the local WSL
environment has no PyTorch/LibTorch.

## Controlled next steps

1. Review the local diff against 4e9b33cd and this report. Confirm the
   previous persistent staging contains only failed-job diagnostics; inspect
   its scratch.log and scratch.exit.json read-only. Do not attempt to reuse
   an unverified scratch collection unless the scheduler confirms it exists.
2. After approval to publish, push this isolated branch. Deploy the exact
   reviewed SHA to a distinct MetaCentrum checkout. Leave the 4e9b33cd
   checkout intact. Verify branch, SHA and a clean worktree before any PBS
   submission.
3. Only after separately approving the small compile allocation, set
   SISAP23_REVIEWED_COMMIT to the deployed SHA and manually submit
   55_recovery_compile_smoke.pbs. Inspect job.log, compile.log,
   compile.resources.txt, toolchain provenance and compile.complete.json.
   A missing marker or nonzero PBS exit is a failed compile gate.
4. Design and run a disposable small Float16 index-only lifecycle test:
   count active and physical segments after indexed-count attainment,
   before shutdown, after a synchronous flush, and after restart. Verify
   full point mappings and vector fingerprint at each durable stage. Include
   a two-active-segment case and a retired-source-file case. This is the
   required test before changing the 10M workflow's durability behavior.
5. If the small test proves a reliable one-segment path, add that explicit
   gate on a fresh patch, compile it on MetaCentrum, and use a new immutable
   10M index generation. Reuse the accepted data generation by its recorded
   producer SHA. Preserve the original failed attempt, fixed grids, and
   campaign/hardware provenance. Update reviewed commit pins and create
   new publication/campaign metadata only for the new generation.
6. If the small test cannot force a single fully verified segment, stop.
   Choose an explicit new topology protocol before spending another large
   job; do not relax locate_segment or publish a mixed-index generation.

After branch publication is approved, these are the exact manual
deployment and compile commands. Replace REVIEWED_SHA with the local
reviewed commit printed in the delivery report; the checkout test refuses
a different revision. Run the qsub line only after the small allocation is
approved.

~~~bash
WORKSPACE=/storage/budejovice1/home/nico/lmi-metacentrum
QDRANT_DIR="$WORKSPACE/qdrant-sisap23-index-topology-diagnostics"
REVIEWED_SHA=REVIEWED_SHA_FROM_DELIVERY
git clone --branch thesis/sisap23-index-topology-diagnostics --single-branch \
  https://github.com/nico-occhino/qdrant.git "$QDRANT_DIR"
test "$(git -C "$QDRANT_DIR" rev-parse HEAD)" = "$REVIEWED_SHA"
test "$(git -C "$QDRANT_DIR" branch --show-current)" = thesis/sisap23-index-topology-diagnostics
test -z "$(git -C "$QDRANT_DIR" status --porcelain)"
qsub -v "WORKSPACE=$WORKSPACE,QDRANT_DIR=$QDRANT_DIR,SISAP23_REVIEWED_COMMIT=$REVIEWED_SHA" \
  "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/55_recovery_compile_smoke.pbs"
~~~

The clone command requires that the destination does not exist. A nonzero
exit at any check stops deployment. No source generation or campaign
manifest is modified by the compile job.

No branch has been pushed and no cluster job or large construction has
been run for this verification pass.
