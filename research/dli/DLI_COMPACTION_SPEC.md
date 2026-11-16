# Temporal compaction policy — arithmetic stage

The buffer is separate from learned levels. Let b > 0 be its threshold and integer arity a >= 2. Learned level i is one-based and has capacity C_i = b a^i (Sommerová thesis §3.1–3.3, printed pp. 31–34). This matches the paper's maximum-capacity convention; the separate deletion-driven minimum occupancy/rebuild rules are not implemented.

When a buffer drain is requested, accumulate its occupancy plus each higher (smaller-index) level that cannot fit into the current target. Select the first target whose free capacity accommodates the accumulated input. A replacement must include the target's existing population as well as the buffer and all intervening levels, even when those levels are empty. Inputs may exceed b, so newly created empty intermediate levels can be skipped. Integer overflow, impossible existing occupancy and invalid configuration return errors instead of publishing a partial plan.

This is a scheduling policy only. Real compaction must choose native segments, obtain a consistent source snapshot, preserve deletes/upserts and writes via proxy/COW, rebuild model and postings for destination offsets, validate membership, coordinate publication and support recovery. The current planner stores no points, does not train, and must not be wired to automatically publish any native level before those contracts are established. Partial filling of an existing level requires rebuilding the immutable Qdrant LMI generation with the existing target population and incoming sources; reusing a stale router/posting pair is not sufficient.

## Continuation 09: native private builder
A controlled `SegmentBuilder` test combines two trained LMI segments and a Plain buffer into a newly trained LMI replacement and reopens it. This establishes a bounded source-to-destination adapter control only: no scheduler-triggered transition, concurrent proxy update or publication is implemented.

Continuation 10: one private builder control uses scheduler input `[3,0,3]` and buffer 2 to select level 2, combines buffer and level 1, and verifies level 3 separately. No automatic trigger, publication, concurrent mutation or hierarchy recovery is established.

Continuation 11 does not implement automatic compaction or optimizer publication; earlier selective private-build evidence remains unchanged.

Continuation 12 does not implement automatic construction or publication; concurrency and recovery prerequisites remain open.

Continuation 13 validates dispatcher fallback during manually invoked native optimizer publication and cancellation. No ordinary-write threshold trigger or scheduler-driven automatic compaction is implemented.

Continuation 14 connects successful ordinary update completion to the scheduler and a private native LMI builder, under test-only research activation. A threshold-crossing collection insert selects buffer plus level1 into private level2 while preserving level3. Seven live output points are verified and manually reopened. Outputs never enter the holder; automatic publication, repeated published transitions and recovery remain absent.

Continuation 15 adds a test-gated LMI `OptimizationStrategy` adapter exercised by the existing real native optimizer publication/cancellation control. The automatic scheduler still does not invoke it: its source plan needs UUID/role/occupancy revalidation coordinated with native source protection. Extra COW segment creation is explicitly unsupported.

Continuation 16: the native optimizer LMI adapter rejects physical UUID-set mismatches after source proxies are installed. The ordinary-update scheduler is not connected to this check or to optimizer publication; level roles and occupancy are not validated at the protection boundary.

Continuation 17 supplies a source-validation seam before native proxy installation; automatic publication remains disabled until scheduled roles and generation are coordinated.

Continuation 18 passes the scheduled UUID set to the native LMI adapter for the existing private builder. It implements restricted native Plain COW-segment construction, tested directly. The scheduled job still does not execute the optimizer or publish.

Continuation 19: test-enabled ordinary insert can schedule real `execute_optimization`, consume the selected level1 and buffer, create an extra COW and publish one trained physical LMI. This is a manually assembled disposable fixture; logical hierarchy advancement, concurrent mutation on this automatic job, repeated transitions and recovery are not validated.
Continuation 20: the test-only maintenance worker now advances a mutable UUID/level map after native publication. A second ordinary insertion batch uses the optimizer-created Plain COW as the buffer and publishes a separate trained level 1 while preserving previously published level 2 and untouched level 3. This is two real physical transitions in one disposable fixture, not a managed production maintenance lifecycle.
Continuation 20 adds a deterministic cancellation before native optimizer entry: buffer threshold crossing schedules an ordinary insert job, cancellation returns without publication, the native exact query sees all buffered updates, and a later ordinary insert causes a successful retry. The third publication merges buffer plus selected levels 1 and 2 into target level 3, while native optimizer preserves the lifecycle. This does not test writes between source validation and proxy installation or during late propagation.
The automatic transition test now includes an upsert to a selected Plain-buffer point after optimizer plan validation and before proxy installation. The update and a concurrent collection query both complete while the job is paused; after native publication, the updated cosine score and all 21 unique external IDs match the fixture oracle. A replayed stale scheduled plan is rejected before temp-COW/proxy setup. Late proxy propagation and concurrent delete are not covered by these automatic-job tests.
# Continuation 22 runtime note

The `dli-research` collection feature now enables ordinary LocalShard update notifications and native optimization. `DliExperimentalConfig` provides the restricted buffer capacity, arity, candidate budget and LMI training parameters. The configured collection fixture drove four actual optimizer publications from normal batches, with target/output pairs `(1,3)`, `(1,6)`, `(2,9)`, `(1,3)`; these demonstrate repeated selective transitions on a small fixture. The final active hierarchy has learned levels 1 and 2 plus a Plain buffer. `DliPublicationEvent` records sources, target, output UUID/count and native optimizer wall time for measurement.

If a persisted descriptor is invalid, active native segments remain exact-search inputs and become role-zero reconstruction sources, but learned search membership stays empty. The next ordinary update drives a full native reconstruction; learning reactivates only after physical optimizer publication and descriptor update. Graceful shard shutdown now cancels and joins the tracked DLI optimizer job.

No full SISAP23 benchmark result is yet recorded. Canonical data hashes and prepared binary extracts are under `continuation-22/benchmark-300k/`.
