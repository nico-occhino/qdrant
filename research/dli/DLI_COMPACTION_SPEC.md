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
