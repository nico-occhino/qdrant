# Post-update learned routing — continuation 23

Revision a1c891046; release benchmark: 1 passed, 0 failed, 879.43 s. Raw run: continuation-23/post-update-300k/run.log and summary.json. Canonical inputs and hashes are unchanged from continuation-22/benchmark-300k/manifest.json. Both systems ingested six ordered 50K batches, queried 1000 official queries, applied 128 same-vector upserts and 128 deletes, then queried 100 queries.

The removed rejection was global nonzero deleted-point count. Native scoring excludes deleted offsets; complete selected-ID checks still reject duplicate visible physical copies, deferred state and identity mismatches. Returned scores/IDs and delete/reinsert versions are checked in the deterministic tests. The retained-old/unselected-new-copy regression still falls back.

| DLI post-update metric | Original | Revised |
|---|---:|---:|
| Learned queries | 0 | 100 |
| Exact fallback queries | 100 | 0 |
| Recall@10 | .997 | .784 |
| Recall@30 | .998 | .751667 |
| Query p50 ms | 41.893 | 6.873 |
| Query p95 ms | 51.358 | 9.907 |

These rows compare exact fallback with learned ANN; they do not show a quality-matched speedup. Revised pre-update Recall@10/30 remains .7625/.725333, selected postings 2,489,314 across 1000 queries. Six native publications rebuilt 750K vectors. The same requested budget and model seed produce the same pre-update selected posting count as the original. Lower ANN post-update recall is approximation, not evidence of stale/deleted hits. The truncated official gold is appropriate only for this unchanged-vector/up-to-128-deletion stream; vector-changing streams need regenerated ground truth.

Shared static/DLI optimizer event boundaries, actual scorer counts, isolated RSS, alternative scheduler, routing and visibility ablations, recall-matched sweeps and replication are NOT IMPLEMENTED in this checkpoint. Maintenance timings remain incomparable. No geometric advantage is established. The first-run report remains preserved.

Tests: DLI suite 17 passed, 0 failed, 1 ignored (deletion-resolution-retry.log). Initial deletion-resolution.log failures were obsolete fallback expectations and missing ambient metrics scope. A later experimental underfill guard was reverted before this checkpoint: it made narrow-budget controls fall back by comparing their result size to whole-population count, changing the primitive's bounded-candidate contract. underfill.log preserves that unsuccessful experiment; it is not the committed source/test evidence. Adaptive replenishment remains unresolved for budgets that yield fewer live hits than requested.
