---
name: database-lifecycle-review
description: Review a database index change for visibility, publication, recovery and concurrent-access defects.
---

Read the pinned commit, parent diff, actual call sites and tests independently.
Trace the exact diff through shard optimizer, proxy writes, segment holder, point-version resolution and persistence. Check Rust lifetime and lock order across search/update, durable file writes versus generation publication, failpoints before/after publication, WAL replay, tombstones, model reuse and stale exact fallbacks. Compare reopen results and route counters against a deterministic update/delete fixture.
Report each finding with severity, path and line, violated invariant, reproducible evidence and smallest suggested correction. Identify untested risks separately. Keep review read-only unless asked to implement a fix.
