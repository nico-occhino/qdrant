# DLI concurrent update and maintenance evaluation (continuation43)

**Harness:** `lib/collection/src/tests/lmi_lifecycle/dli_concurrent_workload.rs` (ignored test). Final version is at `072be2e0c`.

**Evidence:** `<admin>/experiments/2026-11-12-dli-architecture/continuation-43/`
- `workload/summary-readers{1,4,8}-ops1500.json`, `workload-r*.log`.
- `perf-binary.sha256`, `source-head.txt`.

## Design (Tier A: correctness and contention mechanism)

**Collection.** Configured DLI:
- 16-dimensional unit-axis vectors, so every cosine score is exactly 0 or 1.
- Buffer capacity 96, arity 2, a 4-bucket model.
- A candidate budget covering every bucket: learned routing is exhaustive, so any wrong hit is a visibility defect, not ANN recall loss.
- Generic optimizers disabled; DLI publishes natively.

**Operations.**
- Seed: 192 points.
- Writer: a deterministic xorshift log of 1,500 operations, all `WaitUntil::Visible`: 50% inserts, 30% re-axis upserts of live points, 20% deletes.
- Readers: R concurrent tasks issue random-axis top-10 queries until the writer finishes.

**Checks**
1. **Interval linearizability.** Every concurrent hit must match some version of that point whose lifetime overlaps the read interval:
   - a version's visibility starts no later than its operation's start,
   - and ends no earlier than the superseding operation's end,
   - and score 1 must mean the version had the query axis.
2. **Exact final state** at quiescence, per axis, with limit 100,000.
3. **The same exact state after a graceful reopen.**
4. **No duplicate IDs** in any result.
5. **No vacuous checks:** at least 90% of concurrent reads must be non-empty.

## Results (optimized `perf` build, binary sha256 in `perf-binary.sha256`)

| Readers | Concurrent reads | Violations | Read p50/p95/p99 ms | Write p50/p95/p99 ms | Publications | Exact fallbacks (busy / proxy / topology) |
|---|---|---|---|---|---|---|
| 1 | 1,809 | 0 | 0.18 / 0.34 / 0.68 | 0.22 / 0.42 / 0.68 | 12 | 220 (36 / 179 / 5) |
| 4 | 7,672 | 0 | 0.20 / 0.40 / 0.86 | 0.26 / 0.48 / 0.90 | 12 | 2,151 (928 / 1,196 / 27) |
| 8 | 17,200 | 0 | 0.27 / 0.57 / 1.08 | 0.40 / 0.77 / 1.29 | 12 | 4,111 (2,554 / 1,499 / 58) |

All runs:
- Exact final state at quiescence and after graceful reopen, which served 16/16 learned queries with 0 fallbacks.
- 0 duplicate IDs.
- No deleted-point resurrection.

**Interpretation**
- **Correctness (VERIFIED for this fixture):** no lost updates, stale versions, resurrections or duplicates under concurrent reads, writes and native publications.
- **F8 mechanism measured.** `busy` fallbacks rise from 2.0% to 12.1% to 14.9% of reads as readers go from 1 to 4 to 8. A waiting writer makes DLI's non-blocking `try_read` of every member fail. Writer p99 roughly doubles.
- **Proxy windows.** 8.7–15.6% of reads fall back while an optimizer proxy is installed. Correct, but it is lost learned work during maintenance.
- **Scale caveat.** This is a 16-dimensional fixture with sub-millisecond queries. At 65K or 10M, learned queries take milliseconds to seconds and hold guards correspondingly longer, so contention should be larger. Not measured here: Tier B is pending.

## New finding F12 (fixed, `072be2e0c`)

**Symptom.** After a graceful stop that followed recent publications:
- Before stop: the descriptor, in-memory roles and holder all agreed (2 members).
- After reopen: the holder contained **7** segments. DLI activation failed closed (`topology`, runtime not ready), and 16/16 queries fell back with a full rebuild scheduled.

**Cause.** Upstream destroys retired optimizer sources through post-flush actions only once their replacement is durable. Until then, a restart reloads them beside the replacement, load-time deduplication resolves the overlap, and the WAL acknowledgement is pinned. That is correct for normal collections. The DLI descriptor requires exact membership, so it conflicted.

**Fix.** `stop_gracefully` issues one final sync flush for DLI shards, after DLI maintenance is cancelled and the update workers have stopped. Post-flush drops are gated on replacement durability, so only safe drops run.

**Evidence:**

| | Holder after reopen | Learned / exact queries | Runtime ready |
|---|---|---|---|
| Without the flush | 7 | 0 / 16 | no |
| With the flush (3/3 runs, no test-side flush) | 2 | 16 / 0 | yes |

**Remaining limit.** A **crash** restart can still reload pending-retired segments: fail closed, then rebuild. Tolerating them at activation would need a version-ownership proof for duplicated points and is not attempted.

## F7 runtime boundary (fixed, `8c74ac1b0`)

Runtime `update_optimizer_params` with threads ≠ 0, or enabling collection-wide quantization, would let generic optimizers restructure DLI members, or make the collection fail to load at the next restart. Both are now rejected before persistence, and `config.json` is unchanged afterwards (regression in `dli_recovery_control`).

## Not done

- **Tier B:** contention at 65K or 10M with real query cost and compaction-triggering write volume.
- A versioned recall oracle under concurrency: the fixture uses exhaustive routing to isolate visibility.
- **Descriptor-save fault injection (F10):** not added.
