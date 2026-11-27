# Q5 local gates (2026-11-27, on one WSL machine)

- **Binaries** (`binaries.sha256`).
  - The base is the perf build of `34dbeffd7`.
  - "new" is the perf build of `5698c41c8`, then rebuilt at the `ffae40bfc` code with the `copy_dir` fix. The
    third line of `binaries.sha256` is the binary that passed.
- **Attempt 1 failed in the harness.** The new E1 run panicked in `copy_dir` with NotFound at the batch-2
  snapshot: a file vanished mid-copy. The fix is in `ffae40bfc`. This is one documented infrastructure retry.
- **Retry 1:**
  - gate 2 (`gate2.json`): E1 base = E1 new on S1 batches 0–2 for `mq_val`, over the 17-point grid. That covers
    1,632 rows, 3 topologies, the 3-batch ledger with routers, and 27 builds. **PASS.**
  - gate 3 (`gate3.json`): E1T on the same prefix. It made 23 decisions with 0 retrains; the largest eligible ratio
    was 2.59×. The decisions agree with the builds (4 trained = 4 first fits). **PASS.**
  - No `COPY_DIR_VANISHED` occurred.
