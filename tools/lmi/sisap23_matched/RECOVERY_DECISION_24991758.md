# Recovery decision: SISAP23 LMI generation from job 24900597

**Decision:** do not rebuild. Treat the retained LMI build as a reusable recovery candidate, but do not accept or publish it yet. The 20-query warmup protocol has not been validated for this index. No recovery-validation job was submitted in this turn because SSH access to Perian failed; PBS evidence retrieval and job submission are currently unavailable.

## Proven so far

The user-reported successful diagnostic job `24991758.pbs-m1.metacentrum.cz` queried row 0 from the prepared SISAP warmup file through ordinary `Collection.search()` on an isolated persistent NFS copy.

- Cold: 60.082 s and a search timeout; 18,543 candidates; routing 7.549 ms; learned route +1; Plain fallback +0; scoring did not finish.
- After the diagnostic's full inventory read: 78.413 ms; 18,543 candidates; routing 8.488 ms; scoring 68.312 ms; learned route +1; Plain fallback +0; ten valid IDs.
- The isolated copy matched 38 files / 15,979,223,137 bytes. The report says before/after index hashes were unchanged.
- PBS exit status was 0. This indicates the diagnostic harness completed and recorded the cold timeout; it does not mean the cold query passed.

Raw files are retained under:

`/auto/budejovice1/nico/lmi-metacentrum/development-search-timeout-24900597-20261026/attempt-24991758.pbs-m1.metacentrum.cz/`

Expected report names are `query-prepared-cold.json`, `query-prepared-warm.json`, `summary.json`, `inventory-diff.json`, `query-cold.resources.txt`, `query-warm.resources.txt`, and the index hash manifests. The supplied summary did not include numeric resource fields or hash values. A read-only retrieval attempt in this turn failed: multiplexed SSH reported a broken pipe, and a fresh BatchMode connection was denied with `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`. No SSH configuration or credentials were changed.

The source-stage inventory from failed job `24900597.pbs-m1.metacentrum.cz` records 38 files totaling 15,979,223,137 bytes. Its pre-search verifier had already reached checks for point accounting, mappings, postings and persisted index metadata before the ordinary search timed out. The stage has no accepted generation descriptor or publication marker. Its path remains:

`/auto/budejovice1/nico/lmi-metacentrum/results/sisap23-lmi-multisegment-20261023-02/.staging/build-lmi-24900597.pbs-m1.metacentrum.cz-b931357cd167`

The accepted Plain source remains read-only at:

`/auto/budejovice1/nico/lmi-metacentrum/results/sisap23-independent-recovery-20261019-01/recover-lmi-24808764.pbs-m1.metacentrum.cz-f46b380a2475`

Its known manifest SHA-256 is `5ab566ba0f641b177248d7fe42b83540a13b6f779da2c455ec815ea155d2ec96`; producer revision is `69216d5ada898feef7066c72da0514aeae2ead89`. Deployment source revision is `9b12d583b1bf9df802b2cc420244bd3c24732d86`.

## Why recovery is plausible but not yet accepted

The failure is reproducibly after learned routing, during candidate scoring on cold NFS. The same query scores quickly after a full file inventory warms the diagnostic copy. That supports reusing the existing built files and does not justify rebuilding the index. It does not prove that the benchmark's prescribed warmups are sufficient, because job `24991758` warmed pages by scanning every file rather than by running those warmups.

The existing verifier requires an ordinary collection query to succeed on the persistent copy under the default 60-second timeout. `recovery_stages.py` embeds that result as `persistent_report.ordinary_lmi_query_path`; `check_generation_metadata()` and `generation_publication.review_staging()` refuse publication without it. A scratch-only query report cannot satisfy the current contract. Removing this requirement would weaken the contract; the justified change is to represent persistence integrity, structural reopen, query correctness, and cold/warm latency as separate evidence while retaining strict hashes and query-path checks.

## Required recovery-validation protocol

Run one non-rerunnable PBS validation job only after SSH access is restored and the job script has been reviewed. Do not submit any index build.

1. Create a new unique recovery root, proposed name `sisap23-lmi-recovered-20261026-01`; first verify remotely that it does not exist. Never write to the failed stage or accepted Plain source.
2. Copy the failed stage's `collection` into new persistent staging and independently into PBS local scratch. Do not use hard links. Inventory every path, size and SHA-256 and require byte-for-byte equivalence before opening either copy.
3. Preserve a cold persistent-NFS ordinary query measurement as an operational result. Record timeout/error, elapsed time, learned/fallback deltas, selected candidates, route time, scoring completion and system resource counters. A timeout is data, not a successful query.
4. On the isolated persistent copy, execute the exact 20 rows in `queries_warmup.f32le` in the project's order, recording each row's latency, IDs, candidate count, errors and route/fallback evidence. Do not use a full inventory scan between cold measurement and this warmup sequence. Then run measured query row 20 and retain its result. Do not discard slow warmup calls.
5. Repeat the official 20-query sequence and measured row on a byte-identical scratch copy. Label the storage as local scratch. Do not compare it to a baseline measured cold on NFS.
6. Reopen the scratch copy in normal Qdrant processes, including the existing restart checks. Re-run structural validation: active topology, all 10,120,191 points, unique/reversible IDs, no missing or duplicate mappings, full postings coverage, valid offsets, trained LMI state for every live indexed segment, persistence hashes, and clean reopen without reingestion or optimization.
7. After each read/open/search phase, re-inventory both persistent and scratch artifacts. Require published persistent bytes to match the original failed-stage inventory exactly; if startup or verification rewrites any file, fail closed and preserve the copy for diagnosis.
8. After the official warmups, issue ordinary `Collection.search()` for a prepared measured query and require valid top-10 IDs, one learned-route increment per active indexed segment, zero Plain-fallback increments, and no omitted segment. Compare measured row 20 with official gold (and the exact Plain reference if its provenance is validated); record recall@10. One row is a focused check, not a benchmark result.
9. Write separate, machine-readable `persistent_integrity`, `structural_reopen`, `cold_nfs_queries`, `official_warmup_queries`, `scratch_queries`, and `accuracy` records. Update publication review to validate all report hashes, source/input provenance, and the matching byte inventory. Accept only if every structural and warm-query criterion passes; retain cold NFS timeouts as explicit operational limitations. This is an explicit schema change, not a suppressed error.

## Warmup sufficiency is unproven

The matched evaluator in `lib/collection/src/tests/lmi_sisap23_matched.rs` executes 20 prepared warmups before measured rows, but `search_one()` calls the segment index directly and discards warmup timing and IDs. For LMI it routes and calls `score_candidates()` without the ordinary collection operation timeout. Consequently, the existing evaluator does not establish ordinary API behavior or report whether any warmup itself is slow. The required diagnostic must preserve all 20 per-query timings and errors, and include at least the first measured row after them.

## Tests and remaining blockers

The lightweight Python suites passed: 41 tests across `test_recovery_stages`, `test_generation_publication`, `test_preservation`, and `test_validate` using `PYTHONPATH` set to the repository root. Rust compilation/tests were not run because remote access failed and Perian must not be used for compilation. No acceptance-schema changes have been made yet; changing the contract before observing the official warmup sequence would be premature.

Current blocker: restore authenticated SSH to Perian to retrieve raw job `24991758` resources/hash files, verify current output-path availability and submit the single isolated PBS validation job. Until then, the generation remains unaccepted and unmodified; no rebuild, publication or matched benchmark is appropriate.
