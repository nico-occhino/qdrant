# SISAP23 300K matched run — first result

**Source revision:** `47ddac824db106014c47e3b56a439d514e921b81`  
**Status:** one optimized run; not replicated or recall-matched.  
**Raw artifacts:** `/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-22/benchmark-300k/`

## Provenance and workload

The run used the canonical pre-existing 300,000-row SISAP23 LAION CLIP-768 HDF5 subset, official 10K public query set and official 10K-by-1000 gold file. Source MD5s: vectors `d238b4b037c32bae41e497f95dffa895`; queries `257b9eb3f7f25776e0d33b22451b7b32`; gold `5d635f26630cced971358fd76f37c32e`. Extracted SHA256s and the full manifest are in `manifest.json`. Vectors are Float16, 768 dimensions, cosine, one-based row IDs. Both variants received the same six ordered 50K batches and same query/update IDs. Updates were 128 same-vector upserts and 128 deletes. The release/optimized Rust test ran sequentially, static first, then DLI. DLI used buffer 50K, arity 2, candidate budget 2000, 548 LMI buckets, sample 32768, hidden size 512, 30 epochs, and seed 42. Static used normal Qdrant optimizer maintenance.

## Results

| Metric | Static LMI + native optimizer | Temporal DLI | DLI minus static |
|---|---:|---:|---:|
| Recall@10, before updates | 0.7962 | 0.7625 | -0.0337 (-3.37 pp) |
| Recall@30, before updates | 0.7628 | 0.7253 | -0.0375 (-3.75 pp) |
| Query p50, ms | 4.358 | 6.900 | +2.542 (+58.3%) |
| Query p95, ms | 6.179 | 12.483 | +6.304 (+102.0%) |
| Six insert-call durations summed, ms | 3608.1 | 3960.2 | +352.1 (+9.8%) |
| 128 upsert call, ms | 4.953 | 3.769 | -1.184 |
| 128 delete call, ms | 0.331 | 0.285 | -0.046 |
| Maintenance timing sum, s | 561.999 | 300.906 | -261.093* |
| LMI auxiliary files, bytes | 3,904,256 | 10,312,768 | +6,408,512 (+164.1%) |
| DLI temporal publications | n/a | 6 | n/a |
| DLI output vectors cumulatively rebuilt | not instrumented | 750,000 | n/a |
| Selected posting entries / 1,000 pre-update queries | 3,087,797 | 2,489,314 | -598,483 |
| Exact fallback after mixed updates | not reported | 100 / 100 | n/a |
| Combined process peak RSS, KiB | not split | 3,105,320 | not comparable |

`*` Static is insert-to-indexed wall wait; DLI is summed native optimizer-job wall time. They have different boundaries and do not establish a 46.5% reduction in equivalent CPU or maintenance work. The query figures are sequential, not interleaved or randomized; cache-order bias is possible. Candidate entries are selected posting offsets, not a measured count of vectors actually scored by the native scorer.

Post-update recall was static 0.832/0.8023 and DLI 0.997/0.998 at 10/30. All 100 DLI post-update queries used exact fallback. The oracle removes deleted IDs from official top-1000 results; same-vector upserts do not change ranking. This truncated-list calculation is not a newly computed exhaustive live-population oracle and should not be interpreted as exact recall.

## Interpretation

This run demonstrates six ordinary-insert-triggered native DLI publications and real learned queries at 300K, then exact fallback after updates. Under the tested fixed configuration, DLI has lower recall, slower query p50/p95, more auxiliary index files, and a higher summed insert-call duration. It selected 750K output vectors across six publications, including a 300K transition, so these observations do not support a general reduced-rebuild claim. The apparent maintenance-time difference is not directly comparable due to instrumentation boundaries. The fallback rate after deletes is 100% on the measured post-update queries.

There is one run/seed, no p99, no recall-matched operating point, no interleaved latency design, no per-system RSS split, no precise static vectors-rewritten/training-time instrumentation, no measured native scorer count, and no fresh exhaustive post-update ground truth. Treat these results as a first bounded evaluation and negative signal, not a final performance conclusion.

## Reproduction and logs

The exact release test command and environment are recorded in `research/dli/TEST_MATRIX.md`; the run log is `logs/run-no-generic-dli-optimizer.log`, raw summary `raw_results/summary.json`, publication events `maintenance_events/publications.json`, and generated comparison `results.csv`. Two earlier runs failed after 150K; `logs/run.log`, `logs/run-diagnostics.log`, and `logs/run-corrected.log` preserve the attempts. Diagnosis: generic Qdrant optimizer workers contended for DLI-owned member topology and produced `new DLI member is not appendable`. The DLI benchmark now sets generic optimization threads to zero while static retains normal optimizer threads. No host security settings or remote HPC resources were changed.

Follow-up: the original observations above are preserved. The revised deletion resolver and separate rerun are documented in [DLI_POST_UPDATE_STUDY.md](DLI_POST_UPDATE_STUDY.md); post-update learned routing increased to 100/100 queries with zero fallbacks.
