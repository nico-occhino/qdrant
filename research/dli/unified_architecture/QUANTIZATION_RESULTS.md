# Completed learned-posting release comparison

Source:10b8f116b plus source.patch recorded in continuation33/handoff.json, committed as6651b4ef7. Log completed1PASS; no duplicate run launched. Native disposable65K Float16 trained segment,32 queries800–831, model-predicted top16 posting lists, identical sorted candidate offsets for S0/S1/S2. All512 independent score assertions outside each timed S2 query? More precisely,30 returned-hit assertions per query/shortlist, with the execution timer captured before them. No assertion-count inference is used as measured work.

Mean selected/quantized inputs2153.8125 per query; S0 scores all these originals. All S2 shortlist settings have conditional Recall30=1.0 versus S0. S1 top30 conditional Recall30=.95. Conditional Recall10 was not logged and cannot be recovered unambiguously from top30 set recall under near ties; whole-pool recall is not recorded here. Counters are known scorer inputs for this deletion-free candidate fixture, not physical reads or hardware-scoring counters.

|Requested shortlist|Mean originals|S2 p50 ms|S2 p95 ms|
|---:|---:|---:|---:|
|64|64.0|0.371|0.562|
|128|128.0|0.276|0.536|
|256|256.0|0.184|0.275|
|512|512.0|0.253|0.319|
|1024|1024.0|0.387|1.118|
|2048|1889.6|0.627|0.971|
|65536|2153.8|0.412|0.834|

S0 p50/p95=0.834/1.169ms. Scalar construction=['309']ms. Scalar128 has16.83x fewer original inputs and a nominal3.02x scorer-pipeline median improvement. **Repeatable end-to-end acceptance gate does not pass:** S0 always executes first; later sweeps reuse original vector pages and score assertions warm data further. No randomized paired repetitions, native collection resolution timing, CPU, RSS, fault/read-byte telemetry or uniformly cold results were captured. Warm and cold p50/p95 cannot be reported separately. Quantized artifact bytes were not recorded before disposable teardown. No missing measurements are zero-filled.

The256 setting is faster than128 despite more original inputs, further illustrating order/cache effects. Quantized-specific concurrent publication, generation/reopen/recovery tests are deferred because the performance gate is unresolved; no scorer promotion. Existing default correctness remains tested by prior suite, but it is not evidence for new compressed-path recovery. Scalar128 remains a promising candidate, rejected for promotion pending repeated collection-level controls. No10M backfill.

Continuation35 adds a warm-only ABBA/BAAB mode, DLI_COUNTERBALANCED=1, to the existing native fixture. Eight alternating S0/S2 passes per query use identical selected offsets, with conditional Recall10/30, original/quantized scorer inputs, task-local CPU ticks and major/minor faults. Independent score assertions are outside timing. Initial S0 is warmup. This is a scorer control, not complete collection-level latency or cold-cache telemetry. Release compilation is pending in continuation35/abba-build.log; no counterbalanced speedup is claimed.

## Continuation36 — counterbalanced warm native control completed

Executed optimized binary sourcef92bcf0ff, hash/config in continuation36/manifest.json. Warm-only32 queries800–831, ABBA/BAAB eightpasses/query,128 samples/mode. No duplicate or new training. Native score assertions outside execution timer.

|Metric|S0 original|S2 scalar128 + original rerank|
|---|---:|---:|
|Pooled p50/p95 ms|.296/.943|.155/.369|
|Conditional Recall10 and30, minimum over passes|1/1|1/1|
|Original scorer inputs, all128 passes|275688|16384|
|Quantized scorer inputs|0|275688|
|Task CPU ticks (100Hz)|3|4|
|Minor/major faults|3/0|7/0|

Median query-paired speed ratio1.825; S2 wins all32 paired query medians. Query-bootstrap95% interval[1.748,2.007] is descriptive variability across reused queries, not independent model replication. CPU totals .03/.04s have10ms granularity and cannot establish a CPU-work improvement. No major faults means this does not measure a cold-I/O advantage. Additional quantized-index bytes/RSS, full-pool recall, complete collection latency and separately timed approximate/rerank stages remain unmeasured in ABBA. Inputs are known unique/live offsets in this fixture, not general production hardware-access accounting. Per-hit assertions outside timing add original accesses outside reported algorithmic work.

**Warm scorer screening gate PASS; promotion gate NOT PASS.** Scalar128 merits controlled integration testing. Newly added path/dimension validation rejects a quantized object from another segment even with the same dimensions/population. The trained native fixture swaps objects to test rejection. Native builder-created quantized learned segment now flushes/reopens, survives learned deletion plus Plain reinsertion, resolves complete-member results, and rejects an unavailable object. Precise dispatcher reasons:quantization_unavailable,quantization_generation_mismatch,quantization_configuration. Regressions26PASS/6ignored; final-regression.log.

Path/dimension checks reject accidental cross-generation object reuse; they are not cryptographic validation of same-path corrupted data or proof against incompatible persisted contents. Quantized-specific concurrent publication and persisted-corruption recovery remain unverified; default original scorer retained. No300K run launched.
