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

### Additional native safety closure

A per-shard research shortlist override avoids process-environment mutation in concurrent tests; zero preserves existing environment/default behavior. The actual native optimizer proxy/publication test now starts from a persisted/reopened quantized learned segment, enables quantized dispatch, performs native upsert/delete during proxy construction, and exercises both successful publication and cancellation. Search during proxies falls back; after cancellation the restored generation executes one learned quantized query with expected current IDs/scores. Successful publication preserves mutation population and old-membership fallback. This does not test an old learned query spanning the final swap or learned reactivation on the new output. Targeted1PASS (quantized-publication.log), full26PASS/6ignored (closure-regression.log).

The reopened learned-segment fixture also rejects corrupt quantization JSON via native loader and, after removing native quantization configuration metadata, reopens with no quantized object: original learned routing remains usable, quantized mode rejects. This is a segment-level control, not full Collection::load corrupted-derived-metadata reconstruction. Original scores and complete resolution remain checked. No300K gate override.

## Continuation37 — native collection lifecycle closure

Added persisted, default-None `DliExperimentalConfig.quantized_shortlist`; nonzero shortlist requires bounded materialization. LocalShard reinstalls the setting after native recovery. Native maintenance builds scalar data in each selected learned destination through SegmentBuilder; native insertion/COW segments explicitly remain unquantized. No geometric schedule/model changes. Existing benchmark configs explicitly retain None.

The configured normal-Collection fixture now automatically publishes four quantized generations, queries through actual learned dispatch after each (4learned/0fallback), retains two active learned levels plus Plain, gracefully closes and loads the complete collection, then resumes actual quantized learned routing (1learned/0fallback). Persisted bytes for this tiny16D fixture:463 and343 per active member, including native metadata. These are not65K memory measurements. Quantization construction runs inside native SegmentBuilder before physical publication and therefore inside native optimizer job wall time. Outputs rebuild scalar data from the selected vectors; Plain/COW is original, not copied compressed data.

Missing scalar metadata on one member yields mixed quantized/unquantized learned membership after normal Collection::load and correct full exact fallback (0learned/1exact), with every expected ID/score checked. Corrupt JSON and incompatible dimension metadata each likewise reopen with native vectors intact and yield0learned/1exact. Native learned-index loader now degrades invalid derived quantization to absent; cancellation is preserved and Plain/HNSW loader error behavior unchanged. Corrupt inputs remain on disk for investigation/rebuild; no custom compressed format or vector rewrite. This is clean-reopen/error-injection evidence, not abrupt-crash or power-loss durability.

Deterministic real-optimizer control pins a quantized learned query before proxy installation, completes the real physical holder swap while that query is barrier-paused, then resumes it and checks its old physical view's expected version/score. Subsequent queries see the new topology and native visibility. Success/cancel controls preserve mutations; all26 regressions PASS/6ignored (`continuation37/corrupt-recovery-regression.log`). All new timings are debug correctness only.

**Still missing:** a65K multi-level Collection S0/S2 release comparison against a full-pool oracle, actual scorer rather than input accounting, compressed RSS/read-byte/phase-cost telemetry, cold controls and the A0–A3 validation/test comparison. No300K gate passes. Prior warm scorer results remain the only optimized performance evidence.

## Continuation38 — full-pool oracle preparation

`prepare_65k_oracle.py` computes a blocked Float64 cosine oracle over all65,536 existing pool vectors after Float32 preprocessing and Float16 storage conversion; ties use external-ID order. Executed32 previously used queries800–831, no collection or model modification. Manifest/input/output hashes and arrays: continuation38/oracle. Native Float16 score near-ties can differ; no claim of native bitwise score equivalence. These32 queries are not an untouched held-out set.

Collection instrumentation currently records aggregate native scorer time and global scorer work; it does not independently attribute successful quantized versus original evaluations per query or isolate the two timings. This blocks the requested controlled collection-level cost attribution. No new S0/S2 collection run or cold experiment was executed, and no speedup or 300K promotion gate is claimed.

## Continuation40 — multi-level 65K Collection S0/S2 completed

At source `ceca71685743b7c3c66c57114faa3cde171d5f28`, the ignored release benchmark constructed a normal Qdrant Collection from 65,536 distinct Float16 CLIP-768 vectors through four automatic native optimizer publications. Verified active topology after publication and before queries:

| Role | UUID | Live population | Native files |
|---|---|---:|---|
| Plain buffer, level 0 | `f9ef5fd5-4a38-4238-b7fa-09f2e385e6da` | 1,536 | original vectors |
| Geometric level 2 | `6544fb04-e99b-41bc-85a9-8ebd16a006d3` | 48,000 | scalar128 + LMI router/postings |
| Geometric level 1 | `178b666e-ae7d-41d9-b73a-fae9159899e7` | 16,000 | scalar128 + LMI router/postings |

The test asserted exactly 65,536 unique active external IDs equal to the oracle pool, disjoint membership, two independent LMI members, a nonempty appendable Plain member, matching posting/segment generations and four automatic publications. Each S0/S2 query pair used the same physical members and exact same selected buckets/posting offsets. All 64 queries recorded successful learned dispatch and zero exact fallback. Every timed query returned 30 hits; returned exact-oracle IDs were checked against native scores outside the timed interval. The complete Collection closed and reopened in 3.396 s; a post-reopen query executed learned routing with zero fallback.

### Collection-level result

Release build, warm repeated access only, 64 paired queries, four timed passes per query in ABBA/BAAB order. The held-out label refers to query rows 932–963; the selection record confirms they were not used in continuations 26–38, but prior use outside those records was not independently excluded.

| Metric | A0: original scorer | A2: scalar128 + exact rerank |
|---|---:|---:|
| Full-pool Recall@10 | 0.9922 | 0.9922 |
| Full-pool Recall@30 | 0.9766 | 0.9766 |
| Nominally held-out Recall@10 / @30 (32 queries) | 0.9906 / 0.9708 | 0.9906 / 0.9708 |
| Collection `search` p50 / p95 | 8.058 / 10.371 ms | 5.441 / 7.157 ms |
| Median per-query paired A0/A2 latency ratio | — | 1.483 (p10 1.293, p90 1.772); A2 faster for 64/64 query medians |
| Learned posting entries evaluated/query | 8,512.8 original | 8,512.8 quantized |
| Learned original full-dimensional evaluations/query | 8,512.8 | 256 (128/member) |
| Plain original evaluations/query | 1,536 | 1,536 |
| Exact fallback | 0/64 | 0/64 |

Conditional A2-vs-A0 overlap was 1.0 at both k=10 and k=30. This says the quantized shortlist/reranker preserved the selected-posting scorer's results on these queries; full-pool recall is reported separately above. A2 reduces learned original scorer inputs by 33.3× (97.0%), and combined full-dimensional inputs including Plain by 82.2%. It still evaluates the same 8,513 learned posting candidates approximately; candidate selection/work did not shrink.

Request-local profiling was disabled during latency passes. In the separate profiling pass, median stages were: A0 route/plan 2.374 ms, learned original scoring 4.706 ms, Plain scoring 0.337 ms, materialization 0.061 ms, complete-member resolution 0.251 ms; A2 route/plan 2.638 ms, quantized scoring 1.394 ms, exact learned reranking 0.246 ms, Plain scoring 0.316 ms, materialization 0.045 ms, resolution 0.208 ms. These stage measurements are not added to or substituted for end-to-end latency. Warm timed calls recorded zero major/minor faults; this is not a controlled cold-access test or hardware-byte measurement. Process CPU ticks were coarse (97 ticks over 128 A0 calls; 70 over 128 A2 calls).

### Storage and construction cost

The raw Float16 pool is 100,663,296 bytes. Correct recursive inspection of persisted mmap chunks found 101,088,768 logical bytes / 100,765,696 allocated bytes (chunk capacity/segment files), 49,408,603 scalar-quantized bytes, and 5,664,508 LMI model/posting-index bytes. Quantized bytes by active member were 37,056,302 at level 2 and 12,352,301 at level 1. Whole test-process peak RSS was 914,536 KiB; incremental RSS attributable to quantization was not isolated.

The native builder recorded 0.646 s total inside the four optimizer builds for quantization construction (per destination: 0.118, 0.173, 0.254, 0.101 s); native job wall-time sum was 143.723 s. There is no matched no-quantization optimizer build, so a construction-cost break-even is not established. A naive division of the measured component by the 2.617 ms median per-query saving yields about 247 searches, but that is only a scenario and excludes RAM residency and storage cost.

The first 20K-batch fixture attempt is preserved at `continuation-40/multilevel-native/`; it failed before search because native scheduling correctly coalesced the data into only levels 0 and 2. The corrected 16K schedule and successful release result are under `continuation-40/multilevel-native-retry1/`. `summary.json` retains the raw output; its one-level storage scan undercounted nested mmap chunks, so use `analysis.json`'s corrected recursive resource accounting. The benchmark executable used commit `ceca7168`; the subsequent diagnostics-only byte-count/profile-key correction is commit `b8d4b9157`. The raw inputs and oracle hashes are in the run manifest.

**Decision:** scalar128 is now a promising, native Collection-level experimental scorer on this 65K warm topology, and advances ahead of the original scorer for further bounded validation. It remains opt-in; do not change defaults yet. Independent untouched-query provenance, controlled cold behavior, quantization-specific incremental RSS and matched quantized/unquantized build overhead remain open. No A1/A3 routing combination or 300K run was executed.

The follow-up request-event instrumentation also records native exact-fallback work independently from successful learned-member profiling. A forced underfill emits linked request events for the learned attempt and exact fallback, including explicit fallback reason, successful exact scorer inputs, fallback-stage elapsed time and whole DLI-shard elapsed time. Concurrent filtered/full exact searches retain separate request-local counts. Focused assertions and the 27-test DLI regression set pass. This closes counter attribution for the tested shard-level fallback path; the recorded warm Collection run itself had zero fallbacks, and its end-to-end latency still comes from the timed Collection caller rather than summing nested stage timers.
