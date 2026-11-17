# Current architecture decision — continuation35

1. **Bucket configuration:** retain548/32768-sample/512-hidden baseline. Completed128 and256 controls plus older548 refined curves reject128 for high-recall scaling;256 is a smaller-model ablation, not proven replacement. Fresh548 replicate is still in the original serial process; it is not silently treated as complete.
2. **Scalar scoring:** retain default-off. The32-query identical-posting result preserves conditional Recall30 with16.83x fewer original inputs, but ordered warming fails the repeatable end-to-end gate. New warm-only ABBA/BAAB test records conditional Recall10/30, task CPU ticks, faults and wall nanoseconds, with score assertions outside timing. Release compilation/measurement status is separate; no warm result is called cold or collection-level timing.
3. **Routing:** T2 remains an experimental quality candidate, not a proven latency winner. No new heuristic or oracle inference policy.
4. **Bottlenecks:** high-recall bucket expansion and original scoring remain supported concerns. Smaller outputs help agreement but lose selectivity. Cold physical I/O remains distinct from warm scorer cost.
5. **Resources:**548 router2.70MB vs2562.10MB; build-duration comparisons are confounded. Quantization persisted bytes/RSS and repeated collection-level work remain unmeasured.
6. **Scaling:** no1M or10M gate passes. A validation/test-separated bounded300K study is reasonable only after finalized factorial and repeated scorer plus compressed-publication/recovery gates. Do not launch it here.
7. **Before10M:** correct counters, artifact generation compatibility, missing/stale metadata, concurrent publication, reopen and exact fallback tests, and matched storage/query/oracle protocol.
8. **Rejected/promotions deferred:**128 high-recall policy, full-shortlist scalar, multiplying isolated gains and claims versus unmatched staticLMI/HNSW. Geometric scheduler unchanged; model reuse remains deferred.
