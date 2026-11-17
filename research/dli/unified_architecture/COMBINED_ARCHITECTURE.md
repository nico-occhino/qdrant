# Architecture comparison status — continuation36

A0 original scoring remains the safe default. A1 T2 is retained as an allocation-quality candidate, not a confirmed latency winner. A2 scalar128 is now a validated warm scorer-pipeline improvement on identical selected native postings (32-query counterbalanced control), with conditional results preserved. This is not a complete collection-level promotion. A3 combined calibrated multi-level allocation plus scalar scoring is not measured; do not multiply isolated gains. A4 256 outputs remains a memory ablation against548; smaller128 rejected for high-recall scaling. No new heuristic or maintenance redesign.

The strongest bounded execution candidate is548 plus scalar128 reranking; the best supported deployed research default remains548 plus original scoring, because quantized concurrent-publication/recovery, cold-I/O/resource and untouched validation/test gates remain open. No matched staticLMI/HNSW superiority is claimed.

Bounded quantized-native proxy/cancel/publication and segment reopen/missing/corrupt metadata controls now pass. They justify further65K integration, not default promotion or a combined T2/multi-level performance claim. Native final scores and visible populations remain authoritative.

Continuation37: explicit collection quantized-shortlist configuration and native construction/reopen are now implemented and exercised through normal updates. Native generation pinning through a barrier-forced final holder swap, mixed-member missing/corrupt/incompatible scalar recovery and exact-fallback attribution pass. Tiny multi-level lifecycle evidence does not supply an A0–A3 performance ranking. Keep original default and defer300K until full65K query/resource measurements.

Continuation38: full65K pool oracle generated and hashed for previously used32 diagnostic queries. It is not a new multi-level collection experiment or untouched test split. Separate per-query native quantized/original work and timing remain necessary for A0/A2 comparison; no new architecture winner is inferred.

## Continuation40 — first native multi-level Collection comparison

The 65K release Collection experiment now compares A0 and A2 on one immutable physical topology: two independently published learned members (16K level 1, 48K level 2), plus 1,536 Plain points. The original and scalar128 modes use identical per-query bucket plans and offsets. At fixed routing, both achieve full-pool Recall@10/30 of .9922/.9766 over 64 queries, with conditional top-k overlap 1.0. A2 reduces original learned scoring from 8,513 to 256 evaluations/query and reduces complete Collection search p50/p95 from 8.058/10.371 ms to 5.441/7.157 ms; per-query median latency favored A2 for 64/64 pairs. No query used exact fallback.

This is evidence for A2 on a warm 65K control, not a 10M result or a measured benefit at matched recall across independent test queries. Query rows 932–963 were not used in recorded continuations 26–38, but earlier-use exclusion is not proven. A2 therefore remains default-off. T=2 routing, A3, cold-I/O behavior, incremental RSS and matched no-quantization construction are unmeasured. Full results and per-query traces: `continuation-40/multilevel-native-retry1/analysis.json` and `per-query.csv`; exact run source: `ceca71685743b7c3c66c57114faa3cde171d5f28`.
