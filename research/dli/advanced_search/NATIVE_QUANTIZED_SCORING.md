# Native quantized candidate scoring — continuation 30

Starting revision: 2a8d78749. Source changes: `lib/segment/src/index/candidate_scoring.rs`, `lib/segment/tests/candidate_scoring_v120.rs`. Raw logs: `/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-30/`.

## Implemented native slice

`score_candidates_generic` now delegates to `score_candidates_with_quantization`, defaulting to None. The explicit optional `QuantizedVectors` argument reaches Qdrant's `BatchFilteredSearcher`. Its native raw-scorer constructor selects quantized storage when present; otherwise it uses original vector storage. Existing DLI calls pass None and remain unchanged. This explains original-vector access despite a potential quantized representation. Offsets are unchanged native segment offsets, sorted/deduplicated and bounds checked. Native vector/point deletion plus VisibleOnly deferred mapping filters run before quantized scoring. This is a caller-pinned physical-generation seam, not an automatic persistent DLI activation.

Scalar quantization was built with native `QuantizedVectors::create`, immutable storage and native configuration. No custom compressed format was added. Native load/missing-artifact and deletion controls are included. Original reranking uses the existing original candidate scorer; approximate scores are not exposed as exact scores.

## Actual native control

Disposable 65,536-point Cosine segment, prepared SISAP subset from continuation28, four official query rows800–803. This fixture uses Float32 native vector storage (not the retained Float1610M backend), no new LMI training, and a complete segment-offset candidate set rather than selected learned posting lists. Scalar construction: **1223ms**, debug build. Every reranked hit is independently rescored through the original native scorer and exact score equality asserted. Full-shortlist reranking equals full scoring.

|Quantized offsets/query|Original shortlist|Recall@30 versus native original control|Combined debug time range|
|---:|---:|---:|---:|
|65536|128|1.0 on all4|27–30ms|
|65536|512|1.0 on all4|29–32ms|
|65536|2048|1.0 on all4|37–43ms|
|65536|65536|1.0 on all4|300–305ms|

Shortlist size bounds originals passed to the rerank scorer; this fixture also performs additional per-returned-hit exact assertions outside that algorithmic count. No page-fault or bytes-read counters were captured, and no cold/release claim follows. These figures compare two-stage shortlist sizes, not timed S0 versus S2. S1 approximate-score ranking is executed internally but separate S1 recall has not yet been reported. Four queries do not establish .90/.95 population-level targets.

## Correctness scope

Small native test quantizes real storage, reopens quantized artifacts, verifies absent scalar metadata returns None, compares full-shortlist exact outputs, deletes the highest original candidate and verifies the quantized result excludes it and fills three hits. Existing candidate scorer regression tests cover bounds, duplicates, deferred visibility and deletion. Cross-segment versions, newer Plain, concurrent optimizer publication and stale artifact generation have **not** been tested through this new optional path. It is therefore not enabled in collection dispatch. No compressed scorer may bypass DLI's complete-member version resolver or underfill fallback.

## Retained hierarchy / decisions

All five active physical member segment configurations at generation41 have `quantization_config: null`, including the6M Cold member. No artifacts were backfilled or baseline mutated. Available resources observed:20GiB available RAM,495GiB free disk. Immutable native quantization can be constructed from native vector storage, but safe disposable-copy/backfill and persisted configuration need validation before performing a10M job. No large backfill launched.

1. **Can DLI use native quantization with posting offsets?** Native scorer accepts those same segment offsets; this control demonstrates explicit offset scoring. Actual learned-posting/collection activation remains untested.
2. **Page faults/materialization reduced?** Not measured. Shortlist reduces original scorer inputs; materialization was already bounded.
3. **Original evaluations at .90/.95?**128 sufficed on these four small-control queries; no reliable10M target established.
4. **Memory/maintenance overhead?** Native construction measured1.223s for this65K control. Disk/RSS and production maintenance overhead not yet measured.
5. **More efficient?** Shortlist control is promising; no release-mode S0/S2 or controlled-cold comparison establishes end-to-end advantage yet.

Reproduce with validated environment and existing target:

```bash
cargo test --locked --offline -p segment --test candidate_scoring_v120 --features 'testing,lmi-training,lmi-research' -- --nocapture
DLI_TEACHER_CONTROL=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-28/teacher-control cargo test --locked --offline -p segment --test candidate_scoring_v120 --features 'testing,lmi-training,lmi-research' native_quantized_sisap_subset -- --ignored --exact --nocapture
```

Initial command incorrectly requested collection-only `dli-research` from segment; corrected to segment's `lmi-research`, failure preserved. No correctness protections removed and no new persistent index format introduced.
