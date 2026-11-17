# Supported decision — continuation36

1. **Model:**548 outputs,32768samples,512hidden remains baseline. Fresh548 is byte-identical to historical control.256 remains a smaller-model ablation;128 rejected for high-recall scaling despite higher assignment agreement. Refined95% coverage work9825/7450/5827 for128/256/548. Training timing is confounded, not used for a claimed cost win.
2. **Scorer:** scalar128 passes warm kernel screening:1.825x median paired speed ratio,32/32 query wins, exact conditional top10/30 preserved,16.83x fewer original inputs. Default remains original; promotion waits for full collection, quantified memory and quantized concurrency/recovery.
3. **Routing:** retain slot baseline and T2density experimental option; no demonstrated combined end-to-end superiority.
4. **Safety implemented:** native physical path/dimension guard, foreign-generation object swap rejection, native builder quantization reopen, deletion/Plain reinsertion and complete resolution; unavailable quantization fails closed with explicit reasons.26regressions pass. This does not close corrupted-artifact or concurrent quantized-publication gates.
5. **Bottleneck/resource:** high-recall expansion/original access remains important; cold I/O is not represented by fault-free warm control.548 router2.70MB vs2562.10MB; persistent quantization bytes/RSS/maintenance still unmeasured.
6. **Scaling:**300K NOT launched; its explicit gates are incomplete. No1M/10M. Next justified work is quantized concurrent publication, persisted stale/missing artifacts and collection-level matched-recall/resource measurements on the65K control.
7. **Rejected:**128 high-recall scaling, full-shortlist scalar, assuming better assignment accuracy wins, multiplying speedups, and claims against unmatched static/HNSW.

All prior managed sessions completed; no experiment monitoring needed. Raw continuation36 holds exact binary/source identity and outputs. Geometric scheduler/full retraining unchanged.

Additional closure: quantization-enabled native proxy/publication/cancellation with upsert/delete passes; cancellation resumes actual learned quantized dispatch. Native segment missing/corrupt quantization metadata rejects compressed usage while original scoring remains usable after missing metadata reopen. Per-shard shortlist override is default0 and avoids unsafe process-env mutation in tests. Full26-regression suite passes. Still no300K: collection-level timed/resource/actual-scoring accounting, in-flight query spanning swap, and full collection corrupted-quantization recovery remain open.

## Continuation37 update

Closed correctness gaps: actual pinned learned quantized query across final native publication; persisted quantized LocalShard activation; four normal-insert publications; complete collection clean reopen; mixed compressed/uncompressed members; missing/corrupt/incompatible quantized metadata with full native exact fallback and0learned attribution. Native26-test regression passes. Added narrowly scoped derived-quantization degradation for LmiTrained only, preserving cancellation and other index load errors.

Quantization is now explicitly configurable per collection but remains defaultNone. Scalar128 is still an experimental warm-scorer candidate,548 baseline and T2 quality option retained. No release collection-level speedup, no full-pool A0–A3 winner and no65K compressed-memory/maintenance economics are claimed. Tiny16D quantized bytes463+343 do not extrapolate to768D.300K remains NOT RUN and its performance/resource gates incomplete. No1M/10M work or maintenance redesign.

Continuation38: generated exact65K pool oracle for32 previously used diagnostic queries (continuation38/oracle). This does not satisfy held-out, multi-level Collection performance, per-stage work or resource gates. Original scorer remains default; scalar remains experimental;300K remains NOT AUTHORIZED by the stated gates.

## Continuation40 decision — native 65K Collection S0/A2

1. **Topology and correctness:** Release run at `ceca71685743b7c3c66c57114faa3cde171d5f28` published four native generations into a final active topology of 48K level 2 (`6544fb04-e99b-41bc-85a9-8ebd16a006d3`), 16K level 1 (`178b666e-ae7d-41d9-b73a-fae9159899e7`), and 1,536 Plain (`f9ef5fd5-4a38-4238-b7fa-09f2e385e6da`). Exactly 65,536 unique live IDs matched the full-pool oracle. All measured requests used learned routing, zero exact fallback, and S0/S2 plans were identical. Reopen took 3.396 s and resumed learned routing without fallback.
2. **Scorer decision:** Across 64 paired queries, S0 and scalar128 + exact rerank both achieved full-pool Recall@10/30 = .9922/.9766. A2 Collection search p50/p95 was 5.441/7.157 ms versus S0 8.058/10.371 ms; median paired ratio 1.483, A2 faster for all 64 query medians. Conditional result overlap was 1.0. Keep scalar128 as the leading opt-in scorer for further validation; original scoring remains the default pending provenance and resource gates.
3. **Work/resource:** Learned original evaluations fell from 8,513 to 256/query; A2 performed 8,513 quantized evaluations plus 256 original reranks, while Plain's 1,536 points used original scoring in both modes. Quantized artifacts total 49,408,603 bytes; model/posting files 5,664,508 bytes; raw Float16 vectors 100,663,296 bytes. Four in-builder quantization phases summed 0.646 s within a 143.723 s native job-wall sum. Whole-process peak RSS was 914,536 KiB and is not incremental quantization RAM. Use corrected recursive storage values in `continuation-40/multilevel-native-retry1/analysis.json`.
4. **Routing/model:** A0/A2 isolate scoring only. A1/A3 T=2 routing, adaptive buckets, model changes and static-LMI/HNSW comparisons were not run or inferred.
5. **Next gate:** Do not launch 300K yet. The 32 nominal test rows were not used in recorded continuations 26–38, but exclusion from earlier work is unverified. Select/document a truly disjoint query test range and regenerate its 65K exact oracle before promotion. Measure incremental quantization RSS and a matched no-quant build before claiming full economic break-even. No 1M/10M campaign.
6. **Execution:** The preserved 20K fixture failed before search because native scheduling left only levels 0 and 2. The corrected 16K schedule passed the release Collection test in 156.83 s after binary build. Focused tests: recursive storage accounting 1/1 and native DLI routing/visibility 13/13. Commits: `ceca71685` fixture correction; `b8d4b9157` recursive byte accounting and profile cleanup.

## Continuation41 decision — independent audit

1. **Continuation40 stands** within its stated limits. Its latency pass ran with profiling disabled, and the request accounting defects found by the audit (early outcome label, guard-held I/O, interleaved concurrent JSONL, static-path feature coupling) did not touch its measurements. No rerun.
2. **Per-request accounting gate is now met** for learned, Plain, quantized, original and exact-fallback work, with one final outcome per request.
3. **Semantics changes to carry into future runs:**
   - Missing derived quantization now degrades the affected member to original scoring (counted) instead of complete exact fallback.
   - DLI activation requires `max_optimization_threads = 0`.
   - Static LMI search is unprofiled in `dli-research` builds.
   - Static-vs-DLI timing must still be compiled from one source with identical features.
4. **Next experimental gate (proposed, not run):** isolate candidate *selection* from candidate *scoring* on the existing 65K topology. Build a recall-versus-work frontier over `candidate_budget` for slot-weighted and T=2 routing. Scoring is held fixed (A0 and A2) and judged against the full-pool oracle and a static-LMI / IVF-style flat-partition baseline at matched candidate counts. This is offline-replayable from saved postings first, before any native run.
   - Prerequisites: a demonstrably disjoint held-out query range with its own oracle; incremental quantization RSS; a matched no-quantization build-cost control.
   - No 300K/1M/10M ingestion until the frontier shows algorithmic competitiveness at matched recall.
