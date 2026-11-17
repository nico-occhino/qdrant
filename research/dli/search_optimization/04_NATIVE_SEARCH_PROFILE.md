# Native scorer and resolver profile

New named counters split router inference, global bucket planning, native scorer execution and full-hit materialization. Existing broad scoring includes preparation, sorting, scorer and materialization; resolution includes membership-wide visibility and aggregation. These counters are nested; do not sum broad scoring with its subphases. Read-guard time includes hook pauses in deterministic tests and must not be treated as production contention.

The new native maintenance `native_phases` additionally separates copy, populate, permit wait, segment build and first propagation. These are wall durations, not CPU time. No new maintenance run is needed for the frozen search experiment.

A bounded reopen diagnostic supports `DLI_PROFILE_QUERY_COUNT` and two identical rounds, or `DLI_PROFILE_COMPARE_TOPK=1` for four alternating full/top-k rounds. It uses native Collection::load/search/stop, no insertion or model training. The initial attempt failed at the missing ambient hardware scope, before completing queries; `frozen-debug-profile.log` preserves the failure. The corrected test establishes an ambient scope. Debug measurements are mechanism diagnostics, not credible optimized latency benchmarks.

Opt-in `materialize_topk` scores all selected vectors but asks each native scorer for only `limit+offset` winners. A discarded local candidate has at least that many local candidates ahead of it. If those possible winners are unique and native-visible, it cannot enter the global requested result. Every retained candidate is still checked against every captured member; duplicated/shadowed/deferred/unsupported winners still trigger complete exact fallback. Thus this is not candidate truncation before scoring and does not remove the newer-Plain-copy safety check. Default remains false pending measurement and regressions. The routing-only primitive retains the full-hit behavior for controls.

Correctness checks compare bounded/full native output on a trained two-level fixture with deletion and Plain reinsertion. The configured runtime fixture exercises both reserved routing and bounded materialization through four publications, clean reopen, corrupt-derived-metadata fallback and reconstruction. Final logs/results are documented in05 after execution.

## Bounded native result — debug mechanism control

Frozen reopen passed1/1,103.89s total including loading. Four alternating rounds of the same three official queries (rows0–2), all12 learned/zero fallback. Each round scores58,610 vectors, evaluates12 models, Recall@10=.333333 and Recall@30=.388889. These are not representative100-query quality estimates.

|Round / materialization|Query total s|Native scorer s|Hit materialization s|Resolver s|Router s|
|---|---:|---:|---:|---:|---:|
|0 / full|24.1894|23.1649|0.2334|0.6613|0.1153|
|1 / top-k|0.5607|0.4223|0.0010|0.0017|0.1201|
|2 / full|1.2603|0.3155|0.1518|0.6524|0.1265|
|3 / top-k|0.4049|0.2930|0.0011|0.0021|0.0965|

The same full-materialization path changes from23.1649s native scoring on first pass to.3155s on the later pass. This proves strong order/cache-associated scorer sensitivity; it does not distinguish page faults, physical I/O, swap or concurrent resource activity, because CPU/I/O/fault counters were not recorded. It falsifies inference-dominates for these initial outliers. No cache was forcibly dropped.

For the subsequent full versus top-k passes, scorer work is identical and scorer duration similar (.3155/.2930s), while materialization changes .1518→.00114s and resolution .6524→.00214s. Query total1.2603→.4049s. This supports the intended allocation/lookup reduction without a recall change on this bounded control. It is an unoptimized, three-query, nonrandomized observation; no release-mode speedup or10M-wide latency claim follows. Data: continuation26/native-phase-deltas.csv and frozen-debug-profile-fixed/{rounds.json}, test log frozen-debug-profile-fixed.log. Source state matches fab3ee7bb code, committed while this bounded process was running.

## Fault/CPU discrimination — official rows20–22

The opt-in `DLI_NATIVE_ACCESS_EVENTS` samples `/proc/thread-self/stat` around candidate preparation/native scoring on the actual search worker. CPU is in clock ticks (`getconf CLK_TCK`=100 here), fault counts are task-local. It emits physical UUID, persisted backend, returned-offset count and wall nanoseconds. It is disabled unless requested. This is detailed diagnostic instrumentation; production phase timers do not read procfs.

|Round|Returned offsets|Vectors scored|Native-access wall s|Thread CPU s|Major faults|
|---|---:|---:|---:|---:|---:|
|First full|61,565|61,565|18.9240|2.44|75,702|
|Top-k repeat|90|61,565|.3094|.30|0|
|Full repeat|61,565|61,565|.3063|.30|0|
|Top-k repeat|90|61,565|.2711|.27|0|

All12 phase records concern the6M Cold UUID. Individual first-pass queries take5.12/4.37/9.44s, with20,044/18,259/37,399 major faults. Repeated full scoring takes.3063s withzero major faults. Thus major-fault-backed native access is a demonstrated large outlier mechanism, while neural inference is outside the traced phase. These records do not identify every fault address/file or measure device bytes; not every fault can be assigned specifically to vector pages. Other non-CPU waits are not excluded. No cache dropping or global host settings were used.

Run PASS1/1,105.40s including reopen; regressions20 PASS/3 ignored/0 FAIL. Raw `continuation-26/access-events.jsonl`, `access-deltas.csv`, `frozen-access-debug/{rounds.json}`, `frozen-access-debug.log`; command `bash continuation-26/run-access-control.sh`. Exact source is the CPU/fault instrumentation patch following fab3ee7bb. Query offset is recorded in output. Units and nested phase definitions are deliberately explicit.

Native `SegmentReadViewFor` wraps the same regular Segment vector data and tracker used by this DLI path. Simply wrapping scoring in `with_view` would not automatically switch this original-segment fixture to an asynchronous read-only backend. No unsupported I/O backend change or parallel vector snapshot was introduced. A useful remaining access experiment is bounded native prefetch/batched I/O against this fault trace; forcing the whole corpus into RAM would change the storage experiment.

Continuation27 optimized phase data and explicit native Cold-cache attempts are now available in [07_RELEASE_RESULTS.md](07_RELEASE_RESULTS.md). The release study confirms materialization/resolver savings, while warm native scoring remains dominant. CPU-only warm tails must not be explained using earlier major-fault observations; the two regimes are separated in the new traces.
