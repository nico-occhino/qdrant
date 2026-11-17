# Optimized matched-work frozen10M study — continuation27

Starting source dfde295c50ab9c707a3273a5bdb2dff725100acb; retained geometric corpus10120191, same official2023 query/gold assets. No ingestion, model training or physical conversion.

## Safety gate

`materialize_topk` keeps only k+offset possible winners per selected physical member, after native deletion/deferred filtering. Every retained identity/version is still checked against complete captured membership. If a possible winner is shadowed, duplicated or ambiguous, the dispatcher falls back over the complete native view; it does not discard that winner and silently return a shorter set. A new dispatch check also detects an insufficient selected population after deleted posting entries consume the bucket budget, and uses complete native exact fallback.

Once all shortlisted identities are uniquely visible, any discarded member-local candidate has at least k preceding local candidates, so cannot be required for global top-k (equal-score ties retain native ANN tie tolerance). If the shortlist underfills k while the captured population can supply more hits, exact fallback closes the gap. This proof relies on unchanged native guards and complete physical membership; proxy/deferred/generation ambiguity remains fail closed.

| Control | Evidence |
|---|---|
| Multiple high-score deletions | Native deletion filtering before bounded heap; live oracle comparison for full/bounded modes |
| Superseded copies/newer Plain excluded by ANN | Complete membership lookup detects ambiguity for full/bounded modes |
| Three physical versions | Constructed retained preimages; exact fallback, no invented numeric-version winner |
| Delete/reinsert | Native mapping deletion followed by Plain insertion; correct live versions/scores |
| Deferred state | Native segment constructed with deferred offset; fallback in both modes |
| Shortlist underfill | Ordinary dispatcher selects deleted bucket while three live Plain vectors exist; exact returns all three |
| Update, proxy, publication, cancellation/stale membership | Existing bounded interleavings now run top-k; native publication/recovery controls retained |

Full targeted suite24 PASS/4 ignored/0 FAIL in `continuation-27/safety-ready.log`. Earlier compiler errors (missing trait import, dynamic versus original guard in cache control) are preserved in safety-first.log, study-compile.log and safety-final.log, and corrected. Old full materialization remains default control. No unsafe state is enabled to improve recall.

## Predeclared protocol

Optimized Cargo `--release`, existing target/environment, no parallel heavy build. Validation rows100..111 and held-out rows200..223 by default, disjoint from previous exploratory rows0..99. Each policy uses native validation candidates10K/20K/30K; selection chooses closest measured mean scoring30K and closest validation Recall10=.60, separately per policy. Choices are locked to disk before any held-out query. If targets cannot be matched within tolerance, analysis must say so, never report unmatched service quality as an economic win.

Policies: preserved slot weighting; reserved highest-probability bucket per learned physical member; population-density ranking `P(bucket|q)*physical_posting_population/bucket_population`. Density does not train parameters or use test gold. All reuse identical native storage, routers, scorer, version resolution, hierarchy and publication/recovery. Policy affects ordering only; whole-bucket overshoot remains explicit.

Each selected test case attempts native cache clearing of this fixture's Cold vector storage, then runs first-touch and repeated passes. This is a **Cold-member cache-clear attempt**, not a claim that every Cached member/query is cold. Per-query major faults establish actual access behavior. No global host cache/security settings change. CPU counters are process clock ticks (CLK_TCK recorded), coarse on millisecond warm queries; native scorer task ticks/faults remain separately available. Phase timers are nested, not additive with broad scoring totals.

Independent materialization control uses slot20K on identical held-out rows: warmup then full/top-k/full/top-k. Ordinary query timing excludes post-query JSON serialization; native selected-plan/access trace overhead is included if enabled and must be qualified. Generic optimizers remain disabled in retained DLI configuration; no maintenance is requested.

The sandbox rejects user-systemd bus access (`Operation not permitted`). Independent survival of managed executions is **not guaranteed**; no nohup/setsid claim substitutes for a supported service. Bounded commands are run in the active session and must report actual completion or timeout. No benchmark results are claimed until artifacts exist.
