# Frozen geometric 10M recall frontier — continuation 28

Source checkpoint: 7f3bfc066 (native controls); starting revision 4136d46f9.
Raw evidence: `<admin>/experiments/2026-11-12-dli-architecture/continuation-28/`.

## Completed evidence

Targeted native regression suite: **25 PASS, 6 ignored, zero failures** (`frontier-safety.log`). Release executable compiled successfully (`release-build.log`). Existing top-k deletion, deferred, duplicate/version, Plain shadow, proxy, publication, cancellation and recovery controls remain enabled. Temperature one equals density; cached and uncached priorities preserve complete ordering/work. No new 10M ingestion or alteration of retained postings/models.

Offline reconstruction uses the retained four-model geometric topology, ordered Float32 inference and exact saved postings/physical ID membership. Validation rows400–499 and held-out rows600–699 are distinct. Shared temperature is fitted on validation neighbor-bucket labels: T=2; NLL improves from2.240 to1.631. This is query relevance calibration, not teacher classification accuracy. Physical-population priors remain fixed in the native temperature control. An additional fitted-prior variant remains offline-only.

The table shows validation-selected smallest tested budgets meeting **both** bucket coverage targets. These are inclusion/work statistics, not native recall or latency. Whole-bucket selection can substantially overshoot nominal budget.

| Policy | Coverage target .80 | .90 | .95 |
|---|---:|---:|---:|
| Slot |200K|280K|500K|
| Reserved |200K|280K|500K|
| Density |90K|240K|400K|
| Temperature-density T2 |80K|180K|340K|

At the .80 selection, mean validation posting work is88,055 for T2 versus135,868 for density and approximately200K for slot. This supports testing calibration; it does not establish an end-to-end Pareto improvement. Full offline curves, logits, NLL, scalar/physical/fitted-prior controls and locked native choices are in `offline-refined/`.

Sommerová §3.5.1 (printed39–40) normalizes by arity-growing output counts. The pinned Plhák implementation similarly grows model output count in add_level; its zero-based search exponent differs by a common learned-model factor from the one-based text. Our fixed548 outputs invalidate importing that heuristic without calibration. Original scope assumptions, rather than the name of the algorithm, matter.

## Predeclared optimized execution

`run_recall_frontier.sh FRESH_OUTPUT offline-refined/native-plan.json teacher-control` runs:

1. Release safety gate, then12 validation-selected policy/budget cases,12 validation queries each,12 held-out first-touch queries and12 held-out warm repeats.
2. Five low-budget uncached/cached passes: rehearsal then ABBA. Same work, ID-score-version sets required.
3. Four full native exact controls against the entire10120191-point population. These deliberate fallbacks are separated from ANN fallback counts.
4. Only after query timing, one private65536-point training control (uniform seeded sample from6M member) captures actual native training labels/centers and sample offsets. A teacher/model placement × teacher/model query-ranking factorial uses a newly computed local exact oracle on32 distinct queries800–831. This does not recover the original6M teacher or measure full10M retraining benefit.

Default first-touch runs do not evict9GB between configurations, so they are **not uniformly cold**. ABBA is a paired cached-priority control; policy order/cache state remain limitations. Native oracle analysis verifies executed scope/bucket coverage, actual scores, versions and work. Matching is descriptive (work spread ≤10%; both recall differences ≤.03), never parameter tuning on held-out queries. Plots distinguish offline bucket coverage from actual native recall.

Full native frontier, exact ceiling, end-to-end priority benefit and teacher conclusions are **pending execution**. No static10M baseline was found in the explicitly searched local summary scope; preparation and resource/probe limitations are documented in09_STATIC_10M_COMPARISON_PLAN.md.

## Persistence and reproducibility

User-systemd access is denied in this sandbox. The launcher records local PID namespace, process ownership, exact source/binary/config/data evidence and explicitly sets independent_persistence_verified=false. A managed foreground session is not a verified detached service. Do not infer survival from background shell syntax. No competing full ingestion is launched.
