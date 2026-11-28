# Qdrant response to the PostgreSQL CLI spec review

- **Reviewed.** `CLI_SPEC_PG_REVIEW.md` and `CLI_PG_DESIGN.md` at PostgreSQL commit `e17cd31640dedb27e4221b6af18800c9879ecc00`
  (SHA-256 `d9ea672e…5679e` and `ee990f2b…0d4741`, verified), of nico-occhino/postgresql-learned-indexes, branch
  research/cli-postgresql-20261128.
- **The result.** Shared spec v2, [CLI_SHARED_SPEC.md](CLI_SHARED_SPEC.md), applies every ACCEPT below and the
  Qdrant counter-proposals. It is still a draft for the author's freeze.
- **The principle for the counters** is the preamble both sides accepted: the thesis where it specifies; otherwise
  `cli_ablation` as the deterministic reference, so that gate G5 (parity with `cli_ablation`) stays meaningful.
  Where neither specifies, one shared convention is chosen.

| Section | PostgreSQL amendment | Qdrant | Note |
|---|---|---|---|
| §1 | metric, stored-representation gold, visibility before budget and scoring, separate counters | **ACCEPT** | Qdrant counts live candidates only (id-tracker filter) and reports dead postings separately, as LMI does |
| §2 | B = max(2, ⌊2n / b_max⌋); S = min(n, ⌊B · b_max / 10⌋); reject S < B or S = n; fixed H_init | **ACCEPT** | identical to `initial_buckets` / `training_sample`; the rejection rule is added in Qdrant |
| §3 | the MVCC / VACUUM delete contract; deferred bucket removal; the idle empty state | **ACCEPT** | Qdrant has no older snapshots, so deferral never triggers there. Both engines report deferred removals (0 in Qdrant) |
| §4 cadence and §4.1 | after each committed plan batch; an empty insert set has no entropy; at most one path; both flags recorded | **ACCEPT** | |
| §4.2 | ascending-id order, replay quotas with remainder and redistribution, untouched assignments preserved, residual overflow reported | **ACCEPT** | |
| §4.3 k | k = max(1, ⌊B/2⌋) | **COUNTER: keep k = round_half_even(0.5 · B), clamped to [1, B]** | the thesis says only "50%"; `cli_ablation` uses `int(round(0.5 · n))` (`reclustering.py:213`). They differ for odd B (B = 7: 3 against 4) |
| §4.3 μ | the conventional median, averaging the two middle values | **COUNTER: keep the lower middle value** | the thesis does not specify; `cli_ablation` uses `torch.median`, which returns the lower middle (`reclustering.py:304`). The G5 run (`evidence/g5-dedrift-reference.txt`) reproduces it |
| §4.3 numerator | the sum of B1's bucket populations (vectors) | **ACCEPT** | already the rule: `l1_size` |
| §4.3 ties | lower stable id for both largest and smallest | **ACCEPT** for both engines | `torch.sort` makes no tie promise. G5 parity is therefore claimed only for distinct sizes, and Qdrant's smallest-tie rule changes from higher-id-first to lower-id |
| §4.3 sample | min(n, 100 × clusters) | **COUNTER: keep min(n, ⌊0.1 · b_max · clusters⌋)** | `cli_ablation`: `train_indices_clustering_ratio · b_max · n_new_clusters` (`reclustering.py:58`), eq. 2.2's ratio. It equals 100 × clusters at b_max = 1,000 |
| §4.3 donor | the argmax donor of the true k-means centroid | **ACCEPT** | a declared deviation: `cli_ablation` uses the sample-mean proxy, which is near-identical under spherical k-means |
| §4.3 label order, pinned k-means iterations, the B_last reset | | **ACCEPT** | iterations = 20, as in `cli_ablation` `utils.py:50–62` |
| §5 | the exact integer rule over all allocated buckets including empties; evaluated before splitting; no firing at live = 0 | **ACCEPT** | identical to `cli::trigger` (`DriftSignal::Imbalance`) |
| §6 | R = ⌊n / 20⌋ capped by the eligible population; pinned streams, orders, versions and threads | **ACCEPT** | Within an engine, determinism is required on a pinned runtime. Across engines, deterministic parts are compared exactly and numerical parts within a frozen tolerance; bit-equal training across tch and PyTorch is not claimed |
| §7 | membership and exact coverage checked separately; engine diagnostics | **ACCEPT** | |

## On the PostgreSQL design choices (for the author)
- **The PostgreSQL design itself.** One mutable relation; replacement pages for changed buckets plus a new LMI1 model;
  one transactional selector; Generic WAL for page images; training out of process with no database locks held.
- **The Qdrant view.** This is the same publication contract as Qdrant's §4: readers see all of G_t or all of
  G_t+1, and maintenance is not a logged data operation. **The Qdrant side recommends approval.**
- **The PostgreSQL-only deferral rule.** Bucket removal and publication are deferred while an invisible posting may
  still be needed by an older snapshot. It is a correctness requirement there, and it is consistent with §3 as
  accepted.

## Author decision (2026-11-28)
- **Approved:** the PostgreSQL CLI design at `e17cd3164` (`CLI_PG_DESIGN.md`, SHA-256 `ee990f2b…0d4741`). That is
  one mutable relation, replacement pages plus a new LMI1 model, one transactional selector, publication deferred
  while an invisible posting may still be needed by an older snapshot, and training out of process.
- **What it permits.** Native PostgreSQL CLI code may start. There is still no MetaCentrum job, no measured data and
  no freeze until the author freezes spec v2.

## Author amendment (2026-11-28): spec v3, replay = 2% of the live index
- **The rule.** R_t = ⌊0.02 · live_t⌋ at each maintenance, capped by the eligible population. It replaces
  ⌊n_init / 20⌋; see the spec, §0.
- **Why.** The Stage A centroid control (`RESULTS_CENTROID_CONTROL.md`).
- **Timing.** Set after that descriptive diagnostic, before any benchmark data or freeze. PostgreSQL's trainer must
  take R_t per maintenance from the live census.
