# Q4 results: snapshot parity seen from Qdrant

**Verdict: at batch 11 the publication schedule explains the cross-engine gap from the Qdrant side, as PostgreSQL
found from its side.**
- **The schedule effect.** Replaying Qdrant E1 at PostgreSQL's drain schedule (same live ids, same plan, same code)
  moves Qdrant's batch-11 Recall@30 at 50K from **91.35% to 67.19%**: S = −24.17 pp [−33.65, −15.31].
- **What is left.** The remaining difference to PostgreSQL's own E1 is **+1.67 pp [−8.44, +11.35]**.
- **The preregistered rule** (|G_b| ≤ 2 pp and |S| ≥ 10 pp at batch 11) holds.
- **Batch 10 is partly different** (see below).
- **Status.** This is a validation-panel diagnostic: 32 MQ validation queries. It changes no frozen result and makes
  no latency claim.

Frozen protocol: [Q4_PREREGISTRATION.md](Q4_PREREGISTRATION.md) at `50b0c6af0`. Arms (c) and (c′), with PostgreSQL's
routers, were added by Amendment A1 (frozen `fda10150c`, before their data). Their results are in §"Full
decomposition (arms c and c′)" below.

**Update 2026-11-28: the full decomposition.** Given the same topology and PostgreSQL's own fitted routers, Qdrant's
engine reproduces PostgreSQL's recall and vectors scored exactly at batches 10 and 11, and within 0.1 pp at batch 12.
- **The engine residual** is 0.0 pp at batches 10 and 11 and 0.1 pp at batch 12.
- **The remaining cross-engine gap is router fit.** That includes the batch-10 residual of +10.42 pp at Qdrant's
  schedule.

## Provenance
- **The plan.** S1, Phase 3 Shift, seed 20261121. Its 205 file hashes and batch ledger equal those of Qdrant job
  25859944. The corpus is the 1M prefix (`6e1c7237…`).
- **Qdrant arms:**
  - (a) is the archived E1 of job 25859944 (MetaCentrum, commit f370a0ef);
  - (a′) and (b) were replayed locally on one WSL machine with the frozen binary (`evidence/q4/binary.sha256`),
    using E1, `mq_val`, the 17-point R3 grid plus the gate-only 2M setting, `DLI_P3_MAX_BATCHES=13` and
    `DLI_P3_EVAL=10,11,12`;
  - (b) adds `DLI_P3_DRAIN_ALIGN=global`.
- **PostgreSQL.** `Research-Results/2026-11-26-pg-r3-followups/shift-topology` (its `ARTIFACTS.SHA256` verified).
  It holds per-query captures at 50K for both schedules.
- **Archive.** `Research-Results/2026-11-27-uniform-q4` (33 files, `STORAGE_SHA256.txt` verified). Derived results
  are in `evidence/q4/analysis.json` and `evidence/q4/rounding_check.json`.

## Gates
- **(b) reproduces PostgreSQL's schedule exactly.**
  - Its publication ledger equals all 117 PostgreSQL publications (target level and points).
  - At 10, 11 and 12, its live-ID digests and level populations equal PostgreSQL's captures. For example, batch 11
    is level 2 = 495,000, level 1 = 45,000 and buffer 304; batch 12 is level 3 = 550,000 and level 1 = 35,000.
- **(a′) reproduces the archive.** Its ledger, live digests and level populations equal job 25859944's.
  - **Router fingerprints differ.** The fits are host-dependent.
  - **The outcomes still agree.** Per-query recall at 50K is identical at all three snapshots, and scored work is
    identical at 10 and 11 (51,236 vs 51,403 at 12). So N = 0 within resolution.
- **The full-search known answer** (gate-only 2M setting) is 0.9969–0.9979, below 0.999.
  - **The cause.** Every miss is explained by stored-representation rounding: 3, 3 and 2 misses at 10, 11 and 12,
    the same in (a′) and (b). Each missing plan-gold id lies outside Qdrant's stored-vector top-30, or within its
    tie slack (`q4_rounding_check.py`, using the bit-exact stored model in `final_benchmark/fb_plan.py`).
  - **A caveat.** The preregistration said only to classify shortfalls "as in the DLI-best pilot". This explicit
    classifier was written after the shortfall was seen. Because it applies equally to both arms, it does not
    affect any contrast.

## Results (mean over 32 MQ validation queries, nominal 50K)
| Batch | Qdrant as run (a) | Qdrant replay (a′) | **Qdrant at PostgreSQL's schedule (b)** | PostgreSQL original | PostgreSQL aligned to Qdrant |
|---:|---:|---:|---:|---:|---:|
| 10 | 61.87% | 61.87% | **70.73%** | 71.46% | 72.29% |
| 11 | 91.35% | 91.35% | **67.19%** | 68.85% | 91.15% |
| 12 | 93.96% | 93.96% | **93.96%** | 92.60% | 92.92% |

Vectors scored at 50K:
| Batch | (a) | (a′) | (b) | PostgreSQL original |
|---:|---:|---:|---:|---:|
| 10 | 59,994 | 59,994 | 66,671 | 77,623 |
| 11 | 51,126 | 51,126 | 63,442 | 75,369 |
| 12 | 51,236 | 51,403 | 51,104 | 50,821 |

Paired contrasts (pp, 95% query-bootstrap CI, 10,000 resamples):
| Batch | Total gap PostgreSQL − (a) | Schedule S = (b) − (a′) | Remaining G_b = PostgreSQL − (b) | Mirror: PostgreSQL aligned − (a′) |
|---:|---|---|---|---|
| 10 | +9.58 [−1.15, +20.42] | +8.85 [+2.60, +15.21] | +0.73 [−10.00, +11.04] | +10.42 [+1.98, +19.69] |
| 11 | −22.50 [−31.98, −14.06] | **−24.17 [−33.65, −15.31]** | **+1.67 [−8.44, +11.35]** | −0.21 [−3.33, +3.13] |
| 12 | −1.35 [−4.27, +1.46] | 0.00 [−2.60, +2.50] | −1.35 [−4.38, +1.56] | −1.04 [−3.54, +1.35] |

## Full decomposition (arms c and c′; Amendment A1)
- **The run.** A local replay of (b) and (a′), each with snapshots 10–12 evaluated twice: natively, and with every
  learned level rebuilt using PostgreSQL's fitted router for that level.
  - **The routers.** Taken from `Q4_ROUTER_EXPORT.md` (`c68a606`); the export's `SHA256SUMS` was verified.
  - **The search.** Qdrant's engine, unchanged: mdn_code, the same budgets and the 32 mq_val queries.
  - **The binary.** `47210b96…1bd2`, commit `fda10150c`.
- **The archive.** `Research-Results/2026-11-28-uniform-q4c/` (58 files, `STORAGE_SHA256.txt` verified).
  `analysis.json` SHA-256 `a65f3ecc…94ee`, `rounding_check.json` `563172d0…89a7`.

### Gates (A1)
| Gate | Result |
|---|---|
| 1. LMI1 export → import → rebuild reproduces every posting (`lmi1_export_import_rebuild_reproduces_postings`) | pass |
| 2. Each of the 15 imported members carries the parsed router; on 1,000 stored points the forward pass equals an independent float64 evaluation, with the same argmax and argmax posting | pass (max abs logit difference 8.2e-6 ≤ 1e-4) |
| 3. The native (a′) and (b) rows of the new replays equal the archived Q4 local rows | pass (identical: 6 × 576 rows) |
| (b) ledger and snapshots equal PostgreSQL's; (a′) ledger equals the archive | pass |
| Full-search shortfalls of (c) and (c′) are stored-representation rounding | pass (0 unexplained) |
| Topology: (b) = PostgreSQL's schedule, (a′) = PostgreSQL's Qdrant-aligned schedule, level for level | pass (checked before data) |

### Results (Recall@30 at 50K, 32 queries; paired bootstrap, 10,000 resamples)
| Batch | (a′) Qdrant schedule | (c′) PostgreSQL routers there | PostgreSQL aligned | (b) PostgreSQL schedule | (c) PostgreSQL routers there | PostgreSQL original |
|---|---|---|---|---|---|---|
| 10 | 61.87% | 72.29% | 72.29% | 70.73% | 71.46% | 71.46% |
| 11 | 91.35% | 91.04% | 91.15% | 67.19% | 68.85% | 68.85% |
| 12 | 93.96% | 92.81% | 92.92% | 93.96% | 92.50% | 92.60% |

| Batch | F′ = (c′) − (a′), router fit | E′ = aligned − (c′), engine | F = (c) − (b), router fit | E = original − (c), engine |
|---|---|---|---|---|
| 10 | **+10.42 pp [+1.87, +20.00]** | 0.00 [0.00, 0.00] | +0.73 [−10.31, +11.67] | 0.00 [0.00, 0.00] |
| 11 | −0.31 [−3.44, +3.02] | +0.10 [0.00, +0.31] | +1.67 [−8.23, +11.46] | 0.00 [0.00, 0.00] |
| 12 | −1.15 [−3.75, +1.25] | +0.10 [0.00, +0.31] | −1.46 [−4.58, +1.46] | +0.10 [0.00, +0.31] |

- **Vectors scored at 50K.** (c) scores exactly PostgreSQL's mean work: 77,623, 75,369 and 50,821 at batches 10–12.
  (b) scores 66,671, 63,442 and 51,104.
- **The full chain at batch 11, PostgreSQL original − (a):**
  - total = −22.50 pp;
  - S (schedule) = −24.17;
  - F (router fit) = +1.67;
  - E (engine) = 0.00;
  - N (host and fit noise) = 0.00.
  - Within rounding, −22.50 = S + F + E + N. That is an identity of the arm differences, not a model.

### Interpretation
- **Engine.** With identical topology and identical routers, the two DLI integrations give the same answers: at
  most 0.1 pp, one query differing by one id. Bucket selection, budget accounting and scoring therefore agree to
  the query.
  - **Preprocessing.** This also shows the preprocessing difference declared in A1 (halfvec normalization in
    PostgreSQL, Qdrant's stored Float16) is negligible here.
- **Batch 10's residual is router fit.** The +10.42 pp at Qdrant's topology is entirely F′. PostgreSQL's level
  routers fitted on that population route mq_val better than Qdrant's own fits do, and E′ = 0.
  - **Interval.** On 32 queries the interval is wide, [+1.87, +20.00], but it excludes zero.
- **At PostgreSQL's schedule, router fit is small and uncertain** (F = +0.73 and +1.67 pp at batches 10–11, both
  intervals spanning zero). So the batch-11 gap is schedule, as reported above.
- **Implication for the thesis.**
  - **Engine and schedule.** Integration-level differences between Qdrant and PostgreSQL vanish once the
    publication schedule is matched and routers are shared. What remains at a snapshot is router fit:
    training-sample and initialisation differences between two fits of the same architecture on the same level.
  - **Variance.** Fit-to-fit variance of a level router can reach about 10 pp on a 32-query panel at a single
    snapshot, so single-snapshot cross-engine comparisons need either shared routers or averaging over fits.
- **Limits.**
  - **The panel.** It is one validation panel of 32 queries, at three snapshots of one Shift stream.
  - **What F measures.** F and F′ compare one PostgreSQL fit with one Qdrant fit. They measure the difference
    between these two fits, not a systematic quality difference between the engines' trainers. No logits for an
    agreed query were exported; gate 2 checks the import only.

## Interpretation
- **Batch 11.** Both directions of the replay now agree.
  - PostgreSQL moved to Qdrant's schedule closes the gap to −0.21 pp; that is PostgreSQL's 0.21 pp, reproduced here
    as the mirror contrast.
  - Qdrant moved to PostgreSQL's schedule loses 24 pp and lands within 1.7 pp of PostgreSQL.
  - **The mechanism.** It is the moment of level-3 promotion. Qdrant's per-batch drains of 5,025 points cross the
    500,000 level-2 capacity during batch 11 (500,279) and publish a freshly trained level 3. PostgreSQL's exact
    5,000-point drains fit exactly 500,000 and promote only in batch 12. Until then, a reused level-2 router serves
    495,000 drifted points, and recall falls in either engine.
- **Batch 10.** The schedule effect is again large (+8.85 pp). At PostgreSQL's schedule, the engines agree
  (+0.73 pp).
  - **But at Qdrant's schedule they do not.** PostgreSQL aligned − (a′) = +10.42 pp [+1.98, +19.69]. That topology is
    level 1 = 45,026 and level 2 = 450,228.
  - So the batch-10 residual PostgreSQL reported is real, and it depends on the topology. Arm (c′) (below) shows that
    it is router fit (F′ = +10.42 pp), with an engine residual of 0.00.
- **Batch 12.** After both engines have published level 3, they agree within about 1.4 pp.

## Limitations
- **The panel.** There are 32 validation queries, so the intervals are wide (about ±10 pp for G_b at batches 10–11).
  "Within 2 pp" refers to the point estimate, as preregistered, not to an equivalence test.
- **The routers.** In (a)–(b) they are each engine's own fits. The router-fit and engine split is in the full
  decomposition above.
- **The budget.** Only nominal 50K was compared across engines, because PostgreSQL captured 50K only. Qdrant's
  full grid is in the archive.

## Thesis meaning
- **The algorithm.** Snapshot-level disagreement between two independently built DLI integrations is dominated, at
  the decisive snapshot, by *when* each engine's compaction promotes a level. It is not dominated by how either
  engine routes.
- **The cause of that timing.** It comes from a detail of buffer draining (per-batch versus global 5,000-point
  drains), not from the algorithm.
- **What to compare.** Cross-engine comparisons should therefore be made at matched publication schedules, or
  averaged over a stream, never at a single snapshot near a level boundary.
