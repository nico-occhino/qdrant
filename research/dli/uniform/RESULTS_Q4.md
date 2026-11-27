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

Frozen protocol: [Q4_PREREGISTRATION.md](Q4_PREREGISTRATION.md) at `50b0c6af0`. Arm (c), with PostgreSQL's routers,
is still blocked.

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
  - So the batch-10 residual PostgreSQL reported is real, and it depends on the topology. With these two
    measurements it cannot be split between router fit and engine. That split is what arm (c) is for.
- **Batch 12.** After both engines have published level 3, they agree within about 1.4 pp.

## Limitations
- **The panel.** There are 32 validation queries, so the intervals are wide (about ±10 pp for G_b at batches 10–11).
  "Within 2 pp" refers to the point estimate, as preregistered, not to an equivalence test.
- **The routers.** They are each engine's own fits. The router-fit and engine components (arm c) need PostgreSQL's
  router weights, which were archived only as hashes.
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
