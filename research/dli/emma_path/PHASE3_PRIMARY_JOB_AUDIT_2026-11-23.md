# Phase 3 primary job audit — 23 November 2026

## Completed Shift run

Job `25859944.pbs-m1.metacentrum.cz` finished on `haldan1` with PBS Exit 0 in 16:18:38 (9:30:06 CPU, 43.90 GiB PBS memory accounting of 128 GiB). The completion record pins source `f370a0ef097e80e5b76c6ffead778eb29dd9b5c4` and binary SHA-256 `43a4d6293756a2b68118cb466a2a3b48604e70fbcd990125a76568f865b1ef30`. The archived analysis JSON and text, Shift plan JSON and three batch ledgers pass their SHA-256 checks. Each ledger has 20 records. I independently matched every ledger's batch number, live count and live-set hash to the plan, then recomputed recall from 180,000 selected rows (3 arms × 20 batches × 3 panels × 1,000 queries). Mean MQ recall was 0.804805 / 0.919230 / 0.838005 for E1 / E1R / E1C. IDS means were 0.881378 / 0.954155 / 0.900302, and OOD means were 0.783655 / 0.869075 / 0.823617. The confidence intervals are from the archived analysis.

The plan uses a 1M-row SISAP23 source prefix, seed 20261121, and 20 ordered batches with held-out data. The primary comparison averages Recall@30 over the 1,000-query MQ panel and then over batches at the frozen 50,000 candidate budget with `mdn_code`. MQ is the fixed panel tied to the last inserted cluster. IDS samples from already inserted clusters. OOD is from the withheld cluster.

| Arm | Mean MQ R@30 | Batch 9 MQ R@30 | Final MQ R@30 | Training (s) | Ingestion (s) | Points written |
|---|---:|---:|---:|---:|---:|---:|
| E1, reuse | 0.805 | 0.618 | 0.810 | 185 | 243 | 8.60M |
| E1R, retrain | 0.919 | 0.883 | 0.967 | 716 | 784 | 8.60M |
| E1C, consolidate | 0.838 | 0.617 | 0.916 | 167 | 233 | 8.97M |

- **H5 supported.** E1R gained **11.44 percentage points** over E1 (95% CI **[11.20, 11.68] pp**), exceeding the frozen +2 pp threshold and passing the Bonferroni check. The largest illustrated separation is at batch 9, about 26.5 pp. The cost was approximately **3.9× training time** and **3.2× ingestion time**, with the same reported points written.
- **H12 descriptive.** E1C gained **3.32 pp** over E1 (95% CI **[3.15, 3.49] pp**), opposite the preregistered direction. Its final layout holds 875,482 live points in level 3, compared with 500,279 in E1. Consolidation changes both the layout and router training opportunities, so this comparison does not isolate either mechanism.
- The job records 180 streaming publications per arm. The Emma branch's existing result note counts 183 builds with initial builds included. The intervals describe query sampling in this one stream and seed. This experiment uses a 1M source prefix and a fixed candidate budget.

## Insert-only snapshot

Job `25859945.pbs-m1.metacentrum.cz` was still running on `haldan11` at the 23 November check (17:34:01 wall, 7:21:55 CPU and 25,048,348 KiB PBS memory accounting). The shared archive showed 21 batch ledger records each for E1 and E1R and 15 for E1C. The E1C batch 15 directory was present. The archive had no completion or aggregate analysis file at the latest listing, so insert-only outcomes remain pending.

## Separate Pilot 2 known-answer diagnosis

Commit `1a1e5253b` reports that all 205 regenerated Pilot 2 Shift plan files matched their archived hashes. Three D-only mismatches were the same MQ validation query in batches 3–5, with exact zero score margin at rank 30 and 88, 169 and 230 byte-identical live rows at the cutoff. A separate 40 shared misses were associated with storage rounding, and rescoring reproduced 39. This is evidence from the static-vs-DLI Pilot 2, a different run from Phase 3 Shift. The tie-aware gate is proposed and the frozen Phase 3 measurements retain their original scoring.

## Evidence

- [Shift PBS job](https://my.metacentrum.cz/jobs/25859944.pbs-m1.metacentrum.cz)
- [Insert-only PBS job](https://my.metacentrum.cz/jobs/25859945.pbs-m1.metacentrum.cz)
- Remote archive root `/storage/budejovice1/home/nico/lmi-metacentrum/dli-emma-path-20261120/results`
- Emma branch `research/dli-emma-path-20261120`, `emma_path/RESULTS_PHASE3_SHIFT.md` and frozen `emma_path/PHASE3_PREREGISTRATION.md`
