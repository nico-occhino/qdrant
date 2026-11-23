# DLI-best: the frozen assembly rule

**Status: FROZEN, 2026-11-23 15:50 UTC, by the author's acceptance.**
- **Timing.** This was frozen before any result of the policy-replay study (`mdn_code` against `proportional`,
  jobs 25888669–25888673) existed. All five were still running, at about 40 minutes of walltime. The check job
  25886307 had shown batches 0–1 of one panel; that is disclosed in `research/dli/policy_replay/PBS_MANIFEST.json`.
- **Changes** after this point are dated amendments that say whether they came before or after data.

## Purpose

DLI-best is the single DLI configuration entered into the final benchmark against IVF, HNSW, E1, CLI and one CLI
variant. It is assembled from E1 (Sommerová's DLI, as specified in [PREREGISTRATION.md](PREREGISTRATION.md)) by
adding only components that preregistered results support.

## Rule

A component is **included** if both of these hold:
1. **Benefit.** Its preregistered contrast, in the workload it targets, has a 95% CI that excludes no effect in the
   beneficial direction: work ratio CI upper bound < 1, or recall-difference CI lower bound > 0. This holds whether
   or not that hypothesis reached its own effect-size bar.
2. **No harm.** In no other *decided* workload does its contrast show a 95% CI entirely in the harmful direction.

Further conditions:
- **Competing alternatives.** Where two components are alternatives for the same function (allocation policies),
  the one the relevant preregistered rule recommends is used. If no single policy is recommended, `mdn_code` stays.
- **Engineering cost.** It is reported, but it does not exclude a component that passes 1 and 2.
- **Interactions.** Components are combined as one configuration, and the combination is not tuned further.
  Interaction effects are measured in the final benchmark, not assumed.

## Applying the rule to what is decided (2026-11-23)

| Component | Contrast (source) | Benefit | Harm elsewhere | Decision |
|---|---|---|---|---|
| **Vacuum** at dead fraction > 0.2 (E1V) | H6, Constant-R: +1.20 pp [1.15, 1.24] ([RESULTS_PHASE3_DELETES.md](RESULTS_PHASE3_DELETES.md)) | yes | none decided (it fires only when there are deletes) | **include** |
| **Consolidation** ρ = 0.25 (E1C) | H9, insert-only: work 0.841 [0.831, 0.852] ([RESULTS_PHASE3_PARTB.md](RESULTS_PHASE3_PARTB.md)); H12, Shift: +3.32 pp [3.15, 3.49] | yes | none (it helps under shift too) | **include** |
| **Calibrated allocation** (E1L) | H10, insert-only: work 0.929 [0.921, 0.936] | yes | none decided | **include**. Its cost is a per-level calibration panel and isotonic fit, which is reported |
| Greedy rebuild (E1G) | H6b: +0.11 pp [0.08, 0.15]; H7: equivalent | yes (H6b, tiny) | none | **superseded by vacuum** for the same function (delete handling): vacuum's effect is 10× larger and the two are alternatives |
| Always retrain (E1R) | H5: +11.44 pp under shift, +0.54 pp insert-only | yes | none in recall, but 3.6–3.9× training | **excluded as a component.** It is the upper-bound reference arm. The thesis's candidate original contribution, an imbalance-triggered retraining rule, is tested separately as its own arm |
| Allocation `mdn_code` vs `proportional` | policy replay (pending) | — | — | decided by that study's frozen recommendation rule. its interaction with calibrated allocation follows the frozen tie-break below |

**DLI-best as of now:** E1 + vacuum (> 0.2 dead) + consolidation (ρ = 0.25) + calibrated allocation. The allocation
policy can change only through the frozen tie-break below.

## Allocation tie-break (FROZEN 2026-11-23 15:52 UTC, author accepted, before any policy-replay result)

The author accepted the rule above. One case it does not settle: if the policy replay recommends `proportional`
over `mdn_code`, should DLI-best use calibrated or proportional allocation? No study compares calibrated allocation
with proportional, and calibrated allocation was tested on insert-only only.

**Rule.** Calibrated allocation stays, because it has a supported contrast. If the policy replay recommends
`proportional`, a calibrated-vs-proportional replay on the insert-only snapshots is run as a short, separately
preregistered check before the final benchmark, and its frozen rule decides. The author accepted this while all
five policy-replay jobs were still running (state R at 15:52 UTC).

## Notes

- **Calibration and consolidation have not been tested together under shift.** The insert-only combination E1CL
  gave 0.799 (descriptive). The final benchmark reports DLI-best per workload against E1, so harm from the
  combination would be visible.
- **No component is re-selected** after the final benchmark's data exist.
