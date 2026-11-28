# CLI Stage A: centroid-routing control (results; descriptive)

**What this is.** The diagnostic of CLI_SHARED_SPEC.md §8 (author decision, 2026-11-28): the same CLI maintenance
(splitting, DEDRIFT, triggers) with two routings on the in-memory Stage A core.
- **MLP routing:** the CLI router, fine-tuned with stored-label replay.
- **Centroid routing:** the spherical k-means centroids, replaced at each split or recluster, with no training.
  Entropy is undefined for it, so its only drift trigger is growth (declared in `cli_core.rs`).

**Status.** Descriptive only. It is not a benchmark arm, it is not preregistered, and it is on validation panels.
It is Stage A: in memory, not Qdrant's segment lifecycle. Gold is the plan's Float64 gold over raw Float16 rows;
both routings score the same normalized f32 vectors.

## Setup
- **Code.** `research/cli-20261128` at `5b5606363` (`cli_core.rs`, `tests/cli_stage_a.rs`). The binary is
  `b6424ff9…`, the perf profile.
- **The analysis.** `analyze_control.py` with its known-answer test, committed at `2912cd847` before the results
  were read.
- **Plans.**
  - **I1:** Phase 3 Insert-only, seed 20261121, panel `val` (32 queries), 21 batches, a 100,000-point build.
  - **S1:** Phase 3 Shift, seed 20261121, panel `mq_val` (32 queries), 20 batches, a 45,025-point build.
- **Parameters.** b_max = 1,000 and spec v2 defaults: hidden 128; 30 build epochs and 20 fine-tune epochs; R =
  ⌊n_init / 20⌋, which is 5,000 on I1 and 2,251 on S1; seed 42; the entropy trigger at +0.3 bits; growth at
  10 × |B|_last.
- **Budgets.** The Phase 3 grid, 1K–320K.
- **The archive.** `Research-Results/2026-11-28-cli-centroid-control\` (17 files,
  `STORAGE_SHA256.txt` verified).
- **Gate G2.** `check_membership` passed inside every maintenance of all four runs: no run aborted.

## Results
**The ratio.** R = mean_t W@0.90(centroid) / mean_t W@0.90(MLP), from a batch-stratified paired query bootstrap,
10,000 resamples.

| Plan | R (centroid / MLP) | 95% CI | W@0.90 MLP, first → last batch | W@0.90 centroid, first → last | R@30 at 50K, last batch, MLP / centroid |
|---|---|---|---|---|---|
| I1 Insert-only | **0.508** | [0.470, 0.567] | 4,247 → 12,326 | 3,583 → 3,974 | 0.967 / 0.992 |
| S1 Shift (mq_val) | **0.534** | [0.489, 0.595] | 13,942 → 21,496 | 9,852 → 9,644 | 0.973 / 0.989 |

**Maintenance:**

| | I1 MLP | I1 centroid | S1 MLP | S1 centroid |
|---|---|---|---|---|
| Buckets split | 1,554 | 1,505 | 970 | 1,057 |
| Reclusters (trigger) | 0 | 0 | 3 (entropy, batches 3–5) | 1 (growth, batch 13) |
| Final buckets | 1,749 | 1,705 | 1,541 | 1,612 |
| Fine-tune seconds, total | 1,231 | 0 | 210 | 0 |
| L_route, mean / max | 0.33 / 0.42 | — | 0.40 / 0.50 | — |

## Reading
1. **Both routings split almost identically,** yet the centroid control reaches 0.90 with about half the vectors
   scored on both plans. The difference is the router, not the partition.
2. **The MLP degrades as it is fine-tuned.**
   - **I1.** Its W@0.90 triples over the stream (4.2K → 12.3K), while the centroid control stays flat at 3.6–5.2K.
   - **Forgetting.** Routing-accuracy loss on untouched points rises steadily, to 0.42 on I1 and 0.50 on S1. Up to
     half of the points that were never moved no longer have their own bucket as the router's top choice.
   - **Replay size.** The spec v2 replay R = ⌊n_init / 20⌋ is 2–3 vectors per bucket at about 1,500–1,700 buckets.
     The thesis's 5,000 at 300K is a larger fraction (1.7%) than ours at 1M (0.5%).
3. **It was already behind at build.** The MLP needed more work than centroids on the first batch too: 4,247 vs
   3,583 on I1 and 13,942 vs 9,852 on S1. This matches the 1M LMI-versus-IVF routing ablation.
4. **Triggers.**
   - **Entropy.** It fired only on Shift: three DEDRIFT reclusters in batches 3–5, at 1.77–1.84 bits against H_init
     = 1.37. On I1, batch entropy *fell* below H_init (1.01 → 0.79 bits) and never fired. Growth did not fire for the
     MLP on either plan.
   - **Recovery under Shift.** The MLP's work spiked to 58K at batch 4, recovered to about 22K after the reclusters,
     then drifted back up to about 37K.
   - **CLI-I's expectation (spec §5).** Before splitting, the largest bucket reached 20.2× the mean on S1 (MLP,
     batch 2) and 14.6× (centroid), but at most 3.6× on I1. The 8× imbalance rule would therefore have fired under
     Shift and not on Insert-only, as expected before data.
5. **What it suggests (not a result).** Under spec v2's parameters, the neural router adds cost, not value, beyond
   the maintenance structure. CLI's claimed advantage would have to come from settings this control did not vary:
   a larger replay, more fine-tune epochs, or a different b_max.

## Limits
- One seed, one b_max (1,000) and 32 validation queries per batch, on two plans.
- Stage A: in memory, with no segment lifecycle and no latency.
- The control's growth-only trigger is not the same drift policy as the MLP's entropy trigger.
- Raw-Float16 gold. The routings are compared with each other, not with the final-benchmark arms.
