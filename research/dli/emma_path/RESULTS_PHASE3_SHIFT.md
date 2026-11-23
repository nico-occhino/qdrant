# Phase 3 results: distribution shift (primary, 1M)

**Status.**
- **Decided:** H5 (frozen rules, [PHASE3_PREREGISTRATION.md](PHASE3_PREREGISTRATION.md) §5, §9b).
- **Descriptive:** H12.
- **Pending:** Part B (insert-only), job 25859945, still running at this writing.

## Provenance

| Item | Value |
|---|---|
| Job | `25859944.pbs-m1.metacentrum.cz`, exit 0, about 24 h requested; finished 2026-11-23 11:54 UTC |
| Source | `f370a0ef0`; compiled source, PBS, planner and analysis identical to the freeze `6538a4483`. Binary `43a4d629…` |
| Data | the first 1M SISAP23 rows; 10 spherical-k-means clusters in nearest-chain order. The last cluster is withheld (OOD), with a 1% holdout per inserted cluster; 20 batches; seed 20261121 |
| Queries | MQ (fixed, from the last inserted cluster's holdout), IDS (resampled per batch from the inserted clusters' holdouts), OOD (withheld cluster); 1,000 test queries each |
| Arms | E1 (router reuse), E1R (always retrain), E1C (consolidation, ρ = 0.25); `mdn_code`, B* = 50,000 |
| Integrity | every batch matched the plan's live-set ledger (driver-asserted) |
| Evidence | remote `dli-emma-path-20261120/results/phase3-primary-shift-25859944…/` |

## Hypotheses (Shift-M = MQ test panel, mean over 20 batches of mean R@30 at B*)

| | Contrast | Δ (pp) | 95% CI | Outcome | Bonferroni |
|---|---|---|---|---|---|
| **H5** | E1R − E1 | **+11.44** | [11.20, 11.68] | **supported** (≥ 2.0, lower bound > 0) | supported |
| H12 (descriptive) | E1C − E1 | +3.32 | [3.15, 3.49] | descriptive | — |

Sommerová's analogous contrast (No Retrain against From Scratch, her Table 4.6) was 8.4 pp.

## Trajectory and cost

| Arm | MQ first / mid (batch 9) / last | MQ mean | IDS mean | OOD mean | Builds (reused) | Training s | Ingestion s | Points written |
|---|---|---|---|---|---|---|---|---|
| E1 | 0.997 / **0.618** / 0.810 | 0.805 | 0.881 | 0.784 | 183 (159) | 185 | 243 | 8.60M |
| E1R | 0.997 / 0.883 / 0.967 | **0.919** | **0.954** | **0.869** | 183 (0) | **716** | 784 | 8.60M |
| E1C | 0.997 / 0.617 / 0.916 | 0.838 | 0.900 | 0.824 | 183 (155) | 167 | 233 | 8.97M |

Final topology:
- E1 and E1R: L2 400K (100 buckets), L3 500K (1,000 buckets).
- E1C: L1 25K, L3 875K.

## Interpretation

1. **Router reuse is the DLI's weak point under drift.**
   - **Why:** Sommerová's design routes new points into an occupied level with that level's *existing* router. Under
     shift, the router was trained on earlier clusters, so it packs the new clusters' points into a few oversized
     buckets: the largest L2 bucket reaches 10–16× the mean, against about 2× with retraining. At a fixed budget,
     the query then buys only a few huge buckets (corrected 2026-11-23, see
     [SHIFT_MECHANISM.md](SHIFT_MECHANISM.md)). Whether a stale router also ranks buckets worse is not separated.
   - **Size:** mid-stream, E1 loses 26.5 pp against E1R on MQ at batch 9 (0.618 against 0.883); the peak gap is +27.5 pp at batch 8. Averaged over the stream it loses
     11.4 pp, and 7.3 pp (IDS) to 8.5 pp (OOD) on the other panels.
2. **The cost of the fix is training.** Always retraining costs 3.9× the training time (716 s against 185 s) and 3.2×
   the ingestion time, and writes the same number of points. Reuse's 16× training saving, measured under uniform
   data, is therefore paid for in recall as soon as the data drifts.
3. **Consolidation helps under shift (H12 was expected to be negative).** It recovers 3.3 pp on average and 10.6 pp
   by the last batch. This contradicts the preregistered expectation, and it is reported as found.
   - **Corrected 2026-11-23.** This note first explained the gain by E1C "retraining more often" on a "more recent
     sample". The ledgers show the opposite: E1C reuses 155 routers against E1's 159, and 97% of its final data sit
     under an *older* router (batch 11) than E1's recent data (batch 12).
   - **The real cause is granularity.** The gain is all after batch 11 (+3.14 of +3.32 pp), where consolidation moves
     the newest clusters into the 1,000-bucket level instead of a 100-bucket one (see
     [SHIFT_MECHANISM.md](SHIFT_MECHANISM.md)).
4. **This motivates the planned original contribution:** retrain only when drift is detected ("reuse until drift").
   E1 and E1R bound such a method from below and above: 0.805 / 185 s and 0.919 / 716 s.

## Limits

- One replicate; the CIs cover query sampling only.
- B* = 50K is a high-recall regime on a 1M collection.
- Only `mdn_code` was replayed; whether population-proportional allocation behaves differently under shift is open.
