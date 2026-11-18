# Amendment 01 to PREREGISTRATION_TIER1.md: finalist selection on validation

**Written:** 2026-11-18 14:05 CET. Tier 1 was still running (B4096-converged and the three B=1024 variants unfinished).

**Not preregistered.** The original preregistration stands unchanged. This document amends it after the fact.

## Problem

The original decision rules D1–D5 and the promotion rule compute W_target on the **test** split (rows 4000–5999) and use it to choose Tier 2 finalists. That lets test data influence model selection, so test is no longer a clean final evaluation.

## Disclosure

Before this amendment was written, an interim `analyze_tier1.py` printout showed test-set W_target values and paired ratios. The runs it covered were B64, B256 and B1024 (native and converged) and B4096-native; the scratchpad output is not kept as evidence. Its qualitative pattern was this: IVF needed less work than native LMI at every B, by a growing margin, and the centroid router was never worse than the MLP. That knowledge cannot be undone. This amendment limits its influence by making selection mechanical, from validation only, under the rules below.

## Amended procedure

1. **Selection set: validation** (rows 1000–2999, 2,000 queries). Apply D1–D5 and the promotion rule exactly as written, but compute W_target, the routing-equivalent totals and the paired bootstrap ratios on validation:
   `analyze_tier1.py --eval val --val val`
2. **Evaluation set: test** (rows 4000–5999). Report every test metric for the chosen finalists and all Tier 1 configurations with the same estimands, as a confirmation, not as a selection input:
   `analyze_tier1.py --eval test --val val`
3. **Mechanical finalist rules (validation)**
   - **F1, native baseline:** the native configuration at 10M is fixed in advance (accepted 10M: B = 3,162, S = 250,000, H = 512, 30 epochs, 5 iterations). It is evaluated with all four partition × router combinations, so the IVF-at-equal-teacher ablation comes free.
   - **F2, best spherical IVF:** the centroid|centroid configuration with the smallest validation `W_total` (scored work plus routing vector-equivalents) at R@10 ≥ 0.95, subject to D3's construction cap.
   - **F3, best learned LMI:** the mlp|mlp configuration with the smallest validation `W_total` under the same rule.
   - **F4, simplification:** included only if D4 passes on validation.
4. **Scale mapping to 10M.** The preregistration does not specify how a 1M bucket count maps to 10M. Fixed now, before Tier 1 completes: keep the **mean bucket size** (N/B) and the **samples per bucket** (S/B) of the selected 1M configuration, with B rounded to a power of two. The native S limit (≤ 1,000,000) caps S.
5. **Disagreement.** If validation and test pick different finalists, both are reported. Tier 2 uses the validation choice.
