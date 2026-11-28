# Phase 3 preflight (job 25842074): pipeline passes, two design flaws found

Job `25842074.pbs-m1.metacentrum.cz`, node elmu1-25, exit 0, 58:42. Commit `f5b8e3235c8b`,
binary `52f1c318…`. N = 100K, 3 batches. Runs: Shift E1/E1R, Constant-R E1/E1G, Constant-O E1/E1G;
1,000 test queries; budgets 1K/4K/16K.

Evidence: remote `results/phase3-preflight-all-25842074…/`; local
`Research-Results/2026-11-20-dli-emma-path/phase3-preflight-25842074/`.
`SHA256SUMS` verify, except `job.log` (the known `tee` final-line case, verified with the last line removed).
Recall values are not interpreted (§6).

## Pipeline: pass

- All 7 Rust tests and the analysis known-answer test pass in-job.
- Every batch of every run matched the plan's live count and live-set SHA-256.
- Every publication succeeded, and all three plans were generated with gold.
- The frontier replay and the static references ran, and the analysis completed.

## Finding 1: B* = 0.5% of N makes the buffer swallow the primary budget

The precision check returned a per-query SD of **0.0** for all three contrasts. In Shift, the arms do
differ:
- E1 and E1R have different level-1 routers;
- 964 of 1,000 queries differ at 4K, and 814 at 16K; mean R@30 at batch 0 is 0.871 vs 0.922 at 16K.

At B* (500, rounded to the grid's 1K), though, the buffer holds about 2,300 points, scores 1 under
`mdn_code`, and alone exceeds the budget. Both arms then return the same buffer-only result. The same
holds at 1M, where B* = 5,000 = b. Sommerová's B* = 50,000 at 10M was **10·b**, so the buffer was at most
10% of the budget. The 0.5%-of-N rule matched that ratio only at her scale.

## Finding 2: Greedy rebuild almost never fires in the Constant scenarios

At 100K, the live set (10K) fits in level 1 above Cmin(1), so E1 and E1G are the same index. In a
planner-arithmetic simulation of the 1M workloads (live counts, Bentley–Saxe, Greedy when no
compaction is due):

| Scenario | E1G rebuilds | Final levels E1 → E1G |
|---|---|---|
| Heavy-R | 4 | L1 10K, L2 426K, L3 474K → L1 45K, L2 220K, L3 645K |
| Medium-R | 4 | L1 5K, L2 102K, L3 443K → L1 43K, L3 507K |
| **Constant-R** (H6) | **1** | identical (L1 31K, L2 69K) |
| Heavy-O | 1 | different |
| Medium-O | 1 | L3 only |
| **Constant-O** (H7) | **0** | identical |

So H6 would compare two near-identical indexes, and H7 would be "equivalent" by construction.

**The mechanism is Qdrant-specific.** Her rule triggers on *live* occupancy (live < Cmin). With her hard
deletes, live equals physical. In Qdrant, deleted points stay in their segment and their postings stay
in the buckets. A level can therefore keep a healthy live count while most of its postings are dead.
The preflight shows exactly that in Constant-R: at budget 16K, about 13.7K of the selected postings are
dead and 4.5K are scored, so **≈ 75% of the budget buys nothing**. Greedy cannot see this. A rule that
triggers on the **dead fraction** of a level can: Qdrant's own vacuum optimizer uses
`deleted_threshold`, default 0.2.

## Decisions needed before freezing

Recorded as author decisions in [PHASE3_PREREGISTRATION.md](PHASE3_PREREGISTRATION.md) §9b (2026-11-22):
- B* = 10·b = 50K;
- the new vacuum arm E1V (dead fraction > 0.2);
- H6 = E1V vs E1 in Constant-R, H6b = Greedy in Medium-R, H7 in Medium-O;
- a 1M precision preflight with a different seed.
