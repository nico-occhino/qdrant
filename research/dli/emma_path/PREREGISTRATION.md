# Preregistration — Sommerová-faithful DLI in native Qdrant ("Emma path")

Written 2026-11-20 before any implementation on branch `research/dli-emma-path-20261120`
(base `7f8e02891`). Amendments are appended; nothing above an amendment is edited after
results exist.

## 1. Motivation

At 10.12M (H-10M, job 25677858) the Qdrant DLI needed 45% more scored work than a static
LMI to reach test R@30 = 0.90, and the gap was the indivisible Plain buffer (120,191
vectors, priority 1.0). Sommerová's design (thesis §3; code `Emma_DLI.zip`,
`experiments/configs/index/bs_10_5k.yml`) keeps **bucket size ≈ buffer size b** by giving
level *i* exactly aⁱ buckets and capacity b·aⁱ, so a buffer priority of 1.0 costs about one
bucket. Our implementation broke that invariant (548 buckets on every level, b = 250K,
a = 2) and also retrains every compaction target from scratch, whereas her code routes data
into a non-empty target with the target's existing model (training only when the target
level is empty).

Question: does restoring her design in native Qdrant remove the Plain-buffer penalty and
reduce maintenance training, at matched candidate work?

## 2. Arms (only selection/maintenance change; scorer fixed to original Float16, A0)

| Arm | b | a | Buckets per learned level | Compaction into non-empty target | Model (hidden / epochs / k-means iters / samples) |
|---|---|---|---|---|---|
| **G0** current | 250,000 | 2 | 548 fixed | full rebuild, retrain | 512 / 30 / 5 / 32,768 |
| **E1** Emma-faithful | 5,000 | 10 | aⁱ | reuse target router (no k-means, no MLP training) | 128 / 10 / 25 / min(n, max(100K, 40·B)) |
| **E1R** ablation (300K only) | 5,000 | 10 | aⁱ | full rebuild, retrain from scratch | as E1 |

E1 values are taken from her code: hidden `N_NEURONS = 128`, `epochs: 10`, `lr 1e-3`,
`batch 256`, `sample_threshold 100000` with the Faiss 40·k floor, Faiss `Kmeans` default
25 iterations, `balance_classes: false`. E1R is not run at 10M: the arithmetic planner
predicts 2,024 publications with 14.4× retraining amplification there.

Routing policies replayed on each frozen final topology (same production
`route_pinned_traced` + `resolve_pinned_candidates`):
`slot` (P·2ⁱ, current default), `density`, `tdensity2`, **`mdn`** (thesis eq. 3.4: P·Bᵢ,
Bᵢ = the member's bucket count, buffer 1.0), **`mdn_code`** (her code: P·Bᵢ/a, i.e.
`arity**level_idx` with 0-based index), `proportional`, and `static_flat`.

Static reference per topology (`static_flat`): one LMI over the same live vectors.
- G0: same model as G0 members (548 buckets), reproducing H-10M.
- E1/E1R: total bucket count matched to the E1 topology (her §4.1.3 protocol) with the
  E1 model rule.

## 3. Data, queries and protocol

- SISAP23 LAION2B CLIP-768, Float16, cosine. The 300K pilot uses the first 300,000 rows of
  the 10,120,191-row corpus (sha256 `f5868d6e…`); 10M uses all rows.
- Query sets as H-10M: validation rows 900–931 (32), test rows 9000–9255 (256).
- Gold: 10M uses the official top-30; 300K uses exact top-30 recomputed on the prefix
  (float64 cosine, ties by smaller ID), produced in-job and hashed.
- Ingestion inserts in chunks of b (E arms) or 250K (G0), each in ID order, and waits for
  the scheduled publication after every chunk that fills the buffer (synchronous drain, as
  in her sequential protocol).
- Work = vectors actually scored per query (learned + Plain). Budgets (minimum,
  whole buckets):
  - 300K: 1K, 2K, 4K, 6K, 8K, 12K, 16K, 24K, 32K, 48K, 64K.
  - 10M: 10K, 20K, 40K, 57K, 80K, 113K, 160K, 226K, 320K, 640K.
- Collection-level search pass after ingestion (100 queries, `candidate_budget` of the arm)
  records latency and exact-fallback count.

## 4. Pilot (300K) gate — engineering only

The pilot cannot settle the scientific question at 300K. 10M is submitted only if all of
these hold for E1:

1. Every scheduled publication succeeds (no optimizer errors) and the final hierarchy
   descriptor validates.
2. The collection search pass uses learned routing for 100/100 queries with 0 exact
   fallbacks.
3. Build events show: every reuse build used the target's router (fingerprint equal) and
   ran no training; every fresh build has B = aⁱ.
4. Projected 10M E1 ingestion wall time, from a per-publication model fitted to pilot
   events (wall = c₀ + c₁·output points + training seconds), is ≤ 40 h. The 10M walltime
   request is then 1.5× projection + 8 h for replay and static reference.

If 1–3 fail, the defect is fixed and the pilot rerun (reported as such). If 4 fails, stop
and report; no 10M run.

## 5. 10M hypotheses (test set; decided by the rules below)

Work at target is the interpolated work to reach mean test R@30 = 0.90 (log-linear in
work between adjacent budgets, as H-10M), with 95% paired query bootstrap (10,000
resamples, seed 20261120).

- **H1 (primary).** E1 with `mdn_code` needs ≥ 30% less work than G0 with `slot`.
  Supported if the point estimate is ≥ 30% and the bootstrap lower bound > 0.
- **H2.** In E1 `mdn_code` at the target operating point, Plain accounts for ≤ 10% of
  scored work.
- **H3 (maintenance).** E1 total build training seconds (teacher + MLP) are lower than
  G0's. Native job wall sums and publication counts are reported for both, without a
  claim of equivalence of scope.
- **H4 (descriptive).** E1 `mdn_code` versus E1 `static_flat` work at target, and every
  policy's curve, are reported. The best E1 policy chosen on validation is reported as
  secondary.

Negative results are reported as such. Thresholds, query sets and the primary policy are
not changed after results exist.

## 6. Phase boundary

Phase 3 (her distribution-shift and delete scenarios) is out of scope here and needs its
own preregistration after the 10M report.

## Amendment 1 — 2026-11-20, before any job or result

Corrections and additions made while writing the tooling. No data had been produced.
1. **Interpolation.** §5 said "log-linear in work … as H-10M"; H-10M (`analyze_10m.py`)
   interpolates the mean Recall@30 curve **linearly** in realized work between the two
   measured budgets that bracket the target, never extrapolating. Linear is used, as in
   H-10M. A curve that starts above or never reaches the target is reported, not filled.
2. **Preflight stage.** Before the pilot, one preflight job runs the whole pipeline at
   100K points (E1, E1R; G0 has no learned level below its 250K buffer; 8 queries per set; budgets 1K/4K/16K). Its numbers are pipeline
   checks only and are not interpreted.
3. **Gold-tool check.** The preflight recomputes the full-corpus top-30 for the val and
   test rows with `prefix_gold.py` and compares it with the official SISAP23 gold. It
   must reach mean set overlap ≥ 0.99 on both sets: official gold comes from float32
   originals, ours from the stored Float16 rows, so bit identity is not expected.
4. **Collection pass gold.** For the 300K pilot the collection search pass (queries 0–99)
   uses prefix gold (exact top-1000 on the prefix) in the official file layout.
5. **Static reference sample size** for E arms: max(B, min(n, max(100K, 40·B), 1M)), with
   B the topology's learned bucket total and the E1 model otherwise.
6. **Analysis.** All estimands are computed by `analyze.py` (committed before any job).
   Comparisons use the same resampled query multiset for every arm and policy.

## Amendment 2 — 2026-11-21, after the failed 10M job 25814270, before any 10M estimand

Job 25814270 failed in the G0 collection search pass. G0 ingested all 10,120,191 points;
the first post-ingestion query then exceeded Qdrant's default 60 s search timeout. The node
(zenon3, EPYC 7351) ingested 2.5× slower than the H-10M node and recorded 16.2M major page
faults, so the query read cold on-disk segments. No frontier replay ran and E1 never
started, so no 10M estimand had been produced. Two engineering changes, neither touching an
estimand, threshold, query set or policy:
1. The collection pass passes an explicit search timeout of 1,800 s
   (`DLI_BENCH_QUERY_TIMEOUT_S`; unset keeps the default). Its latency stays descriptive.
2. The 10M job runs E1 before G0, so the new arm is not lost to a G0 failure. Arms use
   separate collections and scratch is removed between them.

## Amendment 3 — 2026-11-21, after the failed 10M job 25820728, before any 10M estimand

E1 ingested 5.25M points (1,099 publications) and then failed at publication 1100, the
first build of a fresh level 4 (5.5M points, 10⁴ buckets, 400K training sample). The
harness waits at most 30 min for each publication. The pilot projection (§4 criterion 4)
predicts 2,717 s for this one publication; every other publication is projected ≤ 80 s.
The gate checked only the projected total. Changes, none touching an estimand, threshold,
query set or policy:
1. The per-publication wait is configurable (`DLI_BENCH_PUBLICATION_TIMEOUT_S`; unset keeps
   30 min). The 10M job uses 14,400 s, 5.3× the longest projected publication.
2. From now on the gate also requires that the per-publication wait exceed 3× the longest
   projected publication.
The model, bucket rule and thread policy of the arm are unchanged.

## Amendment 4 — 2026-11-21, during 10M job 25823237, before any 10M estimand

In job 25823237, publication 1100 (the fresh level 4, 10⁴ buckets) was still running after
1 h 53 min, against a projection of 45 min. Stack samples on the compute node showed one
busy thread, in spherical k-means assignment (`spherical_kmeans::assign_all`). Cause: the
native DLI maintenance path built every segment with a hard-coded budget of one CPU
(`ResourceBudget::new(1, 1)` in `dli_maintenance.rs`), whatever the job requested. The
pilot's per-FLOP training rate was fitted on 100-bucket levels, whose centroids fit in
cache. At 10⁴ centroids (31 MB) the rate is several times worse, so the projection
(Amendment 3) underestimated this publication.

Change: the DLI build CPU permit is read from `QDRANT_DLI_BUILD_CPUS`; unset keeps 1 CPU.
The 10M job uses 16, the job's allocation, for both arms. This changes no result:
- k-means assignment with several threads is bit-identical to one thread (unit test
  `parallel_assignment_is_bit_identical_to_single_thread`);
- the MLP keeps its single-thread Torch policy.

Wall and training seconds become 16-CPU measurements for both arms. H3 compares E1 and G0
within the job, so it stays like for like. Comparisons with H-10M seconds (one CPU) are
descriptive only.
