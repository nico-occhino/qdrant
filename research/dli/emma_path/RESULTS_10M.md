# Emma path — 10M results (job 25827532)

**Verdict:**
- **H1** supported: E1 needs 72% less work than G0.
- **H2** supported: E1's Plain buffer accounts for 0.2% of work.
- **H3** not supported: E1 trains 1% *more* than G0. One build accounts for 87% of E1's
  training.
- **H4** (descriptive): E1 needs **2.4×** the work of a static LMI with the same bucket
  total.

The Plain-buffer penalty of H-10M is gone. A multi-level penalty remains, larger in
relative terms than before.

Decision rules are in [PREREGISTRATION.md](PREREGISTRATION.md) §5 and Amendments 1–4. The derived numbers are in [`derived/analysis_10m_25827532.json`](derived/analysis_10m_25827532.json). They
come from `analyze.py confirm`, run in-job; the script was committed before any job ran.

## Provenance

| Item | Value |
|---|---|
| Job | `25827532.pbs-m1.metacentrum.cz`, node `alfrid3`, 16 CPU / 192 GB, exit 0, walltime 04:24:03, peak memory 167 GB |
| Source | `1bdd6c024b3d` (clean, verified in-job); binary sha256 `b1890e0b…` |
| Data | full SISAP23 LAION2B 10,120,191 × 768 Float16 (`f5868d6e…`); official top-30 gold |
| Queries | validation rows 900–931 (32), test rows 9000–9255 (256) |
| Build CPUs | 16 per DLI build (Amendment 4), both arms; the MLP stays single-threaded |
| Evidence | remote `dli-emma-path-20261120/results/10m-25827532…/`; local copy `Research-Results/2026-11-20-dli-emma-path/10m-25827532/`. `SHA256SUMS` verify except `job.log` (known `tee` final-line case, verified minus its last line). |
| Earlier attempts | 25814270, 25820728, 25823237: harness limits and a 1-CPU build permit, no estimand produced (Amendments 2–4) |

**Reproduction check.** G0 reproduces H-10M (job 25677858) exactly:
- the same topology, down to bucket sizes;
- identical mean R@30 and mean work at all 7 budgets the two runs share, for `slot` and
  `static_flat`.

This also confirms that the 16-CPU change (Amendment 4) did not alter results. G0's slot
work at the target is 295.6K here against 304.9K in H-10M. The difference comes only from
this run's finer budget grid: the target now falls between 226K and 320K instead of between
160K and 320K, so the linear interpolation differs.

## Topology at 10,120,191 points

| Arm | Members: live points / buckets / largest bucket |
|---|---|
| G0 | L5 6.0M / 548 / 51,723; L4 3.0M / 548 / 25,578; L2 750K / 548 / 5,830; L1 250K / 548 / 2,203; **Plain 120,191** |
| E1 | L4 5.5M / 10,000 / 8,799; L3 4.4M / 1,000 / 19,999; L2 220K / 100 / 5,065; **Plain 191** |

The final E1 levels equal the planner projection [0, 220K, 4.4M, 5.5M].

## Hypotheses (test set, 256 queries, paired bootstrap 10,000 resamples, seed 20261120)

| | Rule | Result | Outcome |
|---|---|---|---|
| **H1** | E1 `mdn_code` needs ≥ 30% less work than G0 `slot` at R@30 = 0.90, CI lower bound > 0 | W = 81,580 vs 295,626; saving **72.4% [67.5, 77.8]** | **Supported** |
| **H2** | Plain ≤ 10% of E1 `mdn_code` work at target | **0.21%** | **Supported** |
| **H3** | E1 build training seconds < G0 | 1,543 s vs 1,522 s | **Not supported** |
| H4 | E1 `mdn_code` vs E1 `static_flat` (11,100 buckets, E1 model), descriptive | W = 81,580 vs 34,023; E1 needs **+140% [103, 180]** | reported |

Secondary, chosen on validation: the best E1 policy on validation is `proportional`
(val W = 25.5K). On test it needs 74.8K, a 74.7% saving vs G0 `slot` [69.1, 79.5].

## Work to mean R@30 = 0.90 (test)

| Policy | G0 | E1 |
|---|---|---|
| slot | 295,626 | 80,335 |
| mdn (thesis) | 267,403 | 80,335 |
| **mdn_code** (her code) | 277,271 | **81,580** |
| density | 280,425 | 85,102 |
| tdensity2 | 210,719 | 79,143 |
| proportional | 281,515 | 74,819 |
| static_flat | 165,191 (548 buckets, G0 model) | **34,023** (11,100 buckets, E1 model) |

G0's Plain share at target is 41–43% under every policy except `tdensity2` (14%).

## Maintenance (16 CPUs per build)

| Arm | Publications | Trained / reused builds | Training s | Routing s | Optimizer wall s | Ingestion s | Write amplification |
|---|---|---|---|---|---|---|---|
| G0 | 40 | 40 / 0 | 1,522 | 562 | 2,259 | 2,429 | 3.1 |
| E1 | 2,024 | 206 / 1,818 | 1,543 | 1,373 | 3,922 | 4,237 | **14.4** |

E1 training seconds by level size:

| Buckets | Builds | Training s |
|---|---|---|
| 10 | 184 | 64 |
| 100 | 19 | 89 |
| 1,000 | 2 | 54 |
| **10,000** | **1** | **1,337** |

## Interpretation

1. **The buffer penalty is solved.** G0 scores the 120K-vector Plain buffer whenever it
   ranks, which is 41% of its work at target. E1's buffer holds 191 vectors. Together with
   finer partitions this cuts work by 72%, so H1 holds with a wide margin. Two causes are
   confounded in this number: the buffer and the finer buckets. The arms differ in both by
   design (Sommerová's invariant couples them).
2. **A multi-level penalty remains, and it is large. It is structural.** With the same
   total of 11,100 buckets and the same model rule, one static LMI needs 34K, while E1's
   three levels need 82K. *Corrected 2026-11-22:* an earlier version of this point guessed
   that the routers of two large levels err independently. The oracle-cost decomposition in
   [MULTILEVEL_DIAGNOSIS.md](MULTILEVEL_DIAGNOSIS.md) (exploratory, post hoc) rejects that:
   - with perfect routing, E1's buckets need **2.44×** the static work (29.5K vs 12.1K);
   - E1 selects as efficiently as the static index (realized / oracle 2.77 vs 2.82).

   The cause is L3: nearly full, with 4.4M points in 1,000 buckets of about 4,444 points,
   against 912 per static bucket. Almost every query needs it. Sommerová's invariant makes a
   full level's buckets about b = 5K, which at 10M is about 5× coarser than the static index.
3. **Partition granularity dominates the model choice.** The static LMI with 11,100 buckets
   (small model: 128 hidden, 10 epochs) needs 34K. The static LMI with 548 buckets (large
   model: 512 hidden, 30 epochs) needs 165K, 4.9× more. This is consistent with the Phase 2
   pilot and the earlier static LMI/IVF results: more buckets matter more than a better router.
4. **Reuse makes training cheap except where a fresh level is born.**
   - The 1,818 reuse builds train nothing.
   - The 205 small fresh builds take 207 s in total.
   - The one fresh level 4 (10,000 buckets, 400K sample) takes 1,337 s and dominates.

   H3 therefore depends on the snapshot. At N = 10.12M, E1 has just created level 4. A
   population before that point would have favoured E1 strongly; one creating level 5
   (10⁵ buckets) would be far worse. It is reported as not supported, as preregistered.
5. **Reuse does not reduce copying.** Write amplification is 14.4 for E1 against 3.1 for G0,
   and E1 ingestion is 1.7× longer. Every publication rewrites the whole target level,
   because Qdrant segments are immutable. This is the main engineering cost of Sommerová's
   design in Qdrant.

## Limitations

- One replicate: one ingestion order, one seed per model. The CIs cover query sampling only.
- Seconds were measured with 16 build CPUs and a single-threaded MLP. H-10M seconds used one
  CPU, so comparisons with them are descriptive only.
- Collection-pass latency (100 queries, budget 2,000) is cold-start and uncontrolled. It is
  not a latency result. Both arms used learned routing for 100/100 queries with 0 exact fallbacks.
- Budget is measured in candidates, not latency. Index memory was not measured in this job.

## Next

Phase 3 (shift and deletes), preregistered in
[PHASE3_PREREGISTRATION.md](PHASE3_PREREGISTRATION.md) (draft, author decisions recorded).
Point 2 above is addressed by Phase 3 Part B (§10 of the Phase 3 preregistration):
consolidation of co-resident large levels (E1C), which attacks the structural cause, and
calibrated level-aware allocation (E1L), whose gain the diagnosis expects to be modest.
