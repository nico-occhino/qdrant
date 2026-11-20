# Development journal: learned indexes inside Qdrant (September – November 2026)

*Thesis: Integration and Evaluation of Learned Indexes in Vector Database Systems (MUNI research traineeship).*

This is the project-wide history: what was tried, why, what came out of it, and especially what went wrong
and what was learned. The work was not linear. It ran on several tracks at once (static LMI, a large HPC
campaign, a dynamic index, controlled ablations). Some of it was dead ends, and two Qdrant versions were involved.
Each part below introduces the theory it needs when that theory first appears, then gives the engineering,
mathematical and statistical content, then the outcome and its lessons.

It is written from an audit of every surviving document (§16 lists them). Numbers come from those documents.
Where a document is the only source of a claim, it is named. Two narrower journals remain the detailed records for
their tracks:
- `Progress_Ladder/STATIC_LMI_RESEARCH_REPORT.md` (Windows side), for static LMI up to 12 November;
- `research/dli/DEVELOPMENT_JOURNAL.md`, for the dynamic index, continuation by continuation.

**Evidence labels.**
- **Accepted:** passed an explicit final validator on immutable artifacts.
- **Measured:** observed in a logged run.
- **Verified:** read in source, manifests or tests.
- **Exploratory:** a bounded diagnostic, a small sample, or post hoc.

---

## Contents

0. [The project in one page](#0-the-project-in-one-page)
1. [Background: nearest-neighbour search, IVF, HNSW and the Learned Metric Index](#1-background)
2. [Part I: exploration and alignment (2–24 September)](#part-i-exploration-and-alignment-224-september)
3. [Part II: making LMI a real Qdrant index (23 September – 7 October)](#part-ii-making-lmi-a-real-qdrant-index-23-september--7-october)
4. [Part III: does the learned router work? Phases F–F.4 at 100K](#part-iii-does-the-learned-router-work-phases-ff4-at-100k)
5. [Part IV: scaling engineering, Phase S (1–6 October)](#part-iv-scaling-engineering-phase-s-16-october)
6. [Part V: Phase G, the 10M failure analysis (5–11 October)](#part-v-phase-g-the-10m-failure-analysis-511-october)
7. [Part VI: the SISAP23 matched campaign on MetaCentrum (13 October – 3 November)](#part-vi-the-sisap23-matched-campaign-on-metacentrum-13-october--3-november)
8. [Part VII: forward-port to Qdrant v1.20 (10–12 November)](#part-vii-forward-port-to-qdrant-v120-1012-november)
9. [Part VIII: the Dynamic Learned Index (12–18 November)](#part-viii-the-dynamic-learned-index-1218-november)
10. [Part IX: static LMI versus IVF, the controlled ablation (18–19 November)](#part-ix-static-lmi-versus-ivf-the-controlled-ablation-1819-november)
11. [Part X: filtered search (19 November)](#part-x-filtered-search-19-november)
12. [Part XI: nearest-neighbour-supervised routing (19–20 November)](#part-xi-nearest-neighbour-supervised-routing-1920-november)
13. [Part XII: repository consolidation and upstream sync (20 November)](#part-xii-repository-consolidation-and-upstream-sync-20-november)
14. [Cross-cutting lessons](#14-cross-cutting-lessons)
15. [Register of negative results and dead ends](#15-register-of-negative-results-and-dead-ends)
16. [Documentation audit: sources, gaps and inconsistencies](#16-documentation-audit)
17. [Where the evidence lives](#17-where-the-evidence-lives)

---

## 0. The project in one page

**Question.** Can a *learned* index, where a neural network decides where to look, become a legitimate physical
index inside a production vector database? The database must keep ownership of storage, identities, deletions, filters,
persistence and exact scoring. If it can, under which workloads does it pay off?

**Short answer, as of 20 November 2026.**
- **Engineering: yes.** LMI is a native Qdrant index on v1.20.2. It is built by the optimizer, persisted, reopened
  without retraining, snapshot-restored and served through the normal search path, including adaptive filtered search.
- **Science: as a static index, no, on this workload.**
  - The learned router never beat plain centroid routing (IVF) on the same partition.
  - IVF needs about half the candidates at matched recall at 10M.
  - HNSW is 10–27× faster at matched recall.
  - Neighbour-supervised training and better calibration did not change this.
  - LMI's real advantages are index size (about 10× smaller than HNSW) and build simplicity.
- **Dynamic index (DLI): it works end to end, but costs more.** A Dynamic Learned Index runs inside Qdrant with
  native optimizer publication, recovery and concurrent reads. At 10M it needs about 45% more candidate work than a
  single static LMI over the same data. The extra work is caused by the unpartitioned insertion buffer and the
  allocation policy, not by the independent level models.

**Timeline.**

| Dates (2026) | Track | What happened | Outcome |
|---|---|---|---|
| 2–22 Sep | Exploration | Reading, Qdrant inspection, Phases A–D: a dummy LMI, then candidate routing and scoring | Integration seam found |
| 24 Sep | Alignment | Supervisor-aligned priorities: a first-class index, with Qdrant owning everything authoritative | Direction fixed |
| 23–26 Sep | Native integration | Phase E/E2: database-owned training, persistence, lifecycle, snapshots | **Implemented and tested** |
| 26–30 Sep | Router science | Phase F (100K): MLP needs 42% more candidates than centroid; F.2–F.4: undertraining diagnosed, oracle headroom found | Negative baseline, then explained |
| 1–6 Oct | Scaling | Phase S/S.3: compact postings, streaming build, batch/tch routing, spherical k-means, 10M builds | 10M build feasible |
| 5–11 Oct | Diagnosis | Phase G: 10.12M recall 0.815 at nprobe 4, router-limited; independent audit | Router is the bottleneck |
| 13 Oct – 3 Nov | HPC campaign | SISAP23 matched Plain/LMI/HNSW at 10.12M; a chain of infrastructure incidents | **Accepted 3 Nov:** LMI 27× slower than HNSW at recall 0.95 |
| 10–12 Nov | Port | v1.20 port, 9,980/9,980 identical rankings | Compatible |
| 12–18 Nov | DLI | 45 continuations: functional DLI, 300K and 10M runs, audit, H-10M | DLI costs more than static |
| 18–19 Nov | Ablation | Static LMI versus IVF, Tiers 1–3B (1M offline to 10M native) | IVF wins everywhere |
| 19 Nov | Filtering | Adaptive filtered search plus a preregistered probing study | B0 stays; calibration fixes the recall cliff |
| 19–20 Nov | Supervised routing | Neighbour-supervised routers over IVF-8192 | **Negative** (gate failed) |
| 20 Nov | Repository | 18 worktrees and 23 branches consolidated into `main` plus tags; synced to v1.20.2 | Clean baseline |

---

## 1. Background

### 1.1 Exact and approximate nearest-neighbour search

A vector database stores embeddings $x_i \in \mathbb{R}^d$. In this project they are CLIP image/text embeddings,
$d = 768$. A query $q$ asks for the $k$ stored vectors most similar to it. With **cosine similarity** on
unit-normalised vectors, similarity is the dot product $q \cdot x$.

**Exact search** (Qdrant's *Plain* index) scores every vector: $O(Nd)$ work per query. It is authoritative and gives
the ground truth, but at $N = 10^7$ that is $7.7 \times 10^9$ multiply-adds per query.

**Approximate (ANN) search** scores only a *candidate set* $C(q)$ and returns the best $k$ among them. Quality is
measured as **Recall@k**: the fraction of the true top-$k$ that is returned,
$$\text{R@}k(q) = \frac{|\text{returned}(q) \cap \text{true}_k(q)|}{k}.$$
Throughout the project, the main cost proxy is the **number of candidates scored**, $|C(q)|$. On this workload,
exact scoring of candidates dominates query time: about 91% of LMI's latency at 10M (§V, §VI). Index-structure
overhead (routing a query, traversing a graph) is second order.

> **Why candidates are the right currency, and their limits.** At matched recall, fewer candidates means less memory
> traffic and less arithmetic. That makes candidates hardware-independent and cheap to compute offline.
> Candidates miss three things: (a) routing cost; (b) *where* candidates live, since random reads from cold storage cost
> far more than cached sequential ones (the NFS incident in §VI, the DLI cold member in §VIII); and (c) per-candidate
> overheads such as filter checks. The project therefore always pairs candidate counts with native latency before
> drawing a final conclusion.

### 1.2 Partition-based indexes: IVF

**IVF (inverted file)** partitions the corpus into $B$ buckets with k-means and stores one *posting list* of IDs per
bucket. At query time it ranks buckets by the similarity of $q$ to each bucket's centroid $c_b$, then scores every
vector in the top **nprobe** buckets.

- **Pros:** simple; small index (centroids plus IDs); cheap to build; work is predictable from nprobe.
- **Cons:** the recall/work trade-off is coarse; an uneven bucket size makes work vary per query; high recall needs
  many probes, because a true neighbour can sit across a Voronoi boundary.

A key fact used repeatedly: **nearest-centroid assignment is an affine rule.**
$$\arg\min_b \lVert x - c_b \rVert^2 = \arg\max_b \left(2\,c_b \cdot x - \lVert c_b \rVert^2\right).$$
So the k-means partition is exactly representable by a linear layer, and a neural router is only useful if it learns
something *better* than the partition that taught it. This is why the project always ran the **centroid and affine
controls** next to the MLP (Phase F onwards).

**Spherical k-means** (Phase S.3C) is the cosine version. It assigns by maximum dot product and re-normalises the
centroids to unit length. Plain Euclidean k-means on unit vectors produces non-unit centroids, whose norms bias the
assignment. That is the reason for the switch.

### 1.3 Graph indexes: HNSW

**HNSW** builds a multi-layer proximity graph and answers a query by greedy best-first traversal; `ef` sets the
size of the search frontier. It visits a few thousand vectors where IVF/LMI score tens of thousands, because it
navigates *towards* the neighbours instead of scanning whole regions.

- **Pros:** very low latency at high recall; Qdrant's default index and heavily optimised.
- **Cons:** large index (graph edges: 516–702 MB at 10M, against LMI's 49 MB); expensive build; filtered search needs
  special handling; incremental updates are cheap but deletions degrade the graph.

### 1.4 The Learned Metric Index (LMI)

LMI (Coda Research Group, MUNI: SISAP challenge papers and theses) replaces IVF's centroid ranking with a neural
**router**:
1. **Teacher:** k-means on a training sample gives $B$ clusters, and every sample point gets a bucket *label*.
2. **Router:** an MLP (768 → hidden → $B$, ReLU) is trained with cross-entropy to predict a point's label.
3. **Postings:** the *trained router* classifies every corpus vector. A vector goes to the bucket the MLP predicts,
   which is not necessarily the k-means bucket.
4. **Query:** the MLP scores all $B$ buckets for $q$; the top nprobe buckets' postings are scored exactly.

The hope is that a non-linear router ranks buckets better than distance to a centroid, so it needs fewer probes at the
same recall. **The cost** is training (a LibTorch dependency), a per-query forward pass ($768 \cdot H + H \cdot B$
multiply-adds) and a partition that depends on the model's errors.

**The oracle bound** used throughout: for a fixed partition, an *oracle* router probes the buckets that actually
contain the most true neighbours. Its recall at nprobe is an upper bound for *any* router on that partition. The gap
(oracle minus router) measures routing quality; the gap (1 minus oracle) measures partition quality.

---

## Part I: exploration and alignment (2–24 September)

**Sources:** `Records/` (starter, plan of work, meeting transcripts) and `Progress_Ladder/` PDFs: Preliminary LMI
characterisation (16 Sep), Qdrant v1.19 static inspection (17 Sep), Phases A–D (17–22 Sep).

The work started from the MUNI LMI line (Python prototypes, SISAP 2023/2024 challenge code, a Rust LMI) and a plan of
work. The first two weeks were reading and inspection:
- **Preliminary characterisation:** what LMI is, its configuration knobs, how its teacher/router/postings fit together.
- **Phase A, Qdrant inspection:** collections → shards → segments → vector index; where a new index type could be inserted.
- **Phase B, dummy LMI:** a placeholder index wired into Qdrant, to learn ownership: who holds vectors, IDs, deletions.
- **Phase C, candidate routing with Qdrant scoring:** the central design choice. LMI produces *candidate offsets*
  only, and Qdrant's own scorer computes the exact scores.
- **Phase D, static learned routing integration:** a trained router serving queries inside a segment.

**The supervisor-aligned decision (24 Sep, `Supervisor_Aligned_LMI_Priorities.md`).** LMI must be a *first-class
physical index*: creatable through Qdrant's normal lifecycle, with no external Python training or manual model import.
Qdrant stays the owner of vectors, identifiers, visibility/deletion, storage and final scoring. LMI owns only auxiliary
routing state and postings of **segment-local offsets** (`PointOffsetType`), never external IDs. Query-plan selection
(choosing Plain/HNSW/LMI per query) was deferred.

> **Why the ownership rule matters.** In a database, an index is an *access path*, not a second copy of the data.
> If LMI stored its own vectors or IDs, every delete, update, snapshot and crash recovery would have to be done twice
> and kept consistent. Postings of segment-local offsets inherit Qdrant's visibility for free: a deleted point is
> masked by the ID tracker and never scored. The price is that postings are only valid for one physical segment.
> A merge or rebuild must regenerate them. This price comes back in the DLI work (§VIII).

**Ideas raised but not pursued then.**
- Giovanni Bellitto (Catania) suggested replacing the MLP with *prototype key–value routing* inspired by continual
  learning's Learning-to-Prompt (`Records/aligns.txt`, 30 Sep).
- A statistical router research note followed (14 Oct).

These were not implemented. The later ablation (§IX) and the analysis in §XI show why a geometry-plus-statistics
router is the natural control any learned router must beat.

---

## Part II: making LMI a real Qdrant index (23 September – 7 October)

**Sources:** `docs/lmi-phase-e.md`, `docs/lmi-phase-e2.md`, `docs/lmi-phase-e2-final.md` (in tag
`archive/thesis/lmi-integration`). Commits `4f3bd3798` (23 Sep) to `a1987b504` (26 Sep).

### What was built
- **Configuration:** `vectors.lmi_config` per named vector, through REST/gRPC → collection → shard optimizer.
- **Construction:** the optimizer's `SegmentBuilder → build_vector_index` calls the LMI builder, which reads Qdrant's own
  vector storage, samples, trains (CPU, LibTorch behind the `lmi-training` feature), exports a native model, and
  classifies the corpus into postings.
- **Persistence:** `lmi_state.json` plus router and postings files, saved with the segment.
- **Native open:** reopening loads and validates state *without training*. A binary without `lmi-training` can serve
  but not build.
- **Lifecycle (E2):** deferred points, update/rebuild after corpus change, mixed named vectors (LMI and HNSW on
  different fields), snapshot restore, rejection of corrupt or mismatched state (version, dimension, count, model shape).

### Engineering problems met
1. **Stable Rust compatibility.** The `kmeans 1.1.0` crate needed nightly features, so clustering was reimplemented.
2. **Persistence could not be postponed.** `SegmentBuilder` drops built indexes, publishes the directory and *reloads*
   the segment. An in-memory model disappears during a normal build, so the persisted format had to exist first.
3. **Native inference does not mean a LibTorch-free server.** Query routing used a native Rust MLP, but a
   training-enabled server still links LibTorch.
4. **Structural validation is not authentication.** A same-shaped state from another corpus would pass. This was
   recorded as a limitation (no corpus fingerprint).

**Outcome (Measured):** 35–36 LMI integration tests passing (candidate scoring, dummy, Phases C/D/E/E2), plus snapshot
and corruption controls. Filtered, quantized and non-default-parameter queries *fell back to Plain*. Filtered
learned search only arrived in November (§X).

---

## Part III: does the learned router work? Phases F–F.4 at 100K

**Sources:** `docs/lmi-phase-f*.md`, `Progress_Ladder/Phase_F*_Report.md`, the supervisor briefing. 99,780 LAION CLIP-768
vectors, cosine, $k = 10$, 64 buckets.

### Phase F (26 Sep): the honest baseline
At Recall@10 ≈ 0.90, centroid routing needed **10,197** candidates and the MLP **14,504** (+42%). HNSW took 0.16 ms
against 3.5 ms. The MLP had **24 of 64 buckets empty**: no corpus point was ever predicted there.

### Phase F.2 (27 Sep): diagnosis, not tuning
- **Finding:** the 30-epoch baseline was **underfit**. All 24 empty classes had training examples (1–29 each), but
  training accuracy was only 71.7%.
  - With more epochs and samples, empty buckets fell to 10, then to 1–2.
  - Selected 4,096-sample / 60-epoch models used **9,372–9,566** candidates against centroid's **10,231** near 0.90.
- **Key conceptual lesson:** *teacher agreement and ANN recall measure different things.* A router can disagree with
  k-means on the top-1 label and still retrieve well, because retrieval probes several buckets. Conversely, perfect
  agreement guarantees nothing about the second-ranked bucket.

### Phase F.3 (30 Sep): frozen models, validation-locked comparison
This was the first use of the statistical protocol that became standard:
1. Choose the operating point on **validation** queries (300): the smallest nprobe whose mean recall reaches the target.
2. Freeze the choices, then evaluate once on **test** queries (1,000).
3. Report **paired query bootstrap** intervals (2,000 resamples).

> **Why paired bootstrap.** Two methods answer the same queries, so their per-query differences are paired.
> Resampling *queries* (not methods independently) gives an interval for the mean difference that accounts for
> query difficulty, which varies far more than the methods differ. A **non-inferiority margin** (here 0.01 recall)
> asks "is B at most 0.01 worse than A?", which is the right question when B is cheaper.

**Result.** At 0.90 the frozen MLPs saved 9.6–10.0% of candidates, but only one of three seeds met the recall margin.
At 0.95 they saved 4.3–4.5% and all met it. This was a real but small local result.

### Phase F.4 (30 Sep): oracle headroom and the first retrieval-target router
- At 4 probes the oracle reaches Recall@10 **0.980** and the MLP **0.882**: 0.098 of headroom is lost to routing.
- A **retrieval-target MLP**, trained on which buckets hold each query's true neighbours instead of on k-means labels,
  reached 0.907, but with more candidates.
- **This is the idea that returned on 19 November as "supervised routing" (§XI)**, seven weeks and many experiments later.

**Pros and cons of the Phase F protocol.** It established controls (centroid/affine), held-out queries and paired
intervals early. Its weakness was scale: 100K vectors, 64 buckets and 100–1,000 queries are far from the 10M regime,
where bucket counts in the thousands change the picture (§IX).

---

## Part IV: scaling engineering, Phase S (1–6 October)

**Sources:** `docs/lmi-phase-s*.md`. Commits `6f7085b9c` (1 Oct) to `e360295aa` (6 Oct). This was the most purely
engineering phase. Each change was validated by byte-identical outputs or exact parity checks.

| Step | Problem | Solution | Theory and trade-off |
|---|---|---|---|
| S.1 compact postings | One vector of IDs per bucket wastes memory and allocations | One `u32` offset array plus a $B+1$ boundary array (CSR layout); v2 three-file format; v1 reader kept | CSR is the standard sparse-row layout: $O(N + B)$ memory, contiguous scans. Offsets are segment-local, which is fine under the ownership rule |
| S.2 streaming build | Holding every label and offset at 10M is too much RAM | Two passes: *count* bucket sizes, prefix-sum boundaries, then *fill* | Trades a second routing pass for $O(B)$ extra memory |
| S.3A top-1 routing and a memory planner | The build-time classifier used full top-k machinery | Dedicated top-1 path; explicit resource planner (policy C) | Planning memory up front avoids out-of-memory failures in the optimizer |
| S.3B1 native batch routing | Row-at-a-time routing is slow | Bounded native batches | Little gain: scalar Rust without BLAS is compute-bound |
| S.3B2 tch batch routing | Same | Route corpus batches through LibTorch: **6–21× faster** | Bigger batches reach BLAS kernels; logits memory is bounded by $K \times B$ |
| S.3B2a tie-safe routing | Torch and native disagreed on 2 of 1M SISAP rows | When the top-2 logits differ by at most $4\varepsilon(1+\max\lvert s\rvert)$, fall back to native | **Floating-point addition is not associative.** Different reduction orders give different last bits, so near-ties flip. The rule is relative, symmetric and nonzero near 0. The multiplier 4 was the smallest *tested* value covering all 137 generated near-boundary cases |
| S.3C spherical k-means | Euclidean Lloyd on cosine data biases toward large-norm centroids | Spherical k-means with an f64 scalar oracle | Correct metric; the f32 SIMD production path may still differ at exact ties |
| S.3D 10M Lance build | First end-to-end 10M | Bounded ingestion, one trained build, persisted and reopened | Showed the routing passes dominate: count 1,374 s plus fill 1,376 s |
| S.3E cached labels | The corpus was routed twice | Cache `u16` labels after pass 1 (2 bytes × $N$ = 20 MB at 10M) | Halves routing; works only while $B \le 65{,}536$ |

**A negative result inside Phase S (S.3B2a).** On a synthetic $B = 10{,}000$ fixture the tie-safe fallback fired on
*every* row: the generated weights repeated every 3,811 buckets and so produced exact ties. Throughput collapsed to
the native speed. The acceleration was therefore never made a default. It stayed an environment-gated option.

---

## Part V: Phase G, the 10M failure analysis (5–11 October)

**Sources:** `docs/lmi-sisap2023.md`, `docs/lmi-phase-g.md`, `Progress_Ladder/lmi-comprehensive-audit.md`
(11 Oct).

**Accepted 5 Oct:** the SISAP 2023 subset (**10,120,191** × 768 Float16, not exactly 10M, because the official gold
references IDs above 10,000,000) was built with $B = 3162$, hidden 512, a 250K sample, 30 epochs and nprobe 4.
Mean Recall@10 was **0.815** at 17,667 candidates per query (0.17% of the corpus). HTTP p50 was 13.3 ms, but p95 was 171 ms.

**Phase G question:** why only 0.815? The **TT/TM/MT/MM decomposition** crosses teacher or model *placement*
(which bucket holds a point) with teacher or model *routing* (which buckets the query probes):

| | Model routing | Teacher routing |
|---|---|---|
| Model placement | MM: what is deployed | MT |
| Teacher placement | TM | TT: k-means with centroid routing |

Comparing the cells separates partition loss from routing loss.

**Coverage at the same postings:**

| nprobe | Oracle coverage | Router coverage |
|---:|---:|---:|
| 1 | 0.664 | 0.559 |
| 4 | 0.949 | 0.819 |
| 8 | 0.999 | 0.892 |
| 16 | 1.000 | 0.940 |

**Decision G1: ROUTER-LIMITED.** The oracle–router gap (0.13) is larger than the partition ceiling (0.05).

**G3/G4, planner and multi-index:** a negative result. In Qdrant's segment format one vector field has exactly one
index, so a three-way Plain/HNSW/LMI planner is not a local change. LMI and HNSW are mutually exclusive per field.

**The 11 Oct independent audit** found the deepest *measurement* caution of the whole project. The official ground
truth ranks normalised source vectors in high precision. Qdrant ranks its **stored Float16** vectors with
half-precision arithmetic. So even an exhaustive scan does not have official recall exactly 1.0. Recall against
official gold and exactness against the stored representation are **two different contracts**. That later broke a
validator (§VI, "Plain row 333").

---

## Part VI: the SISAP23 matched campaign on MetaCentrum (13 October – 3 November)

**Sources:** `tools/lmi/sisap23_matched/*.md` (tag `archive/rescue/sisap23-small-lifecycle-wip`),
`Progress_Ladder/SISAP23_LMI_INCIDENT_REPORT_2026-10-26.md`, `STATIC_LMI_RESEARCH_REPORT.md` §6 and §8.

### Design
A **matched benchmark** in the same Qdrant (1.19.2): Plain, LMI and two HNSW configurations (A: m 16/ef_c 100;
B: m 32/ef_c 200).
- **Shared data:** identical prepared corpus bytes, one shard, one segment, Float16 cold vector storage.
- **Queries:** a 1,000-query panel × 3 repetitions, plus the full 9,980 official queries.
- **Sweeps:** LMI nprobe 1–256 and HNSW ef 16–1024.
- **Infrastructure:** PBS jobs on MetaCentrum with immutable generations, hash manifests and a final independent validator.

### The incident chain
This campaign took three weeks instead of days. Each incident taught something general:

| Date | Incident | Root cause | Fix and lesson |
|---|---|---|---|
| 13 Oct | Plain 10M build: **1,679 points missing** after reopen | `WaitUntil::Visible` waits for search visibility, not a durable segment flush; the job stopped before the flush | WAL inspection and replay on scratch copies restored all 1,679 vectors. **Visibility ≠ durability**; count checks must read *persisted* state |
| 14–15 Oct | Diagnosing safely | The original collection was evidence | Rule adopted: never open the original; copy it, verify byte-for-byte, then experiment |
| 16 Oct | LMI build rejected for memory (exit 101); the fixture then waited up to 48 h while the optimizer retried | The admission guard computed cgroup limit minus usage, and **kernel usage includes reclaimable page cache**; `OptimizersStatus::Error` was ignored | Cache-aware admission; fail on optimizer error. **Never trust a wait loop that cannot fail** |
| 19–20 Oct | Publishing a generation failed: `renameat2(RENAME_NOREPLACE)` returned errno 22 | The NFS filesystem does not support that flag | Publication by immutable generation descriptors instead of directory rename; tested on tiny fixtures first |
| 22 Oct | Build "complete" but with **two live segments** | `wait_for_index` accepted the *aggregate* indexed count, not one segment; Green is a snapshot, not durable | Require exact point count, Green, and a persisted topology check |
| 23–26 Oct | Full LMI build: **search timeout** in validation | Exact scoring of 18,543 routed candidates from **cold NFS** took more than 60 s (page faults); the router took 7.5 ms. Warm: 78 ms | It was I/O, not routing or fallback. **Low CPU does not mean stalled.** Stage data to local SSD and separate integrity checks from functional checks |
| 26 Oct | Diagnostic job died in 10 s | `set -Eeuo pipefail` plus a variable read before assignment in `common-env.sh` | Assign before use; validate the environment before the expensive stage |
| (later) | A validator demanded that Plain have official Recall@10 = 1.0 | The Float16 contract (§V) | Measure official recall separately; prove exhaustive-scan validity by other means |
| (later) | Plain "v7" tie: a different top-k order from history | Exact equal scores; history is not a unique oracle for ties | Accept equal-score permutations after rescoring; no near-tie tolerance |

### Accepted result (3 November)
The final five jobs (Plain 25206053, LMI 25228294, HNSW-A 25228296, HNSW-B 25228298, validator 25228301) all
exited 0 and passed the final validator.

| Method at recall ≥ 0.95 | Median latency | Recall@10 (panel) |
|---|---:|---:|
| LMI nprobe 30 | 36.9 ms | 0.960 |
| HNSW-A ef 256 | 1.38 ms | 0.958 |
| HNSW-B ef 128 | 1.34 ms | 0.957 |
| Plain (exhaustive) | 564 ms | 0.997 |

- **LMI is about 27× slower than HNSW at matched recall.**
- **Decomposition:** 117,212 candidates (1.16% of the corpus), 33.4 ms of scoring and 3.1 ms of routing. Scoring is
  91% of the time.
- **Index-only bytes:** LMI 48.6 MB against HNSW 516–702 MB, about 10–14× smaller.
- **Build time:** LMI 1,989 s against HNSW-A 6,911 s.

**Interpretation.** The gap is the *volume* of vectors that bucket partitioning must score to reach high recall, not
neural inference. That diagnosis drove everything after it: the IVF ablation (§IX) and supervised routing (§XI) both
attack routing quality, because routing quality sets the candidate volume.

---

## Part VII: forward-port to Qdrant v1.20 (10–12 November)

**Sources:** `research/v120-validation/README.md`, `experiments/2026-11-11-v120-parity`.

The integration was ported to official **v1.20.0** in a few commits (`925c3c55d` and following), with static and
scoring regressions and the mixed-segment lifecycle re-validated.

**Parity (Verified):** a freshly built shared v3 state gave **9,980/9,980 identical rankings** between the historical
v1.19 code and v1.20.

**Catch:** the historical 300K collection had `lmi_state.json` version 2, while both current lines require version 3.
Old artifacts cannot simply be reopened; parity had to be shown on a fresh shared state.

**Profiling (12 Nov):** opt-in native profiling showed that *observing* changes performance. Debug builds and
per-query logging distort timings, so they are diagnostics only. Release-mode linking exceeded a 30-minute cap, which
taught the habit of reporting build failures instead of inferring release results from debug.

**CI hygiene:** two PRs against the v1.19 line (#2 lifecycle tests, #3 CI hygiene) separated new defects from
pre-existing nightly-format and lint debt. They were superseded by the v1.20 line and archived in November.

---

## Part VIII: the Dynamic Learned Index (12–18 November)

**Sources:** `research/dli/` (about 50 documents), especially `DEVELOPMENT_JOURNAL.md`,
`FINAL_IMPLEMENTATION_STATUS.md`, `SISAP23_300K_FIRST_RUN.md`, `sisap23_10m/*`, `search_optimization/*`,
`unified_architecture/*`, `selection_frontier/*`. These are 45 "continuations" (numbered work sessions).

### Theory: why a dynamic learned index
A static learned index must be **rebuilt** when data changes. The router reflects the build-time sample, and new
points cannot be placed without retraining. The MUNI **Dynamic Learned Index** (Sommerová, Plhák and Jakubík theses;
`LITERATURE_AND_REFERENCE.md`) applies the classic **Bentley–Saxe logarithmic method**:
- New points go into an unindexed **buffer** of capacity $b$.
- When the buffer fills, it is merged with lower levels into level $i$, of capacity $b \cdot a^i$ (arity $a$).
  Each level is an independently trained LMI.
- A point is rewritten $O(\log_a N)$ times over its life, which amortises rebuild cost.
- At query time every level's router scores its buckets. A **global allocation policy** chooses which buckets, across
  all levels plus the buffer, to scan within a candidate budget.

- **Pros:** amortised updates; bounded rebuild cost; old levels are immutable.
- **Cons:** a query must consult several models whose scores are *not calibrated against each other*; the buffer is
  scanned exhaustively; total write amplification is about $\log_a N$.

### Architecture decision: levels across Qdrant segments (ADR 001)
Placing levels *across segments* (option B) keeps Qdrant's ownership of WAL, IDs and appendable segments. Three hard
facts from the source audit (`QDRANT_ARCHITECTURE_AUDIT.md`, `DLI_RECOVERY_MATRIX.md`) shaped the work:
1. **Holder IDs are transient:** `swap_new` assigns new IDs, so a registry cannot key on them.
2. **The segment manifest is superset-biased:** it may list incomplete or obsolete segments, so it cannot be the
   authoritative level membership.
3. **COW and flush ordering:** deleting an old copy before its replacement is durable can lose data on WAL replay.

### Milestones
- **Continuations 01–21 (12–16 Nov):** test-only registry, pinned views, the native optimizer driving private LMI
  builds, insert-triggered publication, stale-plan rejection, worker panic safety.
- **Continuation 22: FUNCTIONAL EXPERIMENTAL DLI.** A configured LocalShard activates maintenance and search; ordinary
  inserts trigger native publications; graceful reopen restores routing; corrupt metadata falls back safely. Not
  exposed through REST/gRPC.
- **First 300K run (16 Nov):**
  - DLI Recall@10 **0.7625** against static **0.7962** (−3.4 pp);
  - p50 **6.9 ms** against **4.4 ms** (+58%);
  - not recall-matched; a single run.
- **Continuation 23, a masked result.**
  - After deletions, the original code rejected *every* query with nonzero deletions and fell back to exact search,
    so its post-update "recall .997" was the exact fallback.
  - Allowing learned routing after deletion gave the true learned recall: **0.784**.
  - **Lesson:** always count how a result was produced (learned against fallback), not just its recall.
- **Continuation 24, maintenance policies:** geometric DLI rewrote fewer points than static rebuilds but more than a
  size-tiered alternative.
- **Continuation 25, first 10M tracks:**
  - Geometric: Recall@10 **0.402** at budget 2,000, p50 **3.1 s**.
  - The size-tiered run exposed a **storage-mode inheritance bug**: destination segments copied
    `sources.first().config()`, so placement (Cached against Cold) depended on source order. Every latency outlier hit
    the Cold member.
  - Fixed in continuation 26.
- **Continuations 26–28, search optimisation:**
  - **Top-k materialisation** (score all selected vectors, but keep only $k$ per member) is proven safe and is a real
    Pareto improvement.
  - Allocation policies (reserved per level, population density, temperature $T = 2$) are evaluated at *matched actual
    work*. Comparisons at nominal budget had misled: population density looked bad at the same nominal budget but was
    good at matched actual work.
- **Continuations 30–40, quantization:** a scalar-quantized shortlist (128) plus exact rerank gave **1.8× faster**
  warm scoring with identical recall on a 65K collection. It stayed default-off pending recovery and cold-I/O evidence.
- **Continuation 41, independent code audit (Claude Code, 17 Nov).** Twelve findings F1–F12, mostly
  *measurement-validity* defects:
  - an outcome label written before the underfill check;
  - diagnostic I/O inside timed, lock-held sections;
  - a feature flag that switched *static* LMI to a profiled path;
  - ordinary optimizers acting on DLI segments.

  All were fixed or bounded.
- **Continuation 43, concurrency:** 1, 4 and 8 readers, **0 violations**. New finding **F12**: after a graceful stop,
  retired optimizer sources were reloaded beside their replacements (7 segments instead of 2). Fixed by a final sync
  flush for DLI shards (`072be2e0c`).
- **Continuation 42, selection frontier (65K):**
  - At equal scored work, a **single static LMI beats every DLI policy** (+0.011 to +0.239 R@30).
  - The router's ranking costs about 3.5× over a gold-bucket oracle.
- **H-10M (job 25677858, 17–18 Nov), preregistered, the main DLI result:**
  - Static LMI reaches R@30 = 0.90 at **166K** scored vectors per query.
  - DLI-slot needs **305K**, so static needs **45% less** work (95% CI 32–51%).
  - Proportional allocation needs 285K (static 42% less).
  - The curves never cross.
  - **Revised diagnosis:** the gap comes from the exhaustively scored Plain buffer (120,191 points at the time) and
    slot's fixed buffer priority. It does *not* come from independent level models: proportional allocation is
    roughly equal to static at equal *learned* work.

**Status at the end of the track.** DLI is a functional research system inside Qdrant with recovery, concurrency
and safety controls. Scientifically it costs more work than a static index over the same data, for a reason now
understood. The open comparison is a replication against Emma Sommerová's reported results.

---

## Part IX: static LMI versus IVF, the controlled ablation (18–19 November)

**Sources:** `research/lmi/LMI_VS_IVF_ROUTING_ABLATION.md`, preregistrations, `STATIC_LMI_RESEARCH_OPPORTUNITIES.md`,
`STATIC_LMI_INDEPENDENT_AUDIT.md`.

**Question.** Is the MLP router better than the centroid router it was distilled from? Everything earlier mixed
partition and router effects. This ablation separates them: IVF and LMI use **identical infrastructure**. A
research-only exact spherical-centroid router (`44e361b27`) turns the native LMI into native IVF-Flat.

**Design.**
- **Tier 1:** 1M offline lab, $B \in \{64, \dots, 4096\}$, native against converged teacher. Preregistered; Amendment
  01 moved selection to validation queries.
- **Tier 2:** 10M offline confirmation on MetaCentrum.
- **Tier 3A/3B:** 10M **native** Qdrant latency (jobs 25738234 and 25743351).

**Findings.**
1. **The MLP never beats centroid routing on a fixed partition** (centroid/MLP work ratio 0.555–0.995). IVF needs
   3–37% less work than native LMI, and the gap grows with $B$.
2. **The MLP memorises its training sample** (train accuracy 0.997–1.000) instead of learning the partition. The
   router's ranking cost relative to the oracle grows to 11.8× against IVF's 6.7× at $B = 4096$.
3. **Native 10M (Tier 3A):**
   - At the locked R@10 ≈ 0.95, IVF scores **0.551×** the candidates of LMI, with **0.588×** the p50.
   - HNSW is about **16× faster** at p50 (1.30 against 20.5 ms) and **20×** in throughput at 16 threads.
4. **Tier 3B, a finer IVF:**
   - IVF-8192 at nprobe 20 reaches R@10 0.946 in **12.4 ms**: 0.49× the candidates and 0.44× the p50 of IVF-3162.
   - HNSW ef 160 reaches 0.952 in 0.95 ms.
   - The best learned-partition index is therefore still about 13× slower than HNSW.

> **Why the MLP cannot beat its teacher with hard labels (math).** Cross-entropy on one-hot k-means labels is
> minimised by predicting the label distribution $P(b \mid x)$, which for a hard partition is the indicator of $x$'s
> Voronoi cell. The loss is therefore indifferent to how the *other* buckets are ranked. Yet multi-probe retrieval
> (nprobe > 1) depends entirely on ranks 2…nprobe. Centroid scores provide a geometrically meaningful ranking for
> free; an MLP trained on labels has no signal to learn one. This is why the next experiment changed the *target*,
> not the model size (§XI).

**Engineering side-gains:**
- the per-query router overhead was removed (shape check, SIMD dot product, partial top-k: `aaca97f24`);
- spherical k-means assignment was parallelised over the build CPU permit (`3e49dabcd`);
- an underfilled-probe defect was fixed: widen until top-$k$ fills (`15727cc2e`).

---

## Part X: filtered search (19 November)

**Sources:** `research/lmi/filtering/`: baseline, preregistration with Amendment 1, report.

**Problem.** Until now, *any* filter sent LMI to a full Plain scan: correct, but a full scan per query (audit L1).

**Theory: the plan choice.** For a filter matching $m$ points out of $N$:
- **Exact plan:** iterate the matching points through the payload index and score all of them. Cost is about $m$ and
  recall is 1.0.
- **Routed plan:** probe nprobe buckets and score only the candidates that pass the filter. Cost is about the *probe
  volume* $V = \text{nprobe} \cdot N / B$.

Break-even: choose exact when $m \lesssim V$. The planner uses Qdrant's cardinality estimate:
- exact if the estimated maximum is at most $V$;
- routed if the estimated minimum exceeds $V$;
- otherwise a seeded sampling check decides.

The routed plan **widens** nprobe (doubling) until $\min(k, \text{matches})$ results are found.

> **Why filtering hurts routed recall.** With a filter independent of the vectors, keeping fraction $s$ of points,
> the filtered top-$k$ is roughly the unfiltered top-$(k/s)$ thinned out. Recall then depends on routing quality at
> depth $k/s$, which is much deeper than $k$. Recall therefore falls as $s$ falls unless more buckets are probed.

**Results (SISAP23 300K, IVF and MLP).**
- **Baseline (B0, nprobe 16):** routed is 12–38× faster than exact. IVF R@10 falls 0.953 → 0.874 from $s = 1$ to 3%.
- **Probing study:** preregistered; validation/test split; 720,000 policy records; 0 native/simulation mismatches.
  - Recall-calibrated policies (R1/R2) keep mean R@10 ≥ 0.94 at every selectivity. B0's recall cliff (0.876 IVF,
    0.835 MLP at 3%) disappears, at 1.8–6× B0's latency.
  - Choosing on cost alone is unsafe. The cost model routed $s = 0.01$ at nprobe 8 and got recall 0.66, where exact
    gives 1.0 in 0.33 ms.
  - A **mean-recall target is not a per-query guarantee**: 20–28% of queries fall below target and the 5th percentile
    is 0.8.
  - **Correlated (slab) filters break calibration transfer** (exploratory). This is the one case where filter-aware
    routing could matter.
  - The preregistered adoption criterion was **not met**, so B0 stays the default.

**Statistical lessons from this study.**
- *Bins keyed by nominal selectivity* sent a 3% filter to the 0.1% bin; they were rekeyed by *estimated* selectivity.
- An unweighted linear cost fit mispredicted by up to 10×. Relative-error fits and empirical log-log tables were
  documented as Amendment 1, *before* evaluation.
- An adoption rule with a strict "≤" fails on timing noise when two policies run the *identical* plan. Future rules
  need a tolerance band for identical plans.

---

## Part XI: nearest-neighbour-supervised routing (19–20 November)

**Sources:** `research/lmi/supervised_routing/`: preregistration, design, training report, results, final report.

**Idea (the theory from §IX applied).** Train the router on the right target. Let $v_b(q)$ be the number of $q$'s true
$k$ neighbours in bucket $b$. Listwise cross-entropy against $t_b(q) = v_b(q)/k$ is a proper scoring rule, so its
population minimiser is
$$p^*(b \mid q) = \mathbb{E}[\,t_b(q) \mid q\,],$$
the **expected neighbour mass**. Ranking by it maximises expected coverage at fixed nprobe. Two consequences follow:
- **Cost-aware ranking:** maximising coverage under a candidate budget is a knapsack problem whose greedy order is
  $\mathbb{E}[v_b]/n_b$, which is the logits minus $\log n_b$.
- **A linear correction to centroid scores** ($W = C + \Delta$) has *zero* extra routing cost.

**Design.** Arms ranging from centroid IVF to full linear corrections, all over a fixed native IVF-8192 partition at 1M.
Router selection used validation only. Promotion to 10M required at least 10% fewer candidates at native matched recall.

**Result: negative.** The best router (A4occ) gave **−1.1%** candidates at native matched 0.95. A *post hoc* continuous
analysis suggested −10.3%, at the edge of the threshold, but post hoc. The stopping rule held, and no 10M job ran.

**Why the training-data budget mattered.** The full correction has $B \cdot d \approx 6.3$M parameters at
$B = 8192$, but only about 5,700 training queries were available, which is enough to fit biases, not full matrices.
The signal is real but small. Learned routing over a fixed partition on CLIP embeddings has little room to beat
geometry.

---

## Part XII: repository consolidation and upstream sync (20 November)

**State before:** 18 worktrees, 23 local and 18 GitHub branches, 3 open PRs on the v1.19 line, three pieces of work
that existed only as uncommitted files, and about 200 GB of build directories.

**What was done, in reversible order:**
1. **Rescue:** the uncommitted work was committed on `rescue/*` branches; large evidence (259 MB) was copied to
   `research-admin/archive/`.
2. **Tags:** 25 `archive/<branch>` tags and 5 `exp/<date>-<name>` tags were created and pushed. Every commit on every
   branch was verified reachable from a tag before anything was retired.
3. **`main`:** the merge of the static ablation, filtering and supervised routing (no conflicts) became GitHub's
   default branch.
4. **Upstream sync to v1.20.2**, by **merge** (not rebase), because experiment provenance files (`source-head.txt`, PBS manifests, reports) cite exact SHAs.
   Upstream had split `segments_searcher.rs` into a module and added `VectorIndexRead::filter_checks`; the port
   needed a semantic adaptation for LMI. The full `segment`, `collection` and `shard` suites passed
   (1,971 + 386 + 196 tests).

**Lessons.**
- **Use each Git mechanism for one purpose:** branches for active work, tags for frozen revisions, experiment folders
  for evidence, documents for methodology.
- **A tool environment in a temporary directory is a liability.** The PyTorch venv that every build used lived in a
  session scratch directory and vanished. It is now documented in `research/BUILD_ENVIRONMENT.md`.

---

## 14. Cross-cutting lessons

### Scientific method
1. **Controls before claims.**
   - The centroid/affine control (Phase F) and the oracle bound (F.4, G) turned "the MLP is worse" into "the
     MLP is worse *because* of X".
   - Every later study kept a control that needs no learning.
2. **Separate selection from evaluation.**
   - Validation-locked operating points (F.3, Tier 1 Amendment 01, filtering, supervised routing) prevent choosing
     the configuration that happens to look best on test.
3. **Preregister, then amend in writing.**
   - Preregistrations with dated amendments (Tier 1–3, H-10M, filtering, supervised routing) made it possible to
     report negative results without doubt about post-hoc tuning.
   - Stopping rules were honoured (no 10M supervised job).
4. **Compare at matched *actual* work, not nominal knobs.**
   - Nominal budgets misled the DLI allocation study.
   - nprobe is not candidates when buckets are uneven.
5. **Paired query bootstrap intervals** are the default uncertainty. They cover query sampling only. Seed and
   construction variance remained unmeasured in most studies (single replicates), and this is a stated limitation.
6. **Mean recall is not per-query recall:** report tails (share below target, 5th percentile).
7. **Know which contract a number measures:**
   - official gold against stored-Float16 exactness;
   - learned against fallback results;
   - in-process against REST latency;
   - debug against release.

### Systems engineering
1. **Separate the gates.** PBS "completed", process exit 0, artifact published and scientifically accepted are four
   different things.
2. **Visibility is not durability.** A search seeing a point does not mean it is flushed (the 1,679 missing points).
3. **Fail closed.** No silent fills, no relaxed gates, no wait loops that cannot fail, no indefinite optimizer retries.
4. **Evidence is read-only.** Diagnose on verified copies (byte-for-byte inventories), never on the original.
5. **Provenance.** Hash inputs, scripts, binaries and outputs; record the source SHA the job *actually ran*.
6. **Low CPU does not mean stalled.** Distinguish wall time from CPU time; inspect I/O and page faults.
7. **Storage locality dominates at scale.** Cold NFS (§VI) and Cold DLI members (§VIII) dwarfed algorithmic differences.
8. **Floating-point non-associativity is real:** different reduction orders flip near-ties, and Float16 storage
   creates exact ties.

### Process
1. **Coding agents** (Codex, then Claude Code) did most implementation and documentation. That needed explicit rules:
   - no push without asking;
   - verify state before acting;
   - an independent audit of agent-written code (continuation 41 found 12 issues);
   - reports that label evidence status.
2. **Handoffs must state verifiable state** (SHA, job IDs, paths), because sessions end mid-task.
3. **Branch sprawl** came from using branches as experiment snapshots. One maintained line plus tags fixed it.

---

## 15. Register of negative results and dead ends

| Date | What was tried | Result | Why it matters |
|---|---|---|---|
| 26 Sep | MLP router (Phase F baseline) | +42% candidates against centroid | Undertraining diagnosed (F.2) |
| 30 Sep | Retrieval-target MLP at 100K (F.4) | Recall +0.025, but more candidates | Idea revisited in §XI |
| 3 Oct | tch acceleration at $B = 10{,}000$ | Tie fallback on every row; no speedup | Kept opt-in only |
| 6 Oct | Three-way Plain/HNSW/LMI planner | Not local: one index per field | Planner deferred |
| 3 Nov | Static LMI against HNSW at 10M | 27× slower at recall 0.95 | Main static result |
| 16 Nov | DLI against static at 300K | −3.4 pp recall, +58% p50 | First DLI cost signal |
| 16 Nov | "Post-update recall 0.997" | It was the exact fallback; learned recall was 0.784 | Count result provenance |
| 17 Nov | T = 2 temperature routing | Helps only at very low work | Not promoted |
| 17 Nov | 128-bucket models for scaling | Rejected for high recall | Better assignment ≠ better retrieval |
| 18 Nov | H-10M: DLI against static | Static needs 45% less work | Buffer and slot priority are the cause |
| 18–19 Nov | MLP against centroid on a fixed partition | Never better; IVF 0.55× candidates | Static LMI router question closed |
| 19 Nov | Cost-model filter probing | Recall 0.66 at $s = 0.01$ | Cost-only planning is unsafe |
| 19 Nov | Filter-probing adoption | Criterion not met | B0 stays default |
| 20 Nov | Neighbour-supervised routing | −1.1% candidates; gate failed | Stopped per rule |
| Sep–Oct | L2P prototype routing; statistical router | Discussed, not implemented | A statistical router stays the natural control for any future router |

---

## 16. Documentation audit

### Sources audited (20 November 2026)
- **v1.19 line**, reachable only through tags: `docs/lmi-phase-{e,e2,f,f2,f3,f4,g,g2,s,s3a..s3e}.md`,
  `docs/lmi-sisap2023.md`, `docs/lmi-upstream-sync.md`, `docs/thesis-progress-*.md`, the supervisor brief, and 12
  documents in `tools/lmi/sisap23_matched/` (tags `archive/thesis/lmi-integration`, `archive/thesis/lmi-mainline`,
  `archive/thesis/lmi-qdrant-sync`, `archive/rescue/sisap23-small-lifecycle-wip`).
- **v1.20 line**, on `main`: about 85 documents under `research/` (DLI 50, LMI 20, filtering 3, supervised routing 7,
  validation 1).
- **Windows `Learning stuff/`:**
  - `Progress_Ladder/` (Phases A–G reports, incident report, static research report, comprehensive audit, roadmap);
  - `Records/` (meeting transcripts and plans, September–October);
  - `DEVELOPMENT_JOURNAL.md`, which is a copy of the early DLI journal.
- **Evidence trees:** `research-admin/experiments/*` (manifests, provenance, logs).
- **Git:** 234 non-merge project commits, 23 Sep – 20 Nov.

### Gaps and inconsistencies found
1. **Phases A–D and the preliminary characterisation exist only as PDFs on Windows.** They have no Git record and no
   source revision; their dates come from file timestamps.
2. **The v1.19 phase reports are invisible from `main`.** Readers need the archive tags (see §17).
3. **There was no single timeline before this journal.** The static report stops at 12 November; the DLI journal
   covers only DLI.
4. **Gap 23 Oct – 10 Nov in Git.** The accepted campaign ran on the cluster in this period and its analysis lived on
   Windows (`final-scientific-analysis-20261103`). The commit history alone understates this work.
5. **Superseded statements that remain in older documents:**
   - "LMI SISAP2023 181 QPS comparable to ours": the 181 QPS figure is SISAP 2024, 100M, recall 0.8, not comparable.
   - Baseline filtering finding 5: filter checks explain only about 0.75 ms of the 2.2 ms (erratum added).
   - The H-10M job's own `analysis-summary.txt`, whose matched-work section was biased (superseded by
     `TEN_MILLION_VECTOR_ANALYSIS.md`).
   - The 65K selection interpretation, revised after H-10M (`CANDIDATE_SELECTION_REPORT.md`, addendum).
6. **Writing style.** Most reports were written by coding agents in a dense, hedged style with heavy inline
   provenance. They are reliable but hard to read. This journal is the human-readable entry point; the reports remain
   the evidence.
7. **Meeting decisions are not in the repository.** Supervisor guidance lives in `Records/` transcripts (Italian and
   English), and only the 24 September alignment was turned into a document.

---

## 17. Where the evidence lives

| Era | Code | Documents | Raw evidence |
|---|---|---|---|
| v1.19 (Sep–Oct) | tags `archive/thesis/lmi-integration`, `archive/thesis/lmi-mainline` | `git show archive/thesis/lmi-integration:docs/lmi-phase-g.md` (and so on) | MetaCentrum `/storage/budejovice1/home/nico/lmi-metacentrum/`; Windows `Progress_Ladder/` |
| SISAP23 campaign | tag `archive/rescue/sisap23-small-lifecycle-wip` | `tools/lmi/sisap23_matched/*.md` in that tag | MetaCentrum `results/sisap23-matched-final-20261028-01`; `research-admin/archive/rescue-20261120/` |
| v1.20 port | `main` | `research/v120-validation/` | `research-admin/experiments/2026-11-11-*` |
| DLI | `main` (`exp/2026-11-18-dli-h10m`) | `research/dli/` | `research-admin/experiments/2026-11-12-dli-architecture/` (bulk archived to external disk, see `BULK_ON_EXTERNAL.txt`) |
| Static ablation | `main` (`exp/2026-11-19-static-tier3b`) | `research/lmi/` | `research-admin/experiments/2026-11-18-static-lmi-ivf/` |
| Filtering | `main` (`exp/2026-11-19-filter-probing-300k`) | `research/lmi/filtering/` | `research-admin/experiments/2026-11-19-lmi-adaptive-filtering/` |
| Supervised routing | `main` (`exp/2026-11-20-supervised-routing`) | `research/lmi/supervised_routing/` | `research-admin/experiments/2026-11-19-static-supervised-routing/` |

**Next priorities** (agreed 19–20 November): Tier 3B synthesis into the thesis; thesis writing; DLI replication
against Sommerová's reported results; CLI; and finally external API tests across Plain, HNSW, LMI, DLI and CLI with
filtering, against one `main` build.
