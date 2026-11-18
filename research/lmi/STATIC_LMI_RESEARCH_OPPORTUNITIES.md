# Static LMI: research opportunities, prior work and reuse audit

Written 2026-11-18, before Tier 1 results. It records what is already known, so that new experiments answer only open questions.

## 1. Source inventory

**Primary documents read**

| Source | Location | What it establishes |
|---|---|---|
| STATIC_LMI_RESEARCH_REPORT.md | `Learning stuff/Progress_Ladder/` | The accepted Nov 3 10M campaign (§6), Phase F–F4 100K router studies (§5), the Phase G decomposition (§7) |
| STATIC_LMI_INDEPENDENT_AUDIT.md | `research/lmi/` | The underfill fix; about 91% of LMI query time is candidate scoring |
| TEN_MILLION_VECTOR_ANALYSIS.md | `research/dli/selection_frontier/` | H-10M: 548-bucket static LMI, router/oracle costs at 10M |
| CANDIDATE_SELECTION_REPORT.md (plus addendum) | same | 65K frontier |
| MODEL_PARTITION_STUDY.md | `research/dli/unified_architecture/` | continuation28 65K 2×2 placement × routing; 128/256/548-output study |
| lmi-sisap2023.md | `work/qdrant/docs/` | Accepted 300K (B=548) and 10M (B=3162) build configurations |
| Historical matched harness | `work/qdrant-sisap-matched-bench`, `lib/collection/src/tests/lmi_sisap23_matched.rs` (v1.19) | Index-level measurement boundary for LMI/HNSW/Plain |
| Upstream LMI, SISAP23 challenge branch | `work/LearnedMetricIndex`, `origin/paper-sisap23-indexing-challenge`, `res.csv` (2,088 runs) | Published LMI hyperparameter sweeps |
| Native code | `lib/segment/src/index/lmi_index/{training,spherical_kmeans,build,routing}.rs` | Actual teacher, router and posting semantics |

**Web sources** (fetched 2026-11-18)

| Source | Facts used |
|---|---|
| MUNI publication 2428378: Procházka, Slanináková, Čerňanský, Oľha, Antol, Dohnal, *Scaling Learned Metric Index to 100M Datasets*, SISAP 2024 (LNCS, 2025), doi 10.1007/978-3-031-75823-2_22 | Improvements listed: parallel query processing, adaptive memory, **pre-constructed memory buckets for contiguous access**, **k-means → spherical k-means**, **fewer epochs and smaller training samples**. No numbers on the landing page. |
| Mendeley 10.17632/3dp7jfv2vh.1 | SISAP23 LMI challenge raw data: a stripped-down LMI with 2,088 experiments |
| FAISS wiki, *Guidelines to choose an index* | IVF list count K = 4√N to 16√N; 30·K to 256·K training vectors; at 1M–10M the suggested index is IVF65536 with an HNSW coarse quantizer |
| FAISS wiki, *Faiss indexes* | IndexIVFFlat = inverted lists with exact post-verification; a Flat coarse quantizer holds the centroids; nprobe is set at query time; nlist ≈ C·√N |
| Qdrant indexing documentation | The only dense index is HNSW (m = 16, ef_construct = 100 by default); there is no IVF; payload-aware HNSW edges support filtering |

The web agent could not search arXiv. Antol et al. (*Learned Metric Index — proposition…*, Information Systems 2021) and Slanináková et al. (*Reproducible experiments with LMI Framework*, Information Systems 118, 2023) are cited from the local `Learned/` PDFs and the upstream repository, not re-read here. No claim below depends on learned-ANN papers that were not retrieved, such as BLISS, Neural LSH or SOAR; they are listed only as unverified pointers.

`Records/statistical_router_research_note.pdf` could not be rendered in this environment (no poppler) and is not used.

## 2. What the current native LMI actually is (verified in code)

- **Teacher.** Spherical k-means (`spherical_kmeans.rs`, called for cosine at `training.rs:335`). Unit centres are seeded from a shuffled reservoir sample, there are `kmeans_iterations` Lloyd steps (5 in every accepted configuration), and empty clusters keep their centre.
- **Router.** Two-layer MLP, 768 → H (ReLU) → B, trained with Adam (learning rate 1e-3, batch 256) on cross-entropy against hard teacher labels, over the same sample. Training is single-threaded (`tch::set_num_threads(1)`).
- **Postings.** Every point goes to `argmax` of the MLP. A posting list is therefore the MLP's partition, not the teacher's Voronoi cells.
- **Query.** Route; take the top `nprobe` buckets (config-fixed); score candidates exactly. If the result underfills, double `nprobe` (the fix in `15727cc2e`). Non-default `SearchParams`, filters and quantization fall back to exact search.
- **Routing cost** in multiply-accumulates: d·H + H·B. Exact centroid routing costs d·B. The two are equal at B = d·H / (d − H) = 1,536 for d = 768, H = 512, so below about 1.5K buckets the MLP is *more* expensive than exact centroid ranking.

## 3. Experiment reuse matrix

| Question | Prior experiment | Configuration | Data / queries | Result | Reusable? |
|---|---|---|---|---|---|
| MLP versus centroid routing | Phase F, F2, F3 (Sep 2026) | B = 64, S = 2,048, H = 64 | 100K LAION, 200–1,000 held-out queries | Near parity. F: centroid needs fewer candidates. F2/F3: MLP 3–10% fewer at 0.90. | Direction only: one tiny B, an old code path |
| Placement × routing 2×2 | continuation28 (DLI member) | 65K, native 548-output member | 32 queries | At about 2,000 work, R@30 coverage: teacher/teacher 0.879, model-place/teacher-route 0.874, teacher-place/model-route 0.854, model/model 0.853. **Centroid routing beats the MLP by about 2.5 pp.** | Yes, as prior evidence; too few queries for a conclusion |
| Oracle versus router gap | Phase F4 | B = 64, 96K | held-out queries | Oracle 0.980 versus MLP 0.882 at P = 4. A retrieval-supervised MLP reached 0.907–0.911. | Yes: a modest objective gain was already found |
| Partition loss versus routing gap at 10M | Phase G | B = 3,162, S = 250K | 9,980 official queries | At nprobe 4: oracle 0.949, router 0.819, returned 0.815 | Yes |
| Bucket count | Accepted Oct/Nov 10M (B = 3,162) against H-10M static (B = 548) | S = 250K against 32,768 | 9,980 queries against 256 test queries | R@10 ≈ 0.96 needs about 117K candidates at B = 3,162 versus about 500K at B = 548 | Indicative: query sets and recall definitions differ |
| Model outputs 128/256/548 | continuation36 | 65K DLI members | 32 queries | 95%-coverage work 9,825 / 7,450 / 5,827: **more buckets, less work** | Yes, as prior evidence |
| 10M latency, LMI versus HNSW | Nov 3 matched campaign (v1.19, deimos11) | LMI B = 3,162, nprobe 1–256; HNSW m16/ef_c100 and m32/ef_c200 | 9,980 queries, 1,000-query panel × 3 | At ≈ 0.96 R@10: LMI 36.9 ms against HNSW 1.34–1.38 ms. Index-only bytes 48.6 MB against 516/702 MB. Builds 1,989 / 6,911 / 22,883 s. | Yes, as the historical reference. Not v1.20, not matched to new variants. |
| Published LMI hyperparameters | SISAP23 challenge `res.csv` | categories 10–200, PCA 32/96, MLP, epochs 40–200 | official 10M | Best CLIP-768 run: R@10 0.981 at 10 categories / 4 searched (scans ≈ 40%), about 0.7 s per query | Shows that published LMI configurations are coarse; no partition-efficiency evidence |
| Float32 versus Float16 | none | — | — | **Not possible from this data.** The SISAP23 10M corpus is distributed as Float16 (`emb` dtype float16); only the queries are Float32. | Study G dropped; no higher-precision originals exist |

**Not redone:**
- the 10M HNSW build (6.9 ks and 22.9 ks), unless Tier 3 needs v1.20 matching;
- Phase F/G-style analyses at 100K;
- the SISAP23 challenge configurations.

## 4. Candidate improvements

The basis column separates demonstrated from speculative.

| # | Idea | Source | Problem addressed | In our LMI? | Change needed | Expected effect | Basis | Cheapest valid test |
|---|---|---|---|---|---|---|---|---|
| 1 | **More buckets** (√N to 16√N) | FAISS guidelines; our 548 → 3,162 data | Coarse partitions force large scans | Configurable (`n_buckets`) | Configuration only | Fewer candidates; routing grows | **Demonstrated** (indicative at 10M, and at 65K) | Tier 1 B-sweep |
| 2 | **Sample size scaled with B** (30–256 per list) | FAISS | Undertrained centroids at large B | Configurable (`sample_size`) | Configuration only | Better partition at large B; build time grows | Standard practice | Tier 1 native-versus-converged teacher |
| 3 | **Centroid routing, i.e. IVF-Flat** | FAISS, standard IVF; continuation28 | MLP routing loss, MLP cost below 1.5K buckets | No | A linear "router" with W = C, b = 0 expresses exact spherical centroid ranking with the **existing** `RouterLayer::Linear`; postings by argmax become Voronoi assignment | Same or fewer candidates; no training | Prior evidence (65K, 100K) | Tier 1 2×2 |
| 4 | Smaller or linear MLP, fewer epochs | Procházka 2024 (fewer epochs); routing-cost algebra | Training and inference cost | Configurable (`hidden_dim`, `epochs`); linear needs H = 0 support | Configuration, plus a small change for linear | Cheaper build and routing | Speculative for recall | Tier 1 D4 |
| 5 | Budget stopping instead of fixed nprobe | H-10M frontier; IVF practice | Imbalanced buckets make nprobe a poor work control | Research harness only | A small change in `routing.rs` | Smoother work control; same order | Demonstrated in replay (realized-work framing) | Tier 1: both stopping rules from the same traces |
| 6 | **Bucket-contiguous storage order** | Procházka 2024 ("pre-construction of memory buckets for contiguous access") | Random 1.5 KB reads per candidate from the vector store | No: posting offsets are scattered | Build the LMI segment with points inserted in bucket order, so each posting list is a contiguous offset range; no new storage abstraction | Lower cost per candidate (locality), same candidates | Published improvement; untested here | Tier 3 micro-benchmark on one built segment |
| 7 | Scalar128 scoring plus exact rerank | Our DLI A2 (1.48× at 65K Collection level) | Cost per candidate | Implemented for DLI members, not for static search | Wire the existing quantized scorer into static `LmiIndex::search` | About 1.5× faster scoring | Demonstrated at 65K (DLI) | Tier 3 factor |
| 8 | Retrieval-oriented objective | Phase F4 (+2.5–3 pp); query-aware routing literature | Router ranking gap | No | New training path | Modest, from F4 | Partially demonstrated, small | Only if Tier 1 shows a large MLP deficit at fixed partition |
| 9 | HNSW coarse quantizer over centroids | FAISS IVF65536_HNSW32 | Routing cost at very large B | No | Larger change | Needed only at B ≫ 10⁴ | Standard | Only if the promoted B exceeds about 16K |
| 10 | Top-k selection instead of a full sort of logits | Engineering | Routing constant factor | Full sort | Small change | Below 0.2 ms | Speculative | Profile first |

**Rejected or deferred**
- Deeper or hierarchical neural models: the user's constraint, plus no supporting evidence.
- New storage layouts beyond insertion order.
- PCA (SISAP23 challenge PCA runs are coarse; dimension reduction changes the recall semantics).
- Float32 corpus: not available.
- Filtering: a separate, high-cost capability.
- CLI: Phase 5.

## 5. Plan

1. **Tier 1** (this document's companion, `ivf_ablation/PREREGISTRATION_TIER1.md`): offline replay at 1M of the 2×2 × B × teacher factorial plus the D4 simplifications.
2. **Tier 2:** at most 4 finalists at 10M offline on HPC: the same lab and the official gold.
3. **Tier 3:** native v1.20 index-level latency for the finalists, against HNSW and Plain, on one node. Ideas 3, 6 and 7 can all be realized without new abstractions: a centroid "router" through the existing Linear layer, insertion order, and the existing quantized scorer.
