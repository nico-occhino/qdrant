# Progress Report: Learned Metric Index Integration in Qdrant

## Executive summary

This MSc thesis / MUNI traineeship investigates whether a **Learned Metric Index (LMI)** can become a legitimate physical vector index inside a database system, rather than an external machine-learning prototype placed beside one. The project has integrated an experimental LMI into Qdrant so that Qdrant retains authority over vectors, identifiers, visibility, deletion, segment lifecycle, snapshots and final distance scoring. LMI supplies only a learned routing function and its auxiliary bucket postings.

The engineering result is substantial: a trained LMI can be configured per vector field, built by Qdrant's optimizer from Qdrant-owned vectors, persisted with a segment, reopened without retraining, restored from a snapshot, rebuilt for a new immutable segment after updates, and served through native and universal read-only paths. This is tested through focused unit/integration tests and isolated HTTP lifecycle acceptance drivers. It is not a production-readiness claim.

The scientific result is nuanced. In the first controlled LAION experiment, an undertrained MLP router gave recall near direct centroid routing but required about 42% more candidates, because its learned partition was badly imbalanced. Phase F.2 identified inadequate optimization as an important cause: holding data, teacher, architecture and optimizer fixed while increasing epochs reduced empty MLP buckets from about 24 to 1--2. A validation-selected improved MLP reduced candidate work by 6.5--8.4% relative to direct centroid routing around Recall@10 near 0.90 on an internal holdout. This advantage was not uniform at high recall, and there is no demonstrated query-latency advantage over HNSW.

The central research question has therefore sharpened. The question is not whether an MLP can simply replace HNSW. It is: **under which data distributions, recall targets, and rebuild/write workloads does learned routing justify its additional model complexity relative to conventional physical indexes and simpler centroid routing?**

## 1. Problem and terminology

A vector DBMS stores high-dimensional numerical representations of data: an image, document, audio clip, user, or product can be encoded as a vector \(x \in \mathbb{R}^d\). A nearest-neighbour query asks for the stored vectors most similar to a query vector under a configured metric, such as cosine similarity, dot product, or Euclidean distance.

Exact k-nearest-neighbour search compares the query with every eligible vector. It is authoritative and provides ground truth, but its work grows with corpus size. Approximate nearest-neighbour (ANN) indexes reduce that work by selecting a promising subset before scoring. HNSW is a widely used graph-based physical access path for this role.

An index in a DBMS is not an isolated ML algorithm. A realistic request path is:

```text
query/request
  -> logical/API interpretation
  -> physical access path
  -> candidate generation
  -> visibility, deletion and filtering semantics
  -> vector scoring
  -> top-k result
```

The index may propose candidates, but the DBMS remains responsible for which points are visible and how the result is scored and returned. This distinction is the foundation of this work.

Learned indexes are motivated by the observation that classical data structures often encode assumptions about a data distribution. A trained model may replace or assist one access-path decision. That does **not** imply that learned indexes universally replace conventional structures; their value is empirical and workload-dependent.

## 2. Learned Metric Index concept

The implemented LMI follows this conceptual pipeline:

```text
vectors -> sample -> KMeans -> pseudo-label/bucket assignment
        -> neural classifier (router)
        -> classify corpus into learned buckets
        -> query top-nprobe buckets
        -> candidate union -> exact scoring -> top-k
```

KMeans is a **teacher**: it produces pseudo-labels for training. The MLP is a learned router trained to predict those labels. Critically, final LMI postings are created from the MLP's predictions for corpus vectors; they are not copied from the KMeans assignment. Thus the MLP defines a learned partition of the corpus.

Two separate parameters are essential. \(k\) is the number of neighbours requested by the user. `nprobe` is the number of learned buckets searched. The model approximates candidate selection, not the metric. Once candidates are chosen, Qdrant performs exact scoring under its configured metric.

ANN quality is measured against exact Qdrant/Plain results:

\[
\operatorname{Recall@k}(q) =
\frac{|\operatorname{exact\_top\_k}(q) \cap \operatorname{approximate\_top\_k}(q)|}{k}.
\]

Candidate count and candidate fraction measure the amount of exact scoring work. Teacher-label accuracy is useful for diagnosis, but it is not Recall@k and is not the ANN objective.

## 3. Why centroid and affine controls are necessary

For Euclidean KMeans centers \(c_j\), nearest-centroid assignment is

\[
\arg\min_j ||x-c_j||^2.
\]

Expanding the square gives

\[
||x-c_j||^2 = ||x||^2 - 2c_j^T x + ||c_j||^2.
\]

Because \(||x||^2\) does not depend on \(j\), this is equivalent to

\[
\arg\max_j [2c_j^T x - ||c_j||^2].
\]

So an affine classifier can reproduce Euclidean nearest-centroid routing exactly. A nonlinear MLP is not necessary merely to simulate KMeans boundaries. The experiment therefore compares direct centroid routing, affine centroid routing, and nonlinear MLP routing. Phase F verified identical bucket orderings and retrieval behaviour for direct and affine routing under the tested precision and deterministic tie rule.

## 4. Database-owned architecture

The architectural decision adopted after supervisor discussions is simple: **Qdrant must remain the database**. Qdrant owns authoritative vector storage, external/internal IDs, `PointOffsetType` mappings, deletion and visibility semantics, query semantics, optimizer scheduling, segment publication, snapshots and final scoring. LMI owns only its reproducible sample, KMeans teacher, MLP router, learned postings, temporary training memory and auxiliary persisted routing state.

This excludes a duplicate vector store, HDF5 as operational storage, Python or PyO3 on the production request path, a custom distance engine, and query-time retraining. Python is used only for experiment orchestration and analysis. Rust/tch performs optional build-time training; exported native Rust code serves inference.

```text
collection configuration
  -> Qdrant optimizer
  -> SegmentBuilder
  -> Qdrant VectorStorage + IdTracker
  -> reproducible eligible-vector sample
  -> CPU Lloyd KMeans -> pseudo-labels
  -> Rust/tch MLP -> native MlpRouter
  -> classify target vectors -> bucket -> PointOffsetType postings
  -> persist lmi_state.json -> publish segment -> normal reopen

dense query -> metric preprocessing -> native MlpRouter
  -> top-nprobe buckets -> candidate PointOffsetTypes
  -> Qdrant validity/deletion/context checks
  -> BatchFilteredSearcher / RawScorer -> top-k
```

This is stronger than calling an LMI library from Qdrant: the candidate generator is integrated while Qdrant's established correctness semantics remain authoritative.

## 5. Project evolution

The history below is supported by the repository and phase reports.

**Phase A** (`a628691f6`) instrumented and understood HNSW/Plain dispatch. This established how Qdrant selects physical search paths and prevented building LMI on an unverified test-only seam.

**Phase B** (`4f3bd3798`) introduced the LMI enum/index structure. This was structural integration, not yet a learned ANN index; temporary paths could delegate to Plain.

**Phase C** established the critical seam: arbitrary `PointOffsetType` candidates can flow into Qdrant's `BatchFilteredSearcher` / `RawScorer`. This preserves database-owned deletion, batching, top-k and metric semantics. Raw scoring alone was insufficient because a DBMS must also enforce visibility and deleted-point handling.

**Phase D** added native Rust MLP routing and router-predicted postings, while retaining Qdrant scoring. Torch/export parity checks established that native `MlpRouter` inference matches the trained model. At this stage model state was deliberately static/test-oriented.

**Phase E** (`7029bcd0f`) made construction database-owned: Qdrant samples its own storage, runs local stable-Rust Lloyd KMeans, trains a Rust/tch MLP, exports the native router, classifies corpus vectors, persists `lmi_state.json`, and builds via the optimizer/`SegmentBuilder`. A standalone KMeans dependency required nightly features, so a bounded seeded Lloyd implementation was used locally. Persistence could not be deferred because `SegmentBuilder` publishes then reloads the completed segment; memory-only learned state would vanish. The phase also repaired shared-config compilation gaps, HNSW/LMI update-time exclusivity, unrelated lockfile churn and unsafe persisted-routing mutation.

**Phase E2.1** (`a807381fb`) proved snapshot creation and fresh-storage restoration, learned reopening without retraining, and rejection of malformed or structurally inconsistent state.

**Phase E2 final** (`a1987b504`) completed the accepted lifecycle scope: deferred eligibility independent of HNSW enablement, optimizer-owned rebuild after corpus changes, replacement by a new segment-local trained state, retirement of old state, restart after rebuild, mixed named LMI/HNSW fields, and universal/read-only trained-LMI support.

Normal writes do not mutate a neural model online. The lifecycle is:

```text
immutable trained segment S_t + later writes
  -> Qdrant mutable/deferred state
  -> optimizer
  -> new immutable segment S_(t+1)
  -> newly trained LMI for S_(t+1)
```

This is segment-level rebuild/adaptation, not continual learning.

## 6. Engineering result and lifecycle evidence

The implementation can truthfully be described as a Qdrant-owned physical vector index whose routing state is built from Qdrant storage, persisted with the segment, reopened without retraining, restored through snapshots, rebuilt by the optimizer when the corpus changes, and served through native and universal read-only segment paths. Qdrant retains vector authority, visibility and final scoring.

Focused evidence includes 36 existing LMI integration tests (candidate scoring 2, dummy 1, Phase C 10, Phase D 12, Phase E/E2 11), six selected unit tests, collection/segment/edge checks, default and training-enabled server builds, and two isolated HTTP lifecycle drivers. The snapshot driver checks fresh storage, matching state/IDs/scores, no build/train on restore, learned open/search, and documented exact/filter fallbacks. The lifecycle driver checks deferred promotion, rebuilt postings, updates/deletions, mixed names and restart. These checks qualify the accepted lifecycle scope; they do not establish production readiness, distributed crash tolerance, exhaustive remote filesystem support, or a cryptographic corpus-to-state identity binding.

## 7. Phase F controlled baseline

Phase F used a LAION source array of 100,000 x 768 float16 vectors, exported as float32 for Qdrant. The indexed corpus contained 99,780 vectors; there were 20 separate warmups, 200 held-out measured queries, two trials per operating point, cosine distance and \(k=10\). The baseline LMI used 64 buckets, a 2,048-vector sample, hidden dimension 64, 30 epochs, batch 256, 20 Lloyd iterations and seed 42. HNSW used `m=16`, `ef_construct=100`, `full_scan_threshold=0` and one indexing thread.

Measurements are in-process, not HTTP end-to-end latency. HNSW uses the native `VectorIndexRead` path. Centroid, affine and MLP use a common candidate-scoring harness. HNSW candidate visits are not exposed by this seam, so they remain unavailable; `hnsw_ef` is not substituted as a false candidate count.

| Around Recall@10 0.90 | Recall | Mean candidates | p50 |
| --- | ---: | ---: | ---: |
| HNSW (`ef=16`) | 0.9000 | unavailable | 0.16 ms |
| Centroid (`nprobe=4`) | 0.8975 | 10,197 | 3.53 ms |
| Affine (`nprobe=4`) | 0.8975 | 10,197 | 3.51 ms |
| Baseline MLP (`nprobe=4`) | 0.9015 | 14,504 | 4.91 ms |

The baseline MLP used about 42% more candidates than centroid for similar recall. Router compute was small; candidate scoring dominated. Its corpus partition had 40 active and 24 empty buckets, maximum size 7,361, versus centroid's 64 active, no empty buckets and maximum size 4,501. The main baseline issue was partition/candidate inflation, not forward-pass cost.

One observed construction comparison was LMI 5.268562424 s and 1,221,407 index bytes versus HNSW 48.610999886 s and 3,857,946 bytes. Shared vector storage is excluded. This is a single build observation, not a general build-speed theorem. HNSW had dramatically lower measured query latency; LMI's possible tradeoff concerns build/rebuild cost and auxiliary index size, not demonstrated workload-level benefit.

## 8. Phase F.2 diagnosis

The initial finding is that the 30-epoch MLP was substantially underfit. All 24 empty output classes had training examples: support ranged from 1 to 29, median 7, versus median 48 for active classes. They represented 10.21% of training labels and 7.00% of full-corpus teacher labels. Baseline training agreement was 71.73%; remaining-corpus agreement was 64.18%. Thus missing labels were ruled out, while rare-class fitting and approximation remained plausible.

Holding the 2,048 samples, teacher, architecture and optimizer fixed and increasing epochs produced:

| Epochs | Empty buckets | Training teacher agreement | Validation agreement |
| ---: | ---: | ---: | ---: |
| 30 | 23--24 | about 72% | 55--63% |
| 60 | 9--10 | about 92% | 76--79% |
| 120 | 1--2 | 99.7--99.8% | 83--84% |

This is direct intervention evidence that insufficient optimization materially contributed to collapse. It does not explain every error. With 2,048 samples and 120 epochs, nearly perfect sample fit still leaves 79.5--79.9% remaining-corpus agreement. At 16,384 samples and 60 epochs, remaining-corpus agreement reached about 86.2%, validation agreement 87--90%, and 1--2 buckets remained empty. Larger samples also create more optimizer updates at fixed epochs, so their effect cannot be attributed purely to coverage.

Errors are structured: 68.85% of wrong corpus assignments go to one of the four nearest teacher centroids, and 83.96% to one of the eight nearest. This supports local geometric confusion rather than random error, but does not alone prove why retrieval remains useful.

### Routing/corpus-assignment decomposition

At baseline `nprobe=4`, counterfactual diagnostics separate teacher query routing (T) from MLP query routing (M), and teacher postings (T) from MLP postings (M). MT and TM are diagnostic hybrids, not deployable indexes.

| Pipeline | Meaning | Recall@10 | Mean candidates |
| --- | --- | ---: | ---: |
| TT | teacher query + teacher postings | 0.8975 | 10,196.6 |
| MT | MLP query + teacher postings | 0.8155 | 11,628.4 |
| TM | teacher query + MLP postings | 0.8440 | 11,627.0 |
| MM | MLP query + MLP postings | 0.9015 | 14,503.6 |

Changing query routing alone hurts; changing corpus assignment alone also hurts. Together they form a new internally consistent partition which recovers roughly the original recall with substantially more work. Relative to TT, MM loses 0.53 exact neighbours/query and recovers 0.57. Candidate accounting gives about 1,432 net extra candidates from query-routing change and about 2,875 from corpus reassignment, totaling roughly 4,307. The decomposition is explanatory accounting, not a unique causal proof because changing the counterfactual order changes attribution.

Low teacher top-1 agreement can coexist with useful ANN recall because a query probes several buckets and Qdrant exactly scores their vectors. The MLP need not reproduce a single teacher label to retrieve useful neighbours. The teacher's labels are not exact-neighbour ground truth.

## 9. Selected improved MLP and final F.2 test

The original 200 Phase F measured queries were split into 100 validation and 100 F.2 final-test queries. The final set was withheld from F.2 model selection, but was already included in Phase F; it is an internal selection holdout, not an untouched external dataset.

Stage 1 used 2,048 samples, 30/60/120 epochs and seeds 42/43/44. Validation selected 60 epochs. Stage 2 used 4,096/8,192/16,384 samples, again with those seeds. The predefined selection criterion considered candidate fraction at validation recall targets 0.90 and 0.95 within ±0.02, selecting closest recall then lower work. Missing bands received a predeclared penalty. A powers-of-two `nprobe` sweep is coarse, so a missed band does not mean a configuration could not reach that target with finer probing. The selected configuration, not a global optimum, was 4,096 samples / 60 epochs.

| Final-test target near 0.90 | Recall | Mean candidates |
| --- | ---: | ---: |
| Centroid | 0.906 | 10,231.5 |
| MLP seed 42 | 0.901 | 9,565.7 |
| MLP seed 43 | 0.907 | 9,500.5 |
| MLP seed 44 | 0.910 | 9,372.2 |

The selected MLPs use 6.5--8.4% fewer candidates around this operating point. Paired 2,000-resample bootstrap intervals for candidate saving exclude zero for each seed; recall-difference intervals include zero. The correct claim is lower candidate work on this test sample, not a statistically supported recall improvement or non-inferiority result.

At the approximate 0.95 target, selected MLPs use 16,464--16,824 candidates at recall 0.952--0.959, versus centroid's 20,231 at 0.961: candidate work is lower but recall is not identical. At about 0.99, centroid reaches 0.993 with 38,856 candidates; MLP at `nprobe=32` reaches 0.994--0.996 with 50--51k candidates. There is no uniform dominance. F.2 timing is not directly paired with the inherited Phase F centroid timing; candidate work and recall are the defensible cross-run comparison.

## 10. Strengths, limitations and non-implemented work

The strongest qualities are genuine DBMS integration, preservation of database semantics, lifecycle coverage, the strict routing-versus-scoring separation, reproducible raw observations, exact Plain ground truth, centroid/affine controls, and retention of the negative baseline. The project diagnoses through controlled interventions rather than retrospectively hiding failure.

Limitations are equally important: one main dataset/domain and 100k scale; one primary 768-dimensional representation; one small MLP architecture; limited seed and build repetition; an internal rather than external F.2 holdout; sample-size/update-count confounding; coarse probing; unavailable HNSW visitation counts; cross-run latency limits; no reliable per-index RAM estimate; limited remote universal-filesystem qualification; no distributed crash campaign; no corpus-state cryptographic binding; incomplete telemetry; and no HNSW query-latency challenge. Thesis-critical next work is paired/denser measurement and external validation; extensive distributed hardening is valuable engineering beyond the immediate thesis scope.

Not implemented as established thesis functionality are DLI, CLI, online or continual neural updates, automatic cost-based Plain/HNSW/LMI planning, GPU training, a public per-query `nprobe` API, alternative neural architectures, a learned metric replacing Qdrant scoring, or broad distributed production qualification.

## 11. Current interpretation and next questions

The evidence supports six careful conclusions: LMI can be integrated as a first-class Qdrant physical index; the first router was poor mainly because training was inadequate; teacher fidelity and ANN recall differ; improved training can reduce candidate work relative to centroid routing at some medium-high recall points; that gain is modest and non-uniform; and HNSW remains the clear measured query-latency leader here.

High-value next investigations are a denser `nprobe` sweep near matched-recall targets, a genuinely new/larger query test set, paired same-harness centroid/MLP timing, matched optimizer-update controls for sample-size effects, repeated independent builds, larger corpora and another embedding domain, then workload-level query/build/update tradeoff analysis. A planner should only be considered after those measurements.

## 12. Supervisor briefing: what to say

The initial task was to determine whether learned routing could live inside a real vector DBMS without bypassing database correctness. It now does: Qdrant builds, persists, restores, rebuilds and serves the LMI while retaining vector ownership and exact scoring. The strongest engineering result is lifecycle integration, including fresh snapshot restore without retraining and optimizer-owned rebuilt immutable segments.

The strongest scientific result is that the negative baseline was understood rather than hidden. The initial 30-epoch router collapsed 24 buckets and needed 42% more candidate work than centroid routing. Controlled epoch increases restored almost all buckets, demonstrating undertraining. A validation-selected model then used 6.5--8.4% fewer candidates than centroid near 0.90 recall on an internal holdout, but this did not persist uniformly at high recall and did not establish a recall or latency win.

The realistic next experiment is not a larger model by default. It is a cleaner measurement: dense probing, an external query set, matched updates, and paired control timing. The guiding question is when learned routing is worthwhile given recall, candidate work, build/rebuild cost, index size and update frequency.

## Appendix: evidence, checkpoints and reproduction

Relevant checkpoints: `a628691f6` (dispatch instrumentation), `4f3bd3798` (structural LMI integration), `7029bcd0f` (database-owned training/persistence), `a807381fb` (snapshot/corruption), `a1987b504` (lifecycle), and `6e3cfdb0f` (Phase F baseline). Current worktree contains uncommitted F.2 evaluation/report files; no commit or push was performed.

Consulted evidence includes `docs/lmi-phase-e.md`, `docs/lmi-phase-e2.md`, `docs/lmi-phase-e2-final.md`, `docs/lmi-phase-f-audit.md`, `docs/lmi-phase-f.md`, `docs/lmi-phase-f2.md`, the LMI implementation under `lib/segment/src/index/lmi_index/`, Phase F/F.2 evaluation code, test drivers, saved raw output and commit history.

Reproduction requires the recorded WSL CPU/Torch environment: Rust 1.98, Python 3.11.16, Torch 2.5.1+cpu, tch 0.18.1 and clang 21.1.8. Phase F/F.2 reports contain exact commands, fixed seeds, hashes, environment settings and raw machine-readable artifacts. Run `cargo check -p collection --tests --features segment/lmi-training --locked`, `cargo check -p edge --locked`, the focused LMI integration suite, and the two isolated lifecycle drivers before treating changes as comparable to the recorded evidence.
