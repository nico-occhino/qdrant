# Phase G — Static LMI failure analysis and Qdrant architecture

Status: PARTIAL PHASE G — G1, G3 and G4 decisions are complete; G2 has a full analytic coverage curve and a bounded native pilot, but no accepted full six-point returned-recall/latency frontier. G5 is consequently conditional. This chapter distinguishes measured results from implementation facts and hypotheses. The accepted SISAP 10.12M state under `/home/nicoo/work/lmi-sisap2023-storage/10m-f16` is read only throughout this phase.

## 1. Motivation and baseline

**MEASURED:** N=10,120,191, d=768, Cosine, Float16 storage; B=3162, H=512, sample=250000, 30 epochs, five KMeans iterations, nprobe=4, seed=42. The accepted 9,980-query run yielded mean Recall@10=0.8152505010, mean candidates=17,667.08 (0.174573%), HTTP p50/p95/p99=13.298/170.992/1888.683 ms. All persisted learned hashes matched before and after. Full provenance: `docs/lmi-sisap2023.md`.

## 2. Initial architecture map, before Phase G edits

```text
HTTP / collection API
  → Collection::core_search_batch / do_core_search_batch
  → shard search
  → Segment::search_batch / read view
  → VectorData.vector_index: Arc<AtomicRefCell<VectorIndexEnum>>
       ├─ PlainVectorIndex
       ├─ HNSWIndex
       └─ LmiIndex
  → Vec<Vec<ScoredPointOffset>> (segment-local offsets)
  → Segment::process_search_result (external point IDs, payload/vector)
  → shard/collection aggregation
```

**IMPLEMENTED:** `lib/segment/src/segment/read_view/search.rs::search_batch` obtains the configured vector index and calls `VectorIndexRead::search` with query batch, optional `Filter`, top-k, optional `SearchParams` and `VectorQueryContext`. The trait and `VectorIndexEnum` are in `lib/segment/src/index/vector_index_base.rs`. The segment has one `VectorData` per vector name; each currently contains exactly one `vector_index` enum and a separate shared `vector_storage`. This is a representation constraint, not proof that vectors would have to be duplicated to support multiple auxiliary indexes.

**IMPLEMENTED:** `lib/segment/src/segment_constructor/segment_constructor_base/vector_index.rs::{open_vector_index,build_vector_index}` matches `VectorDataConfig.index` (`Indexes` in `lib/segment/src/types.rs`) to Plain/HNSW/LMI. `lib/segment/src/segment_constructor/segment_builder.rs` builds replacement segment indexes before publication. `lib/segment/src/index/lmi_index/build.rs` persists state, router and offset postings; `open_trained` validates and opens without training. `lib/segment/src/index/lmi_index/postings.rs::CompactPostings` stores per-bucket segment-local `PointOffsetType`s, not vector copies.

**IMPLEMENTED:** `lib/segment/src/index/lmi_index/read.rs::search` delegates filtered requests, non-default `SearchParams`, or quantized storage to Plain. For supported dense nearest-neighbor requests, `LmiRoutingState::candidates_for_query` in `routing.rs` ranks learned buckets and gathers posting offsets. `score_candidates_for_query` passes offsets to `BatchFilteredSearcher` in `lib/segment/src/index/hnsw_index/point_scorer.rs`; Qdrant applies point/vector deletion and deferred visibility checks and scores the authoritative vectors. `SearchParams` in `lib/segment/src/types.rs` includes `hnsw_ef`, `exact`, `quantization`, `indexed_only`, `acorn`, and `idf`. Default params can still use trained LMI; non-default params use Plain.

**IMPLEMENTED:** `IdTrackerRead::internal_id_with_behavior` in `lib/segment/src/id_tracker/id_tracker_base/trait_def.rs` maps external `PointIdType` to segment-local `PointOffsetType`. Diagnostic gold-neighbor mapping must call this method. External SISAP IDs must never be treated as offsets.

## 3. Ownership table

| State | Owner / Rust type | Persistence and restart | Optimization |
|---|---|---|---|
| User-visible IDs and segment offsets | `Segment.id_tracker`, `IdTrackerEnum` | ID mapping files | New target segment can assign new offsets. |
| Authoritative vectors | `VectorData.vector_storage`, `VectorStorageEnum` | Vector storage files; Float16 matrix in accepted run | Copied/repacked into target segment. |
| Payload and filter indexes | `Segment.payload_index`, `StructPayloadIndex` | Payload storage/index files | Rebuilt for target segment. |
| One configured physical vector index | `VectorData.vector_index`, `VectorIndexEnum` | Plain/HNSW/LMI files as applicable | `SegmentBuilder` builds according to target `Indexes`. |
| LMI learned state | `LmiIndex` and `LmiRoutingState` | `lmi_state.json`, `lmi_router.bin`, `lmi_postings.bin` | New postings must refer to new target offsets. |

## 4. Candidate generation and scoring

```text
External query → distance preprocessing → MLP bucket ranking
  → top-P bucket posting union → segment-local offsets
  → IdTracker/deletion/deferred checks → BatchFilteredSearcher / RawScorer
  → top-k offsets → external IDs
```

The vector index proposes candidate offsets; Qdrant owns vector bytes and scoring semantics. RawScorer alone does not automatically enforce all database visibility rules. The LMI seam therefore uses Qdrant's filtered searcher and point-level deletion mask.

## 5. Current scientific questions

- **HYPOTHESIZED:** The low-recall tail may reflect partition dispersion, router errors, Float16/tie semantics, or a scoring/output issue. G1 measures these separately.
- **MEASURED:** G2 has complete 9,980-query oracle/candidate-coverage and posting-count curves. Actual native return sets exist for a first-100-query diagnostic pilot; the full returned-recall frontier remains open.
- **MEASURED:** G3 traces HNSW and LMI dispatch, verifies live REST SearchParams behavior, and evaluates five synthetic indexed payload selectivities.
- **SOURCE AUDIT:** G4 finds shared storage feasible but multi-index serving requires a cross-cutting representation/lifecycle refactor.

This document is updated as each result is obtained. A negative or incomplete experiment will be labelled explicitly.
## 6. Exact REST-to-scorer path

**IMPLEMENTED / source audited:** `src/actix/api/search_api.rs::search_points` accepts `POST /collections/{collection}/points/search` and calls `src/common/query.rs::do_core_search_points`. The table of contents selects a collection; `lib/collection/src/collection/search.rs::core_search_batch` calls `do_core_search_batch`, selects shard replica sets and merges shard results. `lib/collection/src/shards/local_shard/search.rs::do_search_impl` invokes `SegmentsSearcher::search`. `lib/collection/src/collection_manager/segments_searcher.rs::search_in_segment` groups compatible requests and `execute_batch_search` calls `ReadSegmentEntry::search_batch`. `lib/segment/src/segment/read_view/search.rs::search_batch` selects `VectorData.vector_index()` and calls its `VectorIndexRead::search`. The selected enum then runs Plain, HNSW or LMI. Segment search converts internal scored offsets back to external IDs and optionally attaches payload/vector; collection search merges shard results. This is the actual source path for the tested legacy search endpoint, not a generic sketch of every Qdrant query API.

```text
search_points → do_core_search_points → TableOfContent → Collection::do_core_search_batch
→ LocalShard::do_search_impl → SegmentsSearcher::search_in_segment
→ execute_batch_search → Segment::search_batch → VectorIndexEnum::search
→ LmiIndex::search / HNSWIndex::search / PlainVectorIndex::search
→ ScoredPointOffset → Segment::process_search_result → ScoredPoint
```

**IMPLEMENTED:** `PointOffsetType` is an integer meaningful only inside one segment. `IdTrackerRead::internal_id_with_behavior` and `external_id` map it to/from user-visible IDs. An optimizer replacement can renumber offsets; postings from the source segment therefore cannot be copied unchanged. The Phase G exporter called the real `IdTrackerRead` mapping for each official gold and returned ID, then verified every saved posting offset belongs to exactly one bucket.

## 7. Vector-index abstraction and ownership

`VectorIndexRead::search` takes borrowed queries, optional filter, top-k, optional `SearchParams` and a `VectorQueryContext`; it returns per-query `ScoredPointOffset` lists. It does not own an external ID namespace. `VectorIndex` adds file enumeration and update methods. `VectorIndexEnum` is the runtime sum type for one configured physical index per vector name. `VectorDataConfig.index` in `types.rs` is the persisted choice. `SegmentBuilder` constructs a new target's vector storage, ID tracker, payload index and one physical vector index in a temporary directory, saves segment state, renames the completed directory, and reopens it. `snapshot.rs` enumerates files through that one index object.

Plain owns no auxiliary graph/model; it scans eligible offsets and uses `BatchFilteredSearcher`. HNSW owns graph metadata and has an internal dispatch between graph and Plain paths. LMI owns a validated MLP router and compact offset postings, but not vector bytes. The three use Qdrant's authoritative `VectorStorage`, `IdTracker` and payload state. Open/restart validates persisted LMI state; training belongs only to build. Trained LMI rejects in-place vector updates; optimization constructs a fresh segment and fresh postings.

## 8. SearchParams and filtering semantics

**IMPLEMENTED:** `SearchParams` contains `hnsw_ef`, `exact`, `quantization`, `indexed_only`, `acorn`, and `idf`. An explicit default object equals `SearchParams::default()`. In `LmiIndex::search`, an unfiltered, dense nearest-neighbor query with absent or default params and no quantized storage uses StaticLearned. A filter or any non-default params selects the owned Plain fallback. Other query shapes lacking dense nearest semantics also fall back to Plain. `exact=true` therefore gets exact Plain semantics. A supplied `hnsw_ef` does not cause HNSW to run on an LMI-configured field; it causes current LMI dispatch to Plain.

**MEASURED REST:** `work/phase_g/g3_planner/rest-dispatch.json` records six live requests on the accepted storage. No params and `{}` each produced one StaticLearned marker. `exact=true`, `hnsw_ef=64`, quantization ignore and a `has_id` filter reached `LmiIndex::search` but produced no StaticLearned marker; all succeeded. The filter returned one ID. The state/router/postings SHA-256 values were equal before and after. The first controller startup attempt encountered an immediate `/proc/<pid>/environ` race and terminated that process before queries; retry opened the same state and passed environment checks. No learned state was changed.

**IMPLEMENTED:** HNSW's real filter planner is in `lib/segment/src/index/hnsw_index/hnsw/read_view/dispatch.rs`. Exact or disabled graph uses Plain. For a filter, `PayloadIndexRead::estimate_cardinality` supplies a cardinality range, `adjust_to_available_vectors` adjusts it, and the full-scan threshold separates certainly-small Plain from certainly-large graph searches. When the estimate straddles the threshold, HNSW samples IDs using a filter context. Its graph path applies filter context during traversal. Plain's filtered path obtains matching point IDs from the payload index and scores those. LMI currently does not run that estimator: every filtered request goes to Plain first.

```text
HNSW + filter
  exact or disabled → Plain
  cardinality.max < threshold → Plain
  cardinality.min > threshold → graph + filter context
  otherwise → sampled decision
LMI + any filter → Plain (current policy)
```

**MEASURED small fixture:** `work/phase_g/g3_planner/filter-small.json` contains five controlled indexed integer payload filters over 1,000 points: 1000, 500, 100, 10 and 1 matching points. Estimated min/expected/max equalled actual cardinality in each case. Filtered LMI output equalled exact Plain output, so Recall@10 against that truth was 1.0 (the one-point case returned one point). Single-query elapsed times are recorded as descriptive observations, not a latency benchmark. Candidate work in this Plain path is the matching point set size, not LMI postings. The fixture is synthetic and does not establish behavior on a large real metadata workload.

## 9. G1 method: partition, router and output

**MEASURED:** `lib/segment/tests/lmi_phase_g_export.rs` loads the persisted segment read only, calls Qdrant's `IdTrackerRead::internal_id_with_behavior(VisibleOnly)` for all 99,800 official gold IDs, and maps each internal offset to its one learned bucket via the persisted `CompactPostings`. It separately maps accepted returned IDs for native parity checks. All 10,120,191 live stored offsets appeared exactly once in postings. The exporter never assumes external ID equals offset. Its 9,980 rows are `work/phase_g/g1_failure_decomposition/gold-buckets.jsonl`.

`tests/lmi_phase_g_analyze.py` loads the validated native router and official queries, reproduces top-P bucket order at P=1,2,4,8,16,32, and checks P=4 candidate counts against the accepted HTTP JSONL. **All 9,980 counts matched exactly.** For each query and P it computes: (a) oracle coverage by the P gold-neighbor buckets with highest multiplicity, (b) coverage by the actual router's first P buckets, and (c) summed posting count. The oracle knows exact neighbors and is unattainable at serving time.

At P=4, define:

```text
partition loss = 1 − oracle gold coverage
router loss = oracle gold coverage − routed candidate gold coverage
scoring/output loss = routed candidate gold coverage − official returned Recall@10
```

These terms telescope to `1 − official Recall@10`. A nonzero third term can reflect scoring precision, ties or output semantics as well as a defect; it is not automatically a scorer bug. Results use official top-10 IDs without modifying the accepted benchmark.

| P=4 mean across 9,980 | Value |
|---|---:|
| Oracle top-10 coverage | 0.9493988 |
| Actual router candidate gold coverage | 0.8188577 |
| Accepted returned Recall@10 | 0.8152505 |
| Partition loss | 0.0506012 |
| Router loss | 0.1305411 |
| Scoring/output loss | 0.0036072 |

**MEASURED:** 750 queries had official recall ≤0.3. In that subgroup, mean partition/router/scoring-output losses were 0.2227/0.5748/0.0076. The 86 zero-recall queries had corresponding means 0.1977/0.7442/0.0581. Thus the worst tail is mostly router-limited but contains partition and semantic subcases. The P=4 oracle reaches full coverage for the median query; its p5 is 0.7. The mean number of distinct gold buckets is 3.248 (median 3, p95 7).

**MEASURED associations, not causes:** Pearson/Spearman correlations with official recall were `d10`: −0.507/−0.528; distinct gold buckets: −0.779/−0.819; partition loss: −0.711/−0.702; router loss: −0.915/−0.939; candidate count: +0.170/+0.171. Loss variables contain recall algebraically, so their correlations should not be read as independent causal evidence. Distance and bucket dispersion offer more independent hardness features. Per-query `d1`, `d10`, spread, mean top-10 distance, largest bucket multiplicity, all three losses and candidate fraction are retained in `per-query.jsonl`.

## 10. Tie and Float16 pathology

**MEASURED:** 298 measured queries have at least one nearly-zero official top-10 distance; 89 have all ten near zero. The official top-10 tie-aware diagnostic using the published 1,000 gold neighbors raises mean recall only from 0.8152505 to 0.8159820. The 1,000-neighbor gold is a finite list, so it cannot enumerate arbitrarily large near-duplicate groups. Exactly 295 queries have positive candidate-coverage minus returned-ID recall. Ten of those have all ten official distances near zero. `tests/lmi_phase_g_scoring_loss.py` recomputes returned-point cosine distances from original Float16 source rows in Float64 for these 295 queries and preserves every value in `scoring-loss-source-check.jsonl`. Forty of 295 return ten points within the official tenth-neighbor radius plus `1e-7`; the others can include genuinely farther points. For query 1269, all ten official gold distances are numerically zero while returned candidates have tiny positive source distances and Qdrant stored scores slightly above 1.0. This is strong evidence that Float16 storage/scoring precision and near-duplicate rank instability contribute to the scoring/output term, but it does not prove that every one of the 295 differences has the same cause. No official benchmark metric was replaced.

## 11. G1 decision

**MEASURED decision:** `ROUTER-LIMITED` overall at four probes; `MIXED` among difficult queries; a small `SCORING/SEMANTIC ISSUE` subgroup. This is supported by a 0.1305 mean oracle-router gap versus 0.0506 partition ceiling loss, the same dominance in low-recall queries, and 9,980 exact candidate-count parity checks. Decision record: `work/phase_g/g1_failure_decomposition/decision.json`. The correct next router comparison should keep this fixed partition and evaluate whether a better routing objective closes the gap; changing corpus assignments is a separate experiment.

## 12. G2 analytical frontier and its validity boundary

**MEASURED candidate work and candidate-gold coverage** across all 9,980 queries on the *same* persisted MLP postings:

| P | Oracle coverage | Router candidate-gold coverage | Mean candidate count | Mean candidate fraction |
|---:|---:|---:|---:|---:|
| 1 | 0.6641884 | 0.5592886 | 4,776.36 | 0.04720% |
| 2 | 0.8343487 | 0.7100902 | 9,231.76 | 0.09122% |
| 4 | 0.9493988 | 0.8188577 | 17,667.08 | 0.17457% |
| 8 | 0.9988277 | 0.8924349 | 33,975.68 | 0.33572% |
| 16 | 1.0000000 | 0.9395190 | 65,352.01 | 0.64576% |
| 32 | 1.0000000 | 0.9684770 | 125,698.18 | 1.24205% |

**Important:** candidate-gold coverage is an upper bound on returned Recall@10. Only P=4 has an accepted full HTTP return set. The bounded native pilot cannot replace a full returned-recall sweep; this table is not a returned-recall Pareto frontier. `work/phase_g/g2_frontier/analytic-frontier.json` stores all quantiles. The oracle chooses buckets by gold multiplicity without accounting for posting sizes, so it is a coverage upper bound, not a cost-optimal oracle.

A pilot native experiment scores candidates using Qdrant's `BatchFilteredSearcher` over the same shared Float16 storage, leaving persisted state untouched. It showed equal P=4 result *sets* for all 100 pilot queries; eight had only ordering differences among very close scores. Thus ID-set/Recall parity, not exact ordering, is the proper gate. The pilot's initial cold-cache timing is not a stable operating-point benchmark. The full six-point native attempt was stopped after 45 complete queries (and three more valid probe rows, followed by a torn final line) because repeated random Float16 scoring progressed too slowly and competed for file-backed memory. Its exact partial bytes and metadata are preserved; it is not an accepted benchmark. The 100-query pilot is diagnostic only and its first-100 query selection and cold-cache timings cannot establish the full frontier.

**NOT PERFORMED:** TT, TM, MT and MM control symmetry at 10.12M. Current postings are MLP corpus assignments (`M`). A centroid query router over them would be `TM`, not a centroid index (`TT`). The existing 100K Phase F controls used proper centroid-generated postings; reproducing `TT` at 10.12M would require an additional corpus classification build and is not needed to answer the fixed-partition oracle question.


**MEASURED bounded native pilot (first 100 measured rows only):** returned Recall@10 at P=1/2/4/8/16/32 was 0.584/0.774/0.856/0.924/0.965/0.979; mean candidates 4,617/9,091/17,697/33,959/66,322/127,278. These rows are not a random/held-out sample and the P=4 pilot recall 0.856 differs from the full accepted 0.8153. P=4 ID sets and candidate counts matched accepted HTTP data for every pilot query. The pilot's native scoring medians were 58.5/50.8/84.3/170.0/425.6/717.1 ms, but cache state changed throughout this single process and the values are **not** comparable to accepted HTTP latency or a warm steady-state curve. `work/phase_g/g2_frontier/native-pilot-100.summary.json` contains full component quantiles. `native-full.partial.metadata.json` explicitly rejects the interrupted 45-query attempt.

**G2 decision:** `INCOMPLETE / no accepted full Pareto frontier`. The full analytic curve shows substantial router headroom and increasing candidate cost. The bounded native pilot confirms the expected direction of returned recall, but cannot choose an optimal operating point for all 9,980 queries. Do not call P=4 optimal or claim that P=8 reaches 0.8924 returned recall; 0.8924 is candidate-gold coverage. `work/phase_g/g2_frontier/decision.json` records this boundary.

## 13. G3 planner decision

**MEASURED / source-audited:** `WORKABLE WITH LIMITATIONS`. LMI already participates in normal segment index dispatch and preserves exact/filter semantics through Plain fallback. HNSW's threshold/cardinality planner is not a generic selector across multiple physical indexes; it lives inside `HNSWIndexReadView`. Adding a new three-way selector today would be misleading because the same vector field cannot concurrently configure HNSW and LMI. We retain the simple current rule: supported dense nearest + default params + no filter → LMI; otherwise Plain. More sophisticated selective-filter Plain, LMI intersected postings, progressive probing and fewer-than-k fallback require a measured filter workload and explicit cost rule. The five-selectivity fixture establishes safety of the current fallback, not a crossover threshold. Decision: `work/phase_g/g3_planner/decision.json`.

## 14. G4 multi-index feasibility and cost

**SOURCE AUDIT:** `VectorDataConfig.index: Indexes` holds one variant; `Segment::VectorData.vector_index` is one `Arc<AtomicRefCell<VectorIndexEnum>>`; `open_vector_index`/`build_vector_index` choose one; `SegmentBuilder` builds one per vector name; `Segment::search_batch` dispatches to one; read-only open, snapshot file enumeration, telemetry and indexed-vector counts are based on one. Collection config currently rejects per-vector HNSW plus LMI. See `work/phase_g/g4_multi_index/decision.json` for each assumption and classification.

**INFERRED feasible ownership:** HNSW and LMI constructors both receive shared `Arc` handles for vector storage, ID tracker and payload index. Multiple auxiliary structures could refer to the same authoritative vectors without duplicating 15,544,613,376 bytes of Float16 payload. The measured 10.12M LMI auxiliary files total 49,569,750 bytes. A Phase F 99,780-indexed-vector HNSW build with `m=16` used 3,857,946 auxiliary bytes. Linear scaling of that historical density suggests roughly 390 MB for 10.12M HNSW and roughly 440 MB combined auxiliary overhead, **an estimate, not a measured 10.12M HNSW size**. Graph density, levels, configuration and allocator/format effects could change it. Do not add vector payload twice.

```text
Shared VectorStorage + IdTracker + PayloadIndex
       ├── Plain full scan (a scoring strategy, little auxiliary state)
       ├── HNSW graph files
       └── LMI router + postings
                   ↓
           explicit per-query selector
```

**INFERRED lifecycle requirement:** `SegmentBuilder` already constructs one replacement in a temporary directory, writes state, then renames and reloads. A multi-index version should build and validate *both* auxiliary indexes against the target segment's new offsets before publication; a missing/corrupt member should reject open or follow an explicit degradation policy. Snapshot file listing must include both. Read-only serving and telemetry must expose both, and `indexed_vector_count` must have a defined non-additive meaning. Deletion/update behavior must maintain visibility while replacement segments build. The current code does not prove atomic multi-index publication because it builds only one.

**G4 decision:** `MULTI-INDEX FEASIBLE BUT REQUIRES MAJOR REFACTOR`. A toy test proving two objects can share an `Arc` would not prove optimizer, snapshot or planner correctness. No prototype was added. The blocker is the one-index-per-field representation and lifecycle, not a need to duplicate vectors. This boundary matters before DLI/CLI: each new index must first satisfy the existing vector-index and segment lifecycle contracts; co-residency is a separate architectural project.

## 15. G5 synthesis and Phase H requirements

**MEASURED:** The fixed MLP corpus partition leaves a substantial four-probe router improvement opportunity (oracle 0.9494 versus actual candidate coverage 0.8189). High probe counts approach complete gold coverage with increasing work; actual native Recall/latency and Pareto choice require a future full G2 result. Existing filtered LMI queries are correct through Plain; no performance crossover is known. HNSW+LMI cannot currently coexist as configured indexes over one vector field.

**FUTURE WORK:** A DLI design should specify how its routing state and postings are built for a segment, validated and reopened without retraining, how fresh/deleted points remain visible, and when an optimizer rebuild publishes a new generation. Continual adaptation cannot silently mutate postings referencing old offsets. A CLI design would add its own cluster/prototype/pivot or other routing state, but should reuse the same Qdrant vector storage, ID tracker, scorer, cancellation and snapshot contracts. Neither requires a three-way planner as a precondition if first evaluated as the sole configured index on a test vector field. Multi-index co-residency becomes necessary only when the research question explicitly requires choosing among simultaneous HNSW/LMI/DLI/CLI structures.

**Phase H recommendation, conditional on G2:** keep the accepted MLP postings fixed and test a retrieval-relevant query-routing objective against the current MLP and unattainable oracle. Use a held-out validation/test separation. If G2 shows probe widening already reaches the desired quality at tolerable candidate work, quantify whether a new router reduces candidates at matched recall. Introduce dynamic index maintenance only after the static retrieval tradeoff and lifecycle contract are explicit. The first Phase H experiment should be a controlled query-router comparison on the fixed 10.12M partition, without rebuilding the corpus and without adding DLI/CLI to Qdrant until the external result justifies integration.

## 16. Negative results and open questions

- **NEGATIVE RESULT:** A three-way Plain/HNSW/LMI planner is not a local rule in the current segment representation. HNSW and LMI are mutually exclusive per field.
- **NEGATIVE RESULT:** The candidate-coverage curve cannot be labelled actual Recall or a latency Pareto frontier before native scores are complete.
- **NEGATIVE RESULT:** Centroid query ranking over MLP postings is a mixed `TM` experiment, not a true centroid control.
- **OPEN:** How much of the 295-row scoring/output discrepancy comes from Float16 normalization/ranking versus official tie handling? Float64 source recalculations narrow the issue but do not exhaustively prove causality.
- **OPEN:** On real payload workloads, at what selectivity would filtered LMI beat Plain after accounting for posting gather and filtering? No measured crossover exists yet.
- **OPEN:** What is the exact 10.12M HNSW auxiliary size and build cost with a matching configuration? No HNSW build was performed in Phase G.
- **OPEN:** How should `indexed_only`, quantization and advanced query forms behave under a future multi-index selector? Their current fallback semantics must be preserved or explicitly redesigned.
- **OPEN:** Does a multi-index target build and publish all configured structures atomically? Current code only builds one.

## 17. Tests, artifacts and reproducibility

Relevant new files: `lib/segment/tests/lmi_phase_g_export.rs` (read-only real ID/posting export), `lib/segment/tests/lmi_phase_g_filter.rs` (controlled payload filter fixture), `lib/segment/tests/lmi_phase_g_native.rs` (explicit native probe sweep), `tests/lmi_phase_g_analyze.py`, `tests/lmi_phase_g_scoring_loss.py`, `tests/lmi_phase_g_rest_probe.py`. Baseline code and storage remain unchanged. The Rust exporter and filter experiment passed. A separate full-segment exporter rerun passed (1 test, 10.94 s); its gold mapping and router exports are byte-identical to the original SHA-256 values. Focused Phase A–E/Phase G tests passed: candidate scoring 2, dummy 1, Phase C 10, Phase D 12, Phase E 15 (one additional expensive test ignored), and Phase G filter 1; total 41 passed, 0 failed, 1 ignored. The native pilot completed 100 queries × six probe counts. Its first strict ordered-ID parity assertion failed because eight of 100 result sets were ordered differently; every P=4 ID set and candidate count matched. An independent check accepted it as a diagnostic pilot. The full 9,980-query native attempt was stopped after 45 complete queries because repeated random scoring was too slow; its partial JSONL has a torn final line and is explicitly non-benchmark evidence. Exact verification commands and final Git state are recorded below. A later ten-query smoke rerun was stopped after several minutes of storage I/O; it adds no accepted results. A one-query native smoke completed with zero P=4 ID-set parity failures (1 passed, 26.76 s).

The authoritative Phase G evidence directories are:

```text
work/phase_g/g1_failure_decomposition/
work/phase_g/g2_frontier/
work/phase_g/g3_planner/
work/phase_g/g4_multi_index/
work/phase_g/g5_synthesis/
```

To reproduce G1 without rebuilding, point `LMI_G_SEGMENT` at the UUID segment above, `LMI_G_QUERIES_JSONL` at the accepted `queries.full.jsonl`, and `LMI_G_OUTPUT` at a fresh output directory; run `cargo test -p segment --test lmi_phase_g_export -- --ignored --nocapture`. Then run `tests/lmi_phase_g_analyze.py` and `tests/lmi_phase_g_scoring_loss.py` under the preserved `lmi-starterpack` Python environment. These scripts require the official query/gold/source HDF5 files at the paths in the source and write only under `work/phase_g`. The native experiment reads the exported gold mapping and a Float32 binary copy of official query rows 20–9999; it scores through Qdrant's `BatchFilteredSearcher` and does not use HTTP. It has an explicit `LMI_G_LIMIT` for a bounded pilot.

## 18. Final verification and reproduction notes

**MEASURED:** The focused test invocation with `--locked` and the repository's existing `lmi-training` feature passed 41 tests: 2 candidate-scoring, 1 dummy, 10 Phase C, 12 Phase D, 15 Phase E, and 1 Phase G filter; 1 intentionally expensive Phase E test remained ignored. The explicit full-segment exporter passed again (1 test, 10.94 s). Its original and verification `gold-buckets.jsonl` files both hash to `81858d1b7fd68bafda3ae1066998e263171387e2ffc20a7d4eb42d273fd1a432`; both router JSON exports hash to `8d1116b6ebc5164373582ff2d3c22a059f213908ee73861cea21fa1629ce6a64`. A one-query, six-probe native smoke test passed (1 test, 26.76 s, zero P=4 ID-set parity failures). The 100-query pilot was independently checked for 100/100 P=4 ID-set and candidate-count parity; its original ordered-result assertion did fail on eight reorderings, then the assertion was corrected to compare sets. No full rerun of that pilot is claimed.

The learned state SHA-256 remained `61cb61886af0ccbda56f013ac4fe5496ad688b7e5831deb67436f53036e4e459`; router `35df7ec2613687950ea422b708c7fe2ad18ebc61776724e5a68dc7773c0ac340`; postings `c82b1cd646d4b349fdc4d5b6a3b5b34154ddcc710cda4af058cc1cc035742bf1`. The accepted query and baseline evidence under `work/phase_s3/` were not rewritten.

The build environment for tests with `lmi-training` needs the existing `/home/nicoo/miniconda3/envs/lmi-starterpack` runtime on `PATH`, `LIBTORCH_USE_PYTORCH=1`, `LIBTORCH_BYPASS_VERSION_CHECK=1`, and `LD_LIBRARY_PATH` including that environment's `lib/python3.12/site-packages/torch/lib`. Without the library path, the test executable cannot load `libtorch_cpu.so`; this is an environment setup failure, not a test assertion failure. `cargo fmt --all` completed with expected stable-rustfmt warnings for nightly-only import settings; it touched two otherwise-clean unrelated files, which were restored to HEAD. `git diff --check` passed. No Phase G production source files were edited.

The full 9,980-query native six-probe sweep was explicitly stopped after 45 complete query groups because this machine's scattered Float16 scoring was progressing at an unsuitable rate and stressed file-backed memory. Its partial file is retained for diagnosis, not included in aggregate claims. Therefore Phase G as specified is **not complete**: a full native returned-recall/candidate/latency frontier, Pareto operating points, and cache-controlled warm-state comparison remain open. The safest next increment is to reuse nested-probe scoring work or add a diagnostic-only runtime probe override, validate P=4 against the accepted API results, then run a bounded full sweep without rebuilding the index.

**Git hygiene:** no commit, staging, or push was made in Phase G. The tracked diff at handoff is the four pre-existing files (`lib/collection/src/config.rs`, `lib/collection/src/operations/types.rs`, `lib/segment/src/index/lmi_index/build.rs`, `lib/segment/tests/lmi_phase_e.rs`): 124 insertions, 7 deletions. Git does not include untracked Phase G files in `git diff --stat`; enumerate them with `git status --short`. The other pre-existing untracked Phase S3 and SISAP files remain untouched.

## 19. Reproduction commands (no index rebuild)

Run in the existing WSL repository. The HDF5 paths hard-coded in the two Python diagnostics must exist under `/mnt/c/datasets/sisap2023/`; they are the accepted SISAP source, query and gold files. Use a new `LMI_G_OUTPUT` directory if retaining an earlier export byte-for-byte.

```bash
cd /home/nicoo/work/qdrant
export PATH=/home/nicoo/miniconda3/envs/lmi-starterpack/bin:$PATH
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
export LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
export LMI_G_SEGMENT=/home/nicoo/work/lmi-sisap2023-storage/10m-f16/storage/collections/sisap2023_10m_f16_lmi/0/segments/8f67bcc1-6bca-49c8-bef5-4ae8f4565ab1
export LMI_G_QUERIES_JSONL=/home/nicoo/work/qdrant/work/phase_s3/sisap2023/10m-f16/queries.full.jsonl
export LMI_G_OUTPUT=/home/nicoo/work/qdrant/work/phase_g/g1_failure_decomposition/verification
cargo test -p segment --features lmi-training --test lmi_phase_g_export --locked -- --ignored --nocapture
python tests/lmi_phase_g_analyze.py
python tests/lmi_phase_g_scoring_loss.py
cargo test -p segment --features lmi-training --test lmi_candidate_scoring --test lmi_dummy --test lmi_phase_c --test lmi_phase_d --test lmi_phase_e --test lmi_phase_g_filter --locked
cargo fmt --all
git diff --check
```

`lmi_phase_g_analyze.py` reads the main G1 export path, so a fresh verification export must either be compared with that original or be copied into a new isolated analysis workspace with adjusted paths. It must not replace the accepted Phase S3 files. To run only the bounded native smoke, set `LMI_G_GOLD_EXPORT` to the main G1 JSONL, `LMI_G_QUERY_F32` to `work/phase_g/g2_frontier/queries.rows20-9999.f32` using an absolute path, `LMI_G_SWEEP_OUTPUT` to a fresh absolute output path, and `LMI_G_LIMIT=1`; then run `cargo test --release -p segment --features lmi-training --test lmi_phase_g_native --locked -- --ignored --nocapture`. The program's default 9,980-query sweep is not recommended until the scoring I/O bottleneck is addressed. Live REST dispatch reproduction additionally requires an isolated server at `127.0.0.1:17038` with the accepted storage mounted; `tests/lmi_phase_g_rest_probe.py` documents the six requests and writes the dispatch record. Never start a second writer against the same Qdrant storage.

## What Nicolò must be able to explain without looking at the code

1. **What is a Segment?** A self-contained group of points with vectors, payload, IDs, indexes and version state; `lib/segment/src/segment/mod.rs::Segment`.
2. **What is VectorStorage?** The authoritative vector bytes and deletion state; `lib/segment/src/vector_storage/`, held in `Segment::VectorData`.
3. **What is PointOffsetType?** An integer position local to a segment, not an external ID; `common::types`, used throughout postings and scorer APIs.
4. **How are external IDs mapped?** `Segment.id_tracker` implements `IdTrackerRead::internal_id_with_behavior` and `external_id`; `lib/segment/src/id_tracker/id_tracker_base/trait_def.rs`.
5. **What is VectorIndexEnum?** The one configured per-field runtime index variant, `Plain`, `Hnsw`, `Lmi`, etc.; `lib/segment/src/index/vector_index_base.rs`.
6. **What is its search contract?** `VectorIndexRead::search` takes queries, filter, top, params and context; returns local scored offsets; same file.
7. **How does HNSW build/open?** `segment_constructor_base/vector_index.rs` selects `HNSWIndex::{build,open}` using shared tracker/storage/payload handles; HNSW graph files live under its index path.
8. **How does LMI build/open?** The same constructor selects `LmiIndex::{build_trained,open_trained}`; `lmi_index/build.rs` saves and validates model/postings; open logs no training.
9. **What does SegmentBuilder do?** Copies/reassigns vectors and IDs into a temporary replacement, builds its configured indexes, saves state, renames, and reloads; `segment_constructor/segment_builder.rs`.
10. **Where is replacement published?** The builder renames a complete target segment directory, then returns its loaded segment; optimizer/holder code subsequently swaps references to it. Source: `segment_builder.rs::build` and collection optimizer modules.
11. **Where does RawScorer enter?** Through Qdrant's candidate scoring path; `hnsw_index/point_scorer.rs::BatchFilteredSearcher` and `vector_storage` scorer builders.
12. **Why doesn't LMI score vectors itself?** It owns bucket predictions and offsets; Qdrant owns datatype, metric, deletion, quantization and top-k semantics; `lmi_index/read.rs`.
13. **How do filters reach an index?** `Filter` flows from REST through collection/shard/segment to `VectorIndexRead::search`; HNSW estimates cardinality, current LMI falls back to Plain.
14. **What is cardinality estimation?** A bounded estimate of matches from payload indexes, used by HNSW to choose graph or scan; `hnsw/read_view/dispatch.rs` and `PayloadIndexRead`.
15. **Why can Plain win for selective filters?** Scoring a small matching set can be cheaper than graph traversal; HNSW dispatch compares estimated scan work to a threshold.
16. **What is in SearchParams?** HNSW beam, exact mode, quantization, indexed-only, ACORN and IDF; `lib/segment/src/types.rs`.
17. **What is query-time dispatch?** Choosing the path when a request arrives; today it occurs within the one configured index (HNSW internal graph/Plain, LMI learned/Plain).
18. **Why is this different from multiple configured indexes?** A planner can only choose an HNSW graph and LMI postings together if both are represented, built and opened for the same field; the current enum stores one.
19. **Why needn't multiple indexes duplicate vectors?** Both constructors accept shared `Arc` handles to one `VectorStorage`; auxiliary files are separate.
20. **What belongs to LMI?** Router model, bucket postings, training/build validation and LMI-specific routing; `lmi_index/{build,routing,postings,read}.rs`.
21. **What belongs to Qdrant?** External IDs, authoritative vectors, payload filters, scoring, segment lifecycle, snapshots, request handling.
22. **Why are offsets segment-local?** Every segment has its own ID tracker and vector storage; a rebuilt target can reorder or compact points.
23. **Why can't old postings simply be copied?** They contain source offsets, which may refer to different points in the replacement segment.
24. **Why does LMI reopen without training?** Persisted router/postings are validated and loaded by `open_trained`; training is only in `build_trained`.
25. **What are the stages?** Partition construction assigns corpus points to buckets; query routing ranks buckets; candidate generation gathers offsets; Qdrant scoring ranks actual vectors; database lifecycle builds and publishes segments; planner selection chooses a physical path; continual adaptation changes state over time and must preserve all these invariants.
