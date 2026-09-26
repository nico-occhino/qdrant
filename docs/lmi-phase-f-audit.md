# Phase F measurement audit and protocol

Base a1987b504, thesis/lmi-integration, initially clean. Evaluation only; no commit/push.

| Seam | Source | Measurement / limitation |
| --- | --- | --- |
| Build, sample, model, postings | lmi_index/build.rs::build_trained; SegmentBuilder::build | Wall construction and test-only stage capture; no query-time clocks added to production. |
| Clustering and MLP | lmi_index/training.rs::cluster/train | Cluster currently discards final centers. Extract a common center-returning clustering helper with a labels-only production wrapper, retaining the existing training behavior. Evaluation consumes exact teacher centers on persisted sample offsets. |
| Routing and union | lmi_index/routing.rs::MlpRouter::top_buckets / LmiRoutingState::candidates_for_query | Evaluation-only separate clocks for routing and deduplicated sorted candidate union. Native MLP state and independent centroid postings. |
| Valid candidates / scoring | lmi_index/read.rs::score_candidates_for_query; BatchFilteredSearcher | Use Qdrant scorer and authoritative masks. Dataset is immutable with no deletions, so eligible count equals corpus count; assert validity before scoring. |
| Full search | VectorIndexRead::search | Actual in-process index call, not REST end-to-end request latency. Suppress legacy stderr markers only in unit-test build to avoid timed-loop logging; normal server unchanged. |
| HNSW | hnsw/read_view/dispatch.rs | Configure full_scan_threshold=0, assert unfiltered_hnsw telemetry increases. hnsw_ef sweep. Exact visited/scored candidate count unavailable without deeper instrumentation: report null, never substitute ef for work. |
| Plain | plain_vector_index/read_view/search.rs | Exact Qdrant ground truth and separately timed exact baseline over same storage representation/metric. |
| Size / reopen | index.files(); load_segment | File bytes, coldness-uncontrolled reopen wall time. Memory: whole-process RSS only; no unsupported per-index attribution. |

Protocol: LAION float16 converted to float32 only by Python dataset preparation. Fixed seeded permutation with disjoint held-out queries; no corpus self matches. Cosine metric, k=10; 64 buckets, 2048 training samples, hidden=64, 30 epochs, batch=256, 20 Lloyd iterations, seed=42. Probe sweep 1,2,4,8,16,32,64; HNSW ef sweep 16,32,64,128,256,512. Separate warmup and repeated measured held-out queries. Release build required for reported timings; pilot correctness runs may use fewer rows. Centroid and affine controls share teacher centers and centroid-assigned corpus postings, never MLP postings. Affine equivalence checked including deterministic ties. Do not infer a winner from one operating point or classifier agreement.
