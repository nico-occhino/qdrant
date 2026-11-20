//! Static LMI index: pure-Rust serving with optional build-only Torch training.
//! Segment optimizer publication is outside this module and remains a separate stage.

#[cfg(feature = "lmi-training")]
mod build;
mod build_plan;
mod config;
pub mod filtered;
mod postings;
pub mod read_only;
mod routing;
#[cfg(feature = "lmi-training")]
mod spherical_kmeans;
mod state;
#[cfg(feature = "lmi-training")]
mod training;
pub use config::LmiConfig;
pub use filtered::{FilteredPlan, FilteredPlanDecision};

/// Run the same checked admission as training, without allocating samples/tensors.
pub fn check_build_memory(
    vectors: usize,
    dim: usize,
    config: &LmiConfig,
) -> OperationResult<serde_json::Value> {
    let plan = build_plan::BuildPlan::estimate(vectors, dim, config)?;
    plan.check_budget()?;
    Ok(serde_json::to_value(plan).expect("serializable LMI plan"))
}

/// Whether this binary can build new trained LMI segments. Serving does not require it.
pub const fn training_available() -> bool {
    cfg!(feature = "lmi-training")
}
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
#[cfg(feature = "testing")]
use std::sync::atomic::{AtomicU64, Ordering};

use atomic_refcell::AtomicRefCell;
use common::types::{PointOffsetType, ScoredPointOffset, TelemetryDetail};
pub use postings::CompactPostings;
pub use routing::{LinearLayer, LmiRoutingState, MlpRouter, RouterLayer};
use sparse::common::types::DimId;
#[cfg(feature = "testing")]
pub use state::save_fixture_state;
pub use state::{
    LMI_POSTINGS_FILE, LMI_ROUTER_FILE, LMI_STATE_FILE, LmiStateMetadata, RouterPreprocessing,
};

use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::{QueryVector, VectorInternal, VectorRef};
use crate::id_tracker::{IdTrackerEnum, IdTrackerRead};
use crate::index::candidate_scoring::score_candidates;
use crate::index::field_index::CardinalityEstimation;
use crate::index::plain_vector_index::PlainVectorIndex;
use crate::index::query_optimization::filter_checks::FilterChecks;
use crate::index::struct_payload_index::StructPayloadIndex;
use crate::index::{PayloadIndexRead, VectorIndex, VectorIndexRead};
use crate::telemetry::VectorIndexSearchesTelemetry;
use crate::types::{Distance, Filter, SearchParams, VectorDataConfig};
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;
use crate::vector_storage::{VectorStorageEnum, VectorStorageRead};

/// Plain-backed physical-index shell. Synthetic candidates exist only under
/// the test feature; production queries take the exact Plain path.
#[derive(Debug)]
pub struct LmiIndex {
    plain: PlainVectorIndex,
    id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
    vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
    quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
    #[cfg(feature = "testing")]
    synthetic_candidates: parking_lot::Mutex<Option<Vec<PointOffsetType>>>,
    #[cfg(feature = "testing")]
    learned_searches: AtomicU64,
    #[cfg(feature = "testing")]
    plain_fallbacks: AtomicU64,
    /// Posting offsets selected by successful learned searches, before visibility filtering.
    #[cfg(feature = "testing")]
    selected_candidate_offsets: AtomicU64,
    #[cfg(feature = "testing")]
    underfill_widenings: AtomicU64,
    /// Filtered learned searches served by exact scoring of the matching points.
    #[cfg(feature = "testing")]
    filtered_exact_searches: AtomicU64,
    /// Filtered learned searches served by routing plus posting intersection.
    #[cfg(feature = "testing")]
    filtered_routed_searches: AtomicU64,
    payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
    routing_state: Option<LmiRoutingState>,
    routing_distance: Option<Distance>,
    state_files: Vec<PathBuf>,
}

impl LmiIndex {
    pub fn new(
        id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
        vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
        quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
        payload_index: Arc<AtomicRefCell<StructPayloadIndex>>,
    ) -> Self {
        let plain = PlainVectorIndex::new(
            id_tracker.clone(),
            vector_storage.clone(),
            quantized_vectors.clone(),
            payload_index.clone(),
        );
        Self {
            plain,
            id_tracker,
            vector_storage,
            quantized_vectors,
            #[cfg(feature = "testing")]
            synthetic_candidates: parking_lot::Mutex::new(None),
            #[cfg(feature = "testing")]
            learned_searches: AtomicU64::new(0),
            #[cfg(feature = "testing")]
            plain_fallbacks: AtomicU64::new(0),
            #[cfg(feature = "testing")]
            selected_candidate_offsets: AtomicU64::new(0),
            #[cfg(feature = "testing")]
            underfill_widenings: AtomicU64::new(0),
            #[cfg(feature = "testing")]
            filtered_exact_searches: AtomicU64::new(0),
            #[cfg(feature = "testing")]
            filtered_routed_searches: AtomicU64::new(0),
            payload_index,
            routing_state: None,
            routing_distance: None,
            state_files: Vec::new(),
        }
    }

    pub(crate) fn open_trained(
        path: &Path,
        vector_config: &VectorDataConfig,
        config: LmiConfig,
        id_tracker: Arc<AtomicRefCell<IdTrackerEnum>>,
        vector_storage: Arc<AtomicRefCell<VectorStorageEnum>>,
        quantized_vectors: Arc<AtomicRefCell<Option<QuantizedVectors>>>,
        payload_index: Arc<AtomicRefCell<crate::index::struct_payload_index::StructPayloadIndex>>,
    ) -> OperationResult<Self> {
        let storage = vector_storage.borrow();
        let routing_state = state::load_local(
            path,
            config,
            vector_config,
            storage.datatype(),
            storage.total_vector_count(),
        )?;
        drop(storage);
        let mut index = Self::new(id_tracker, vector_storage, quantized_vectors, payload_index);
        index.routing_state = Some(routing_state);
        index.routing_distance = Some(vector_config.distance);
        index.state_files = state::state_files(path);
        Ok(index)
    }

    pub fn routing_state(&self) -> Option<&LmiRoutingState> {
        self.routing_state.as_ref()
    }

    /// Resident auxiliary index data only; authoritative vectors belong to VectorStorage.
    pub(crate) fn auxiliary_heap_bytes(&self) -> u64 {
        self.routing_state
            .as_ref()
            .map_or(0, |state| state.heap_bytes() as u64)
    }

    pub(crate) fn state_file_paths(&self) -> Vec<PathBuf> {
        self.state_files.clone()
    }

    /// Install deterministic offsets only in test builds. They are never persisted.
    #[cfg(feature = "testing")]
    pub fn set_synthetic_candidates_for_test(&self, candidates: Vec<PointOffsetType>) {
        *self.synthetic_candidates.lock() = Some(candidates);
    }

    #[cfg(feature = "testing")]
    pub fn search_path_counts_for_test(&self) -> (u64, u64) {
        (
            self.learned_searches.load(Ordering::Relaxed),
            self.plain_fallbacks.load(Ordering::Relaxed),
        )
    }

    /// Counts selected posting offsets, not necessarily scored or visible points.
    #[cfg(feature = "testing")]
    pub fn selected_candidate_offsets_for_test(&self) -> u64 {
        self.selected_candidate_offsets.load(Ordering::Relaxed)
    }

    fn widen_until_filled(
        &self,
        state: &LmiRoutingState,
        routed_query: &QueryVector,
        query: &QueryVector,
        top: usize,
        result: Vec<ScoredPointOffset>,
        tracker: &IdTrackerEnum,
        storage: &VectorStorageEnum,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        widen_until_filled(
            state,
            routed_query,
            query,
            top,
            result,
            tracker,
            storage,
            query_context,
            || {
                #[cfg(feature = "testing")]
                self.underfill_widenings.fetch_add(1, Ordering::Relaxed);
            },
        )
    }

    /// Learned searches that had to widen the probe to fill `top`.
    #[cfg(feature = "testing")]
    pub fn underfill_widenings_for_test(&self) -> u64 {
        self.underfill_widenings.load(Ordering::Relaxed)
    }

    /// Filtered learned searches as `(exact matches, routed intersection)`.
    #[cfg(feature = "testing")]
    pub fn filtered_plan_counts_for_test(&self) -> (u64, u64) {
        (
            self.filtered_exact_searches.load(Ordering::Relaxed),
            self.filtered_routed_searches.load(Ordering::Relaxed),
        )
    }

    /// Planner decision for `filter`, without searching.
    pub fn filtered_plan(&self, filter: &Filter) -> OperationResult<Option<FilteredPlanDecision>> {
        let Some(state) = &self.routing_state else {
            return Ok(None);
        };
        let tracker = self.id_tracker.borrow();
        let storage = self.vector_storage.borrow();
        let payload_index = self.payload_index.borrow();
        payload_index
            .with_view(|view| {
                filtered::choose_filtered_plan(state, &view, filter, &*tracker, &*storage)
            })
            .map(Some)
    }

    /// Query shapes and parameters the learned path serves, independent of filtering.
    fn learned_eligible(&self, vectors: &[&QueryVector], params: Option<&SearchParams>) -> bool {
        params.is_none_or(|p| *p == SearchParams::default())
            && self.quantized_vectors.borrow().is_none()
            && vectors
                .iter()
                .all(|q| matches!(q, QueryVector::Nearest(VectorInternal::Dense(_))))
    }

    fn supports_candidates(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        params: Option<&SearchParams>,
    ) -> bool {
        filter.is_none() && self.learned_eligible(vectors, params)
    }

    /// Adaptive filtered search; see [`filtered`] for the plan choice.
    fn filtered_search(
        &self,
        state: &LmiRoutingState,
        vectors: &[&QueryVector],
        filter: &Filter,
        top: usize,
        params: Option<&SearchParams>,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        if top == 0 {
            return Ok(vec![Vec::new(); vectors.len()]);
        }
        let tracker = self.id_tracker.borrow();
        let storage = self.vector_storage.borrow();
        let payload_index = self.payload_index.borrow();
        payload_index.with_view(|view| {
            let decision =
                filtered::choose_filtered_plan(state, &view, filter, &*tracker, &*storage)?;
            match decision.plan {
                FilteredPlan::ExactMatches => {
                    #[cfg(feature = "testing")]
                    self.filtered_exact_searches
                        .fetch_add(vectors.len() as u64, Ordering::Relaxed);
                    #[cfg(feature = "lmi-research")]
                    log::info!(target: "lmi_research", "{}", serde_json::json!({
                        "event": "lmi_filtered_search", "decision": decision,
                        "query_count": vectors.len(),
                    }));
                    self.plain
                        .search(vectors, Some(filter), top, params, query_context)
                }
                FilteredPlan::RoutedIntersect => {
                    #[cfg(feature = "testing")]
                    self.filtered_routed_searches
                        .fetch_add(vectors.len() as u64, Ordering::Relaxed);
                    let checker = view.filter_context(filter)?;
                    vectors
                        .iter()
                        .map(|query| {
                            let routed_query =
                                filtered::routing_query(self.routing_distance, query);
                            let (result, _stats) = filtered::routed_filtered_search(
                                state,
                                &routed_query,
                                query,
                                top,
                                state.nprobe(),
                                &checker,
                                &*tracker,
                                &*storage,
                                query_context,
                            )?;
                            #[cfg(feature = "testing")]
                            self.underfill_widenings
                                .fetch_add(_stats.widenings as u64, Ordering::Relaxed);
                            #[cfg(feature = "lmi-research")]
                            log::info!(target: "lmi_research", "{}", serde_json::json!({
                                "event": "lmi_filtered_search", "decision": decision,
                                "routed": _stats, "returned_hits": result.len(),
                            }));
                            Ok(result)
                        })
                        .collect()
                }
            }
        })
    }

    /// Payload-index estimate for `filter`, as the planner sees it, without searching.
    #[cfg(any(feature = "testing", feature = "lmi-research"))]
    pub fn filter_estimate(&self, filter: &Filter) -> OperationResult<filtered::FilterEstimate> {
        let tracker = self.id_tracker.borrow();
        let storage = self.vector_storage.borrow();
        let payload_index = self.payload_index.borrow();
        payload_index
            .with_view(|view| filtered::estimate_filter(&view, filter, &*tracker, &*storage))
    }

    /// Baseline break-even decision as if the index were configured with `nprobe`.
    #[cfg(any(feature = "testing", feature = "lmi-research"))]
    pub fn filtered_plan_at(
        &self,
        filter: &Filter,
        nprobe: usize,
    ) -> OperationResult<Option<FilteredPlanDecision>> {
        let Some(state) = &self.routing_state else {
            return Ok(None);
        };
        let tracker = self.id_tracker.borrow();
        let storage = self.vector_storage.borrow();
        let payload_index = self.payload_index.borrow();
        payload_index
            .with_view(|view| {
                filtered::choose_filtered_plan_with_threshold(
                    &view,
                    filter,
                    &*tracker,
                    &*storage,
                    filtered::probe_volume_at(state, nprobe),
                )
            })
            .map(Some)
    }

    /// Research entry point: one filtered query under an explicit mode, with
    /// its plan and candidate accounting. `FilteredSearchMode::Baseline` runs
    /// the same decision and search code as the production `search()`.
    #[cfg(any(feature = "testing", feature = "lmi-research"))]
    pub fn filtered_search_research(
        &self,
        query: &QueryVector,
        filter: &Filter,
        top: usize,
        mode: &filtered::FilteredSearchMode,
        query_context: &VectorQueryContext,
    ) -> OperationResult<(Vec<ScoredPointOffset>, filtered::FilteredSearchRecord)> {
        let Some(state) = &self.routing_state else {
            return Err(OperationError::service_error(
                "LMI research search requires a trained index",
            ));
        };
        if !self.learned_eligible(&[query], None) {
            return Err(OperationError::service_error(
                "LMI research search requires a dense nearest query",
            ));
        }
        let tracker = self.id_tracker.borrow();
        let storage = self.vector_storage.borrow();
        let payload_index = self.payload_index.borrow();
        payload_index.with_view(|view| {
            let mut policy_decision = None;
            let (plan, nprobe, decision) = match mode {
                filtered::FilteredSearchMode::Baseline => {
                    let decision =
                        filtered::choose_filtered_plan(state, &view, filter, &*tracker, &*storage)?;
                    (decision.plan, state.nprobe(), Some(decision))
                }
                filtered::FilteredSearchMode::ForceExact => {
                    (FilteredPlan::ExactMatches, state.nprobe(), None)
                }
                filtered::FilteredSearchMode::ForceRouted { nprobe } => {
                    (FilteredPlan::RoutedIntersect, *nprobe, None)
                }
                filtered::FilteredSearchMode::Policy(policy) => {
                    let decision = filtered::decide_policy(
                        policy, state, &view, filter, &*tracker, &*storage,
                    )?;
                    policy_decision = Some(decision);
                    (decision.plan, decision.nprobe, decision.break_even)
                }
            };
            let mut record = filtered::FilteredSearchRecord {
                plan,
                nprobe,
                decision,
                policy: policy_decision,
                routed: None,
            };
            if top == 0 {
                return Ok((Vec::new(), record));
            }
            let result = match plan {
                FilteredPlan::ExactMatches => self
                    .plain
                    .search(&[query], Some(filter), top, None, query_context)?
                    .remove(0),
                FilteredPlan::RoutedIntersect => {
                    let checker = view.filter_context(filter)?;
                    let routed_query = filtered::routing_query(self.routing_distance, query);
                    let (result, stats) = filtered::routed_filtered_search(
                        state,
                        &routed_query,
                        query,
                        top,
                        nprobe,
                        &checker,
                        &*tracker,
                        &*storage,
                        query_context,
                    )?;
                    record.routed = Some(stats);
                    result
                }
            };
            Ok((result, record))
        })
    }
}

/// An unfiltered segment search must return `min(top, live points)` hits.
/// When the configured probe prefix holds fewer live candidates (small or
/// empty buckets, deleted postings), the probe count is doubled until the
/// result fills. With every bucket probed, the postings cover all vectors
/// eligible at build time; a trained segment is immutable except for
/// deletions, so that final step equals exact search over live points.
/// Shared by the mutable and read-only LMI indexes.
#[allow(clippy::too_many_arguments)]
pub(super) fn widen_until_filled<V, I>(
    state: &LmiRoutingState,
    routed_query: &QueryVector,
    query: &QueryVector,
    top: usize,
    mut result: Vec<ScoredPointOffset>,
    tracker: &I,
    storage: &V,
    query_context: &VectorQueryContext,
    mut on_widen: impl FnMut(),
) -> OperationResult<Vec<ScoredPointOffset>>
where
    V: VectorStorageRead + crate::vector_storage::raw_scorer::RawScorerBuilder,
    I: IdTrackerRead,
{
    let needed = top.min(tracker.available_point_count());
    let buckets = state.postings().len();
    let mut nprobe = state.nprobe();
    while result.len() < needed && nprobe < buckets {
        nprobe = nprobe.saturating_mul(2).min(buckets);
        on_widen();
        let candidates = state
            .candidates_for_query_with_nprobe(routed_query, nprobe, &query_context.is_stopped())?
            .ok_or_else(|| {
                OperationError::service_error("LMI candidate routing received unsupported query")
            })?;
        result = crate::index::candidate_scoring::score_candidates_generic(
            query,
            &candidates,
            top,
            tracker,
            storage,
            query_context,
        )?;
    }
    Ok(result)
}

impl VectorIndexRead for LmiIndex {
    fn search(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        if let Some(filter) = filter
            && let Some(state) = &self.routing_state
            && self.learned_eligible(vectors, params)
        {
            return self.filtered_search(state, vectors, filter, top, params, query_context);
        }
        if self.supports_candidates(vectors, filter, params) {
            if let Some(state) = &self.routing_state {
                let tracker = self.id_tracker.borrow();
                let storage = self.vector_storage.borrow();
                return vectors
                    .iter()
                    .map(|query| {
                        #[cfg(feature = "lmi-research")]
                        let segment_began = std::time::Instant::now();
                        let routed_query = match (self.routing_distance, *query) {
                            (
                                Some(Distance::Cosine),
                                QueryVector::Nearest(VectorInternal::Dense(vector)),
                            ) => QueryVector::Nearest(VectorInternal::Dense(
                                Distance::Cosine.preprocess_vector::<f32>(vector.clone()),
                            )),
                            _ => (*query).clone(),
                        };
                        #[cfg(feature = "lmi-research")]
                        {
                            let query_preprocessing_ns = segment_began.elapsed().as_nanos() as u64;
                            let (candidates, route) = state
                                .candidates_for_query_profiled(&routed_query, &query_context.is_stopped())?
                                .ok_or_else(|| OperationError::service_error("LMI candidate routing received unsupported query"))?;
                            let (result, scoring) = crate::index::candidate_scoring::score_candidates_profiled(
                                query, &candidates, top, &*tracker, &*storage, query_context,
                            )?;
                            #[cfg(feature = "testing")]
                            {
                                self.learned_searches.fetch_add(1, Ordering::Relaxed);
                                self.selected_candidate_offsets.fetch_add(
                                    u64::try_from(candidates.len()).unwrap_or(u64::MAX), Ordering::Relaxed,
                                );
                            }
                            let segment_total_ns = segment_began.elapsed().as_nanos() as u64;
                            log::info!(target: "lmi_research", "{}", serde_json::json!({
                                "event": "lmi_segment_search", "path": "learned", "nprobe": state.nprobe(),
                                "selected_unique_offsets": candidates.len(), "query_preprocessing_ns": query_preprocessing_ns,
                                "route": route, "scoring": scoring,
                                "segment_total_ns": segment_total_ns,
                            }));
                            return self.widen_until_filled(
                                state, &routed_query, query, top, result, &*tracker, &*storage, query_context,
                            );
                        }
                        #[cfg(not(feature = "lmi-research"))]
                        {
                        let candidates = state
                            .candidates_for_query(&routed_query, &query_context.is_stopped())?
                            .ok_or_else(|| {
                                OperationError::service_error(
                                    "LMI candidate routing received unsupported query",
                                )
                            })?;
                        let result = score_candidates(
                            query,
                            &candidates,
                            top,
                            &*tracker,
                            &*storage,
                            query_context,
                        )?;
                        #[cfg(feature = "testing")]
                        {
                            self.learned_searches.fetch_add(1, Ordering::Relaxed);
                            self.selected_candidate_offsets.fetch_add(
                                u64::try_from(candidates.len()).unwrap_or(u64::MAX),
                                Ordering::Relaxed,
                            );
                        }
                        self.widen_until_filled(
                            state,
                            &routed_query,
                            query,
                            top,
                            result,
                            &*tracker,
                            &*storage,
                            query_context,
                        )
                        }
                    })
                    .collect();
            }
            #[cfg(feature = "testing")]
            if let Some(candidates) = self.synthetic_candidates.lock().clone() {
                let tracker = self.id_tracker.borrow();
                let storage = self.vector_storage.borrow();
                return vectors
                    .iter()
                    .map(|query| {
                        score_candidates(
                            query,
                            &candidates,
                            top,
                            &*tracker,
                            &*storage,
                            query_context,
                        )
                    })
                    .collect();
            }
        }
        let result = self
            .plain
            .search(vectors, filter, top, params, query_context)?;
        #[cfg(feature = "lmi-research")]
        {
            let reason = if params.is_some_and(|p| p.exact) {
                "exact_override"
            } else if filter.is_some() {
                "filter"
            } else if params.is_some_and(|p| *p != SearchParams::default()) {
                "non_default_params"
            } else if self.quantized_vectors.borrow().is_some() {
                "quantized"
            } else if vectors
                .iter()
                .any(|q| !matches!(q, QueryVector::Nearest(VectorInternal::Dense(_))))
            {
                "query_type"
            } else {
                "untrained_or_other"
            };
            log::info!(target: "lmi_research", "{}", serde_json::json!({
                "event": "lmi_segment_search", "path": "plain_fallback", "reason": reason,
                "query_count": vectors.len(), "returned_hits": result.iter().map(Vec::len).collect::<Vec<_>>(),
            }));
        }
        #[cfg(feature = "testing")]
        self.plain_fallbacks
            .fetch_add(vectors.len() as u64, Ordering::Relaxed);
        Ok(result)
    }

    fn filter_checks(
        &self,
        query_vectors: &[QueryVector],
        cardinality: &CardinalityEstimation,
        top: usize,
        params: Option<&SearchParams>,
    ) -> OperationResult<FilterChecks> {
        let queries: Vec<&QueryVector> = query_vectors.iter().collect();
        match &self.routing_state {
            Some(state) if self.learned_eligible(&queries, params) => Ok(
                filtered::learned_filter_checks(state, queries.len(), cardinality, top),
            ),
            _ => self
                .plain
                .filter_checks(query_vectors, cardinality, top, params),
        }
    }

    fn get_telemetry_data(&self, detail: TelemetryDetail) -> VectorIndexSearchesTelemetry {
        self.plain.get_telemetry_data(detail)
    }

    fn indexed_vector_count(&self) -> usize {
        self.routing_state.as_ref().map_or_else(
            || self.plain.indexed_vector_count(),
            |state| state.postings().point_count(),
        )
    }

    fn size_of_searchable_vectors_in_bytes(&self) -> usize {
        self.plain.size_of_searchable_vectors_in_bytes()
    }

    fn fill_idf_statistics(
        &self,
        idf: &mut HashMap<DimId, usize>,
        corpus: Option<&Filter>,
        is_stopped: &AtomicBool,
    ) -> OperationResult<usize> {
        self.plain.fill_idf_statistics(idf, corpus, is_stopped)
    }

    fn is_index(&self) -> bool {
        self.routing_state.is_some()
    }
}

impl VectorIndex for LmiIndex {
    fn files(&self) -> Vec<PathBuf> {
        self.state_files.clone()
    }

    fn immutable_files(&self) -> Vec<PathBuf> {
        self.state_files.clone()
    }

    fn update_vector(
        &mut self,
        id: PointOffsetType,
        vector: Option<VectorRef>,
    ) -> OperationResult<()> {
        if self.routing_state.is_some() && vector.is_some() {
            return Err(OperationError::service_error(
                "Trained LMI requires segment rebuild for vector updates",
            ));
        }
        self.plain.update_vector(id, vector)
    }

    fn update_vector_raw(
        &mut self,
        id: PointOffsetType,
        vector: Option<&[u8]>,
    ) -> OperationResult<()> {
        if self.routing_state.is_some() && vector.is_some() {
            return Err(OperationError::service_error(
                "Trained LMI requires segment rebuild for vector updates",
            ));
        }
        self.plain.update_vector_raw(id, vector)
    }
}
