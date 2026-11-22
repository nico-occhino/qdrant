use std::sync::Arc;
use std::time::Duration;

use segment::types::ScoredPoint;
use shard::common::stopping_guard::StoppingGuard;
use shard::query::query_enum::QueryEnum;
use shard::search::CoreSearchRequestBatch;

use super::LocalShard;
use crate::collection_manager::segments_searcher::SegmentsSearcher;
use crate::common::adaptive_handle::AdaptiveSearchHandle;
use crate::operations::types::{CollectionError, CollectionResult};

// Chunk requests for parallelism in certain scenarios
//
// Deeper down, each segment gets its own dedicated search thread. If this shard has just
// one segment, all requests will be executed on a single thread.
//
// To prevent this from being a bottleneck if we have a lot of requests, we can chunk the
// requests into multiple searches to allow more parallelism.
//
// For simplicity, we use a fixed chunk size. Using chunks helps to ensure our 'filter
// reuse optimization' is still properly utilized.
// See: <https://github.com/qdrant/qdrant/pull/813>
// See: <https://github.com/qdrant/qdrant/pull/6326>
const CHUNK_SIZE: usize = 16;

impl LocalShard {
    /// Activate the feature-gated local temporal DLI after native recovery has
    /// reached its WAL visibility boundary. The derived descriptor is validated
    /// against the current holder; invalid metadata leaves an empty learned
    /// membership so collection search takes native exact fallback.
    #[cfg(feature = "dli-research")]
    pub async fn configure_dli_experimental(
        &self,
        config: crate::config::DliExperimentalConfig,
    ) -> CollectionResult<bool> {
        use crate::collection_manager::dli_maintenance::DliMaintenance;
        use crate::collection_manager::dli_schedule::HierarchyConfig;
        use crate::collection_manager::segments_searcher::DliResearchConfig;
        use crate::operations::types::{Datatype, VectorsConfig};
        use segment::types::Distance;
        use std::sync::atomic::Ordering;

        config.validate()?;
        let collection = self.collection_config.read().await;
        let supported = collection.params.shard_number.get() == 1
            && collection.params.replication_factor.get() == 1
            && collection.quantization_config.is_none()
            && collection.params.sparse_vectors.is_none()
            && matches!(
                &collection.params.vectors,
                VectorsConfig::Single(vector)
                    if vector.distance == Distance::Cosine
                        && vector.datatype == Some(Datatype::Float16)
            );
        if !supported {
            return Err(CollectionError::bad_input(
                "Experimental DLI currently requires one local shard/replica, one dense Float16 cosine vector and no quantization or sparse vectors".to_owned(),
            ));
        }
        // DLI maintenance owns the physical topology of every member. Native
        // merge/indexing/vacuum workers would publish segments that are absent
        // from the DLI membership: search then falls back permanently and
        // `DliMaintenance::observe` stops scheduling (continuation 22 observed
        // "new DLI member is not appendable" until fixtures disabled them).
        // Fail closed at activation instead of degrading silently later.
        if collection.optimizer_config.max_optimization_threads != Some(0) {
            return Err(CollectionError::bad_input(
                "Experimental DLI requires optimizer_config.max_optimization_threads = 0: generic Qdrant optimizers must not restructure DLI-managed segments".to_owned(),
            ));
        }
        #[allow(deprecated)] // Match native optimizer precedence for legacy collection files.
        let destination_on_disk = match &collection.params.vectors {
            VectorsConfig::Single(vector) => {
                segment::types::Memory::resolve(
                    vector.memory,
                    vector.on_disk.map(segment::types::Memory::from_on_disk),
                )
                .unwrap_or(segment::types::Memory::Cached)
                    == segment::types::Memory::Cold
            }
            _ => unreachable!("validated single-vector scope"),
        };
        drop(collection);

        let holder = self.segments.clone();
        let initial_roles: Vec<_> = {
            let guard = holder.read();
            guard
                .iter()
                .map(|(_, segment)| (segment.get().read().segment_uuid(), 0usize))
                .collect()
        };
        let Some(search) =
            DliResearchConfig::new(initial_roles.clone(), config.arity, config.candidate_budget)
        else {
            return Err(CollectionError::bad_input(
                "Experimental DLI requires at least one unique active native segment".to_owned(),
            ));
        };
        *search.routing_policy.write() = config.routing_policy;
        search.quantized_shortlist.store(config.quantized_shortlist.unwrap_or(0), Ordering::Relaxed);
        search
            .materialize_topk
            .store(config.materialize_topk, Ordering::Relaxed);
        let search = Arc::new(search);
        let maintenance = DliMaintenance::new(
            HierarchyConfig {
                policy: config.maintenance_policy,
                buffer_capacity: config.buffer_capacity,
                arity: config.arity,
            },
            initial_roles.into_iter().collect(),
            config.model,
            self.path.clone(),
        );
        maintenance
            .destination_on_disk
            .store(destination_on_disk, Ordering::Release);
        *maintenance.level_model.lock() = config.level_model;
        maintenance
            .greedy_rebuild
            .store(config.greedy_rebuild, std::sync::atomic::Ordering::Release);
        maintenance
            .vacuum_dead_permille
            .store(config.vacuum_dead_permille, std::sync::atomic::Ordering::Release);
        maintenance.attach_search(search.clone());
        maintenance.native_enabled.store(true, Ordering::Release);

        let descriptor_path = self.path.join("dli_hierarchy.json");
        let activation = if descriptor_path.exists() {
            maintenance.load_validated_descriptor(&holder).map(|_| ())
        } else {
            maintenance.initialize_plain_buffer(&holder)
        };
        let (hierarchy_valid, maintenance_available) = match activation {
            Ok(()) => (true, true),
            Err(error) => {
                log::warn!(
                    "Experimental DLI descriptor is unavailable or invalid; using native exact fallback: {error}"
                );
                search.members.write().clear();
                let available = match maintenance.initialize_native_fallback(&holder) {
                    Ok(()) => true,
                    Err(rebuild_error) => {
                        log::warn!(
                            "Experimental DLI maintenance reconstruction is unavailable: {rebuild_error}"
                        );
                        false
                    }
                };
                (false, available)
            }
        };

        let previous = std::mem::replace(
            &mut *self.update_tracker.dli_maintenance.write(),
            maintenance_available.then_some(maintenance),
        );
        *self.dli_research.write() = Some(search);
        if let Some(previous) = previous {
            previous.cancel_and_wait().await;
        }
        Ok(hierarchy_valid)
    }

    /// Enable or disable the non-persistent internal DLI research path.
    #[cfg(all(feature = "dli-research", feature = "testing"))]
    pub fn configure_dli_research(
        &self,
        config: Option<Arc<crate::collection_manager::segments_searcher::DliResearchConfig>>,
    ) {
        *self.dli_research.write() = config;
    }

    pub async fn do_search(
        &self,
        core_request: Arc<CoreSearchRequestBatch>,
        search_runtime_handle: &AdaptiveSearchHandle,
        timeout: Duration,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        if core_request.searches.is_empty() {
            return Ok(vec![]);
        }

        let skip_batching = if core_request.searches.len() <= CHUNK_SIZE {
            // Don't batch if we have few searches, prevents cloning request
            true
        } else if self.segments.read().len() > self.shared_storage_config.search_thread_count {
            // Don't batch if we have more segments than search threads
            // Not a perfect condition, but it helps to prevent consuming a lot of search threads
            // if the number of segments is large
            // Note: search threads are shared with all other search threads on this Qdrant
            // instance, and other shards also have segments. For simplicity this only considers
            // the global search thread count and local segment count.
            // See: <https://github.com/qdrant/qdrant/pull/6478>
            true
        } else {
            false
        };

        let is_stopped_guard = StoppingGuard::new();

        if skip_batching {
            return self
                .do_search_impl(
                    core_request,
                    search_runtime_handle,
                    timeout,
                    &is_stopped_guard,
                )
                .await;
        }

        // Batch if we have many searches, allows for more parallelism
        let CoreSearchRequestBatch { searches } = core_request.as_ref();

        let chunk_futures = searches
            .chunks(CHUNK_SIZE)
            .map(|chunk| {
                let core_request = CoreSearchRequestBatch {
                    searches: chunk.to_vec(),
                };
                self.do_search_impl(
                    Arc::new(core_request),
                    search_runtime_handle,
                    timeout,
                    &is_stopped_guard,
                )
            })
            .collect::<Vec<_>>();

        let results = futures::future::try_join_all(chunk_futures)
            .await?
            .into_iter()
            .flatten()
            .collect();

        Ok(results)
    }

    async fn do_search_impl(
        &self,
        core_request: Arc<CoreSearchRequestBatch>,
        search_runtime_handle: &AdaptiveSearchHandle,
        timeout: Duration,
        is_stopped_guard: &StoppingGuard,
    ) -> CollectionResult<Vec<Vec<ScoredPoint>>> {
        let start = std::time::Instant::now();
        let (query_context, distances) = {
            let collection_config = self.collection_config.read().await;
            let query_context_opt = SegmentsSearcher::prepare_query_context(
                self.segments.clone(),
                &core_request,
                &collection_config,
                timeout,
                search_runtime_handle,
                is_stopped_guard,
            )
            .await?;

            let Some(query_context) = query_context_opt else {
                // No segments to search
                return Ok(vec![]);
            };

            // Resolve the per-request distances now, rather than cloning all collection params
            let distances = core_request
                .searches
                .iter()
                .map(|req| {
                    collection_config
                        .params
                        .get_distance(req.query.get_vector_name())
                })
                .collect::<Vec<_>>();

            (query_context, distances)
        };

        // update timeout
        let timeout = timeout.saturating_sub(start.elapsed());

        let search_request = async {
            #[cfg(feature = "dli-research")]
            {
                let config = self.dli_research.read().clone();
                if let Some(config) = config {
                    return SegmentsSearcher::search_dli_research(
                        self.segments.clone(),
                        core_request.clone(),
                        search_runtime_handle,
                        query_context,
                        timeout,
                        config,
                    )
                    .await;
                }
            }
            SegmentsSearcher::search(
                self.segments.clone(),
                core_request.clone(),
                search_runtime_handle,
                true,
                query_context,
                timeout,
            )
            .await
        };

        let res = tokio::time::timeout(timeout, search_request)
            .await
            .map_err(|_| {
                log::debug!("Search timeout reached: {timeout:?}");
                // StoppingGuard takes care of setting is_stopped to true
                CollectionError::timeout(timeout, "Search")
            })??;

        let top_results = res
            .into_iter()
            .zip(core_request.searches.iter())
            .zip(distances)
            .map(|((vector_res, req), distance)| {
                let distance = distance.unwrap();
                let processed_res = vector_res.into_iter().map(|mut scored_point| {
                    match req.query {
                        QueryEnum::Nearest(_) => {
                            scored_point.score = distance.postprocess_score(scored_point.score);
                        }
                        // Don't post-process if we are dealing with custom scoring
                        QueryEnum::RecommendBestScore(_)
                        | QueryEnum::RecommendSumScores(_)
                        | QueryEnum::Discover(_)
                        | QueryEnum::Context(_)
                        | QueryEnum::FeedbackNaive(_) => {}
                    };
                    scored_point
                });

                if let Some(threshold) = req.score_threshold {
                    let order = req.query.score_order(distance);
                    processed_res
                        .take_while(|scored_point| {
                            order.check_threshold(scored_point.score, threshold)
                        })
                        .collect()
                } else {
                    processed_res.collect()
                }
            })
            .collect();
        Ok(top_results)
    }
}
