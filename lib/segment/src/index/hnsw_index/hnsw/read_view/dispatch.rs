use common::condition_checker::ConditionChecker;
use common::types::ScoredPointOffset;

use super::HNSWIndexReadView;
use crate::common::operation_error::OperationResult;
use crate::common::operation_time_statistics::ScopeDurationMeasurer;
use crate::data_types::query_context::VectorQueryContext;
use crate::data_types::vectors::QueryVector;
use crate::id_tracker::IdTrackerRead;
use crate::index::PayloadIndexRead;
use crate::index::query_estimator::adjust_to_available_vectors;
use crate::index::sample_estimation::sample_check_cardinality;
use crate::types::{Filter, QuantizationSearchParams, SearchParams};
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectorsRead;
use crate::vector_storage::{RawScorerBuilder, VectorStorageRead};

impl<'a, I, V, Q, P> HNSWIndexReadView<'a, I, V, Q, P>
where
    I: IdTrackerRead,
    V: VectorStorageRead + RawScorerBuilder,
    Q: QuantizedVectorsRead,
    P: PayloadIndexRead,
{
    pub(crate) fn search(
        &self,
        vectors: &[&QueryVector],
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        query_context: &VectorQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        if top == 0 {
            return Ok(vec![vec![]; vectors.len()]);
        }

        // If neither `m` nor `payload_m` is set, HNSW doesn't have any links.
        // And if so, we need to fall back to plain search (optionally, with quantization).
        let is_hnsw_disabled = self.config.m == 0 && self.config.payload_m.unwrap_or(0) == 0;

        let exact = params.is_some_and(|params| params.exact);

        eprintln!(
            "[HNSW-DISPATCH] batch={} top={} filtered={} exact={} disabled={} available={} threshold={}",
            vectors.len(),
            top,
            filter.is_some(),
            exact,
            is_hnsw_disabled,
            self.vector_storage.available_vector_count(),
            self.config.full_scan_threshold,
        );

        let exact_params = if exact {
            params.map(|params| {
                let mut params = params.clone();
                params.quantization = Some(QuantizationSearchParams {
                    ignore: true,
                    rescore: Some(false),
                    oversampling: None,
                }); // disable quantization for exact search
                params
            })
        } else {
            None
        };

        match filter {
            None => {
                // Determine whether to do a plain or graph search.
                let plain_search = exact
                    || is_hnsw_disabled
                    || self.vector_storage.available_vector_count()
                        < self.config.full_scan_threshold;

                if plain_search {
                    let _timer = ScopeDurationMeasurer::new(if exact {
                        &self.searches_telemetry.exact_unfiltered
                    } else {
                        &self.searches_telemetry.unfiltered_plain
                    });

                    let params_ref = if exact { exact_params.as_ref() } else { params };

                    let reason = if exact {
                        "exact"
                    } else if is_hnsw_disabled {
                        "hnsw_disabled"
                    } else {
                        "below_full_scan_threshold"
                    };

                    eprintln!("[HNSW-DISPATCH] path=plain_unfiltered reason={}", reason);

                    self.search_plain_unfiltered_batched(vectors, top, params_ref, query_context)
                } else {
                    let _timer =
                        ScopeDurationMeasurer::new(&self.searches_telemetry.unfiltered_hnsw);

                    eprintln!("[HNSW-DISPATCH] path=hnsw_unfiltered");

                    self.search_vectors_with_graph(vectors, None, top, params, query_context)
                }
            }

            Some(query_filter) => {
                // Depending on the amount of filtered-out points, the optimal strategy could be:
                // - retrieve possible points and score them afterwards
                // - use HNSW with a filtering condition

                // Exact search must not use the HNSW graph.
                if exact || is_hnsw_disabled {
                    let reason = if exact { "exact" } else { "hnsw_disabled" };

                    eprintln!("[HNSW-DISPATCH] path=plain_filtered reason={}", reason);

                    let _timer = ScopeDurationMeasurer::new(if exact {
                        &self.searches_telemetry.exact_filtered
                    } else {
                        &self.searches_telemetry.filtered_plain
                    });

                    let params_ref = if exact { exact_params.as_ref() } else { params };

                    return self.search_vectors_plain(
                        vectors,
                        query_filter,
                        top,
                        params_ref,
                        query_context,
                    );
                }

                let available_vector_count = self.vector_storage.available_vector_count();

                let hw_counter = query_context.hardware_counter();

                let query_point_cardinality = self
                    .payload_index
                    .estimate_cardinality(query_filter, &hw_counter)?;

                let query_cardinality = adjust_to_available_vectors(
                    query_point_cardinality,
                    available_vector_count,
                    self.id_tracker.available_point_count(),
                );

                eprintln!(
                    "[HNSW-DISPATCH] filter_cardinality min={} max={} threshold={} available={}",
                    query_cardinality.min,
                    query_cardinality.max,
                    self.config.full_scan_threshold,
                    available_vector_count,
                );

                if query_cardinality.max < self.config.full_scan_threshold {
                    eprintln!("[HNSW-DISPATCH] path=plain_filtered_small_cardinality");

                    let _timer =
                        ScopeDurationMeasurer::new(&self.searches_telemetry.small_cardinality);

                    return self.search_vectors_plain(
                        vectors,
                        query_filter,
                        top,
                        params,
                        query_context,
                    );
                }

                if query_cardinality.min > self.config.full_scan_threshold {
                    eprintln!("[HNSW-DISPATCH] path=hnsw_filtered_large_cardinality");

                    let _timer =
                        ScopeDurationMeasurer::new(&self.searches_telemetry.large_cardinality);

                    return self.search_vectors_with_graph(
                        vectors,
                        filter,
                        top,
                        params,
                        query_context,
                    );
                }

                // Fast cardinality estimation is inconclusive.
                // Sample actual point IDs to decide whether graph search is worthwhile.
                let use_graph = {
                    let filter_context = self
                        .payload_index
                        .filter_context(query_filter, &hw_counter)?;

                    sample_check_cardinality(
                        self.id_tracker
                            .sample_ids(Some(self.vector_storage.deleted_vector_bitslice())),
                        |idx| filter_context.check(idx),
                        self.config.full_scan_threshold,
                        available_vector_count,
                    )?
                };

                eprintln!("[HNSW-DISPATCH] sampled_decision use_graph={}", use_graph);

                if use_graph {
                    eprintln!("[HNSW-DISPATCH] path=hnsw_filtered_sampled");

                    let _timer =
                        ScopeDurationMeasurer::new(&self.searches_telemetry.large_cardinality);

                    self.search_vectors_with_graph(vectors, filter, top, params, query_context)
                } else {
                    eprintln!("[HNSW-DISPATCH] path=plain_filtered_sampled");

                    let _timer =
                        ScopeDurationMeasurer::new(&self.searches_telemetry.small_cardinality);

                    self.search_vectors_plain(vectors, query_filter, top, params, query_context)
                }
            }
        }
    }
}
