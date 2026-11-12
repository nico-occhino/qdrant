//! Experimental native routing primitive, not a collection search entry point.
//!
//! Callers must keep every segment read-locked for the whole call. The returned
//! per-segment results are deliberately not merged: candidate-only version
//! aggregation cannot establish that a newer, unselected copy does not exist.
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use segment::common::operation_error::{OperationError, OperationResult};
use segment::data_types::query_context::VectorQueryContext;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, QueryVector};
use segment::entry::ReadSegmentEntry;
use segment::id_tracker::IdTrackerRead;
use segment::index::candidate_scoring::score_candidates;
use segment::index::{VectorIndexEnum, VectorIndexRead};
use segment::segment::Segment;
use segment::types::{Distance, ScoredPoint, WithPayload, WithVector};
use uuid::Uuid;

use super::dli_bucket_plan::{BucketPrediction, SelectedBuckets, select};

/// Output is partitioned by physical segment. It must not be returned to clients
/// until complete native point-version/visibility resolution has been applied.
pub struct NativeRouteResult {
    pub plan: SelectedBuckets,
    pub routing_and_planning_time: Duration,
    pub scoring_time: Duration,
    pub posting_candidates: usize,
    pub unique_posting_offsets: usize,
    /// Live point count, which can exceed the number of vectors in the buffer.
    pub buffer_point_estimate: usize,
    pub per_segment: Vec<(Uuid, Vec<ScoredPoint>)>,
}

/// Route and score one dense, unfiltered query on explicitly pinned native
/// segment references. Level zero denotes a Plain buffer, all other levels
/// denote independently trained LMI indexes. The caller supplies complete
/// membership; this function cannot validate an unseen holder topology.
pub fn route_pinned(
    members: &[(usize, &Segment)],
    query: &[f32],
    arity: usize,
    budget: usize,
    context: &VectorQueryContext,
) -> OperationResult<NativeRouteResult> {
    let fail = |message: &str| OperationError::service_error(message.to_owned());
    let start = Instant::now();
    let mut predictions = Vec::new();
    let mut seen = HashSet::new();
    let mut distance = None;
    for &(level, segment) in members {
        if !seen.insert(segment.uuid) {
            return Err(fail("duplicate DLI physical member"));
        }
        let config = segment
            .config()
            .vector_data
            .get(DEFAULT_VECTOR_NAME)
            .ok_or_else(|| fail("missing default vector configuration"))?;
        if config.size != query.len() || distance.is_some_and(|d| d != config.distance) {
            return Err(fail("incompatible DLI vector configuration"));
        }
        distance = Some(config.distance);
        let data = segment
            .vector_data
            .get(DEFAULT_VECTOR_NAME)
            .ok_or_else(|| fail("missing default vector storage"))?;
        if config.multivector_config.is_some() || data.quantized_vectors.borrow().is_some() {
            return Err(fail("unsupported DLI vector storage"));
        }
        let index = data.vector_index.borrow();
        match (level, &*index) {
            (0, VectorIndexEnum::Plain(_)) => predictions.push(BucketPrediction {
                level,
                physical: segment.uuid,
                bucket: None,
                probability: 0.0,
                occupancy: segment.id_tracker.borrow().available_point_count(),
            }),
            (1.., VectorIndexEnum::Lmi(lmi)) => {
                let state = lmi
                    .routing_state()
                    .ok_or_else(|| fail("missing LMI routing state"))?;
                let routed = if config.distance == Distance::Cosine {
                    Distance::Cosine.preprocess_vector::<f32>(query.to_vec())
                } else {
                    query.to_vec()
                };
                let logits = state.router().forward(&routed)?;
                if logits.len() != state.postings().len() || logits.is_empty() {
                    return Err(fail("LMI router/postings bucket mismatch"));
                }
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = logits.iter().map(|v| (v - max).exp()).sum();
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(fail("invalid LMI router probabilities"));
                }
                for (bucket, logit) in logits.iter().enumerate() {
                    predictions.push(BucketPrediction {
                        level,
                        physical: segment.uuid,
                        bucket: Some(bucket),
                        probability: (logit - max).exp() / denominator,
                        occupancy: state
                            .postings()
                            .get(bucket)
                            .expect("validated bucket")
                            .len(),
                    });
                }
            }
            _ => return Err(fail("DLI member role/index mismatch")),
        }
    }
    let plan = select(predictions, arity, budget)
        .map_err(|e| fail(&format!("invalid DLI candidate plan: {e:?}")))?;
    let routing_and_planning_time = start.elapsed();
    let started_scoring = Instant::now();
    let mut posting_candidates = 0usize;
    let mut unique_posting_offsets = 0usize;
    let mut buffer_point_estimate = 0usize;
    let mut per_segment = Vec::new();
    let query_vector: QueryVector = query.to_vec().into();
    let mut grouped: HashMap<Uuid, Vec<usize>> = HashMap::new();
    for bucket in &plan.buckets {
        grouped
            .entry(bucket.physical)
            .or_default()
            .extend(bucket.bucket);
    }
    for &(level, segment) in members {
        let Some(buckets) = grouped.get(&segment.uuid) else {
            continue;
        };
        let data = &segment.vector_data[DEFAULT_VECTOR_NAME];
        let index = data.vector_index.borrow();
        let offsets = if level == 0 {
            // Whole buffer, not a partial top-k: the outer caller owns global top-k.
            let count = segment.id_tracker.borrow().available_point_count();
            buffer_point_estimate += count;
            index
                .search(&[&query_vector], None, count, None, context)?
                .pop()
                .expect("one buffer query")
        } else {
            let VectorIndexEnum::Lmi(lmi) = &*index else {
                unreachable!()
            };
            let state = lmi.routing_state().expect("validated while pinned");
            let mut candidates = Vec::new();
            for &bucket in buckets {
                candidates.extend_from_slice(state.postings().get(bucket).expect("planned bucket"));
            }
            posting_candidates += candidates.len();
            candidates.sort_unstable();
            candidates.dedup();
            unique_posting_offsets += candidates.len();
            let tracker = segment.id_tracker.borrow();
            let storage = data.vector_storage.borrow();
            score_candidates(
                &query_vector,
                &candidates,
                candidates.len(),
                &tracker,
                &storage,
                context,
            )?
        };
        let hits = segment.with_view(|view| {
            view.process_search_result(
                offsets,
                &WithPayload::default(),
                &WithVector::default(),
                &context.is_stopped(),
            )
        })?;
        per_segment.push((segment.uuid, hits));
    }
    Ok(NativeRouteResult {
        plan,
        routing_and_planning_time,
        scoring_time: started_scoring.elapsed(),
        posting_candidates,
        unique_posting_offsets,
        buffer_point_estimate,
        per_segment,
    })
}

#[cfg(all(test, feature = "lmi-training", feature = "testing"))]
mod tests {
    use common::flags::FeatureFlags;
    use segment::data_types::vectors::only_default_vector;
    use segment::entry::entry_point::{
        NonAppendableSegmentEntry, SegmentEntry, StorageSegmentEntry,
    };
    use segment::index::lmi_index::LmiConfig;
    use segment::segment_constructor::load_segment;
    use segment::segment_constructor::segment_builder::SegmentBuilder;
    use segment::segment_constructor::simple_segment_constructor::build_simple_segment;
    use segment::types::{HnswGlobalConfig, Indexes, PointIdType};
    use std::sync::atomic::AtomicBool;

    use super::*;

    fn trained(root: &std::path::Path, rows: &[(u64, [f32; 2])], seed: u64) -> Segment {
        let staging = tempfile::tempdir().unwrap();
        let mut src = build_simple_segment(root, 2, Distance::Cosine).unwrap();
        for (id, vector) in rows {
            src.upsert_point(*id, (*id).into(), only_default_vector(vector))
                .unwrap();
        }
        let mut config = src.config().clone();
        config
            .vector_data
            .get_mut(DEFAULT_VECTOR_NAME)
            .unwrap()
            .index = Indexes::LmiTrained(LmiConfig {
            n_buckets: 2,
            sample_size: 4,
            hidden_dim: 4,
            epochs: 1,
            batch_size: 4,
            routing_batch_size: 4,
            kmeans_iterations: 2,
            nprobe: 2,
            seed,
        });
        let mut builder = SegmentBuilder::new(
            staging.path(),
            &config,
            &HnswGlobalConfig::default(),
            FeatureFlags::default(),
        )
        .unwrap();
        builder.update(&[&src], &AtomicBool::new(false)).unwrap();
        builder.build_for_test(root)
    }

    #[test]
    fn two_native_routers_global_plan_and_per_segment_scoring() {
        let _scope = common::ambient::unmeasured_guard(common::reason::reason("DLI native route"));
        let root = tempfile::tempdir().unwrap();
        let rows = [
            (11, [1., 0.]),
            (12, [0.8, 0.6]),
            (13, [0., 1.]),
            (14, [-0.6, 0.8]),
            (21, [-1., 0.]),
            (22, [-0.8, -0.6]),
            (23, [0., -1.]),
            (24, [0.6, -0.8]),
        ];
        let mut a = trained(root.path(), &rows[..4], 17);
        let mut b = trained(root.path(), &rows[4..], 29);
        let mut buffer = build_simple_segment(root.path(), 2, Distance::Cosine).unwrap();
        a.delete_point(100, 11_u64.into()).unwrap();
        b.delete_point(101, 21_u64.into()).unwrap();
        buffer
            .upsert_point(102, 11_u64.into(), only_default_vector(&[0., 1.]))
            .unwrap();
        buffer
            .upsert_point(103, 31_u64.into(), only_default_vector(&[0.6, 0.8]))
            .unwrap();
        let members = [(1, &a), (2, &b), (0, &buffer)];
        let context = VectorQueryContext::default();
        let all = route_pinned(&members, &[1., 0.], 2, 100, &context).unwrap();
        assert_eq!(
            all.plan
                .buckets
                .iter()
                .map(|b| b.level)
                .collect::<HashSet<_>>(),
            HashSet::from([0, 1, 2])
        );
        let mut ids = HashSet::new();
        for (_, hits) in &all.per_segment {
            for hit in hits {
                let PointIdType::NumId(id) = hit.id else {
                    panic!("numeric fixture")
                };
                assert!(ids.insert(id));
                if id == 11 {
                    assert_eq!(hit.version, 102);
                }
                let expected = match id {
                    11 => 0.,
                    31 => 0.6,
                    21 => panic!("deleted"),
                    _ => rows.iter().find(|r| r.0 == id).unwrap().1[0],
                };
                assert!((hit.score - expected).abs() < 1e-5, "id={id}");
            }
        }
        assert_eq!(ids, HashSet::from([11, 12, 13, 14, 22, 23, 24, 31]));
        let limited = route_pinned(&members, &[1., 0.], 2, 2, &context).unwrap();
        assert!(limited.plan.estimated_candidates >= 2);
        assert!(
            limited.unique_posting_offsets + limited.buffer_point_estimate
                <= limited.plan.estimated_candidates
        );
        assert!(route_pinned(&[(1, &a), (2, &a)], &[1., 0.], 2, 2, &context).is_err());
        assert!(route_pinned(&[(1, &buffer)], &[1., 0.], 2, 2, &context).is_err());
        println!(
            "DLI_ROUTE budget=2 selected={} postings={} unique_offsets_plus_buffer_estimate={} returned={}",
            limited.plan.buckets.len(),
            limited.posting_candidates,
            limited.unique_posting_offsets + limited.buffer_point_estimate,
            limited
                .per_segment
                .iter()
                .map(|(_, h)| h.len())
                .sum::<usize>()
        );

        // Private native build from the three inputs; not optimizer publication.
        let staging = tempfile::tempdir().unwrap();
        let config = a.config().clone();
        let mut builder = SegmentBuilder::new(
            staging.path(),
            &config,
            &HnswGlobalConfig::default(),
            FeatureFlags::default(),
        )
        .unwrap();
        builder
            .update(&[&a, &b, &buffer], &AtomicBool::new(false))
            .unwrap();
        let replacement = builder.build_for_test(root.path());
        let rebuilt = route_pinned(&[(1, &replacement)], &[1., 0.], 2, 100, &context).unwrap();
        let rebuilt_ids: HashSet<_> = rebuilt
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits.iter().map(|h| h.id))
            .collect();
        assert_eq!(
            rebuilt_ids,
            ids.into_iter().map(PointIdType::from).collect()
        );
        let expected_scores: HashMap<_, _> = all
            .per_segment
            .iter()
            .flat_map(|(_, hits)| hits.iter().map(|h| (h.id, h.score)))
            .collect();
        for (_, hits) in &rebuilt.per_segment {
            for hit in hits {
                assert!((hit.score - expected_scores[&hit.id]).abs() < 1e-5);
            }
        }
        replacement.flush(true).unwrap();
        let path = replacement.segment_path.clone();
        let uuid = replacement.uuid;
        drop(replacement);
        let reopened = load_segment(&path, uuid, None, &AtomicBool::new(false), true).unwrap();
        let after = route_pinned(&[(1, &reopened)], &[1., 0.], 2, 100, &context).unwrap();
        assert_eq!(after.per_segment[0].1.len(), rebuilt_ids.len());
        for hit in &after.per_segment[0].1 {
            assert!((hit.score - expected_scores[&hit.id]).abs() < 1e-5);
        }
    }
}
