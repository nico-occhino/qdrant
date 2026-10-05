use std::sync::atomic::AtomicBool;

use common::generic_consts::Random;
use common::types::PointOffsetType;

use super::CompactPostings;

use crate::common::operation_error::{check_process_stopped, OperationError, OperationResult};
use crate::data_types::named_vectors::CowVector;
use crate::data_types::vectors::{QueryVector, VectorInternal};
use crate::vector_storage::VectorStorageRead;

/// Candidate source used by the current experimental LMI integration.
///
/// `AllValidPoints` remains the exact/default behavior.
/// `DeterministicTwoBuckets` is the Phase C fixture.
/// `StaticLearned` is the Phase D3 learned-routing fixture.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LmiCandidateMode {
    /// Exact full-precision baseline.
    #[default]
    AllValidPoints,

    /// Phase C fixture:
    /// query sign selects even/odd synthetic postings.
    DeterministicTwoBuckets,

    /// Phase D3:
    /// native MLP router + router-predicted static postings.
    StaticLearned,
}

// ============================================================
// Phase C deterministic fixture
// ============================================================

/// Phase C only.
///
/// Rebuilt from offsets on every search batch.
/// No learned semantics.
pub(super) fn fake_postings(
    vector_count: usize,
    stopped: &AtomicBool,
) -> OperationResult<[Vec<PointOffsetType>; 2]> {
    let mut buckets = [Vec::new(), Vec::new()];

    for offset in 0..vector_count {
        check_process_stopped(stopped)?;

        buckets[offset % 2].push(offset as PointOffsetType);
    }

    Ok(buckets)
}

/// Phase C only.
///
/// Nonnegative first coordinate -> bucket 0.
/// Negative first coordinate -> bucket 1.
pub(super) fn fake_route(query: &QueryVector) -> Option<usize> {
    let QueryVector::Nearest(VectorInternal::Dense(vector)) = query else {
        return None;
    };

    let first = *vector.first()?;

    first.is_finite().then_some(usize::from(first < 0.0))
}

// ============================================================
// Phase D3 native learned router
// ============================================================

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LinearLayer {
    pub in_features: usize,
    pub out_features: usize,

    /// Row-major matrix:
    ///
    /// [out_features, in_features]
    pub weights: Vec<f32>,

    pub bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RouterLayer {
    Linear(LinearLayer),

    ReLU,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MlpRouter {
    pub layers: Vec<RouterLayer>,
}

impl MlpRouter {
    /// Validate ordered layers before any indexing/allocation based on shapes.
    /// The public draft representation stays easy to construct in tiny tests.
    pub fn validate(&self) -> OperationResult<(usize, usize)> {
        let Some(RouterLayer::Linear(first)) = self.layers.first() else {
            return Err(OperationError::service_error(
                "LMI router must start with Linear",
            ));
        };
        if !matches!(self.layers.last(), Some(RouterLayer::Linear(_))) {
            return Err(OperationError::service_error(
                "LMI router must end with Linear logits",
            ));
        }
        let mut dim = first.in_features;
        for layer in &self.layers {
            if let RouterLayer::Linear(linear) = layer {
                if linear.in_features == 0 || linear.out_features == 0 || linear.in_features != dim
                {
                    return Err(OperationError::service_error(
                        "LMI router has zero or incompatible layer dimensions",
                    ));
                }
                let count = linear
                    .in_features
                    .checked_mul(linear.out_features)
                    .ok_or_else(|| {
                        OperationError::service_error("LMI router weight dimensions overflow")
                    })?;
                if linear.weights.len() != count || linear.bias.len() != linear.out_features {
                    return Err(OperationError::service_error(
                        "LMI router malformed weight or bias count",
                    ));
                }
                if linear
                    .weights
                    .iter()
                    .chain(&linear.bias)
                    .any(|v| !v.is_finite())
                {
                    return Err(OperationError::service_error(
                        "LMI router parameters must be finite",
                    ));
                }
                dim = linear.out_features;
            }
        }
        Ok((first.in_features, dim))
    }

    pub fn output_dim(&self) -> OperationResult<usize> {
        self.validate().map(|(_, output)| output)
    }

    pub fn forward(&self, query: &[f32]) -> OperationResult<Vec<f32>> {
        self.forward_with_stop(query, &AtomicBool::new(false))
    }

    fn forward_with_stop(&self, query: &[f32], stopped: &AtomicBool) -> OperationResult<Vec<f32>> {
        check_process_stopped(stopped)?;
        let (input_dim, _) = self.validate()?;
        if query.len() != input_dim || query.iter().any(|v| !v.is_finite()) {
            return Err(OperationError::service_error(
                "LMI router input has wrong dimension or non-finite values",
            ));
        }
        let mut values = query.to_vec();
        for layer in &self.layers {
            check_process_stopped(stopped)?;
            match layer {
                RouterLayer::Linear(linear) => {
                    let mut output = Vec::with_capacity(linear.out_features);
                    for (row, &bias) in linear
                        .weights
                        .chunks_exact(linear.in_features)
                        .zip(&linear.bias)
                    {
                        check_process_stopped(stopped)?;
                        let mut sum = bias;
                        for (&weight, &value) in row.iter().zip(&values) {
                            sum += weight * value;
                        }
                        // Check before ReLU so it cannot hide NaNs or -infinity.
                        if !sum.is_finite() {
                            return Err(OperationError::service_error(
                                "LMI router produced non-finite activations/logits",
                            ));
                        }
                        output.push(sum);
                    }
                    values = output;
                }
                RouterLayer::ReLU => values.iter_mut().for_each(|v| *v = v.max(0.0)),
            }
        }
        Ok(values)
    }

    pub fn top_buckets(&self, query: &[f32], nprobe: usize) -> OperationResult<Vec<usize>> {
        self.top_buckets_with_stop(query, nprobe, &AtomicBool::new(false))
    }

    pub(super) fn top_buckets_with_stop(
        &self,
        query: &[f32],
        nprobe: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Vec<usize>> {
        check_process_stopped(stopped)?;
        let count = self.output_dim()?;
        if nprobe == 0 || nprobe > count {
            return Err(OperationError::service_error(format!(
                "LMI nprobe must be in 1..={count}, got {nprobe}"
            )));
        }
        let logits = self.forward_with_stop(query, stopped)?;
        let mut bucket_ids: Vec<usize> = (0..count).collect();
        // Finite logits suffice: softmax preserves order. Equal logits (including
        // signed zero) prefer the smaller bucket ID; tch need not share that tie rule.
        bucket_ids.sort_by(|&a, &b| {
            if logits[a] == logits[b] {
                a.cmp(&b)
            } else {
                logits[b].total_cmp(&logits[a])
            }
        });
        bucket_ids.truncate(nprobe);
        check_process_stopped(stopped)?;
        Ok(bucket_ids)
    }
}

// ============================================================
// Static learned routing state
// ============================================================

#[derive(Debug, Clone, PartialEq)]
pub struct LmiRoutingState {
    router: MlpRouter,

    /// postings[bucket]
    ///     = internal Qdrant point offsets assigned to that bucket.
    ///
    /// Crucially, these are router-predicted assignments.
    postings: CompactPostings,

    nprobe: usize,
}

impl LmiRoutingState {
    pub fn postings(&self) -> &CompactPostings {
        &self.postings
    }

    pub fn nprobe(&self) -> usize {
        self.nprobe
    }
    #[cfg(test)]
    pub(super) fn top_buckets_for_test(
        &self,
        normalized_query: &[f32],
        nprobe: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Vec<usize>> {
        self.router
            .top_buckets_with_stop(normalized_query, nprobe, stopped)
    }

    pub fn new(
        router: MlpRouter,
        postings: Vec<Vec<PointOffsetType>>,
        nprobe: usize,
    ) -> OperationResult<Self> {
        Self::from_compact(router, CompactPostings::from_buckets(postings)?, nprobe)
    }

    pub(super) fn from_compact(
        router: MlpRouter,
        postings: CompactPostings,
        nprobe: usize,
    ) -> OperationResult<Self> {
        postings.validate()?;
        let bucket_count = router.output_dim()?;

        if postings.len() != bucket_count {
            return Err(OperationError::service_error(format!(
                concat!(
                    "LMI postings/router mismatch: ",
                    "{} postings buckets, ",
                    "{} router outputs"
                ),
                postings.len(),
                bucket_count,
            )));
        }

        if nprobe == 0 || nprobe > bucket_count {
            return Err(OperationError::service_error(format!(
                concat!("LMI nprobe must be in ", "1..={}, got {}"),
                bucket_count, nprobe,
            )));
        }

        Ok(Self {
            router,
            postings,
            nprobe,
        })
    }

    /// Route a dense nearest-neighbor query to top-nprobe
    /// buckets and produce the union of their postings.
    ///
    /// Returns None for unsupported query types so the caller
    /// can fall back to Plain.
    pub fn candidates_for_query(
        &self,
        query: &QueryVector,
        stopped: &AtomicBool,
    ) -> OperationResult<Option<Vec<PointOffsetType>>> {
        check_process_stopped(stopped)?;
        let QueryVector::Nearest(VectorInternal::Dense(vector)) = query else {
            return Ok(None);
        };

        let buckets = self
            .router
            .top_buckets_with_stop(vector.as_slice(), self.nprobe, stopped)?;

        let mut candidates = Vec::new();

        for bucket in buckets {
            let Some(posting) = self.postings.get(bucket) else {
                return Err(OperationError::service_error(format!(
                    concat!("LMI router selected ", "missing bucket {}"),
                    bucket,
                )));
            };

            for &point_offset in posting {
                check_process_stopped(stopped)?;
                candidates.push(point_offset);
            }
        }

        // Native top-k can keep a different equal-score boundary point when
        // arrival order differs. Ascending offsets match Plain full-scan order.
        candidates.sort_unstable();
        candidates.dedup();
        check_process_stopped(stopped)?;
        Ok(Some(candidates))
    }
}

// ============================================================
// Database-vector -> learned postings
// ============================================================

/// Build final LMI postings from the router's own predictions.
///
/// This is intentionally NOT based on the original KMeans labels.
///
/// For each stored vector x_i:
///
///     bucket_i = argmax f_theta(x_i)
///
/// and:
///
///     postings[bucket_i].push(point_offset_i)
///
/// Qdrant remains authoritative for the actual vectors.
/// Only internal point offsets are stored here.
pub fn build_router_postings(
    router: &MlpRouter,
    vector_storage: &impl VectorStorageRead,
    stopped: &AtomicBool,
) -> OperationResult<Vec<Vec<PointOffsetType>>> {
    check_process_stopped(stopped)?;
    let bucket_count = router.output_dim()?;

    let mut postings = vec![Vec::new(); bucket_count];

    for offset in 0..vector_storage.total_vector_count() {
        check_process_stopped(stopped)?;
        let point_offset = PointOffsetType::try_from(offset).map_err(|_| {
            OperationError::service_error("LMI vector offset exceeds PointOffsetType")
        })?;
        // Tombstones may have no readable vector. Point/deferred visibility is
        // still checked by the scoring seam, both now and after installation.
        if vector_storage.is_deleted_vector(point_offset) {
            continue;
        }
        let vector = vector_storage
            .get_vector_opt::<Random>(point_offset)
            .ok_or_else(|| OperationError::service_error("LMI stored vector is missing"))?;

        let CowVector::Dense(dense) = vector else {
            return Err(OperationError::service_error(concat!(
                "Phase D3 static LMI ",
                "currently supports only ",
                "dense vector storage"
            )));
        };

        let bucket = router
            .top_buckets_with_stop(dense.as_ref(), 1, stopped)?
            .into_iter()
            .next()
            .ok_or_else(|| OperationError::service_error("LMI router returned no bucket"))?;

        postings[bucket].push(point_offset);
    }

    Ok(postings)
}

/// A build-local immutable model borrow and reusable inference workspace.
/// Construction validates all weights/shapes once; the borrow prevents mutation.
#[cfg(any(feature = "lmi-training", test))]
#[allow(dead_code)] // Retained as the S.3B1 scalar correctness/performance reference.
pub(super) struct BuildRouter<'a> {
    router: &'a MlpRouter,
    input_dim: usize,
    values: Vec<f32>,
    output: Vec<f32>,
}

#[cfg(any(feature = "lmi-training", test))]
#[allow(dead_code)] // Its methods are exercised by focused parity tests.
impl<'a> BuildRouter<'a> {
    pub(super) fn new(router: &'a MlpRouter, stopped: &AtomicBool) -> OperationResult<Self> {
        check_process_stopped(stopped)?;
        let (input_dim, _) = router.validate()?;
        let width = router
            .layers
            .iter()
            .fold(input_dim, |width, layer| match layer {
                RouterLayer::Linear(layer) => width.max(layer.out_features),
                RouterLayer::ReLU => width,
            });
        let allocate = || {
            let mut buffer = Vec::new();
            buffer.try_reserve_exact(width).map_err(|e| {
                OperationError::service_error(format!("LMI inference workspace allocation: {e}"))
            })?;
            Ok::<_, OperationError>(buffer)
        };
        let result = Self {
            router,
            input_dim,
            values: allocate()?,
            output: allocate()?,
        };
        check_process_stopped(stopped)?;
        Ok(result)
    }

    pub(super) fn top_bucket(
        &mut self,
        query: &[f32],
        stopped: &AtomicBool,
    ) -> OperationResult<usize> {
        check_process_stopped(stopped)?;
        if query.len() != self.input_dim || query.iter().any(|v| !v.is_finite()) {
            return Err(OperationError::service_error(
                "LMI router input has wrong dimension or non-finite values",
            ));
        }
        self.values.clear();
        self.values.extend_from_slice(query);
        for layer in &self.router.layers {
            check_process_stopped(stopped)?;
            match layer {
                RouterLayer::Linear(linear) => {
                    self.output.clear();
                    for (row, &bias) in linear
                        .weights
                        .chunks_exact(linear.in_features)
                        .zip(&linear.bias)
                    {
                        check_process_stopped(stopped)?;
                        let mut sum = bias;
                        for (&weight, &value) in row.iter().zip(&self.values) {
                            sum += weight * value;
                        }
                        if !sum.is_finite() {
                            return Err(OperationError::service_error(
                                "LMI router produced non-finite activations/logits",
                            ));
                        }
                        self.output.push(sum);
                    }
                    std::mem::swap(&mut self.values, &mut self.output);
                }
                RouterLayer::ReLU => self.values.iter_mut().for_each(|v| *v = v.max(0.0)),
            }
        }
        // Strict comparison preserves smaller-ID preference, including signed zero.
        let mut best = 0;
        for bucket in 1..self.values.len() {
            if self.values[bucket] > self.values[best] {
                best = bucket;
            }
        }
        check_process_stopped(stopped)?;
        Ok(best)
    }
}

/// A build-only, single-threaded batch router. It retains the scalar
/// multiply/add order for each row, but reuses contiguous K-row workspaces.
/// Therefore every top-1 assignment is bit/partition-equivalent to
/// [`BuildRouter`] for the same model and inputs.
#[cfg(any(feature = "lmi-training", test))]
pub(super) struct BuildBatchRouter<'a> {
    router: &'a MlpRouter,
    input_dim: usize,
    batch_capacity: usize,
    values: Vec<f32>,
    output: Vec<f32>,
    buckets: Vec<usize>,
}

#[cfg(any(feature = "lmi-training", test))]
impl<'a> BuildBatchRouter<'a> {
    pub(super) fn new(
        router: &'a MlpRouter,
        batch_capacity: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        check_process_stopped(stopped)?;
        if batch_capacity == 0 {
            return Err(OperationError::service_error(
                "LMI build routing batch capacity must be nonzero",
            ));
        }
        let (input_dim, _) = router.validate()?;
        let width = router
            .layers
            .iter()
            .fold(input_dim, |width, layer| match layer {
                RouterLayer::Linear(layer) => width.max(layer.out_features),
                RouterLayer::ReLU => width,
            });
        let allocate_f32 = |rows: usize, columns: usize| -> OperationResult<Vec<f32>> {
            let capacity = rows.checked_mul(columns).ok_or_else(|| {
                OperationError::service_error("LMI build routing workspace size overflow")
            })?;
            let mut buffer = Vec::new();
            buffer.try_reserve_exact(capacity).map_err(|e| {
                OperationError::service_error(format!("LMI inference workspace allocation: {e}"))
            })?;
            Ok(buffer)
        };
        let mut buckets = Vec::new();
        buckets.try_reserve_exact(batch_capacity).map_err(|e| {
            OperationError::service_error(format!("LMI inference workspace allocation: {e}"))
        })?;
        let result = Self {
            router,
            input_dim,
            batch_capacity,
            values: allocate_f32(batch_capacity, width)?,
            output: allocate_f32(batch_capacity, width)?,
            buckets,
        };
        check_process_stopped(stopped)?;
        Ok(result)
    }

    /// Actual reusable workspace: two K×max(d,H,B) activation buffers plus
    /// K bucket identifiers. The caller owns the K×d gathered input buffer.
    pub(super) fn workspace_bytes(&self) -> usize {
        (self.values.capacity() + self.output.capacity()) * std::mem::size_of::<f32>()
            + self.buckets.capacity() * std::mem::size_of::<usize>()
    }

    pub(super) fn top_buckets(
        &mut self,
        inputs: &[f32],
        stopped: &AtomicBool,
    ) -> OperationResult<&[usize]> {
        check_process_stopped(stopped)?;
        if inputs.len() % self.input_dim != 0 {
            return Err(OperationError::service_error(
                "LMI batch router input has wrong dimension",
            ));
        }
        let rows = inputs.len() / self.input_dim;
        if rows > self.batch_capacity {
            return Err(OperationError::service_error(
                "LMI batch router input exceeds configured batch capacity",
            ));
        }
        if inputs.iter().any(|value| !value.is_finite()) {
            return Err(OperationError::service_error(
                "LMI router input has non-finite values",
            ));
        }
        self.values.clear();
        self.values.extend_from_slice(inputs);
        let mut columns = self.input_dim;
        for layer in &self.router.layers {
            check_process_stopped(stopped)?;
            match layer {
                RouterLayer::Linear(linear) => {
                    self.output.clear();
                    for input in self.values.chunks_exact(columns) {
                        check_process_stopped(stopped)?;
                        for (weights, &bias) in linear
                            .weights
                            .chunks_exact(linear.in_features)
                            .zip(&linear.bias)
                        {
                            // Preserve BuildRouter's ordered f32 reduction exactly.
                            let mut sum = bias;
                            for (&weight, &value) in weights.iter().zip(input) {
                                sum += weight * value;
                            }
                            if !sum.is_finite() {
                                return Err(OperationError::service_error(
                                    "LMI router produced non-finite activations/logits",
                                ));
                            }
                            self.output.push(sum);
                        }
                    }
                    columns = linear.out_features;
                    std::mem::swap(&mut self.values, &mut self.output);
                }
                RouterLayer::ReLU => self
                    .values
                    .iter_mut()
                    .for_each(|value| *value = value.max(0.0)),
            }
        }
        self.buckets.clear();
        for logits in self.values.chunks_exact(columns) {
            // Strict comparison preserves smaller-ID preference, including signed zero.
            let mut best = 0;
            for bucket in 1..logits.len() {
                if logits[bucket] > logits[best] {
                    best = bucket;
                }
            }
            self.buckets.push(best);
        }
        check_process_stopped(stopped)?;
        Ok(&self.buckets)
    }
}

#[cfg(test)]
mod build_router_tests {
    use super::*;
    use crate::types::Distance;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn build_top1_matches_query_order_and_reuses_scratch() {
        let stopped = AtomicBool::new(false);
        let mut rng = rand::rngs::StdRng::seed_from_u64(43);
        for buckets in [1, 2, 64, 316] {
            let router = MlpRouter {
                layers: vec![
                    RouterLayer::Linear(LinearLayer {
                        in_features: 7,
                        out_features: 11,
                        weights: (0..77).map(|_| rng.random_range(-1.0..1.0)).collect(),
                        bias: vec![0.0; 11],
                    }),
                    RouterLayer::ReLU,
                    RouterLayer::Linear(LinearLayer {
                        in_features: 11,
                        out_features: buckets,
                        weights: (0..11 * buckets)
                            .map(|_| rng.random_range(-1.0..1.0))
                            .collect(),
                        bias: vec![0.0; buckets],
                    }),
                ],
            };
            let mut build = BuildRouter::new(&router, &stopped).unwrap();
            let capacities = (build.values.capacity(), build.output.capacity());
            for distance in [
                Distance::Cosine,
                Distance::Dot,
                Distance::Euclid,
                Distance::Manhattan,
            ] {
                for _ in 0..20 {
                    let row = distance.preprocess_vector::<f32>(
                        (0..7).map(|_| rng.random_range(-2.0..2.0)).collect(),
                    );
                    assert_eq!(
                        build.top_bucket(&row, &stopped).unwrap(),
                        router.top_buckets(&row, 1).unwrap()[0]
                    );
                }
            }
            assert_eq!(
                capacities,
                (build.values.capacity(), build.output.capacity())
            );
            assert!(build.top_bucket(&[], &stopped).is_err());
            assert!(build.top_bucket(&[f32::NAN; 7], &stopped).is_err());
        }
    }

    #[test]
    fn build_top1_ties_invalid_models_overflow_and_cancellation() {
        let stopped = AtomicBool::new(false);
        for bias in [vec![0.0, -0.0, 0.0], vec![1.0, 1.0, 0.0]] {
            let router = MlpRouter {
                layers: vec![RouterLayer::Linear(LinearLayer {
                    in_features: 1,
                    out_features: 3,
                    weights: vec![0.0; 3],
                    bias,
                })],
            };
            let mut build = BuildRouter::new(&router, &stopped).unwrap();
            assert_eq!(build.top_bucket(&[0.0], &stopped).unwrap(), 0);
            assert_eq!(
                build.top_bucket(&[0.0], &stopped).unwrap(),
                router.top_buckets(&[0.0], 1).unwrap()[0]
            );
            assert!(build.top_bucket(&[0.0], &AtomicBool::new(true)).is_err());
        }
        for router in [
            MlpRouter { layers: vec![] },
            MlpRouter {
                layers: vec![RouterLayer::ReLU],
            },
            MlpRouter {
                layers: vec![RouterLayer::Linear(LinearLayer {
                    in_features: 1,
                    out_features: 0,
                    weights: vec![],
                    bias: vec![],
                })],
            },
            MlpRouter {
                layers: vec![RouterLayer::Linear(LinearLayer {
                    in_features: 1,
                    out_features: 1,
                    weights: vec![f32::NAN],
                    bias: vec![0.0],
                })],
            },
        ] {
            assert!(BuildRouter::new(&router, &stopped).is_err());
        }
        let router = MlpRouter {
            layers: vec![RouterLayer::Linear(LinearLayer {
                in_features: 1,
                out_features: 1,
                weights: vec![f32::MAX],
                bias: vec![0.0],
            })],
        };
        assert!(BuildRouter::new(&router, &AtomicBool::new(true)).is_err());
        assert!(BuildRouter::new(&router, &stopped)
            .unwrap()
            .top_bucket(&[2.0], &stopped)
            .is_err());
    }
}

#[cfg(test)]
mod build_batch_router_tests {
    use super::*;
    use rand::{RngExt, SeedableRng};

    fn router(buckets: usize) -> MlpRouter {
        let mut rng = rand::rngs::StdRng::seed_from_u64(91 + buckets as u64);
        MlpRouter {
            layers: vec![
                RouterLayer::Linear(LinearLayer {
                    in_features: 3,
                    out_features: 5,
                    weights: (0..15).map(|_| rng.random_range(-1.0..1.0)).collect(),
                    bias: (0..5).map(|_| rng.random_range(-1.0..1.0)).collect(),
                }),
                RouterLayer::ReLU,
                RouterLayer::Linear(LinearLayer {
                    in_features: 5,
                    out_features: buckets,
                    weights: (0..5 * buckets)
                        .map(|_| rng.random_range(-1.0..1.0))
                        .collect(),
                    bias: (0..buckets).map(|_| rng.random_range(-1.0..1.0)).collect(),
                }),
            ],
        }
    }

    #[test]
    fn batch_top1_is_scalar_identical_for_full_partial_and_empty_batches() {
        let stopped = AtomicBool::new(false);
        for buckets in [1, 2, 7, 64] {
            let router = router(buckets);
            let mut rng = rand::rngs::StdRng::seed_from_u64(103 + buckets as u64);
            let rows: Vec<Vec<f32>> = (0..5)
                .map(|_| (0..3).map(|_| rng.random_range(-2.0..2.0)).collect())
                .collect();
            let expected: Vec<usize> = rows
                .iter()
                .map(|row| {
                    BuildRouter::new(&router, &stopped)
                        .unwrap()
                        .top_bucket(row, &stopped)
                        .unwrap()
                })
                .collect();
            for capacity in [1, 2, 3, 8] {
                let mut batch = BuildBatchRouter::new(&router, capacity, &stopped).unwrap();
                assert!(batch.top_buckets(&[], &stopped).unwrap().is_empty());
                let mut actual = Vec::new();
                for chunk in rows.chunks(capacity) {
                    let inputs: Vec<f32> = chunk.iter().flatten().copied().collect();
                    actual.extend_from_slice(batch.top_buckets(&inputs, &stopped).unwrap());
                }
                assert_eq!(actual, expected, "buckets={buckets} capacity={capacity}");
            }
        }
    }

    #[test]
    fn batch_router_rejects_bad_inputs_and_observes_cancellation() {
        let router = router(3);
        let stopped = AtomicBool::new(false);
        assert!(BuildBatchRouter::new(&router, 0, &stopped).is_err());
        let mut batch = BuildBatchRouter::new(&router, 2, &stopped).unwrap();
        assert!(batch.top_buckets(&[1.0, 2.0], &stopped).is_err());
        assert!(batch.top_buckets(&[f32::NAN, 0.0, 0.0], &stopped).is_err());
        assert!(batch.top_buckets(&[0.0; 9], &stopped).is_err());
        assert!(batch
            .top_buckets(&[0.0; 3], &AtomicBool::new(true))
            .is_err());
        assert!(batch.workspace_bytes() > 0);
    }
}
