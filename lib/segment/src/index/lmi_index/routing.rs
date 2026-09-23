use std::collections::HashSet;
use std::sync::atomic::AtomicBool;

use common::generic_consts::Random;
use common::types::PointOffsetType;

use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};
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

#[derive(Debug, Clone, PartialEq)]
pub struct LinearLayer {
    pub in_features: usize,
    pub out_features: usize,

    /// Row-major matrix:
    ///
    /// [out_features, in_features]
    pub weights: Vec<f32>,

    pub bias: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RouterLayer {
    Linear(LinearLayer),

    ReLU,
}

#[derive(Debug, Clone, PartialEq)]
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

    fn top_buckets_with_stop(
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
    postings: Vec<Vec<PointOffsetType>>,

    nprobe: usize,
}

impl LmiRoutingState {
    pub fn postings(&self) -> &[Vec<PointOffsetType>] {
        &self.postings
    }

    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    pub fn new(
        router: MlpRouter,
        postings: Vec<Vec<PointOffsetType>>,
        nprobe: usize,
    ) -> OperationResult<Self> {
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

        let mut seen = HashSet::new();

        for bucket in buckets {
            let Some(posting) = self.postings.get(bucket) else {
                return Err(OperationError::service_error(format!(
                    concat!("LMI router selected ", "missing bucket {}"),
                    bucket,
                )));
            };

            for &point_offset in posting {
                check_process_stopped(stopped)?;
                if seen.insert(point_offset) {
                    candidates.push(point_offset);
                }
            }
        }

        // Native top-k can keep a different equal-score boundary point when
        // arrival order differs. Ascending offsets match Plain full-scan order.
        candidates.sort_unstable();
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
