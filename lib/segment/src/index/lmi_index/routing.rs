//! Native serving-only MLP routing copied from the final frozen implementation.
use super::CompactPostings;
use crate::common::operation_error::{OperationError, OperationResult, check_process_stopped};
use crate::data_types::vectors::{QueryVector, VectorInternal};
use common::types::PointOffsetType;
use std::sync::atomic::AtomicBool;

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

    pub fn top_buckets_with_stop(
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
    pub fn router(&self) -> &MlpRouter {
        &self.router
    }

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

    pub fn from_compact(
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
