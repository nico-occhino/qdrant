//! Checked admission accounting for native LMI builds, excluding Qdrant-owned storage.
use std::collections::BTreeMap;

use super::LmiConfig;
use crate::common::operation_error::{OperationError, OperationResult};

/// Retain the former component rejection envelope while introducing total accounting.
const COMPONENT_BUDGET_BYTES: u64 = 128_000_000;

fn overflow() -> OperationError {
    OperationError::service_error("LMI build plan arithmetic overflow")
}
fn product(values: &[u64]) -> OperationResult<u64> {
    values
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b).ok_or_else(overflow))
}
fn sum(values: &[u64]) -> OperationResult<u64> {
    values
        .iter()
        .try_fold(0u64, |a, &b| a.checked_add(b).ok_or_else(overflow))
}

#[derive(Debug, serde::Serialize)]
pub(super) struct BuildPlan {
    vectors: u64,
    effective_sample: u64,
    dimension: u64,
    buckets: u64,
    hidden: u64,
    batch: u64,
    routing_batch: u64,
    pub(super) bytes: BTreeMap<&'static str, u64>,
    explicit_bytes: u64,
    backend_allowance_bytes: u64,
    estimated_bytes: u64,
    component_ceiling_bytes: u64,
    requested_sample_bytes: u64,
    model_weight_bytes: u64,
    /// Coordinate-distance operations, including final teacher assignment.
    lloyd_coordinate_operations: u64,
    /// Approximate forward/backward dense MACs; excludes optimizer/activation work.
    training_macs: u64,
    corpus_two_pass_macs: u64,
}

impl BuildPlan {
    /// `vectors` may be a conservative physical-slot upper bound before scanning.
    /// Config validation belongs to the caller; this estimator also handles hypothetical plans.
    pub(super) fn estimate(
        vectors: usize,
        dim: usize,
        config: &LmiConfig,
    ) -> OperationResult<Self> {
        let n = u64::try_from(vectors).map_err(|_| overflow())?;
        let d = u64::try_from(dim).map_err(|_| overflow())?;
        let b = u64::try_from(config.n_buckets).map_err(|_| overflow())?;
        let h = u64::try_from(config.hidden_dim).map_err(|_| overflow())?;
        let requested = u64::try_from(config.sample_size).map_err(|_| overflow())?;
        let s = requested.min(n);
        let routing_k = u64::try_from(config.routing_batch_size)
            .map_err(|_| overflow())?
            .min(n);
        let k = u64::try_from(config.batch_size)
            .map_err(|_| overflow())?
            .min(s);
        if d == 0
            || b == 0
            || h == 0
            || requested == 0
            || config.batch_size == 0
            || config.routing_batch_size == 0
        {
            return Err(OperationError::service_error(
                "LMI build plan requires nonzero dimensions and configuration",
            ));
        }
        let weights = product(&[sum(&[d, b])?, h])?;
        let parameters = sum(&[weights, h, b])?;
        let mut bytes = BTreeMap::new();
        bytes.insert("sample_rust", product(&[4, s, d])?);
        bytes.insert("sample_torch", product(&[4, s, d])?);
        bytes.insert("sample_offsets_two_copies", product(&[8, s])?);
        bytes.insert("cluster_assignments_and_orders", product(&[32, s])?);
        bytes.insert("centroids_and_sums", product(&[16, b, d])?);
        bytes.insert("cluster_sizes", product(&[8, b])?);
        for name in [
            "parameters",
            "gradients",
            "adam_first",
            "adam_second",
            "router_export",
            "initialization_workspace",
        ] {
            bytes.insert(name, product(&[4, parameters])?);
        }
        bytes.insert("batch_input_and_gradient_allowance", product(&[8, k, d])?);
        bytes.insert(
            "hidden_activation_and_gradient_allowance",
            product(&[8, k, h])?,
        );
        bytes.insert("logits_loss_and_gradient_allowance", product(&[16, k, b])?);
        bytes.insert("batch_indices_and_labels", product(&[16, k])?);
        bytes.insert(
            "posting_counts_boundaries_cursors",
            sum(&[product(&[24, b])?, 8])?,
        );
        bytes.insert("postings", product(&[4, n])?);
        // Universal decoding may retain an encoded buffer alongside decoded state.
        bytes.insert(
            "encoded_postings_open_allowance",
            sum(&[product(&[4, n])?, product(&[8, sum(&[b, 1])?])?, 16])?,
        );
        bytes.insert(
            "encoded_router_open_allowance",
            sum(&[product(&[4, parameters])?, product(&[4, s])?, 256])?,
        );
        bytes.insert("open_validation_bits", sum(&[n, 7])? / 8);
        bytes.insert("routing_batch_gathered_input", product(&[4, routing_k, d])?);
        bytes.insert(
            "routing_batch_two_activation_buffers",
            product(&[8, routing_k, d.max(h).max(b)])?,
        );
        bytes.insert(
            "routing_batch_offsets_and_buckets",
            product(&[12, routing_k])?,
        );
        bytes.insert("inference_scratch", product(&[8, d.max(h).max(b)])?);
        let explicit_bytes = bytes
            .values()
            .try_fold(0u64, |a, &b| a.checked_add(b).ok_or_else(overflow))?;
        // Not a measured bound: opaque Torch/allocator workspaces vary by backend.
        let backend_allowance_bytes = sum(&[explicit_bytes / 4, 32 * 1024 * 1024])?;
        Ok(Self {
            vectors: n,
            effective_sample: s,
            dimension: d,
            buckets: b,
            hidden: h,
            batch: k,
            routing_batch: routing_k,
            bytes,
            explicit_bytes,
            backend_allowance_bytes,
            estimated_bytes: sum(&[explicit_bytes, backend_allowance_bytes])?,
            component_ceiling_bytes: COMPONENT_BUDGET_BYTES,
            requested_sample_bytes: product(&[4, requested, d])?,
            model_weight_bytes: product(&[4, weights])?,
            lloyd_coordinate_operations: product(&[
                sum(&[config.kmeans_iterations as u64, 1])?,
                s,
                b,
                d,
            ])?,
            training_macs: product(&[3, config.epochs as u64, s, weights])?,
            corpus_two_pass_macs: product(&[2, n, weights])?,
        })
    }

    pub(super) fn check_budget(&self) -> OperationResult<()> {
        if self.requested_sample_bytes > COMPONENT_BUDGET_BYTES
            || self.model_weight_bytes > COMPONENT_BUDGET_BYTES
        {
            return Err(OperationError::service_error(format!(
                "LMI build plan exceeds component budget {COMPONENT_BUDGET_BYTES} bytes: requested sample={} model weights={}; reduce sample_size/hidden_dim/buckets",
                self.requested_sample_bytes, self.model_weight_bytes
            )));
        }
        // Option C: aggregate bytes/work are reported, not admitted against an
        // invented global memory policy. CPU/IO permits do not reserve RAM.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiny_default_and_former_boundaries() {
        for n in [0, 1, 99780, 1_000_000] {
            let plan = BuildPlan::estimate(n, 768, &LmiConfig::default()).unwrap();
            plan.check_budget().unwrap();
            assert_eq!(plan.bytes["postings"], 4 * n as u64);
        }
        let mut config = LmiConfig {
            sample_size: 1_000_000,
            n_buckets: 1,
            hidden_dim: 1,
            ..LmiConfig::default()
        };
        config.nprobe = 1;
        BuildPlan::estimate(1_000_000, 32, &config)
            .unwrap()
            .check_budget()
            .unwrap();
        assert!(BuildPlan::estimate(1_000_000, 33, &config)
            .unwrap()
            .check_budget()
            .is_err());
        // Same requested-component envelope even when the segment is much smaller.
        assert!(BuildPlan::estimate(1, 33, &config)
            .unwrap()
            .check_budget()
            .is_err());
        config.sample_size = 1;
        config.hidden_dim = 4096;
        BuildPlan::estimate(1, 7811, &config)
            .unwrap()
            .check_budget()
            .unwrap();
        assert!(BuildPlan::estimate(1, 7812, &config)
            .unwrap()
            .check_budget()
            .is_err());
    }
    #[test]
    fn impossible_and_overflowing_plans_fail_before_allocation() {
        assert!(BuildPlan::estimate(usize::MAX, 768, &LmiConfig::default()).is_err());
        assert!(BuildPlan::estimate(1, usize::MAX, &LmiConfig::default()).is_err());
        assert!(BuildPlan::estimate(1, 0, &LmiConfig::default()).is_err());
        assert!(BuildPlan::estimate(
            1_000_000,
            768,
            &LmiConfig {
                sample_size: 1_000_000,
                ..LmiConfig::default()
            }
        )
        .unwrap()
        .check_budget()
        .is_err());
        let large = BuildPlan::estimate(1_000_000_000, 768, &LmiConfig::default()).unwrap();
        assert!(large.estimated_bytes > 8_000_000_000);
        // Reporting this estimate is not a guarantee that the host can build it.
        large.check_budget().unwrap();
    }
    #[test]
    fn report_scale_estimates_without_allocating_tensors() {
        for (name, n, d, s, b, h) in [
            ("baseline", 1_000_000, 128, 2048, 64, 64),
            ("sample_1m", 1_000_000, 768, 1_000_000, 64, 64),
            ("corpus_100m", 100_000_000, 768, 1_000_000, 10_000, 512),
            ("corpus_1b", 1_000_000_000, 768, 1_000_000, 31_622, 512),
        ] {
            let config = LmiConfig {
                sample_size: s,
                n_buckets: b,
                hidden_dim: h,
                ..LmiConfig::default()
            };
            let plan = BuildPlan::estimate(n, d, &config).unwrap();
            assert_eq!(plan.bytes["postings"], 4 * n as u64);
            println!("S3_PLAN {name} {}", serde_json::to_string(&plan).unwrap());
        }
    }
}
