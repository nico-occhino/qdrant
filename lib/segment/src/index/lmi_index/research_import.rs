//! Research only (`testing`): routers trained by another engine, imported for cross-engine replays (Q4 arm c).
//!
//! The format is PostgreSQL DLI's LMI1 export (`docs/research/Q4_ROUTER_EXPORT.md` in the PostgreSQL repository):
//! ASCII `LMI1`, then little-endian u32 d, h, b, then four contiguous row-major little-endian f32 arrays
//! W1[h,d], bias1[h], W2[b,h], bias2[b]; Linear, ReLU, Linear. That is exactly [`MlpRouter`]'s trained
//! architecture, with weights stored [out, in].
//!
//! A registered router is found by [`super::MlpRouter::fingerprint`] when a build names it in
//! [`super::LmiConfig::reuse_router`] and no build source holds it.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::{LinearLayer, MlpRouter, RouterLayer};
use crate::common::operation_error::{OperationError, OperationResult};

fn registry() -> &'static Mutex<HashMap<u64, MlpRouter>> {
    static REGISTRY: OnceLock<Mutex<HashMap<u64, MlpRouter>>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

/// Parses an LMI1 file. Rejects a wrong magic, a size that differs from the header, or non-finite values.
pub fn parse_lmi1(bytes: &[u8]) -> OperationResult<MlpRouter> {
    let bad = |what: &str| OperationError::service_error(format!("LMI1: {what}"));
    if bytes.len() < 16 || &bytes[..4] != b"LMI1" {
        return Err(bad("missing magic or header"));
    }
    let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
    let (d, h, b) = (u32_at(4), u32_at(8), u32_at(12));
    let floats = h * d + h + b * h + b;
    if d == 0 || h == 0 || b == 0 || bytes.len() != 16 + 4 * floats {
        return Err(bad("size does not match header"));
    }
    let values: Vec<f32> = bytes[16..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(bad("non-finite weight"));
    }
    let (w1, rest) = values.split_at(h * d);
    let (b1, rest) = rest.split_at(h);
    let (w2, b2) = rest.split_at(b * h);
    let router = MlpRouter {
        layers: vec![
            RouterLayer::Linear(LinearLayer { in_features: d, out_features: h, weights: w1.to_vec(), bias: b1.to_vec() }),
            RouterLayer::ReLU,
            RouterLayer::Linear(LinearLayer { in_features: h, out_features: b, weights: w2.to_vec(), bias: b2.to_vec() }),
        ],
    };
    router.validate()?;
    Ok(router)
}

/// Registers a router for reuse by fingerprint and returns the fingerprint.
pub fn register_router(router: MlpRouter) -> u64 {
    let fingerprint = router.fingerprint();
    registry().lock().unwrap().insert(fingerprint, router);
    fingerprint
}

pub(super) fn registered_router(fingerprint: u64) -> Option<MlpRouter> {
    registry().lock().unwrap().get(&fingerprint).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lmi1(d: u32, h: u32, b: u32, values: &[f32]) -> Vec<u8> {
        let mut out = b"LMI1".to_vec();
        for v in [d, h, b] {
            out.extend(v.to_le_bytes());
        }
        values.iter().for_each(|v| out.extend(v.to_le_bytes()));
        out
    }

    #[test]
    fn lmi1_round_trips_into_the_trained_architecture() {
        // d = 2, h = 2, b = 3. W1 = [[1, 0], [0, -1]], b1 = [0, 0.5], W2 = [[1, 1], [2, 0], [0, 3]], b2 = [0, 0, -1].
        let v = [1., 0., 0., -1., 0., 0.5, 1., 1., 2., 0., 0., 3., 0., 0., -1.];
        let router = parse_lmi1(&lmi1(2, 2, 3, &v)).unwrap();
        // x = [0.6, -0.8]: hidden = relu([0.6, 0.8 + 0.5]) = [0.6, 1.3]; logits = [1.9, 1.2, 2.9].
        let logits = router.forward(&[0.6, -0.8]).unwrap();
        for (got, want) in logits.iter().zip([1.9f32, 1.2, 2.9]) {
            assert!((got - want).abs() < 1e-6, "{logits:?}");
        }
        let fp = register_router(router.clone());
        assert_eq!(registered_router(fp), Some(router));
        assert!(parse_lmi1(&lmi1(2, 2, 3, &v[..14])).is_err(), "short payload");
        assert!(parse_lmi1(&lmi1(2, 2, 3, &[f32::NAN; 15])).is_err(), "non-finite");
        assert!(parse_lmi1(b"LMI0\0\0\0\0\0\0\0\0\0\0\0\0").is_err(), "magic");
    }
}
