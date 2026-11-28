//! Jakubík's Continually Learned Index (CLI), Stage A algorithm core: the pure rules of
//! `research/dli/cli/CLI_SHARED_SPEC.md`. Each function cites its spec section and thesis equation, and has a
//! known-answer test. Nothing here touches segments, persistence or training; those come in later stages.

use super::{LinearLayer, MlpRouter, RouterLayer};
use crate::common::operation_error::{OperationError, OperationResult};

/// Spec §2.1, thesis eq. 2.1: |B| = ⌊|D_init| / (0.5 · b_max)⌋, at least 2. Exact in integers: ⌊2n / b_max⌋.
/// (The code floors before dividing, audit A4; the thesis is followed.)
pub fn initial_buckets(n: usize, b_max: usize) -> usize {
    assert!(b_max > 0);
    (2 * n / b_max).max(2)
}

/// Spec §2.2, thesis eq. 2.2: |S| = ⌊0.1 · |B| · b_max⌋, capped at |D_init|. Exact: ⌊|B| · b_max / 10⌋.
pub fn training_sample(buckets: usize, b_max: usize, n: usize) -> usize {
    (buckets * b_max / 10).min(n)
}

/// Spec §4.2, thesis eq. 2.3: an overflowing bucket of n live points splits into k = ⌈n / b_max⌉. `None` if it does
/// not overflow (n ≤ b_max).
pub fn split_k(n: usize, b_max: usize) -> Option<usize> {
    (n > b_max).then(|| n.div_ceil(b_max))
}

/// Spec §4.2, thesis eq. 2.4: the split's k-means sample, min(n, 100 · k).
pub fn split_sample(n: usize, k: usize) -> usize {
    (100 * k).min(n)
}

/// Spec §4.3 step 2: k_new = ⌈N / (0.5 · b_max)⌉ = ⌈2N / b_max⌉, and its k-means sample min(N, ⌊0.1 · b_max · k_new⌋).
pub fn recluster_k(pooled: usize, b_max: usize) -> (usize, usize) {
    let k_new = (2 * pooled).div_ceil(b_max).max(1);
    (k_new, (b_max * k_new / 10).min(pooled))
}

/// Shannon entropy in bits of softmax(logits), computed stably (spec §4.1, thesis eq. 3.6).
pub fn entropy_bits(logits: &[f32]) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&z| (z as f64 - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    -exps
        .iter()
        .map(|&e| e / total)
        .filter(|&p| p > 0.0)
        .map(|p| p * p.log2())
        .sum::<f64>()
}

/// The maintenance path for one batch (spec §4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// No trigger: the splitting path (§4.2).
    None,
    Entropy,
    Growth,
    /// CLI-I only (spec §5): the frozen Q5 bucket-imbalance rule.
    Imbalance,
}

/// Which drift signal the arm uses: CLI's entropy rule, or CLI-I's imbalance rule (spec §5). Growth is common.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftSignal {
    Entropy,
    Imbalance,
}

/// Spec §4.1 and §5. `h_batch` and `h_init` are mean entropies in bits; `sizes` are live bucket sizes over all
/// allocated buckets. Entropy fires on H_batch > H_init + 0.3 (absolute term only, audit A7). Growth fires on
/// |B| > 10 · |B|_last. Imbalance fires on max · |B| · 1000 ≥ 8000 · live, in exact integers (the frozen Q5 rule).
/// When more than one fires, the drift signal is reported first, then growth.
pub fn trigger(signal: DriftSignal, h_batch: f64, h_init: f64, sizes: &[usize], buckets_last: usize) -> Trigger {
    let drift = match signal {
        DriftSignal::Entropy => h_batch > h_init + 0.3,
        DriftSignal::Imbalance => {
            let live: u128 = sizes.iter().map(|&s| s as u128).sum();
            let max = sizes.iter().copied().max().unwrap_or(0) as u128;
            live > 0 && max * sizes.len() as u128 * 1000 >= 8000 * live
        }
    };
    if drift {
        match signal {
            DriftSignal::Entropy => Trigger::Entropy,
            DriftSignal::Imbalance => Trigger::Imbalance,
        }
    } else if sizes.len() > 10 * buckets_last {
        Trigger::Growth
    } else {
        Trigger::None
    }
}

/// Python's `round` (half to even) on a non-negative value, as `cli_ablation` uses for the DEDRIFT fraction.
fn round_half_even(x: f64) -> usize {
    let floor = x.floor();
    let diff = x - floor;
    let n = floor as usize;
    if diff > 0.5 || (diff == 0.5 && n % 2 == 1) { n + 1 } else { n }
}

/// Spec §4.3 step 1, exactly as `DedriftReclusteringStrategy` (`reclustering.py:206–330`): the buckets to
/// recluster, ascending. Empty if nothing is reclustered (at most k non-empty buckets). Ties (spec v2 §0): the lower
/// stable id wins in both the largest and the smallest selection. `torch.sort` makes no tie promise, so G5 parity
/// with `cli_ablation` is claimed for distinct sizes only.
pub fn dedrift_select(sizes: &[usize], b_max: usize) -> Vec<usize> {
    let n = sizes.len();
    if n == 0 || sizes.iter().all(|&s| s == 0) {
        return Vec::new();
    }
    let k = round_half_even(0.5 * n as f64).clamp(1, n);
    if sizes.iter().filter(|&&s| s > 0).count() <= k {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    let largest: Vec<usize> = order[..k].to_vec();
    let l1: usize = largest.iter().map(|&b| sizes[b]).sum();
    let mut nonzero: Vec<usize> = sizes.iter().copied().filter(|&s| s > 0).collect();
    nonzero.sort_unstable();
    let mu = nonzero[(nonzero.len() - 1) / 2]; // torch.median: the lower middle value
    let k2 = l1.div_ceil(mu).clamp(k, n);
    let mut selected: std::collections::BTreeSet<usize> = largest.iter().copied().collect();
    let mut added = 0;
    let mut ascending: Vec<usize> = (0..n).collect();
    ascending.sort_by(|&a, &b| sizes[a].cmp(&sizes[b]).then(a.cmp(&b)));
    for &b in &ascending {
        if added >= k2 - k {
            break;
        }
        if largest.contains(&b) || sizes[b] == 0 {
            continue;
        }
        selected.insert(b);
        added += 1;
    }
    selected.extend((0..n).filter(|&b| sizes[b] > b_max));
    selected.into_iter().collect()
}

fn output_layer(router: &mut MlpRouter) -> OperationResult<&mut LinearLayer> {
    match router.layers.as_mut_slice() {
        [RouterLayer::Linear(_), RouterLayer::ReLU, RouterLayer::Linear(out)] => Ok(out),
        _ => Err(OperationError::service_error("CLI needs the Linear–ReLU–Linear router")),
    }
}

/// Spec §4.2 step 2, thesis Fig. 2.3: adds `copies` output neurons that copy `parent`'s weight row and bias. Returns
/// the new bucket ids (appended at the end, so existing ids keep their neurons).
pub fn grow_output(router: &mut MlpRouter, parent: usize, copies: usize) -> OperationResult<Vec<usize>> {
    let out = output_layer(router)?;
    if parent >= out.out_features {
        return Err(OperationError::service_error("CLI grow: parent bucket out of range"));
    }
    let width = out.in_features;
    let row = out.weights[parent * width..(parent + 1) * width].to_vec();
    let bias = out.bias[parent];
    let first = out.out_features;
    for _ in 0..copies {
        out.weights.extend_from_slice(&row);
        out.bias.push(bias);
    }
    out.out_features += copies;
    router.validate()?;
    Ok((first..first + copies).collect())
}

/// Spec §4.2 step 5: removes the output neurons of `remove` (empty buckets). Returns the old-id → new-id map, with
/// `None` for removed ids; surviving buckets keep their relative order.
pub fn shrink_output(router: &mut MlpRouter, remove: &[usize]) -> OperationResult<Vec<Option<usize>>> {
    let out = output_layer(router)?;
    let n = out.out_features;
    let gone: std::collections::HashSet<usize> = remove.iter().copied().collect();
    if gone.iter().any(|&b| b >= n) || gone.len() >= n {
        return Err(OperationError::service_error("CLI shrink: bad bucket set"));
    }
    let width = out.in_features;
    let mut map = vec![None; n];
    let (mut weights, mut bias) = (Vec::with_capacity((n - gone.len()) * width), Vec::new());
    for b in 0..n {
        if !gone.contains(&b) {
            map[b] = Some(bias.len());
            weights.extend_from_slice(&out.weights[b * width..(b + 1) * width]);
            bias.push(out.bias[b]);
        }
    }
    out.weights = weights;
    out.out_features = bias.len();
    out.bias = bias;
    router.validate()?;
    Ok(map)
}

/// Spec §7: L_route = 1 − share of untouched points whose argmax under the new router equals their stored bucket.
/// `stored[i]` is the stored bucket of untouched point i, and `argmax_new[i]` its argmax under the new router.
pub fn routing_accuracy_loss(stored: &[usize], argmax_new: &[usize]) -> f64 {
    assert_eq!(stored.len(), argmax_new.len());
    if stored.is_empty() {
        return 0.0;
    }
    let same = stored.iter().zip(argmax_new).filter(|(a, b)| a == b).count();
    1.0 - same as f64 / stored.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router(buckets: usize) -> MlpRouter {
        let (d, h) = (3, 2);
        MlpRouter {
            layers: vec![
                RouterLayer::Linear(LinearLayer { in_features: d, out_features: h, weights: vec![0.1; d * h], bias: vec![0.0; h] }),
                RouterLayer::ReLU,
                RouterLayer::Linear(LinearLayer {
                    in_features: h,
                    out_features: buckets,
                    weights: (0..buckets * h).map(|i| i as f32).collect(),
                    bias: (0..buckets).map(|b| b as f32 * 10.0).collect(),
                }),
            ],
        }
    }

    #[test]
    fn formulas_match_the_thesis_equations() {
        // Eq. 2.1 at the study example: N0 = 100,000, b_max = 1,024 -> 195 buckets; eq. 2.2 -> 19,968 samples.
        assert_eq!(initial_buckets(100_000, 1_024), 195);
        assert_eq!(training_sample(195, 1_024, 100_000), 19_968);
        // Audit A4: the code gives 4 at n = 30,000, b_max = 10,240; the thesis gives 5. n < b_max gives the floor of 2.
        assert_eq!(initial_buckets(30_000, 10_240), 5);
        assert_eq!(initial_buckets(100, 10_240), 2);
        assert_eq!(training_sample(2, 10_240, 100), 100, "capped at |D_init|");
        // Eq. 2.3/2.4: 2,300 points at b_max 1,024 -> 3 children, a 300-vector sample; no split at exactly b_max.
        assert_eq!(split_k(2_300, 1_024), Some(3));
        assert_eq!(split_sample(2_300, 3), 300);
        assert_eq!(split_k(1_024, 1_024), None);
        assert_eq!(split_k(1_025, 1_024), Some(2));
        assert_eq!(split_sample(150, 2), 150);
        // §4.3: 10,000 pooled at b_max 1,000 -> 20 clusters, sample min(10,000, 0.1 * 1,000 * 20 = 2,000).
        assert_eq!(recluster_k(10_000, 1_000), (20, 2_000));
        assert_eq!(recluster_k(1, 1_000), (1, 1));
    }

    #[test]
    fn entropy_and_triggers() {
        assert!((entropy_bits(&[0.0; 8]) - 3.0).abs() < 1e-12, "uniform over 8 = 3 bits");
        assert!(entropy_bits(&[100.0, 0.0, 0.0]) < 1e-30);
        assert!((entropy_bits(&[1.0, 1.0, f32::NEG_INFINITY]) - 1.0).abs() < 1e-12);
        let sizes = [100usize; 10];
        assert_eq!(trigger(DriftSignal::Entropy, 2.31, 2.0, &sizes, 10), Trigger::Entropy);
        assert_eq!(trigger(DriftSignal::Entropy, 2.3, 2.0, &sizes, 10), Trigger::None, "strictly greater");
        assert_eq!(trigger(DriftSignal::Entropy, 0.0, 2.0, &[1; 101], 10), Trigger::Growth);
        assert_eq!(trigger(DriftSignal::Entropy, 0.0, 2.0, &[1; 100], 10), Trigger::None, "growth is strict");
        // Imbalance: exactly 8.0x fires (>=), 7.9x does not; the mean is over all allocated buckets, empties included.
        let mut s = vec![0usize; 8];
        s[0] = 10;
        assert_eq!(trigger(DriftSignal::Imbalance, 9.0, 0.0, &s, 8), Trigger::Imbalance, "10*8 = 8*10");
        let mut s = vec![1usize; 80];
        s[0] = 8;
        assert_eq!(trigger(DriftSignal::Imbalance, 9.0, 0.0, &s, 80), Trigger::None, "8*80 < 8*87");
        assert_eq!(trigger(DriftSignal::Imbalance, 9.0, 0.0, &[0; 4], 4), Trigger::None, "empty index");
    }

    #[test]
    fn dedrift_selection_matches_cli_ablation() {
        // 6 buckets, b_max 100. k = round(3.0) = 3 largest: ids 2 (90), 0 (80), 4 (70), sum 240.
        // Non-empty sizes sorted [5, 10, 20, 70, 80, 90]: lower median 20. k2 = ceil(240 / 20) = 12 -> clamp 6.
        // B2 = 3 smallest non-empty not in B1: 3 (5), 1 (10), 5 (20).
        assert_eq!(dedrift_select(&[80, 10, 90, 5, 70, 20], 100), vec![0, 1, 2, 3, 4, 5]);
        // Larger median: sizes [50, 60, 70, 80, 300]: k = round(2.5) = 2 (half to even): ids 4 (300), 3 (80), 380.
        // Median of [50, 60, 70, 80, 300] = 70; k2 = ceil(380 / 70) = 6 -> clamp 5; B2 = 0 (50), 1 (60), 2 (70).
        assert_eq!(dedrift_select(&[50, 60, 70, 80, 300], 1_000), vec![0, 1, 2, 3, 4]);
        // k2 == k: no small buckets. sizes [100, 100, 1, 1]: k = 2, l1 = 200, lower median of [1, 1, 100, 100] = 1,
        // k2 = 200 -> clamp 4, B2 = 2 smallest = 3, 2. With b_max 50 the overflowing 0, 1 are already in B1.
        assert_eq!(dedrift_select(&[100, 100, 1, 1], 50), vec![0, 1, 2, 3]);
        // A real k2 < n case: sizes [10, 10, 10, 10, 10, 10, 10, 10, 10, 11]: k = 5,
        // B1 = 9 (11), 0, 1, 2, 3 (10 each); l1 = 51; median 10; k2 = ceil(5.1) = 6; B2 = one smallest not in B1.
        // Ties at size 10 among 4..8: the lower id, 4 (spec v2). cli_ablation's torch.sort happened to give 8; ties
        // are outside G5 parity.
        assert_eq!(dedrift_select(&[10, 10, 10, 10, 10, 10, 10, 10, 10, 11], 1_000), vec![0, 1, 2, 3, 4, 9]);
        // Odd B: k = round_half_even(3.5) = 4, not floor(3.5) = 3 (the PostgreSQL proposal the Qdrant side countered).
        // sizes 7 distinct: [70, 60, 50, 40, 30, 20, 10]: B1 = 0..3 (sum 220); lower median of 7 values = 40;
        // k2 = ceil(220 / 40) = 6; B2 = 2 smallest = 6 (10), 5 (20).
        assert_eq!(dedrift_select(&[70, 60, 50, 40, 30, 20, 10], 1_000), vec![0, 1, 2, 3, 5, 6]);
        // At most k non-empty buckets -> nothing is reclustered, even an overflowing one (the code returns first).
        assert_eq!(dedrift_select(&[0, 0, 0, 5], 1), Vec::<usize>::new());
        assert_eq!(dedrift_select(&[], 1), Vec::<usize>::new());
    }

    #[test]
    fn output_layer_grows_by_copy_and_shrinks_by_map() {
        let mut r = router(3);
        let new = grow_output(&mut r, 1, 2).unwrap();
        assert_eq!(new, vec![3, 4]);
        let x = [1.0f32, 2.0, 3.0];
        let z = r.forward(&x).unwrap();
        assert_eq!(z.len(), 5);
        assert_eq!(z[3], z[1]);
        assert_eq!(z[4], z[1], "copies produce the parent's logit");
        let before = z.clone();
        let map = shrink_output(&mut r, &[0, 3]).unwrap();
        assert_eq!(map, vec![None, Some(0), Some(1), None, Some(2)]);
        let after = r.forward(&x).unwrap();
        assert_eq!(after, vec![before[1], before[2], before[4]], "survivors keep their neurons");
        assert!(grow_output(&mut r, 9, 1).is_err());
        assert!(shrink_output(&mut r, &[0, 1, 2]).is_err(), "cannot remove every bucket");
        assert!((routing_accuracy_loss(&[1, 2, 3, 4], &[1, 2, 0, 4]) - 0.25).abs() < 1e-12);
        assert_eq!(routing_accuracy_loss(&[], &[]), 0.0);
    }
}
