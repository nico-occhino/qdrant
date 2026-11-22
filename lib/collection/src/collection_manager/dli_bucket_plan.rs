//! Global whole-bucket planning. Physical generation and posting validity must
//! be established by the caller's pinned native search view before scoring.
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Experimental routing choices; topology and native visibility semantics are unchanged.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RoutingPolicy {
    #[default]
    SlotWeighted,
    ReservedPerLevel,
    PopulationDensity,
    /// Explicit same-pipeline control for priority-cache timing.
    PopulationDensityUncached,
    /// Shared positive temperature fitted on validation queries, milli-units.
    TemperatureDensity {
        temperature_milli: u32,
    },
    /// Sommerová thesis eq. 3.4: P(bucket|q) times the member's bucket count, buffer 1.
    ModelDrivenNormalized,
    /// Her released code: the same score divided by the arity (0-based level exponent).
    ModelDrivenNormalizedCode,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct BucketPrediction {
    /// Buffer is level 0, learned levels use paper's one-based numbering.
    pub level: usize,
    pub physical: Uuid,
    /// `None` is a Plain buffer segment; learned buckets use `Some`.
    pub bucket: Option<usize>,
    pub probability: f32,
    pub occupancy: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    InvalidArity,
    ZeroBudget,
    InvalidPrediction,
    DuplicateBucket,
    NumericOverflow,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedBuckets {
    pub requested: usize,
    /// Sum of recorded bucket occupancies, not a count of visible scored points.
    pub estimated_candidates: usize,
    pub buckets: Vec<BucketPrediction>,
}

/// Paper-style heuristic P(bucket|q) * a^level, buffer priority 1.
/// This is a ranking heuristic, not probability calibration. The budget is a
/// minimum target and may be exceeded by the last whole bucket.
pub fn select(
    mut predictions: Vec<BucketPrediction>,
    arity: usize,
    budget: usize,
) -> Result<SelectedBuckets, PlanError> {
    if arity < 2 {
        return Err(PlanError::InvalidArity);
    }
    if budget == 0 {
        return Err(PlanError::ZeroBudget);
    }
    let mut seen = HashSet::new();
    let mut physical_levels = HashMap::new();
    let mut ranked = Vec::with_capacity(predictions.len());
    for p in predictions.drain(..) {
        if (p.level == 0) != p.bucket.is_none()
            || !p.probability.is_finite()
            || !(0.0..=1.0).contains(&p.probability)
        {
            return Err(PlanError::InvalidPrediction);
        }
        if physical_levels
            .insert(p.physical, p.level)
            .is_some_and(|previous| previous != p.level)
        {
            return Err(PlanError::InvalidPrediction);
        }
        if !seen.insert((p.physical, p.bucket)) {
            return Err(PlanError::DuplicateBucket);
        }
        if p.occupancy == 0 {
            continue;
        }
        let score = if p.level == 0 {
            1.0
        } else {
            let exponent = i32::try_from(p.level).map_err(|_| PlanError::NumericOverflow)?;
            let value = f64::from(p.probability) * (arity as f64).powi(exponent);
            if !value.is_finite() {
                return Err(PlanError::NumericOverflow);
            }
            value
        };
        ranked.push((score, p));
    }
    ranked.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.level.cmp(&b.1.level))
            .then(a.1.physical.cmp(&b.1.physical))
            .then(a.1.bucket.cmp(&b.1.bucket))
    });
    let mut selected = Vec::new();
    let mut count = 0usize;
    for (_, p) in ranked {
        if count >= budget {
            break;
        }
        count = count
            .checked_add(p.occupancy)
            .ok_or(PlanError::NumericOverflow)?;
        selected.push(p);
    }
    Ok(SelectedBuckets {
        requested: budget,
        estimated_candidates: count,
        buckets: selected,
    })
}

/// Population prior times predicted bucket relevance per unit posting work.
/// No parameters are fitted to test queries. Population comes from all nonempty
/// posting occupancies for this pinned generation, not logical slot capacity.
pub fn select_density(
    predictions: Vec<BucketPrediction>,
    arity: usize,
    budget: usize,
) -> Result<SelectedBuckets, PlanError> {
    select_density_impl::<true>(predictions, arity, budget)
}

pub fn select_density_uncached(
    predictions: Vec<BucketPrediction>,
    arity: usize,
    budget: usize,
) -> Result<SelectedBuckets, PlanError> {
    select_density_impl::<false>(predictions, arity, budget)
}

fn select_density_impl<const CACHE: bool>(
    predictions: Vec<BucketPrediction>,
    arity: usize,
    budget: usize,
) -> Result<SelectedBuckets, PlanError> {
    if budget == 0 {
        return Err(PlanError::ZeroBudget);
    }
    let ranked = select(predictions, arity, usize::MAX)?.buckets;
    let mut populations = HashMap::<Uuid, usize>::new();
    for p in &ranked {
        let count = populations.entry(p.physical).or_default();
        *count = count
            .checked_add(p.occupancy)
            .ok_or(PlanError::NumericOverflow)?;
    }
    let score = |p: &BucketPrediction| {
        if p.level == 0 {
            1.
        } else {
            f64::from(p.probability) * populations[&p.physical] as f64 / p.occupancy as f64
        }
    };
    // A priority is immutable for a pinned generation/query. Cache it once;
    // otherwise every sort comparison repeats a UUID hash lookup and division.
    let mut scored: Vec<_> = ranked
        .into_iter()
        .map(|p| (if CACHE { score(&p) } else { 0. }, p))
        .collect();
    scored.sort_by(|a, b| {
        let sa = if CACHE { a.0 } else { score(&a.1) };
        let sb = if CACHE { b.0 } else { score(&b.1) };
        sb.total_cmp(&sa)
            .then(a.1.level.cmp(&b.1.level))
            .then(a.1.physical.cmp(&b.1.physical))
            .then(a.1.bucket.cmp(&b.1.bucket))
    });
    let mut count = 0usize;
    let mut buckets = Vec::new();
    for (_, p) in scored {
        if count >= budget {
            break;
        }
        count = count
            .checked_add(p.occupancy)
            .ok_or(PlanError::NumericOverflow)?;
        buckets.push(p);
    }
    Ok(SelectedBuckets {
        requested: budget,
        estimated_candidates: count,
        buckets,
    })
}

/// Model-driven normalized ordering (Sommerová §3.5.1). A learned bucket scores
/// P(bucket|q)·B_m, where B_m is its member's bucket count (all predictions of that
/// member, empty buckets included), so a uniformly uncertain model scores 1 on every
/// level; `divide_by_arity` reproduces her code's `arity**level_idx` with a 0-based
/// index. The Plain buffer scores 1. Whole buckets are taken until `budget` is met.
pub fn select_model_normalized(
    predictions: Vec<BucketPrediction>,
    arity: usize,
    budget: usize,
    divide_by_arity: bool,
) -> Result<SelectedBuckets, PlanError> {
    if budget == 0 {
        return Err(PlanError::ZeroBudget);
    }
    let mut bucket_counts = HashMap::<Uuid, usize>::new();
    for p in &predictions {
        if p.level > 0 {
            *bucket_counts.entry(p.physical).or_default() += 1;
        }
    }
    // Reuse the baseline validation and its deterministic tie order.
    let ranked = select(predictions, arity, usize::MAX)?.buckets;
    let divisor = if divide_by_arity { arity as f64 } else { 1. };
    let mut scored: Vec<_> = ranked
        .into_iter()
        .map(|p| {
            let score = if p.level == 0 {
                1.
            } else {
                f64::from(p.probability) * bucket_counts[&p.physical] as f64 / divisor
            };
            (score, p)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.level.cmp(&b.1.level))
            .then(a.1.physical.cmp(&b.1.physical))
            .then(a.1.bucket.cmp(&b.1.bucket))
    });
    let mut count = 0usize;
    let mut buckets = Vec::new();
    for (_, p) in scored {
        if count >= budget {
            break;
        }
        count = count
            .checked_add(p.occupancy)
            .ok_or(PlanError::NumericOverflow)?;
        buckets.push(p);
    }
    Ok(SelectedBuckets {
        requested: budget,
        estimated_candidates: count,
        buckets,
    })
}

/// Reserve the highest-probability nonempty bucket of each learned physical
/// member, then fill remaining work in the baseline order. Whole buckets impose
/// an explicit cost floor; `budget` remains a minimum, never a hard work bound.
pub fn select_reserved(
    predictions: Vec<BucketPrediction>,
    arity: usize,
    budget: usize,
) -> Result<SelectedBuckets, PlanError> {
    if budget == 0 {
        return Err(PlanError::ZeroBudget);
    }
    // Reuse all baseline validation and deterministic ordering.
    let ranked = select(predictions, arity, usize::MAX)?.buckets;
    let mut covered = HashSet::new();
    let mut chosen = HashSet::new();
    let mut buckets = Vec::new();
    let mut count = 0usize;
    for p in &ranked {
        if p.level > 0 && covered.insert(p.physical) {
            count = count
                .checked_add(p.occupancy)
                .ok_or(PlanError::NumericOverflow)?;
            chosen.insert((p.physical, p.bucket));
            buckets.push(p.clone());
        }
    }
    for p in ranked {
        if count >= budget {
            break;
        }
        if chosen.insert((p.physical, p.bucket)) {
            count = count
                .checked_add(p.occupancy)
                .ok_or(PlanError::NumericOverflow)?;
            buckets.push(p);
        }
    }
    Ok(SelectedBuckets {
        requested: budget,
        estimated_candidates: count,
        buckets,
    })
}

/// Monotone (non-decreasing) step map fitted by pool-adjacent-violators: the value
/// at `x` is the mean of the block whose smallest input is the largest one <= `x`
/// (the first block below its range).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IsotonicMap {
    pub thresholds: Vec<f64>,
    pub values: Vec<f64>,
}

impl IsotonicMap {
    /// Least-squares isotonic regression of `y` on `x` (equal weights). Ties in `x`
    /// are pooled into one block. Empty input gives the constant map 0.
    pub fn fit(mut pairs: Vec<(f64, f64)>) -> Self {
        pairs.retain(|(x, y)| x.is_finite() && y.is_finite());
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        // (min x, sum y, count)
        let mut blocks: Vec<(f64, f64, f64)> = Vec::new();
        for (x, y) in pairs {
            match blocks.last_mut() {
                Some(last) if last.0 == x => {
                    last.1 += y;
                    last.2 += 1.;
                }
                _ => blocks.push((x, y, 1.)),
            }
            while blocks.len() >= 2 {
                let n = blocks.len();
                let (a, b) = (blocks[n - 2], blocks[n - 1]);
                if a.1 / a.2 <= b.1 / b.2 {
                    break;
                }
                blocks[n - 2] = (a.0, a.1 + b.1, a.2 + b.2);
                blocks.pop();
            }
        }
        if blocks.is_empty() {
            return Self { thresholds: vec![f64::NEG_INFINITY], values: vec![0.] };
        }
        Self {
            thresholds: blocks.iter().map(|b| b.0).collect(),
            values: blocks.iter().map(|b| b.1 / b.2).collect(),
        }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let idx = self.thresholds.partition_point(|t| *t <= x);
        self.values[idx.saturating_sub(1)]
    }
}

/// Per-level calibration for `select_calibrated` (Phase 3 Part B, arm E1L): for each
/// learned level, the expected number of the query's true top-k neighbours in a bucket
/// given its router probability; `plain_density` is the expected neighbours per
/// buffer point (k / N under exchangeability).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LevelCalibration {
    pub levels: std::collections::BTreeMap<usize, IsotonicMap>,
    pub plain_density: f64,
}

/// Calibrated level-aware allocation: buckets in decreasing expected neighbours per
/// unit of cost, g_level(P) / occupancy (the buffer: `plain_density`), the greedy
/// solution of the fractional knapsack with whole buckets, until `budget` is met.
/// Empty learned buckets cost and yield nothing and are skipped. A learned level
/// without a calibration map is rejected.
pub fn select_calibrated(
    predictions: Vec<BucketPrediction>,
    calibration: &LevelCalibration,
    budget: usize,
) -> Result<SelectedBuckets, PlanError> {
    if budget == 0 {
        return Err(PlanError::ZeroBudget);
    }
    if !calibration.plain_density.is_finite() || calibration.plain_density < 0. {
        return Err(PlanError::InvalidPrediction);
    }
    // Reuse the baseline validation and deterministic tie order.
    let ranked = select(predictions, 2, usize::MAX)?.buckets;
    let mut scored = Vec::with_capacity(ranked.len());
    for p in ranked {
        let score = if p.level == 0 {
            calibration.plain_density
        } else {
            if p.occupancy == 0 {
                continue;
            }
            let map = calibration.levels.get(&p.level).ok_or(PlanError::InvalidPrediction)?;
            map.eval(f64::from(p.probability)) / p.occupancy as f64
        };
        scored.push((score, p));
    }
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.level.cmp(&b.1.level))
            .then(a.1.physical.cmp(&b.1.physical))
            .then(a.1.bucket.cmp(&b.1.bucket))
    });
    let mut count = 0usize;
    let mut buckets = Vec::new();
    for (_, p) in scored {
        if count >= budget {
            break;
        }
        count = count.checked_add(p.occupancy).ok_or(PlanError::NumericOverflow)?;
        buckets.push(p);
    }
    Ok(SelectedBuckets { requested: budget, estimated_candidates: count, buckets })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn b(
        level: usize,
        id: u128,
        bucket: Option<usize>,
        probability: f32,
        occupancy: usize,
    ) -> BucketPrediction {
        BucketPrediction {
            level,
            physical: Uuid::from_u128(id),
            bucket,
            probability,
            occupancy,
        }
    }
    #[test]
    fn cached_and_uncached_density_keep_complete_order_and_work() {
        for seed in 1..21usize {
            let input: Vec<_> = (0..200)
                .map(|i| {
                    b(
                        i % 4 + 1,
                        (i % 4 + 1) as u128,
                        Some(i / 4),
                        ((i * seed % 99 + 1) as f32) / 100.,
                        i * seed % 1000 + 1,
                    )
                })
                .collect();
            for budget in [1, 100, 2000, usize::MAX] {
                assert_eq!(
                    select_density(input.clone(), 2, budget),
                    select_density_uncached(input.clone(), 2, budget)
                );
            }
        }
    }

    #[test]
    fn density_uses_physical_population_and_cost_not_logical_slot() {
        let a = vec![
            b(1, 1, Some(0), 0.5, 10),
            b(1, 1, Some(1), 0.5, 90),
            b(5, 2, Some(0), 0.9, 900),
            b(5, 2, Some(1), 0.1, 100),
        ];
        assert_eq!(
            select_density(a.clone(), 2, 1).unwrap().buckets[0].physical,
            Uuid::from_u128(1)
        );
        let mut shifted = a;
        for p in &mut shifted {
            p.level += 3;
        }
        assert_eq!(
            select_density(shifted, 2, 1).unwrap().buckets[0].physical,
            Uuid::from_u128(1)
        );
    }

    #[test]
    fn model_normalized_scales_by_member_bucket_count_and_code_variant_divides_by_arity() {
        // Level 1: 10 buckets; level 2: 100 buckets (arity 10). Uniform level-2 model
        // (0.01 each) scores 1.0, a confident level-1 bucket 0.5*10 = 5.0.
        let mut input = vec![b(0, 9, None, 0., 5)];
        for bucket in 0..10 {
            input.push(b(1, 1, Some(bucket), if bucket == 0 { 0.5 } else { 0.05 }, 3));
        }
        for bucket in 0..100 {
            input.push(b(2, 2, Some(bucket), 0.01, if bucket == 7 { 0 } else { 4 }));
        }
        let thesis = select_model_normalized(input.clone(), 10, 1, false).unwrap();
        assert_eq!((thesis.buckets[0].level, thesis.buckets[0].bucket), (1, Some(0)));
        // Thesis: second-best level-1 buckets score 0.5, below the buffer and the
        // uniform level 2 (both 1.0); the buffer wins the tie by level order.
        let thesis = select_model_normalized(input.clone(), 10, 9, false).unwrap();
        assert_eq!(thesis.buckets[1].level, 0);
        assert_eq!(thesis.buckets[2].level, 2);
        // Code variant: every learned score shrinks by the arity (confident bucket 0.5,
        // level 2 about 0.1), so the buffer (1.0) is visited first.
        let code = select_model_normalized(input.clone(), 10, 9, true).unwrap();
        assert_eq!(
            code.buckets.iter().map(|p| p.level).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        // Empty buckets are never selected and the budget is a minimum.
        let all = select_model_normalized(input, 10, usize::MAX, false).unwrap();
        assert_eq!(all.estimated_candidates, 5 + 30 + 99 * 4);
        assert!(all.buckets.iter().all(|p| p.occupancy > 0));
        assert_eq!(
            select_model_normalized(vec![], 10, 0, false),
            Err(PlanError::ZeroBudget)
        );
    }

    #[test]
    fn reserved_coverage_has_explicit_cost_floor_and_no_duplicates() {
        let input = vec![
            b(1, 1, Some(0), 0.8, 10),
            b(1, 1, Some(1), 0.2, 10),
            b(5, 2, Some(0), 0.9, 100),
            b(0, 3, None, 1., 3),
        ];
        let baseline = select(input.clone(), 2, 1).unwrap();
        assert_eq!(baseline.buckets.len(), 1);
        let reserved = select_reserved(input.clone(), 2, 1).unwrap();
        assert_eq!(reserved.estimated_candidates, 110);
        assert_eq!(reserved.buckets.len(), 2);
        let all = select_reserved(input, 2, 1000).unwrap();
        assert_eq!(all.buckets.len(), 4);
        assert_eq!(all.estimated_candidates, 123);
        assert_eq!(select_reserved(vec![], 2, 0), Err(PlanError::ZeroBudget));
    }

    #[test]
    fn globally_ranks_across_levels_and_buffer_with_whole_bucket_overshoot() {
        let result = select(
            vec![
                b(0, 1, None, 0.0, 2),
                b(1, 2, Some(0), 0.4, 4),
                b(2, 3, Some(1), 0.3, 3),
            ],
            3,
            5,
        )
        .unwrap();
        assert_eq!(result.estimated_candidates, 7);
        assert_eq!(
            result.buckets.iter().map(|x| x.level).collect::<Vec<_>>(),
            vec![2, 1]
        );
    }
    #[test]
    fn stable_ties_empty_buckets_and_insufficient_candidates() {
        let result = select(
            vec![
                b(1, 3, Some(1), 0.5, 1),
                b(0, 1, None, 0.0, 2),
                b(1, 2, Some(0), 0.5, 0),
            ],
            2,
            10,
        )
        .unwrap();
        assert_eq!(result.estimated_candidates, 3);
        assert_eq!(
            result.buckets.iter().map(|x| x.level).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }
    #[test]
    fn rejects_malformed_duplicate_and_overflow_inputs() {
        assert_eq!(select(vec![], 2, 0), Err(PlanError::ZeroBudget));
        assert_eq!(select(vec![], 1, 1), Err(PlanError::InvalidArity));
        assert_eq!(
            select(vec![b(1, 1, Some(0), f32::NAN, 1)], 2, 1),
            Err(PlanError::InvalidPrediction)
        );
        assert_eq!(
            select(vec![b(0, 1, Some(0), 0.0, 1)], 2, 1),
            Err(PlanError::InvalidPrediction)
        );
        assert_eq!(
            select(
                vec![b(1, 1, Some(0), 0.5, 1), b(2, 1, Some(0), 0.5, 1)],
                2,
                1
            ),
            Err(PlanError::InvalidPrediction)
        );
        assert_eq!(
            select(
                vec![b(1, 1, Some(0), 0.5, 1), b(1, 1, Some(0), 0.5, 1)],
                2,
                1
            ),
            Err(PlanError::DuplicateBucket)
        );
        assert_eq!(
            select(vec![b(usize::MAX, 1, Some(0), 0.5, 1)], 2, 1),
            Err(PlanError::NumericOverflow)
        );
    }

    #[test]
    fn isotonic_fit_matches_hand_computed_pav() {
        // (0.1,0) (0.2,2) (0.3,1) (0.4,3): the violator pair 2,1 pools to 1.5.
        let map = IsotonicMap::fit(vec![(0.3, 1.), (0.1, 0.), (0.4, 3.), (0.2, 2.)]);
        assert_eq!(map.thresholds, vec![0.1, 0.2, 0.4]);
        assert_eq!(map.values, vec![0., 1.5, 3.]);
        assert_eq!(map.eval(0.05), 0.);
        assert_eq!(map.eval(0.25), 1.5);
        assert_eq!(map.eval(0.3999), 1.5);
        assert_eq!(map.eval(0.9), 3.);
        // Ties in x pool; a fully decreasing sequence pools to one block (its mean).
        let tie = IsotonicMap::fit(vec![(0.5, 1.), (0.5, 3.), (0.6, 0.)]);
        assert_eq!((tie.thresholds, tie.values), (vec![0.5], vec![4. / 3.]));
        assert_eq!(IsotonicMap::fit(vec![]).eval(0.3), 0.);
        // Output is non-decreasing on random input.
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        let pairs: Vec<_> = (0..500).map(|_| (next(), next() * 3.)).collect();
        let map = IsotonicMap::fit(pairs);
        assert!(map.values.windows(2).all(|w| w[0] <= w[1]));
        assert!(map.thresholds.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn calibrated_selection_is_greedy_value_per_cost_with_whole_buckets() {
        let (l1, l2, plain) = (Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3));
        let p = |level, physical, bucket, probability, occupancy| BucketPrediction {
            level, physical, bucket, probability, occupancy,
        };
        let calibration = LevelCalibration {
            // Level 1: value = 10·P (as a step map on these inputs); level 2: value = 2 flat.
            levels: [
                (1, IsotonicMap { thresholds: vec![0., 0.5, 0.9], values: vec![1., 5., 9.] }),
                (2, IsotonicMap { thresholds: vec![0.], values: vec![2.] }),
            ]
            .into_iter()
            .collect(),
            plain_density: 0.01,
        };
        let predictions = vec![
            p(1, l1, Some(0), 0.95, 100), // 9 / 100 = 0.09
            p(1, l1, Some(1), 0.6, 10),   // 5 / 10  = 0.5
            p(1, l1, Some(2), 0.1, 0),    // empty: skipped
            p(2, l2, Some(0), 0.2, 4),    // 2 / 4   = 0.5 (tie: lower level first)
            p(2, l2, Some(1), 0.1, 40),   // 2 / 40  = 0.05
            p(0, plain, None, 0., 50),    // 0.01
        ];
        let order = |budget| {
            select_calibrated(predictions.clone(), &calibration, budget)
                .unwrap()
                .buckets
                .iter()
                .map(|b| (b.level, b.bucket))
                .collect::<Vec<_>>()
        };
        let all = order(usize::MAX / 2);
        assert_eq!(all, vec![(1, Some(1)), (2, Some(0)), (1, Some(0)), (2, Some(1)), (0, None)]);
        // Reference implementation: sort by value/cost descending (same tie rule).
        let mut reference: Vec<(f64, usize, Option<usize>)> = vec![
            (9. / 100., 1, Some(0)), (0.5, 1, Some(1)), (0.5, 2, Some(0)), (0.05, 2, Some(1)), (0.01, 0, None),
        ];
        reference.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        assert_eq!(all, reference.iter().map(|r| (r.1, r.2)).collect::<Vec<_>>());
        // Budget is a minimum met by whole buckets: 10 < 12 so a second bucket is taken.
        assert_eq!(order(12), vec![(1, Some(1)), (2, Some(0))]);
        assert_eq!(order(10), vec![(1, Some(1))]);
        // Missing calibration for a learned level is rejected; zero budget too.
        let partial = LevelCalibration { levels: calibration.levels.iter().take(1).map(|(k, v)| (*k, v.clone())).collect(), plain_density: 0.01 };
        assert_eq!(select_calibrated(predictions.clone(), &partial, 5), Err(PlanError::InvalidPrediction));
        assert_eq!(select_calibrated(predictions, &calibration, 0), Err(PlanError::ZeroBudget));
    }
}
