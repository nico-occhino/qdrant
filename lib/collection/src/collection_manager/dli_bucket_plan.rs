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
}
