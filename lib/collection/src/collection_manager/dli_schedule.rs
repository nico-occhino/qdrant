//! Arithmetic-only Bentley–Saxe planning for a temporal learned hierarchy.
//!
//! The buffer is not a learned level. Learned levels are one-based: level i
//! has capacity `buffer_capacity * arity^i` (Sommerová, §3.1–3.3).
//! This module does not read vectors, mutate native segments, train or publish.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MaintenancePolicy {
    #[default]
    Geometric,
    /// Flush independent runs until `arity` runs exist, then merge the buffer
    /// with the smallest run. Logical slots retain physical capacity checks.
    SizeTiered,
}

/// How many buckets a freshly trained learned level gets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LevelBuckets {
    /// Every level uses the collection model's `n_buckets` (original Qdrant DLI).
    #[default]
    Fixed,
    /// Level i gets arity^i buckets, so a full level has buckets of about the buffer
    /// size (Sommerová eq. 3.1–3.2).
    ArityPower,
}

/// Per-level model rule for new learned destinations. The default reproduces the
/// original behaviour: one model configuration, full retraining on every compaction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LevelModelPolicy {
    pub buckets: LevelBuckets,
    /// Teacher sample floor of this many samples per bucket (Faiss's 40·k rule);
    /// the collection model's `sample_size` stays the preferred size. 0 disables.
    pub min_samples_per_bucket: usize,
    /// Compaction into a non-empty target reuses the target's trained router instead
    /// of training (Sommerová's naive insert); an empty target is trained fresh.
    pub reuse_target_router: bool,
}

/// Largest sample the LMI configuration accepts.
const MAX_SAMPLE_SIZE: usize = 1_000_000;
/// Largest bucket count the LMI configuration accepts.
const MAX_BUCKETS: usize = 65_536;

impl LevelModelPolicy {
    /// Model for a learned destination at `level` that is trained from scratch.
    pub fn fresh_config(
        &self,
        base: segment::index::lmi_index::LmiConfig,
        arity: usize,
        level: usize,
    ) -> Result<segment::index::lmi_index::LmiConfig, ScheduleError> {
        let mut config = base;
        config.reuse_router = None;
        if self.buckets == LevelBuckets::ArityPower {
            let exponent = u32::try_from(level).map_err(|_| ScheduleError::CapacityOverflow)?;
            config.n_buckets = arity
                .checked_pow(exponent)
                .filter(|&n| n <= MAX_BUCKETS)
                .ok_or(ScheduleError::CapacityOverflow)?;
        }
        if self.min_samples_per_bucket > 0 {
            let floor = self
                .min_samples_per_bucket
                .checked_mul(config.n_buckets)
                .ok_or(ScheduleError::CapacityOverflow)?;
            config.sample_size = config.sample_size.max(floor).min(MAX_SAMPLE_SIZE);
        }
        config.sample_size = config.sample_size.max(config.n_buckets);
        config.nprobe = config.nprobe.clamp(1, config.n_buckets);
        Ok(config)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HierarchyConfig {
    pub buffer_capacity: usize,
    pub arity: usize,
    pub policy: MaintenancePolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    ZeroBufferCapacity,
    InvalidArity,
    CapacityOverflow,
    OccupancyOverflow,
    LevelOverCapacity {
        level: usize,
        occupied: usize,
        capacity: usize,
    },
}

/// One planned replacement. `source_levels` contains the one-based levels
/// merged with the buffer. The target's existing contents also belong in the
/// replacement. No caller may treat this arithmetic plan as published state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionPlan {
    pub target_level: usize,
    pub source_levels: Vec<usize>,
    pub buffer_points: usize,
    pub target_existing_points: usize,
    pub output_points: usize,
    pub target_capacity: usize,
}

impl HierarchyConfig {
    pub fn validate(self) -> Result<(), ScheduleError> {
        if self.buffer_capacity == 0 {
            return Err(ScheduleError::ZeroBufferCapacity);
        }
        if self.arity < 2 {
            return Err(ScheduleError::InvalidArity);
        }
        Ok(())
    }

    pub fn level_capacity(self, level: usize) -> Result<usize, ScheduleError> {
        self.validate()?;
        if level == 0 {
            return Ok(self.buffer_capacity);
        }
        let exponent = u32::try_from(level).map_err(|_| ScheduleError::CapacityOverflow)?;
        self.arity
            .checked_pow(exponent)
            .and_then(|factor| self.buffer_capacity.checked_mul(factor))
            .ok_or(ScheduleError::CapacityOverflow)
    }

    /// Plan a single complete buffer drain when its threshold is reached.
    /// `levels[0]` is learned level 1. Oversized input batches may skip empty
    /// intermediate levels; no vectors are modified by planning.
    pub fn plan(
        self,
        buffer_points: usize,
        levels: &[usize],
    ) -> Result<Option<CompactionPlan>, ScheduleError> {
        self.validate()?;
        for (idx, &occupied) in levels.iter().enumerate() {
            let level = idx.checked_add(1).ok_or(ScheduleError::CapacityOverflow)?;
            let capacity = self.level_capacity(level)?;
            if occupied > capacity {
                return Err(ScheduleError::LevelOverCapacity {
                    level,
                    occupied,
                    capacity,
                });
            }
        }
        if buffer_points < self.buffer_capacity {
            return Ok(None);
        }
        if self.policy == MaintenancePolicy::SizeTiered {
            let occupied: Vec<_> = levels.iter().enumerate().filter(|(_, n)| **n > 0).collect();
            let selected = if occupied.len() >= self.arity {
                occupied
                    .into_iter()
                    .min_by_key(|(i, n)| (**n, *i))
                    .map(|(i, n)| (i + 1, *n))
            } else {
                None
            };
            let output = buffer_points
                .checked_add(selected.map_or(0, |(_, n)| n))
                .ok_or(ScheduleError::OccupancyOverflow)?;
            let mut target = 1usize;
            loop {
                let capacity = self.level_capacity(target)?;
                let existing = levels.get(target - 1).copied().unwrap_or(0);
                if (existing == 0 || selected.is_some_and(|(level, _)| level == target))
                    && output <= capacity
                {
                    return Ok(Some(CompactionPlan {
                        target_level: target,
                        source_levels: selected
                            .into_iter()
                            .map(|(level, _)| level)
                            .filter(|level| *level != target)
                            .collect(),
                        buffer_points,
                        target_existing_points: existing,
                        output_points: output,
                        target_capacity: capacity,
                    }));
                }
                target = target
                    .checked_add(1)
                    .ok_or(ScheduleError::CapacityOverflow)?;
            }
        }
        let mut incoming = buffer_points;
        let mut target = 1usize;
        loop {
            let capacity = self.level_capacity(target)?;
            let occupied = levels.get(target - 1).copied().unwrap_or(0);
            if incoming <= capacity - occupied {
                let output_points = incoming
                    .checked_add(occupied)
                    .ok_or(ScheduleError::OccupancyOverflow)?;
                return Ok(Some(CompactionPlan {
                    target_level: target,
                    source_levels: (1..target).collect(),
                    buffer_points,
                    target_existing_points: occupied,
                    output_points,
                    target_capacity: capacity,
                }));
            }
            incoming = incoming
                .checked_add(occupied)
                .ok_or(ScheduleError::OccupancyOverflow)?;
            target = target
                .checked_add(1)
                .ok_or(ScheduleError::CapacityOverflow)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> HierarchyConfig {
        HierarchyConfig {
            policy: Default::default(),
            buffer_capacity: 2,
            arity: 2,
        }
    }

    // This applies only the count model. Native segment movement and publication
    // remain separate and require their own correctness and recovery tests.
    fn apply(levels: &mut Vec<usize>, p: &CompactionPlan) {
        levels.resize(levels.len().max(p.target_level), 0);
        for level in &p.source_levels {
            levels[level - 1] = 0;
        }
        levels[p.target_level - 1] = p.output_points;
    }

    #[test]
    fn level_model_policy_sizes_buckets_and_samples_per_level() {
        let base = segment::index::lmi_index::LmiConfig {
            n_buckets: 548,
            sample_size: 100_000,
            nprobe: 4,
            ..Default::default()
        };
        let legacy = LevelModelPolicy::default();
        assert_eq!(legacy.fresh_config(base, 10, 3).unwrap(), base);
        let emma = LevelModelPolicy {
            buckets: LevelBuckets::ArityPower,
            min_samples_per_bucket: 40,
            reuse_target_router: true,
        };
        let level1 = emma.fresh_config(base, 10, 1).unwrap();
        assert_eq!((level1.n_buckets, level1.sample_size, level1.nprobe), (10, 100_000, 4));
        let level2 = emma.fresh_config(base, 2, 1).unwrap();
        assert_eq!((level2.n_buckets, level2.nprobe), (2, 2));
        let level4 = emma.fresh_config(base, 10, 4).unwrap();
        assert_eq!((level4.n_buckets, level4.sample_size), (10_000, 400_000));
        let reused = segment::index::lmi_index::LmiConfig {
            reuse_router: Some(7),
            ..base
        };
        assert_eq!(emma.fresh_config(reused, 10, 1).unwrap().reuse_router, None);
        assert_eq!(emma.fresh_config(base, 10, 5), Err(ScheduleError::CapacityOverflow));
        let huge = LevelModelPolicy {
            min_samples_per_bucket: 1_000,
            ..emma
        };
        assert_eq!(huge.fresh_config(base, 10, 4).unwrap().sample_size, MAX_SAMPLE_SIZE);
    }

    #[test]
    fn size_tiered_preserves_population_and_selects_smallest_run() {
        let c = HierarchyConfig {
            buffer_capacity: 2,
            arity: 2,
            policy: MaintenancePolicy::SizeTiered,
        };
        let mut levels = Vec::new();
        for batch in 1..=12 {
            let p = c.plan(2, &levels).unwrap().unwrap();
            let sources: Vec<_> = p
                .source_levels
                .iter()
                .copied()
                .chain((p.target_existing_points > 0).then_some(p.target_level))
                .collect();
            assert!(sources.len() <= 1);
            if levels.iter().filter(|n| **n > 0).count() >= 2 {
                assert_eq!(sources.len(), 1);
                assert_eq!(
                    levels[sources[0] - 1],
                    *levels.iter().filter(|n| **n > 0).min().unwrap()
                );
            }
            apply(&mut levels, &p);
            assert_eq!(levels.iter().sum::<usize>(), batch * 2);
            for (i, n) in levels.iter().enumerate() {
                assert!(*n <= c.level_capacity(i + 1).unwrap());
            }
        }
    }

    #[test]
    fn repeated_batches_create_three_geometric_levels_without_losing_counts() {
        let c = config();
        let mut levels = Vec::new();
        assert_eq!(c.plan(1, &levels).unwrap(), None);
        for batch in 1..=15 {
            let plan = c.plan(2, &levels).unwrap().unwrap();
            apply(&mut levels, &plan);
            assert_eq!(levels.iter().sum::<usize>(), batch * 2);
            for (i, &n) in levels.iter().enumerate() {
                assert!(n <= c.level_capacity(i + 1).unwrap());
            }
        }
        assert!(levels.len() >= 3);
        assert_eq!(
            (
                c.level_capacity(1).unwrap(),
                c.level_capacity(2).unwrap(),
                c.level_capacity(3).unwrap()
            ),
            (4, 8, 16)
        );
    }

    #[test]
    fn targets_partial_levels_and_skips_empty_levels_for_large_batches() {
        let c = config();
        let p = c.plan(2, &[3, 1]).unwrap().unwrap();
        assert_eq!(
            (p.target_level, p.source_levels, p.output_points),
            (2, vec![1], 6)
        );
        let p = c.plan(9, &[]).unwrap().unwrap();
        assert_eq!(
            (
                p.target_level,
                p.source_levels,
                p.output_points,
                p.target_capacity
            ),
            (3, vec![1, 2], 9, 16)
        );
        let p = c.plan(2, &[4, 0, 5]).unwrap().unwrap();
        assert_eq!((p.target_level, p.output_points), (2, 6));
    }

    #[test]
    fn invalid_configuration_occupancy_and_overflow_are_rejected() {
        assert_eq!(
            HierarchyConfig {
                policy: Default::default(),
                buffer_capacity: 0,
                arity: 2
            }
            .plan(2, &[]),
            Err(ScheduleError::ZeroBufferCapacity)
        );
        assert_eq!(
            HierarchyConfig {
                policy: Default::default(),
                buffer_capacity: 2,
                arity: 1
            }
            .plan(2, &[]),
            Err(ScheduleError::InvalidArity)
        );
        assert!(matches!(
            config().plan(2, &[5]),
            Err(ScheduleError::LevelOverCapacity { level: 1, .. })
        ));
        assert_eq!(
            HierarchyConfig {
                policy: Default::default(),
                buffer_capacity: usize::MAX,
                arity: 2
            }
            .plan(usize::MAX, &[]),
            Err(ScheduleError::CapacityOverflow)
        );
        assert_eq!(
            config().level_capacity(usize::MAX),
            Err(ScheduleError::CapacityOverflow)
        );
    }
}
