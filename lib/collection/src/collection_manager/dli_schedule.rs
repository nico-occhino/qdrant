//! Arithmetic-only Bentley–Saxe planning for a temporal learned hierarchy.
//!
//! The buffer is not a learned level. Learned levels are one-based: level i
//! has capacity `buffer_capacity * arity^i` (Sommerová, §3.1–3.3).
//! This module does not read vectors, mutate native segments, train or publish.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HierarchyConfig {
    pub buffer_capacity: usize,
    pub arity: usize,
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
                buffer_capacity: 0,
                arity: 2
            }
            .plan(2, &[]),
            Err(ScheduleError::ZeroBufferCapacity)
        );
        assert_eq!(
            HierarchyConfig {
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
