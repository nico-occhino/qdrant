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
    /// Bentley–Saxe, except that a level above the deepest non-empty level D may
    /// hold at most `rho_permille / 1000 · live(D)` points. A compaction that would
    /// exceed it merges the buffer and every level above D into D instead, if the
    /// result fits D's capacity. Phase 3 Part B arm E1C (`MULTILEVEL_DIAGNOSIS.md`).
    GeometricConsolidating { rho_permille: u16 },
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
    /// Exception to `reuse_target_router`: retrain an occupied destination whose bucket
    /// census is too imbalanced. `None` (the default) is plain reuse; it is not
    /// serialized, so existing configurations keep their bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub imbalance_retrain: Option<ImbalanceRetrain>,
}

/// Imbalance-triggered retraining, ported from the PostgreSQL rule frozen in
/// `PREREGISTRATION_DLI_IMBALANCE_TRIGGER.md` (research/dli/uniform/Q5). At a
/// compaction into an occupied destination at level `min_level` or deeper, retrain if
/// the destination's largest live bucket divided by the mean live bucket size over
/// all allocated buckets is at least `ratio_permille / 1000`; otherwise reuse.
/// Shallower occupied levels always reuse; empty destinations always train.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImbalanceRetrain {
    pub min_level: usize,
    pub ratio_permille: u32,
}

impl ImbalanceRetrain {
    /// PostgreSQL's frozen rule: level 2 or deeper, threshold 8.0.
    pub const POSTGRES_FROZEN: Self = Self {
        min_level: 2,
        ratio_permille: 8_000,
    };

    /// Whether an occupied destination at `level` with live bucket sizes `sizes`
    /// (one entry per allocated bucket, empty buckets included) is retrained. The
    /// comparison max / (total / buckets) >= ratio / 1000 is done exactly in integers.
    pub fn retrain(&self, level: usize, sizes: &[usize]) -> bool {
        let total: u128 = sizes.iter().map(|&s| s as u128).sum();
        if level < self.min_level || sizes.is_empty() || total == 0 {
            return false;
        }
        let max = sizes.iter().copied().max().unwrap_or(0) as u128;
        max * sizes.len() as u128 * 1_000 >= u128::from(self.ratio_permille) * total
    }
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
                if let MaintenancePolicy::GeometricConsolidating { rho_permille } = self.policy
                    && let Some(plan) = self.consolidation(
                        buffer_points,
                        levels,
                        target,
                        output_points,
                        rho_permille,
                    )?
                {
                    return Ok(Some(plan));
                }
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

impl HierarchyConfig {
    /// Consolidation override for a Bentley–Saxe plan into `target` with
    /// `output_points`. `None` keeps the Bentley–Saxe plan.
    fn consolidation(
        self,
        buffer_points: usize,
        levels: &[usize],
        target: usize,
        output_points: usize,
        rho_permille: u16,
    ) -> Result<Option<CompactionPlan>, ScheduleError> {
        let Some(deepest) = levels.iter().rposition(|&n| n > 0).map(|i| i + 1) else {
            return Ok(None);
        };
        if target >= deepest {
            return Ok(None);
        }
        let deepest_points = levels[deepest - 1];
        // output > rho · live(D), in integers: output · 1000 > rho_permille · live(D).
        let lhs = (output_points as u128) * 1000;
        let rhs = u128::from(rho_permille) * deepest_points as u128;
        if lhs <= rhs {
            return Ok(None);
        }
        let above: usize = levels[..deepest - 1].iter().sum();
        let merged = buffer_points
            .checked_add(above)
            .and_then(|n| n.checked_add(deepest_points))
            .ok_or(ScheduleError::OccupancyOverflow)?;
        let capacity = self.level_capacity(deepest)?;
        if merged > capacity {
            return Ok(None);
        }
        Ok(Some(CompactionPlan {
            target_level: deepest,
            source_levels: (1..deepest).collect(),
            buffer_points,
            target_existing_points: deepest_points,
            output_points: merged,
            target_capacity: capacity,
        }))
    }

    /// Minimum capacity of learned level `level`: Cmin(i) = b·a^(i-1) (Sommerová eq. 3.3).
    pub fn minimum_capacity(self, level: usize) -> Result<usize, ScheduleError> {
        if level == 0 {
            return Ok(0);
        }
        self.level_capacity(level - 1)
    }

    /// Vacuum rebuild (Phase 3 arm E1V, Qdrant-specific): Qdrant deletes are soft, so a
    /// level keeps its deleted points, and their postings keep consuming routing budget,
    /// until the level is rewritten. The deepest learned level whose dead fraction
    /// (physical − live) / physical exceeds `dead_permille / 1000` is rewritten in place
    /// (target = that level, no sources, buffer untouched), which drops its dead points;
    /// the target's router is reused. Mirrors Qdrant's vacuum optimizer
    /// (`deleted_threshold`, default 0.2).
    pub fn plan_vacuum(
        self,
        live: &[usize],
        physical: &[usize],
        dead_permille: u16,
    ) -> Result<Option<CompactionPlan>, ScheduleError> {
        self.validate()?;
        if live.len() != physical.len() {
            return Err(ScheduleError::OccupancyOverflow);
        }
        for level in (1..=live.len()).rev() {
            let (n, p) = (live[level - 1], physical[level - 1]);
            if n > p {
                return Err(ScheduleError::OccupancyOverflow);
            }
            if p == 0 || n == 0 {
                continue;
            }
            // (p − n) / p > dead_permille / 1000, in integers.
            if ((p - n) as u128) * 1000 > u128::from(dead_permille) * p as u128 {
                return Ok(Some(CompactionPlan {
                    target_level: level,
                    source_levels: Vec::new(),
                    buffer_points: 0,
                    target_existing_points: n,
                    output_points: n,
                    target_capacity: self.level_capacity(level)?,
                }));
            }
        }
        Ok(None)
    }

    /// Greedy underutilisation rebuild (Sommerová §3.4.2, Algorithm 11), planned from
    /// **live** occupancies after deletes. `levels[0]` is learned level 1. The deepest
    /// underutilised level is handled first, since it holds the oldest data. Level 1 is
    /// never rebuilt into the buffer: her third rule would target level 0, which is not a
    /// learned level here. Returns `None` when no level needs a rebuild.
    ///
    /// Target preference, first match wins:
    /// 1. the nearest non-empty deeper level whose free capacity exceeds live(i);
    /// 2. level i itself, if some non-empty level lies above it;
    /// 3. level i−1 (empty by then; it holds live(i) < Cmin(i) = Cmax(i−1)).
    ///
    /// Greedy then adds, moving upward from the target, every contiguous level that
    /// still fits the remaining capacity; it stops at the first one that does not, so
    /// temporal order is preserved. Empty levels are passed over.
    pub fn plan_rebuild(self, levels: &[usize]) -> Result<Option<CompactionPlan>, ScheduleError> {
        self.validate()?;
        for (idx, &occupied) in levels.iter().enumerate() {
            let capacity = self.level_capacity(idx + 1)?;
            if occupied > capacity {
                return Err(ScheduleError::LevelOverCapacity {
                    level: idx + 1,
                    occupied,
                    capacity,
                });
            }
        }
        let live = |level: usize| levels.get(level - 1).copied().unwrap_or(0);
        let mut underutilised = None;
        for level in (1..=levels.len()).rev() {
            let n = live(level);
            if n > 0 && n < self.minimum_capacity(level)? {
                underutilised = Some(level);
                break;
            }
        }
        let Some(level) = underutilised else {
            return Ok(None);
        };
        let deeper = (level + 1..=levels.len()).find(|&j| live(j) > 0);
        let above_non_empty = (1..level).any(|j| live(j) > 0);
        let target = match deeper {
            Some(j) if self.level_capacity(j)? - live(j) > live(level) => j,
            _ if above_non_empty => level,
            _ if level >= 2 => level - 1,
            _ => return Ok(None),
        };
        let capacity = self.level_capacity(target)?;
        let mut output = live(target);
        let mut sources = Vec::new();
        // Levels to absorb, from just above the target upward; when the target is
        // level − 1 (rule 3) the only source is `level` itself.
        let candidates: Vec<usize> = if target < level {
            vec![level]
        } else {
            (1..target).rev().collect()
        };
        for source in candidates {
            let n = live(source);
            if n == 0 {
                continue;
            }
            match output.checked_add(n) {
                Some(total) if total <= capacity => {
                    output = total;
                    sources.push(source);
                }
                _ => break,
            }
        }
        if !sources.contains(&level) && target != level {
            // The underutilised level must move, or the rebuild achieves nothing.
            return Ok(None);
        }
        if sources.is_empty() {
            return Ok(None);
        }
        sources.sort_unstable();
        Ok(Some(CompactionPlan {
            target_level: target,
            source_levels: sources,
            buffer_points: 0,
            target_existing_points: live(target),
            output_points: output,
            target_capacity: capacity,
        }))
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
    fn imbalance_retrain_matches_the_frozen_postgres_rule() {
        let rule = ImbalanceRetrain::POSTGRES_FROZEN;
        // 10 buckets, total 100, mean 10: a largest bucket of 80 is exactly 8.0x.
        let exact = [80usize, 20, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(rule.retrain(2, &exact), "8.0x retrains (>=)");
        let below = [79usize, 21, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(!rule.retrain(2, &below), "7.9x reuses");
        // Empty buckets count in the mean (all allocated buckets): 10 / (20 / 10) = 5.0x.
        let sparse = [10usize, 10, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(!rule.retrain(2, &sparse));
        assert!(!rule.retrain(1, &exact), "level 1 always reuses");
        assert!(rule.retrain(3, &exact), "deeper levels are eligible");
        assert!(!rule.retrain(2, &[0, 0, 0]), "an empty census never retrains");
        assert!(!rule.retrain(2, &[]));
    }

    #[test]
    fn imbalance_retrain_off_keeps_the_serialized_policy() {
        let e1 = LevelModelPolicy {
            buckets: LevelBuckets::ArityPower,
            min_samples_per_bucket: 40,
            reuse_target_router: true,
            imbalance_retrain: None,
        };
        let text = serde_json::to_string(&e1).unwrap();
        assert_eq!(
            text,
            r#"{"buckets":"ArityPower","min_samples_per_bucket":40,"reuse_target_router":true}"#
        );
        assert_eq!(serde_json::from_str::<LevelModelPolicy>(&text).unwrap(), e1);
        let trigger = LevelModelPolicy {
            imbalance_retrain: Some(ImbalanceRetrain::POSTGRES_FROZEN),
            ..e1
        };
        let text = serde_json::to_string(&trigger).unwrap();
        assert_eq!(serde_json::from_str::<LevelModelPolicy>(&text).unwrap(), trigger);
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
            imbalance_retrain: None,
        };
        let level1 = emma.fresh_config(base, 10, 1).unwrap();
        assert_eq!(
            (level1.n_buckets, level1.sample_size, level1.nprobe),
            (10, 100_000, 4)
        );
        let level2 = emma.fresh_config(base, 2, 1).unwrap();
        assert_eq!((level2.n_buckets, level2.nprobe), (2, 2));
        let level4 = emma.fresh_config(base, 10, 4).unwrap();
        assert_eq!((level4.n_buckets, level4.sample_size), (10_000, 400_000));
        let reused = segment::index::lmi_index::LmiConfig {
            reuse_router: Some(7),
            ..base
        };
        assert_eq!(emma.fresh_config(reused, 10, 1).unwrap().reuse_router, None);
        assert_eq!(
            emma.fresh_config(base, 10, 5),
            Err(ScheduleError::CapacityOverflow)
        );
        let huge = LevelModelPolicy {
            min_samples_per_bucket: 1_000,
            ..emma
        };
        assert_eq!(
            huge.fresh_config(base, 10, 4).unwrap().sample_size,
            MAX_SAMPLE_SIZE
        );
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

    fn consolidating(buffer_capacity: usize, arity: usize, rho_permille: u16) -> HierarchyConfig {
        HierarchyConfig {
            policy: MaintenancePolicy::GeometricConsolidating { rho_permille },
            buffer_capacity,
            arity,
        }
    }

    /// Replays whole buffer drains; returns final levels and points written.
    fn ingest(config: HierarchyConfig, population: usize) -> (Vec<usize>, usize) {
        let (mut levels, mut written, mut n) = (Vec::new(), 0usize, 0usize);
        while n + config.buffer_capacity <= population {
            n += config.buffer_capacity;
            let plan = config
                .plan(config.buffer_capacity, &levels)
                .unwrap()
                .unwrap();
            assert!(plan.output_points <= plan.target_capacity);
            apply(&mut levels, &plan);
            written += plan.output_points;
            assert_eq!(levels.iter().sum::<usize>(), n, "population conserved");
        }
        while levels.last() == Some(&0) {
            levels.pop();
        }
        (levels, written)
    }

    #[test]
    fn consolidation_reproduces_the_planner_simulation() {
        // Final levels printed by research/dli/emma_path/consolidation_sim.py.
        let plain = HierarchyConfig {
            policy: MaintenancePolicy::Geometric,
            buffer_capacity: 5_000,
            arity: 10,
        };
        assert_eq!(
            ingest(plain, 10_120_191).0,
            vec![0, 220_000, 4_400_000, 5_500_000]
        );
        let (levels, written) = ingest(consolidating(5_000, 10, 250), 10_120_191);
        assert_eq!(levels, vec![40_000, 0, 1_100_000, 8_980_000]);
        assert_eq!((written as f64 / 10_120_191.0 * 10.0).round() / 10.0, 14.2);
        assert_eq!(
            ingest(consolidating(5_000, 10, 500), 10_120_191).0,
            vec![35_000, 55_000, 2_200_000, 7_830_000]
        );
        // rho = 1000 can only trigger when a level would outgrow the deepest one.
        assert_eq!(
            ingest(consolidating(5_000, 10, 1000), 10_120_191).0,
            vec![0, 220_000, 4_400_000, 5_500_000]
        );
    }

    #[test]
    fn consolidation_boundary_fit_guard_and_shape() {
        let c = consolidating(10, 10, 250);
        // Bentley–Saxe would put 10 + 30 = 40 into level 1; deepest is level 3 with 160.
        // 40 > 0.25 · 160 = 40 is false: the boundary keeps Bentley–Saxe.
        let p = c.plan(10, &[30, 0, 160]).unwrap().unwrap();
        assert_eq!((p.target_level, p.output_points), (1, 40));
        // One more point above the boundary consolidates everything into level 3.
        let p = c.plan(10, &[31, 0, 160]).unwrap().unwrap();
        assert_eq!(p.target_level, 3);
        assert_eq!(p.source_levels, vec![1, 2]);
        assert_eq!((p.target_existing_points, p.output_points), (160, 201));
        // Fit guard: level 3 capacity is 10_000; a nearly full deepest level keeps Bentley–Saxe.
        let p = c.plan(10, &[95, 0, 9_950]).unwrap().unwrap();
        assert_ne!(p.target_level, 3);
        // A plan already targeting the deepest level is unchanged.
        let p = c.plan(10, &[90]).unwrap().unwrap();
        assert_eq!((p.target_level, p.output_points), (1, 100));
        // No learned level yet: plain Bentley–Saxe.
        assert_eq!(c.plan(10, &[]).unwrap().unwrap().target_level, 1);
    }

    fn rebuild(levels: &[usize]) -> Option<(usize, Vec<usize>, usize)> {
        // b = 10, a = 10: Cmin = [10, 100, 1_000], Cmax = [100, 1_000, 10_000].
        HierarchyConfig {
            policy: MaintenancePolicy::Geometric,
            buffer_capacity: 10,
            arity: 10,
        }
        .plan_rebuild(levels)
        .unwrap()
        .map(|p| (p.target_level, p.source_levels, p.output_points))
    }

    #[test]
    fn greedy_rebuild_follows_the_three_target_rules() {
        // Nothing below its minimum capacity.
        assert_eq!(rebuild(&[50, 500, 5_000]), None);
        assert_eq!(rebuild(&[]), None);
        // Rule 1: level 2 (50 < 100) moves into the nearest non-empty deeper level 3,
        // and Greedy also absorbs level 1 because it still fits.
        assert_eq!(rebuild(&[40, 50, 5_000]), Some((3, vec![1, 2], 5_090)));
        // Rule 1 across an empty intermediate level 3 into level 4.
        let c = HierarchyConfig {
            policy: MaintenancePolicy::Geometric,
            buffer_capacity: 10,
            arity: 10,
        };
        let p = c.plan_rebuild(&[0, 60, 0, 20_000]).unwrap().unwrap();
        assert_eq!(
            (p.target_level, p.source_levels, p.output_points),
            (4, vec![2], 20_060)
        );
        // Greedy stops at the first level that does not fit (temporal order): level 3
        // takes level 2 (9_900 + 60 = 9_960), but adding level 1's 80 would exceed 10_000.
        assert_eq!(rebuild(&[80, 60, 9_900]), Some((3, vec![2], 9_960)));
        // Rule 2: the deeper level has no room, an upper level is non-empty, so level 2
        // is the target and absorbs level 1.
        assert_eq!(rebuild(&[70, 60, 9_990]), Some((2, vec![1], 130)));
        // Rule 3: no deeper level, nothing above: level 2's 60 points move up into level 1.
        assert_eq!(rebuild(&[0, 60]), Some((1, vec![2], 60)));
        // Deepest underutilised level first (its data is oldest).
        assert_eq!(rebuild(&[5, 60, 500]).map(|r| r.0), Some(3));
        // Level 1 alone below Cmin(1) is never moved into the buffer.
        assert_eq!(rebuild(&[5]), None);
        // Over-capacity input is rejected.
        assert!(c.plan_rebuild(&[101]).is_err());
    }

    #[test]
    fn vacuum_rewrites_the_deepest_level_above_the_dead_threshold() {
        let c = HierarchyConfig { policy: MaintenancePolicy::Geometric, buffer_capacity: 10, arity: 10 };
        let plan = |live: &[usize], physical: &[usize]| {
            c.plan_vacuum(live, physical, 200)
                .unwrap()
                .map(|p| (p.target_level, p.source_levels, p.buffer_points, p.output_points))
        };
        // Exactly 20% dead is not above the threshold; one more dead point is.
        assert_eq!(plan(&[80], &[100]), None);
        assert_eq!(plan(&[79], &[100]), Some((1, vec![], 0, 79)));
        // Deepest qualifying level first; levels without deletes are left alone.
        assert_eq!(plan(&[10, 500, 4_000], &[50, 900, 4_000]), Some((2, vec![], 0, 500)));
        assert_eq!(plan(&[10, 500, 4_000], &[50, 900, 5_100]), Some((3, vec![], 0, 4_000)));
        // Empty and fully deleted levels are skipped (a fully deleted level has nothing to keep).
        assert_eq!(plan(&[0, 0], &[0, 30]), None);
        // live > physical or mismatched lengths are rejected.
        assert!(c.plan_vacuum(&[5], &[4], 200).is_err());
        assert!(c.plan_vacuum(&[5], &[], 200).is_err());
    }
}
