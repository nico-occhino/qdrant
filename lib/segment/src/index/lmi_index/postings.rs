//! Contiguous, offset-only LMI postings. Vectors remain in Qdrant VectorStorage.
use common::types::PointOffsetType;
use serde::{Deserialize, Serialize};

use crate::common::operation_error::{OperationError, OperationResult};

/// Binary layout (bincode fixed-width): u64 boundary count, u64 boundaries,
/// u64 posting count, u32 point offsets. All integer encoding is little-endian.
/// Deserialization is followed by `validate` before any ranges are exposed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactPostings {
    boundaries: Vec<u64>,
    points: Vec<PointOffsetType>,
}

impl CompactPostings {
    pub(crate) fn heap_bytes(&self) -> usize {
        self.boundaries
            .capacity()
            .saturating_mul(std::mem::size_of::<u64>())
            .saturating_add(
                self.points
                    .capacity()
                    .saturating_mul(std::mem::size_of::<PointOffsetType>()),
            )
    }

    pub fn from_buckets(buckets: Vec<Vec<PointOffsetType>>) -> OperationResult<Self> {
        let count = buckets
            .iter()
            .try_fold(0usize, |sum, b| sum.checked_add(b.len()))
            .ok_or_else(|| OperationError::service_error("LMI posting count overflow"))?;
        let mut points = Vec::new();
        points
            .try_reserve_exact(count)
            .map_err(|e| OperationError::service_error(format!("LMI posting allocation: {e}")))?;
        let mut boundaries = Vec::with_capacity(buckets.len() + 1);
        boundaries.push(0);
        for mut bucket in buckets {
            points.append(&mut bucket);
            boundaries.push(points.len() as u64);
        }
        Self::from_parts(boundaries, points)
    }

    pub fn from_parts(boundaries: Vec<u64>, points: Vec<PointOffsetType>) -> OperationResult<Self> {
        let result = Self { boundaries, points };
        result.validate()?;
        Ok(result)
    }

    pub(super) fn validate(&self) -> OperationResult<()> {
        if self.boundaries.first() != Some(&0)
            || self.boundaries.last().copied() != Some(self.points.len() as u64)
            || self.boundaries.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(OperationError::service_error(
                "LMI invalid posting boundaries",
            ));
        }
        Ok(())
    }

    /// Count, allocate exactly once, then repeat the same stable stream to fill.
    /// The caller owns a stable snapshot/read borrow for the duration of both passes.
    #[cfg(test)]
    pub(super) fn build_two_pass(
        bucket_count: usize,
        stopped: &std::sync::atomic::AtomicBool,
        mut visit: impl FnMut(
            &mut dyn FnMut(PointOffsetType, usize) -> OperationResult<()>,
        ) -> OperationResult<()>,
    ) -> OperationResult<(Self, [f64; 3])> {
        use crate::common::operation_error::check_process_stopped;
        use std::time::Instant;
        check_process_stopped(stopped)?;
        let started = Instant::now();
        let mut counts = vec![0usize; bucket_count];
        visit(&mut |_, bucket| {
            check_process_stopped(stopped)?;
            let count = counts.get_mut(bucket).ok_or_else(|| {
                OperationError::service_error("LMI predicted bucket out of range")
            })?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| OperationError::service_error("LMI bucket count overflow"))?;
            Ok(())
        })?;
        let count_seconds = started.elapsed().as_secs_f64();
        check_process_stopped(stopped)?;
        let started = Instant::now();
        let mut boundaries = Vec::with_capacity(bucket_count + 1);
        boundaries.push(0u64);
        let mut total = 0usize;
        for count in counts {
            total = total
                .checked_add(count)
                .ok_or_else(|| OperationError::service_error("LMI posting count overflow"))?;
            boundaries.push(total as u64);
        }
        let mut points = Vec::new();
        points
            .try_reserve_exact(total)
            .map_err(|e| OperationError::service_error(format!("LMI posting allocation: {e}")))?;
        points.resize(total, 0);
        let mut cursors = boundaries[..bucket_count].to_vec();
        let allocation_seconds = started.elapsed().as_secs_f64();
        check_process_stopped(stopped)?;
        let started = Instant::now();
        visit(&mut |id, bucket| {
            check_process_stopped(stopped)?;
            let cursor = cursors.get_mut(bucket).ok_or_else(|| {
                OperationError::service_error("LMI predicted bucket out of range")
            })?;
            if *cursor >= boundaries[bucket + 1] {
                return Err(OperationError::service_error(
                    "LMI posting count changed between passes",
                ));
            }
            points[*cursor as usize] = id;
            *cursor += 1;
            Ok(())
        })?;
        check_process_stopped(stopped)?;
        if cursors.iter().zip(&boundaries[1..]).any(|(a, b)| a != b) {
            return Err(OperationError::service_error(
                "LMI posting count changed between passes",
            ));
        }
        let fill_seconds = started.elapsed().as_secs_f64();
        Ok((
            Self::from_parts(boundaries, points)?,
            [count_seconds, allocation_seconds, fill_seconds],
        ))
    }

    /// Route each eligible point once, then replay only its offset to fill postings.
    /// The caller must hold immutable ID-tracker and vector-storage borrows across
    /// both traversals, so the replay has the same eligible offsets and order.
    /// No sentinel is used: all 65_536 u16 bucket IDs are available.
    #[cfg(any(feature = "lmi-training", test))]
    pub(super) fn build_cached_u16(
        bucket_count: usize,
        eligible_count: usize,
        stopped: &std::sync::atomic::AtomicBool,
        mut route: impl FnMut(
            &mut dyn FnMut(PointOffsetType, usize) -> OperationResult<()>,
        ) -> OperationResult<()>,
        mut replay: impl FnMut(
            &mut dyn FnMut(PointOffsetType) -> OperationResult<()>,
        ) -> OperationResult<()>,
    ) -> OperationResult<(Self, [f64; 3])> {
        use crate::common::operation_error::check_process_stopped;
        use std::time::Instant;
        if bucket_count > u16::MAX as usize + 1 {
            return Err(OperationError::service_error(
                "LMI u16 label cache cannot represent bucket count",
            ));
        }
        check_process_stopped(stopped)?;
        let started = Instant::now();
        let mut labels = Vec::<u16>::new();
        labels.try_reserve_exact(eligible_count).map_err(|e| {
            OperationError::service_error(format!("LMI label cache allocation: {e}"))
        })?;
        let mut counts = vec![0usize; bucket_count];
        route(&mut |_, bucket| {
            check_process_stopped(stopped)?;
            let label = u16::try_from(bucket)
                .map_err(|_| OperationError::service_error("LMI bucket ID exceeds u16"))?;
            let count = counts.get_mut(bucket).ok_or_else(|| {
                OperationError::service_error("LMI predicted bucket out of range")
            })?;
            if labels.len() == eligible_count {
                return Err(OperationError::service_error(
                    "LMI routed more points than eligible scan",
                ));
            }
            *count = count
                .checked_add(1)
                .ok_or_else(|| OperationError::service_error("LMI bucket count overflow"))?;
            labels.push(label);
            Ok(())
        })?;
        if labels.len() != eligible_count {
            return Err(OperationError::service_error(
                "LMI routed fewer points than eligible scan",
            ));
        }
        let count_seconds = started.elapsed().as_secs_f64();
        check_process_stopped(stopped)?;
        let started = Instant::now();
        let mut boundaries = Vec::with_capacity(bucket_count + 1);
        boundaries.push(0u64);
        let mut total = 0usize;
        for count in counts {
            total = total
                .checked_add(count)
                .ok_or_else(|| OperationError::service_error("LMI posting count overflow"))?;
            boundaries.push(total as u64);
        }
        let mut points = Vec::new();
        points
            .try_reserve_exact(total)
            .map_err(|e| OperationError::service_error(format!("LMI posting allocation: {e}")))?;
        points.resize(total, 0);
        let mut cursors = boundaries[..bucket_count].to_vec();
        let allocation_seconds = started.elapsed().as_secs_f64();
        check_process_stopped(stopped)?;
        let started = Instant::now();
        let mut position = 0usize;
        replay(&mut |id| {
            check_process_stopped(stopped)?;
            let Some(&label) = labels.get(position) else {
                return Err(OperationError::service_error(
                    "LMI eligible offset count changed during label replay",
                ));
            };
            position += 1;
            let bucket = usize::from(label);
            let cursor = &mut cursors[bucket];
            points[*cursor as usize] = id;
            *cursor += 1;
            Ok(())
        })?;
        check_process_stopped(stopped)?;
        if position != labels.len() || cursors.iter().zip(&boundaries[1..]).any(|(a, b)| a != b) {
            return Err(OperationError::service_error(
                "LMI eligible offset count changed during label replay",
            ));
        }
        let fill_seconds = started.elapsed().as_secs_f64();
        Ok((
            Self::from_parts(boundaries, points)?,
            [count_seconds, allocation_seconds, fill_seconds],
        ))
    }

    pub fn len(&self) -> usize {
        self.boundaries.len() - 1
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn point_count(&self) -> usize {
        self.points.len()
    }
    pub fn get(&self, bucket: usize) -> Option<&[PointOffsetType]> {
        if bucket >= self.len() {
            return None;
        }
        Some(&self.points[self.boundaries[bucket] as usize..self.boundaries[bucket + 1] as usize])
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[PointOffsetType]> {
        self.boundaries
            .windows(2)
            .map(|r| &self.points[r[0] as usize..r[1] as usize])
    }
    /// Diagnostic/legacy fixture conversion; never used by native build or search.
    pub fn to_vec(&self) -> Vec<Vec<PointOffsetType>> {
        self.iter().map(<[u32]>::to_vec).collect()
    }
}

impl std::ops::Index<usize> for CompactPostings {
    type Output = [PointOffsetType];
    fn index(&self, bucket: usize) -> &Self::Output {
        self.get(bucket).expect("bucket in range")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_bucket_ranges_round_trip() {
        for buckets in [
            vec![],
            vec![vec![]],
            vec![vec![3, 1]],
            vec![vec![], vec![8, 2], vec![], vec![5]],
        ] {
            let p = CompactPostings::from_buckets(buckets.clone()).unwrap();
            assert_eq!(p.to_vec(), buckets);
            assert!(p.get(p.len()).is_none());
            let bytes = bincode::serialize(&p).unwrap();
            assert_eq!(bytes.len(), 16 + 8 * (p.len() + 1) + 4 * p.point_count());
            let decoded: CompactPostings = bincode::deserialize(&bytes).unwrap();
            decoded.validate().unwrap();
            assert_eq!(decoded, p);
        }
    }
    #[test]
    fn compact_rejects_corrupt_boundaries() {
        for boundaries in [
            vec![],
            vec![1, 2],
            vec![0, 3, 2],
            vec![0, 1],
            vec![0, u64::MAX, 2],
        ] {
            assert!(CompactPostings::from_parts(boundaries, vec![0, 1]).is_err());
        }
    }
    #[test]
    fn compact_100k_offsets_have_exact_binary_size() {
        let mut buckets = vec![vec![]; 316];
        for id in 0..100_000u32 {
            buckets[id as usize % 316].push(id);
        }
        let p = CompactPostings::from_buckets(buckets).unwrap();
        assert_eq!(p.point_count(), 100_000);
        assert_eq!(bincode::serialized_size(&p).unwrap(), 402_552);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("postings.bin");
        common::fs::atomic_save_bin(&path, &p).unwrap();
        let reopened: CompactPostings =
            common::universal_io::read_bin_via(&common::universal_io::MmapFs, &path).unwrap();
        reopened.validate().unwrap();
        assert_eq!(reopened, p);
        assert_eq!(std::fs::metadata(path).unwrap().len(), 402_552);
    }
    #[test]
    fn two_pass_matches_legacy_with_holes_and_empty_buckets() {
        let stopped = std::sync::atomic::AtomicBool::new(false);
        for buckets in [1, 2, 64, 316] {
            let source: Vec<(u32, usize)> = (0..10_000u32)
                .filter(|id| id % 3 != 0)
                .map(|id| (id, (id as usize * 7) % buckets))
                .collect();
            let mut legacy = vec![vec![]; buckets];
            for &(id, b) in &source {
                legacy[b].push(id);
            }
            let mut passes = 0;
            let (result, _) = CompactPostings::build_two_pass(buckets, &stopped, |push| {
                passes += 1;
                for &(id, b) in &source {
                    push(id, b)?;
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(passes, 2);
            assert_eq!(result.to_vec(), legacy);
        }
        let (empty, _) = CompactPostings::build_two_pass(64, &stopped, |_| Ok(())).unwrap();
        assert_eq!(empty.point_count(), 0);
        assert_eq!(empty.len(), 64);
    }

    #[test]
    fn two_pass_cancellation_and_changed_source_return_errors() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stopped = AtomicBool::new(true);
        assert!(
            CompactPostings::build_two_pass(2, &stopped, |_| panic!("must stop first")).is_err()
        );
        for cancel_pass in [1, 2] {
            stopped.store(false, Ordering::Relaxed);
            let mut pass = 0;
            assert!(
                CompactPostings::build_two_pass(2, &stopped, |push| {
                    pass += 1;
                    push(3, 0)?;
                    if pass == cancel_pass {
                        stopped.store(true, Ordering::Relaxed);
                    }
                    push(7, 1)
                })
                .is_err()
            );
        }
        stopped.store(false, Ordering::Relaxed);
        for extra in [false, true] {
            let mut pass = 0;
            assert!(
                CompactPostings::build_two_pass(2, &stopped, |push| {
                    pass += 1;
                    push(3, 0)?;
                    if (pass == 1) != extra {
                        push(7, 1)?;
                    }
                    Ok(())
                })
                .is_err()
            );
        }
        assert!(CompactPostings::build_two_pass(2, &stopped, |push| push(0, 2)).is_err());
    }

    #[test]
    fn cached_labels_match_two_routing_passes_with_sparse_and_deleted_offsets() {
        use std::sync::atomic::AtomicBool;
        let stopped = AtomicBool::new(false);
        // Holes at both ends, in the middle, and several consecutive tombstones.
        let eligible = [2u32, 5, 9, 10, 17, 23, 29];
        for bucket_count in [1, 4, 64, 65_536] {
            let bucket = |id: u32| {
                if bucket_count == 65_536 && id == 29 {
                    65_535
                } else {
                    (id as usize * 7) % bucket_count
                }
            };
            let (old, _) = CompactPostings::build_two_pass(bucket_count, &stopped, |push| {
                for id in eligible {
                    push(id, bucket(id))?;
                }
                Ok(())
            })
            .unwrap();
            let mut route_calls = 0;
            let (cached, _) = CompactPostings::build_cached_u16(
                bucket_count,
                eligible.len(),
                &stopped,
                |push| {
                    for id in eligible {
                        route_calls += 1;
                        push(id, bucket(id))?;
                    }
                    Ok(())
                },
                |push| {
                    for id in eligible {
                        push(id)?;
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(route_calls, eligible.len());
            assert_eq!(cached, old);
            assert_eq!(
                bincode::serialize(&cached).unwrap(),
                bincode::serialize(&old).unwrap()
            );
        }
    }

    #[test]
    fn cached_labels_reject_bad_replay_and_cancellation() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stopped = AtomicBool::new(false);
        let run = |replay: &[u32]| {
            CompactPostings::build_cached_u16(
                2,
                2,
                &stopped,
                |push| {
                    push(2, 0)?;
                    push(9, 1)
                },
                |push| {
                    for &id in replay {
                        push(id)?;
                    }
                    Ok(())
                },
            )
        };
        assert!(run(&[2]).is_err());
        assert!(run(&[2, 9, 12]).is_err());
        stopped.store(true, Ordering::Relaxed);
        assert!(run(&[2, 9]).is_err());
        stopped.store(false, Ordering::Relaxed);
        assert!(
            CompactPostings::build_cached_u16(65_537, 0, &stopped, |_| Ok(()), |_| Ok(())).is_err()
        );
    }

    #[test]
    #[ignore = "Explicit bounded 100K posting-builder comparison"]
    fn cached_labels_100k_synthetic_benchmark() {
        use std::sync::atomic::AtomicBool;
        use std::time::Instant;

        let stopped = AtomicBool::new(false);
        let mode = std::env::var("LMI_S3E_BENCH_MODE").unwrap_or_else(|_| "both".into());
        let classify = |id: u32| {
            let mut value = u64::from(id);
            for _ in 0..64 {
                value = value
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
            }
            std::hint::black_box((value % 64) as usize)
        };
        let mut old = None;
        if mode != "cached" {
            let started = Instant::now();
            let (postings, stages) = CompactPostings::build_two_pass(64, &stopped, |push| {
                for id in 0..100_000u32 {
                    push(id, classify(id))?;
                }
                Ok(())
            })
            .unwrap();
            println!(
                "s3e_bench mode=old n=100000 route1_s={:.6} alloc_s={:.6} route2_s={:.6} total_s={:.6}",
                stages[0],
                stages[1],
                stages[2],
                started.elapsed().as_secs_f64()
            );
            old = Some(postings);
        }
        if mode != "old" {
            let started = Instant::now();
            let (postings, stages) = CompactPostings::build_cached_u16(
                64,
                100_000,
                &stopped,
                |push| {
                    for id in 0..100_000u32 {
                        push(id, classify(id))?;
                    }
                    Ok(())
                },
                |push| {
                    for id in 0..100_000u32 {
                        push(id)?;
                    }
                    Ok(())
                },
            )
            .unwrap();
            println!(
                "s3e_bench mode=cached n=100000 route_cache_s={:.6} alloc_s={:.6} fill_s={:.6} total_s={:.6}",
                stages[0],
                stages[1],
                stages[2],
                started.elapsed().as_secs_f64()
            );
            if let Some(old) = old {
                assert_eq!(
                    bincode::serialize(&postings).unwrap(),
                    bincode::serialize(&old).unwrap()
                );
            }
        }
    }
}
