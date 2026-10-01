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

    pub(super) fn from_parts(
        boundaries: Vec<u64>,
        points: Vec<PointOffsetType>,
    ) -> OperationResult<Self> {
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
    #[cfg(any(feature = "lmi-training", test))]
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
}
