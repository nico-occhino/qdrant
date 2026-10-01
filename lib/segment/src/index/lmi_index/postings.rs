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
    }
}
