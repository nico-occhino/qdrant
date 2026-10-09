"""Independently validate persisted v3 postings and full query results."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


N = 10_120_191
B = 3_162
out = Path('/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap10m/pre_g2_119_attempt2')
build = json.loads((out / 'build.json').read_text())
state = Path(build['state_path'])
index = state.parent
names = ('lmi_state.json', 'lmi_router.bin', 'lmi_postings.bin')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
files = {name: {'size': (index / name).stat().st_size, 'sha256': digest(index / name)} for name in names}
before = json.loads((out / 'hashes_after_build.json').read_text())
assert files == before, 'LMI files changed after publication and query evaluation'
postings = index / 'lmi_postings.bin'
boundary_count = np.memmap(postings, dtype='<u8', mode='r', shape=(1,))[0]
assert boundary_count == B + 1, boundary_count
boundaries = np.memmap(postings, dtype='<u8', mode='r', offset=8, shape=(B + 1,))
assert boundaries[0] == 0 and boundaries[-1] == N
assert np.all(boundaries[1:] >= boundaries[:-1])
count_offset = 8 + 8 * (B + 1)
posting_count = np.memmap(postings, dtype='<u8', mode='r', offset=count_offset, shape=(1,))[0]
assert posting_count == N
points = np.memmap(postings, dtype='<u4', mode='r', offset=count_offset + 8, shape=(N,))
assert postings.stat().st_size == count_offset + 8 + N * 4
assert points.min() == 0 and points.max() == N - 1
hist = np.bincount(points, minlength=N)
assert len(hist) == N and np.all(hist == 1)
rows = [json.loads(line) for line in (out / 'queries.full.jsonl').read_text().splitlines()]
assert len(rows) == 9980
assert [r['query_row'] for r in rows] == list(range(20, 10000))
with h5py.File('/mnt/c/datasets/sisap2023/laion2B-en-public-gold-standard-v2-10M-F64-IEEE754.h5') as f:
    gold = f['knns']
    for row in rows:
        ids = row['returned_ids_1_based']
        assert len(ids) == 10 and len(set(ids)) == 10
        assert all(1 <= x <= N for x in ids)
        expected = set(map(int, gold[row['query_row'], :10]))
        assert abs(row['recall_at_10'] - len(expected.intersection(ids)) / 10) < 1e-12
recalls = np.array([r['recall_at_10'] for r in rows])
candidates = np.array([r['candidate_count'] for r in rows])
latency = np.array([r['native_search_ms'] for r in rows])
warm_latency = np.array([r['native_search_ms'] for r in rows if r['query_row'] >= 1000])
summary = json.loads((out / 'summary.json').read_text())
assert abs(summary['recall_at_10_mean'] - float(recalls.mean())) < 1e-12
assert abs(summary['mean_candidates'] - float(candidates.mean())) < 1e-9
assert abs(summary['mean_candidate_fraction'] - float(candidates.mean()) / N) < 1e-12
result = {
    'exact_coverage': True, 'posting_count': N, 'min_offset': 0,
    'max_offset': N - 1, 'duplicate_offsets': 0, 'missing_offsets': 0,
    'files': files, 'state_unchanged_after_queries': True,
    'query_rows_validated': len(rows),
    'mean_recall': float(recalls.mean()), 'mean_candidates': float(candidates.mean()),
    'native_latency_ms': {
        'mean': float(latency.mean()), 'p50': float(np.quantile(latency, 0.5)),
        'p95': float(np.quantile(latency, 0.95)), 'p99': float(np.quantile(latency, 0.99)),
        'max': float(latency.max()),
    },
    'native_warm_suffix_rows_1000_9999_ms': {
        'count': len(warm_latency), 'mean': float(warm_latency.mean()),
        'p50': float(np.quantile(warm_latency, 0.5)),
        'p95': float(np.quantile(warm_latency, 0.95)),
        'p99': float(np.quantile(warm_latency, 0.99)),
        'max': float(warm_latency.max()),
    },
    'bucket_occupancy': {
        'empty': int(np.sum(np.diff(boundaries) == 0)),
        'median': float(np.median(np.diff(boundaries))),
        'largest': int(np.max(np.diff(boundaries))),
    },
}
(out / 'independent_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
