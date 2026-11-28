"""Postprocess executed native queries; never select parameters from held-out results.

Usage: python analyze_matched_study.py RUN_DIRECTORY
Large retained datasets stay outside Git. No training or database mutation.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
a = p.parse_args()
root = Path(os.environ['ADMIN'], 'experiments/2026-11-12-dli-architecture')
base = root / 'continuation-25'
run = a.run
cases = json.loads((run / 'study.json').read_text())
locked = json.loads((run / 'locked-selection.json').read_text())
def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]
queries = jsonl(run / 'query-events.jsonl')
routes = jsonl(run / 'routing-events.jsonl')
access = jsonl(run / 'access-events.jsonl')
assert len(queries) == len(routes) == sum(c['query_count'] for c in cases)
assert locked['validation_rows'][1] <= locked['test_rows'][0]
gold = np.memmap(base / 'data/ground_truth.i32', mode='r', dtype='<i4', shape=(10000, 1000))
qvectors = np.memmap(base / 'data/queries.f32', mode='r', dtype='<f4', shape=(10000, 768))
vectors = np.memmap(base / 'data/vectors.f16', mode='r', dtype='<f2', shape=(10120191, 768))
used_rows = sorted({i for c in cases for i in range(c['query_start'], c['query_start'] + c['query_count'])})
wanted = set(map(int, gold[used_rows, :30].ravel()))
wanted.update(int(h['id']) for q in queries for h in q['hits'])
ids = np.array(sorted(wanted), dtype=np.uint64)
assert ids.min() >= 1 and ids.max() <= 10120191

# Authoritative offset mapping and persisted versions; no offset == ID assumption.
descriptor_path = base / 'geometric-10120191/collection/storage/0/dli_hierarchy.json'
descriptor = json.loads(descriptor_path.read_text())
owner, bucket_of, versions = {}, {}, {}
for uuid, level in descriptor['members']:
    if not level:
        plain_uuid = uuid
        continue
    directory = descriptor_path.parent / 'segments' / uuid
    path = directory / 'id_tracker.mappings'
    n = int(np.fromfile(path, dtype='<u8', count=1)[0])
    assert path.stat().st_size == 8 + 13*n
    entries = np.memmap(path, mode='r', offset=8,
                        dtype=[('kind', 'u1'), ('id', '<u8'), ('offset', '<u4')], shape=(n,))
    assert np.all(entries['kind'] == 0)
    order = np.argsort(entries['id'])
    sorted_ids = entries['id'][order]
    positions = np.searchsorted(sorted_ids, ids)
    valid = positions < n
    valid[valid] &= sorted_ids[positions[valid]] == ids[valid]
    with (directory / 'vector_index/lmi_postings.bin').open('rb') as f:
        boundary_count = int(np.frombuffer(f.read(8), dtype='<u8')[0])
        boundaries = np.frombuffer(f.read(boundary_count*8), dtype='<u8')
        count = int(np.frombuffer(f.read(8), dtype='<u8')[0])
        posting = np.frombuffer(f.read(count*4), dtype='<u4')
    assert count == n and len(np.unique(posting)) == n and posting.max() < n
    inverse = np.full(n, -1, dtype=np.int32)
    for bucket, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        inverse[posting[int(lo):int(hi)]] = bucket
    ver = np.memmap(directory / 'id_tracker.versions', mode='r', dtype='<u8')
    for external, offset in zip(ids[valid], entries['offset'][order[positions[valid]]]):
        external, offset = int(external), int(offset)
        assert external not in owner
        owner[external] = uuid
        bucket_of[external] = int(inverse[offset])
        versions[external] = int(ver[offset])
for external in wanted - owner.keys():
    # The retained insertion-only lineage contains exactly40x250K indexed IDs.
    assert 10_000_000 < external <= 10_120_191
    owner[external] = plain_uuid
    bucket_of[external] = None

rows, summaries = [], []
access_cursor = 0
max_error, version_checks = 0., 0
batch_version_checks=0
score_cache = {}
for case_id, case in enumerate(cases):
    begin = case['ordinal_begin']
    selected_rows = []
    for ordinal in range(begin, begin + case['query_count']):
        q, route = queries[ordinal], routes[ordinal]
        query_id = case['query_start'] + q['query']
        assert case['policy'] == route['policy'] and case['budget'] == route['budget']
        selected = {(b['physical'], b['bucket']) for b in route['buckets']}
        members = {u for u, _ in selected}
        assert sum(b['occupancy'] for b in route['buckets']) == q['scored_vectors']
        phases = q['phases']
        assert phases['selected_buckets'] == len(selected)
        assert phases['selected_members'] == len(members)
        stage = access[access_cursor:access_cursor+len(members)]
        access_cursor += len(members)
        assert {s['physical'] for s in stage} == members
        assert sum(s['returned_offsets'] for s in stage) == phases['materialized_hits']
        actual = [int(h['id']) for h in q['hits']]
        assert len(actual) == len(set(actual)) == 30
        for h in q['hits']:
            external = int(h['id'])
            # Recorded insertion-only stream:41 sequential upserts of <=250K rows.
            assert h['version'] == (external-1)//250000+1
            batch_version_checks+=1
            if external in versions:
                assert h['version'] == versions[external]
                version_checks += 1
            key = query_id, external
            if key not in score_cache:
                v = vectors[external-1].astype(np.float64)
                query = qvectors[query_id].astype(np.float64)
                score_cache[key] = float(v @ query / (np.linalg.norm(v)*np.linalg.norm(query)))
            error = abs(float(h['score']) - score_cache[key])
            max_error = max(max_error, error)
            assert error < .0005, (key, h['score'], score_cache[key], error)
        quality = {}
        for k in [10, 30]:
            truth = list(map(int, gold[query_id, :k]))
            quality[f'recall{k}'] = len(set(actual[:k]) & set(truth)) / k
            quality[f'scope{k}'] = sum(owner[g] in members for g in truth)/k
            quality[f'bucket{k}'] = sum((owner[g], bucket_of[g]) in selected for g in truth)/k
        row = dict(case=case_id, ordinal=ordinal, query=query_id, policy=case['policy'],
                   budget=case['budget'], topk=case['topk'], pass_name=case['pass'],
                   scored=q['scored_vectors'], selected_members=len(members),
                   selected_models=sum(u != plain_uuid for u in members), selected_buckets=len(selected),
                   materialized=phases['materialized_hits'], latency_ms=q['latency_ms'],
                   process_cpu_ticks=q['process_cpu_ticks'], major_faults=q['process_major_faults'],
                   minor_faults=q['process_minor_faults'], **quality)
        row.update({k:phases[k]/1e6 for k in ['native_scorer_nanos', 'hit_materialization_nanos',
                                            'resolution_nanos', 'router_inference_nanos',
                                            'bucket_planning_nanos', 'read_guard_nanos']})
        rows.append(row)
        selected_rows.append(row)
    mean = lambda key: float(np.mean([r[key] for r in selected_rows]))
    for k in [10, 30]:
        assert abs(mean(f'recall{k}') - case[f'recall{k}']) < 1e-10
    assert sum(r['scored'] for r in selected_rows) == case['scored_vectors']
    summary = dict(case=case_id, policy=case['policy'], budget=case['budget'], topk=case['topk'],
                   pass_name=case['pass'], queries=len(selected_rows),
                   mean_scored=mean('scored'), mean_materialized=mean('materialized'),
                   mean_models=mean('selected_models'), mean_buckets=mean('selected_buckets'),
                   recall10=mean('recall10'), recall30=mean('recall30'), scope10=mean('scope10'),
                   bucket10=mean('bucket10'), scope30=mean('scope30'), bucket30=mean('bucket30'),
                   p50_ms=case['p50_ms'], p95_ms=case['p95_ms'], mean_ms=case['mean_ms'],
                   major_faults=sum(r['major_faults'] for r in selected_rows),
                   process_cpu_ticks=sum(r['process_cpu_ticks'] for r in selected_rows))
    summary.update({k:mean(k) for k in ['native_scorer_nanos','hit_materialization_nanos',
                                      'resolution_nanos','router_inference_nanos','bucket_planning_nanos']})
    summaries.append(summary)
assert access_cursor == len(access)
def write_csv(name, data):
    with (run / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
write_csv('per-query.csv', rows)
write_csv('cases.csv', summaries)
# Values in *_nanos columns above are converted to milliseconds, rename explicitly.
for name in ['per-query.csv', 'cases.csv']:
    path = run / name
    path.write_text(path.read_text().replace('_nanos', '_ms'))

# Native ordering compares scores only; equal-score order need not be stable.
controls=[c for c in cases if c['pass']=='materialization_control']
ordered_pairs=tie_pairs=0
if controls:
    reference=controls[0]
    for c in controls[1:]:
        for i in range(c['query_count']):
            aa=queries[c['ordinal_begin']+i]['hits']
            bb=queries[reference['ordinal_begin']+i]['hits']
            assert {h['id']:(h['score'],h['version']) for h in aa} == {h['id']:(h['score'],h['version']) for h in bb}
            assert [h['score'] for h in aa]==[h['score'] for h in bb]
            if aa==bb:ordered_pairs+=1
            else:tie_pairs+=1
# Evaluate match quality instead of assuming validation targets transfer perfectly.
validation = [s for s in summaries if s['pass_name'] == 'validation']
test = [s for s in summaries if s['pass_name'] == 'test_warm']
comparisons = {}
for objective in (['work30000', 'recall060'] if validation else []):
    group = []
    for policy in ['SlotWeighted','ReservedPerLevel','PopulationDensity']:
        relevant = [s for s in validation if s['policy'] == policy]
        key, target = ('mean_scored', 30000.) if objective == 'work30000' else ('recall10', .6)
        chosen = min(relevant, key=lambda r: abs(r[key] - target))
        selected = next(s for s in test if s['policy']==policy and s['budget']==chosen['budget'])
        group.append(selected)
    work = [s['mean_scored'] for s in group]
    comparisons[objective] = dict(cases=[s['case'] for s in group],
        work_spread_ratio=max(work)/min(work)-1,
        approximately_matched_work=max(work)/min(work)<=1.10,
        recall10_spread=max(s['recall10'] for s in group)-min(s['recall10'] for s in group),
        recall30_spread=max(s['recall30'] for s in group)-min(s['recall30'] for s in group))
(run/'checks.json').write_text(json.dumps(dict(population=10120191, query_rows=used_rows,
    processed_queries=len(rows), max_independent_cosine_error=max_error,
    checked_persisted_learned_versions=version_checks,
    checked_insertion_stream_versions=batch_version_checks,
    plain_versions='Checked against recorded41-batch insertion stream; mutable-tracker file not independently decoded',
    identical_hit_set_score_version_pairs=ordered_pairs+tie_pairs, exact_ordered_pairs=ordered_pairs, equal_score_permutation_pairs=tie_pairs,
    match_checks=comparisons, cpu_units='process ticks; use recorded CLK_TCK; not per-query high-resolution CPU',
    timing='release; detailed tracing enabled; cold-member clearing is an attempt, not full-dataset cache isolation'), indent=2))
print(json.dumps(comparisons, indent=2))
