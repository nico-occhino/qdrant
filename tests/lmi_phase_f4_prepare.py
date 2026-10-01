"""Create a fresh, immutable F.4 split for oracle and retrieval-target routing."""
from pathlib import Path
import hashlib
import json
import os
import random
import subprocess

import numpy as np

base = Path('/home/nicoo/work/lmi-phase-f-data')
f2 = Path('/home/nicoo/work/lmi-phase-f2-data')
f3 = Path('/home/nicoo/work/lmi-phase-f3-data')
out = Path(os.environ.get('LMI_PHASE_F4_DIR', '/home/nicoo/work/lmi-phase-f4-data'))
assert not out.exists(), 'F.4 output already exists; preserve it rather than overwrite it'
out.mkdir()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    with path.open('x') as target:
        json.dump(value, target, indent=2, allow_nan=False)


source_hashes = {
    str(path): sha(path)
    for root in (base, f2, f3)
    for path in root.rglob('*')
    if path.is_file()
}
raw = np.fromfile(base / 'corpus.f32', dtype='<f4').reshape(-1, 768)
norms = np.linalg.norm(raw, axis=1, keepdims=True)
normalized = raw / norms
keys = [hashlib.sha256(row.tobytes()).digest() for row in normalized]
f2_protocol = json.loads((f2 / 'protocol.json').read_text())
f3_protocol = json.loads((f3 / 'protocol.json').read_text())
old_queries = np.fromfile(base / 'queries.f32', dtype='<f4').reshape(-1, 768)
old_queries /= np.linalg.norm(old_queries, axis=1, keepdims=True)
reserved = set(f2_protocol['sample_order'][:16384]) | set(f3_protocol['query_offsets'])
blocked_keys = {keys[index] for index in reserved}
blocked_keys |= {hashlib.sha256(row.tobytes()).digest() for row in old_queries}

pool = list(range(len(raw)))
random.Random(20261001).shuffle(pool)
selected = []
used = set(blocked_keys)
for index in pool:
    if keys[index] not in used:
        selected.append(index)
        used.add(keys[index])
        if len(selected) == 3368:
            break
assert len(selected) == 3368
assert not (set(selected) & reserved)
selected_keys = {keys[index] for index in selected}
removed = [index for index, key in enumerate(keys) if key in selected_keys]
assert len(removed) >= len(selected)

query_reference = np.concatenate([normalized[sorted(reserved)], old_queries])
near = []
for start in range(0, len(selected), 64):
    near.extend((normalized[selected[start:start + 64]] @ query_reference.T).max(axis=1).tolist())

meta = json.loads((base / 'dataset.json').read_text())
dump(out / 'protocol.json', {
    'base': str(base), 'f2': str(f2), 'f3': str(f3), 'seed': 20261001,
    'dimension': 768, 'original_count': len(raw), 'live_count': len(raw) - len(removed),
    'query_offsets': selected, 'removed_offsets': removed,
    'warmup': 20, 'train': 2048, 'validation': 300, 'test': 1000,
    'partition_model': 'n4096_e60_s42',
    'partition_selection': 'Canonical F.2 seed 42 selected before F.4 observations; no seed was chosen from F.3 test performance.',
    'router_seeds': [42, 43, 44],
    'router_config': {'hidden_dim': 64, 'epochs': 60, 'batch_size': 256, 'learning_rate': 0.001},
    'target_definition': 'For each train query, repeat the normalized query once for each canonical exact top-10 neighbour and assign that neighbour fixed corpus-bucket label. This is an empirical cross-entropy target with bucket mass r_b(q)/10.',
    'oracle_definition': 'For each query and p, sort fixed buckets by descending r_b(q), tie by bucket id; oracle recall is sum selected r_b(q)/10. p=4 is the requested headroom bound.',
    'selection': 'For each practical router and target recall 0.90/0.95, select the smallest p on validation. Persist selection before test. Oracle is reported at fixed p=4 and as a dense upper-bound curve; it is never timed as a practical query method.',
    'targets': [0.90, 0.95], 'dense_nprobe': list(range(1, 65)), 'trials': 2,
    'tie_rule': 'Score descending then segment offset ascending. Exact and candidate paths retrieve k+1 and expand through all equal-score boundary ties before canonical top-10 truncation.',
    'query_source': 'Former corpus rows excluded from F.2 teacher/training sample and F.3 query split. They were visible as corpus in earlier studies; this is fresh internal routing evaluation, not an external dataset.',
    'deduplication': 'Queries are unique by normalized f32 hash; all exact normalized duplicates are removed from the search corpus. Exact duplicates to earlier training/query rows are excluded.',
    'near_duplicate_threshold': 0.9999,
    'near_training_or_old_query_count': sum(value >= 0.9999 for value in near),
    'max_cosine_to_training_or_old_queries': near,
    'source_dataset_metadata': meta, 'source_hashes': source_hashes,
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
})
print(json.dumps({'out': str(out), 'queries': len(selected), 'removed': len(removed), 'live': len(raw) - len(removed), 'near_duplicate_queries': sum(value >= 0.9999 for value in near), 'protected_files': len(source_hashes)}))
