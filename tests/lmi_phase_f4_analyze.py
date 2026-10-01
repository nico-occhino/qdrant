"""Analyze F.4 without changing any selection, model or raw observation."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

root = Path(os.environ.get('LMI_PHASE_F4_DIR', '/home/nicoo/work/lmi-phase-f4-data'))
protocol = json.loads((root / 'protocol.json').read_text())
done = json.loads((root / 'done.json').read_text())
assert done['complete']
selection = json.loads((root / 'selection.json').read_text())
names = selection['methods']


def dump(name, value):
    with (root / name).open('x') as target:
        json.dump(value, target, indent=2, allow_nan=False)


def write_csv(name, rows):
    with (root / name).open('x', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


preserved = {path: sha(Path(path)) == value for path, value in protocol['source_hashes'].items()}
assert all(preserved.values())
dump('source_preservation.json', {'verified_files': len(preserved), 'all_unchanged': True, 'checks': preserved})

routers = defaultdict(list)
with (root / 'router_queries.jsonl').open() as source:
    for line in source:
        row = json.loads(line)
        routers[(row['subset'], row['model'], row['nprobe'])].append(row)
oracles = defaultdict(list)
with (root / 'oracle_queries.jsonl').open() as source:
    for line in source:
        row = json.loads(line)
        oracles[(row['subset'], row['nprobe'])].append(row)
assert sum(map(len, routers.values())) == (300 + 1000) * len(names) * 64
assert sum(map(len, oracles.values())) == (300 + 1000) * 64

summary = []
for (subset, model, nprobe), rows in routers.items():
    assert len(rows) == (300 if subset == 'validation' else 1000)
    summary.append({
        'kind': 'practical_router', 'subset': subset, 'model': model, 'nprobe': nprobe,
        'query_count': len(rows),
        'recall': float(np.mean([row['recall'] for row in rows])),
        'candidate_mean': float(np.mean([row['candidate_count'] for row in rows])),
        'candidate_fraction': float(np.mean([row['candidate_fraction'] for row in rows])),
    })
for (subset, nprobe), rows in oracles.items():
    assert len(rows) == (300 if subset == 'validation' else 1000)
    summary.append({
        'kind': 'oracle_upper_bound', 'subset': subset, 'model': 'oracle', 'nprobe': nprobe,
        'query_count': len(rows),
        'recall': float(np.mean([row['oracle_recall'] for row in rows])),
        'candidate_mean': float(np.mean([row['candidate_count'] for row in rows])),
        'candidate_fraction': float(np.mean([row['candidate_fraction'] for row in rows])),
    })
summary.sort(key=lambda row: (row['subset'], row['kind'], row['model'], row['nprobe']))
write_csv('dense_summary.csv', summary)
dump('dense_summary.json', summary)

def row(kind, subset, model, probe):
    return next(value for value in summary if value['kind'] == kind and value['subset'] == subset and value['model'] == model and value['nprobe'] == probe)

observed = []
for subset in ('validation', 'test'):
    oracle = row('oracle_upper_bound', subset, 'oracle', 4)
    existing = row('practical_router', subset, 'existing_mlp_s42', 4)
    observed.append({
        'subset': subset, 'method': 'oracle_p4', **{key: oracle[key] for key in ('recall', 'candidate_mean', 'candidate_fraction')},
    })
    for model in names:
        item = row('practical_router', subset, model, 4)
        observed.append({
            'subset': subset, 'method': f'{model}_p4', **{key: item[key] for key in ('recall', 'candidate_mean', 'candidate_fraction')},
        })
write_csv('p4_summary.csv', observed)

selected = []
for model, probes in zip(names, selection['nprobe']):
    for target, probe in zip(selection['targets'], probes):
        validation = row('practical_router', 'validation', model, probe)
        test = row('practical_router', 'test', model, probe)
        selected.append({
            'model': model, 'target': target, 'nprobe': probe,
            'validation_recall': validation['recall'], 'test_recall': test['recall'],
            'candidate_mean': test['candidate_mean'], 'candidate_fraction': test['candidate_fraction'],
        })
write_csv('selected_operating_points.csv', selected)
dump('selected_operating_points.json', selected)

oracle_p4 = {item['query']: item for item in oracles[('test', 4)]}
router_p4 = {(model, item['query']): item for model in names for item in routers[('test', model, 4)]}
assert len(oracle_p4) == 1000
rng = np.random.default_rng(20261001)
draws = rng.integers(0, 1000, size=(2000, 1000))
headroom = []
for model in names:
    practical = np.array([router_p4[(model, query)]['recall'] for query in sorted(oracle_p4)])
    oracle = np.array([oracle_p4[query]['oracle_recall'] for query in sorted(oracle_p4)])
    gap = oracle - practical
    interval = np.percentile(gap[draws].mean(axis=1), [2.5, 97.5])
    headroom.append({
        'model': model,
        'p': 4,
        'oracle_recall': float(oracle.mean()),
        'practical_recall': float(practical.mean()),
        'oracle_minus_practical_recall': float(gap.mean()),
        'gap_ci95': [float(interval[0]), float(interval[1])],
        'queries_at_oracle_ceiling': int(np.sum(gap == 0)),
        'queries_with_positive_headroom': int(np.sum(gap > 0)),
        'maximum_per_query_gap': float(gap.max()),
    })
dump('oracle_headroom.json', {
    'resamples': 2000, 'seed': 20261001,
    'unit': 'test query; fixed partition and fixed trained router',
    'caveat': 'Exploratory query-resampling interval. It does not represent corpus partition, target-query sample, architecture, training-seed or dataset uncertainty.',
    'comparisons': headroom,
})

targets = json.loads((root / 'train_targets.json').read_text())
target_support = np.asarray([entry['bucket_counts'] for entry in targets], dtype=np.int64)
dump('target_diagnostics.json', {
    'train_queries': len(targets), 'label_rows': int(target_support.sum()),
    'active_target_buckets': int(np.sum(target_support.sum(axis=0) > 0)),
    'mean_distinct_relevant_buckets_per_query': float(np.mean((target_support > 0).sum(axis=1))),
    'single_bucket_queries': int(np.sum((target_support > 0).sum(axis=1) == 1)),
    'bucket_mass': target_support.sum(axis=0).tolist(),
})

sys.path.insert(0, '/home/nicoo/work/lmi-phase-f-python')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

colors = {'oracle': '#111111', 'existing_mlp_s42': '#2166ac', 'retrieval_target_mlp_s42': '#d6604d', 'retrieval_target_mlp_s43': '#1b9e77', 'retrieval_target_mlp_s44': '#8e63b6'}
fig, ax = plt.subplots(figsize=(8, 5))
for model in ['oracle'] + names:
    kind = 'oracle_upper_bound' if model == 'oracle' else 'practical_router'
    rows = [item for item in summary if item['kind'] == kind and item['subset'] == 'test' and item['model'] == model]
    rows.sort(key=lambda item: item['nprobe'])
    ax.plot([item['candidate_fraction'] for item in rows], [item['recall'] for item in rows], label=model, color=colors[model], linestyle='--' if model == 'oracle' else '-')
ax.set(xlabel='Mean fraction of fixed corpus partition scored', ylabel='Recall@10', title='F.4: oracle headroom and retrieval-target query routers')
ax.set_ylim(0.5, 1.005)
ax.grid(alpha=.2)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(root / 'oracle_and_router_curves.png', dpi=180)
plt.close(fig)

existing_gap = next(item for item in headroom if item['model'] == 'existing_mlp_s42')
control = [item for item in headroom if item['model'].startswith('retrieval_target')]
oracle_test = row('oracle_upper_bound', 'test', 'oracle', 4)
existing_test = row('practical_router', 'test', 'existing_mlp_s42', 4)
report = [
    '# Phase F.4: fixed-partition oracle headroom and retrieval-target MLP control', '',
    '## Main result', '',
    f"At four probes, the fixed-partition oracle reaches Recall@10 {oracle_test['recall']:.4f}. The existing corpus-label MLP reaches {existing_test['recall']:.4f}, leaving {existing_gap['oracle_minus_practical_recall']:.4f} mean recall headroom (95% exploratory paired query interval [{existing_gap['gap_ci95'][0]:.4f}, {existing_gap['gap_ci95'][1]:.4f}]). This is substantial enough to justify testing an alternative query router while keeping the corpus partition fixed.", '',
    'The retrieval-target MLP control uses the same 768 -> 64 ReLU -> 64 architecture and fixed postings as the existing MLP. It changes only the supervised query-routing target. Each training query contributes ten labeled copies, one for each exact top-10 neighbour bucket; this is empirical cross entropy on r_b(q)/10. Therefore any difference from the existing MLP cannot be attributed solely to prototype memory, which is not implemented in F.4.', '',
    '## Experimental scope', '',
    f"The experiment fixes the canonical F.2 `n4096_e60_s42` corpus partition before F.4 observations. Its seed was predeclared rather than selected from F.3 test results. The indexed corpus has {protocol['live_count']:,} live vectors after withholding every F.4 query row and exact normalized duplicates. Query split: 2,048 routing-training, 300 validation, 1,000 test and 20 warmups. The selected rows were excluded from F.2 training and F.3 queries but had been corpus rows in earlier experiments; this is fresh internal routing evaluation, not external validation.", '',
    f"One selected query has cosine similarity >= {protocol['near_duplicate_threshold']} to an earlier training/query row, although it is not an exact normalized duplicate. The split was frozen before running, so it is retained and reported. Exact duplicate query rows were removed from the corpus. No query is permitted to retrieve itself.", '',
    '## Four-probe comparison on the final test split', '',
    '| Method | Recall@10 | Mean candidates | Corpus fraction | Oracle gap |', '|---|---:|---:|---:|---:|',
]
for item in observed:
    if item['subset'] != 'test':
        continue
    label = item['method']
    gap = '' if label == 'oracle_p4' else f"{oracle_test['recall'] - item['recall']:.4f}"
    report.append(f"| {label} | {item['recall']:.4f} | {item['candidate_mean']:.1f} | {item['candidate_fraction']:.4f} | {gap} |")
report += ['', '## Oracle interpretation', '',
    'The oracle knows the exact answer before choosing buckets. It is an upper bound on Recall@10 for any query-side method that selects four buckets from this exact frozen partition and then scores their contents exactly. It is not deployable and its candidate counts are only descriptive because equal-relevance bucket ties can be broken in multiple ways. The bound does not cover improvements that change the corpus partition, number of probes, scoring method, metric, or k.', '',
    f"For the existing MLP, {existing_gap['queries_with_positive_headroom']} of 1,000 test queries have positive oracle headroom and {existing_gap['queries_at_oracle_ceiling']} already meet the oracle ceiling. The maximum per-query gap is {existing_gap['maximum_per_query_gap']:.1f} Recall@10. This shows where a better query selector may help; it does not guarantee that a trainable selector can realize the bound.", '',
    '## Retrieval-target control', '',
    '| Model | Oracle minus practical recall at p=4 [95% CI] |', '|---|---:|']
for item in headroom:
    report.append(f"| {item['model']} | {item['oracle_minus_practical_recall']:.4f} [{item['gap_ci95'][0]:.4f}, {item['gap_ci95'][1]:.4f}] |")
report += ['', 'The practical curves and validation-selected operating points are retained in the CSV/JSON artifacts. The validation policy chooses the smallest p reaching each target recall and is frozen before test; it must not be reselected from test curves. The p=4 oracle result remains the principal headroom question requested for this phase.', '',
    '## Verification', '',
    f"- {len(preserved)} F/F.2/F.3 source artifacts retained identical SHA-256 hashes.",
    f"- Fixed postings cover exactly {protocol['live_count']:,} live offsets once; no corpus reassignment occurs.",
    f"- {done['native_checks']} native Qdrant scorer checks compare fixed-bucket membership recall to actual scored top-k results; full-probe warmups match exact canonical IDs and scores.",
    f"- {done['trained_query_routers']} retrieval-target routers trained; Qdrant serving, persistence, configuration and existing learned postings were not changed.",
    '- Full probe reaches all ten exact neighbours for every dense membership observation.', '',
    '## Limits and next decision', '',
    'F.4 is a controlled diagnostic, not a claim that the objective-controlled router generalizes beyond this dataset/split. The empirical soft target is derived from exact search and therefore expensive to create. The three seeds vary initialization and minibatch order only; they do not cover partition, query-split or dataset uncertainty. No HNSW, build, persistence, lifecycle or HTTP result is added.', '',
    'Decision rule: if a retrieval-target MLP closes most of the oracle gap, objective choice is a stronger explanation than prototype memory. If material headroom remains after this control, prototype routing has a clearer rationale. Either result preserves the completed Qdrant integration and evaluates new routing ideas independently.', '',
    '## Reproduction', '',
    'Use a fresh F.4 output directory. Run `tests/lmi_phase_f4_prepare.py`, the ignored `phase_f4_oracle_and_objective_control` test with `--features lmi-training --locked`, then `tests/lmi_phase_f4_analyze.py`. Raw per-query oracle and router observations, targets, routers, selection, plots and source-preservation evidence are retained in the F.4 directory. No commit or push was performed.', '',
]
(root / 'Phase_F4_Report.md').write_text('\n'.join(report))
print(json.dumps({'headroom': headroom, 'selected': selected, 'protected_files': len(preserved)}, indent=2))
