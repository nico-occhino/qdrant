"""Summarize raw Phase F query rows; run only after a successful benchmark."""
import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=Path(a.directory)
rows=[json.loads(line) for line in (root/'queries.jsonl').read_text().splitlines()]
build=json.loads((root/'build.json').read_text())
def percentile(xs,p):
    xs=sorted(xs); t=(len(xs)-1)*p; lo=int(t); hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(t-lo)
groups=defaultdict(list)
for row in rows: groups[row['method'],row['effort']].append(row)
summary=[]
for (method,effort),g in sorted(groups.items()):
    s={'method':method,'effort':effort,'observations':len(g),'recall_at_10':statistics.mean(r['recall'] for r in g)}
    for column in ['total_ns','router_ns','preparation_ns','scoring_ns','native_mlp_index_ns']:
        values=[r[column]/1e6 for r in g if r[column] is not None]
        for label,q in [('p50',.5),('p95',.95),('p99',.99)]:
            s[column.removesuffix('_ns')+'_'+label+'_ms']=percentile(values,q) if values else None
    c=[r['candidate_count'] for r in g if r['candidate_count'] is not None]
    s['candidate_mean']=statistics.mean(c) if c else None
    s['candidate_fraction_mean']=statistics.mean(c)/build['dataset']['corpus_count'] if c else None
    s['candidate_p95']=percentile(c,.95) if c else None
    for trial in sorted({r['trial'] for r in g}):
        s[f'trial_{trial}_p50_ms']=statistics.median(r['total_ns']/1e6 for r in g if r['trial']==trial)
    summary.append(s)
with (root/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
(root/'summary.json').write_text(json.dumps(summary,indent=2))
matches=[]
for target in [.5,.8,.9,.95,.99,1.0]:
    for method in ['plain','hnsw','centroid','affine','mlp']:
        tolerance = 0.0 if target == 1.0 else .02
        possible=[s for s in summary if s['method']==method and abs(s['recall_at_10']-target)<=tolerance]
        best=min(possible,key=lambda s:(abs(s['recall_at_10']-target),s['total_p50_ms'])) if possible else None
        matches.append({'target':target,'tolerance':tolerance,'method':method,'operating_point':best})
(root/'matched_recall.json').write_text(json.dumps(matches,indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'figure.dpi':160})
fig, axes=plt.subplots(1,2,figsize=(11,4.5))
colors={'plain':'tab:blue','hnsw':'tab:orange','centroid':'tab:green','affine':'tab:red','mlp':'tab:purple'}
for method in ['plain','hnsw','centroid','affine','mlp']:
    g=sorted((s for s in summary if s['method']==method),key=lambda s:s['recall_at_10'])
    axes[0].plot([s['recall_at_10'] for s in g],[s['total_p50_ms'] for s in g],'-o',label=method,color=colors[method],markersize=4)
    observed=[s for s in g if s['candidate_fraction_mean'] is not None]
    if observed: axes[1].plot([s['recall_at_10'] for s in observed],[s['candidate_fraction_mean'] for s in observed],'-o',label=method,color=colors[method],markersize=4)
axes[0].set_ylabel('Median measured search / harness time (ms)');axes[0].set_yscale('log')
axes[1].set_ylabel('Mean candidate fraction (HNSW unobserved)')
for ax in axes: ax.set_xlabel('Recall@10');ax.grid(alpha=.25);ax.legend()
fig.suptitle('LAION / Qdrant — controlled in-process evaluation; not HTTP latency')
fig.tight_layout();fig.savefig(root/'recall_tradeoffs.png');fig.savefig(root/'recall_tradeoffs.svg');plt.close(fig)

lines=['# Phase F measured operating points','',
    '| Method | Effort | Recall@10 | Candidates mean | p50 ms | p95 ms | p99 ms |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for s in summary:
    c='unavailable' if s['candidate_mean'] is None else f"{s['candidate_mean']:.0f}"
    lines.append(f"| {s['method']} | {s['effort']} | {s['recall_at_10']:.4f} | {c} | {s['total_p50_ms']:.2f} | {s['total_p95_ms']:.2f} | {s['total_p99_ms']:.2f} |")
lines+=['','## Approximately matched recall','',
    'Only measured operating points within ±0.02 absolute recall are admitted; no interpolation. Select closest recall, breaking ties by median latency. Target 1.0 requires measured recall exactly 1.0. Actual recall must still be compared; absence is not evidence of inability.', '',
    '| Target | Method | Effort | Actual recall | p50 ms |','| --- | --- | ---: | ---: | ---: |']
for m in matches:
    s=m['operating_point']
    lines.append(f"| {m['target']:.2f} | {m['method']} | "+(f"{s['effort']} | {s['recall_at_10']:.4f} | {s['total_p50_ms']:.2f} |" if s else '— | no point in band | — |'))
lines+=['','## Teacher diagnostics','',f"Held-out query teacher top-1 agreement (including warmup): {build['mlp_teacher_agreement']:.4f}.",
    f"Direct/affine query ordering mismatches: {build['centroid_affine_order_mismatches']}.",
    'Teacher agreement and coverage are diagnostics, not retrieval recall.','']
for name in ['mlp','centroid']:
    sizes=build[name+'_bucket_sizes']
    lines.append(f"{name}: {sum(v==0 for v in sizes)} empty buckets; min/median/max sizes {min(sizes)}/{statistics.median(sizes):.1f}/{max(sizes)}; largest fraction {max(sizes)/sum(sizes):.4f}.")
(root/'measured_tables.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))


