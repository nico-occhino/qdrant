"""Scientific plots from actual native per-query/case CSVs. No synthetic estimates."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
with (a.run/'cases.csv').open() as f:cases=list(csv.DictReader(f))
with (a.run/'per-query.csv').open() as f:queries=list(csv.DictReader(f))
colors={'SlotWeighted':'#4878cf','ReservedPerLevel':'#d65f5f','PopulationDensity':'#6acc64'}
for pass_name in ['validation','test_warm','test_cache_clear_attempt']:
    selected=[r for r in cases if r['pass_name']==pass_name]
    for x,label,out in [('mean_scored','Mean actually scored vectors','recall-work'),('p50_ms','Native query p50 (ms)','recall-latency')]:
        fig,ax=plt.subplots(figsize=(6,4))
        for policy,color in colors.items():
            rr=sorted((r for r in selected if r['policy']==policy),key=lambda r:float(r[x]))
            if not rr:continue
            ax.plot([float(r[x]) for r in rr],[float(r['recall10']) for r in rr],marker='o',label=policy,color=color)
            for r in rr:ax.annotate(r['budget'],(float(r[x]),float(r['recall10'])),fontsize=7)
        ax.set(xlabel=label,ylabel='Measured Recall@10',title=f'Release frozen10M — {pass_name}')
        ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(a.run/f'{out}-{pass_name}.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4))
for topk,marker in [('False','x'),('True','o')]:
    rr=[r for r in queries if r['pass_name']=='materialization_control' and r['topk']==topk]
    ax.scatter([float(r['scored']) for r in rr],[float(r['resolution_ms']) for r in rr],marker=marker,label='top-k' if topk=='True' else 'full',alpha=.6)
ax.set(xlabel='Actually scored vectors',ylabel='Candidate resolution (ms)',title='Same policy/budget: materialization control')
ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(a.run/'resolution-work.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4))
for pass_name,marker in [('test_cache_clear_attempt','x'),('test_warm','o')]:
    rr=[r for r in queries if r['pass_name']==pass_name]
    ax.scatter([float(r['major_faults']) for r in rr],[float(r['latency_ms']) for r in rr],marker=marker,label=pass_name,alpha=.5)
ax.set(xlabel='Process major faults during query',ylabel='Native query wall (ms)',title='Observed faults; cache-clear attempt ≠ uniformly cold corpus')
ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(a.run/'faults-latency.svg')
