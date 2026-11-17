"""Plot descriptive offline inclusion and measured native frontiers separately."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('offline',type=Path);p.add_argument('native',type=Path);p.add_argument('teacher',type=Path);a=p.parse_args()
with (a.offline/'frontier.csv').open() as f:offline=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(6,4))
for method in ['slot','reserved','density','temperature_density','temperature_population']:
 rr=[r for r in offline if r['method']==method and r['split']=='test'];ax.plot([float(x['candidates']) for x in rr],[float(x['bucket30']) for x in rr],marker='.',label=method)
ax.set(xscale='log',xlabel='Selected posting offsets (offline)',ylabel='Held-out gold bucket coverage@30',title='Frozen10M:100 held-out queries, no native-latency inference');ax.legend(fontsize=7);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(a.native/'offline-frontier.svg');plt.close(fig)
with (a.native/'oracle-analysis/cases.csv').open() as f:cases=list(csv.DictReader(f))
for x,label,name in [('mean_scored','Actually scored vectors','recall-work'),('p50_ms','Warm query p50 (ms)','recall-latency'),('p95_ms','Warm query p95 (ms)','recall-p95')]:
 fig,ax=plt.subplots(figsize=(6,4))
 for policy in sorted({r['policy'] for r in cases if r['pass_name']=='test_warm'}):
  rr=sorted([r for r in cases if r['pass_name']=='test_warm' and r['policy']==policy],key=lambda r:float(r[x]));ax.plot([float(r[x]) for r in rr],[float(r['recall30']) for r in rr],marker='o',label=policy)
 ax.set(xlabel=label,ylabel='Measured held-out Recall@30',title='Frozen10M native release frontier');ax.legend(fontsize=7);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(a.native/(name+'.svg'));plt.close(fig)
if (a.teacher/'teacher-frontier.csv').exists():
 with (a.teacher/'teacher-frontier.csv').open() as f:rows=list(csv.DictReader(f))
 fig,ax=plt.subplots(figsize=(6,4))
 for placement in ['teacher','model']:
  for routing in ['teacher','model']:
   rr=[r for r in rows if r['placement']==placement and r['routing']==routing];ax.plot([float(r['candidates']) for r in rr],[float(r['bucket30']) for r in rr],marker='.',label=f'{placement} points / {routing} query')
 ax.set(xscale='log',xlabel='Selected subset candidates',ylabel='Exact subset-neighbor coverage@30',title='Private65K teacher/router control: not10M recall');ax.legend(fontsize=7);ax.grid(alpha=.3);fig.tight_layout();fig.savefig(a.teacher/'teacher-router-frontier.svg')
