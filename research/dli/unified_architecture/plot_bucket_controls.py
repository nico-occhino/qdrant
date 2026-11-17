import csv,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(sys.argv[1]);fig,ax=plt.subplots()
for k in (128,256,548):
 rows=list(csv.DictReader((base/f'buckets-{k}'/'teacher-frontier.csv').open()))
 for placement,routing,style in [('model','model','-'),('teacher','teacher','--')]:
  r=[x for x in rows if x['placement']==placement and x['routing']==routing]
  ax.plot([float(x['candidates']) for x in r],[float(x['bucket30']) for x in r],style,label=f'{k} {routing}')
ax.set_xscale('log');ax.set_xlabel('Selected posting entries (offline)');ax.set_ylabel('Exact-neighbor bucket coverage@30');ax.legend();ax.set_title('65K diagnostic: 548 is the earlier control');fig.tight_layout();fig.savefig(base/'coverage-work.svg')
