"""Aggregate completed native bucket controls; no inference-time gold use."""
import csv,json,re,struct,sys
from pathlib import Path
import numpy as np
base=Path(sys.argv[1]);rows=[]
for k in (128,256,548):
 p=base/f'buckets-{k}'; checks=json.loads((p/'teacher-checks.json').read_text());meta=json.loads((p/'index.json').read_text());index=Path(meta['path'])
 with (index/'vector_index/lmi_postings.bin').open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];b=np.frombuffer(f.read(n*8),dtype='<u8')
 sizes=np.diff(b).astype('float64');rr=list(csv.DictReader((p/'teacher-frontier.csv').open()));build=(p/'build.log').read_text();match=re.search(r'test result: ok.*finished in ([\d.]+)s',build);assert match
 for placement,routing in [('teacher','teacher'),('model','model')]:
  for target in (.9,.95):
   eligible=[r for r in rr if r['placement']==placement and r['routing']==routing and float(r['bucket10'])>=target and float(r['bucket30'])>=target]
   best=min(eligible,key=lambda r:float(r['candidates']))
   rows.append(dict(buckets=k,placement=placement,routing=routing,target=target,candidates=float(best['candidates']),coverage10=float(best['bucket10']),coverage30=float(best['bucket30']),train_agreement=checks['train_classifier_agreement'],heldout_agreement=checks['heldout_pool_classifier_agreement'],occupancy_min=float(sizes.min()),occupancy_max=float(sizes.max()),occupancy_cv=float(sizes.std()/sizes.mean()),empty_buckets=int((sizes==0).sum()),model_bytes=(index/'vector_index/lmi_router.bin').stat().st_size,control_wall_seconds=float(match[1])))
with (base/'bucket-summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
