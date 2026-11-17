"""Summarize native log; conditional recall versus identical selected offsets."""
import re,json,sys,numpy as np
from pathlib import Path
p=Path(sys.argv[1]);text=p.read_text();assert 'test result: ok.' in text
s0={int(q):int(ns) for q,ns in re.findall(r'S0 query=(\d+) original=\d+ wall_ns=(\d+)',text)}
rows=[]
for line in text.splitlines():
 if not line.startswith('NATIVE_SCALAR '):continue
 d=dict(re.findall(r'(\w+)=([\d.eE+-]+)',line));d={k:float(v) for k,v in d.items()};rows.append(d)
per_query=len([r for r in rows if r['query']==rows[0]['query']]); limits=[64,128,256,512,1024,2048,65536] if per_query==7 else [128,512,2048,65536]
assert per_query==len(limits)
for i,r in enumerate(rows):r['requested']=limits[i%per_query]
summary=[]
for count in limits:
 rr=[r for r in rows if r['requested']==count];ns=[r['execution_ns']/1e6 for r in rr]
 summary.append(dict(requested_shortlist=count,mean_original_inputs=float(np.mean([r["original"] for r in rr])),queries=len(rr),mean_quantized_inputs=float(np.mean([r['quantized'] for r in rr])),conditional_recall30=float(np.mean([r['recall30'] for r in rr])),s1_recall30=float(np.mean([r['s1_recall30'] for r in rr])),p50_ms=float(np.median(ns)),p95_ms=float(np.percentile(ns,95)),s0_p50_ms=float(np.median([s0[int(r['query'])]/1e6 for r in rr]))))
print(json.dumps(summary,indent=2));p.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2))
