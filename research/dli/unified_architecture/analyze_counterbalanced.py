"""Paired warm scorer analysis. Does not infer cold or collection timings."""
import json,re,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);txt=p.read_text();assert 'test result: ok.' in txt
rows=[{k:float(v) for k,v in re.findall(r'(\w+)=([\d.eE+-]+)',s)} for s in txt.splitlines() if s.startswith('ABBA ')]
assert len(rows)==256 and {r['query'] for r in rows}==set(range(32))
summary={}
for mode in (0,1):
 a=[r for r in rows if bool(r['quantized'])==bool(mode)];assert len(a)==128
 summary[f'S{2 if mode else 0}']=dict(p50_ms=float(np.median([r['ns']/1e6 for r in a])),p95_ms=float(np.percentile([r['ns']/1e6 for r in a],95)),cpu_ticks=sum(r['cpu_ticks'] for r in a),minor=sum(r['minor'] for r in a),major=sum(r['major'] for r in a),min_recall10=min(r['recall10'] for r in a),min_recall30=min(r['recall30'] for r in a),original_inputs=sum(r['original'] for r in a),quantized_inputs=sum(r['quantized'] for r in a))
ratios=[]
for q in range(32):
 a=[r for r in rows if r['query']==q];assert len(a)==8
 ratios.append(np.median([r['ns'] for r in a if r['quantized']==0])/np.median([r['ns'] for r in a if r['quantized']>0]))
rng=np.random.default_rng(42);ci=np.percentile(np.median(rng.choice(ratios,size=(10000,32),replace=True),axis=1),[2.5,97.5])
summary.update(paired_ratio_median=float(np.median(ratios)),paired_ratio_bootstrap95=ci.tolist(),s2_winning_queries=int(sum(r>1 for r in ratios)),scope='warm-only scorer pipeline;32 reused diagnostic queries;four passes/mode; conditional oracle')
p.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
