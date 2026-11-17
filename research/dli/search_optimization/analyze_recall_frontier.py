"""Native executed frontier with exact-control separation and shared strict oracle."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
cases=json.loads((run/'frontier.json').read_text())
queries=[json.loads(x) for x in (run/'query-events.jsonl').read_text().splitlines()]
routes=[json.loads(x) for x in (run/'routing-events.jsonl').read_text().splitlines()]
normal=[c for c in cases if not c.get('exact',False)];exact=[c for c in cases if c.get('exact',False)]
assert len(exact)==1 and cases[-1]==exact[0]
label=lambda p:f"TemperatureDensityT{p['TemperatureDensity']['temperature_milli']/1000:g}" if isinstance(p,dict) else p
for c in normal:c['policy']=label(c['policy'])
for r in routes:r['policy']=label(r['policy'])
# Derived metadata/events only; preserve all original raw logs.
out=run/'oracle-analysis';out.mkdir(exist_ok=True)
(out/'study.json').write_text(json.dumps(normal));(out/'locked-selection.json').write_text(json.dumps({'validation_rows':[400,500],'test_rows':[600,700]}))
(out/'query-events.jsonl').write_text('\n'.join(json.dumps(q) for q in queries[:exact[0]['ordinal_begin']])+'\n')
(out/'routing-events.jsonl').write_text('\n'.join(json.dumps(q) for q in routes)+'\n')
path=out/'access-events.jsonl'
if not path.exists():path.symlink_to((run/'access-events.jsonl').resolve())
subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_matched_study.py')),str(out)],check=True)
with (out/'cases.csv').open() as f:summaries=list(csv.DictReader(f))
with (out/'per-query.csv').open() as f:rows=list(csv.DictReader(f))
# Cache/uncached comparisons: same actual work, sets, scores, versions.
control=[c for c in normal if c['pass']=='priority_control'];ref=control[0];comparisons=0
for c in control[1:]:
 for i in range(c['query_count']):
  aa=queries[ref['ordinal_begin']+i];bb=queries[c['ordinal_begin']+i]
  assert aa['scored_vectors']==bb['scored_vectors']
  assert {h['id']:(h['score'],h['version']) for h in aa['hits']}=={h['id']:(h['score'],h['version']) for h in bb['hits']}
  comparisons+=1
# Native exact physically scores full population: report its official-gold recall.
e=exact[0];base=Path('/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-25/data')
vectors=np.memmap(base/'vectors.f16',mode='r',dtype='<f2',shape=(10120191,768));qvec=np.memmap(base/'queries.f32',mode='r',dtype='<f4',shape=(10000,768));maxerr=0.
for i in range(e['query_count']):
 q=queries[e['ordinal_begin']+i];assert q['scored_vectors']==10120191
 query=qvec[e['query_start']+i].astype(np.float64);query/=np.linalg.norm(query)
 for h in q['hits']:
  assert h['version']==(int(h['id'])-1)//250000+1
  v=vectors[int(h['id'])-1].astype(np.float64);v/=np.linalg.norm(v)
  error=abs(float(h['score'])-float(v@query));maxerr=max(maxerr,error);assert error<.0005
# Descriptive matching, no parameter selection from test results.
warm=[s for s in summaries if s['pass_name']=='test_warm'];matches=[]
for i,x in enumerate(warm):
 for y in warm[i+1:]:
  if x['policy']==y['policy']:continue
  work=max(float(x['mean_scored']),float(y['mean_scored']))/min(float(x['mean_scored']),float(y['mean_scored']))-1
  r10=abs(float(x['recall10'])-float(y['recall10']));r30=abs(float(x['recall30'])-float(y['recall30']))
  if work<=.10 or (r10<=.03 and r30<=.03):matches.append(dict(a=x['case'],b=y['case'],work_spread=work,recall10_difference=r10,recall30_difference=r30,approximately_work_matched=work<=.10,approximately_recall_matched=r10<=.03 and r30<=.03))
(run/'frontier-checks.json').write_text(json.dumps(dict(learned_queries=sum(c['query_count'] for c in normal),deliberate_exact_controls=e['query_count'],priority_identical_pairs=comparisons,exact_control=e,exact_max_cosine_error=maxerr,matching_pairs=matches,scope='pilot; matching is descriptive, configurations locked from validation before test'),indent=2))
print('learned',sum(c['query_count'] for c in normal),'exact',e['query_count'],'priority pairs',comparisons)
for s in warm:print(s['policy'],s['budget'],s['mean_scored'],s['recall10'],s['recall30'],s['p50_ms'],s['p95_ms'])
