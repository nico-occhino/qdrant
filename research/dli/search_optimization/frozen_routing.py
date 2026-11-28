"""Read-only frozen geometric routing reconstruction; no collection mutation or training.

Uses retained continuation-25 numeric mappings and verifies all 100 baseline
native score-count/call observations before interpreting counterfactual coverage.
Bucket coverage is an oracle inclusion statistic, not native recall or latency.
Only insertion-only numeric-ID populations in this artifact are supported.
"""
import json,struct,math,statistics,csv,hashlib
import os
from pathlib import Path
import numpy as np
root=Path(os.environ['ADMIN'], 'experiments/2026-11-12-dli-architecture')
r=root/'continuation-25/geometric-10120191';out=root/'continuation-26/geometric-routing';out.mkdir(parents=True,exist_ok=True)
s=json.loads((r/'summary.json').read_text());q=[json.loads(l) for l in (r/'query-events.jsonl').read_text().splitlines()];e=[json.loads(l) for l in (r/'native-events.jsonl').read_text().splitlines()];b=[json.loads(l) for l in (r/'build-events.jsonl').read_text().splitlines()]
assert len(q)==100 and [x['query'] for x in q]==list(range(100))
assert sum(x['scored_vectors'] for x in q)==s['scored_vectors']==s['dli_attribution'][2]
assert sum(x['scorer_calls'] for x in q)==s['scorer_calls']
assert all(x['returned']==30 for x in q)
models=[]
descriptor=next((r/'collection').rglob('dli_hierarchy.json'));d=json.loads(descriptor.read_text())
for uuid,level in d['members']:
 if not level:continue
 directory=descriptor.parent/'segments'/uuid/'vector_index'
 with (directory/'lmi_postings.bin').open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];bounds=struct.unpack('<'+'Q'*n,f.read(n*8));count=struct.unpack('<Q',f.read(8))[0]
 layers=[]
 with (directory/'lmi_router.bin').open('rb') as f:
  def u64():return struct.unpack('<Q',f.read(8))[0]
  def vec():return np.frombuffer(f.read(u64()*4),dtype='<f4').copy()
  for _ in range(u64()):
   kind=struct.unpack('<I',f.read(4))[0]
   if kind==0:
    ins,outs=u64(),u64();w=vec().reshape(outs,ins);bias=vec();layers.append((w,bias))
   else:assert kind==1;layers.append(None)
  assert not f.read(1)
 lens=np.diff(bounds);models.append(dict(uuid=uuid,level=level,lengths=lens,layers=layers,count=count))
queries=np.memmap(root/'continuation-25/data/queries.f32',mode='r',dtype='<f4',shape=(10000,768))[:100].copy()
queries/=np.sqrt(np.sum(queries*queries,axis=1,dtype=np.float32))[:,None]
for model in models:
 v=queries
 for layer in model['layers']:
  if layer is None:v=np.maximum(v,0);continue
  w,bias=layer;acc=np.broadcast_to(bias,(len(v),len(bias))).copy()
  # Same ordered Float32 accumulation as native forward; no vector search.
  for j in range(w.shape[1]):acc+=v[:,j,None]*w[None,:,j]
  v=acc
 model['logits']=v.copy()
 v-=np.max(v,axis=1,keepdims=True);p=np.exp(v);p/=np.sum(p,axis=1,keepdims=True);model['p']=p
# Decode authoritative numeric external/internal mappings (immutable tracker format).
# No assumption that posting offset equals external ID.
gold=np.memmap(root/'continuation-25/data/ground_truth.i32',mode='r',dtype='<i4',shape=(10000,1000))[:100,:30].copy()
flat=gold.ravel().astype(np.uint64)
gold_bucket={}
for m in models:
 directory=descriptor.parent/'segments'/m['uuid']
 path=directory/'id_tracker.mappings'
 with path.open('rb') as f:count=struct.unpack('<Q',f.read(8))[0]
 assert path.stat().st_size==8+13*count
 entries=np.memmap(path,mode='r',offset=8,dtype=np.dtype([('kind','u1'),('id','<u8'),('offset','<u4')]),shape=(count,))
 assert np.all(entries['kind']==0)
 order=np.argsort(entries['id']);ids=entries['id'][order]
 positions=np.searchsorted(ids,flat);valid=positions<count
 valid[valid] &= ids[positions[valid]]==flat[valid]
 with (directory/'vector_index/lmi_postings.bin').open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];bounds=np.frombuffer(f.read(n*8),dtype='<u8').copy();length=struct.unpack('<Q',f.read(8))[0]
  posting=np.frombuffer(f.read(length*4),dtype='<u4').copy()
 inverse=np.full(count,-1,dtype=np.int32)
 assert np.max(posting)<count and len(np.unique(posting))==count
 for bucket,(a,z) in enumerate(zip(bounds[:-1],bounds[1:])):inverse[posting[int(a):int(z)]]=bucket
 mapped=np.full(len(flat),-1,dtype=np.int32)
 mapped[valid]=inverse[entries['offset'][order[positions[valid]]]]
 gold_bucket[m['uuid']]=mapped.reshape(gold.shape)
# Ingestion lineage is verified against native destination counts, not offset order.
lineage={}
for i,event in enumerate(e,1):
 assert event['outcome']=='published'
 batches=set().union(*(lineage.get(u,set()) for u in event['sources']))|{i}
 assert len(batches)*250000==event['destination_points']==event['source_live_points']
 lineage[event['destination']]=batches
owner={batch:m['uuid'] for m in models for batch in lineage[m['uuid']]}
assert len(owner)==40
gold=np.memmap(root/'continuation-25/data/ground_truth.i32',mode='r',dtype='<i4',shape=(10000,1000))[:100]
rows=[]
for i in range(100):
 predictions=[]
 for m in models:
  predictions += [(float(p),m['level'],m['uuid'],bucket,int(n),m['count']) for bucket,(p,n) in enumerate(zip(m['p'][i],m['lengths'])) if n]
 plain=(1.,0,'plain',-1,120191,120191)
 for policy in ['slot','reserved','population_density']:
  for budget in [2000,10000,20000,30000,60000,120000]:
   def priority(x):
    if policy in ['slot','reserved']:return x[0]*2**x[1]
    # Expected neighborhood mass per scored vector, population prior.
    return x[0]*x[5]/max(1,x[4])
   ranked=sorted(predictions+[plain],key=lambda x:(-priority(x),x[1],x[2],x[3]))
   selected=[]
   if policy=='reserved':
    # One best whole bucket per learned member. This has a potentially large cost floor.
    selected=[max((x for x in predictions if x[2]==m['uuid']),key=lambda x:x[0]) for m in models]
   count=sum(x[4] for x in selected)
   for x in ranked:
    if count>=budget:break
    if x not in selected:selected.append(x);count+=x[4]
   grouped={}
   for x in selected:grouped[x[2]]=grouped.get(x[2],0)+x[4]
   touched=set(grouped)
   ceilings=[sum(owner.get((int(g)-1)//250000+1,'plain') in touched for g in gold[i,:k])/k for k in [10,30]]
   rows.append(dict(query=i,policy=policy,budget=budget,candidates=count,buckets=len(selected),models=len(touched-{'plain'}),scope10=ceilings[0],scope30=ceilings[1],count_match=count==q[i]['scored_vectors'] if policy=='slot' and budget==2000 else None,calls_match=sum(math.ceil(n/64) for n in grouped.values())==q[i]['scorer_calls'] if policy=='slot' and budget==2000 else None,baseline_latency_ms=q[i]['latency_ms'],bucket10=sum(any(gold_bucket[u][i,j]==bucket for _,_,u,bucket,_,_ in selected if u!='plain') or (int(gold[i,j])>10000000 and 'plain' in touched) for j in range(10))/10,bucket30=sum(any(gold_bucket[u][i,j]==bucket for _,_,u,bucket,_,_ in selected if u!='plain') or (int(gold[i,j])>10000000 and 'plain' in touched) for j in range(30))/30,cold_selected='44bee22d-72cb-4b49-a175-76b3e54d9dce' in touched,selected=';'.join(f'{x[2]}:{x[3]}:{x[4]}' for x in selected)))
with (out/'routing.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
np.savez_compressed(out/'probabilities.npz',**{m['uuid']+suffix:m[field] for m in models for suffix,field in [('_probabilities','p'),('_logits','logits')]})
stats={}
for policy in ['slot','reserved','population_density']:
 for budget in [2000,10000,20000,30000,60000,120000]:
  rr=[x for x in rows if x['policy']==policy and x['budget']==budget]
  key=f'{policy}:{budget}'
  stats[key]={k:float(np.mean([x[k] for x in rr])) for k in ['candidates','buckets','models','scope10','scope30','bucket10','bucket30','cold_selected']}
  if policy=='slot' and budget==2000:stats[key].update(count_matches=sum(x['count_match'] for x in rr),calls_matches=sum(x['calls_match'] for x in rr))
stats['members']=[dict(uuid=m['uuid'],level=m['level'],population=m['count'],median_bucket=float(np.median(m['lengths'])),min_bucket=int(min(m['lengths'])),max_bucket=int(max(m['lengths']))) for m in models]
assert stats['slot:2000']['count_matches']==100 and stats['slot:2000']['calls_matches']==100
model_stats=[]
for m in models:
 ranks=np.argsort(np.argsort(-m['p'],axis=1),axis=1)
 covered=gold_bucket[m['uuid']]>=0
 rr=np.zeros_like(gold_bucket[m['uuid']])
 for i in range(100):
  mask=covered[i];rr[i,mask]=ranks[i,gold_bucket[m['uuid']][i,mask]]
 model_stats.append(dict(uuid=m['uuid'],level=m['level'],population=m['count'],mean_entropy=float(np.mean(-np.sum(m['p']*np.log(np.maximum(m['p'],1e-30)),axis=1))),mean_max_probability=float(np.mean(np.max(m['p'],axis=1))),gold_neighbours_in_member=int(covered.sum()),**{f'conditional_gold_bucket_top{k}':float(np.sum(covered&(rr<k))/covered.sum()) for k in [1,4,8]}))
(out/'model-diagnostic.json').write_text(json.dumps(model_stats,indent=2))
(out/'summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
