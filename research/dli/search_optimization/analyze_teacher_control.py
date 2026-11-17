"""Small native-built teacher versus router factorial; no10M quality extrapolation."""
import argparse,json,struct,csv
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--budget-step',type=int,default=0);a=p.parse_args();r=a.run
budgets=[100,250,500,1000,2000,4000,8000,16000,32000,65536]
if a.budget_step:
 assert a.budget_step>0
 budgets=sorted(set(budgets+list(range(a.budget_step,16001,a.budget_step))))
teacher=json.loads((r/'teacher.json').read_text());centers=np.array(teacher['centers'],dtype=np.float32)
metadata=json.loads((r/'index.json').read_text());index=Path(metadata['path'])
ids=np.fromfile(r/'pool_ids.u64',dtype='<u8');raw=np.fromfile(r/'pool.f16',dtype='<f2').reshape(65536,768).astype(np.float32)
# Match native insertion preprocessing and Float16 storage. The resulting oracle
# is this controlled subset, never the official full10M gold file.
raw/=np.sqrt((raw*raw).sum(axis=1,dtype=np.float32))[:,None];data=raw.astype(np.float16).astype(np.float32)
queries=np.fromfile(r/'queries.f32',dtype='<f4').reshape(-1,768);queries/=np.sqrt((queries*queries).sum(axis=1,dtype=np.float32))[:,None]
mapping=np.memmap(index/'id_tracker.mappings',mode='r',offset=8,dtype=[('kind','u1'),('id','<u8'),('offset','<u4')],shape=(65536,));assert np.all(mapping['kind']==0)
external=np.empty(65536,dtype=np.uint64);external[mapping['offset']]=mapping['id'];assert np.array_equal(external,ids)
with (index/'vector_index/lmi_postings.bin').open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];bounds=np.frombuffer(f.read(n*8),dtype='<u8');count=struct.unpack('<Q',f.read(8))[0];posting=np.frombuffer(f.read(count*4),dtype='<u4')
assert count==65536 and len(np.unique(posting))==65536
model_labels=np.full(65536,-1,dtype=np.int32)
for b,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):model_labels[posting[int(lo):int(hi)]]=b
teacher_labels=np.empty(65536,dtype=np.int32)
for start in range(0,65536,1024):teacher_labels[start:start+1024]=(data[start:start+1024]@centers.T).argmax(axis=1)
offsets=np.array(json.loads((r/'teacher.offsets.json').read_text()),dtype=np.int64);trainmask=np.zeros(65536,dtype=bool);trainmask[offsets]=True;assert trainmask.sum()==32768
capture_labels=np.array(teacher['labels']);capture_agreement=float(np.mean(capture_labels==teacher_labels[offsets]))
with (index/'vector_index/lmi_router.bin').open('rb') as f:
 u64=lambda:struct.unpack('<Q',f.read(8))[0]
 vec=lambda:np.frombuffer(f.read(u64()*4),dtype='<f4').copy()
 v=queries.copy()
 for _ in range(u64()):
  kind=struct.unpack('<I',f.read(4))[0]
  if kind==0:
   ni,no=u64(),u64();w,bias=vec().reshape(no,ni),vec();out=np.broadcast_to(bias,(len(v),no)).copy()
   for j in range(ni):out+=v[:,j,None]*w[None,:,j]
   v=out
  else:assert kind==1;v=np.maximum(v,0)
model_rank=np.argsort(-v,axis=1,kind='stable');teacher_rank=np.argsort(-(queries@centers.T),axis=1,kind='stable')
# Exact normalized Float64 cosine on65K pool, bounded32x65K matrix.
d64=data.astype(np.float64);d64/=np.linalg.norm(d64,axis=1)[:,None]
q64=queries.astype(np.float64);q64/=np.linalg.norm(q64,axis=1)[:,None]
scores=q64@d64.T;gold=np.argsort(-scores,axis=1,kind='stable')[:,:30]
rows=[]
for placement,labels in [('teacher',teacher_labels),('model',model_labels)]:
 sizes=np.bincount(labels,minlength=len(centers))
 for routing,rank in [('teacher',teacher_rank),('model',model_rank)]:
  for qi in range(len(queries)):
   order=rank[qi];cumulative=np.cumsum(sizes[order]);position=np.argsort(order)
   for budget in budgets:
    n=min(len(order),int(np.searchsorted(cumulative,budget,side='left'))+1)
    rows.append(dict(placement=placement,routing=routing,query=qi+800,budget=budget,candidates=int(cumulative[n-1]),bucket10=float(np.mean(position[labels[gold[qi,:10]]]<n)),bucket30=float(np.mean(position[labels[gold[qi]]]<n))))
with (r/'teacher-query-frontier.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
summary=[]
for placement in ['teacher','model']:
 for routing in ['teacher','model']:
  for budget in budgets:
   rr=[x for x in rows if x['placement']==placement and x['routing']==routing and x['budget']==budget]
   summary.append(dict(placement=placement,routing=routing,budget=budget,**{k:float(np.mean([x[k] for x in rr])) for k in ['candidates','bucket10','bucket30']}))
with (r/'teacher-frontier.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
result=dict(population=65536,training_sample=32768,queries=len(queries),buckets=len(centers),model_config=metadata['model'],sample_capture_vs_blas_teacher_agreement=capture_agreement,train_classifier_agreement=float(np.mean(model_labels[trainmask]==teacher_labels[trainmask])),heldout_pool_classifier_agreement=float(np.mean(model_labels[~trainmask]==teacher_labels[~trainmask])),query_teacher_bucket_in_model_top={str(k):float(np.mean([teacher_rank[i,0] in model_rank[i,:k] for i in range(len(queries))])) for k in [1,4,8,16]},scope='controlled65K subset; exact local oracle, not official10M gold; not reconstruction of saved6M teacher; BLAS/native teacher boundary differences measured')
(r/'teacher-checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
