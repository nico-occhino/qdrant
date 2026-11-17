"""Validation-only scalar calibration; frozen-model counterfactual, not native recall.

No teacher labels are claimed. Targets are gold-neighbor posting buckets.
Validation100:200; held-out200:300. No index/model is changed.
"""
import argparse,json,struct
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
base=Path('/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-25')
BUDGETS=sorted(set([2000,10000,20000,40000,80000,160000,320000,640000,1280000,2560000,5120000,10120191]+list(range(10000,100001,10000))+list(range(120000,400001,20000))+list(range(500000,1000001,100000))))
dp=base/'geometric-10120191/collection/storage/0/dli_hierarchy.json';desc=json.loads(dp.read_text())
query_rows=list(range(400,500))+list(range(600,700))
q=np.memmap(base/'data/queries.f32',mode='r',dtype='<f4',shape=(10000,768))[query_rows].copy();q/=np.sqrt((q*q).sum(axis=1,dtype=np.float32))[:,None]
gold=np.memmap(base/'data/ground_truth.i32',mode='r',dtype='<i4',shape=(10000,1000))[query_rows,:30].copy();flat=gold.ravel().astype(np.uint64)
models=[]
for uuid,level in desc['members']:
 if level==0:plain=uuid;continue
 path=dp.parent/'segments'/uuid;layers=[]
 with (path/'vector_index/lmi_router.bin').open('rb') as f:
  u64=lambda:struct.unpack('<Q',f.read(8))[0]
  vec=lambda:np.frombuffer(f.read(u64()*4),dtype='<f4').copy()
  for _ in range(u64()):
   kind=struct.unpack('<I',f.read(4))[0]
   if kind==0:
    ins,outs=u64(),u64();layers.append((vec().reshape(outs,ins),vec()))
   else:assert kind==1;layers.append(None)
 v=q
 for layer in layers:
  if layer is None:v=np.maximum(v,0);continue
  w,b=layer;acc=np.broadcast_to(b,(len(v),len(b))).copy()
  for j in range(w.shape[1]):acc+=v[:,j,None]*w[None,:,j]
  v=acc
 with (path/'vector_index/lmi_postings.bin').open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];bounds=np.frombuffer(f.read(n*8),dtype='<u8');count=struct.unpack('<Q',f.read(8))[0];posting=np.frombuffer(f.read(count*4),dtype='<u4')
 inverse=np.full(count,-1,dtype=np.int32)
 for bucket,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):inverse[posting[int(lo):int(hi)]]=bucket
 mpath=path/'id_tracker.mappings';n=int(np.fromfile(mpath,dtype='<u8',count=1)[0]);assert n==count and mpath.stat().st_size==8+13*n
 entries=np.memmap(mpath,mode='r',offset=8,dtype=[('kind','u1'),('id','<u8'),('offset','<u4')],shape=(n,));assert np.all(entries['kind']==0)
 order=np.argsort(entries['id']);ids=entries['id'][order];pos=np.searchsorted(ids,flat);valid=pos<n;valid[valid]&=ids[pos[valid]]==flat[valid]
 labels=np.full(len(flat),-1,dtype=np.int32);labels[valid]=inverse[entries['offset'][order[pos[valid]]]]
 models.append(dict(uuid=uuid,level=level,logits=v,lengths=np.diff(bounds),population=count,labels=labels.reshape(gold.shape)))
# Fit one shared temperature against validation gold bucket mass only.
def prob(logits,t):
 z=logits/t;z=z-z.max(axis=1,keepdims=True);p=np.exp(z);return p/p.sum(axis=1,keepdims=True)
losses={}
for t in [.5,.75,1.,1.25,1.5,2.,3.]:
 loss=0.;n=0
 for m in models:
  pp=prob(m['logits'][:100],t)
  for i,labels in enumerate(m['labels'][:100]):
   labels=labels[labels>=0];loss-=float(np.log(np.maximum(pp[i,labels],1e-30)).sum());n+=len(labels)
 losses[t]=loss/n
chosen_t=min(losses,key=losses.get)
counts=[int((m['labels'][:100]>=0).sum()) for m in models];plain_count=int((gold[:100]>10000000).sum());total=sum(counts)+plain_count
assert total==3000
priors=[(n+1)/(total+len(models)+1) for n in counts];plain_prior=(plain_count+1)/(total+len(models)+1)
np.savez_compressed(a.out/'logits.npz',**{m['uuid']:m['logits'] for m in models})
all_rows=[]
for method,t in [('slot',1.),('reserved',1.),('density',1.),('temperature_density',chosen_t),('temperature_population',chosen_t)]:
 pp=[prob(m['logits'],t) for m in models]
 for i in range(200):
  ranked=[]
  for mi,m in enumerate(models):
   prior=m['population']/10120191 if method!='temperature_population' else priors[mi]
   for bucket,(pval,length) in enumerate(zip(pp[mi][i],m['lengths'])):
    if length:ranked.append((float(pval)*2**m['level'] if method in ['slot','reserved'] else float(pval)*prior/int(length),mi,bucket,int(length)))
  prior=120191/10120191 if method!='temperature_population' else plain_prior
  ranked.append((1. if method in ['slot','reserved'] else prior/120191,-1,-1,120191));ranked.sort(key=lambda x:(-x[0],x[1],x[2]))
  for budget in BUDGETS:
   selected=set();count=0
   if method=='reserved':
    for mi in range(len(models)):
     item=max((x for x in ranked if x[1]==mi),key=lambda x:x[0]);selected.add((mi,item[2]));count+=item[3]
   for _,mi,bucket,length in ranked:
    if count>=budget:break
    if (mi,bucket) in selected:continue
    selected.add((mi,bucket));count+=length
   covered=[];scope=[];touched={x[0] for x in selected}
   for k in [10,30]:
    scope.append(sum(any(m['labels'][i,j]>=0 and mi in touched for mi,m in enumerate(models)) or (int(gold[i,j])>10000000 and (-1,-1) in selected) for j in range(k))/k)
    covered.append(sum(any(m['labels'][i,j]>=0 and (mi,int(m['labels'][i,j])) in selected for mi,m in enumerate(models)) or (int(gold[i,j])>10000000 and (-1,-1) in selected) for j in range(k))/k)
   all_rows.append(dict(query=query_rows[i],split='validation' if i<100 else 'test',method=method,budget=budget,candidates=count,bucket10=covered[0],bucket30=covered[1],scope10=scope[0],scope30=scope[1]))
# Budget selection is locked from validation only. Test curves are descriptive.
import csv
budgets=BUDGETS
methods=['slot','reserved','density','temperature_density','temperature_population']
summary=[];selected=[]
for method in methods:
 for split in ['validation','test']:
  for budget in budgets:
   rr=[r for r in all_rows if r['method']==method and r['split']==split and r['budget']==budget]
   summary.append(dict(method=method,split=split,budget=budget,**{k:float(np.mean([r[k] for r in rr])) for k in ['candidates','bucket10','bucket30','scope10','scope30']}))
 if method!='temperature_population':
  rr=[r for r in summary if r['method']==method and r['split']=='validation']
  for target in [.8,.9,.95]:
   good=[r for r in rr if r['bucket10']>=target and r['bucket30']>=target]
   pick=min(good,key=lambda r:r['candidates']) if good else rr[-1]
   policy={'slot':'SlotWeighted','reserved':'ReservedPerLevel','density':'PopulationDensity','temperature_density':{'TemperatureDensity':{'temperature_milli':int(round(chosen_t*1000))}}}[method]
   if not any(x['policy']==policy and x['budget']==pick['budget'] for x in selected):selected.append(dict(policy=policy,budget=pick['budget'],validation_target=target,validation=pick))
with (a.out/'frontier.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
with (a.out/'queries.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=all_rows[0].keys());w.writeheader();w.writerows(all_rows)
(a.out/'native-plan.json').write_text(json.dumps(selected,indent=2))
(a.out/'calibration.json').write_text(json.dumps(dict(validation_rows=[400,500],heldout_rows=[600,700],temperature=chosen_t,validation_nll_by_temperature=losses,priors=priors,plain_prior=plain_prior,scope='offline selected-posting work and bucket inclusion; not native recall or latency'),indent=2))
print(json.dumps(selected,indent=2))
