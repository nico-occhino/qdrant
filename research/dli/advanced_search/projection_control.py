"""Bounded exact-oracle random projection control; no production index changes."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]); x=np.fromfile(p/'pool.f16',dtype='<f2').reshape(-1,768).astype('float32'); q=np.fromfile(p/'queries.f32',dtype='<f4').reshape(-1,768)
x/=np.linalg.norm(x,axis=1)[:,None]; q/=np.linalg.norm(q,axis=1)[:,None]
t=time.perf_counter(); exact=q@x.T; oracle=np.argsort(-exact,axis=1,kind='stable')[:,:30]; oracle_time=time.perf_counter()-t
rows=[]
for dim in (64,128,256):
 rng=np.random.default_rng(42+dim); projection=rng.standard_normal((768,dim),dtype=np.float32)/np.sqrt(dim)
 t=time.perf_counter(); xp=(x@projection).astype('float16'); build=time.perf_counter()-t
 t=time.perf_counter(); qp=q@projection; approx=qp@xp.astype('float32').T; approximate=time.perf_counter()-t
 for k in (512,2048,8192,16384):
  t=time.perf_counter(); selected=np.argpartition(-approx,k-1,axis=1)[:,:k]; scores=np.take_along_axis(exact,selected,axis=1); rerank=np.take_along_axis(selected,np.argsort(-scores,axis=1,kind='stable')[:,:30],axis=1); rerank_time=time.perf_counter()-t
  recall=lambda n:float(np.mean([len(set(rerank[i,:n])&set(oracle[i,:n]))/n for i in range(len(q))]))
  rows.append(dict(dim=dim,shortlist=k,recall10=recall(10),recall30=recall(30),projected_bytes=xp.nbytes,build_seconds=build,approx_seconds_32_queries=approximate,rerank_from_precomputed_exact_seconds=rerank_time))
with (Path(sys.argv[2])/'projection-control.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps({'population':len(x),'queries':len(q),'oracle_seconds':oracle_time,'rows':rows},indent=2))
