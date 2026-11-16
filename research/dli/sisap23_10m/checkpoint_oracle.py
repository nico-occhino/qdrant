"""Exact blocked cosine top1000 for first100 official queries at a checkpoint.
Unused query rows are zero and must not be queried. Memory independent of N.
"""
import json,sys
from pathlib import Path
import numpy as np
source,output=map(Path,sys.argv[1:3]);n=int(sys.argv[3]);output.mkdir(parents=True,exist_ok=True)
for name in ['vectors.f16','queries.f32']:
 if not (output/name).exists():(output/name).symlink_to(source/name)
q=np.memmap(source/'queries.f32',dtype='<f4',mode='r',shape=(10000,768))[:100].astype('float64')
q/=np.linalg.norm(q,axis=1,keepdims=True)
v=np.memmap(source/'vectors.f16',dtype='<f2',mode='r',shape=(10120191,768))
best=np.empty((100,0));ids=np.empty((100,0),dtype=np.int32)
for first in range(0,n,8192):
 x=v[first:min(n,first+8192)].astype('float64');x/=np.linalg.norm(x,axis=1,keepdims=True)
 scores=q@x.T
 candidates=np.concatenate([best,scores],axis=1)
 cand_ids=np.concatenate([ids,np.broadcast_to(np.arange(first+1,first+len(x)+1,dtype=np.int32),(100,len(x)))],axis=1)
 keep=np.argpartition(-candidates,999,axis=1)[:,:1000]
 best=np.take_along_axis(candidates,keep,axis=1);ids=np.take_along_axis(cand_ids,keep,axis=1)
order=np.argsort(-best,axis=1,kind='stable');ids=np.take_along_axis(ids,order,axis=1)
g=np.memmap(output/'ground_truth.i32',dtype='<i4',mode='w+',shape=(10000,1000));g[:]=0;g[:100]=ids;g.flush()
(output/'oracle.json').write_text(json.dumps(dict(population=n,query_rows=100,precision='float64',metric='normalized cosine',block_rows=8192,unused_rows='zero; do not query'),indent=2))
print('oracle complete',n,flush=True)
