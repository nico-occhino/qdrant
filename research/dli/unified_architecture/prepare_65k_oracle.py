"""Exact cosine oracle for the frozen native 65K pool; writes new artifacts only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('pool',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
paths=[a.pool/'pool.f16',a.pool/'pool_ids.u64',a.pool/'queries.f32']
ids=np.fromfile(paths[1],dtype='<u8');assert len(ids)==65536 and len(np.unique(ids))==len(ids)
x=np.memmap(paths[0],dtype='<f2',mode='r',shape=(len(ids),768));q=np.fromfile(paths[2],dtype='<f4').reshape(-1,768).astype('float64');q/=np.linalg.norm(q,axis=1)[:,None]
# Reproduce native cosine preprocessing followed by Float16 storage.
scores=np.empty((len(q),len(ids)),dtype='float64')
for start in range(0,len(ids),2048):
 v=x[start:start+2048].astype('float32');v/=np.linalg.norm(v,axis=1)[:,None];v=v.astype('float16').astype('float64');v/=np.linalg.norm(v,axis=1)[:,None]
 scores[:,start:start+len(v)]=q@v.T
# Stable external-ID ordering breaks exact ties deterministically.
order=np.lexsort((np.broadcast_to(ids,scores.shape),-scores),axis=1)[:,:30]
gold=ids[order]; np.save(a.output/'gold_ids.npy',gold);np.save(a.output/'gold_cosines.npy',np.take_along_axis(scores,order,axis=1))
hashfile=lambda path:hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
manifest={'population':len(ids),'query_count':len(q),'dimensions':768,'input_hashes':{str(x):hashfile(x) for x in paths},'outputs':{x.name:hashfile(x) for x in a.output.glob('*.npy')},'oracle':'Float32 cosine preprocessing, Float16 stored vectors, Float64 renormalized cosine; ties by external ID','scope':'exact local pool only; native Float16 scoring near ties may differ'}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
