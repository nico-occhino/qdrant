"""Diagnostic gold-aware bucket oracle; never an inference policy."""
import json,struct,csv,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);o=Path(sys.argv[2]);meta=json.loads((p/'index.json').read_text());idx=Path(meta['path']);x=np.fromfile(p/'pool.f16',dtype='<f2').reshape(-1,768).astype('float32');q=np.fromfile(p/'queries.f32',dtype='<f4').reshape(-1,768)
x/=np.linalg.norm(x,axis=1)[:,None];q/=np.linalg.norm(q,axis=1)[:,None]
with (idx/'vector_index/lmi_postings.bin').open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];b=np.frombuffer(f.read(n*8),dtype='<u8');nn=struct.unpack('<Q',f.read(8))[0];ids=np.frombuffer(f.read(nn*4),dtype='<u4')
labels=np.empty(len(x),dtype='int32')
for k in range(n-1):labels[ids[int(b[k]):int(b[k+1])]]=k
sizes=np.diff(b).astype('int64');gold=np.argsort(-(q@x.T),axis=1,kind='stable')[:,:30];rows=[]
for qi,g in enumerate(gold):
 hits=np.bincount(labels[g],minlength=n-1);order=np.argsort(-hits/np.maximum(sizes,1),kind='stable');counts=np.cumsum(hits[order]);work=np.cumsum(sizes[order])
 for target in (.9,.95,1.):
  z=int(np.searchsorted(counts,int(np.ceil(30*target))));rows.append(dict(query=qi+800,target=target,buckets=z+1,candidates=int(work[z]),covered=int(counts[z])))
with (o/'partition-oracle.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for t in (.9,.95,1.):print(t,np.mean([r['candidates'] for r in rows if r['target']==t]))
