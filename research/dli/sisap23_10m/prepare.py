"""Validate official 2023 inputs and extract bounded raw records for Rust.
Never trusts the 10M filename as a row count. No whole-corpus RAM allocation.
"""
import hashlib, json, sys
from pathlib import Path
import h5py
import numpy as np
source, output = map(Path, sys.argv[1:3]); output.mkdir(parents=True, exist_ok=True)
files = [('laion2B-en-clip768v2-n=10M.h5','c05e4b1d2b2a0c7663ac9767753e25e1','vectors.f16','<f2'),
 ('public-queries-10k-clip768v2.h5','257b9eb3f7f25776e0d33b22451b7b32','queries.f32','<f4'),
 ('laion2B-en-public-gold-standard-v2-10M-F64-IEEE754.h5','45b05e4d60b8a66088b378ae7e0d278f','ground_truth.i32','<i4')]
manifest = {'edition':2023,'metric':'cosine','ids':'one-based row index','files':[]}
for name, expected, destination, dtype in files:
 path=source/name; digest=hashlib.md5()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):digest.update(b)
 assert digest.hexdigest()==expected, (name,digest.hexdigest())
 with h5py.File(path,'r') as h:
  datasets={}
  h.visititems(lambda k,v: datasets.update({k:v}) if isinstance(v,h5py.Dataset) else None)
  key='knns' if destination.startswith('ground') else ('emb' if 'emb' in datasets else next(iter(datasets)))
  d=datasets[key]
  shape=list(d.shape); transposed=destination.startswith('ground') and shape[0]==1000
  normalized_shape=shape[::-1] if transposed else shape
  if destination=='vectors.f16':assert shape[1]==768 and d.dtype==np.float16
  target=output/destination
  if not target.exists():
   tmp=target.with_suffix(target.suffix+'.partial')
   with tmp.open('wb') as f:
    for first in range(0,normalized_shape[0],4096):
     block=d[:,first:first+4096].T if transposed else d[first:first+4096]
     f.write(np.asarray(block,dtype=dtype).tobytes(order='C'))
   tmp.rename(target)
  assert target.stat().st_size==int(np.prod(normalized_shape))*np.dtype(dtype).itemsize
  manifest['files'].append(dict(source=str(path),md5=expected,hdf5_key=key,shape=shape,normalized_shape=normalized_shape,dtype=str(d.dtype),raw=str(target)))
  (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
  print(name,shape,d.dtype,'verified/extracted',flush=True)
