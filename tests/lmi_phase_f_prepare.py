"""Dataset orchestration only; operational training stays in Rust/Qdrant."""
import argparse, hashlib, json, platform, subprocess
from pathlib import Path
import h5py
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--source',default='/home/nicoo/work/LearnedMetricIndex/laion2B-en-clip768v2-n=100K.h5')
p.add_argument('--output',required=True)
p.add_argument('--corpus',type=int,default=0)
p.add_argument('--queries',type=int,default=200)
p.add_argument('--warmup',type=int,default=20)
p.add_argument('--trials',type=int,default=2)
a=p.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
with h5py.File(a.source) as f: data=np.asarray(f['emb'],dtype='<f4')
assert np.isfinite(data).all()
seed=20260926
order=np.random.default_rng(seed).permutation(len(data))
q=order[:a.queries+a.warmup]
c=order[a.queries+a.warmup:]
if a.corpus: c=c[:a.corpus]
assert not set(c).intersection(q)
data[c].tofile(out/'corpus.f32');data[q].tofile(out/'queries.f32')
meta={'source':a.source,'source_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),
    'source_shape':list(data.shape),'source_dtype':'float16','export_dtype':'little-endian float32',
    'dimension':data.shape[1],'corpus_count':len(c),'query_count':a.queries,'warmup':a.warmup,'trials':a.trials,
    'split_seed':seed,'generator':'numpy.default_rng / PCG64','metric':'Cosine','k':10,
    'corpus_source_rows':c.tolist(),'query_source_rows':q.tolist(),
    'hardware':subprocess.check_output(['lscpu'],text=True),'platform':platform.platform()}
(out/'dataset.json').write_text(json.dumps(meta,indent=2))
print({k:v for k,v in meta.items() if k not in ['corpus_source_rows','query_source_rows','hardware']})
