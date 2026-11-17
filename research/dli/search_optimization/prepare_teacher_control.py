"""Prepare one deterministic64K private subset; never copies/rebuilds10M index."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=Path('/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-25/data')
# Restrict to original6M learned member's population, but do not claim its teacher is reproduced.
ids=np.sort(np.random.default_rng(42).choice(6000000,size=65536,replace=False)).astype('<u8')
v=np.memmap(base/'vectors.f16',mode='r',dtype='<f2',shape=(10120191,768))
v[ids].tofile(a.output/'pool.f16');(ids+1).tofile(a.output/'pool_ids.u64')
q=np.memmap(base/'queries.f32',mode='r',dtype='<f4',shape=(10000,768));q[800:832].tofile(a.output/'queries.f32')
manifest={'population':65536,'source_population':6000000,'ids':'one-based original point IDs; sampled without replacement','seed':42,'query_rows':[800,832],'training_sample':32768,'buckets':548,'hidden':512,'epochs':30,'kmeans_iterations':5,'scope':'small controlled subset, new exact subset oracle; no official10M-gold reuse','sha256':{name:hashlib.sha256((a.output/name).read_bytes()).hexdigest() for name in ['pool.f16','pool_ids.u64','queries.f32']}}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
