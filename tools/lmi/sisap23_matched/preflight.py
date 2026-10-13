#!/usr/bin/env python3
"""Compute-node SISAP 2023 input verification and preparation."""
import argparse, hashlib, json, os
from pathlib import Path
EXPECTED = {
 "corpus": ("laion2B-en-clip768v2-n=10M.h5","c05e4b1d2b2a0c7663ac9767753e25e1",(10120191,768),"float16","emb"),
 "queries": ("public-queries-10k-clip768v2.h5","257b9eb3f7f25776e0d33b22451b7b32",(10000,768),"float32","emb"),
 "gold": ("laion2B-en-public-gold-standard-v2-10M-F64-IEEE754.h5","45b05e4d60b8a66088b378ae7e0d278f",(10000,1000),"int32","knns")}
N,DIM,SEED=10120191,768,20261013
def digest(path,alg):
 h=hashlib.new(alg)
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
def write_array(path,a,dtype):
 with open(path,"xb") as f: a.astype(dtype,copy=False).tofile(f); f.flush(); os.fsync(f.fileno())
 return {"path":path.name,"bytes":path.stat().st_size,"sha256":digest(path,"sha256")}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
 if a.out.exists(): raise SystemExit(f"refusing existing output directory: {a.out}")
 try: import h5py,numpy as np
 except ImportError as e: raise SystemExit("Python requires h5py and numpy") from e
 a.out.mkdir(parents=True); sources={}
 for key,(name,md5,shape,dtype,dataset) in EXPECTED.items():
  p=a.source/name
  if not p.is_file(): raise SystemExit(f"missing input: {p}")
  got=digest(p,"md5")
  if got!=md5: raise SystemExit(f"{key} MD5 mismatch: expected {md5}, got {got}")
  with h5py.File(p,"r") as hf:
   ds=hf[dataset]
   if tuple(ds.shape)!=shape or np.dtype(ds.dtype).name!=dtype: raise SystemExit(f"{key} shape/dtype mismatch: {ds.shape} {ds.dtype}")
  sources[key]={"file":name,"dataset":dataset,"shape":list(shape),"dtype":dtype,"md5":got,"sha256":digest(p,"sha256")}
 outputs={}
 with h5py.File(a.source/EXPECTED["corpus"][0],"r") as hf:
  ds=hf["emb"]; p=a.out/"corpus.f16le"; h=hashlib.sha256()
  with open(p,"xb") as f:
   for s in range(0,N,8192):
    raw=np.asarray(ds[s:min(s+8192,N)],dtype="<f2").tobytes(order="C"); f.write(raw); h.update(raw)
   f.flush(); os.fsync(f.fileno())
  outputs["corpus"]={"path":p.name,"bytes":p.stat().st_size,"sha256":h.hexdigest()}
 with h5py.File(a.source/EXPECTED["queries"][0],"r") as hf:
  ds=hf["emb"]
  outputs["warmup_queries"]=write_array(a.out/"queries_warmup.f32le",np.asarray(ds[:20]),"<f4")
  outputs["queries"]=write_array(a.out/"queries_20_9999.f32le",np.asarray(ds[20:10000]),"<f4")
  rows=np.sort(np.random.default_rng(SEED).choice(np.arange(20,10000),1000,replace=False)).astype("<u4")
  outputs["panel_rows"]=write_array(a.out/"panel_rows.u32le",rows,"<u4")
 with h5py.File(a.source/EXPECTED["gold"][0],"r") as hf:
  outputs["official_gold_top10"]=write_array(a.out/"gold_20_9999_top10.i32le",np.asarray(hf["knns"][20:10000,:10]),"<i4")
 manifest={"format":"sisap23-matched-v1","qdrant_commit":"97c20f5f0bd7d65239970285a848706f39458c84","dataset":{"corpus_count":N,"dimension":DIM,"query_rows":[20,9999],"measured_queries":9980,"metric":"Cosine","storage":"Float16","external_id_mapping":"corpus row i has one-based ID i+1"},"sources":sources,"prepared":outputs,"panel_seed":SEED,"note":"All variants ingest this identical little-endian Float16 corpus file."}
 with open(a.out/"manifest.json","x",encoding="utf-8") as f: json.dump(manifest,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
 print(json.dumps({"manifest":str(a.out/"manifest.json"),"prepared":outputs},indent=2))
if __name__=="__main__": main()
