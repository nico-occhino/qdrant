from pathlib import Path
import json, hashlib, random, subprocess, os
import numpy as np

base=Path('/home/nicoo/work/lmi-phase-f-data')
f2=Path('/home/nicoo/work/lmi-phase-f2-data')
out=Path(os.environ.get('LMI_PHASE_F3_DIR','/home/nicoo/work/lmi-phase-f3-data'))
assert not out.exists(), 'Preserve completed/partial F.3; do not overwrite'
out.mkdir()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def dump(p,x):
    if p.name=='protocol.json':
        x['tie_rule']='Canonical score descending, offset ascending. Request k+1 and expand through every boundary tie before taking k=10, identically for exact and candidate scoring. Extra scoring is included in latency.'
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False)
hashes={str(p):sha(p) for root in (base,f2) for p in root.rglob('*') if p.is_file()}
proto=json.loads((f2/'protocol.json').read_text())
raw=np.fromfile(base/'corpus.f32',dtype='<f4').reshape(-1,768)
norm=raw/np.linalg.norm(raw,axis=1,keepdims=True)
keys=[hashlib.sha256(r.tobytes()).digest() for r in norm]
old=np.fromfile(base/'queries.f32',dtype='<f4').reshape(-1,768)
old/=np.linalg.norm(old,axis=1,keepdims=True)
excluded=set(proto['sample_order'][:16384])
bad={keys[i] for i in excluded}|{hashlib.sha256(r.tobytes()).digest() for r in old}
pool=list(range(len(raw))); random.Random(20260930).shuffle(pool)
selected=[]; used=set(bad)
for i in pool:
    if keys[i] not in used:
        selected.append(i); used.add(keys[i])
        if len(selected)==1320: break
assert len(selected)==1320 and not set(selected)&excluded
query_keys={keys[i] for i in selected}
removed=[i for i,k in enumerate(keys) if k in query_keys]
near=[]
reference=np.concatenate([norm[sorted(excluded)],old])
for start in range(0,len(selected),64):
    near.extend((norm[selected[start:start+64]]@reference.T).max(axis=1).tolist())
models=['n4096_e60_s42','n4096_e60_s43','n4096_e60_s44']
meta=json.loads((base/'dataset.json').read_text())
dump(out/'protocol.json',dict(base=str(base),f2=str(f2),seed=20260930,dimension=768,original_count=len(raw),live_count=len(raw)-len(removed),query_offsets=selected,removed_offsets=removed,warmup=20,validation=300,test=1000,models=models,trials=2,targets=[.90,.95],dense_nprobe=list(range(1,65)),selection='For each method and target choose smallest nprobe with validation mean Recall@10 >= target. Freeze all choices before evaluating test. Same choices for both timing trials. Do not reselect on test.',training_excluded_count=len(excluded),query_source='Former corpus rows never used for teacher or any F.2 model training; previously observed as corpus in F/F.2. Not external unseen data.',deduplication='Exclude exact normalized-f32 duplicates of training/old queries; query keys unique; remove all exact normalized query duplicates from search corpus.',near_duplicate_threshold=.9999,query_near_training_or_old_query_count=sum(v>=.9999 for v in near),max_cosine_to_training_or_old_queries=near,noninferiority_margin=.01,noninferiority='Exploratory paired query bootstrap: lower 95% recall-difference endpoint >= -0.01. No claim about teacher/sample variability or multiplicity-adjusted confirmatory testing.',timing='same process/storage/scorer, rotated method order per query and trial, CPU0, no HTTP, warm cache',source_dataset_metadata=meta,source_hashes=hashes,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
print(json.dumps(dict(out=str(out),queries=len(selected),removed=len(removed),live=len(raw)-len(removed),near_duplicate_queries=sum(v>=.9999 for v in near),protected_files=len(hashes))))
