"""Replay frozen native bucket selections; compare logical offset locality, no I/O claims."""
import csv,json,struct,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); run=root/'native-frontier'; collection=root.parent/'continuation-25/geometric-10120191/collection/storage/0/segments'
cases=json.loads((run/'frontier.json').read_text()); routes=[json.loads(x) for x in (run/'routing-events.jsonl').read_text().splitlines()]
cache={}; rows=[]
for case_id in [2,32,35]:
 c=cases[case_id]
 for ordinal in range(c['ordinal_begin'],c['ordinal_begin']+c['query_count']):
  route=routes[ordinal]
  for uuid in dict.fromkeys(b['physical'] for b in route['buckets'] if b['bucket'] is not None):
   if uuid not in cache:
    p=collection/uuid/'vector_index/lmi_postings.bin'
    with p.open('rb') as f:
     n=struct.unpack('<Q',f.read(8))[0]; bounds=np.frombuffer(f.read(8*n),dtype='<u8'); count=struct.unpack('<Q',f.read(8))[0]; offsets=np.frombuffer(f.read(count*4),dtype='<u4')
    cache[uuid]=(bounds,offsets)
   bounds,offsets=cache[uuid]; buckets=[b['bucket'] for b in route['buckets'] if b['physical']==uuid and b['bucket'] is not None]
   original=np.concatenate([offsets[int(bounds[b]):int(bounds[b+1])] for b in buckets]).astype('int64'); ordered=np.unique(original)
   for name,arr in [('bucket_order',original),('native_sorted',ordered)]:
    jumps=np.abs(np.diff(arr)); pages=(arr*1536)//4096
    rows.append(dict(case=case_id,query=ordinal-c['ordinal_begin']+600,uuid=uuid,order=name,entries=len(arr),unique=len(ordered),mean_offset_jump=float(jumps.mean()) if len(jumps) else 0,backward=int(np.sum(np.diff(arr)<0)),page_transitions=int(np.sum(np.diff(pages)!=0)),distinct_logical_pages=int(len(np.unique(pages)))))
with (Path(sys.argv[2])/'locality-replay.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for name in ('bucket_order','native_sorted'):
 a=[r for r in rows if r['order']==name]; print(name,'member-query traces',len(a),'mean jump',round(np.mean([r['mean_offset_jump'] for r in a]),2),'backwards',sum(r['backward'] for r in a),'logical page transitions',sum(r['page_transitions'] for r in a),'distinct logical pages',sum(r['distinct_logical_pages'] for r in a))
