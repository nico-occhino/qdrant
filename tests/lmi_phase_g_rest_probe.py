"""Phase G read-only live REST dispatch probe on persisted SISAP LMI."""
import json,urllib.request,hashlib
from pathlib import Path
import h5py
base='http://127.0.0.1:17038';coll='sisap2023_10m_f16_lmi'
out=Path('/home/nicoo/work/qdrant/work/phase_g/g3_planner');log=out/'server/server.log'
seg=Path('/home/nicoo/work/lmi-sisap2023-storage/10m-f16/storage/collections/sisap2023_10m_f16_lmi/0/segments/8f67bcc1-6bca-49c8-bef5-4ae8f4565ab1/vector_index')
names=['lmi_state.json','lmi_router.bin','lmi_postings.bin']
def digests():return {n:hashlib.sha256((seg/n).read_bytes()).hexdigest() for n in names}
before=digests()
with h5py.File('/mnt/c/datasets/sisap2023/public-queries-10k-clip768v2.h5') as h:q=h['emb'][0].astype(float).tolist()
cases=[('no_params',{}),('default_params',{'params':{}}),('exact',{'params':{'exact':True}}),('hnsw_ef',{'params':{'hnsw_ef':64}}),('quantization_ignore',{'params':{'quantization':{'ignore':True}}}),('has_id_filter',{'filter':{'must':[{'has_id':[1]}]}})]
results=[]
for name,opts in cases:
 start=log.stat().st_size
 body={'vector':q,'limit':10,'with_payload':False,'with_vector':False,**opts}
 req=urllib.request.Request(base+'/collections/'+coll+'/points/search',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
 try:
  with urllib.request.urlopen(req,timeout=120) as response: answer=json.load(response)
  status=answer.get('status');ids=[x['id'] for x in answer.get('result',[])]
 except Exception as e:
  status=repr(e);ids=[]
 suffix=log.read_bytes()[start:].decode(errors='replace')
 results.append({'case':name,'request_options':opts,'status':status,'returned_ids':ids,
  'log_static_learned':suffix.count('candidate_source=StaticLearned'),
  'log_plain':suffix.count('candidate_source=all_valid_points'),
  'log_lmi_search':suffix.count('[LMI-DUMMY] search'),
  'log_excerpt':[line for line in suffix.splitlines() if '[LMI-' in line][-8:]})
after=digests();assert before==after
(out/'rest-dispatch.json').write_text(json.dumps({'cases':results,'learned_hashes_before':before,'learned_hashes_after':after,'unchanged':True},indent=2)+'\n')
print(json.dumps([{'case':r['case'],'status':r['status'],'n':len(r['returned_ids']),'learned_markers':r['log_static_learned'],'lmi_calls':r['log_lmi_search']} for r in results],indent=2))
