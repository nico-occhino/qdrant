"""Phase G read-only partition/router analysis of the persisted SISAP generation.
Inputs: Qdrant IdTracker export, immutable postings/router, official HDF5, accepted JSONL.
This script never opens writable Qdrant storage or issues requests.
"""
import hashlib, json, os, struct
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path('/home/nicoo/work/qdrant')
BASE = REPO/'work/phase_s3/sisap2023/10m-f16'
G1 = REPO/'work/phase_g/g1_failure_decomposition'
G2 = REPO/'work/phase_g/g2_frontier'
SEG = Path('/home/nicoo/work/lmi-sisap2023-storage/10m-f16/storage/collections/sisap2023_10m_f16_lmi/0/segments/8f67bcc1-6bca-49c8-bef5-4ae8f4565ab1')
QFILE = Path('/mnt/c/datasets/sisap2023/public-queries-10k-clip768v2.h5')
GFILE = Path('/mnt/c/datasets/sisap2023/laion2B-en-public-gold-standard-v2-10M-F64-IEEE754.h5')
P = [1,2,4,8,16,32]
G2.mkdir(parents=True,exist_ok=True)

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''): h.update(block)
 return h.hexdigest()

expected=json.loads((BASE/'queries.full.summary.json').read_text())
for name in ('lmi_router.bin','lmi_postings.bin'):
 key=next(k for k in expected['lmi_hashes_before'] if k.endswith(name))
 assert digest(SEG/'vector_index'/name)==expected['lmi_hashes_before'][key]
assert digest(BASE/'queries.full.jsonl')==expected['raw_jsonl_sha256']
router=json.loads((G1/'router.json').read_text())
layers=[]
for layer in router['layers']:
 if layer=='ReLU': layers.append(('relu',))
 else:
  spec=layer['Linear']; layers.append(('linear', np.asarray(spec['weights'],dtype=np.float32).reshape(spec['out_features'],spec['in_features']),np.asarray(spec['bias'],dtype=np.float32)))
assert len(layers)==3 and layers[0][1].shape==(512,768) and layers[2][1].shape==(3162,512)
postings_path=SEG/'vector_index/lmi_postings.bin'
with open(postings_path,'rb') as f:
 nbound=struct.unpack('<Q',f.read(8))[0]
 boundaries=np.frombuffer(f.read(nbound*8),dtype='<u8').copy()
 npoints=struct.unpack('<Q',f.read(8))[0]
 assert npoints==10120191 and nbound==3163
 # Confirm format and all offsets were independently validated by the Rust exporter.
 assert f.seek(0,2)==8+8*nbound+8+4*npoints
bucket_sizes=np.diff(boundaries).astype(np.int64)
records=[json.loads(x) for x in (BASE/'queries.full.jsonl').open()]
mapped=[json.loads(x) for x in (G1/'gold-buckets.jsonl').open()]
assert len(records)==len(mapped)==9980
assert all(a['query_row']==b['query_row']==i+20 for i,(a,b) in enumerate(zip(records,mapped)))
with h5py.File(QFILE,'r') as h:
 queries=np.asarray(h['emb'][20:10000],dtype=np.float32)
assert queries.shape==(9980,768)
# Qdrant normalizes Cosine query vectors before routing. Use float32 normalization;
# exact native parity is checked against accepted P=4 candidate counts below.
norms=np.linalg.norm(queries,axis=1,keepdims=True)
queries=np.divide(queries,norms,out=np.zeros_like(queries),where=norms!=0)
ranked=np.empty((len(queries),32),dtype=np.int32)
for start in range(0,len(queries),64):
 x=queries[start:start+64]
 for layer in layers:
  if layer[0]=='relu': x=np.maximum(x,0)
  else: x=x@layer[1].T+layer[2]
 ranked[start:start+len(x)]=np.argsort(-x,axis=1,kind='stable')[:,:32]

per=[]
frontier={str(p):{'coverage':[],'oracle':[],'candidates':[]} for p in P}
mismatches=[]
with h5py.File(GFILE,'r') as h:
 for i,(raw,gold) in enumerate(zip(records,mapped)):
  labels=[int(x) for x in gold['gold_buckets']]
  mult=Counter(labels)
  oracle={str(p):sum(sorted(mult.values(),reverse=True)[:p])/10 for p in P}
  chosen={str(p):[int(v) for v in ranked[i,:p]] for p in P}
  coverage={str(p):sum(mult.get(b,0) for b in chosen[str(p)])/10 for p in P}
  candidates={str(p):int(bucket_sizes[ranked[i,:p]].sum()) for p in P}
  if candidates['4']!=int(raw['candidate_count']):
   mismatches.append({'query_row':int(raw['query_row']),'expected':int(raw['candidate_count']),'computed':candidates['4']})
  actual=float(raw['recall10_id_overlap'])
  if coverage['4']+1e-6<actual:
   raise AssertionError(('actual recall exceeds candidate gold coverage',i,coverage['4'],actual))
  for p in P:
   frontier[str(p)]['coverage'].append(coverage[str(p)])
   frontier[str(p)]['oracle'].append(oracle[str(p)])
   frontier[str(p)]['candidates'].append(candidates[str(p)])
  d=np.asarray(raw['gold_distances'],dtype=float)
  all_gold_ids=np.asarray(h['knns'][int(raw['query_row'])],dtype=np.int64)
  all_gold_dist=np.asarray(h['dists'][int(raw['query_row'])],dtype=np.float64)
  threshold=float(d[9])+1e-6
  qualifying=set(int(v) for v in all_gold_ids[all_gold_dist<=threshold])
  tie_aware=min(10,sum(int(v) in qualifying for v in raw['result_ids_1_based']))/10
  per.append({'query_row':int(raw['query_row']),'gold_ids':gold['gold_ids'],
      'gold_offsets':gold['gold_offsets'],'gold_buckets':labels,
      'distinct_gold_buckets':len(mult),'bucket_multiplicities':dict(sorted(mult.items())),
      'router_top32_buckets':chosen['32'],'oracle_recall':oracle,
      'router_gold_coverage':coverage,'candidate_count_by_p':candidates,
      'partition_loss_4':1-oracle['4'],'router_loss_4':oracle['4']-coverage['4'],
      'scoring_output_loss_4':coverage['4']-actual,'actual_recall_4':actual,
      'candidate_count_4':int(raw['candidate_count']),'candidate_fraction_4':float(raw['candidate_fraction']),
      'd1':float(d[0]),'d10':float(d[9]),'d10_minus_d1':float(d[9]-d[0]),
      'mean_top10_distance':float(d.mean()),'largest_gold_bucket_multiplicity':max(mult.values()),
      'gold_zero_distance_count':int(np.sum(d<=1e-6)),
      'gold_tie_qualified_count_in_top1000':len(qualifying),'tie_aware_recall_diagnostic':tie_aware,
      'official_recall':actual})
with (G1/'per-query.jsonl').open('w') as f:
 for row in per: f.write(json.dumps(row,separators=(',',':'))+'\n')

def stats(v):
 a=np.asarray(v,dtype=float)
 return {'mean':float(a.mean()),'p5':float(np.percentile(a,5)),
  'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95)),
  'min':float(a.min()),'max':float(a.max())}

def corr(a,b):
 x=np.asarray([v[a] for v in per],dtype=float)
 y=np.asarray([v[b] for v in per],dtype=float)
 return {'pearson_r':float(pearsonr(x,y).statistic),'spearman_rho':float(spearmanr(x,y).statistic)}

summary={'measured_rows':len(per),'mapping':'real Qdrant IdTracker; not ID-as-offset assumption',
 'postings_cover_all_vectors':True,'accepted_raw_sha256':expected['raw_jsonl_sha256'],
 'p4_candidate_count_native_parity':{'matching_rows':len(per)-len(mismatches),'mismatches':len(mismatches),'examples':mismatches[:20]},
 'p4':{'oracle_recall':stats([x['oracle_recall']['4'] for x in per]),
  'router_gold_coverage':stats([x['router_gold_coverage']['4'] for x in per]),
  'actual_recall':stats([x['actual_recall_4'] for x in per]),
  'partition_loss':stats([x['partition_loss_4'] for x in per]),
  'router_loss':stats([x['router_loss_4'] for x in per]),
  'scoring_output_loss':stats([x['scoring_output_loss_4'] for x in per])},
 'query_hardness':{'distinct_gold_buckets':stats([x['distinct_gold_buckets'] for x in per]),
  'zero_distance_queries':sum(x['gold_zero_distance_count']>0 for x in per),
  'ten_zero_distance_queries':sum(x['gold_zero_distance_count']==10 for x in per),
  'tie_aware_recall':stats([x['tie_aware_recall_diagnostic'] for x in per]),
  'official_recall':stats([x['official_recall'] for x in per]),
  'ties_cutoff_more_than_10':sum(x['gold_tie_qualified_count_in_top1000']>10 for x in per)},
 'correlations':{a+'_vs_recall':corr(a,'actual_recall_4') for a in ['d10','distinct_gold_buckets','partition_loss_4','router_loss_4','candidate_count_4']}}
(G1/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
front=[]
for p in P:
 v=frontier[str(p)]
 front.append({'nprobe':p,'oracle_recall':stats(v['oracle']),
   'router_gold_coverage':stats(v['coverage']),
   'candidate_count':stats(v['candidates']),
   'candidate_fraction_mean':float(np.mean(v['candidates'])/10120191),
   'actual_recall':stats([x['actual_recall_4'] for x in per]) if p==4 else None,
   'interpretation':'actual Qdrant result' if p==4 else 'candidate gold coverage, upper bound on actual retrieval; no candidate scoring'})
(G2/'analytic-frontier.json').write_text(json.dumps({'source':'fixed persisted LMI, no rebuild','rows':9980,'points':10120191,'frontier':front},indent=2)+'\n')
print(json.dumps({'p4':summary['p4'],'hardness':summary['query_hardness'],'parity':summary['p4_candidate_count_native_parity'],'frontier':[{'p':x['nprobe'],'oracle':x['oracle_recall']['mean'],'coverage':x['router_gold_coverage']['mean'],'candidates':x['candidate_count']['mean']} for x in front]},indent=2))
