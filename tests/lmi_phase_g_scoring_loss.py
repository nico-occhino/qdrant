"""Investigate P=4 candidate-covered official neighbors lost at output."""
import json
from pathlib import Path
import h5py
import numpy as np
p=Path('/home/nicoo/work/qdrant/work/phase_g/g1_failure_decomposition')
raw=Path('/home/nicoo/work/qdrant/work/phase_s3/sisap2023/10m-f16/queries.full.jsonl')
rows=[json.loads(s) for s in (p/'per-query.jsonl').open()]
loss={x['query_row']:x for x in rows if x['scoring_output_loss_4']>1e-8}
accepted={x['query_row']:x for x in map(json.loads,raw.open()) if x['query_row'] in loss}
assert len(loss)==len(accepted)
per=[]
with h5py.File('/mnt/c/datasets/sisap2023/laion2B-en-clip768v2-n=10M.h5') as corpus,h5py.File('/mnt/c/datasets/sisap2023/public-queries-10k-clip768v2.h5') as q:
 for row,analysis in loss.items():
  original=accepted[row]
  query=np.asarray(q['emb'][row],dtype=np.float64)
  query/=np.linalg.norm(query)
  threshold=float(original['gold_distances'][9])
  results=[]
  for external,score in zip(original['result_ids_1_based'],original['result_scores']):
   vector=np.asarray(corpus['emb'][external-1],dtype=np.float64)
   vector/=np.linalg.norm(vector)
   distance=float(1-query@vector)
   results.append({'id':external,'exact_source_cosine_distance':distance,'stored_score':score,
                   'within_gold_d10_1e-7':distance<=threshold+1e-7,
                   'official_gold_top10':external in original['gold_ids_1_based']})
  per.append({'query_row':row,'gold_d10':threshold,'gold_zero_count':analysis['gold_zero_distance_count'],
              'official_recall':analysis['official_recall'],'candidate_gold_coverage':analysis['router_gold_coverage']['4'],
              'returned':results,'returned_within_gold_radius':sum(x['within_gold_d10_1e-7'] for x in results)})
(p/'scoring-loss-source-check.jsonl').write_text(''.join(json.dumps(v,separators=(',',':'))+'\n' for v in per))
s={'queries_with_positive_scoring_output_loss':len(per),
   'queries_all_gold_distances_zero':sum(x['gold_zero_count']==10 for x in per),
   'queries_any_gold_zero':sum(x['gold_zero_count']>0 for x in per),
   'queries_all_10_returned_within_gold_d10_plus_1e-7':sum(x['returned_within_gold_radius']==10 for x in per),
   'mean_returned_within_gold_radius':float(np.mean([x['returned_within_gold_radius'] for x in per])),
   'examples':per[:3],
   'interpretation':'Original source Float16 rows and original Float32 query were recomputed in Float64. Values are diagnostic and not a substitute for the official gold or Qdrant stored Float16 ranking. Official ID-overlap remains the benchmark.'}
(p/'scoring-loss-source-summary.json').write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps({k:v for k,v in s.items() if k!='examples'},indent=2))
