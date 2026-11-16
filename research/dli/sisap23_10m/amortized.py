"""QPI projections from measured means at a matched recall target.
Input JSON list: name,N,recall10,recall30,maintenance_wall_s,query_mean_s,
acknowledged_writes_s. CPU model requires separate measured inputs.
"""
import json,sys
rows=json.load(open(sys.argv[1]));target10=float(sys.argv[2]);target30=float(sys.argv[3])
feasible=[r for r in rows if r['recall10']>=target10 and r['recall30']>=target30]
if not feasible:raise SystemExit('No measured feasible recall operating point')
if any(abs(a[k]-b[k])>.01 for a in feasible for b in feasible for k in ['recall10','recall30']):
 raise SystemExit('Recall spread >1 percentage point: do not compare unmatched services')
for qpi in [.01,.1,1,10,100]:
 for r in feasible:
  q=r['N']*qpi
  print(json.dumps(dict(name=r['name'],qpi=qpi,kind='stationary final-population extrapolation; not executed',
   maintenance_and_query_per_query_s=r['maintenance_wall_s']/q+r['query_mean_s'],
   including_acknowledged_writes_per_query_s=(r['maintenance_wall_s']+r['acknowledged_writes_s'])/q+r['query_mean_s'])))
