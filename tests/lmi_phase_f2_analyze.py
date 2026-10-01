"""F.2 diagnostics and staged selection. Reads immutable F results; writes F.2 only."""
import argparse, csv, hashlib, json, math, os
from pathlib import Path
from collections import defaultdict
import numpy as np
p=argparse.ArgumentParser();p.add_argument('action',choices=['diagnose','epoch_jobs','sample_jobs','select','final']);p.add_argument('--root',default=os.environ.get('LMI_PHASE_F2_DIR','/home/nicoo/work/lmi-phase-f2-data'));a=p.parse_args()
root=Path(a.root);base=Path(os.environ.get('LMI_PHASE_F_DIR','/home/nicoo/work/lmi-phase-f-data'));protocol=json.loads((root/'protocol.json').read_text())
def dump(path,v):path.write_text(json.dumps(v,indent=2,allow_nan=False))
def csv_out(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def make_job(n,e,s):return dict(id=f'n{n}_e{e}_s{s}',sample_size=n,epochs=e,seed=s)
def entropy(v):
    v=np.asarray(v,dtype=float);v=v[v>0];v=v/v.sum() if len(v) else v
    return float(-(v*np.log2(v)).sum())
if a.action=='epoch_jobs':
    jobs=[make_job(2048,e,s) for e in [30,60,120] for s in protocol['seeds']]
    dump(root/'jobs.json',jobs);dump(root/'epoch_jobs.json',jobs);print('9 epoch jobs');raise SystemExit
teacher=np.fromfile(root/'teacher_labels.u8',dtype=np.uint8)
ground=json.loads((root/'ground_truth.json').read_text());tq=np.asarray(ground['teacher_query_order'])
all_models=sorted([d for d in root.iterdir() if d.is_dir() and (d/'done.json').exists()])
summary=[];diagnostics={}
for d in all_models:
    tr=json.loads((d/'training.json').read_text()); sample=np.asarray(tr['sample_offsets']);job=tr['job']
    ranks=np.fromfile(d/'corpus_top8.u8',dtype=np.uint8).reshape(-1,8)
    qr=np.asarray(json.loads((d/'query_order.json').read_text()))
    remaining=np.ones(len(teacher),bool);remaining[sample]=False
    groups={'sampled_training':(teacher[sample],ranks[sample]),'remaining_corpus':(teacher[remaining],ranks[remaining]),'warmup':(tq[:20,0],qr[:20,:8]),'validation':(tq[protocol['validation'],0],qr[protocol['validation'],:8])}
    if d.name=='baseline' or a.action=='final':
        groups['heldout_measured']=(tq[20:,0],qr[20:,:8])
        groups['test']=(tq[protocol['test'],0],qr[protocol['test'],:8])
    diag={}
    diagdir=d/'diagnostics';diagdir.mkdir(exist_ok=True)
    for group,(truth,pred) in groups.items():
        matrix=np.zeros((64,64),dtype=np.int64);np.add.at(matrix,(truth,pred[:,0]),1)
        np.savetxt(diagdir/(group+'_confusion.csv'),matrix,fmt='%d',delimiter=',')
        support=matrix.sum(1);assigned=matrix.sum(0)
        per_class=[]
        for i in range(64):
            per_class.append(dict(bucket=i,teacher_count=int(support[i]),mlp_count=int(assigned[i]),teacher_correct=int(matrix[i,i]),teacher_accuracy=float(matrix[i,i]/support[i]) if support[i] else None,dominant_mlp_bucket=int(matrix[i].argmax()) if support[i] else None,dominant_fraction=float(matrix[i].max()/support[i]) if support[i] else None,teacher_assignment_entropy_bits=entropy(matrix[i]),mlp_teacher_classes_merged=int(np.count_nonzero(matrix[:,i])),mlp_teacher_entropy_bits=entropy(matrix[:,i])))
        csv_out(diagdir/(group+'_buckets.csv'),per_class)
        diag[group]=dict(count=len(truth),top_p_agreement={str(k):float(np.any(pred[:,:k]==truth[:,None],axis=1).mean()) for k in [1,2,4,8]},teacher_sizes=support.tolist(),mlp_sizes=assigned.tolist(),active=int(np.count_nonzero(assigned)),empty=int(np.count_nonzero(assigned==0)),assignment_entropy_bits=entropy(assigned),largest_fraction=float(assigned.max()/len(truth)))
    diag['corpus_sizes']=np.bincount(ranks[:,0],minlength=64).tolist()
    diag['corpus_empty']=int(sum(v==0 for v in diag['corpus_sizes']))
    dump(diagdir/'summary.json',diag);diagnostics[d.name]=diag
    paths=[d/'queries.jsonl']+([d/'test_queries.jsonl'] if (d/'test_queries.jsonl').exists() else [])
    groups=defaultdict(list)
    for path in paths:
        for line in path.read_text().splitlines():
            row=json.loads(line);groups[row['subset'],row['nprobe']].append(row)
    for (subset,nprobe),rows in sorted(groups.items()):
        if subset=='test' and d.name!='baseline' and a.action!='final':continue
        v=dict(model=d.name,sample_size=job['sample_size'],epochs=job['epochs'],seed=job['seed'],subset=subset,nprobe=nprobe,observations=len(rows),recall=float(np.mean([x['recall'] for x in rows])),candidate_mean=float(np.mean([x['candidate_count'] for x in rows])),candidate_fraction=float(np.mean([x['candidate_fraction'] for x in rows])),empty=diag['corpus_empty'],training_accuracy=diag['sampled_training']['top_p_agreement']['1'],validation_agreement=diag['validation']['top_p_agreement']['1'])
        for field in ['router','preparation','scoring','total']:
            for pct in [50,95,99]:v[f'{field}_p{pct}_ms']=float(np.percentile([x[field+'_ns']/1e6 for x in rows],pct))
        for pct in [0,25,50,75,95,99,100]:v[f'candidates_p{pct}']=float(np.percentile([x['candidate_count'] for x in rows],pct))
        summary.append(v)
dump(root/'summary.json',summary);csv_out(root/'summary.csv',summary)
dump(root/'diagnostics.json',diagnostics)
if a.action in ['diagnose','final']:
    original={(x['query']+20,x['effort'],x['trial']):x for x in map(json.loads,(base/'queries.jsonl').read_text().splitlines()) if x['method']=='mlp'}
    replay=list(map(json.loads,(root/'baseline/queries.jsonl').read_text().splitlines()))
    for x in replay:
        previous=original[x['query'],x['nprobe'],x['trial']]
        assert x['candidate_count']==previous['candidate_count'] and x['recall']==previous['recall']
    dump(root/'baseline_reproduction.json',dict(observations=len(replay),exact_candidate_and_recall_match=True,saved_router_and_postings_asserted=True))
    # These descriptive counterfactuals do not assert a unique causal decomposition.
    trace=[json.loads(x) for x in (root/'baseline/trace.jsonl').read_text().splitlines()]
    decomposed=[]
    for p in [1,2,4,8,16,32,64]:
        rows=[x for x in trace if x['nprobe']==p];v={'nprobe':p}
        for i,key in enumerate(['tt','mt','tm','mm']):
            v[key+'_recall']=float(np.mean([sum(n[key] for n in x['neighbors'])/10 for x in rows]))
            v[key+'_candidates']=float(np.mean([x['counts_tt_mt_tm_mm'][i] for x in rows]))
        for key in ['query_route_added_removed','corpus_reassignment_added_removed']:
            for i,label in enumerate(['added','removed']):v[key+'_'+label]=float(np.mean([x[key][i] for x in rows]))
        v['teacher_partition_missed_neighbors']=float(np.mean([sum(not n['tt'] for n in x['neighbors']) for x in rows]))
        for before,after in [('tt','mt'),('mt','mm'),('tt','mm')]:
            v[before+'_to_'+after+'_lost']=float(np.mean([sum(n[before] and not n[after] for n in x['neighbors']) for x in rows]))
            v[before+'_to_'+after+'_gained']=float(np.mean([sum(not n[before] and n[after] for n in x['neighbors']) for x in rows]))
        decomposed.append(v)
    dump(root/'baseline_decomposition.json',decomposed);csv_out(root/'baseline_decomposition.csv',decomposed)
    diag=diagnostics['baseline'];train=np.asarray(diag['sampled_training']['teacher_sizes']);corpus=np.bincount(teacher,minlength=64);mlp=np.asarray(diag['corpus_sizes']);empty=mlp==0
    centers=np.asarray(json.loads((base/'centroid_state.json').read_text())['centers'])
    dist=((centers[:,None,:]-centers[None,:,:])**2).sum(2);centroid_rank=np.argsort(np.argsort(dist,axis=1),axis=1)
    matrix=np.loadtxt(root/'baseline/diagnostics/remaining_corpus_confusion.csv',delimiter=',',dtype=int)
    np.fill_diagonal(matrix,0); errors=int(matrix.sum())
    empty_analysis=dict(empty_classes=np.flatnonzero(empty).tolist(),empty_training_support=train[empty].tolist(),active_training_support=train[~empty].tolist(),empty_corpus_teacher_sizes=corpus[empty].tolist(),empty_training_mass=float(train[empty].sum()/train.sum()),empty_teacher_corpus_mass=float(corpus[empty].sum()/corpus.sum()),empty_with_zero_training_support=int(sum(train[empty]==0)),median_training_support_empty=float(np.median(train[empty])),median_training_support_active=float(np.median(train[~empty])),wrong_remaining_assignments=errors,wrong_assignments_to_nearest_4_centroids=float(matrix[(centroid_rank>=1)&(centroid_rank<=4)].sum()/errors),wrong_assignments_to_nearest_8_centroids=float(matrix[(centroid_rank>=1)&(centroid_rank<=8)].sum()/errors),mean_wrong_target_centroid_rank=float((matrix*centroid_rank).sum()/errors))
    dump(root/'empty_bucket_analysis.json',empty_analysis);print(json.dumps(empty_analysis,indent=2));print('nprobe4',decomposed[2])

def select_configs(jobs,tag):
    scores=[];matches=[]
    for n,e in sorted({(j['sample_size'],j['epochs']) for j in jobs}):
        costs=[];missing=0
        for seed in protocol['seeds']:
            model=make_job(n,e,seed)['id'];points=[x for x in summary if x['model']==model and x['subset']=='validation']
            assert len(points)==7,model
            for target in [.9,.95]:
                possible=[x for x in points if abs(x['recall']-target)<=.02+1e-12]
                best=min(possible,key=lambda x:(abs(x['recall']-target),x['candidate_fraction'])) if possible else None
                costs.append(best['candidate_fraction'] if best else 1.0);missing+=best is None
                matches.append(dict(model=model,target=target,point=best))
        scores.append(dict(sample_size=n,epochs=e,mean_validation_cost=float(np.mean(costs)),missing_bands=missing))
    best=min(scores,key=lambda x:(x['mean_validation_cost'],x['sample_size'],x['epochs']))
    result=dict(scores=scores,matches=matches,selected=best,criterion=protocol['selection'])
    path=root/(tag+'_selection.json');assert not path.exists();dump(path,result);print(json.dumps(scores,indent=2));return best
if a.action=='sample_jobs':
    for seed in protocol['seeds']:
        curves=[json.loads((root/make_job(2048,e,seed)['id']/'training.json').read_text())['curve'] for e in [30,60,120]]
        for field in ['loss','training_accuracy']:
            assert [x[field] for x in curves[0]]==[x[field] for x in curves[1][:30]]==[x[field] for x in curves[2][:30]], (seed,field)
            assert [x[field] for x in curves[1]]==[x[field] for x in curves[2][:60]], (seed,field)
    dump(root/'epoch_prefix_check.json',dict(identical_loss_accuracy_prefixes=True,seeds=protocol['seeds']))
    best=select_configs(json.loads((root/'epoch_jobs.json').read_text()),'epoch')
    jobs=[make_job(n,best['epochs'],s) for n in [4096,8192,16384] for s in protocol['seeds']]
    dump(root/'sample_jobs.json',jobs);dump(root/'jobs.json',jobs)
if a.action=='select':
    jobs=json.loads((root/'epoch_jobs.json').read_text())+json.loads((root/'sample_jobs.json').read_text())
    best=select_configs(jobs,'final')
    selected=[make_job(best['sample_size'],best['epochs'],s) for s in protocol['seeds']]
    dump(root/'selected_jobs.json',selected)
for name,expected in protocol['baseline_hashes'].items():
    assert hashlib.sha256((base/name).read_bytes()).hexdigest()==expected,name
print('Baseline hashes unchanged. Models:',len(all_models))
