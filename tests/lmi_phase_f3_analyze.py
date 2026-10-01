"""F.3 analysis: fixed validation decisions; never select operating points on test."""
from pathlib import Path
from collections import defaultdict
import json, csv, hashlib, sys, subprocess, platform, os
import numpy as np

root=Path(os.environ.get('LMI_PHASE_F3_DIR','/home/nicoo/work/lmi-phase-f3-data'))
p=json.loads((root/'protocol.json').read_text())
done=json.loads((root/'done.json').read_text()); assert done['complete']
selection=json.loads((root/'selection.json').read_text())
names=selection['methods']
def dump(name,x):
    with (root/name).open('x') as f: json.dump(x,f,indent=2,allow_nan=False)
def table(name,rows):
    with (root/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
preserved={k:sha(Path(k))==v for k,v in p['source_hashes'].items()}
assert all(preserved.values()), 'Preserved source changed'
dump('source_preservation.json',{'verified_files':len(preserved),'all_unchanged':True,'checks':preserved})
dense=defaultdict(list)
with (root/'dense_queries.jsonl').open() as f:
    for line in f:
        r=json.loads(line);dense[(r['subset'],r['model'],r['nprobe'])].append(r)
assert sum(map(len,dense.values()))==1300*5*64
curves=[]
for (subset,m,probe),rs in dense.items():
    assert len(rs)==(300 if subset=='validation' else 1000)
    curves.append(dict(subset=subset,model=m,nprobe=probe,query_count=len(rs),recall=float(np.mean([r['recall'] for r in rs])),candidate_mean=float(np.mean([r['candidate_count'] for r in rs])),candidate_fraction=float(np.mean([r['candidate_fraction'] for r in rs]))))
table('dense_summary.csv',curves);dump('dense_summary.json',curves)
timing=defaultdict(list)
with (root/'timing_queries.jsonl').open() as f:
    for line in f:
        r=json.loads(line);timing[(r['model'],r['target'])].append(r)
assert sum(map(len,timing.values()))==20000
summary=[]; per_query={}
for (m,target),rs in timing.items():
    assert len(rs)==2000
    byq=defaultdict(list)
    for r in rs: byq[r['query']].append(r)
    assert len(byq)==1000
    for pair in byq.values():
        assert len(pair)==2 and {r['trial'] for r in pair}=={0,1}
        assert pair[0]['recall']==pair[1]['recall'] and pair[0]['candidate_count']==pair[1]['candidate_count']
    probe=rs[0]['nprobe']; assert {r['nprobe'] for r in rs}=={probe}
    native={qi:pair[0] for qi,pair in byq.items()}
    for r in dense[('test',m,probe)]:
        assert r['recall']==native[r['query']]['recall'] and r['candidate_count']==native[r['query']]['candidate_count']
    row=dict(model=m,target=target,nprobe=probe,queries=1000,trials=2,validation_recall=float(np.mean([r['recall'] for r in dense[('validation',m,probe)]])),test_recall=float(np.mean([r['recall'] for r in rs])),candidate_mean=float(np.mean([r['candidate_count'] for r in rs])),candidate_fraction=float(np.mean([r['candidate_fraction'] for r in rs])))
    for key in ['router_ns','preparation_ns','scoring_ns','total_ns']:
        for pct in [50,95,99]:row[f'{key[:-3]}_p{pct}_ms']=float(np.percentile([r[key] for r in rs],pct)/1e6)
    summary.append(row)
    per_query[(m,target)]=np.array([[pair[0]['recall'],pair[0]['candidate_count'],np.mean([r['total_ns'] for r in pair])/1e6] for qi,pair in sorted(byq.items())])
summary.sort(key=lambda r:(r['target'],names.index(r['model'])))
table('paired_summary.csv',summary);dump('paired_summary.json',summary)
rng=np.random.default_rng(20260930)
draws=rng.integers(0,1000,size=(2000,1000))
comparisons=[]
for target in [.90,.95]:
    ref=per_query[('centroid',target)]
    for m in names[1:]:
        delta=per_query[(m,target)]-ref
        boots=delta[draws].mean(axis=1)
        low,high=np.percentile(boots,[2.5,97.5],axis=0)
        comparisons.append(dict(model=m,target=target,reference='centroid',recall_delta=float(delta[:,0].mean()),recall_ci95= [float(low[0]),float(high[0])],candidate_delta=float(delta[:,1].mean()),candidate_ci95=[float(low[1]),float(high[1])],candidate_saving_percent=float(-delta[:,1].mean()/ref[:,1].mean()*100),mean_paired_latency_delta_ms=float(delta[:,2].mean()),mean_latency_ci95_ms=[float(low[2]),float(high[2])],exploratory_recall_noninferiority_pass=bool(low[0]>=-.01),candidate_reduction_ci_excludes_zero=bool(high[1]<0),paired_mean_latency_reduction_ci_excludes_zero=bool(high[2]<0)))
dump('paired_bootstrap.json',{'resamples':2000,'seed':20260930,'unit':'query; average timings over two trials before paired bootstrap','caveat':'Exploratory fixed-model query intervals; no multiple-comparison adjustment, no training/teacher uncertainty; latency CI refers to paired mean, not median.','comparisons':comparisons})
ground=json.loads((root/'ground_truth.json').read_text())
dump('query_audit.json',{'query_count':len(ground),'exact_query_duplicates_removed':len(p['removed_offsets'])-len(p['query_offsets']),'near_training_or_old_queries_at_09999':p['query_near_training_or_old_query_count'],'queries_with_nearest_live_neighbor_score_at_least_09999':sum(g['neighbors'][0]['score']>=.9999 for g in ground),'no_self_matches':all(g['original_offset'] not in [n['offset'] for n in g['neighbors']] for g in ground)})
env={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'cpu':subprocess.check_output(['bash','-lc','lscpu'],text=True),'rustc':subprocess.check_output(['/home/nicoo/.cargo/bin/rustc','--version'],text=True).strip(),'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
dump('environment.json',env)

sys.path.insert(0,'/home/nicoo/work/lmi-phase-f-python')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors={'centroid':'#2166ac','affine':'#555555',names[2]:'#d6604d',names[3]:'#1b9e77',names[4]:'#8e63b6'}
fig,ax=plt.subplots(figsize=(8,5))
for m in names:
    rows=sorted([r for r in curves if r['subset']=='test' and r['model']==m],key=lambda r:r['nprobe'])
    ax.plot([r['candidate_fraction'] for r in rows],[r['recall'] for r in rows],label=m,color=colors[m],linestyle='--' if m=='affine' else '-')
ax.set(xlabel='Mean fraction of searchable corpus scored',ylabel='Recall@10',title='F.3 frozen models: dense test curves (1,000 queries)',ylim=(.5,1.005))
ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(root/'recall_vs_candidates.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
for m in names:
    rows=[r for r in summary if r['model']==m]
    ax.plot([r['total_p50_ms'] for r in rows],[r['test_recall'] for r in rows],marker='o',label=m,color=colors[m],linestyle='--' if m=='affine' else '-')
ax.set(xlabel='Paired common-harness total p50 (ms)',ylabel='Recall@10',title='F.3: operating points selected on validation')
ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(root/'recall_vs_latency.png',dpi=180);plt.close(fig)

lines=['# Phase F.3: frozen-model consolidation','', '## Outcome and scope','',
'F.3 evaluates the already-selected 4096/60 MLP models (seeds 42/43/44) without any training, teacher refit or HNSW rebuild. This is a new internal evaluation on a reduced corpus, not a rerun or replacement of Phase F/F.2 and not an external-dataset validation.','',
f"The original corpus had 99,780 vectors. We selected 20 warmup, 300 validation and 1,000 test queries from rows excluded from every F.2 training sample and the teacher sample. Removing queries and four additional exact normalized duplicates leaves {p['live_count']:,} searchable vectors. These rows were previously observed as corpus during F/F.2; they are fresh as query evaluations, not historically untouched data. No near duplicate to training/old queries at cosine >=0.9999 was found. Semantic near duplicates are not ruled out.",'',
'## Fixed validation policy','',
'For targets 0.90 and 0.95, choose the smallest integer nprobe reaching that mean recall on the 300 validation queries. Save the choices before evaluating test rankings. All p=1..64 are evaluated by exact-neighbor membership for recall/candidate curves. Native scoring and timing run only at the two fixed points per method, plus full-probe correctness checks. No interpolation, test-time selection or architecture/hyperparameter tuning is used.','',
'| Method | Target | nprobe | Validation recall | Test recall | Mean candidates | Total p50 ms | p95 ms | p99 ms |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['model']} | {r['target']:.2f} | {r['nprobe']} | {r['validation_recall']:.4f} | {r['test_recall']:.4f} | {r['candidate_mean']:.1f} | {r['total_p50_ms']:.3f} | {r['total_p95_ms']:.3f} | {r['total_p99_ms']:.3f} |")
lines+=['','## Paired uncertainty and decision gates','','Intervals use 2,000 paired query resamples, preserving both trials within each query. Recall non-inferiority is exploratory with a predeclared 0.01 margin; interval containment is not a formal multiplicity-adjusted test. Training/teacher/sample uncertainty is not covered. Lower candidates at lower recall must not be reported as unconditional superiority.','', '| Model | Target | Recall delta [95% CI] | Candidate saving | Candidate delta [95% CI] | Recall margin met? |','|---|---:|---|---:|---|---|']
for r in comparisons:
    if r['model']=='affine':continue
    lines.append(f"| {r['model']} | {r['target']:.2f} | {r['recall_delta']:+.4f} [{r['recall_ci95'][0]:+.4f}, {r['recall_ci95'][1]:+.4f}] | {r['candidate_saving_percent']:.2f}% | {r['candidate_delta']:+.1f} [{r['candidate_ci95'][0]:+.1f}, {r['candidate_ci95'][1]:+.1f}] | {r['exploratory_recall_noninferiority_pass']} |")
lines+=['','## Timing methodology','','Same process, segment, Qdrant vector storage, native BatchFilteredSearcher, candidate preparation and deletion checks for all methods. CPU0 affinity; OMP/OpenBLAS/Rayon one thread. Method order rotates within each query, target and trial. Twenty warmup queries precede each trial/target block. The common harness computes a full 64-bucket ranking for every router, then takes its prefix. This is CPU in-process warm-cache latency, not HTTP latency, cold-cache performance, a production LmiIndex call or HNSW timing. Two trials are intentionally bounded and do not establish long-run system variance. Components and p50/p95/p99 appear in paired_summary.csv/json. Percentiles pool query/trial observations; paired mean-latency intervals are separate.','',
'No current HNSW comparison is made on the changed corpus. Its historical F measurement is not copied into F.3 tables. Candidate visits for HNSW remain unavailable. No build-time improvement can be inferred: the models are loaded, and source partitions are pruned rather than rebuilt.','',
'## Verification and preservation','',f"- All {len(preserved)} protected F/F.2 source files retained their SHA-256 hashes.",f"- {done['native_assignment_checks']:,} native MLP assignment checks reproduce saved postings on retained vectors.",f"- {done['full_probe_native_checks']:,} full-probe native scoring checks equal Plain exact IDs/scores.",'- Zero centroid/affine query-order mismatches; shared postings guarantee identical deterministic retrieval.', '- 416,000 dense per-query operating-point rows and 20,000 per-query/per-trial timing rows are preserved.','- Native recall and candidate counts match dense membership predictions at every timed point; trials have identical retrieval results.','- Removed query rows never appear in exact or approximate results. Training is never invoked.','',
'## Remaining limits and next step','','Interpret F.3 as a controlled replication on fresh internal queries with frozen models and inherited partitions. The corpus is smaller and queries were previously corpus observations. F.2 model-selection history cannot be undone; an external query dataset, independent teacher/sample seeds and broader datasets remain necessary for generalization claims. The denser curve exposes high-recall tradeoffs but no 0.99 policy is selected for timing. No memory/build/reopen or lifecycle performance claims are added. Existing F.2 lifecycle results are historical evidence, not rerun here because production behavior is unchanged.','',
'The next research increment, if justified, should compare retrieval-supervised affine routing and a simple query-prototype memory over fixed centroid postings. Preserve this study as a frozen baseline; do not tune the current models in response to its test results.','',
'## Artifacts and reproduction','','- protocol.json: full row mapping, split, policy, environment assumptions and hashes.','- ground_truth.json: exact top-10 offsets/scores per query.','- dense_queries.jsonl and dense_summary.csv/json: complete p=1..64 observations.','- timing_queries.jsonl and paired_summary.csv/json: native timing components and retrieval results.','- paired_bootstrap.json: paired query uncertainty.','- source_preservation.json, query_audit.json, done.json: executed checks.','- recall_vs_candidates.png and recall_vs_latency.png: scientific plots.','- environment.json and verification/native-run.log: machine/compiler and execution evidence.','',
'Use a fresh output directory for reproduction; preparation and artifact writers refuse overwrite. Run the saved run_f3.sh with the existing Rust/tch CPU environment, then lmi_phase_f3_analyze.py. The protocol freezes source file hashes and the exact seed. No commit or push was performed.','']
lines += ['## Boundary-tie correction and interpretation','',
    'The initial native run stopped before measured timings: two valid top-10 results disagreed by an ID at an equal-score boundary. Eleven queries had equal scores in the last two positions of the initial exact top-10. The failed run and its original outputs remain in lmi-phase-f3-attempt1-tie-failure. This was a harness correctness issue, not a demonstrated production search bug.', '',
    'Both exact and candidate paths now request k+1 and expand when the score at rank 10 equals the last retrieved score, until the entire boundary tie is covered. Both sort by score descending then internal offset ascending and truncate to k=10. Recall remains canonical ID Recall@10, not a post-hoc score-tolerance metric. Full-probe results must match exact IDs AND scores. This benchmark-only canonicalization, including repeated scoring for ties, is included in measured latency. It is not identical to an ordinary production k=10 call.', '',
    f"Tie-related additional scoring calls across ground truth, correctness checks and timing: {done['tie_expansion_calls']}. The same split, frozen models, validation selection rule and two-trial schedule are retained. No performance-based tuning followed the failure.", '',
    'Existing centroid/affine controls compute with f64 parameters; native MLP inference uses the existing f32 implementation. Contemporary paired timing compares these implementations, not a proof against a maximally optimized centroid kernel.', '']
(root/'Phase_F3_Report.md').write_text('\n'.join(lines))
print(json.dumps({'summary':summary,'comparisons':comparisons,'protected_files':len(preserved)},indent=2))
