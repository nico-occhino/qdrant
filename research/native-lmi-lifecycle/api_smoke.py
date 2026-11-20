"""Disposable loopback REST smoke; run only with this worktree's built binary."""
import json, os, pathlib, signal, socket, subprocess, time, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent / 'api-smoke'
OUT.mkdir(exist_ok=False)
PORT = 16433
BINARY = pathlib.Path(os.environ['CARGO_TARGET_DIR']) / 'debug/qdrant'
BASE = f'http://127.0.0.1:{PORT}'
NAME = 'native_api_lmi'
records = []

def request(method, path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=payload, method=method,
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.load(response)
    records.append({'method':method,'path':path,'request':body,'response':result})
    (OUT/'requests-responses.json').write_text(json.dumps(records,indent=2)+'\n')
    assert result['status']=='ok',result
    return result['result']

def axis(a):
    v=[0.0]*16;v[a]=1.0;return v

def search(axes):
    v=[0.0]*16
    for a in axes:v[a]=1.0
    return request('POST',f'/collections/{NAME}/points/search',{'vector':v,'limit':2})

def start(number):
    env=os.environ.copy()
    env.update({'QDRANT__SERVICE__HOST':'127.0.0.1','QDRANT__SERVICE__HTTP_PORT':str(PORT),
                'QDRANT__SERVICE__GRPC_PORT':str(PORT+1),'QDRANT__SERVICE__MAX_WORKERS':'2',
                'QDRANT__STORAGE__STORAGE_PATH':str(OUT/'storage'),
                'QDRANT__STORAGE__SNAPSHOTS_PATH':str(OUT/'snapshots'),
                'QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS':'2',
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    log=(OUT/f'server-{number}.log').open('w')
    p=subprocess.Popen([str(BINARY),'--config-path',str(ROOT/'config/config.yaml'),'--disable-telemetry'],
                       cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT)
    for _ in range(300):
        assert p.poll() is None, (OUT/f'server-{number}.log').read_text()[-4000:]
        try:
            with urllib.request.urlopen(BASE+'/readyz',timeout=1) as r:
                if r.status==200:return p,log
        except OSError:pass
        time.sleep(.1)
    raise RuntimeError('service startup timed out')

def stop(p,log):
    p.send_signal(signal.SIGTERM)
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill();p.wait();raise
    log.close()
    assert p.returncode==0,p.returncode

for port in [PORT,PORT+1]:
    with socket.socket() as s:s.bind(('127.0.0.1',port))
p,log=start(1)
try:
    request('PUT',f'/collections/{NAME}',{
        'vectors':{'size':16,'distance':'Cosine','datatype':'float16','lmi_config':{
            'n_buckets':2,'sample_size':256,'hidden_dim':8,'epochs':1,'batch_size':64,
            'routing_batch_size':64,'kmeans_iterations':2,'nprobe':2,'seed':7}},
        'hnsw_config':{'max_indexing_threads':1},
        'optimizers_config':{'default_segment_number':1,'indexing_threshold':1,
                             'max_optimization_threads':0}})
    points=[{'id':10000+i*3,'vector':axis(i%8)} for i in range(10000)]
    points.append({'id':1000003,'vector':axis(15)})
    result=request('PUT',f'/collections/{NAME}/points?wait=true',{'points':points})
    assert result['status']=='completed',result
    request('PATCH',f'/collections/{NAME}',{'optimizers_config':{'max_optimization_threads':1}})
    for _ in range(600):
        info=request('GET',f'/collections/{NAME}')
        if info['indexed_vectors_count']==10001:break
        time.sleep(.2)
    else:raise RuntimeError('LMI publication timeout')
    states=list((OUT/'storage').rglob('lmi_state.json'))
    assert states,'no native learned artifacts'
    for state in states:
        meta=json.loads(state.read_text());assert meta['config']['nprobe']==2
    request('PATCH',f'/collections/{NAME}',{'optimizers_config':{'max_optimization_threads':0}})
    time.sleep(.5)
    result=request('PUT',f'/collections/{NAME}/points?wait=true',
                   {'points':[{'id':7000019,'vector':axis(14)}]})
    assert result['status']=='completed'
    assert {x['id'] for x in search([14,15])}=={1000003,7000019}
    result=request('PUT',f'/collections/{NAME}/points?wait=true',
                   {'points':[{'id':1000003,'vector':axis(13)}]})
    assert result['status']=='completed'
    assert {x['id'] for x in search([13,14])}=={1000003,7000019}
    result=request('POST',f'/collections/{NAME}/points/delete?wait=true',{'points':[7000019]})
    assert result['status']=='completed'
    assert search([13])[0]['id']==1000003
finally:stop(p,log)
p,log=start(2)
try:
    after=search([13]);assert after[0]['id']==1000003
    assert all(x['id']!=7000019 for x in after)
    summary={'status':'passed','service_binary':str(BINARY),'source_commit':'9b12d583b1bf9df802b2cc420244bd3c24732d86',
             'initial_points':10001,'configured_nprobe':2,'query_params':None,
             'restarted':True,'api_learned_counter_observation':'unavailable: not exposed by REST; internal Rust test verifies learned/fallback counters',
             'path_attribution':'eligible request plus persisted LMI configuration; source-derived, not independent API counter telemetry'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
finally:stop(p,log)
