"""Isolated Phase E E2 deferred/rebuild/named-index lifecycle acceptance test (Python is only the test driver)."""
import argparse, hashlib, json, os, signal, subprocess, tempfile, time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/debug/qdrant')
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=16433)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='qdrant-lmi-phase-e2-lifecycle-'))
    base = f'http://127.0.0.1:{args.port}'
    config = root / 'config.yaml'
    config.write_text(f'''storage:
  storage_path: {root}/storage
  snapshots_path: {root}/snapshots
  optimizers:
    default_segment_number: 1
    indexing_threshold_kb: 1
service:
  host: 127.0.0.1
  http_port: {args.port}
  grpc_port: {args.port + 1}
cluster:
  enabled: false
telemetry_disabled: true
''')
    def request(method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = Request(base + path, data=data, method=method, headers={'Content-Type':'application/json'})
        with urlopen(req, timeout=15) as response:
            return json.load(response)
    # Refuse to use a port already serving another process.
    import socket
    for port in (args.port, args.port + 1):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    proc = None
    logfile = None
    def start(number, extra_args=()):
        nonlocal proc, logfile
        logfile = (out / f'server-{number}.log').open('w')
        proc = subprocess.Popen([str(Path(args.binary).resolve()), '--config-path', str(config), '--disable-telemetry', *extra_args], stdout=logfile, stderr=subprocess.STDOUT)
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError(f'server exited: {proc.returncode}; see server-{number}.log')
            try:
                request('GET', '/collections')
                return
            except (URLError, TimeoutError):
                time.sleep(.25)
        raise RuntimeError('server readiness timeout')
    def stop():
        nonlocal proc, logfile
        if proc is not None:
            proc.send_signal(signal.SIGTERM)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            proc = None
        if logfile: logfile.close(); logfile = None
    evidence = {'storage': str(root), 'cases': []}
    def wait_for(predicate, label):
        for _ in range(240):
            value = predicate()
            if value:
                return value
            time.sleep(.25)
        raise AssertionError(f'timeout: {label}')
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        start(1)
        cfg = {'n_buckets':2, 'sample_size':32, 'hidden_dim':8, 'epochs':60,
               'batch_size':8, 'kmeans_iterations':8, 'nprobe':1, 'seed':42}
        records = []
        for mixed in (False, True):
            name = 'mixed' if mixed else 'solo'
            endpoint = '/collections/' + name
            vectors = {'learned': {'size':2, 'distance':'Dot', 'lmi_config':cfg}}
            if mixed:
                vectors['graph'] = {'size':2, 'distance':'Dot', 'hnsw_config':{'m':8}}
            create = {'vectors':vectors, 'hnsw_config':{'m':0, 'payload_m':0},
                      'optimizers_config':{'indexing_threshold':1, 'default_segment_number':1,
                          'prevent_unoptimized':True, 'flush_interval_sec':1,
                          'deleted_threshold':.05, 'vacuum_min_vector_number':100}, 'shard_number':1}
            request('PUT', endpoint, create)
            before_config = request('GET', endpoint)['result']['config']
            collection_path = root / 'storage' / 'collections' / name
            def state_files():
                return list(collection_path.glob('0/segments/*/vector_index-learned/lmi_state.json'))
            def point(i, flipped=False):
                sign = (1 if i%2==0 else -1) * (-1 if flipped else 1)
                v = [sign*(2+i/100), .1+(i%5)/20]
                return {'id':i, 'vector':{k:v for k in vectors}, 'payload':{'group':'all'}}
            def query(v, using='learned', **extra):
                return request('POST', endpoint+'/points/query', {'query':v,'using':using,'limit':1000,**extra})['result']['points']
            def exact_ids():
                return {p['id'] for p in query([3,.1], params={'exact':True})}
            request('PUT',endpoint+'/points?wait=true',{'points':[point(i) for i in range(300)]})
            wait_for(lambda: state_files() and request('GET',endpoint)['result']['status']=='green', 'initial LMI')
            original = {p: digest(p) for p in state_files()}
            assert len(original) == 1, original
            original_json = json.loads(next(iter(original)).read_text())
            # Keep the optimizer paused while writes fill the mutable segment,
            # so visibility before deferred promotion can be observed.
            request('PATCH',endpoint,{'optimizers_config':{'max_optimization_threads':0}})
            request('PUT',endpoint+'/points?wait=false',{'points':[point(i) for i in range(300,600)]})
            visible_before = wait_for(lambda: (ids if len(ids := exact_ids()) > 300 else None), 'mutable inserts applied')
            assert len(visible_before) < 600, 'deferred points must not be visible before optimization'
            for p, expected in original.items():
                assert digest(p) == expected, 'persisted LMI was mutated by append'
            # Flip the sign of existing vectors and remove other old IDs.
            request('PUT',endpoint+'/points?wait=false',{'points':[point(i,True) for i in range(20)]})
            request('POST',endpoint+'/points/delete?wait=false',{'points':list(range(20,50))})
            for p, expected in original.items():
                assert digest(p) == expected, 'persisted LMI was mutated by update/delete'
            request('PATCH',endpoint,{'optimizers_config':{'max_optimization_threads':1}})
            expected_ids = set(range(600)) - set(range(20,50))
            def rebuilt():
                files = state_files()
                info = request('GET',endpoint)['result']
                return (files and not any(p in original for p in files)
                        and info['status']=='green' and exact_ids()==expected_ids and files)
            files = wait_for(rebuilt, 'new generation publication and old learned-state retirement')
            new_hashes = {str(p):digest(p) for p in files}
            assert not set(new_hashes.values()).intersection(original.values())
            states = [json.loads(p.read_text()) for p in files]
            assert sum(sum(map(len, st['postings'])) for st in states) == len(expected_ids)
            for st in states:
                offsets = [v for bucket in st['postings'] for v in bucket]
                assert len(offsets)==len(set(offsets))
                assert all(0 <= v < st['total_vectors'] for v in offsets)
                assert st['router'] != original_json['router'], 'fixture should train a new model'
            learned_a, learned_b = query([3,.1]), query([-3,.1])
            seen = {p['id'] for p in learned_a+learned_b}
            assert seen == expected_ids, (len(seen),len(expected_ids))
            assert not {p['id'] for p in learned_a}.intersection(p['id'] for p in learned_b)
            expected_vectors = {i:point(i,i<20)['vector']['learned'] for i in expected_ids}
            for q, rows in (([3,.1],learned_a), ([-3,.1],learned_b)):
                for row in rows:
                    vector = expected_vectors[row['id']]
                    assert abs(row['score']-sum(a*b for a,b in zip(q,vector))) < 1e-4
            assert 599 in seen
            assert request('GET',endpoint)['result']['config']['params']['vectors'] == before_config['params']['vectors']
            if mixed:
                for p in files:
                    segment_path = p.parent.parent
                    segment_state = json.loads((segment_path/'segment.json').read_text())
                    # Actual graph files and segment index config, not only API config.
                    assert (segment_path/'vector_index-graph'/'graph.bin').is_file()
                    assert not (segment_path/'vector_index-graph'/'lmi_state.json').exists()
                    assert segment_state['config']['vector_data']['learned']['index'].get('type') == 'lmi_trained', segment_state
                    assert segment_state['config']['vector_data']['graph']['index'].get('type') == 'hnsw', segment_state
                graph = query([3,.1], using='graph', params={'exact':True})
                assert {p['id'] for p in graph} == expected_ids
            records.append((name,learned_a,learned_b,new_hashes))
            evidence['cases'].append({'name':name,'global_hnsw_m':0,'deferred_points_hidden':True,'visible_before_promotion':len(visible_before),
                'old_state_immutable':True,'old_state_retired':True,'new_state_hashes':new_hashes,
                'expected_live_points':len(expected_ids),'new_models':True,'postings_unique_in_target_range':True,
                'updated_scores_correct':True,'deleted_ids_absent':True,'named_indexes_independent':mixed})
        stop(); start(2)
        for name, a, b, hashes in records:
            endpoint='/collections/'+name
            assert query([3,.1]) == a
            assert query([-3,.1]) == b
            for path, expected in hashes.items():
                assert digest(Path(path)) == expected
        stop()
        restart_log=(out/'server-2.log').read_text()
        assert 'LMI build:' not in restart_log
        assert 'LMI open: mode=StaticLearned; no training' in restart_log
        assert 'candidate_source=StaticLearned' in restart_log
        evidence.update(restart_no_training=True,restart_same_state_ids_scores=True)
        (out/'result.json').write_text(json.dumps(evidence,indent=2))
        print(json.dumps(evidence,indent=2))
    finally:
        stop()

if __name__ == '__main__': main()
