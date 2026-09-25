"""Isolated Phase E REST/optimizer/restart/fresh-snapshot-restore smoke test (Python is only the test driver)."""
import argparse, hashlib, json, os, signal, subprocess, tempfile, time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/debug/qdrant')
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=16333)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='qdrant-lmi-phase-e-'))
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
    evidence = {'storage': str(root), 'port':args.port}
    try:
        start(1)
        cfg = {'n_buckets':2, 'sample_size':32, 'hidden_dim':8, 'epochs':60,
               'batch_size':8, 'kmeans_iterations':8, 'nprobe':1, 'seed':42}
        create = {'vectors':{'size':2, 'distance':'Dot', 'lmi_config':cfg},
                  'optimizers_config':{'indexing_threshold':1, 'default_segment_number':1}, 'shard_number':1}
        invalid = json.loads(json.dumps(create)); invalid['vectors']['lmi_config']['nprobe'] = 3
        try:
            request('PUT','/collections/invalid_lmi',invalid)
            raise AssertionError('invalid LMI config accepted')
        except HTTPError as err:
            assert err.code == 422, err.read().decode()
        request('PUT', '/collections/phase_e', create)
        before_config = request('GET', '/collections/phase_e')['result']['config']
        for hnsw in [{'m':8}, {}]:
            try:
                request('PATCH', '/collections/phase_e', {'vectors':{'':{'hnsw_config':hnsw}}})
                raise AssertionError('conflicting HNSW update accepted')
            except HTTPError as err:
                assert err.code == 400, err.read().decode()
            assert request('GET', '/collections/phase_e')['result']['config'] == before_config
        evidence['conflicting_patch_rejected'] = True
        evidence['rejected_patch_preserved_configuration'] = True
        points = [{'id':i, 'vector':[(1 if i%2==0 else -1)*(2+i/100), .1+(i%5)/20], 'payload':{'group':'all'}} for i in range(300)]
        request('PUT', '/collections/phase_e/points?wait=true', {'points':points})
        for _ in range(180):
            info = request('GET','/collections/phase_e')['result']
            files = list((root/'storage').rglob('lmi_state.json'))
            if info.get('indexed_vectors_count',0) >= 300 and files and info['status']=='green': break
            time.sleep(.5)
        else: raise AssertionError(f'optimizer did not publish LMI: {info}; files={files}')
        def query(vector, **extra):
            return request('POST', '/collections/phase_e/points/query', {'query':vector,'limit':1000,**extra})['result']['points']
        a=query([3,.1]); b=query([-3,.1]); default=query([3,.1],params={})
        exact=query([3,.1],params={'exact':True})
        filtered=query([3,.1],filter={'must':[{'key':'group','match':{'value':'all'}}]})
        ids=lambda rows:[r['id'] for r in rows]
        assert 0 < len(a) < 300 and 0 < len(b) < 300
        assert set(ids(a)).isdisjoint(ids(b))
        assert ids(a) == ids(default)
        assert len(exact) == len(filtered) == 300
        assert ids(a) == [p['id'] for p in exact if p['id'] in set(ids(a))]
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        snapshot=request('POST','/collections/phase_e/snapshots')['result']
        evidence.update(indexed_vectors=info['indexed_vectors_count'],positive_candidates=len(a),negative_candidates=len(b),exact_results=len(exact),filtered_results=len(filtered),snapshot=snapshot,model_hashes=hashes)
        stop(); start(2)
        assert request('GET', '/collections/phase_e')['result']['config'] == before_config
        evidence['restart_preserved_configuration'] = True
        assert ids(query([3,.1]))==ids(a)
        assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        stop()
        assert 'LMI build: training' not in (out/'server-2.log').read_text()
        evidence.update(restart_same_results=True,restart_same_state=True,restart_did_not_train=True)
        # The source process is stopped. Restore the actual snapshot into a new
        # directory; no source collection/storage directory is copied or reused.
        source_root = root
        snapshot_path = source_root / 'snapshots' / 'phase_e' / snapshot['name']
        assert snapshot_path.is_file(), snapshot_path
        root = Path(tempfile.mkdtemp(prefix='qdrant-lmi-phase-e2-restored-'))
        restored_config = root / 'config.yaml'
        restored_config.write_text(config.read_text().replace(str(source_root), str(root)))
        config = restored_config
        assert not (root / 'storage').exists()
        start(3, ['--snapshot', f'{snapshot_path}:phase_e'])
        assert request('GET', '/collections/phase_e')['result']['config'] == before_config
        restored_files = list((root / 'storage').rglob('lmi_state.json'))
        restored_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in restored_files}
        assert Counter(restored_hashes.values()) == Counter(hashes.values())
        # Compare IDs AND scores, but also require pruned results and the native
        # StaticLearned serving marker below: equality alone is insufficient.
        assert query([3,.1]) == a
        assert query([-3,.1]) == b
        assert query([3,.1], params={}) == default
        assert query([3,.1], params={'exact':True}) == exact
        assert query([3,.1], filter={'must':[{'key':'group','match':{'value':'all'}}]}) == filtered
        stop()
        restore_log = (out / 'server-3.log').read_text()
        assert 'LMI build:' not in restore_log, 'restore must not build or train LMI'
        assert 'LMI open: mode=StaticLearned; no training' in restore_log
        assert 'candidate_source=StaticLearned' in restore_log
        evidence.update(
            restored_storage=str(root), restored_model_hashes=restored_hashes,
            restore_source_stopped=True, restore_fresh_storage=True,
            restore_preserved_configuration=True, restore_same_state=True,
            restore_same_ids_and_scores=True, restore_did_not_build_or_train=True,
            restore_static_learned_open=True, restore_static_learned_search=True,
            restore_exact_and_filter_fallbacks=True,
        )
        (out/'result.json').write_text(json.dumps(evidence,indent=2))
        print(json.dumps(evidence,indent=2))
    finally:
        stop()

if __name__ == '__main__': main()
