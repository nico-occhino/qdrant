from pathlib import Path
import hashlib,json,random,subprocess,os
r=Path.cwd(); base=Path(os.environ.get('LMI_PHASE_F_DIR','/home/nicoo/work/lmi-phase-f-data')); out=Path(os.environ.get('LMI_PHASE_F2_DIR','/home/nicoo/work/lmi-phase-f2-data'))
assert base.resolve()!=out.resolve(), 'F.2 must have a separate directory'
out.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if not (out/'protocol.json').exists():
    q=list(range(20,220));random.Random(20260927).shuffle(q)
    disk=json.loads((base/'mlp_state.json').read_text())
    sample=disk['sample_offsets']; remainder=sorted(set(range(99780))-set(sample));random.Random(20260928).shuffle(remainder)
    protocol=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),baseline_hashes={p.name:sha(p) for p in base.iterdir() if p.is_file()},warmup=list(range(20)),validation=sorted(q[:100]),test=sorted(q[100:]),sample_order=sample+remainder,seeds=[42,43,44],sample_sizes=[2048,4096,8192,16384],epochs=[30,60,120],hidden_dim=64,nprobe=[1,2,4,8,16,32,64],trials=2,selection='Validation-only: at target recall .90 and .95, admit +/- .02, choose closest achieved recall then candidate count. Compare mean candidate fraction across seeds; missing bands incur 1.0. Choose lower epochs/sample on exact ties. Test is opened only after selected configuration is frozen.',split_seed=20260927,sample_extension_seed=20260928)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'jobs.json').write_text('[]')

print(out)
