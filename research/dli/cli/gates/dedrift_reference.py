"""G5 (deterministic parts): run cli_ablation's DedriftReclusteringStrategy selection, read-only, on the size vectors of
the Rust test `dedrift_selection_matches_cli_ablation`. Usage: python -I dedrift_reference.py <copy of codes/cli/cli_ablation>.
k-means and plotting imports are stubbed; the selection path never calls them. recluster_k_fraction = 0.5 (DEDRIFT 50%).
"""
import sys, types, importlib, torch
class _Stub(types.ModuleType):
    def __getattr__(self, name): return _Stub(name)
for m in ("faiss","matplotlib","matplotlib.pyplot","sklearn","sklearn.cluster","tqdm","seaborn","pandas","h5py","yaml","wandb"):
    try: importlib.import_module(m)
    except Exception: sys.modules[m] = _Stub(m)  # stubs: the selection path never calls them
sys.path.insert(0, sys.argv[1])
R = importlib.import_module("cli.reclustering")
captured = {}
def fake_build(cli, affected):
    captured["a"] = list(affected); return None, list(affected), 0
R.build_reclustering_train_dataset = fake_build
R.finalize_train_dataset = lambda ds, a, n, nb: None
def run(sizes, b_max):
    cli = types.SimpleNamespace(n_buckets=len(sizes), config={"max_bucket_size": b_max, "recluster_k_fraction": 0.5},
                                bucket_indices=[torch.zeros(s) for s in sizes])
    captured.clear(); R.DedriftReclusteringStrategy().recluster(cli); return captured.get("a", [])
for sizes, b in (([80,10,90,5,70,20],100), ([50,60,70,80,300],1000), ([100,100,1,1],50), ([10]*9+[11],1000), ([0,0,0,5],1), ([70,60,50,40,30,20,10],1000)):
    print(sizes, b, run(sizes, b))
