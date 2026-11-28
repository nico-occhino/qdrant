"""Known-answer test for analyze_control.py.

Two batches, two queries, budgets 1,000 and 2,000; both queries score the budget exactly.
* MLP: recall 0.8 then 1.0 -> W@0.90 = 1,500 in both batches.
* Centroid: recall 0.9 at 1,000 (upper bound) in batch 0, and 0.7 / 0.9 in batch 1 -> W = 2,000.
R = mean(1,000, 2,000) / mean(1,500, 1,500) = 1.0. Identical queries make every resample equal, so the CI is
[1.0, 1.0].
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(root, name, recalls):
    d = root / name
    d.mkdir()
    with open(d / "rows.jsonl", "w") as f:
        for t, (r1, r2) in enumerate(recalls):
            for q in (0, 1):
                for b, r in ((1000, r1), (2000, r2)):
                    f.write(json.dumps({"batch": t, "set": "val", "query": q, "budget": b, "scored": b,
                                        "recall30": round(r * 30)}) + "\n")
    m = {"split_buckets": 1, "path": "split", "trigger": "None", "train_seconds": 1.0, "l_route": 0.1,
         "max_over_mean_before": 2.0}
    with open(d / "batches.jsonl", "w") as f:
        for t in range(2):
            f.write(json.dumps({"batch": t, "buckets": 10, "maintenance": m}) + "\n")
    return d


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    mlp = run(root, "mlp", [(0.8, 1.0), (0.8, 1.0)])
    cen = run(root, "cen", [(0.9, 1.0), (0.7, 0.9)])
    out = root / "a.json"
    subprocess.run([sys.executable, "-I", str(HERE / "analyze_control.py"), "--mlp", str(mlp), "--centroid", str(cen),
                    "--out", str(out), "--resamples", "50"], check=True, capture_output=True)
    r = json.loads(out.read_text())
    assert r["per_batch"]["0"]["mlp"]["w090"] == 1500 and r["per_batch"]["0"]["centroid"]["kind"] == "upper_bound", r
    assert r["per_batch"]["1"]["centroid"]["w090"] == 2000, r
    assert abs(r["R_centroid_over_mlp"] - 1.0) < 1e-12 and r["ci95"] == [1.0, 1.0], r
print("analyze_control known-answer test: OK")
