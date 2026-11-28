"""Known-answer test for analyze_pilot.py.

W1, 3 batches (0-2), 2 queries, budgets 1,000 / 2,000 plus the gate budget. Candidate 500: G1 0.8 then 1.0 -> W = 1,500.
Candidate 1000: 0.9 at 1,000 (upper bound) -> W = 1,000 -> chosen. Both reach 1.0 at the gate budget.
A third run with G1 0.9 (27/30) at the gate budget fails the known answer.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_pilot as A  # noqa: E402


def run(root, name, r1, r2, gate=1.0):
    d = root / name
    d.mkdir()
    with open(d / "rows.jsonl", "w") as f:
        for t in range(3):
            for q in (0, 1):
                for b, r in ((1000, r1), (2000, r2), (2_000_000, gate)):
                    f.write(json.dumps({"batch": t, "set": "val", "query": q, "budget": b, "scored": min(b, 5000),
                                        "recall30": 0, "recall30_g1": round(r * 30)}) + "\n")
    with open(d / "batches.jsonl", "w") as f:
        for t in range(3):
            f.write(json.dumps({"batch": t, "batch_seconds": 1.0, "maintenance": {"train_seconds": 2.0}}) + "\n")
    return str(d)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    r = A.analyze({"W1:500": run(root, "a", 0.8, 1.0), "W1:1000": run(root, "b", 0.9, 1.0)})
    assert r["choice"]["b_max"] == 1000 and r["choice"]["batches_used"] == [1, 2], r["choice"]
    assert abs(r["choice"]["mean_w090"]["W1:500"] - 1500) < 1e-9
    assert r["runs"]["W1:1000"]["bracketing"] == {"val": [(0, "upper_bound"), (1, "upper_bound"), (2, "upper_bound")]}
    assert all(v["gates"]["pass"] for v in r["runs"].values())
    bad = A.analyze({"W1:2000": run(root, "c", 0.8, 1.0, gate=0.9)})
    assert not bad["runs"]["W1:2000"]["gates"]["known_answer"]
print("analyze_pilot known-answer test: OK")
