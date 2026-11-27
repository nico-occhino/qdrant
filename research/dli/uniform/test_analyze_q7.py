"""Known-answer test for analyze_q7.py.

One predicate, 2 queries:
* S-LMI nprobe curve 1, 2, 4. Mean recall 0.5 / 0.8 / 1.0 and mean work 100 / 80 / 60,000.
  - The 1→2 drop is a widening restart: per-query drops of 20 < 30.
  - W@0.90 = 80 + (0.9 − 0.8)/0.2 · (60,000 − 80) = 30,040, interpolated.
  - R@30 at 50K is measured at nprobe 4 (1.0).
* HNSW ef curve 30, 45: recall 0.95 and 1.0, so W@0.90 is the ef=30 work (upper_bound) and 50K is unreachable.
* S-LMI exact: query 0 has recall 29/30, but its gold has a rank-30 tie, so it is admitted.
* Second case: the same, with no tie and a 1→2 drop of 40 per query. Both are problems.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_q7 as A  # noqa: E402


def write(root, tie, drop):
    root.mkdir()
    rows = []
    def row(arm, strat, q, rec, work):
        rows.append({"arm": arm, "strategy": strat, "predicate": "1%", "query": q, "recall30": rec, "underfill": 0,
                     "duplicates": 0, "nonqualifying": 0, "work": work, "plans": [{"plan": "RoutedIntersect"}]})
    for q in (0, 1):
        row("S-LMI", "nprobe=1", q, 0.5, 100)
        row("S-LMI", "nprobe=2", q, 0.8, 100 - drop)
        row("S-LMI", "nprobe=4", q, 1.0, 60_000)
        row("HNSW", "ef=30", q, 0.95, 500)
        row("HNSW", "ef=45", q, 1.0, 700)
        row("S-LMI", "exact", q, 29 / 30 if q == 0 else 1.0, 600)
    (root / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (root / "gold.jsonl").write_text("\n".join(json.dumps({"predicate": "1%", "query": q, "qualifying": 600,
                                                           "tie_at_boundary": tie and q == 0}) for q in (0, 1)) + "\n")


with tempfile.TemporaryDirectory() as tmp:
    good = Path(tmp) / "good"
    write(good, tie=True, drop=20)
    r = A.analyze(good)
    c = r["curves"]["S-LMI/nprobe/1%"]
    assert c["w090_kind"] == "interpolated" and abs(c["w090"] - 30_040) < 1e-6, c
    assert c["at_50k"] == "measured" and c["recall_at_50k"]["knob"] == 4
    h = r["curves"]["HNSW/ef/1%"]
    assert h["w090_kind"] == "upper_bound" and h["w090"] == 500 and h["at_50k"] == "unreachable", h
    assert len(r["widening_restarts"]) == 1 and r["pass"], r
    assert abs(r["summary"]["S-LMI/exact/1%"]["recall30"] - (29 / 30 + 1) / 2) < 1e-12
    bad = Path(tmp) / "bad"
    write(bad, tie=False, drop=40)
    r = A.analyze(bad)
    kinds = sorted(next(iter(p)) for p in r["problems"])
    assert not r["pass"] and kinds == ["exact_shortfall_without_tie", "non_monotone"], r["problems"]
print("analyze_q7 known-answer test: OK")
