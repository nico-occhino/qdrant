"""Known-answer test for analyze_integration.py on a synthetic run directory.

Fixture: 3 batches, panel `val`, 4 queries, budgets {1000, 2_000_000}. `proportional_native`
equals `proportional` except one recall difference (batch 1, query 2) and, in a second fixture,
one work mismatch. Publications: a buffer drain, then a single-source rewrite of a learned
segment (a vacuum). The checks: row counts, mismatch and difference detection, the known-answer
minimum, vacuum detection, the dead-fraction gate and the consolidation shape."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_integration as I  # noqa: E402


def write(root, work_mismatch=False):
    d = root / ("bad" if work_mismatch else "ok")
    pubs = [[{"source_uuids": ["buf0"], "destination_uuid": "L1a"}],
            [{"source_uuids": ["L1a"], "destination_uuid": "L1b"}],
            []]
    for b in range(3):
        (d / f"batch_{b:02d}").mkdir(parents=True)
        with open(d / f"batch_{b:02d}" / "rows.jsonl", "w") as f:
            for policy in ("mdn_code", "proportional", "proportional_native"):
                for budget in (1000, 2_000_000):
                    for q in range(4):
                        recall = 30 if budget == 2_000_000 else 25
                        work = 1000 if budget == 1000 else 5000
                        if policy == "proportional_native" and b == 1 and q == 2 and budget == 2_000_000:
                            recall = 29
                        if work_mismatch and policy == "proportional_native" and b == 2 and q == 0:
                            work += 1
                        f.write(json.dumps({"set": "val", "query": q, "policy": policy, "budget": budget,
                                            "recall30": recall, "scored_total": work, "dead_offsets": 0,
                                            "unique_offsets": work}) + "\n")
    with open(d / "batches.jsonl", "w") as f:
        for b in range(3):
            f.write(json.dumps({"batch": b, "publications": pubs[b],
                                "segments": [{"uuid": "L1b", "appendable": False, "live": 90, "deleted": 10},
                                             {"uuid": "buf", "appendable": True, "live": 5, "deleted": 0}],
                                "levels": [{"level": 1, "live": 20, "buckets": 10},
                                           {"level": 2, "live": 100, "buckets": 100}]}) + "\n")
    return d


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    out = I.analyze(write(root), None, "constant_r", ["val"], resamples=50)
    n = out["native_vs_control"]
    assert n["rows"] == 3 * 2 * 4, n["rows"]
    assert n["work_mismatches"] == [], n["work_mismatches"]
    assert n["recall_differences"] == [[1, "val", 2, 2_000_000, 30, 29]], n["recall_differences"]
    ka = out["known_answer"]["val"]
    assert abs(ka["min"] - (3 * 30 + 29) / (4 * 30)) < 1e-12 and ka["pass_raw"] is False, ka
    m = out["mechanisms"]
    assert m["single_source_rewrites_of_learned_levels"] == 1, m
    assert abs(m["worst_dead_fraction_after_settle"] - 0.10) < 1e-12 and m["dead_gate"] is True, m
    assert m["consolidation_shape"] is True, m  # 20 <= 0.25 * 100
    bad = I.analyze(write(root, work_mismatch=True), None, "constant_r", ["val"], resamples=50)
    assert len(bad["native_vs_control"]["work_mismatches"]) == 2, bad["native_vs_control"]["work_mismatches"]
    print("analyze_integration known-answer test: OK")
