"""Known-answer tests for analyze_tiebreak.py: identical recall curves, `calibrated` scores
`scale` times the work of `proportional` at every budget, so R = scale exactly. Checks the
relabelled decisions and the frozen choice rule (calibrated only if decided better)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_tiebreak as T  # noqa: E402

BUDGETS = [1000, 5000, 20000, 50000, 80000, 160000]


def write(root, name, scale, batches=4, queries=300):
    rng = np.random.default_rng(3)
    base = rng.uniform(0.5, 0.75, size=queries)
    d = root / name
    for b in range(batches):
        (d / f"batch_{b:02d}").mkdir(parents=True)
        with open(d / f"batch_{b:02d}" / "rows.jsonl", "w") as f:
            for policy, s in (("proportional", 1.0), ("calibrated", scale), ("mdn_code", 1.1)):
                for i, budget in enumerate(BUDGETS):
                    for q in range(queries):
                        r = min(1.0, base[q] + 0.06 * i)
                        f.write(json.dumps({
                            "set": "partb", "query": q, "policy": policy, "budget": budget,
                            "recall30": r * 30, "scored_total": budget * s, "plain_scored": 0,
                            "dead_offsets": 0, "unique_offsets": budget * s, "buckets_per_level": {"1": 1},
                            "unselected_level_rank": [], "miss_plain_unselected": 0,
                            "miss_selected_but_lost": 0}) + "\n")
    return d


assert T.choose("calibrated better") == "calibrated"
for other in ("proportional better", "equivalent", "inconclusive", "difference below the smallest effect of interest"):
    assert T.choose(other) == "proportional", other

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    better = T.analyze(write(root, "better", 0.9), "partb", reference=write(root, "ref", 0.9), resamples=500)
    assert better["gate_problems"] == [], better["gate_problems"]
    assert abs(better["primary"]["R_cal_over_prop"] - 0.9) < 1e-9, better["primary"]
    assert better["primary"]["decision"] == "calibrated better" and better["choice"] == "calibrated"
    assert better["S5_reproduction"]["mdn_code"]["identical_query_share"] == 1.0
    assert better["S5_reproduction"]["calibrated"]["identical_query_share"] == 1.0
    eq = T.analyze(write(root, "eq", 1.01), "partb", resamples=500)
    assert eq["primary"]["decision"] == "equivalent" and eq["choice"] == "proportional", eq["primary"]
    worse = T.analyze(write(root, "worse", 1.2), "partb", resamples=500)
    assert worse["primary"]["decision"] == "proportional better" and worse["choice"] == "proportional"
    print("analyze_tiebreak known-answer test: OK", better["primary"]["R_cal_over_prop"], better["primary"]["ci95"])
