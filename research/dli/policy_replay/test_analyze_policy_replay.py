"""Known-answer tests for analyze_policy_replay.py.

A fixture run holds both policies with identical recall curves; `proportional` scores `scale`
times the work of `mdn_code` at every budget, so R = scale exactly. Checks: the ratio and its
CI, every decision branch, the S1 recall difference (0 by construction), the level
diagnostics, the reproduction check, and the gates (non-monotone work, missing policy)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_policy_replay as P  # noqa: E402

BUDGETS = [1000, 5000, 20000, 50000, 80000, 160000]


def write(root, name, scale, batches=4, queries=300, drop_policy=None, break_monotone=False):
    rng = np.random.default_rng(7)
    base = rng.uniform(0.5, 0.75, size=queries)
    d = root / name
    for b in range(batches):
        (d / f"batch_{b:02d}").mkdir(parents=True)
        with open(d / f"batch_{b:02d}" / "rows.jsonl", "w") as f:
            for policy, s in (("mdn_code", 1.0), ("proportional", scale)):
                if policy == drop_policy:
                    continue
                for i, budget in enumerate(BUDGETS):
                    work = budget * s
                    if break_monotone and policy == "mdn_code" and i == 3:
                        work = 1.0
                    for q in range(queries):
                        r = min(1.0, base[q] + 0.06 * i)
                        f.write(json.dumps({
                            "set": "test", "query": q, "policy": policy, "budget": budget,
                            "recall30": r * 30, "scored_total": work, "plain_scored": 100 if policy == "proportional" else 0,
                            "dead_offsets": 0, "unique_offsets": work, "buckets_per_level": {"1": 2, "2": 3},
                            "unselected_level_rank": [[2, 1]] if r < 1 else [], "miss_plain_unselected": 0,
                            "miss_selected_but_lost": 0}) + "\n")
    return d


assert P.decide(0.90, [0.88, 0.92]) == "proportional better"
assert P.decide(1.10, [1.07, 1.13]) == "mdn_code better"
assert P.decide(1.00, [0.97, 1.03]) == "equivalent"
assert P.decide(0.97, [0.94, 0.99]) == "difference below the smallest effect of interest"
assert P.decide(0.97, [0.90, 1.04]) == "inconclusive"

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    run = write(root, "run", 0.9)
    out = P.analyze(run, "test", reference=run, resamples=1000)
    pr = out["primary"]
    assert out["gate_problems"] == [], out["gate_problems"]
    assert abs(pr["R_prop_over_mdn"] - 0.9) < 1e-9, pr
    assert pr["ci95"][0] > 0.899 and pr["ci95"][1] < 0.901, pr["ci95"]
    assert pr["decision"] == "proportional better" and pr["status"] == "ok", pr
    assert abs(out["S1_delta_recall_at_Bstar"]["delta_pp"]) < 1e-12
    lv = out["S3_S4_levels_at_Bstar"]
    assert lv["proportional"]["mean_buckets_per_level"] == {"1": 2.0, "2": 3.0}
    assert abs(lv["proportional"]["buffer_share_of_work"] - 100 / (50000 * 0.9)) < 1e-12
    assert out["S5_reproduction"]["identical_query_share"] == 1.0
    eq = P.analyze(write(root, "eq", 1.01), "test", resamples=500)["primary"]
    assert eq["decision"] == "equivalent", eq
    bad = P.analyze(write(root, "bad", 0.9, break_monotone=True), "test", resamples=50)
    assert any("not monotone" in p for p in bad["gate_problems"]), bad["gate_problems"]
    miss = P.analyze(write(root, "miss", 0.9, drop_policy="proportional"), "test", resamples=50)
    assert any("missing" in p for p in miss["gate_problems"]), miss["gate_problems"]
    print("analyze_policy_replay known-answer test: OK", pr["R_prop_over_mdn"], pr["ci95"])
