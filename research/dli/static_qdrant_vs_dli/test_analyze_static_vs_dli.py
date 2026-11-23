"""Known-answer test for analyze_static_vs_dli.py: arm D needs exactly 1.5x the work of
arm S at every recall level -> ratio 1.5 with a tight CI; a curve that never reaches 0.90
in one batch is reported as not attained (no invented value)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_static_vs_dli as A  # noqa: E402


def write(root, name, policy, scale, batches=4, queries=300, cap_batch=None):
    rng = np.random.default_rng(1)
    base = rng.uniform(0.5, 0.75, size=queries)
    d = root / name
    for b in range(batches):
        (d / f"batch_{b:02d}").mkdir(parents=True)
        with open(d / f"batch_{b:02d}" / "rows.jsonl", "w") as f:
            for i, knob in enumerate([1, 2, 4, 8, 16, 32]):
                for q in range(queries):
                    r = min(1.0, base[q] + 0.06 * i)
                    if cap_batch == b:
                        r = min(r, 0.8)
                    f.write(json.dumps({"set": "test", "query": q, "policy": policy, "budget": knob * (1000 if policy == "mdn_code" else 1),
                                        "recall30": r * 30, "scored_total": 1000 * knob * scale,
                                        "dead_offsets": 0, "unique_offsets": 1000 * knob}) + "\n")
    return d


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    s = write(root, "s", "static_nprobe", 1.0)
    d = write(root, "d", "mdn_code", 1.5)
    out = A.analyze_pair(s, d, "test", resamples=2000)
    assert abs(out["ratio_D_over_S"] - 1.5) < 1e-9, out
    assert out["ci95"][0] > 1.49 and out["ci95"][1] < 1.51, out
    assert out["attained_share"] == 1.0 and out["status"] == "ok"
    d2 = write(root, "d2", "mdn_code", 1.5, cap_batch=2)
    out2 = A.analyze_pair(s, d2, "test", resamples=500)
    assert out2["not_attained_D"] == [2] and out2["per_batch"][2]["W_D"] is None, out2
    assert out2["attained_both"] == 3 and out2["status"] == "descriptive (attainment below 80%)", out2
    # A1: a curve already at target on its first grid point gives W = work(first point).
    s3 = write(root, "s3", "static_nprobe", 1.0, cap_batch=None)
    rows = (s3 / "batch_00" / "rows.jsonl").read_text().splitlines()
    import json as _j
    fixed = []
    for line in rows:
        r = _j.loads(line); r["recall30"] = 30.0; fixed.append(_j.dumps(r))
    (s3 / "batch_00" / "rows.jsonl").write_text("\n".join(fixed) + "\n")
    out3 = A.analyze_pair(s3, d, "test", resamples=200)
    assert out3["per_batch"][0]["S_upper_bound"] and out3["per_batch"][0]["W_S"] == 1000.0, out3["per_batch"][0]
    # A2: the fixture's largest point is not full recall (max 0.75 + 0.30 capped at 1.0 -> mean < 0.995).
    assert out["known_answer_pass"] is False and out["known_answer_min"]["S"] < 0.995
    print("analyze_static_vs_dli known-answer test: OK", out["ratio_D_over_S"], out["ci95"])
