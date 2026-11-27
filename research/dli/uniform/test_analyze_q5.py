"""Known-answer test for analyze_q5.py.

Synthetic runs: 3 batches, 4 queries, budgets 10K / 50K / 100K, recall@30 identical across queries.
* Recall at 50K: E1 0.5, E1R 0.9, E1T 0.8. So E1T−E1 = 0.3, E1R−E1 = 0.4, recovery 0.75, all CIs degenerate.
* Builds (identical schedule, points 10/20/30/40, each trained build 1.0 s):
  - E1 trains only build 0;
  - E1R trains all four;
  - E1T trains builds 0 and 2 (one retrain decision).
  So the trainer ratio is 2/4 = 0.5 and criterion 2 holds (<=).
* E1T's decisions are: reuse of router r0 (build 1), a retrain (build 2), reuse of router r2 (build 3).
* E1R's W@0.90 is the 50K point exactly (curve 0.8, 0.9, 1.0 → interpolated at fraction 1).
* Second case: E1T identical to E1 with only reuse decisions, so the verdict is "not met (trigger never fired)".
* Third case: a reuse decision naming the wrong router fails the decision gate.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_q5 as Q  # noqa: E402

BUDGETS = [10_000, 50_000, 100_000]
CURVE = {"E1": [0.4, 0.5, 0.6], "E1R": [0.8, 0.9, 1.0], "E1T": [0.7, 0.8, 0.9]}
TRAINED = {"E1": [True, False, False, False], "E1R": [True] * 4, "E1T": [True, False, True, False]}
ROUTERS = ["r0", "r1", "r2", "r3"]


def write(root, arm, curve, trained, decisions):
    d = root / arm
    for t in range(3):
        (d / f"batch_{t:02d}").mkdir(parents=True)
        with open(d / f"batch_{t:02d}" / "rows.jsonl", "w") as f:
            for q in range(4):
                for b, r in zip(BUDGETS, curve):
                    f.write(json.dumps({"set": "mq", "query": q, "policy": "mdn_code", "budget": b,
                                        "recall30": round(30 * r), "scored_total": b}) + "\n")
        (d / f"batch_{t:02d}" / "topology.json").write_text(json.dumps(
            {"members": [{"level": 2, "bucket_sizes": [8, 0, 0, 0]}, {"level": 0, "bucket_sizes": None}]}))
    with open(d / "build-events.jsonl", "w") as f:
        for i, (pts, tr) in enumerate(zip([10, 20, 30, 40], trained)):
            reused = None if tr else ROUTERS[i - 1 if i != 3 else 2]
            f.write(json.dumps({"stage": "build_completed", "physical_count": pts, "n_buckets": 10,
                                "reused_router": reused, "router_fingerprint": ROUTERS[i] if tr else reused,
                                "sample_read_seconds": 0.25 if tr else 0.0, "teacher_mlp_seconds": 0.75 if tr else 0.0}) + "\n")
    with open(d / "batches.jsonl", "w") as f:
        for t in range(3):
            rec = {"batch": t}
            if t == 1 and decisions:
                rec["trigger_decisions"] = decisions
            f.write(json.dumps(rec) + "\n")
    return str(d)


def dec(job, retrain, router, level=2):
    return {"job": job, "target_level": level, "target_uuid": "u", "target_live": 10, "buckets": 10,
            "max_bucket": 9, "max_over_mean": 9.0 if retrain else 2.0, "retrain": retrain,
            "target_router": router, "census_micros": 5}


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    good = [dec(1, False, "r0"), dec(2, True, "r1"), dec(3, False, "r2")]
    runs = {arm: write(root / "a", arm, CURVE[arm], TRAINED[arm], good if arm == "E1T" else []) for arm in Q.ARMS}
    r = Q.analyze(runs, "mq")
    assert abs(r["E1T-E1"]["point"] - 0.3) < 1e-6 and abs(r["E1R-E1"]["point"] - 0.4) < 1e-6, r
    assert abs(r["E1T-E1"]["ci95"][0] - 0.3) < 1e-6 and abs(r["E1T-E1"]["sensitivity_ci95"][1] - 0.3) < 1e-6
    assert abs(r["recovery"] - 0.75) < 1e-6, r["recovery"]
    assert r["training"]["E1T"]["trained_seconds"] == 2.0 and r["training"]["E1R"]["trained_seconds"] == 4.0
    assert r["training"]["E1R"]["first_fit_seconds"] == 1.0
    assert r["trainer_ratio_E1T_over_E1R"] == 0.5
    assert r["criterion_1"] and r["criterion_2"] and r["verdict"] == "met", r["verdict"]
    assert r["decisions"]["firing_batches"] == [1] and r["decisions"]["check"]["pass"], r["decisions"]["check"]
    assert r["decisions"]["max_ratio_eligible"] == 9.0 and r["decisions"]["census_micros_total"] == 15
    w = r["per_batch"]["E1R"][0]
    assert w["w090"] == 50_000 and w["w090_kind"] == "interpolated" and abs(w["recall_50k"] - 0.9) < 1e-6, w
    assert r["per_batch"]["E1"][0]["w090_kind"] == "not_attained"
    assert r["imbalance"]["E1"][0] == {2: 4.0}

    # Never fires: E1T = E1 in every row and build, reuse decisions only.
    never = [dec(1, False, "r0"), dec(2, False, "r1"), dec(3, False, "r2")]
    t2 = {"E1": TRAINED["E1"], "E1R": TRAINED["E1R"], "E1T": TRAINED["E1"]}
    runs = {arm: write(root / "b", arm, CURVE["E1"] if arm == "E1T" else CURVE[arm], t2[arm],
                       never if arm == "E1T" else []) for arm in Q.ARMS}
    r = Q.analyze(runs, "mq")
    assert r["E1T-E1"]["point"] == 0 and r["recovery"] == 0 and not r["criterion_1"], r["E1T-E1"]
    assert r["verdict"] == "not met (trigger never fired)", r["verdict"]
    # In the never-fire build file, builds 1-3 reuse r0, r1, r2 in turn, matching the decisions.
    assert r["decisions"]["check"]["pass"], r["decisions"]["check"]

    # A reuse decision naming the wrong router fails the gate.
    bad = [dec(1, False, "zz"), dec(2, True, "r1"), dec(3, False, "r2")]
    runs = {arm: write(root / "c", arm, CURVE[arm], TRAINED[arm], bad if arm == "E1T" else []) for arm in Q.ARMS}
    chk = Q.analyze(runs, "mq")["decisions"]["check"]
    assert not chk["pass"] and len(chk["problems"]) == 2, chk
print("analyze_q5 known-answer test: OK")
