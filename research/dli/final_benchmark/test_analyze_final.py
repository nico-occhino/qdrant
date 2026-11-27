"""Known-answer test for analyze_final.py.

Synthetic W1 runs, 3 batches (0–2), 4 test queries, budgets 300..3000 step 300. G1 recall is linear in budget:
* DLI-best 30·b/1500 (capped at 30), so W@0.90 = 1,350, interpolated between 1,200 and 1,500 at fraction 0.5;
* E1 and S-LMI 30·b/3000, so W = 2,700 at a grid point;
* HNSW floor(30·b/1350), so W = 1,275 (26/30 at 1,200, 30/30 at 1,500, fraction 0.25);
* IVF never reaches 0.90.

Expected:
* R(DLI-best, E1) = 0.5, Family A supported; vs S-LMI better on work; vs HNSW R = 1350/1275 = 1.059, worse;
  vs IVF not decidable.
* Latency = work in ns, so latency ratios use the first grid point at or above 0.90: DLI-best 1,500,
  S-LMI 2,700, HNSW 1,500, giving 0.556 and 1.0.
* Seed 2 is identical, so every verdict is replicated.
* Every gate passes except IVF's known answer (it never reaches full recall). A second manifest with a broken ledger and a native/control mismatch fails both gates.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_final as A  # noqa: E402

BUDGETS = list(range(300, 3001, 300))
SCALE = {"DLI-best": 1500, "E1": 3000, "S-LMI": 3000, "HNSW": 1350, "IVF": 10**9}


def g1(arm, b):
    return min(30, int(30 * b // SCALE[arm]))


def write_run(root, seed, arm, plan, break_ledger=False, break_control=False, panel="test", scale=None, tag=""):
    d = root / f"{seed}-{arm}{tag}"
    policy = A.ARM_POLICY[arm]
    for t in range(3):
        (d / f"batch_{t:02d}").mkdir(parents=True)
        with open(d / f"batch_{t:02d}" / "rows.jsonl", "w") as f, open(d / f"batch_{t:02d}" / "timed_rows.jsonl", "w") as ft:
            pols = [policy] + (["proportional", "mdn_code"] if arm == "DLI-best" else [])
            for pol in pols:
                for q in range(4):
                    for b in BUDGETS:
                        rec = min(30, int(30 * b // scale)) if scale else g1("E1" if pol == "mdn_code" else arm, b)
                        work = b + (1 if (break_control and pol == "proportional" and t == 1 and q == 0) else 0)
                        f.write(json.dumps({"set": panel, "query": q, "policy": pol, "budget": b, "recall30": rec,
                                            "recall30_g1": rec, "scored_total": work}) + "\n")
                        if pol == policy:
                            ft.write(json.dumps({"set": panel, "query": q, "policy": pol, "budget": b,
                                                 "elapsed_ns": b, "recall30": rec, "recall30_g1": rec}) + "\n")
                if arm == "HNSW":
                    for q in range(4):
                        f.write(json.dumps({"set": panel, "query": q, "policy": "hnsw_exact", "budget": 0,
                                            "recall30": 30, "recall30_g1": 30, "scored_total": 9999}) + "\n")
    with open(d / "batches.jsonl", "w") as f:
        for m in plan["batches"]:
            sha = "0" * 64 if break_ledger else m["live_sha256"]
            f.write(json.dumps({"live": m["live_count"], "live_sha256": sha}) + "\n")
    return str(d)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    plan = {"batches": [{"live_count": 100 * (t + 1), "live_sha256": hashlib.sha256(bytes([t])).hexdigest()}
                        for t in range(3)]}
    (root / "plan.json").write_text(json.dumps(plan))
    runs = [{"workload": "W1", "seed": s, "arm": arm, "plan": str(root / "plan.json"),
             "dir": write_run(root, s, arm, plan)}
            for s in (1, 2) for arm in ("DLI-best", "E1", "S-LMI", "IVF", "HNSW")]
    res = A.analyze({"runs": runs}, resamples=200)

    w, info = A.batch_w(A.load_rows(runs[0]["dir"], "rows.jsonl", "proportional_native", "test"), 1)
    assert abs(w - 1350) < 1e-9 and info["kind"] == "interpolated" and abs(info["fraction"] - 0.5) < 1e-12, (w, info)
    assert A.interp([0.95, 1.0], [10, 20]) == (10.0, {"kind": "upper_bound", "bracket": [0, 0]})
    assert A.interp([0.1, 0.5], [10, 20])[0] is None

    a = res["family_A"]["W1/1/test"]
    assert abs(a["R"] - 0.5) < 1e-12 and a["decision"] == "supported" and a["attained_both"] == 2, a
    assert a["ci95"] == [0.5, 0.5] and abs(a["bonferroni_level"] - 0.99) < 1e-12, a
    b = res["family_B"]
    assert b["W1/1/test/S-LMI"]["work"]["decision"] == "better"
    assert abs(b["W1/1/test/HNSW"]["work"]["R"] - 1350 / 1275) < 1e-12 and b["W1/1/test/HNSW"]["work"]["decision"] == "worse"
    assert b["W1/1/test/IVF"]["work"]["decision"] == "not decidable"
    assert abs(b["W1/1/test/S-LMI"]["latency"]["R"] - 1500 / 2700) < 1e-12, b["W1/1/test/S-LMI"]["latency"]
    assert abs(b["W1/1/test/HNSW"]["latency"]["R"] - 1.0) < 1e-12 and b["W1/1/test/HNSW"]["latency"]["decision"] == "equivalent"
    assert abs(res["secondary_allocation"]["W1/1/test"]["R"] - 0.5) < 1e-12
    assert all(v["replicated"] for v in res["replication"].values()) and len(res["replication"]) == 1 + 3 * 2
    for k, g in res["gates"].items():
        # The synthetic IVF never reaches full recall, so its known-answer gate must fail; all else passes.
        assert g["pass"] == ("IVF" not in k), (k, g)
        assert g["known_answer"] == ("IVF" not in k) and g["ledger"] and g["timed_matches_work"], (k, g)

    bad_root = root / "bad"
    bad_root.mkdir()
    bad = [{"workload": "W1", "seed": 1, "arm": "DLI-best", "plan": str(root / "plan.json"),
            "dir": write_run(bad_root, 1, "DLI-best", plan, break_ledger=True, break_control=True)}]
    gb = A.analyze({"runs": bad}, resamples=10)["gates"]["W1/1/DLI-best"]
    assert not gb["ledger"] and not gb["native_equals_control"] and gb["native_control_work_mismatches"] == len(BUDGETS), gb
    assert not gb["pass"]
    # Selection pilot: S-LMI with 250 buckets (W = 2,700) against 1,000 buckets (W = 1,350) on W1 `val`;
    # the 1,000-bucket configuration must be chosen, and both bracket 0.90 in every batch.
    pil = root / "pilot"
    pil.mkdir()
    pruns = [{"workload": "W1", "seed": 20261202, "arm": "S-LMI", "cfg": c, "plan": str(root / "plan.json"),
              "dir": write_run(pil, 20261202, "S-LMI", plan, panel="val", scale=sc, tag=f"-{c}")}
             for c, sc in ((250, 3000), (1000, 1500))]
    sel = A.selection({"mode": "pilot", "runs": pruns})
    assert sel["choice"]["S-LMI"]["cfg"] == 1000, sel["choice"]
    assert sel["choice"]["S-LMI"]["mean_W"] == {250: 2700.0, 1000: 1350.0}, sel["choice"]
    assert all(v["all_interpolated"] for v in sel["bracketing"].values()), sel["bracketing"]
    # Amendment A2: upper-bound rule. Synthetic ratio results with one batch whose W is an upper bound.
    def fake(decision, ub_a, ub_b):
        info = lambda ub: {"kind": "upper_bound" if ub else "interpolated"}
        return {"decision": decision, "per_batch": {1: {"A": 1.0, "B": 1.0, "A_info": info(False), "B_info": info(False)},
                                                    2: {"A": 1.0, "B": 1.0, "A_info": info(ub_a), "B_info": info(ub_b)},
                                                    3: {"A": None, "B": 1.0, "A_info": {"kind": "not_attained"},
                                                        "B_info": {"kind": "upper_bound"}}}}
    for d, ua, ub, want in [("worse", False, True, "worse"), ("better", False, True, "not decidable (upper bound)"),
                            ("equivalent", False, True, "not decidable (upper bound)"),
                            ("inconclusive", False, True, "not decidable (upper bound)"),
                            ("better", True, False, "better"), ("worse", True, False, "not decidable (upper bound)"),
                            ("worse", True, True, "not decidable (upper bound)"),
                            ("not supported", False, True, "not supported"),
                            ("supported", False, True, "not decidable (upper bound)"),
                            ("supported", True, False, "supported"), ("better", False, False, "better"),
                            ("descriptive", False, True, "descriptive")]:
        r = fake(d, ua, ub)
        got = A.bound_adjust(r)
        assert got == want, (d, ua, ub, got)
        # Batch 3 is not used (A not attained), so its upper bound never counts.
        assert r["upper_bound_batches"] == {"A": [2] if ua else [], "B": [2] if ub else []}, r["upper_bound_batches"]
    # End to end: an HNSW whose first grid point already reaches 0.90 (W 300, upper bound in every batch) leaves the
    # DLI-best "worse" verdict (R = 1350/300) intact.
    ubr = root / "ub"
    ubr.mkdir()
    uruns = [{"workload": "W1", "seed": 1, "arm": arm, "plan": str(root / "plan.json"),
              "dir": write_run(ubr, 1, arm, plan, scale=sc)} for arm, sc in (("DLI-best", None), ("HNSW", 300))]
    ub_b = A.analyze({"runs": uruns}, resamples=50)["family_B"]["W1/1/test/HNSW"]["work"]
    assert ub_b["decision"] == "worse" and ub_b["upper_bound_batches"]["B"] == [1, 2], ub_b
    # Amendment A3: the widening-restart exception to the monotone-work gate.
    # Batch 1 of a static run: every query widens at nprobe 300 (first pass 20 < K vectors), so
    # work(300) = 20 + work(600) and the batch mean falls from 620 to 600.
    def widen_run(arm, extra, tag):
        r = {"workload": "W1", "seed": 3, "arm": arm, "plan": str(root / "plan.json"),
             "dir": write_run(root / "widen", 3, arm, plan, tag=tag)}
        rows = Path(r["dir"]) / "batch_01" / "rows.jsonl"
        out = []
        for line in rows.read_text().splitlines():
            x = json.loads(line)
            if x["budget"] == 300 and x["policy"] != "hnsw_exact":
                x["scored_total"] = 600 + extra
            out.append(json.dumps(x))
        rows.write_text("\n".join(out) + "\n")
        return r
    (root / "widen").mkdir()
    ok = A.gates(widen_run("S-LMI", 20, "-ok"), ["test"])
    assert ok["monotone_work"] and ok["pass"] and not ok["non_monotone_points"], ok
    assert ok["widening_restart_points"] == [{"panel": "test", "batch": 1, "knobs": [300, 600],
                                              "mean_work": [620.0, 600.0]}], ok["widening_restart_points"]
    big = A.gates(widen_run("S-LMI", 2000, "-big"), ["test"])      # a drop of >= K is not a restart
    assert not big["monotone_work"] and big["non_monotone_points"][0]["knobs"] == [300, 600], big
    other = A.gates(widen_run("HNSW", 20, "-hnsw"), ["test"])     # only static arms widen
    assert not other["monotone_work"], other
print("analyze_final known-answer test: OK")
