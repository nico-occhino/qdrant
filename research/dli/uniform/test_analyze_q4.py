"""Known-answer test for analyze_q4.py.

Synthetic snapshots 10–12, 4 queries, recall at 50K identical across queries:
* PostgreSQL original 0.70, aligned 0.90; Qdrant (a′) 0.90, (b) 0.70, and (a) 0.91, which rounds to 27/30 = 0.90.
* So S = (b)−(a′) = −0.20, N = 0, G_b = 0, and the mirror check = 0.
* At batch 11 the schedule explains the gap (|G_b| ≤ 0.02, |S| ≥ 0.10).
* (b)'s ledger equals PostgreSQL's (two publications), and its snapshot digests and populations match.
* A second (b) with a wrong publication fails the ledger check.
* Amendment A1: (c) = 0.80 and (c′) = 0.80, so F = (c)−(b) = +0.10 (0.80 rounds to 24/30), E = PostgreSQL
  original − (c) = −0.10, F′ = (c′)−(a′) = −0.10 and E′ = PostgreSQL aligned − (c′) = +0.10.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_q4 as Q  # noqa: E402

PUBS = [(1, 5000), (1, 10000)]
TOPO = [[1, 10, 10000]]


def pg(root):
    d = root / "pg" / "diagnostic"
    d.mkdir(parents=True)
    with open(d / "captures.jsonl", "w") as f:
        for sched, r in (("postgres_schedule", 0.70), ("qdrant_schedule", 0.90)):
            for t in Q.SNAPSHOTS:
                f.write(json.dumps({"kind": "capture", "schedule": sched, "batch": t, "phase": "snapshot",
                                    "live_sha256": f"h{t}", "topology": TOPO,
                                    "observations": [{"query": q, "recall": r, "scored": 50000} for q in range(4)]}) + "\n")
    for sched in ("postgres_schedule", "qdrant_schedule"):
        with open(d / f"{sched}-publications.jsonl", "w") as f:
            for lv, n in PUBS:
                f.write(json.dumps({"target_level": lv, "entries": n, "published": True}) + "\n")
    return root / "pg"


def qrun(root, name, recall, pubs=PUBS):
    d = root / name
    for t in Q.SNAPSHOTS:
        (d / f"batch_{t:02d}").mkdir(parents=True)
        with open(d / f"batch_{t:02d}" / "rows.jsonl", "w") as f:
            for q in range(4):
                f.write(json.dumps({"set": "mq_val", "policy": "mdn_code", "budget": 50000, "query": q,
                                    "recall30": round(recall * 30), "scored_total": 50000}) + "\n")
    with open(d / "batches.jsonl", "w") as f:
        for t in Q.SNAPSHOTS:
            f.write(json.dumps({"batch": t, "live_sha256": f"h{t}", "levels": [{"level": 1, "live": 10000}],
                                "publications": [{"target_level": lv, "published_output_points": n}
                                                 for lv, n in (pubs if t == 10 else [])]}) + "\n")
    return str(d)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    pgd = pg(root)
    runs = {"a": qrun(root, "a", 0.91), "a1": qrun(root, "a1", 0.90), "b": qrun(root, "b", 0.70)}
    r = Q.analyze(pgd, runs, a_ledger=runs["a"])
    c = r["snapshots"][11]["contrasts"]
    assert abs(c["S_schedule_b_minus_a1"]["point"] + 0.2) < 1e-9, c
    assert c["N_noise_a1_minus_a"]["point"] == 0, c
    assert abs(c["Gb_pg_original_minus_b"]["point"]) < 0.01, c
    assert abs(c["mirror_pg_aligned_minus_a1"]["point"]) < 1e-9, c
    assert r["batch11_schedule_explains_gap"] is True
    assert r["checks"]["b_ledger_equals_postgres"] and r["checks"]["b_snapshots_equal_postgres"], r["checks"]
    assert r["checks"]["a1_ledger_equals_archive"], r["checks"]
    runs.update(c=qrun(root, "c", 0.80), c1=qrun(root, "c1", 0.80))
    c = Q.analyze(pgd, runs)["snapshots"][10]["contrasts"]
    for key, want in (("F_router_c_minus_b", 0.10), ("E_engine_pg_original_minus_c", -0.10),
                      ("F1_router_c1_minus_a1", -0.10), ("E1_engine_pg_aligned_minus_c1", 0.10)):
        assert abs(c[key]["point"] - want) < 1e-9, (key, c[key])
    bad = qrun(root, "b_bad", 0.70, pubs=[(1, 5000), (2, 10000)])
    r = Q.analyze(pgd, {"b": bad, "a1": runs["a1"]})
    assert not r["checks"]["b_ledger_equals_postgres"], r["checks"]
print("analyze_q4 known-answer test: OK")
