"""Q4 analysis (Q4_PREREGISTRATION.md §2–§3). It is committed, with its known-answer test, before arms (a′) and (b).

    analyze_q4.py --pg-dir SHIFT_TOPOLOGY_DIR --qdrant a=DIR --qdrant a1=DIR --qdrant b=DIR [--qdrant c=DIR] [--qdrant c1=DIR]
                  [--a-ledger DIR] [--out FILE]

Each Qdrant DIR is a `dli_phase3_workload` run with `batches.jsonl` and `batch_XX/rows.jsonl` (set `mq_val`,
policy `mdn_code`). The PostgreSQL directory holds `diagnostic/captures.jsonl` and
`diagnostic/<schedule>-publications.jsonl`.
"""
import argparse
import json
from pathlib import Path

import numpy as np

BUDGET = 50_000
SNAPSHOTS = (10, 11, 12)
RESAMPLES = 10_000
SEED = 20261121


def qdrant_rows(run_dir, set_name="mq_val", policy="mdn_code"):
    """{batch: (recall[q], scored[q])} at BUDGET, queries in id order."""
    out = {}
    for t in SNAPSHOTS:
        f = Path(run_dir) / f"batch_{t:02d}" / "rows.jsonl"
        if not f.exists():
            continue
        acc = {}
        for line in open(f):
            r = json.loads(line)
            if r["set"] == set_name and r["policy"] == policy and r["budget"] == BUDGET:
                acc[r["query"]] = (r["recall30"] / 30, r["scored_total"])
        q = sorted(acc)
        out[t] = (np.array([acc[i][0] for i in q]), np.array([acc[i][1] for i in q], dtype=float), q)
    return out


def pg_rows(pg_dir, schedule):
    out = {}
    for line in open(Path(pg_dir) / "diagnostic" / "captures.jsonl"):
        c = json.loads(line)
        if c.get("kind") == "capture" and c["schedule"] == schedule and c["phase"] == "snapshot":
            obs = sorted(c["observations"], key=lambda o: o["query"])
            out[c["batch"]] = (np.array([o["recall"] for o in obs]), np.array([o["scored"] for o in obs], dtype=float),
                               [o["query"] for o in obs], c["live_sha256"], c["topology"])
    return out


def pg_ledger(pg_dir, schedule):
    out = []
    for line in open(Path(pg_dir) / "diagnostic" / f"{schedule}-publications.jsonl"):
        p = json.loads(line)
        if p is not None and p.get("published", True):
            out.append((p["target_level"], p["entries"]))
    return out


def qdrant_ledger(run_dir):
    pubs, snaps = [], {}
    for line in open(Path(run_dir) / "batches.jsonl"):
        b = json.loads(line)
        pubs += [(p["target_level"], p["published_output_points"]) for p in b["publications"]]
        snaps[b["batch"]] = (b["live_sha256"], sorted((x["level"], x["live"]) for x in b["levels"]))
    return pubs, snaps


def paired(x, y, rng):
    d = x - y
    idx = rng.integers(0, len(d), size=(RESAMPLES, len(d)))
    boots = d[idx].mean(axis=1)
    return {"point": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def analyze(pg_dir, qdrant, a_ledger=None):
    rng = np.random.default_rng(SEED)
    pg = {s: pg_rows(pg_dir, s) for s in ("postgres_schedule", "qdrant_schedule")}
    q = {k: qdrant_rows(v) for k, v in qdrant.items()}
    out = {"checks": {}, "snapshots": {}}
    # Known answers: (b) reproduces PostgreSQL's own schedule; (a′) reproduces Qdrant's archived schedule.
    if "b" in qdrant:
        pubs, snaps = qdrant_ledger(qdrant["b"])
        want = pg_ledger(pg_dir, "postgres_schedule")
        topo_ok = all(snaps[t][0] == pg["postgres_schedule"][t][3]
                      and snaps[t][1] == sorted((lv, n) for lv, _, n in pg["postgres_schedule"][t][4])
                      for t in SNAPSHOTS if t in snaps and t in pg["postgres_schedule"])
        out["checks"]["b_ledger_equals_postgres"] = pubs[: len(want)] == want and len(pubs) >= len(want)
        out["checks"]["b_snapshots_equal_postgres"] = topo_ok
    if "a1" in qdrant and a_ledger:
        pa, sa = qdrant_ledger(qdrant["a1"])
        pr, sr = qdrant_ledger(a_ledger)
        n = min(len(pa), len(pr))
        out["checks"]["a1_ledger_equals_archive"] = pa[:n] == pr[:n] and all(sa[t] == sr[t] for t in sa if t in sr)
    for t in SNAPSHOTS:
        if not all(t in q[k] for k in q) or t not in pg["postgres_schedule"]:
            continue
        ref = pg["postgres_schedule"][t]
        for k in q:
            assert q[k][t][2] == ref[2], f"batch {t}: query ids differ ({k} vs PostgreSQL)"
        snap = {"recall_50k": {k: float(q[k][t][0].mean()) for k in q},
                "scored_50k": {k: float(q[k][t][1].mean()) for k in q}}
        snap["recall_50k"]["pg_original"] = float(ref[0].mean())
        snap["scored_50k"]["pg_original"] = float(ref[1].mean())
        if t in pg["qdrant_schedule"]:
            snap["recall_50k"]["pg_aligned"] = float(pg["qdrant_schedule"][t][0].mean())
        c = {}
        if "b" in q and "a1" in q:
            c["S_schedule_b_minus_a1"] = paired(q["b"][t][0], q["a1"][t][0], rng)
        if "a1" in q and "a" in q:
            c["N_noise_a1_minus_a"] = paired(q["a1"][t][0], q["a"][t][0], rng)
        if "b" in q:
            c["Gb_pg_original_minus_b"] = paired(ref[0], q["b"][t][0], rng)
        if "c" in q:
            c["F_router_c_minus_b"] = paired(q["c"][t][0], q["b"][t][0], rng)
            c["E_engine_pg_original_minus_c"] = paired(ref[0], q["c"][t][0], rng)
        if "c1" in q and "a1" in q and t in pg["qdrant_schedule"]:  # Amendment A1: the mirror at Qdrant's schedule
            c["F1_router_c1_minus_a1"] = paired(q["c1"][t][0], q["a1"][t][0], rng)
            c["E1_engine_pg_aligned_minus_c1"] = paired(pg["qdrant_schedule"][t][0], q["c1"][t][0], rng)
        if "a1" in q and t in pg["qdrant_schedule"]:
            c["mirror_pg_aligned_minus_a1"] = paired(pg["qdrant_schedule"][t][0], q["a1"][t][0], rng)
        if "a" in q:
            c["total_pg_original_minus_a"] = paired(ref[0], q["a"][t][0], rng)
        snap["contrasts"] = c
        out["snapshots"][t] = snap
    s11 = out["snapshots"].get(11, {}).get("contrasts", {})
    if "Gb_pg_original_minus_b" in s11 and "S_schedule_b_minus_a1" in s11:
        out["batch11_schedule_explains_gap"] = (abs(s11["Gb_pg_original_minus_b"]["point"]) <= 0.02
                                                and abs(s11["S_schedule_b_minus_a1"]["point"]) >= 0.10)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pg-dir", type=Path, required=True)
    p.add_argument("--qdrant", action="append", required=True, help="ARM=DIR for a, a1, b (and c, c1)")
    p.add_argument("--a-ledger", type=Path, help="archived Qdrant run whose ledger (a′) must reproduce")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    res = analyze(a.pg_dir, dict(x.split("=", 1) for x in a.qdrant), a.a_ledger)
    text = json.dumps(res, indent=1, default=float)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
