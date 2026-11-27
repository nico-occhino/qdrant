"""Q5 analysis (Q5_PREREGISTRATION.md §4). It is committed, with its known-answer test, before any Q5 job.

    analyze_q5.py --plan PLAN_DIR --run E1=DIR --run E1R=DIR --run E1T=DIR --set mq [--out FILE]

Each DIR is one `dli_phase3_workload` run: `batches.jsonl`, `build-events.jsonl` and `batch_XX/rows.jsonl`
(policy `mdn_code`), plus `batch_XX/topology.json`.

Primary: the equal-batch mean of paired Recall@30 at nominal 50K over all measured batches, for E1T−E1,
E1R−E1 and E1T−E1R. The bootstrap is a paired query bootstrap stratified by batch (10,000 resamples, seed
20261127). The sensitivity interval resamples one shared vector of query positions for every batch.
Recovery = (E1T−E1)/(E1R−E1), undefined if E1R−E1 <= 0.

Criterion 1: CI95 lower bound of E1T−E1 > 0 and recovery >= 0.5.
Criterion 2: trained-build time of E1T <= 0.5 x E1R. Trained-build time is sample_read + teacher_mlp seconds,
summed over build_completed events with reused_router null.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGET = 50_000
TARGET = 0.90
RESAMPLES = 10_000
SEED = 20261127
ARMS = ("E1", "E1R", "E1T")


def load_rows(run_dir, set_name, policy="mdn_code"):
    """{batch: (budgets, recall[q, b] in [0, 1], work[q, b], query ids)} for one set."""
    out = {}
    for bdir in sorted(Path(run_dir).glob("batch_*")):
        rows = bdir / "rows.jsonl"
        if not rows.exists():
            continue
        acc = defaultdict(dict)
        for line in open(rows):
            r = json.loads(line)
            if r["set"] == set_name and r["policy"] == policy:
                acc[r["query"]][r["budget"]] = (r["recall30"] / 30, r["scored_total"])
        if not acc:
            continue
        queries = sorted(acc)
        budgets = sorted(acc[queries[0]])
        assert all(sorted(acc[q]) == budgets for q in queries), f"{rows}: ragged budget grid"
        rec = np.array([[acc[q][b][0] for b in budgets] for q in queries])
        wrk = np.array([[acc[q][b][1] for b in budgets] for q in queries], dtype=float)
        out[int(bdir.name.split("_")[1])] = (budgets, rec, wrk, queries)
    return out


def w_at(recall_curve, work_curve, target=TARGET):
    """Work at mean recall = target by the first measured crossing. Returns (W, kind)."""
    r, w = np.asarray(recall_curve), np.asarray(work_curve)
    if r[0] >= target:
        return float(w[0]), "upper_bound"
    for i in range(1, len(r)):
        if r[i] >= target:
            return float(w[i - 1] + (target - r[i - 1]) / (r[i] - r[i - 1]) * (w[i] - w[i - 1])), "interpolated"
    return None, "not_attained"


def contrast(a, b, rng, shared=None):
    """Equal-batch mean paired difference of recall at BUDGET with bootstrap CIs."""
    batches = sorted(set(a) & set(b))
    pts, boots = [], np.zeros(RESAMPLES)
    nq = {t: a[t][1].shape[0] for t in batches}
    for t in batches:
        assert a[t][3] == b[t][3], f"batch {t}: query ids differ between arms"
        j = a[t][0].index(BUDGET)
        d = a[t][1][:, j] - b[t][1][:, j]
        pts.append(d.mean())
        idx = shared[: , :nq[t]] if shared is not None else rng.integers(0, nq[t], size=(RESAMPLES, nq[t]))
        boots += d[idx].mean(axis=1)
    boots /= len(batches)
    point = float(np.mean(pts))
    return {"point": point, "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "batches": batches, "per_batch": {t: float(p) for t, p in zip(batches, pts)}}


def trainer_seconds(run_dir):
    """Trained-build seconds (in build order) of one run."""
    out = []
    path = Path(run_dir) / "build-events.jsonl"
    for line in open(path):
        e = json.loads(line)
        if e.get("stage") == "build_completed":
            out.append({"trained": e["reused_router"] is None, "points": e["physical_count"],
                        "buckets": e["n_buckets"], "router": e["router_fingerprint"],
                        "reused": e["reused_router"],
                        "seconds": e["sample_read_seconds"] + e["teacher_mlp_seconds"]})
    return out


def imbalance(run_dir):
    """{batch: {level: max/mean over all allocated buckets}} from topology.json."""
    out = {}
    for t in sorted(Path(run_dir).glob("batch_*/topology.json")):
        topo = json.loads(t.read_text())
        out[int(t.parent.name.split("_")[1])] = {
            m["level"]: max(m["bucket_sizes"]) * len(m["bucket_sizes"]) / sum(m["bucket_sizes"])
            for m in topo["members"] if m["level"] >= 1 and m["bucket_sizes"] and sum(m["bucket_sizes"])}
    return out


def decisions(run_dir):
    out = []
    for line in open(Path(run_dir) / "batches.jsonl"):
        rec = json.loads(line)
        for d in rec.get("trigger_decisions", []):
            out.append({**d, "batch": rec["batch"]})
    return out


def check_decisions(decs, builds, first_fits):
    """Gate 3: decisions agree with builds.
    * Each reuse decision matches a distinct build that reused exactly the decision's router at the same bucket
      count, and every reuse build is so matched.
    * Trained builds = first fits (E1's trained builds; the schedule is identical across arms) + retrain decisions.
    """
    reuse_builds = [b for b in builds if not b["trained"]]
    used, problems = set(), []
    for d in decs:
        if d["retrain"]:
            continue
        hit = next((i for i, b in enumerate(reuse_builds)
                    if i not in used and b["reused"] == d["target_router"] and b["buckets"] == d["buckets"]), None)
        if hit is None:
            problems.append(f"reuse decision job {d['job']} has no build reusing router {d['target_router']}")
        else:
            used.add(hit)
    if len(used) != len(reuse_builds):
        problems.append(f"{len(reuse_builds) - len(used)} reuse builds have no reuse decision")
    retrains = sum(d["retrain"] for d in decs)
    trained = sum(b["trained"] for b in builds)
    if trained != first_fits + retrains:
        problems.append(f"trained builds {trained} != first fits {first_fits} + retrain decisions {retrains}")
    return {"decisions": len(decs), "retrain_decisions": retrains, "reuse_builds": len(reuse_builds),
            "trained_builds": trained, "first_fits": first_fits, "problems": problems, "pass": not problems}


def analyze(runs, set_name, resamples_seed=SEED):
    data = {arm: load_rows(d, set_name) for arm, d in runs.items()}
    rng = np.random.default_rng(resamples_seed)
    out = {"set": set_name, "budget": BUDGET}
    nq = min(x[1].shape[0] for x in data["E1"].values())
    shared = np.random.default_rng(resamples_seed + 1).integers(0, nq, size=(RESAMPLES, nq))
    for a, b in (("E1T", "E1"), ("E1R", "E1"), ("E1T", "E1R")):
        out[f"{a}-{b}"] = contrast(data[a], data[b], rng)
        out[f"{a}-{b}"]["sensitivity_ci95"] = contrast(data[a], data[b], rng, shared)["ci95"]
    gain_t, gain_r = out["E1T-E1"]["point"], out["E1R-E1"]["point"]
    out["recovery"] = None if gain_r <= 0 else gain_t / gain_r
    builds = {arm: trainer_seconds(d) for arm, d in runs.items()}
    sched = {arm: [b["points"] for b in builds[arm]] for arm in runs}
    assert sched["E1"] == sched["E1R"] == sched["E1T"], "build schedules differ across arms"
    first = [i for i, b in enumerate(builds["E1"]) if b["trained"]]   # E1 trains only empty destinations
    out["training"] = {arm: {"trained_builds": sum(b["trained"] for b in builds[arm]),
                             "trained_seconds": sum(b["seconds"] for b in builds[arm] if b["trained"]),
                             "first_fit_seconds": sum(builds[arm][i]["seconds"] for i in first)}
                       for arm in runs}
    tr = out["training"]
    out["trainer_ratio_E1T_over_E1R"] = tr["E1T"]["trained_seconds"] / tr["E1R"]["trained_seconds"]
    decs = decisions(runs["E1T"])
    out["decisions"] = {"count": len(decs), "fired": [d for d in decs if d["retrain"]],
                        "firing_batches": sorted({d["batch"] for d in decs if d["retrain"]}),
                        "census_micros_total": sum(d["census_micros"] for d in decs),
                        "max_ratio_eligible": max((d["max_over_mean"] for d in decs if d["target_level"] >= 2),
                                                  default=None),
                        "check": check_decisions(decs, builds["E1T"], len(first))}
    lo = out["E1T-E1"]["ci95"][0]
    c1 = out["recovery"] is not None and lo > 0 and out["recovery"] >= 0.5
    c2 = out["trainer_ratio_E1T_over_E1R"] <= 0.5
    out["criterion_1"], out["criterion_2"] = bool(c1), bool(c2)
    out["verdict"] = ("met" if c1 and c2 else
                      "not met (trigger never fired)" if not out["decisions"]["fired"] else "not met")
    per = {}
    for arm in runs:
        per[arm] = {}
        for t, (budgets, rec, wrk, _) in data[arm].items():
            w, kind = w_at(rec.mean(axis=0), wrk.mean(axis=0))
            j = budgets.index(BUDGET)
            per[arm][t] = {"recall_50k": float(rec[:, j].mean()), "work_50k": float(wrk[:, j].mean()),
                           "w090": w, "w090_kind": kind}
    out["per_batch"] = per
    out["imbalance"] = {arm: imbalance(d) for arm, d in runs.items()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, help="ARM=DIR for E1, E1R and E1T")
    p.add_argument("--set", required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    runs = dict(x.split("=", 1) for x in a.run)
    assert set(runs) == set(ARMS), f"need exactly {ARMS}"
    res = analyze(runs, a.set)
    text = json.dumps(res, indent=1, default=float)
    if a.out:
        a.out.write_text(text + "\n")
    print(json.dumps({k: res[k] for k in ("E1T-E1", "E1R-E1", "E1T-E1R", "recovery", "trainer_ratio_E1T_over_E1R",
                                          "criterion_1", "criterion_2", "verdict")}, indent=1, default=float))


if __name__ == "__main__":
    main()
