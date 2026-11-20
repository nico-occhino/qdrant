"""Analysis for the Emma-path DLI study (see PREREGISTRATION.md).

Each arm directory holds the job's copies of: summary.json (ingestion harness),
build-events.jsonl (QDRANT_LMI_BUILD_EVENTS of the ingestion only) and frontier/
{rows.jsonl, topology.json} (native-replay selection frontier).

Estimands follow research/dli/selection_frontier/ten_million/analyze_10m.py:
* mean curves per (arm, policy, budget) over the evaluation queries;
* W(target): realized mean work where the mean Recall@30 curve reaches the target,
  linear in work between the two measured budgets that bracket it; never
  extrapolated (None if the sweep starts above or never reaches the target);
* saving(A vs B) = 1 - W_A / W_B, paired percentile bootstrap over query identities
  (one resampled query multiset shared by every arm and policy).

`pilot` adds the engineering gate (preregistration section 4) and the 10M projection.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def arm_name(path):
    return json.loads((path / "summary.json").read_text())["arm"]


def curves(rows, eval_set):
    """{policy: (budgets, R[q,b], W[q,b], P[q,b])} on one query set."""
    out = {}
    rows = [r for r in rows if r["set"] == eval_set]
    queries = sorted({r["query"] for r in rows})
    qi = {q: i for i, q in enumerate(queries)}
    for policy in sorted({r["policy"] for r in rows}):
        sel = [r for r in rows if r["policy"] == policy]
        budgets = sorted({r["budget"] for r in sel})
        bi = {b: i for i, b in enumerate(budgets)}
        shape = (len(queries), len(budgets))
        R, W, P = (np.full(shape, np.nan) for _ in range(3))
        for r in sel:
            i, j = qi[r["query"]], bi[r["budget"]]
            assert math.isnan(R[i, j]), f"duplicate row {policy} {r['query']} {r['budget']}"
            R[i, j] = r["recall30"] / 30
            W[i, j] = r["scored_total"]
            P[i, j] = r["plain_scored"]
        assert not np.isnan(R).any(), f"missing rows for {policy}"
        out[policy] = (budgets, R, W, P)
    return out, queries


def work_at(mean_r, mean_w, mean_p, target):
    """(work, plain share, status) where the mean curve first reaches target."""
    if mean_r[0] >= target:
        return None, None, "below_range"
    for j in range(1, len(mean_r)):
        if mean_r[j] >= target:
            t = (target - mean_r[j - 1]) / (mean_r[j] - mean_r[j - 1])
            w = mean_w[j - 1] + t * (mean_w[j] - mean_w[j - 1])
            p = mean_p[j - 1] + t * (mean_p[j] - mean_p[j - 1])
            return float(w), float(p / w), "ok"
    return None, None, "not_reached"


def frontier_summary(curve, target):
    budgets, R, W, P = curve
    w, share, status = work_at(R.mean(0), W.mean(0), P.mean(0), target)
    return {
        "budgets": budgets,
        "mean_recall30": R.mean(0).round(5).tolist(),
        "mean_work": W.mean(0).round(1).tolist(),
        "mean_plain": P.mean(0).round(1).tolist(),
        "work_at_target": w,
        "plain_share_at_target": share,
        "status": status,
    }


def bootstrap_saving(curve_a, curve_b, target, resamples, seed):
    """Saving 1 - W_a/W_b with a paired query bootstrap (same query order in both)."""
    rng = np.random.default_rng(seed)
    q = curve_a[1].shape[0]
    assert curve_b[1].shape[0] == q
    idx = rng.integers(0, q, size=(resamples, q))
    values = []
    for row in idx:
        wa = work_at(curve_a[1][row].mean(0), curve_a[2][row].mean(0), curve_a[3][row].mean(0), target)[0]
        wb = work_at(curve_b[1][row].mean(0), curve_b[2][row].mean(0), curve_b[3][row].mean(0), target)[0]
        if wa is not None and wb is not None:
            values.append(1 - wa / wb)
    point_a = work_at(curve_a[1].mean(0), curve_a[2].mean(0), curve_a[3].mean(0), target)[0]
    point_b = work_at(curve_b[1].mean(0), curve_b[2].mean(0), curve_b[3].mean(0), target)[0]
    point = None if point_a is None or point_b is None else 1 - point_a / point_b
    lo, hi = (np.percentile(values, [2.5, 97.5]).tolist() if values else (None, None))
    return {"point": point, "ci95": [lo, hi], "valid_resamples": len(values), "resamples": resamples}


def maintenance(arm_dir):
    summary = json.loads((arm_dir / "summary.json").read_text())
    builds = load_jsonl(arm_dir / "build-events.jsonl")
    builds = [b for b in builds if b.get("stage") == "build_completed"]
    pubs = summary["publication_events"]
    reused = [b for b in builds if b.get("reused_router")]
    return {
        "population": summary["population"],
        "publications": len(pubs),
        "builds": len(builds),
        "reused_builds": len(reused),
        "trained_builds": len(builds) - len(reused),
        "training_seconds": sum(b["teacher_mlp_seconds"] for b in builds),
        "routing_postings_seconds": sum(b["routing_postings_seconds"] for b in builds),
        "optimizer_wall_seconds": sum(p["optimizer_micros"] for p in pubs) / 1e6,
        "written_points": sum(p["published_output_points"] for p in pubs),
        "write_amplification": sum(p["published_output_points"] for p in pubs) / summary["population"],
        "ingestion_seconds": summary["ingestion_to_checkpoint_seconds"],
        "collection_pass": {k: summary[k] for k in ("recall10", "recall30", "query_p50_ms", "query_p95_ms")},
        "dli_attribution": summary["dli_attribution"],
    }


def pilot_gate(arm_dir, arity):
    """Preregistration section 4, criteria 1-3, for one arm."""
    summary = json.loads((arm_dir / "summary.json").read_text())
    builds = [b for b in load_jsonl(arm_dir / "build-events.jsonl") if b.get("stage") == "build_completed"]
    pubs = summary["publication_events"]
    checks = {}
    checks["one_build_per_publication"] = len(builds) == len(pubs)
    checks["outputs_match_plans"] = all(p["expected_output_points"] == p["published_output_points"] for p in pubs)
    learned, fallbacks = summary["dli_attribution"][0], summary["dli_attribution"][1]
    checks["collection_pass_all_learned"] = learned == summary["query_count"] and fallbacks == 0
    reuse_ok, fresh_ok = True, True
    if checks["one_build_per_publication"]:
        for build, pub in zip(builds, pubs):
            if build.get("reused_router"):
                reuse_ok &= build["reused_router"] == build["router_fingerprint"] and build["teacher_mlp_seconds"] == 0
            else:
                fresh_ok &= build["n_buckets"] == arity ** pub["target_level"]
    checks["reuse_builds_keep_router_without_training"] = reuse_ok
    checks["fresh_builds_have_arity_power_buckets"] = fresh_ok
    return checks


def projection_10m(arm_dir, arity, buffer, population=10_120_191):
    """Projected E1 10M ingestion wall: per-publication model fitted on the pilot."""
    summary = json.loads((arm_dir / "summary.json").read_text())
    builds = [b for b in load_jsonl(arm_dir / "build-events.jsonl") if b.get("stage") == "build_completed"]
    pubs = summary["publication_events"]
    # Non-training wall per publication: c0 + c1 * output points (least squares).
    x = np.array([p["published_output_points"] for p in pubs], dtype=float)
    y = np.array([p["optimizer_micros"] / 1e6 - b["teacher_mlp_seconds"] for p, b in zip(pubs, builds)])
    c1, c0 = np.polyfit(x, y, 1)
    # Training seconds per unit of k-means + MLP work, fitted on trained pilot builds.
    trained = [b for b in builds if not b.get("reused_router")]
    def flops(samples, buckets, hidden, epochs, iters=25):
        return samples * buckets * 768 * iters + samples * epochs * hidden * (768 + buckets) * 3
    rate = sum(b["teacher_mlp_seconds"] for b in trained) / sum(
        flops(b["sample_count"], b["n_buckets"], b["hidden_dim"], b["epochs"]) for b in trained)
    # Replay the geometric plan with router reuse (as dli_schedule::plan).
    levels, wall, n, trains = [], 0.0, 0, 0
    while n + buffer <= population:
        n += buffer
        incoming, target = buffer, 1
        while True:
            cap = buffer * arity ** target
            occ = levels[target - 1] if target - 1 < len(levels) else 0
            if incoming <= cap - occ:
                break
            incoming += occ
            target += 1
        levels += [0] * (target - len(levels))
        out = incoming + levels[target - 1]
        wall += c0 + c1 * out
        if levels[target - 1] == 0:
            buckets = arity ** target
            samples = min(out, max(100_000, 40 * buckets))
            wall += rate * flops(samples, buckets, 128, 10)
            trains += 1
        for s in range(1, target):
            levels[s - 1] = 0
        levels[target - 1] = out
    return {"c0_s": c0, "c1_s_per_point": c1, "training_s_per_flop": rate,
            "projected_publications": n // buffer, "projected_trainings": trains,
            "projected_wall_hours": wall / 3600, "final_levels": levels}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["pilot", "confirm"])
    p.add_argument("arms", type=Path, nargs="+")
    p.add_argument("--target", type=float, default=0.90)
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = {"mode": a.mode, "target": a.target, "seed": a.seed, "arms": {}}
    arm_curves = {}
    for arm_dir in a.arms:
        name = arm_name(arm_dir)
        summary = json.loads((arm_dir / "summary.json").read_text())
        rows = load_jsonl(arm_dir / "frontier" / "rows.jsonl")
        entry = {"dir": str(arm_dir), "maintenance": maintenance(arm_dir),
                 "topology": json.loads((arm_dir / "frontier" / "topology.json").read_text())}
        entry["topology"] = [{k: m[k] for k in ("level", "live", "buckets", "bucket_size_max")}
                             for m in entry["topology"]["members"]]
        for eval_set in ("val", "test"):
            c, queries = curves(rows, eval_set)
            entry[eval_set] = {pol: frontier_summary(curve, a.target) for pol, curve in c.items()}
            entry[eval_set + "_queries"] = len(queries)
            if eval_set == "test":
                arm_curves[name] = c
        if a.mode == "pilot":
            entry["gate"] = pilot_gate(arm_dir, summary["arity"])
            if name == "E1":
                entry["projection_10m"] = projection_10m(arm_dir, summary["arity"], summary["buffer_capacity"])
        report["arms"][name] = entry
    comparisons = {}
    if "G0" in arm_curves:
        for name, c in arm_curves.items():
            if name == "G0":
                continue
            for policy in c:
                comparisons[f"{name}:{policy} vs G0:slot"] = bootstrap_saving(
                    c[policy], arm_curves["G0"]["slot"], a.target, a.resamples, a.seed)
    if "E1" in arm_curves and "static_flat" in arm_curves["E1"]:
        comparisons["E1:mdn_code vs E1:static_flat"] = bootstrap_saving(
            arm_curves["E1"]["mdn_code"], arm_curves["E1"]["static_flat"], a.target, a.resamples, a.seed)
    report["comparisons"] = comparisons
    if a.mode == "confirm" and "E1" in report["arms"] and "G0" in report["arms"]:
        h1 = comparisons["E1:mdn_code vs G0:slot"]
        e1 = report["arms"]["E1"]["test"]["mdn_code"]
        m = {k: report["arms"][k]["maintenance"]["training_seconds"] for k in ("E1", "G0")}
        report["hypotheses"] = {
            "H1_saving_ge_30pct": h1["point"] is not None and h1["point"] >= 0.30
            and h1["ci95"][0] is not None and h1["ci95"][0] > 0,
            "H2_plain_share_le_10pct": e1["plain_share_at_target"] is not None and e1["plain_share_at_target"] <= 0.10,
            "H3_training_seconds_lower": m["E1"] < m["G0"],
        }
    a.out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=1, default=float))
    for name, entry in report["arms"].items():
        print(name, json.dumps(entry.get("gate")), json.dumps(entry.get("projection_10m"), default=float))
        for pol, s in entry["test"].items():
            print(f"  {pol:12s} W@{a.target}={s['work_at_target']} plain_share={s['plain_share_at_target']} {s['status']}")


if __name__ == "__main__":
    main()
