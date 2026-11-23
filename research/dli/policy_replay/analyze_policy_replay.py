"""Analysis for research/dli/policy_replay (PREREGISTRATION.md §2, §5, §6).

One run directory of `dli_phase3_workload` holds both policies on the same frozen snapshots.
Per batch t, W(t) is the work (scored vectors) at mean R@30 = 0.90, by linear interpolation
over budgets (first-point upper bound as in the static study's amendment A1).

Primary: R = mean_t W_prop(t) / mean_t W_mdn(t), paired query bootstrap stratified by batch.
Secondary: S1 Δ mean R@30 at B* (50K); S2 per-batch R(t); S3 per-level selection and misses
at B*; S4 buffer share of proportional work; S5 reproduction of mdn_code against a reference
run (Phase 3 E1) at shared budgets.

    analyze_policy_replay.py --run DIR --set test [--reference DIR] [--out FILE]
"""
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p3 = _load("p3", HERE.parent / "emma_path" / "analyze_phase3.py")
sv = _load("sv", HERE.parent / "static_qdrant_vs_dli" / "analyze_static_vs_dli.py")

SEED = 20261124
TARGET = 0.90
B_STAR = 50_000
MARGIN = 0.95  # smallest effect of interest: 5% of work
MIN_ATTAINED_SHARE = 0.8
A, B = "proportional", "mdn_code"


def decide(point, c95, margin=MARGIN):
    lo, hi = c95
    up = 1 / margin
    if margin <= lo and hi <= up:
        return "equivalent"
    if point <= margin and hi < 1:
        return "proportional better"
    if point >= up and lo > 1:
        return "mdn_code better"
    if lo > 1 or hi < 1:
        return "difference below the smallest effect of interest"
    return "inconclusive"


def gates(run_a, run_b):
    problems = []
    for t in sorted(set(run_a) | set(run_b)):
        if t not in run_a or t not in run_b:
            problems.append(f"batch {t}: a policy is missing")
            continue
        if set(run_a[t]) != set(run_b[t]):
            problems.append(f"batch {t}: budget grids differ")
        for name, run in ((A, run_a), (B, run_b)):
            budgets = sorted(run[t])
            sizes = {len(run[t][x][0]) for x in budgets}
            if len(sizes) != 1:
                problems.append(f"batch {t} {name}: query counts differ across budgets")
            means = [run[t][x][1].mean() for x in budgets]
            if any(b2 < b1 - 1e-9 for b1, b2 in zip(means, means[1:])):
                problems.append(f"batch {t} {name}: mean work not monotone in budget")
        if len(run_a[t][min(run_a[t])][0]) != len(run_b[t][min(run_b[t])][0]):
            problems.append(f"batch {t}: policies answered different query counts")
    return problems


def work_ratio(run_a, run_b, resamples, rng):
    batches = sorted(set(run_a) & set(run_b))
    per_batch, pa_pt, pb_pt, ba_bs, bb_bs = {}, [], [], [], []
    for t in batches:
        ra, wa = sv.curve(run_a, t)
        rb, wb = sv.curve(run_b, t)
        (pa,), (fa,) = sv.interp_or_first(ra.mean(axis=1), wa.mean(axis=1))
        (pb,), (fb,) = sv.interp_or_first(rb.mean(axis=1), wb.mean(axis=1))
        per_batch[t] = {"W_prop": None if np.isnan(pa) else float(pa), "W_mdn": None if np.isnan(pb) else float(pb),
                        "ratio": None if np.isnan(pa) or np.isnan(pb) else float(pa / pb),
                        "prop_upper_bound": bool(fa), "mdn_upper_bound": bool(fb)}
        if np.isnan(pa) or np.isnan(pb):
            continue
        idx = rng.integers(0, ra.shape[1], size=(resamples, ra.shape[1]))
        ba_bs.append(sv.interp_or_first(ra[:, idx].mean(axis=2).T, wa[:, idx].mean(axis=2).T)[0])
        bb_bs.append(sv.interp_or_first(rb[:, idx].mean(axis=2).T, wb[:, idx].mean(axis=2).T)[0])
        pa_pt.append(pa)
        pb_pt.append(pb)
    out = {"batches": batches, "attained_both": len(pa_pt),
           "attained_share": len(pa_pt) / len(batches) if batches else 0.0, "per_batch": per_batch}
    if not pa_pt:
        out["status"] = "target not attained"
        return out
    boots = np.nanmean(np.stack(ba_bs), axis=0) / np.nanmean(np.stack(bb_bs), axis=0)
    point = float(np.mean(pa_pt) / np.mean(pb_pt))
    c95, c99 = p3.ci(boots, 0.95), p3.ci(boots, 0.99)
    out.update({"R_prop_over_mdn": point, "ci95": c95, "ci99_bonferroni": c99,
                "resamples_not_reached": int(np.isnan(boots).sum()),
                "decision": decide(point, c95), "decision_bonferroni": decide(point, c99),
                "status": "ok" if out["attained_share"] >= MIN_ATTAINED_SHARE else "descriptive (attainment below 80%)"})
    return out


def recall_diff_at(run_a, run_b, budget, resamples, rng):
    batches = [t for t in sorted(set(run_a) & set(run_b)) if budget in run_a[t] and budget in run_b[t]]
    if not batches:
        return None
    point, boots = [], np.zeros(resamples)
    for t in batches:
        d = run_a[t][budget][0] - run_b[t][budget][0]
        point.append(d.mean())
        boots += d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    boots /= len(batches)
    return {"budget": budget, "delta_pp": 100 * float(np.mean(point)), "ci95_pp": [100 * x for x in p3.ci(boots, 0.95)],
            "per_batch_pp": {t: 100 * float(p) for t, p in zip(batches, point)}}


def level_diagnostics(run_dir, set_name, budget):
    """S3/S4 at `budget`: mean selected buckets per level, missed gold by level, buffer share."""
    acc = {p: {"buckets": defaultdict(float), "miss": defaultdict(float), "plain": 0.0, "scored": 0.0, "n": 0}
           for p in (A, B)}
    for rows in sorted(Path(run_dir).glob("batch_*/rows.jsonl")):
        for line in open(rows):
            r = json.loads(line)
            if r["set"] != set_name or r["budget"] != budget or r["policy"] not in acc:
                continue
            a = acc[r["policy"]]
            a["n"] += 1
            a["plain"] += r.get("plain_scored", 0)
            a["scored"] += r["scored_total"]
            for lvl, k in r.get("buckets_per_level", {}).items():
                a["buckets"][int(lvl)] += k
            for lvl, _rank in r.get("unselected_level_rank", []):
                a["miss"][int(lvl)] += 1
            a["miss"]["buffer"] += r.get("miss_plain_unselected", 0)
            a["miss"]["scored_but_lost"] += r.get("miss_selected_but_lost", 0)
    out = {}
    for p, a in acc.items():
        n = max(a["n"], 1)
        out[p] = {"rows": a["n"],
                  "mean_buckets_per_level": {str(k): v / n for k, v in sorted(a["buckets"].items())},
                  "mean_missed_gold_per_query": {str(k): v / n for k, v in a["miss"].items()},
                  "buffer_share_of_work": a["plain"] / a["scored"] if a["scored"] else None}
    return out


def reproduction(run_b, ref_b):
    """S5: mdn_code here vs a reference run's mdn_code at shared (batch, budget)."""
    same, total, diffs = 0, 0, []
    for t in sorted(set(run_b) & set(ref_b)):
        for budget in sorted(set(run_b[t]) & set(ref_b[t])):
            x, y = run_b[t][budget][0], ref_b[t][budget][0]
            if len(x) != len(y):
                continue
            same += int((x == y).sum())
            total += len(x)
            diffs.append(float((x - y).mean()))
    if not total:
        return None
    return {"identical_query_share": same / total, "mean_recall_diff_pp": 100 * float(np.mean(diffs)),
            "max_abs_cell_diff_pp": 100 * float(np.max(np.abs(diffs))), "cells": len(diffs)}


def analyze(run_dir, set_name, reference=None, resamples=p3.RESAMPLES, seed=SEED):
    run_a = p3.load_run(run_dir, A, set_name)
    run_b = p3.load_run(run_dir, B, set_name)
    rng = np.random.default_rng(seed)
    out = {"set": set_name, "gate_problems": gates(run_a, run_b)}
    out["primary"] = work_ratio(run_a, run_b, resamples, rng)
    out["S1_delta_recall_at_Bstar"] = recall_diff_at(run_a, run_b, B_STAR, resamples, rng)
    out["S3_S4_levels_at_Bstar"] = level_diagnostics(run_dir, set_name, B_STAR)
    if reference:
        out["S5_reproduction"] = reproduction(run_b, p3.load_run(reference, B, set_name))
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--set", required=True)
    a.add_argument("--reference", type=Path)
    a.add_argument("--resamples", type=int, default=p3.RESAMPLES)
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    result = analyze(args.run, args.set, args.reference, args.resamples)
    text = json.dumps(result, indent=1, default=float)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
