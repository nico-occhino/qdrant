"""Q7 analysis (Q7_PREREGISTRATION.md §4–§5).

    analyze_q7.py --run OUT_DIR [--out FILE]

Input: `rows.jsonl` and `gold.jsonl` from `q7_filter_fixture`.
Per (arm, strategy, predicate): mean Recall@30, mean underfill, mean vectors scored and the plan mix.
Per (arm, predicate) curve (LMI and DLI nprobe=…, HNSW ef=…, ordered by knob):
* W@0.90 by the first measured crossing (interpolated, `upper_bound` or `not_attained`);
* R@30 at the first point with mean work >= 50,000, or `unreachable` when every point scores less.
Gates:
* no duplicate or non-qualifying ids;
* exact strategies have Recall@30 = 1 except where the gold has an exact tie at rank 30 (s30 == s31);
* monotone mean work along each curve, except widening restarts (a drop between n and 2n where every per-query
  drop is < 30, as in final-benchmark A3; for a multi-member arm, every per-member drop, Amendment A2); every drop
  is listed.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGET, NOMINAL, K = 0.90, 50_000, 30


def knob(strategy):
    m = re.fullmatch(r"(nprobe|ef)=(\d+)", strategy)
    return (m.group(1), int(m.group(2))) if m else None


def w_at(points):
    """points: [(knob, mean recall, mean work)] in knob order."""
    if not points:
        return None, "not_attained"
    if points[0][1] >= TARGET:
        return points[0][2], "upper_bound"
    for (k0, r0, w0), (k1, r1, w1) in zip(points, points[1:]):
        if r1 >= TARGET:
            return w0 + (TARGET - r0) / (r1 - r0) * (w1 - w0), "interpolated"
    return None, "not_attained"


def analyze(run):
    rows = [json.loads(l) for l in open(Path(run) / "rows.jsonl")]
    gold = {(g["predicate"], g["query"]): g for g in map(json.loads, open(Path(run) / "gold.jsonl"))}
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["strategy"], r["predicate"])].append(r)
    summary = {}
    for (arm, strat, pred), v in sorted(cells.items()):
        plans = defaultdict(int)
        for x in v:
            plans["+".join(str(p.get("plan")) for p in x["plans"])] += 1
        summary[f"{arm}/{strat}/{pred}"] = {
            "queries": len(v), "recall30": float(np.mean([x["recall30"] for x in v])),
            "underfill": float(np.mean([x["underfill"] for x in v])), "work": float(np.mean([x["work"] for x in v])),
            "qualifying": gold[(pred, 0)]["qualifying"], "plans": dict(plans)}
    curves, problems, restarts = {}, [], []
    by_curve = defaultdict(list)
    for (arm, strat, pred), v in cells.items():
        k = knob(strat)
        if k:
            by_curve[(arm, k[0], pred)].append((k[1], v))
    for (arm, kind, pred), pts in sorted(by_curve.items()):
        pts.sort(key=lambda p: p[0])
        mean = [(k, float(np.mean([x["recall30"] for x in v])), float(np.mean([x["work"] for x in v]))) for k, v in pts]
        w, wk = w_at(mean)
        at = next(((k, r, wo) for k, r, wo in mean if wo >= NOMINAL), None)
        curves[f"{arm}/{kind}/{pred}"] = {"points": mean, "w090": w, "w090_kind": wk,
                                          "recall_at_50k": None if at is None else {"knob": at[0], "recall30": at[1],
                                                                                    "work": at[2]},
                                          "at_50k": "measured" if at else "unreachable"}
        for (k0, v0), (k1, v1) in zip(pts, pts[1:]):
            m0, m1 = np.mean([x["work"] for x in v0]), np.mean([x["work"] for x in v1])
            if m1 < m0 - 1e-9:
                q0 = {x["query"]: x["work"] for x in v0}
                drops = [q0[x["query"]] - x["work"] for x in v1 if q0[x["query"]] > x["work"]]
                # Amendment A2: for a multi-member arm (DLI), the restart bound applies to each member's own work
                # (`plans[].work`), because each member runs its own routed search and widening restart.
                p0 = {x["query"]: {m.get("level"): m.get("work", 0) for m in x["plans"]} for x in v0}
                member_drops = [p0[x["query"]].get(m.get("level"), 0) - m.get("work", 0)
                                for x in v1 for m in x["plans"] if len(x["plans"]) > 1]
                max_member = max(member_drops, default=None)
                rec = {"curve": f"{arm}/{kind}/{pred}", "knobs": [k0, k1], "mean_work": [float(m0), float(m1)],
                       "max_query_drop": max(drops), "max_member_drop": max_member}
                bound = max(drops) if max_member is None else max_member
                if kind == "nprobe" and k1 <= 2 * k0 and bound < K:
                    restarts.append(rec)
                else:
                    problems.append({"non_monotone": rec})
    for r in rows:
        if r["duplicates"] or r["nonqualifying"]:
            problems.append({"bad_ids": {k: r[k] for k in ("arm", "strategy", "predicate", "query")}})
        if r["strategy"] == "exact" and r["recall30"] < 1:
            g = gold[(r["predicate"], r["query"])]
            if not g["tie_at_boundary"]:
                problems.append({"exact_shortfall_without_tie": {k: r[k] for k in ("arm", "predicate", "query", "recall30")}})
    return {"summary": summary, "curves": curves, "widening_restarts": restarts, "problems": problems,
            "pass": not problems}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", required=True)
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    res = analyze(args.run)
    text = json.dumps(res, indent=1, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({"pass": res["pass"], "problems": res["problems"][:10], "restarts": len(res["widening_restarts"])},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
