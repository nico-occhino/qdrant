"""Analyze native-replay selection-frontier rows (continuation42).

Protocol (fixed before inspecting results):
* Work = live scored vectors (original + Plain), the scorer being held fixed.
* Operating points are selected on the validation set only (smallest budget
  with mean Recall@30 >= target), then reported unchanged on test sets.
* Matched-work comparisons interpolate each query's recall along its own
  budget sweep at fixed work levels and compare policies paired per query,
  with a percentile bootstrap (10,000 resamples, seed 20261117) over queries.
"""
import argparse, json, collections, statistics, random
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--validation", default="val")
p.add_argument("--target", type=float, default=0.95)
p.add_argument("--work", default="2000,3000,4000,6000,8000,10000,12000")
a = p.parse_args()
rows = [json.loads(l) for l in (a.run / "rows.jsonl").open()]
refs = [json.loads(l) for l in (a.run / "reference_costs.jsonl").open()]
sets = sorted({r["set"] for r in rows})
policies = sorted({r["policy"] for r in rows})
budgets = sorted({r["budget"] for r in rows})
work_levels = [int(x) for x in a.work.split(",")]
rng = random.Random(20261117)

by = collections.defaultdict(list)  # (set, policy, budget) -> rows
for r in rows:
    by[(r["set"], r["policy"], r["budget"])].append(r)

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def boot_ci(diffs, n=10000):
    if not diffs:
        return (float("nan"),) * 2
    m = len(diffs)
    stats = sorted(mean([diffs[rng.randrange(m)] for _ in range(m)]) for _ in range(n))
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]

out = {"sets": sets, "policies": policies, "budgets": budgets, "curves": {}, "selected": {},
       "matched_work": {}, "reference": {}, "misses": {}}

# 1. Mean curves.
for s in sets:
    for pol in policies:
        curve = []
        for b in budgets:
            rs = by[(s, pol, b)]
            if not rs:
                continue
            curve.append({
                "budget": b,
                "recall10": mean([r["recall10"] / 10 for r in rs]),
                "recall30": mean([r["recall30"] / 30 for r in rs]),
                "scored": mean([r["scored_total"] for r in rs]),
                "posting_entries": mean([r["posting_entries"] for r in rs]),
                "unique_offsets": mean([r["unique_offsets"] for r in rs]),
                "plain_scored": mean([r["plain_scored"] for r in rs]),
                "miss_bucket_unselected": mean([r["miss_bucket_unselected"] for r in rs]),
                "miss_plain_unselected": mean([r["miss_plain_unselected"] for r in rs]),
                "miss_selected_but_lost": mean([r["miss_selected_but_lost"] for r in rs]),
                "queries": len(rs),
            })
        out["curves"][f"{s}/{pol}"] = curve

# 2. Validation-locked operating points.
for pol in policies:
    chosen = None
    for pt in out["curves"].get(f"{a.validation}/{pol}", []):
        if pt["recall30"] >= a.target:
            chosen = pt["budget"]
            break
    entry = {"budget": chosen}
    if chosen is not None:
        for s in sets:
            pt = next((c for c in out["curves"][f"{s}/{pol}"] if c["budget"] == chosen), None)
            entry[s] = pt
    out["selected"][pol] = entry

# 3. Per-query interpolation at matched work.
def interp(points, w):
    """points: sorted (work, recall) for one query; recall at work w (clamped)."""
    pts = sorted(points)
    if w <= pts[0][0]:
        return pts[0][1]
    for (w0, r0), (w1, r1) in zip(pts, pts[1:]):
        if w0 <= w <= w1:
            return r0 if w1 == w0 else r0 + (r1 - r0) * (w - w0) / (w1 - w0)
    return None  # beyond the measured range: not extrapolated

for s in sets:
    per_q = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["set"] == s:
            per_q[r["policy"]][r["query"]].append((r["scored_total"], r["recall30"] / 30))
    base = per_q.get("slot", {})
    res = {}
    for pol in policies:
        res[pol] = {}
        for w in work_levels:
            vals, diffs = [], []
            for q, pts in per_q[pol].items():
                v = interp(pts, w)
                if v is None:
                    continue
                vals.append(v)
                if pol != "slot" and q in base:
                    b = interp(base[q], w)
                    if b is not None:
                        diffs.append(v - b)
            lo, hi = boot_ci(diffs) if pol != "slot" else (None, None)
            res[pol][w] = {"recall30": mean(vals), "n": len(vals),
                           "diff_vs_slot": mean(diffs) if diffs else None, "ci95": [lo, hi]}
    out["matched_work"][s] = res

# 4. Reference costs vs policies' coverage costs.
for s in sets:
    rr = [r for r in refs if r["set"] == s]
    def summ(key):
        xs = [r[key] for r in rr if key in r]
        if not xs:
            return None
        xs.sort()
        return {"mean": mean(xs), "median": statistics.median(xs), "p90": xs[int(0.9 * (len(xs) - 1))]}
    ref = {k: summ(k) for k in ["oracle_cost_top10", "rank_cost_top10", "oracle_cost_top30", "rank_cost_top30",
                                "static_oracle_cost_top10", "static_rank_cost_top10",
                                "static_oracle_cost_top30", "static_rank_cost_top30"]}
    # Policy cost to fully cover each query's top-10 / top-30 (first budget reaching it).
    for pol in policies:
        for k, key in [(10, "recall10"), (30, "recall30")]:
            costs, uncovered = [], 0
            qs = {r["query"] for r in rows if r["set"] == s}
            for q in qs:
                pts = sorted((r["budget"], r["scored_total"], r[key]) for r in rows
                             if r["set"] == s and r["policy"] == pol and r["query"] == q)
                hit = next((w for (_, w, rec) in pts if rec == k), None)
                if hit is None:
                    uncovered += 1
                else:
                    costs.append(hit)
            costs.sort()
            ref[f"{pol}_cover_top{k}"] = {"mean": mean(costs), "median": statistics.median(costs) if costs else None,
                                          "uncovered_queries": uncovered, "n": len(qs)}
    out["reference"][s] = ref

# 5. Miss attribution at each budget for the default policy and rank depth of missed buckets.
for s in sets:
    for pol in policies:
        ranks = collections.Counter()
        for r in rows:
            if r["set"] == s and r["policy"] == pol and r["budget"] == 10000:
                for level, rank in r["unselected_level_rank"]:
                    band = "1-5" if rank < 5 else "6-20" if rank < 20 else "21-50" if rank < 50 else ">50"
                    ranks[f"L{level}:{band}"] += 1
        out["misses"][f"{s}/{pol}@10000"] = dict(ranks)

(a.run / "analysis.json").write_text(json.dumps(out, indent=2))
# Compact console summary.
for s in sets:
    print(f"== {s}")
    for pol in policies:
        line = " ".join(f"{pt['budget']}:{pt['recall30']:.3f}/{pt['scored']:.0f}" for pt in out["curves"][f"{s}/{pol}"])
        print(f"  {pol:13s} {line}")
print("== validation-locked operating points (target R@30 >= %.2f)" % a.target)
for pol, e in out["selected"].items():
    print(" ", pol, e["budget"], {s: (round(e[s]["recall30"], 4), round(e[s]["scored"])) for s in sets if e.get(s)})
print("== reference costs (mean scored vectors to cover gold)")
for s in sets:
    print(" ", s, {k: (round(v["mean"]) if v and v.get("mean") == v.get("mean") else v) for k, v in out["reference"][s].items() if v and "cover" not in k})
    print(" ", s, {k: (round(v["mean"]) if v["mean"] == v["mean"] else None, v["uncovered_queries"]) for k, v in out["reference"][s].items() if v and "cover" in k})
