"""Independent analysis of a native-replay selection frontier (H-10M, job 25677858).

Written after the job finished, to replace the matched-work part of
`analyze_frontier.py`, which (a) clamps per-query interpolation below a
policy's smallest measured work (assigning recall measured at *higher* work to
lower work, i.e. extrapolation), (b) drops queries above the measured range,
so paired samples differ in size between work levels, (c) only compares
against `slot`, and (d) computes coverage costs over covered queries only.

Estimands (all on the evaluation set, queries are the resampling unit):

* Mean curves: per (policy, requested budget) the mean over queries of
  Recall@10, Recall@30 and realized work. Work = live scored vectors
  (original + Plain); the scorer is fixed, so work is the selection cost.
* W(target): realized mean work at which the mean Recall@30 curve reaches the
  target, by linear interpolation in work between the two *measured* budgets
  that bracket it. If the curve never reaches the target inside the sweep the
  value is "not reached"; nothing is extrapolated.
* Saving(static vs P) = 1 - W_static / W_P.
* Matched-work difference at work w: interpolated mean-curve recall of A minus
  B, only for w inside both policies' measured mean-work ranges.
* Equal-budget paired difference: per-query, no interpolation (empirical).

Uncertainty: paired percentile bootstrap over query identities (the same
resampled query multiset for every policy), B resamples, fixed seed. This is
uncertainty over queries for ONE trained topology and ONE static model. It is
not uncertainty over training seeds, ingestion orders or hardware runs.
"""
import argparse, collections, json, math
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("run", type=Path, help="directory with rows.jsonl, reference_costs.jsonl, topology.json")
p.add_argument("--eval-set", default="test")
p.add_argument("--val-set", default="val")
p.add_argument("--targets", default="0.90,0.95")
p.add_argument("--lock-target", type=float, default=0.90)
p.add_argument("--resamples", type=int, default=10000)
p.add_argument("--seed", type=int, default=20261118)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

rows = [json.loads(l) for l in (a.run / "rows.jsonl").open()]
refs = [json.loads(l) for l in (a.run / "reference_costs.jsonl").open()]
topo = json.loads((a.run / "topology.json").read_text())
N_TOTAL = topo["total_live"]
POLICIES = [p for p in ["static_flat", "slot", "proportional", "tdensity2", "density", "reserved"]
            if any(r["policy"] == p for r in rows)]
DLI = [p for p in POLICIES if p != "static_flat"]
BUDGETS = sorted({r["budget"] for r in rows})
TARGETS = [float(x) for x in a.targets.split(",")]
rng = np.random.default_rng(a.seed)
out = {"run": str(a.run), "eval_set": a.eval_set, "val_set": a.val_set, "resamples": a.resamples,
       "seed": a.seed, "total_live": N_TOTAL, "policies": POLICIES, "budgets": BUDGETS}

# ---------------------------------------------------------------- integrity
by_key = collections.Counter((r["set"], r["query"], r["policy"], r["budget"]) for r in rows)
sets = sorted({r["set"] for r in rows})
integrity = {"rows": len(rows), "duplicate_keys": sum(v > 1 for v in by_key.values()), "sets": {}}
for s in sets:
    qs = sorted({r["query"] for r in rows if r["set"] == s})
    integrity["sets"][s] = {
        "queries": len(qs), "contiguous_ids": qs == list(range(len(qs))),
        "complete_grid": all(by_key[(s, q, pol, b)] == 1 for q in qs for pol in POLICIES for b in BUDGETS),
    }
viol = collections.Counter()
for r in rows:
    if r["scored_total"] != r["original_scored"] + r["plain_scored"]:
        viol["scored_total != original + plain"] += 1
    if r["unique_offsets"] > r["posting_entries"]:
        viol["unique_offsets > posting_entries"] += 1
    if not (0 <= r["recall10"] <= 10 and 0 <= r["recall30"] <= 30):
        viol["recall out of range"] += 1
    if r["policy"] != "static_flat":
        if r["miss_plain_unselected"] + r["miss_bucket_unselected"] + r["miss_selected_but_lost"] != 30 - r["recall30"]:
            viol["DLI miss attribution does not sum to 30 - recall30"] += 1
        if len(r["unselected_level_rank"]) != r["miss_bucket_unselected"]:
            viol["unselected_level_rank length != miss_bucket_unselected"] += 1
    elif r["miss_plain_unselected"] or r["miss_bucket_unselected"] or r["miss_selected_but_lost"]:
        viol["static row carries miss attribution (harness computes none)"] += 1
# Per-query monotonicity along the budget sweep (selection is greedy, so
# work must not decrease; recall may, through Float16 tie order).
pq = collections.defaultdict(list)
for r in rows:
    pq[(r["set"], r["policy"], r["query"])].append(r)
mono = collections.Counter()
for (s, pol, q), rs in pq.items():
    rs.sort(key=lambda r: r["budget"])
    for x, y in zip(rs, rs[1:]):
        if y["scored_total"] < x["scored_total"]:
            mono[f"{pol}: work decreases with budget"] += 1
        if y["recall30"] < x["recall30"]:
            mono[f"{pol}: recall30 decreases with budget"] += 1
integrity["violations"] = dict(viol)
integrity["monotonicity_events"] = dict(mono)
plain_sizes = sorted({r["plain_scored"] for r in rows} - {0})
integrity["plain_scored_values"] = plain_sizes
out["integrity"] = integrity

# ------------------------------------------------------------ arrays [P,B,Q]
def arrays(s):
    qs = sorted({r["query"] for r in rows if r["set"] == s})
    qi = {q: i for i, q in enumerate(qs)}
    shape = (len(POLICIES), len(BUDGETS), len(qs))
    A = {k: np.zeros(shape) for k in ["r10", "r30", "work", "post", "uniq", "plain", "orig", "elapsed",
                                      "miss_plain", "miss_bucket", "miss_lost"]}
    for r in rows:
        if r["set"] != s:
            continue
        i, j, k = POLICIES.index(r["policy"]), BUDGETS.index(r["budget"]), qi[r["query"]]
        A["r10"][i, j, k] = r["recall10"] / 10
        A["r30"][i, j, k] = r["recall30"] / 30
        A["work"][i, j, k] = r["scored_total"]
        A["post"][i, j, k] = r["posting_entries"]
        A["uniq"][i, j, k] = r["unique_offsets"]
        A["plain"][i, j, k] = r["plain_scored"]
        A["orig"][i, j, k] = r["original_scored"]
        A["elapsed"][i, j, k] = r.get("elapsed_ns", float("nan"))
        A["miss_plain"][i, j, k] = r["miss_plain_unselected"]
        A["miss_bucket"][i, j, k] = r["miss_bucket_unselected"]
        A["miss_lost"][i, j, k] = r["miss_selected_but_lost"]
    return qs, A

def crossing(work, rec, target):
    """work, rec: mean curves over budgets (1-D). Linear interpolation between
    bracketing measured points; None if not reached in range. Returns (w, kind)."""
    for j in range(len(rec)):
        if rec[j] >= target:
            if j == 0:
                return (float(work[0]), "at_or_below_first_point")  # reached already at smallest budget
            w0, w1, r0, r1 = work[j - 1], work[j], rec[j - 1], rec[j]
            return (float(w0 + (target - r0) * (w1 - w0) / (r1 - r0)), "interpolated")
    return (None, "not_reached")

def interp_curve(work, rec, w):
    """Mean-curve recall at work w; None outside the measured work range.
    Requires work increasing along the budget sweep (checked)."""
    if w < work[0] or w > work[-1]:
        return None
    return float(np.interp(w, work, rec))

def ci(xs):
    xs = np.asarray([x for x in xs if x is not None and not math.isnan(x)])
    if len(xs) == 0:
        return [None, None]
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]

def evaluate(s):
    qs, A = arrays(s)
    nq = len(qs)
    res = {"queries": nq}
    idx = rng.integers(0, nq, size=(a.resamples, nq))
    # Bootstrap mean curves: [B, P, budgets]
    boot = {k: A[k][:, :, idx].mean(axis=3).transpose(2, 0, 1) for k in ["r10", "r30", "work"]}
    mean = {k: A[k].mean(axis=2) for k in A}

    # 1. curves with CIs and distributions
    curves = {}
    for i, pol in enumerate(POLICIES):
        pts = []
        for j, b in enumerate(BUDGETS):
            r30 = A["r30"][i, j]; w = A["work"][i, j]
            pts.append({
                "budget": b,
                "recall10": float(mean["r10"][i, j]), "recall10_ci": ci(boot["r10"][:, i, j]),
                "recall30": float(mean["r30"][i, j]), "recall30_ci": ci(boot["r30"][:, i, j]),
                "recall30_q10_q50_q90": [float(np.percentile(r30, x)) for x in (10, 50, 90)],
                "frac_queries_r30_ge_0.90": float((r30 >= 0.9 - 1e-9).mean()),
                "work_mean": float(w.mean()), "work_ci": ci(boot["work"][:, i, j]),
                "work_q10_q50_q90": [float(np.percentile(w, x)) for x in (10, 50, 90)],
                "realized_over_requested": float(w.mean() / b),
                "selectivity_pct": float(100 * w.mean() / N_TOTAL),
                "posting_entries": float(mean["post"][i, j]), "unique_offsets": float(mean["uniq"][i, j]),
                "learned_work": float(mean["orig"][i, j]), "plain_work": float(mean["plain"][i, j]),
                "plain_selected_frac": float((A["plain"][i, j] > 0).mean()),
                "miss_plain_unselected": float(mean["miss_plain"][i, j]),
                "miss_bucket_unselected": float(mean["miss_bucket"][i, j]),
                "miss_selected_but_lost": float(mean["miss_lost"][i, j]),
                "elapsed_ms_median": float(np.median(A["elapsed"][i, j]) / 1e6),
                "ns_per_scored_vector_median": float(np.median(A["elapsed"][i, j] / np.maximum(w, 1))),
            })
        curves[pol] = pts
    res["curves"] = curves

    # 2. Work to reach targets, savings vs static (paired bootstrap)
    si = POLICIES.index("static_flat")
    wt = {}
    for t in TARGETS:
        entry = {}
        point = {pol: crossing(mean["work"][i], mean["r30"][i], t) for i, pol in enumerate(POLICIES)}
        bw = {pol: [crossing(boot["work"][b, i], boot["r30"][b, i], t)[0] for b in range(a.resamples)]
              for i, pol in enumerate(POLICIES)}
        for pol in POLICIES:
            w, kind = point[pol]
            e = {"work": w, "kind": kind, "work_ci": ci(bw[pol]),
                 "resamples_not_reached": sum(x is None for x in bw[pol]) / a.resamples}
            # bracketing empirical points
            i = POLICIES.index(pol)
            e["bracket"] = [(BUDGETS[j], float(mean["work"][i, j]), float(mean["r30"][i, j]))
                            for j in range(len(BUDGETS))
                            if (j + 1 < len(BUDGETS) and mean["r30"][i, j] < t <= mean["r30"][i, j + 1])
                            or (j > 0 and mean["r30"][i, j - 1] < t <= mean["r30"][i, j])]
            if pol != "static_flat":
                sv = [None if (x is None or y is None) else 1 - x / y for x, y in zip(bw["static_flat"], bw[pol])]
                e["static_saving"] = None if (w is None or point["static_flat"][0] is None) else 1 - point["static_flat"][0] / w
                e["static_saving_ci"] = ci(sv)
                e["P(saving >= 0.30)"] = float(np.mean([x is not None and x >= 0.30 for x in sv]))
                e["P(saving > 0)"] = float(np.mean([x is not None and x > 0 for x in sv]))
            entry[pol] = e
        wt[str(t)] = entry
    res["work_to_target"] = wt

    # 3. Matched-work differences on mean curves, all pairs vs static and DLI vs slot.
    monotone = {pol: bool(np.all(np.diff(mean["work"][i]) > 0)) for i, pol in enumerate(POLICIES)}
    res["mean_work_increasing_with_budget"] = monotone
    lo = max(mean["work"][:, 0].min(), 1.0)
    hi = mean["work"][:, -1].max()
    grid = [float(x) for x in np.geomspace(lo, hi, 25)]
    pairs = [("static_flat", d) for d in DLI] + [(d, "slot") for d in DLI if d != "slot"]
    mw = {}
    for A_, B_ in pairs:
        ia, ib = POLICIES.index(A_), POLICIES.index(B_)
        lst = []
        for w in grid:
            ra = interp_curve(mean["work"][ia], mean["r30"][ia], w)
            rb = interp_curve(mean["work"][ib], mean["r30"][ib], w)
            if ra is None or rb is None:
                continue
            diffs = []
            for b in range(a.resamples):
                x = interp_curve(boot["work"][b, ia], boot["r30"][b, ia], w)
                y = interp_curve(boot["work"][b, ib], boot["r30"][b, ib], w)
                diffs.append(None if x is None or y is None else x - y)
            c = ci(diffs)
            lst.append({"work": w, "recall30_a": ra, "recall30_b": rb, "diff": ra - rb, "ci95": c,
                        "excluded_resamples": sum(d is None for d in diffs) / a.resamples,
                        "significant": c[0] is not None and (c[0] > 0 or c[1] < 0)})
        mw[f"{A_} - {B_}"] = lst
    res["matched_work"] = mw

    # 4. Equal requested budget, per-query paired (empirical, no interpolation).
    eb = {}
    for d in DLI:
        ib = POLICIES.index(d)
        lst = []
        for j, b in enumerate(BUDGETS):
            dr = A["r30"][si, j] - A["r30"][ib, j]
            dw = A["work"][si, j] - A["work"][ib, j]
            bdr = dr[idx].mean(axis=1); bdw = dw[idx].mean(axis=1)
            lst.append({"budget": b, "recall30_diff": float(dr.mean()), "recall30_diff_ci": ci(bdr),
                        "work_diff": float(dw.mean()), "work_diff_ci": ci(bdw),
                        "queries_static_better": int((dr > 0).sum()), "queries_dli_better": int((dr < 0).sum())})
        eb[f"static_flat - {d}"] = lst
    res["equal_budget"] = eb

    # 5. Old method reproduced, to quantify its bias (per-query interp, clamped below).
    def old_interp(points, w):
        pts = sorted(points)
        if w <= pts[0][0]:
            return pts[0][1]
        for (w0, r0), (w1, r1) in zip(pts, pts[1:]):
            if w0 <= w <= w1:
                return r0 if w1 == w0 else r0 + (r1 - r0) * (w - w0) / (w1 - w0)
        return None
    old = {}
    for i, pol in enumerate(POLICIES):
        o = {}
        for w in [20000, 40000, 80000, 160000, 320000, 640000]:
            vals = [old_interp(list(zip(A["work"][i, :, k], A["r30"][i, :, k])), w) for k in range(nq)]
            vv = [v for v in vals if v is not None]
            clamped = sum(1 for k in range(nq) if w <= A["work"][i, 0, k])
            o[w] = {"recall30": float(np.mean(vv)) if vv else None, "n": len(vv), "clamped_below": clamped}
        old[pol] = o
    res["old_method_per_query_interp"] = old
    return res, qs, A, mean

ev, qs_eval, A_eval, mean_eval = evaluate(a.eval_set)
out["eval"] = ev
if a.val_set in sets:
    out["val"], _, A_val, mean_val = evaluate(a.val_set)
    # validation-locked operating points (smallest budget with val mean R@30 >= lock target)
    locked = {}
    for i, pol in enumerate(POLICIES):
        j = next((j for j in range(len(BUDGETS)) if mean_val["r30"][i, j] >= a.lock_target), None)
        if j is None:
            locked[pol] = {"budget": None}
            continue
        c = out["eval"]["curves"][pol][j]
        locked[pol] = {"budget": BUDGETS[j], "val_recall30": float(mean_val["r30"][i, j]),
                       "val_work": float(mean_val["work"][i, j]),
                       "eval_recall30": c["recall30"], "eval_recall30_ci": c["recall30_ci"],
                       "eval_work": c["work_mean"], "eval_work_ci": c["work_ci"]}
    out["validation_locked"] = {"target": a.lock_target, "points": locked}

# ---------------------------------------------------- reference costs (eval)
rr = [r for r in refs if r["set"] == a.eval_set]
def summ(key):
    xs = np.array([r[key] for r in rr], dtype=float)
    return {"mean": float(xs.mean()), "median": float(np.median(xs)), "p90": float(np.percentile(xs, 90))}
ref = {k: summ(k) for k in ["oracle_cost_top10", "rank_cost_top10", "oracle_cost_top30", "rank_cost_top30",
                            "static_oracle_cost_top10", "static_rank_cost_top10",
                            "static_oracle_cost_top30", "static_rank_cost_top30"]}
for k in (10, 30):
    o = np.array([r[f"oracle_cost_top{k}"] for r in rr], float); rk = np.array([r[f"rank_cost_top{k}"] for r in rr], float)
    so = np.array([r[f"static_oracle_cost_top{k}"] for r in rr], float); sr = np.array([r[f"static_rank_cost_top{k}"] for r in rr], float)
    ref[f"rank_over_oracle_top{k}"] = {"dli_ratio_of_means": float(rk.mean() / o.mean()),
                                       "static_ratio_of_means": float(sr.mean() / so.mean()),
                                       "dli_median_query_ratio": float(np.median(rk / o)),
                                       "static_median_query_ratio": float(np.median(sr / so))}
    # gold needing the Plain buffer (oracle cost includes plain_size when any gold is in Plain)
    ref[f"queries_with_gold_levels_top{k}"] = dict(collections.Counter(
        lvl for r in rr for lvl, _ in r[f"rank_depth_top{k}"]))
# Coverage cost: per query the realized work of the first budget that covers all top-k gold.
# Uncovered queries are right-censored (cost > largest measured work); medians include them as +inf.
cover = {}
for pol in POLICIES:
    i = POLICIES.index(pol)
    for k, key in ((10, "r10"), (30, "r30")):
        costs = []
        for q in range(len(qs_eval)):
            hit = next((A_eval["work"][i, j, q] for j in range(len(BUDGETS)) if A_eval[key][i, j, q] >= 1 - 1e-9), math.inf)
            costs.append(hit)
        c = np.array(costs)
        fin = c[np.isfinite(c)]
        cover[f"{pol}_top{k}"] = {"covered": int(len(fin)), "n": len(c),
                                  "median_censored": float(np.median(c)) if np.isfinite(np.median(c)) else None,
                                  "mean_over_covered_only": float(fin.mean()) if len(fin) else None}
ref["coverage"] = cover
out["reference_costs"] = ref

# --------------------------------------- per-level miss attribution (eval)
lvl = {}
for pol in DLI:
    for b in BUDGETS:
        c = collections.Counter(); bands = collections.Counter(); per_level = collections.Counter(); nq = 0
        for r in rows:
            if r["set"] == a.eval_set and r["policy"] == pol and r["budget"] == b:
                nq += 1
                c["plain"] += r["miss_plain_unselected"]
                c["lost"] += r["miss_selected_but_lost"]
                for level, rank in r["unselected_level_rank"]:
                    c[f"L{level}"] += 1
                    band = "1-5" if rank < 5 else "6-20" if rank < 20 else "21-50" if rank < 50 else ">50"
                    bands[band] += 1
                for level, n in r["buckets_per_level"].items():
                    per_level[level] += n
        lvl[f"{pol}@{b}"] = {"misses_per_query": {k: v / nq for k, v in sorted(c.items())},
                             "unselected_rank_band_per_query": {k: v / nq for k, v in bands.items()},
                             "buckets_per_level_mean": {k: v / nq for k, v in sorted(per_level.items())}}
out["per_level"] = lvl
members = [{k: m[k] for k in ("level", "live", "buckets", "bucket_size_max")} for m in topo["members"]]
for m, full in zip(members, topo["members"]):
    bs = full.get("bucket_sizes") or []
    if bs:
        x = np.array(bs, float)
        m.update({"bucket_mean": float(x.mean()), "bucket_median": float(np.median(x)),
                  "bucket_cv": float(x.std() / x.mean()), "bucket_min": int(x.min()),
                  "top10pct_bucket_share_of_points": float(np.sort(x)[-max(1, len(x) // 10):].sum() / x.sum())})
st = topo.get("static_reference")
if st:
    x = np.array(st["bucket_sizes"], float)
    members.append({"level": "static", "live": int(x.sum()), "buckets": len(x), "bucket_size_max": int(x.max()),
                    "bucket_mean": float(x.mean()), "bucket_median": float(np.median(x)),
                    "bucket_cv": float(x.std() / x.mean()), "bucket_min": int(x.min()),
                    "top10pct_bucket_share_of_points": float(np.sort(x)[-len(x) // 10:].sum() / x.sum()),
                    "build_seconds": st.get("build_seconds")})
out["topology"] = members

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
print("wrote", a.out)
