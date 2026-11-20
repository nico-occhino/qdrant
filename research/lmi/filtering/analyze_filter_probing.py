#!/usr/bin/env python3
"""Analysis for PREREGISTRATION_FILTER_PROBING.md.

    calibrate WORK            validation only, independent filters only -> WORK/policies.json
    report    WORK            test simulation + native policy latency    -> WORK/analysis.json, WORK/analysis.md

Inputs are the JSONL files written by lib/segment/tests/lmi_filter_probing_sisap300k.rs.
Policy decisions are recomputed here with the same rules as filtered.rs::decide_policy;
the native `policies` phase checks every result against the simulated grid cell.
Calibrated recall is a mean over the validation query distribution, never a per-query
guarantee; tails are reported for every policy.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P_GRID = [1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 128, 256]
TAUS = [0.90, 0.95]
ROUTERS = ["ivf", "mlp"]
FAMILIES = ["ind", "slab"]
BOOT = 2000
RNG = np.random.default_rng(20261119)


def jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def skey(s):
    return json.dumps(s)


def boot_mean_ci(values, n=BOOT):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ------------------------------------------------------------------ loading

class Data:
    def __init__(self, work, splits):
        self.work = Path(work)
        self.filters = {}
        for r in jsonl(self.work / "grid/filters.jsonl"):
            self.filters[(r["router"], r["family"], skey(r["s"]))] = r
        # grid[(router, family, split, s, cell)] -> {q: record}
        self.grid = defaultdict(dict)
        for r in jsonl(self.work / "grid/records.jsonl"):
            if r["split"] not in splits:
                continue
            routed = r["routed"]
            self.grid[(r["router"], r["family"], r["split"], skey(r["s"]), r["cell"])][r["q"]] = {
                "recall": r["recall"],
                "k_eff": r["k_eff"],
                "hash": r["hash"],
                "final": routed["final_probe"] if routed else None,
                "cum": routed["cumulative"] if routed else None,
                "widenings": routed["widenings"] if routed else 0,
            }
        self.selectivities = sorted({json.loads(k[3]) for k in self.grid}, reverse=True)
        # timing[(router, family, split, s, cell)] -> {q: median ns over repetitions}
        reps = defaultdict(lambda: defaultdict(list))
        path = self.work / "timing/timing.jsonl"
        if path.exists():
            for r in jsonl(path):
                if r["split"] in splits:
                    reps[(r["router"], r["family"], r["split"], skey(r["s"]), r["cell"])][r["q"]].append(r["ns"])
        self.timing = {k: {q: float(np.median(v)) for q, v in d.items()} for k, d in reps.items()}
        self.stages = []
        path = self.work / "stages/stages.jsonl"
        if path.exists():
            self.stages = list(jsonl(path))

    def queries(self, router, family, split, s):
        return sorted(self.grid[(router, family, split, skey(s), "exact")])


# ------------------------------------------------------------------ cost model

def lstsq_nonneg(X, y):
    """Relative-error least squares (weights 1/y^2, Amendment 1): latencies span four orders
    of magnitude, and an unweighted fit is dominated by the largest cells. Any negative
    coefficient is dropped (fixed to 0) and the rest refitted."""
    w = 1.0 / np.maximum(np.asarray(y, float), 1.0)
    X, y = X * w[:, None], np.asarray(y, float) * w
    active = list(range(X.shape[1]))
    while True:
        coef = np.zeros(X.shape[1])
        sol, *_ = np.linalg.lstsq(X[:, active], y, rcond=None)
        coef[active] = sol
        negative = [i for i in active if coef[i] < 0]
        if not negative:
            return coef
        active = [i for i in active if i not in negative]


def fit_costs(data, router):
    """Validation, independent family only."""
    family, split = "ind", "val"
    # Exact plan: per-query median latency against the payload estimate, per iteration path.
    exact = {True: ([], []), False: ([], [])}
    for s in data.selectivities:
        f = data.filters[(router, family, skey(s))]
        est = f["estimate"]
        t = data.timing.get((router, family, split, skey(s), "exact"), {})
        for ns in t.values():
            exact[est["primary_iteration"]][0].append(est["exp"])
            exact[est["primary_iteration"]][1].append(ns)
    exact_fit = {}
    for path, (x, y) in exact.items():
        if len(x) < 2 or len(set(x)) < 2:
            exact_fit[path] = None
            continue
        X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
        a, b = lstsq_nonneg(X, np.asarray(y, float))
        exact_fit[path] = {"fixed_ns": float(a), "per_unit_ns": float(b)}
    # One path never observed: fall back to the other (recorded in the calibration output).
    observed = [v for v in exact_fit.values() if v]
    for path in exact_fit:
        if exact_fit[path] is None:
            exact_fit[path] = observed[0]
    # Routed plan: latency against cumulative filter evaluations and scored vectors.
    rows, y = [], []
    for s in data.selectivities:
        for p in P_GRID:
            cell = (router, family, split, skey(s), str(p))
            t = data.timing.get(cell, {})
            g = data.grid[cell]
            for q, ns in t.items():
                cum = g[q]["cum"]
                rows.append([1.0, cum["filter_evaluations"], cum["scored"]])
                y.append(ns)
    a, b_check, b_score = lstsq_nonneg(np.asarray(rows), np.asarray(y, float))
    # Unique candidates per probe at s = 100% (no widening there).
    unique_at = []
    for p in P_GRID:
        g = data.grid[(router, family, split, skey(1.0), str(p))]
        unique_at.append([p, float(np.mean([r["final"]["unique_candidates"] for r in g.values()]))])
    linear = {
        "exact_iteration": exact_fit[True],
        "exact_scan": exact_fit[False],
        "route_fixed_ns": float(a),
        "route_per_candidate_ns": float(b_check),
        "route_per_scored_ns": float(b_score),
        "unique_candidates_at": unique_at,
        "exact_table": None,
        "routed_table": None,
    }
    # Empirical tables (Amendment 1): mean validation latency at each filter's estimate.
    exact_table, routed = [], {p: [] for p in P_GRID}
    for s in data.selectivities:
        est = data.filters[(router, family, skey(s))]["estimate"]
        t = data.timing.get((router, family, split, skey(s), "exact"))
        if t:
            exact_table.append([float(est["exp"]), float(np.mean(list(t.values())))])
        for p in P_GRID:
            t = data.timing.get((router, family, split, skey(s), str(p)))
            if t:
                routed[p].append([est["exp"] / max(est["available"], 1), float(np.mean(list(t.values())))])
    table = dict(linear, exact_table=sorted(exact_table),
                 routed_table=[[p, sorted(v)] for p, v in routed.items()])
    return linear, table


def prediction_error(data, router, costs):
    """Predicted / measured mean latency per validation cell (independent family)."""
    rows = []
    for s in data.selectivities:
        est = data.filters[(router, "ind", skey(s))]["estimate"]
        for cell in [str(p) for p in P_GRID] + ["exact"]:
            t = data.timing.get((router, "ind", "val", skey(s), cell))
            if not t:
                continue
            measured = float(np.mean(list(t.values())))
            predicted = exact_ns(costs, est) if cell == "exact" else routed_ns(costs, int(cell), est)
            rows.append({"s": s, "cell": cell, "measured_ms": measured / 1e6, "ratio": predicted / measured})
    return rows


def stage_check_cost(data, router):
    """Filter-evaluation cost from the stage pass (ns per evaluation), independent family."""
    ns, evals = 0, 0
    for r in data.stages:
        if r["router"] == router and r["family"] == "ind" and r["routed"] and r["routed"]["stage_ns"]:
            ns += r["routed"]["stage_ns"]["filter"]
            evals += r["routed"]["cumulative"]["filter_evaluations"]
    return ns / evals if evals else None


# ------------------------------------------------------------------ decisions (mirror filtered.rs)

def interpolate_log(points, x):
    """Mirror of filtered.rs::interpolate_log."""
    if not points:
        return None
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    i = next(i for i, (px, _) in enumerate(points) if px > x)
    (x0, y0), (x1, y1) = points[i - 1], points[i]
    if x0 <= 0 or y0 <= 0 or y1 <= 0:
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    t = (np.log(x) - np.log(x0)) / (np.log(x1) - np.log(x0))
    return float(np.exp(np.log(y0) + t * (np.log(y1) - np.log(y0))))


def exact_ns(costs, est):
    if costs.get("exact_table"):
        return interpolate_log(costs["exact_table"], est["exp"])
    c = costs["exact_iteration"] if est["primary_iteration"] else costs["exact_scan"]
    return c["fixed_ns"] + c["per_unit_ns"] * est["exp"]


def routed_ns(costs, p, est):
    sel = est["exp"] / max(est["available"], 1)
    if costs.get("routed_table"):
        return interpolate_log(dict((int(k), v) for k, v in costs["routed_table"])[p], sel)
    unique = dict((int(k), v) for k, v in costs["unique_candidates_at"])[p]
    return costs["route_fixed_ns"] + costs["route_per_candidate_ns"] * unique + costs["route_per_scored_ns"] * sel * unique


def by_cost(costs, p, est):
    e, r = exact_ns(costs, est), routed_ns(costs, p, est)
    return ("exact" if est["exp"] == 0 or e <= r else str(p)), e, r


def bin_for(table, sel):
    bins = table["bins"]
    for b in reversed(bins):
        if b["selectivity"] <= sel + 1e-12:
            return b
    return bins[0] if bins else None


def decide(policy, f):
    """The grid cell a policy selects for a filter (query-independent here)."""
    est = f["estimate"]
    be = {d["p"]: d for d in f["break_even"]}
    if policy == "baseline":
        d = be[16]
        return "exact" if d["plan"] == "exact_matches" else "16"
    (kind, args), = policy.items()
    if kind == "fixed_probe":
        d = be[args["nprobe"]]
        return "exact" if d["plan"] == "exact_matches" else str(args["nprobe"])
    if kind == "cost_model":
        return by_cost(args["costs"], args["nprobe"], est)[0]
    if kind == "calibrated":
        b = bin_for(args["table"], est["exp"] / max(est["available"], 1))
        if b is None or b["nprobe"] is None:
            return "exact"
        return by_cost(args["costs"], b["nprobe"], est)[0]
    raise ValueError(kind)


# ------------------------------------------------------------------ calibrate

def calibrate(work):
    data = Data(work, {"val"})
    out = {"note": "Selected on validation (rows 1000-2999), independent filters only. "
                   "Calibrated recall is a mean over the validation query distribution, not a per-query guarantee.",
           "policies": {}, "calibration": {}}
    for router in ROUTERS:
        costs, table_costs = fit_costs(data, router)
        recall = {}
        for s in data.selectivities:
            for p in P_GRID:
                vals = [r["recall"] for r in data.grid[(router, "ind", "val", skey(s), str(p))].values() if r["recall"] is not None]
                recall[(s, p)] = (float(np.mean(vals)), boot_mean_ci(vals)[0])
        pol = {"B0": "baseline",
               "EXACT": {"calibrated": {"table": {"target_recall": 1.0, "bins": [{"selectivity": 0.0, "nprobe": None}]}, "costs": costs}}}
        cal = {"costs": costs, "table_costs": table_costs, "stage_filter_ns_per_evaluation": stage_check_cost(data, router),
               "validation_recall": {f"{s}|{p}": v for (s, p), v in recall.items()}}
        for tau in TAUS:
            b1 = next((p for p in P_GRID if recall[(1.0, p)][0] >= tau), P_GRID[-1])
            pol[f"B1@{tau}"] = {"fixed_probe": {"nprobe": b1}}
            pol[f"C1@{tau}"] = {"cost_model": {"nprobe": b1, "costs": costs}}
            for name, idx in (("R1", 0), ("R2", 1)):
                # Bins are keyed by each validation filter's *estimated* selectivity, the
                # quantity the runtime policy sees (Amendment 1), not the nominal grid value.
                bins = []
                for s in data.selectivities:
                    est = data.filters[(router, "ind", skey(s))]["estimate"]
                    p = next((p for p in P_GRID if recall[(s, p)][idx] >= tau), None)
                    bins.append({"selectivity": est["exp"] / max(est["available"], 1), "nprobe": p})
                bins.sort(key=lambda b: b["selectivity"])
                pol[f"{name}@{tau}"] = {"calibrated": {"table": {"target_recall": tau, "bins": bins}, "costs": table_costs}}
            cal[f"B1@{tau}"] = b1
        cal["prediction_error"] = {"linear": prediction_error(data, router, costs),
                                   "table": prediction_error(data, router, table_costs)}
        out["policies"][router] = pol
        out["calibration"][router] = cal
    (Path(work) / "policies.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {Path(work) / 'policies.json'}")


# ------------------------------------------------------------------ report

def report(work, split="test"):
    """`split` is "test" for the study; the pilot validates the code path on "val"."""
    work = Path(work)
    spec = json.loads((work / "policies.json").read_text())
    data = Data(work, {split})
    native = defaultdict(lambda: defaultdict(list))
    mismatches = 0
    for r in jsonl(work / "policies/policies.jsonl"):
        native[(r["router"], r["family"], skey(r["s"]), r["policy"])][r["q"]].append(r["ns"])
        mismatches += 0 if r["matches_simulation"] else 1
    rows = []
    paired = []
    for router in ROUTERS:
        for family in FAMILIES:
            for s in data.selectivities:
                f = data.filters[(router, family, skey(s))]
                qs = data.queries(router, family, split, s)
                per_policy = {}
                for name, policy in spec["policies"][router].items():
                    cell = decide(policy, f)
                    g = data.grid[(router, family, split, skey(s), cell)]
                    rec = [g[q]["recall"] for q in qs]
                    valid = [x for x in rec if x is not None]
                    lat = native.get((router, family, skey(s), name), {})
                    med = [float(np.median(v)) for v in lat.values()]
                    cum = [g[q]["cum"] for q in qs if g[q]["cum"]]
                    fin = [g[q]["final"] for q in qs if g[q]["final"]]
                    tau = float(name.split("@")[1]) if "@" in name else None
                    row = {
                        "router": router, "family": family, "s": s, "policy": name, "cell": cell,
                        "matching": f["matching"], "estimate_exp": f["estimate"]["exp"],
                        "empty_cases": sum(1 for x in rec if x is None),
                        "mean_recall": float(np.mean(valid)) if valid else None,
                        "recall_ci": boot_mean_ci(valid),
                        "recall_p5": float(np.percentile(valid, 5)) if valid else None,
                        "share_below_tau": (float(np.mean([x < tau for x in valid])) if tau else
                                            {str(t): float(np.mean([x < t for x in valid])) for t in TAUS}),
                        "exact_fallback": 1.0 if cell == "exact" else 0.0,
                        "p50_ms": float(np.percentile(med, 50)) / 1e6 if med else None,
                        "p95_ms": float(np.percentile(med, 95)) / 1e6 if med else None,
                        "native_queries": len(med),
                        "cum_unique": float(np.mean([c["unique_candidates"] for c in cum])) if cum else 0.0,
                        "cum_filter_evals": float(np.mean([c["filter_evaluations"] for c in cum])) if cum else 0.0,
                        "cum_scored": float(np.mean([c["scored"] for c in cum])) if cum else float(f["matching"]),
                        "final_unique": float(np.mean([c["unique_candidates"] for c in fin])) if fin else 0.0,
                        "final_scored": float(np.mean([c["scored"] for c in fin])) if fin else float(f["matching"]),
                        "widenings": float(np.mean([g[q]["widenings"] for q in qs])),
                    }
                    rows.append(row)
                    per_policy[name] = (rec, lat)
                # Paired comparisons within router (query bootstrap).
                for a, b in [("R1@0.95", "B0"), ("R1@0.95", "B1@0.95"), ("R2@0.95", "B0"), ("R2@0.95", "B1@0.95"),
                             ("C1@0.95", "B0"), ("R1@0.9", "B0"), ("R1@0.9", "B1@0.9"), ("C1@0.9", "B0")]:
                    paired.append(paired_diff(router, family, s, a, b, per_policy[a], per_policy[b]))
            # Oracle bound on the timing subsample.
    # IVF versus MLP within each policy.
    for family in FAMILIES:
        for s in data.selectivities:
            for name in spec["policies"]["ivf"]:
                recs = {}
                for router in ROUTERS:
                    f = data.filters[(router, family, skey(s))]
                    cell = decide(spec["policies"][router][name], f)
                    g = data.grid[(router, family, split, skey(s), cell)]
                    lat = native.get((router, family, skey(s), name), {})
                    recs[router] = ([g[q]["recall"] for q in data.queries(router, family, split, s)], lat)
                paired.append(paired_diff("ivf-mlp", family, s, name + "[ivf]", name + "[mlp]", recs["ivf"], recs["mlp"]))
    oracle = oracle_rows(data, split)
    adoption = adoption_check(rows)
    result = {"simulation_mismatches": mismatches, "rows": rows, "paired": paired, "oracle": oracle,
              "adoption": adoption, "filters": list(data.filters.values())}
    (work / "analysis.json").write_text(json.dumps(result, indent=1))
    (work / "analysis.md").write_text(markdown(result))
    print(f"wrote {work / 'analysis.json'} and analysis.md; native/simulation mismatches: {mismatches}")


def paired_diff(router, family, s, a, b, ra, rb):
    rec_a, lat_a = ra
    rec_b, lat_b = rb
    d = [x - y for x, y in zip(rec_a, rec_b) if x is not None and y is not None]
    common = sorted(set(lat_a) & set(lat_b))
    dl = [float(np.median(lat_a[q]) - np.median(lat_b[q])) / 1e6 for q in common]
    return {"router": router, "family": family, "s": s, "a": a, "b": b,
            "recall_diff": float(np.mean(d)) if d else None, "recall_diff_ci": boot_mean_ci(d),
            "latency_diff_ms_median": float(np.median(dl)) if dl else None,
            "latency_diff_ms_mean_ci": boot_mean_ci(dl)}


def oracle_rows(data, split):
    """Per query: the cheapest measured cell reaching tau (exact always does). A bound only; uses gold."""
    rows = []
    for router in ROUTERS:
        for family in FAMILIES:
            for s in data.selectivities:
                for tau in TAUS:
                    lat, rec = [], []
                    t_exact = data.timing.get((router, family, split, skey(s), "exact"), {})
                    for q in t_exact:
                        best = (t_exact[q], 1.0)
                        for p in P_GRID:
                            cell = (router, family, split, skey(s), str(p))
                            r = data.grid[cell][q]["recall"]
                            t = data.timing.get(cell, {}).get(q)
                            if r is not None and r >= tau and t is not None and t < best[0]:
                                best = (t, r)
                        lat.append(best[0])
                        rec.append(best[1])
                    if lat:
                        rows.append({"router": router, "family": family, "s": s, "tau": tau,
                                     "p50_ms": float(np.percentile(lat, 50)) / 1e6,
                                     "mean_recall": float(np.mean(rec))})
    return rows


def adoption_check(rows):
    """Preregistered criterion, tau = 0.95, independent family, test."""
    out = {}
    for router in ROUTERS:
        by = {(r["policy"], r["s"]): r for r in rows if r["router"] == router and r["family"] == "ind"}
        sel = sorted({s for (_, s) in by}, reverse=True)
        for name in ["C1@0.95", "R1@0.95", "R2@0.95"]:
            ok_recall = all(by[(name, s)]["mean_recall"] is not None and by[(name, s)]["mean_recall"] >= 0.95 - 0.01 for s in sel)
            ok_exact = all(by[(name, s)]["p50_ms"] <= by[("EXACT", s)]["p50_ms"] for s in sel)
            # "No worse than B1 at matched recall": wherever both meet tau - 0.01, p50 <= B1's.
            ok_b1 = all(by[(name, s)]["p50_ms"] <= by[("B1@0.95", s)]["p50_ms"]
                        for s in sel
                        if by[("B1@0.95", s)]["mean_recall"] >= 0.94 and by[(name, s)]["mean_recall"] >= 0.94)
            out[f"{router}/{name}"] = {"recall": ok_recall, "not_slower_than_exact": ok_exact,
                                       "not_slower_than_b1_at_matched_recall": ok_b1,
                                       "adoptable": ok_recall and ok_exact and ok_b1}
    return out


def markdown(result):
    lines = [f"Native/simulation mismatches: {result['simulation_mismatches']}", ""]
    for family in FAMILIES:
        for router in ROUTERS:
            lines += [f"### {router.upper()}, {family} filters (test)", "",
                      "| s | policy | cell | mean R@10 [95% CI] | share < tau | p5 | p50 ms | p95 ms | filter evals | scored |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for r in result["rows"]:
                if r["router"] != router or r["family"] != family:
                    continue
                below = r["share_below_tau"]
                below = below if isinstance(below, float) else below["0.95"]  # EXACT, B0: tau 0.95
                ci = r["recall_ci"]
                lines.append(
                    f"| {r['s']:g} | {r['policy']} | {r['cell']} | {r['mean_recall']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | "
                    f"{below:.3f} | {r['recall_p5']:.2f} | {r['p50_ms'] or float('nan'):.3f} | {r['p95_ms'] or float('nan'):.3f} | "
                    f"{r['cum_filter_evals']:.0f} | {r['cum_scored']:.0f} |")
            lines.append("")
    lines += ["### Adoption criterion (tau = 0.95, independent, test)", "", "```", json.dumps(result["adoption"], indent=1), "```"]
    return "\n".join(lines)


if __name__ == "__main__":
    {"calibrate": calibrate, "report": report}[sys.argv[1]](*sys.argv[2:])
