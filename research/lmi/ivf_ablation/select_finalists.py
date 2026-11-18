"""Mechanical Tier 2 finalist selection (Amendment 01): validation-set analysis only.

Input: analysis JSON from `analyze_tier1.py --eval val --val val`.
F2 = centroid|centroid, F3 = mlp|mlp with the smallest W_total at R@10 >= 0.95 among
configurations satisfying both preregistered D3 conditions: routing vector-equivalents <= 10%
of W_total, and construction FLOPs <= 10x the native baseline of the same B (analytic FLOPs).
(First run of this script omitted the routing condition; corrected 2026-11-18 before Tier 2.) F4 = a D4 simplification passing on validation.
"""
import json, sys

D = 768
a = json.load(open(sys.argv[1]))
cfg = a["configs"]


def flops(c):
    """Analytic construction FLOPs (teacher + router training + N-point assignment)."""
    B, S, it, H, E = c["B"], c["sample"], c["iters"], c["hidden"], c["epochs"]
    N = 1_000_000
    teacher = 2 * S * B * D * (it + 1)
    if c["router"] == "mlp" or c["partition"] == "mlp":
        macs = D * H + H * B if H else D * B
        train = 6 * S * E * macs
        assign = 2 * N * macs
    else:
        train = 0
        assign = 2 * N * B * D
    return teacher + train + assign


def base_flops(B):
    k = f"B{B}-native|mlp|mlp"
    return flops(cfg[k]) if k in cfg else None


def best(combo):
    rows = []
    for k, c in cfg.items():
        run, part, router = k.split("|")
        if f"{part}|{router}" != combo or c["r10"]["W_total"] is None:
            continue
        bf = base_flops(c["B"])
        if bf is not None and flops(c) > 10 * bf:
            continue
        # D3 (as preregistered): routing <= 10% of total work.
        if c["route_eq_vectors"] > 0.10 * c["r10"]["W_total"]:
            continue
        rows.append((c["r10"]["W_total"], k, c["r30"]["W_total"], flops(c)))
    rows.sort()
    return rows


out = {"rule": "Amendment 01: validation W_total at R@10>=0.95 (scored + routing vector-equivalents)"}
for name, combo in (("F2_ivf", "centroid|centroid"), ("F3_lmi", "mlp|mlp")):
    r = best(combo)
    out[name] = {"choice": r[0][1], "W_total_r10": r[0][0], "W_total_r30": r[0][2], "flops": r[0][3],
                 "ranking": [{"config": k, "W_total_r10": w, "W_total_r30": w30, "flops": f} for w, k, w30, f in r]}
d4 = []
for c in a["comparisons"]:
    if c["rule"].startswith("D4") and c["metric"] == "r10":
        passes = c["ratio"] <= 1.05 and c["ci"][1] <= 1.10
        d4.append({"variant": c["a"], "ratio": c["ratio"], "ci": c["ci"], "passes": passes})
out["D4_simplifications"] = d4
out["F4"] = next((x["variant"] for x in sorted(d4, key=lambda x: x["ratio"]) if x["passes"]), None)
print(json.dumps(out, indent=1))
