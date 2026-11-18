"""Analyze lab.py traces (Tier 1/2). Preregistered rules in PREREGISTRATION_TIER1.md.

For every run directory and (partition, router) trace:
  * mean curves over the fine budget grid (realized work vs Recall@10/30) per query set;
  * W_target on the evaluation set: mean work where the mean recall curve reaches the target,
    linear between adjacent fine-grid budgets (never extrapolated; 'not reached' otherwise);
  * validation-locked operating point (smallest grid budget with val mean >= target);
  * total work = scored vectors + routing MACs / 768 (scored-vector equivalents);
  * paired bootstrap over evaluation queries (same resample for every configuration) for
    W_target ratios between named pairs.
"""
import argparse, json, math
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", type=Path)
ap.add_argument("--eval", default="test"); ap.add_argument("--val", default="val")
ap.add_argument("--resamples", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261118)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
TARGETS = {"r10": 0.95, "r30": 0.90}
DIV = {"r10": 10, "r30": 30}

configs = {}
for meta_path in sorted(a.runs.glob("*/meta.json")):
    run = meta_path.parent
    if run.name.startswith("smoke"):
        continue
    meta = json.loads(meta_path.read_text())
    for tr in sorted(run.glob(f"trace-*-{a.eval}.npz")):
        _, part, router, _ = tr.stem.split("-")
        key = f"{run.name}|{part}|{router}"
        ev = dict(np.load(tr)); va = dict(np.load(run / f"trace-{part}-{router}-{a.val}.npz"))
        macs = meta["mlp_macs"] if router == "mlp" else meta["centroid_macs"]
        configs[key] = {"meta": meta, "part": part, "router": router, "ev": ev, "va": va,
                        "route_eq": macs / 768.0}

rng = np.random.default_rng(a.seed)
nq = next(iter(configs.values()))["ev"]["b_work"].shape[0]
IDX = rng.integers(0, nq, size=(a.resamples, nq))


def crossing(work, rec, t):
    for j in range(len(rec)):
        if rec[j] >= t:
            if j == 0:
                return float(work[0])
            return float(work[j - 1] + (t - rec[j - 1]) * (work[j] - work[j - 1]) / (rec[j] - rec[j - 1]))
    return None


def wt(tr, metric, idx=None, extra=0.0):
    w = tr["b_work"] if idx is None else tr["b_work"][idx]
    r = tr[f"b_{metric}"] if idx is None else tr[f"b_{metric}"][idx]
    x = crossing(w.mean(0) + extra, r.mean(0) / DIV[metric], TARGETS[metric])
    return x


res = {}
boot = {}
for key, c in configs.items():
    e, v = c["ev"], c["va"]
    out = {"partition": c["part"], "router": c["router"], "B": c["meta"]["B"], "teacher": c["meta"]["teacher"],
           "sample": c["meta"]["sample"], "iters": c["meta"]["iters"], "hidden": c["meta"]["hidden"],
           "epochs": c["meta"]["epochs"], "route_eq_vectors": c["route_eq"],
           "stats": c["meta"]["stats"][c["part"]], "timing": c["meta"]["timing"],
           "mlp_bytes": c["meta"].get("mlp_bytes"), "centroid_bytes": c["meta"]["centroid_bytes"],
           "mlp_train_acc_final": (c["meta"].get("train_history") or [[None, None, None]])[-1][2],
           "mlp_teacher_agreement_sample": c["meta"].get("mlp_teacher_agreement_sample"),
           "partition_agreement_all": c["meta"].get("partition_agreement_all")}
    out["curve_eval"] = [{"budget": int(b), "work": float(w), "r10": float(r1), "r30": float(r3)} for b, w, r1, r3 in
                         zip(e["budgets"], e["b_work"].mean(0), e["b_r10"].mean(0) / 10, e["b_r30"].mean(0) / 30)]
    out["nprobe_eval"] = [{"nprobe": int(p), "work": float(w), "r10": float(r1), "r30": float(r3)} for p, w, r1, r3 in
                          zip(e["probes"], e["p_work"].mean(0), e["p_r10"].mean(0) / 10, e["p_r30"].mean(0) / 30)]
    for m in TARGETS:
        W = wt(e, m); Wt = wt(e, m, extra=c["route_eq"])
        bs = [wt(e, m, idx) for idx in IDX]
        boot[(key, m)] = np.array([np.nan if x is None else x for x in bs])
        # nprobe stopping rule (same order, different stopping) for comparison
        Wp = crossing(e["p_work"].mean(0), e[f"p_{m}"].mean(0) / DIV[m], TARGETS[m])
        vm = v[f"b_{m}"].mean(0) / DIV[m]
        j = next((j for j in range(len(vm)) if vm[j] >= TARGETS[m]), None)
        locked = None if j is None else {"budget": int(e["budgets"][j]), "val": float(vm[j]),
                                         "eval": float(e[f"b_{m}"][:, j].mean() / DIV[m]),
                                         "eval_work": float(e["b_work"][:, j].mean())}
        out[m] = {"W": W, "W_total": Wt, "W_ci": [float(np.nanpercentile(boot[(key, m)], 2.5)),
                                                  float(np.nanpercentile(boot[(key, m)], 97.5))] if W else None,
                  "W_nprobe_rule": Wp, "val_locked": locked,
                  "oracle_cost_mean": float(e[f"oracle_cost_top{DIV[m]}"].mean()),
                  "rank_cost_mean": float(e[f"rank_cost_top{DIV[m]}"].mean()),
                  "rank_cost_median": float(np.median(e[f"rank_cost_top{DIV[m]}"]))}
    res[key] = out


def ratio(k1, k2, m):
    """W(k1)/W(k2) with paired bootstrap CI; <1 means k1 needs less work."""
    if k1 not in res or k2 not in res or res[k1][m]["W"] is None or res[k2][m]["W"] is None:
        return None
    r = boot[(k1, m)] / boot[(k2, m)]
    return {"a": k1, "b": k2, "metric": m, "ratio": res[k1][m]["W"] / res[k2][m]["W"],
            "ci": [float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))]}


comparisons = []
runs = sorted({k.split("|")[0] for k in res})
for run in runs:
    for m in TARGETS:
        # D1: router at fixed partition; D2: partition at fixed router
        for part in ("mlp", "centroid"):
            comparisons.append(dict(rule="D1 router (centroid/mlp) @partition=" + part, **(ratio(f"{run}|{part}|centroid", f"{run}|{part}|mlp", m) or {})))
        for router in ("mlp", "centroid"):
            comparisons.append(dict(rule="D2 partition (centroid/mlp) @router=" + router, **(ratio(f"{run}|centroid|{router}", f"{run}|mlp|{router}", m) or {})))
        # IVF vs native LMI
        comparisons.append(dict(rule="IVF (centroid|centroid) / LMI (mlp|mlp)", **(ratio(f"{run}|centroid|centroid", f"{run}|mlp|mlp", m) or {})))
for B in (64, 256, 1024, 4096):
    for m in TARGETS:
        for combo in ("mlp|mlp", "centroid|centroid"):
            comparisons.append(dict(rule=f"D5 teacher converged/native B={B} {combo}", **(ratio(f"B{B}-converged|{combo}", f"B{B}-native|{combo}", m) or {})))
for v in ("H128", "linear", "E10"):
    for m in TARGETS:
        comparisons.append(dict(rule=f"D4 {v}/H512 B=1024", **(ratio(f"B1024-native-{v}|mlp|mlp", "B1024-native|mlp|mlp", m) or {})))
comparisons = [c for c in comparisons if "ratio" in c]
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({"eval": a.eval, "val": a.val, "queries": int(nq), "resamples": a.resamples,
                             "targets": TARGETS, "configs": res, "comparisons": comparisons}, indent=1))
# console summary
print(f"{'config':45s} {'W r10@.95':>10s} {'W r30@.90':>10s} {'+route':>8s} {'oracle30':>9s} {'rank30':>9s} cv")
for k, r in sorted(res.items(), key=lambda kv: (kv[1]["B"], kv[0])):
    f = lambda x: "n/r" if x is None else f"{x:,.0f}"
    print(f"{k:45s} {f(r['r10']['W']):>10s} {f(r['r30']['W']):>10s} {r['route_eq_vectors']:8.0f} "
          f"{r['r30']['oracle_cost_mean']:9.0f} {r['r30']['rank_cost_mean']:9.0f} {r['stats']['cv']:.2f}")
for c in comparisons:
    print(f"{c['rule']:52s} {c['metric']} {c['a'].split('|')[0]:22s} ratio={c['ratio']:.3f} [{c['ci'][0]:.3f},{c['ci'][1]:.3f}]")
