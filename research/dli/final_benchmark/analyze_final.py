"""Final-benchmark analysis (PREREGISTRATION.md §4–§6, implementation item I7).

    analyze_final.py --manifest RUNS.json --out FILE [--resamples 10000]

RUNS.json lists every arm run (`cfg` for S-LMI, IVF and HNSW). `"mode": "pilot"` analyses the selection pilot
(validation panels, §8) instead of the primary:
    {"mode": "primary", "runs": [{"workload": "W1", "seed": 20261201, "arm": "DLI-best", "dir": ".../run",
                                  "plan": ".../plan.json"}, ...]}
Arms and the policy their rows carry: E1 → mdn_code, DLI-best → proportional_native, S-LMI and IVF →
static_nprobe, HNSW → hnsw_ef. CLI is added by amendment. The DLI-best run also carries `mdn_code` (the
secondary allocation replay) and `proportional` (the offline control).

Primary metric: W@0.90 under G1 (`recall30_g1`). Per batch, mean G1 Recall@30 over a panel's queries at every
grid point gives a curve. W is linearly interpolated between the two bracketing measured points and never
extrapolated:
* "not attained" if no grid point reaches 0.90;
* the first grid point's work, flagged `upper_bound`, if the first point already reaches 0.90.
Each W records its bracket and fraction.

R(X, Y) = mean_t W_X(t) / mean_t W_Y(t) over the batches where both attain (W1/W2 batches 1–20, W3 batches
0–19). Paired query bootstrap, stratified by batch: one resample of query indices per batch, shared by both
arms. 10,000 resamples, seed 20261204.

Family A (hypothesis): DLI-best vs E1 is supported iff R <= 0.90 and the 95% CI upper bound is < 1.
Family B (no direction): better if the CI is entirely < 0.95, equivalent if it lies in [0.95, 1.05], worse if it
is entirely > 1.05, inconclusive otherwise. Applied to work, and to latency.
* Latency per batch: the median timed query latency at the measured grid point nearest above 0.90 (the first
  grid point whose batch mean G1 recall is >= 0.90, chosen on the full panel and held fixed in resamples).
* Latency R = mean_t L_X(t) / mean_t L_Y(t).
* Amendment A2 (`bound_adjust`): a decision that an `upper_bound` W (or the latency at that grid point) could
  overturn becomes "not decidable (upper bound)".

A ratio whose attainment share is below 80% is descriptive.

Replication: a verdict is replicated iff the second seed gives the same category.

Gates per run: ledger (live SHA per batch), completeness, monotone work, known answer (G1 ≥ 0.999 per
batch-panel at the arm's exact point), native = control (DLI-best), timed/work agreement (G1 recall equal on
every timed row). The oracle gate is `oracle_check.py`, run in the job.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGET = 0.90
K = 30
SEED = 20261204
RESAMPLES = 10_000
MIN_ATTAIN = 0.80
A_BAR = 0.90
MARGIN = (0.95, 1.05)
KNOWN = 0.999
ARM_POLICY = {"E1": "mdn_code", "DLI-best": "proportional_native", "S-LMI": "static_nprobe", "IVF": "static_nprobe",
              "HNSW": "hnsw_ef"}
PANELS = {"W1": ["test"], "W2": ["test"], "W3": ["mq", "ids", "ood"], "W4": ["test"], "W5": ["test"]}  # W4/W5: A6
PILOT_PANELS = {"W1": ["val"], "W2": ["val"], "W3": ["mq_val", "ids_val", "ood_val"], "W4": ["val"], "W5": ["val"]}
FAMILY_B = ["S-LMI", "IVF", "HNSW"]
FAMILY_A_SIZE = 5  # panel decisions
FAMILY_B_SIZE = len(FAMILY_B) * 5 * 2  # baselines x panels x (work, latency)


# ---------- loading ----------

def load_rows(run_dir, name, policy, panel):
    """{batch: (budgets[], recall[q, b], work[q, b])} from `name` rows of one policy and panel."""
    out = {}
    for f in sorted(Path(run_dir).glob(f"batch_*/{name}")):
        t = int(f.parent.name.split("_")[1])
        cells = defaultdict(dict)
        for line in open(f):
            r = json.loads(line)
            if r["policy"] == policy and r["set"] == panel:
                cells[r["budget"]][r["query"]] = r
        if not cells:
            continue
        budgets = sorted(cells)
        queries = sorted(cells[budgets[0]])
        for b in budgets:
            assert sorted(cells[b]) == queries, (f, b, "incomplete query set")
        g1 = [[cells[b][q]["recall30_g1"] for b in budgets] for q in queries]
        assert all(v is not None for row in g1 for v in row), (f, "recall30_g1 missing: not a G1 plan")
        recall = np.array(g1, dtype=float) / K
        work = np.array([[cells[b][q].get("scored_total", np.nan) for b in budgets] for q in queries], dtype=float)
        lat = np.array([[cells[b][q].get("elapsed_ns", np.nan) for b in budgets] for q in queries], dtype=float)
        out[t] = {"budgets": budgets, "recall": recall, "work": work, "latency": lat}
    return out


# ---------- W@0.90 ----------

def interp(recall_curve, work_curve):
    """W at TARGET from one batch's mean curves (indexed by grid point). Returns (W | None, info)."""
    r, w = np.asarray(recall_curve), np.asarray(work_curve)
    if r[0] >= TARGET:
        return float(w[0]), {"kind": "upper_bound", "bracket": [0, 0]}
    for i in range(1, len(r)):
        if r[i] >= TARGET:
            lo, hi = r[i - 1], r[i]
            if hi <= lo:
                raise ValueError(f"flat or non-monotone bracket {lo} -> {hi}")
            frac = (TARGET - lo) / (hi - lo)
            return float(w[i - 1] + frac * (w[i] - w[i - 1])), {"kind": "interpolated", "bracket": [i - 1, i],
                                                                 "fraction": float(frac)}
    return None, {"kind": "not_attained"}


def batch_w(data, t, idx=None):
    d = data[t]
    rec = d["recall"] if idx is None else d["recall"][idx]
    wrk = d["work"] if idx is None else d["work"][idx]
    return interp(rec.mean(axis=0), wrk.mean(axis=0))


def eligible(workload, batches):
    return [t for t in batches if workload == "W3" or t >= 1]


def ratio(a, b, workload, resamples, rng, metric="work", la=None, lb=None, family_size=1):
    """R(X, Y) with a paired, batch-stratified query bootstrap. a, b: work data; la, lb: timed data (latency).
    Also reports the Bonferroni-adjusted interval for a family of `family_size` decisions."""
    batches = eligible(workload, sorted(set(a) & set(b)))
    per, pa, pb, used = {}, [], [], []
    for t in batches:
        wa, ia = batch_w(a, t)
        wb, ib = batch_w(b, t)
        if metric == "latency":
            xa, xb = latency_at(a, la, t), latency_at(b, lb, t)
        else:
            xa, xb = wa, wb
        per[t] = {"A": xa, "B": xb, "A_info": ia, "B_info": ib}
        if xa is None or xb is None:
            continue
        used.append(t)
        pa.append(xa)
        pb.append(xb)
    out = {"batches": batches, "attained_both": len(used), "attained_share": len(used) / len(batches) if batches else 0.0,
           "per_batch": per}
    if not used:
        out["status"] = "not attained"
        return out
    point = float(np.mean(pa) / np.mean(pb))
    boots = np.empty(resamples)
    for k in range(resamples):
        sa, sb = [], []
        for t in used:
            nq = a[t]["recall"].shape[0]
            idx = rng.integers(0, nq, nq)
            if metric == "latency":
                xa, xb = latency_at(a, la, t, idx), latency_at(b, lb, t, idx)
            else:
                xa, xb = batch_w(a, t, idx)[0], batch_w(b, t, idx)[0]
            if xa is None or xb is None:
                sa, sb = None, None
                break
            sa.append(xa)
            sb.append(xb)
        boots[k] = np.nan if sa is None else np.mean(sa) / np.mean(sb)
    ok = boots[~np.isnan(boots)]
    ci = lambda lvl: [float(np.quantile(ok, (1 - lvl) / 2)), float(np.quantile(ok, 1 - (1 - lvl) / 2))]
    lvl = 1 - 0.05 / family_size
    out.update({"R": point, "ci95": ci(0.95), "ci_bonferroni": ci(lvl), "bonferroni_level": lvl,
                "resamples_not_reached": int(np.isnan(boots).sum()),
                "status": "ok" if out["attained_share"] >= MIN_ATTAIN else "descriptive (attainment below 80%)"})
    return out


def latency_at(work, timed, t, idx=None):
    """Median timed latency at the first grid point whose full-panel mean G1 recall >= TARGET (held fixed)."""
    if t not in timed:
        return None
    mean_r = work[t]["recall"].mean(axis=0)
    hit = np.flatnonzero(mean_r >= TARGET)
    if not hit.size:
        return None
    budget = work[t]["budgets"][hit[0]]
    j = timed[t]["budgets"].index(budget)
    lat = timed[t]["latency"][:, j] if idx is None else timed[t]["latency"][idx, j]
    return float(np.median(lat))


# ---------- decisions ----------

def family_a(r):
    if "R" not in r:
        return "not decidable"
    if r["status"] != "ok":
        return "descriptive"
    return "supported" if r["R"] <= A_BAR and r["ci95"][1] < 1 else "not supported"


def family_b(r):
    if "R" not in r:
        return "not decidable"
    if r["status"] != "ok":
        return "descriptive"
    lo, hi = r["ci95"]
    if hi < MARGIN[0]:
        return "better"
    if lo > MARGIN[1]:
        return "worse"
    if lo >= MARGIN[0] and hi <= MARGIN[1]:
        return "equivalent"
    return "inconclusive"


def bound_adjust(r):
    """Amendment A2: a W flagged `upper_bound` overstates that arm's true W (the cheapest grid point already reaches
    0.90). With R = W_X / W_Y, an upper bound on the baseline Y means the true R is at least the reported R, and an
    upper bound on X means it is at most the reported R. A decision survives only if moving R in the possible
    direction cannot change it; otherwise it becomes "not decidable (upper bound)". The kinds come from the full
    panel of each batch used (the same batches the point estimate uses), for both work and latency, since the latency
    grid point is the first one reaching 0.90."""
    used = [t for t, v in r.get("per_batch", {}).items() if v["A"] is not None and v["B"] is not None]
    ub = {s: [t for t in used if r["per_batch"][t][f"{s}_info"]["kind"] == "upper_bound"] for s in ("A", "B")}
    r["upper_bound_batches"] = ub
    d = r["decision"]
    if not (ub["A"] or ub["B"]) or d in ("not decidable", "descriptive"):
        return d
    survives_if_r_grows = {"worse", "not supported"}   # baseline overstated: true R >= reported R
    survives_if_r_shrinks = {"better", "supported"}    # X overstated: true R <= reported R
    if ub["A"] and ub["B"]:
        ok = False
    elif ub["B"]:
        ok = d in survives_if_r_grows
    else:
        ok = d in survives_if_r_shrinks
    if not ok:
        r["decision_before_bound_rule"] = d
        return "not decidable (upper bound)"
    return d


# ---------- gates ----------

def widening_restart(arm, x, i):
    """Amendment A3: whether a drop in batch-mean work from knob i to knob i+1 is Qdrant's static-LMI widening
    restart. When the probed buckets hold fewer than `top` points, `widen_until_filled`
    (lib/segment/src/index/lmi_index/mod.rs) doubles nprobe (capped at the bucket count) and rescores the widened
    set from scratch. So for a query that widens at nprobe n: work(n) = s(n) + work(min(2n, B)), where
    s(n) < K is the underfilled first pass. The drop is accepted only for S-LMI and IVF, only on a step
    n -> min(2n, B), and only if every query whose work falls on that step falls by fewer than K vectors."""
    if arm not in ("S-LMI", "IVF"):
        return False
    b = x["budgets"]
    if b[i + 1] != min(2 * b[i], b[-1]):
        return False
    drop = x["work"][:, i] - x["work"][:, i + 1]
    return bool(np.all(drop[drop > 0] < K))


def gates(run, panels):
    d = Path(run["dir"])
    g = {}
    plan = json.loads(Path(run["plan"]).read_text())
    log = [json.loads(l) for l in open(d / "batches.jsonl")]
    g["ledger"] = len(log) == len(plan["batches"]) and all(
        x["live_sha256"] == m["live_sha256"] and x["live"] == m["live_count"] for x, m in zip(log, plan["batches"]))
    policy = ARM_POLICY[run["arm"]]
    known, mono, complete = [], True, True
    non_mono, restarts = [], []
    for p in panels:
        data = load_rows(d, "rows.jsonl", policy, p)
        complete &= len(data) == len(plan["batches"])
        for t, x in data.items():
            mean_w = np.nanmean(x["work"], axis=0)
            for i in np.flatnonzero(np.diff(mean_w) < -1e-9):
                point = {"panel": p, "batch": t, "knobs": [x["budgets"][i], x["budgets"][i + 1]],
                         "mean_work": [float(mean_w[i]), float(mean_w[i + 1])]}
                if widening_restart(run["arm"], x, i):
                    restarts.append(point)
                else:
                    mono = False
                    non_mono.append(point)
            if run["arm"] != "HNSW":
                known.append(float(x["recall"][:, -1].mean()))
        if run["arm"] == "HNSW":
            exact = load_rows(d, "rows.jsonl", "hnsw_exact", p)
            known += [float(x["recall"][:, 0].mean()) for x in exact.values()]
    g["completeness"] = bool(complete)
    g["monotone_work"] = bool(mono)
    g["non_monotone_points"] = non_mono
    g["widening_restart_points"] = restarts
    g["known_answer_min"] = min(known) if known else None
    g["known_answer"] = bool(known) and min(known) >= KNOWN
    if run["arm"] == "DLI-best":
        mism = 0
        for p in panels:
            n = load_rows(d, "rows.jsonl", "proportional_native", p)
            c = load_rows(d, "rows.jsonl", "proportional", p)
            mism += sum(int(np.sum(n[t]["work"] != c[t]["work"])) for t in n if t in c)
        g["native_equals_control"] = mism == 0
        g["native_control_work_mismatches"] = mism
    timed_bad = 0
    for p in panels:
        w = load_rows(d, "rows.jsonl", policy, p)
        tm = load_rows(d, "timed_rows.jsonl", policy, p)
        for t in tm:
            if t in w and tm[t]["budgets"] == w[t]["budgets"]:
                timed_bad += int(np.sum(tm[t]["recall"] != w[t]["recall"]))
            else:
                timed_bad += 1
    g["timed_matches_work"] = timed_bad == 0
    g["timed_mismatches"] = timed_bad
    g["pass"] = all(v for k, v in g.items() if isinstance(v, bool))
    return g


# ---------- driver ----------

def selection(manifest):
    """Selection pilot (PREREGISTRATION.md §8): for S-LMI, IVF and HNSW, the configuration (`cfg`: bucket count or m)
    with the lowest mean validation W@0.90 on W1 over batches 1–20, taken over the batches where every candidate of
    that arm attains. Exact ties are broken by the lower mean p50 latency. Also reports, for every arm and workload,
    whether each validation panel brackets 0.90 in every batch."""
    out = {"choice": {}, "bracketing": {}}
    for arm in FAMILY_B:
        cands = [r for r in manifest["runs"] if r["arm"] == arm and r["workload"] == "W1"]
        if not cands:
            continue
        data = {r["cfg"]: load_rows(r["dir"], "rows.jsonl", ARM_POLICY[arm], "val") for r in cands}
        timed = {r["cfg"]: load_rows(r["dir"], "timed_rows.jsonl", ARM_POLICY[arm], "val") for r in cands}
        common = sorted(set.intersection(*[set(d) for d in data.values()]))
        ws = {c: {t: batch_w(d, t)[0] for t in eligible("W1", common)} for c, d in data.items()}
        both = [t for t in eligible("W1", common) if all(ws[c][t] is not None for c in ws)]
        mean_w = {c: float(np.mean([ws[c][t] for t in both])) if both else None for c in ws}
        lat = {c: float(np.mean([latency_at(data[c], timed[c], t) or np.nan for t in both])) if both else None for c in ws}
        ranked = sorted((c for c in ws if mean_w[c] is not None), key=lambda c: (mean_w[c], lat[c]))
        out["choice"][arm] = {"cfg": ranked[0] if ranked else None, "mean_W": mean_w, "mean_p50_ns": lat,
                              "batches_used": both}
    for r in manifest["runs"]:
        for p in PILOT_PANELS[r["workload"]]:
            d = load_rows(r["dir"], "rows.jsonl", ARM_POLICY[r["arm"]], p)
            kinds = {t: batch_w(d, t)[1]["kind"] for t in d}
            out["bracketing"][f'{r["workload"]}/{r["arm"]}/{r.get("cfg", "-")}/{p}'] = {
                "all_interpolated": all(k == "interpolated" for k in kinds.values()),
                "upper_bound_batches": [t for t, k in kinds.items() if k == "upper_bound"],
                "not_attained_batches": [t for t, k in kinds.items() if k == "not_attained"]}
    return out


def analyze(manifest, resamples=RESAMPLES):
    panels_by_workload = PILOT_PANELS if manifest.get("mode") == "pilot" else PANELS
    runs = manifest["runs"]
    by = {(r["workload"], r["seed"], r["arm"]): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    result = {"gates": {}, "family_A": {}, "family_B": {}, "secondary_allocation": {}, "replication": {}}
    for r in runs:
        result["gates"][f'{r["workload"]}/{r["seed"]}/{r["arm"]}{"/" + str(r["cfg"]) if "cfg" in r else ""}'] = gates(
            r, panels_by_workload[r["workload"]])
    for seed in seeds:
        rng = np.random.default_rng(SEED)
        for wl, panels in panels_by_workload.items():
            for p in panels:
                key = f"{wl}/{seed}/{p}"
                get = lambda arm, pol=None, name="rows.jsonl": load_rows(by[(wl, seed, arm)]["dir"], name,
                                                                          pol or ARM_POLICY[arm], p)
                if (wl, seed, "DLI-best") not in by:
                    continue
                best, best_t = get("DLI-best"), get("DLI-best", name="timed_rows.jsonl")
                if (wl, seed, "E1") in by:
                    r = ratio(best, get("E1"), wl, resamples, rng, family_size=FAMILY_A_SIZE)
                    r["decision"] = family_a(r)
                    r["decision"] = bound_adjust(r)
                    result["family_A"][key] = r
                for arm in FAMILY_B:
                    if (wl, seed, arm) not in by:
                        continue
                    other, other_t = get(arm), get(arm, name="timed_rows.jsonl")
                    rw = ratio(best, other, wl, resamples, rng, family_size=FAMILY_B_SIZE)
                    rl = ratio(best, other, wl, resamples, rng, "latency", best_t, other_t, family_size=FAMILY_B_SIZE)
                    rw["decision"], rl["decision"] = family_b(rw), family_b(rl)
                    rw["decision"], rl["decision"] = bound_adjust(rw), bound_adjust(rl)
                    result["family_B"][f"{key}/{arm}"] = {"work": rw, "latency": rl}
                mdn = get("DLI-best", "mdn_code")
                if mdn:
                    result["secondary_allocation"][key] = ratio(best, mdn, wl, resamples, rng)
    if len(seeds) >= 2:
        s1, s2 = seeds[0], seeds[1]
        for fam in ("family_A", "family_B"):
            for k, v in result[fam].items():
                if f"/{s1}/" not in k:
                    continue
                twin = result[fam].get(k.replace(f"/{s1}/", f"/{s2}/"))
                if twin is None:
                    continue
                if fam == "family_A":
                    result["replication"][k] = {"seed1": v["decision"], "seed2": twin["decision"],
                                                "replicated": v["decision"] == twin["decision"]}
                else:
                    for m in ("work", "latency"):
                        result["replication"][f"{k}/{m}"] = {"seed1": v[m]["decision"], "seed2": twin[m]["decision"],
                                                             "replicated": v[m]["decision"] == twin[m]["decision"]}
    return result


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--resamples", type=int, default=RESAMPLES)
    args = a.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("mode") == "pilot":
        res = {"selection": selection(manifest),
               "gates": {f'{r["workload"]}/{r["arm"]}/{r.get("cfg", "-")}': gates(r, PILOT_PANELS[r["workload"]])
                         for r in manifest["runs"]}}
        args.out.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({"choice": {a: v["cfg"] for a, v in res["selection"]["choice"].items()},
                          "gates_pass": {k: v["pass"] for k, v in res["gates"].items()}}, indent=1))
        return
    res = analyze(manifest, args.resamples)
    args.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"gates_pass": {k: v["pass"] for k, v in res["gates"].items()},
                      "family_A": {k: (v.get("R"), v.get("ci95"), v["decision"]) for k, v in res["family_A"].items()},
                      "family_B": {k: (v["work"]["decision"], v["latency"]["decision"]) for k, v in res["family_B"].items()},
                      "replication": res["replication"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
