"""Phase 3 analysis (PHASE3_PREREGISTRATION.md §5, §10.3–10.4). Committed before any Phase 3 data.

Input: one directory per (scenario, arm) run of `dli_phase3_workload`, each holding
`batches.jsonl`, `summary.json` and `batch_XX/rows.jsonl`, plus the scenario's `plan.json`.

    analyze_phase3.py --run NAME=SCENARIO:ARM:DIR ... --plans SCENARIO=PLANDIR ... --out FILE
    analyze_phase3.py --precision --run ... --plans ...   (preflight noise check, §5)

Arms are labelled as in the preregistration: E1, E1G, E1R, E1C, E1C50 (and policies within a
run, e.g. E1L = E1 run with policy `calibrated`). A contrast names "ARM[:policy]".

Estimands:
* Part A: mean over batches of mean R@30 at B* = 0.5% of N (budget on the grid), test set;
  paired query bootstrap within each batch, 10,000 resamples, seed 20261121.
* Part B: ratio of the batch-mean work at mean R@30 = 0.90 (linear interpolation between
  bracketing budgets, never extrapolating), over batches where both arms reach the target;
  seed 20261122.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESAMPLES = 10_000
TARGET = 0.90


def load_run(directory, policy, set_name):
    """{batch: {budget: (recall30[q], work[q], dead[q], selected[q])}} for one policy and set."""
    out = defaultdict(dict)
    for bdir in sorted(Path(directory).glob("batch_*")):
        rows = bdir / "rows.jsonl"
        if not rows.exists():
            continue
        b = int(bdir.name.split("_")[1])
        acc = defaultdict(list)
        for line in open(rows):
            r = json.loads(line)
            if r["set"] != set_name or r["policy"] != policy:
                continue
            acc[r["budget"]].append((r["query"], r["recall30"] / 30, r["scored_total"],
                                     r.get("dead_offsets", 0), r["unique_offsets"]))
        for budget, v in acc.items():
            v.sort()
            a = np.array([x[1:] for x in v], dtype=float)
            out[b][budget] = (a[:, 0], a[:, 1], a[:, 2], a[:, 3])
    return dict(out)


def resample_indices(rng, n, resamples=RESAMPLES):
    return rng.integers(0, n, size=(resamples, n))


def mean_recall_diff(a, b, budget, rng):
    """Δ = mean_b mean_q (a − b) at `budget`, with a batch-stratified paired bootstrap."""
    batches = sorted(set(a) & set(b))
    point, boots = [], np.zeros(RESAMPLES)
    for bt in batches:
        d = a[bt][budget][0] - b[bt][budget][0]
        point.append(d.mean())
        boots += d[resample_indices(rng, len(d))].mean(axis=1)
    boots /= len(batches)
    return float(np.mean(point)), boots, batches


def interp_work(recall, work, target=TARGET):
    """Work at mean recall = target, by budget order; NaN if the curve starts above or never reaches."""
    recall, work = np.atleast_2d(recall), np.atleast_2d(work)
    reached = recall >= target
    k = reached.argmax(axis=1)
    ok = reached.any(axis=1) & (k > 0)
    out = np.full(recall.shape[0], np.nan)
    i = np.flatnonzero(ok)
    r0, r1 = recall[i, k[i] - 1], recall[i, k[i]]
    w0, w1 = work[i, k[i] - 1], work[i, k[i]]
    out[i] = w0 + (target - r0) / (r1 - r0) * (w1 - w0)
    return out


def work_ratio(a, b, rng):
    """Ratio of batch-mean W@0.90 (a / b) over batches where both reach, with bootstrap."""
    batches = sorted(set(a) & set(b))
    budgets = sorted(set.intersection(*[set(a[t]) & set(b[t]) for t in batches]))
    wa_pt, wb_pt, wa_bs, wb_bs, used, excluded = [], [], [], [], [], []
    for t in batches:
        ra = np.stack([a[t][x][0] for x in budgets]); wa = np.stack([a[t][x][1] for x in budgets])
        rb = np.stack([b[t][x][0] for x in budgets]); wb = np.stack([b[t][x][1] for x in budgets])
        pa = interp_work(ra.mean(axis=1), wa.mean(axis=1))[0]
        pb = interp_work(rb.mean(axis=1), wb.mean(axis=1))[0]
        if np.isnan(pa) or np.isnan(pb):
            excluded.append(t)
            continue
        idx = resample_indices(rng, ra.shape[1])
        ba = interp_work(ra[:, idx].mean(axis=2).T, wa[:, idx].mean(axis=2).T)
        bb = interp_work(rb[:, idx].mean(axis=2).T, wb[:, idx].mean(axis=2).T)
        wa_pt.append(pa); wb_pt.append(pb); wa_bs.append(ba); wb_bs.append(bb); used.append(t)
    if not used:
        return None
    point = np.mean(wa_pt) / np.mean(wb_pt)
    num, den = np.nanmean(np.stack(wa_bs), axis=0), np.nanmean(np.stack(wb_bs), axis=0)
    boots = num / den
    return {"point": float(point), "boots": boots, "batches_used": used, "batches_excluded": excluded,
            "resamples_not_reached": int(np.isnan(np.stack(wa_bs + wb_bs)).any(axis=0).sum()),
            "per_batch_ratio": {t: float(x / y) for t, x, y in zip(used, wa_pt, wb_pt)}}


def ci(boots, level):
    boots = boots[~np.isnan(boots)]
    lo = (1 - level) / 2
    return [float(np.quantile(boots, lo)), float(np.quantile(boots, 1 - lo))]


def decide_superiority(point, c95, threshold, higher_is_better=True):
    half = (c95[1] - c95[0]) / 2
    if half > threshold:
        return "inconclusive"
    if higher_is_better:
        return "supported" if point >= threshold and c95[0] > 0 else "not supported"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", default=[], help="NAME=SCENARIO:ARM:DIR")
    p.add_argument("--plans", action="append", default=[], help="SCENARIO=PLANDIR")
    p.add_argument("--out", type=Path)
    p.add_argument("--precision", action="store_true")
    a = p.parse_args()
    plans = {k: json.load(open(Path(v) / "plan.json")) for k, v in (x.split("=", 1) for x in a.plans)}
    runs = {}
    for spec in a.run:
        name, rest = spec.split("=", 1)
        scenario, arm, directory = rest.split(":", 2)
        runs[(scenario, arm)] = directory

    def get(scenario, arm, set_name, policy="mdn_code"):
        directory = runs.get((scenario, arm))
        return None if directory is None else load_run(directory, policy, set_name)

    def bstar(scenario, data):
        n = plans[scenario]["rows"]
        grid = sorted(next(iter(data.values())).keys())
        target = 0.005 * n
        return min(grid, key=lambda g: (abs(g - target), g))

    result = {"hypotheses": {}, "descriptive": {}}

    def part_a(label, scenario, set_name, arm_a, arm_b, threshold, equivalence=False, seed=20261121):
        A, B = get(scenario, arm_a, set_name), get(scenario, arm_b, set_name)
        if not A or not B:
            result["hypotheses"][label] = {"status": "missing runs"}
            return
        budget = bstar(scenario, A)
        point, boots, batches = mean_recall_diff(A, B, budget, np.random.default_rng(seed))
        point, boots = point * 100, boots * 100
        entry = {"scenario": scenario, "set": set_name, "contrast": f"{arm_a} - {arm_b}", "budget": budget,
                 "batches": batches, "delta_pp": point, "ci95": ci(boots, 0.95), "ci90": ci(boots, 0.90),
                 "bonferroni_ci": ci(boots, 1 - 0.05 / 3) if not equivalence else ci(boots, 1 - 2 * 0.05 / 3),
                 "threshold_pp": threshold}
        if equivalence:
            lo, hi = entry["ci90"]
            half = (hi - lo) / 2
            entry["outcome"] = ("supported (equivalent)" if -threshold < lo and hi < threshold
                                else "not supported" if lo >= threshold or hi <= -threshold
                                else "inconclusive" if half > threshold else "inconclusive")
            blo, bhi = entry["bonferroni_ci"]
            entry["outcome_bonferroni"] = "equivalent" if -threshold < blo and bhi < threshold else "not equivalent"
        else:
            entry["outcome"] = decide_superiority(point, entry["ci95"], threshold)
            entry["outcome_bonferroni"] = decide_superiority(point, entry["bonferroni_ci"], threshold)
        result["hypotheses"][label] = entry

    def part_b(label, arm_a, arm_b, max_ratio, width, policy_a="mdn_code", policy_b="mdn_code"):
        A = get("insert_only", arm_a, "partb", policy_a)
        B = get("insert_only", arm_b, "partb", policy_b)
        if not A or not B:
            result["hypotheses"][label] = {"status": "missing runs"}
            return
        r = work_ratio(A, B, np.random.default_rng(20261122))
        if r is None:
            result["hypotheses"][label] = {"status": "no batch reaches the target in both arms"}
            return
        c95 = ci(r["boots"], 0.95)
        outcome = ("inconclusive" if c95[1] - c95[0] > width
                   else "supported" if r["point"] <= max_ratio and c95[1] < 1 else "not supported")
        r.pop("boots")
        result["hypotheses"][label] = {**r, "contrast": f"{arm_a}:{policy_a} / {arm_b}:{policy_b}",
                                       "ci95": c95, "max_ratio": max_ratio, "outcome": outcome}

    if a.precision:
        # §5 precision check: per-query SD of each primary contrast at B*, projected half-width.
        for label, scenario, set_name, x, y in [("H5", "shift", "mq", "E1R", "E1"),
                                                ("H6", "constant_r", "test", "E1G", "E1"),
                                                ("H7", "constant_o", "test", "E1G", "E1")]:
            A, B = get(scenario, x, set_name), get(scenario, y, set_name)
            if not A or not B:
                continue
            budget = bstar(scenario, A)
            sds = [float((A[t][budget][0] - B[t][budget][0]).std(ddof=1) * 100) for t in sorted(set(A) & set(B))]
            n = len(next(iter(A.values()))[budget][0])
            sd = float(np.mean(sds))
            result["descriptive"][f"precision_{label}"] = {
                "per_query_sd_pp": sd, "queries": n, "snapshot_half_width_pp_95": 1.96 * sd / np.sqrt(n),
                "snapshot_half_width_pp_90": 1.645 * sd / np.sqrt(n), "budget": budget}
    else:
        part_a("H5", "shift", "mq", "E1R", "E1", 2.0)
        part_a("H5_insert_only", "insert_only", "test", "E1R", "E1", 2.0)
        part_a("H6", "constant_r", "test", "E1G", "E1", 1.0)
        part_a("H7", "constant_o", "test", "E1G", "E1", 1.0, equivalence=True)
        part_a("H12", "shift", "mq", "E1C", "E1", 0.0)
        if "outcome" in result["hypotheses"].get("H12", {}):
            result["hypotheses"]["H12"]["outcome"] = "descriptive"
            result["hypotheses"]["H12"].pop("outcome_bonferroni", None)
        part_b("H9", "E1C", "E1", 0.80, 0.20)
        part_b("H10", "E1", "E1", 0.90, 0.10, policy_a="calibrated")
        # Descriptive: dead-posting share and maintenance per run.
        for (scenario, arm), directory in runs.items():
            data = load_run(directory, "mdn_code", "test" if scenario != "shift" else "mq")
            batches = [json.loads(l) for l in open(Path(directory) / "batches.jsonl")]
            dead = {t: float(v[max(v)][2].sum() / max(v[max(v)][3].sum(), 1)) for t, v in data.items()} if data else {}
            result["descriptive"][f"{scenario}:{arm}"] = {
                "dead_share_at_max_budget": dead,
                "publications": sum(len(b["publications"]) for b in batches),
                "levels_last": batches[-1]["levels"] if batches else None,
                "ingest_seconds": sum(b["ingest_seconds"] for b in batches)}
    text = json.dumps(result, indent=1, default=float)
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
