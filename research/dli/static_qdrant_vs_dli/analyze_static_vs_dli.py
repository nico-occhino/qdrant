"""Analysis for research/dli/static_qdrant_vs_dli (PREREGISTRATION.md §5–§6).

Per workload and evaluated batch t, the work to reach mean Recall@30 = 0.90 on the test
set: W_S(t) from arm S (policy `static_nprobe`, swept over nprobe) and W_D(t) from arm D
(policy `mdn_code`, swept over the global candidate budget). Linear interpolation between
bracketing grid points, never extrapolation; a batch where an arm does not reach the target
is "not attained" (no value invented).

Primary estimand per workload: R = mean_t W_D(t) / mean_t W_S(t) over the batches where
both arms attain, with a paired query bootstrap stratified by batch (the same resampled
query indices for both arms within a batch). The snapshots of one workload share queries
and one ingestion stream, so the batch-mean is a single trajectory statistic; the CI covers
query sampling only, not ingestion order or model-training variance.

    analyze_static_vs_dli.py --pair W=S_DIR:D_DIR ... [--set test] [--out FILE]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p3", HERE.parent / "emma_path" / "analyze_phase3.py")
p3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3)

SEED = 20261123
TARGET = 0.90
MIN_ATTAINED_SHARE = 0.8


def curve(run, t):
    """(recall[budget, q], work[budget, q]) with budgets in increasing mean-work order."""
    budgets = sorted(run[t], key=lambda b: run[t][b][1].mean())
    return (np.stack([run[t][b][0] for b in budgets]), np.stack([run[t][b][1] for b in budgets]))


def analyze_pair(s_dir, d_dir, set_name, resamples=p3.RESAMPLES, seed=SEED):
    S = p3.load_run(s_dir, "static_nprobe", set_name)
    D = p3.load_run(d_dir, "mdn_code", set_name)
    batches = sorted(set(S) & set(D))
    rng = np.random.default_rng(seed)
    per_batch, ws_pt, wd_pt, ws_bs, wd_bs = {}, [], [], [], []
    for t in batches:
        rs, ws = curve(S, t)
        rd, wd = curve(D, t)
        assert rs.shape[1] == rd.shape[1], "both arms must answer the same queries"
        ps = p3.interp_work(rs.mean(axis=1), ws.mean(axis=1))[0]
        pd = p3.interp_work(rd.mean(axis=1), wd.mean(axis=1))[0]
        per_batch[t] = {"W_S": None if np.isnan(ps) else float(ps), "W_D": None if np.isnan(pd) else float(pd),
                        "ratio": None if np.isnan(ps) or np.isnan(pd) else float(pd / ps)}
        if np.isnan(ps) or np.isnan(pd):
            continue
        idx = rng.integers(0, rs.shape[1], size=(resamples, rs.shape[1]))
        ws_bs.append(p3.interp_work(rs[:, idx].mean(axis=2).T, ws[:, idx].mean(axis=2).T))
        wd_bs.append(p3.interp_work(rd[:, idx].mean(axis=2).T, wd[:, idx].mean(axis=2).T))
        ws_pt.append(ps)
        wd_pt.append(pd)
    attained = len(ws_pt)
    out = {"set": set_name, "batches": batches, "attained_both": attained,
           "attained_share": attained / len(batches) if batches else 0.0, "per_batch": per_batch,
           "not_attained_S": [t for t in batches if per_batch[t]["W_S"] is None],
           "not_attained_D": [t for t in batches if per_batch[t]["W_D"] is None]}
    if attained == 0:
        out["status"] = "target not attained"
        return out
    num, den = np.nanmean(np.stack(wd_bs), axis=0), np.nanmean(np.stack(ws_bs), axis=0)
    boots = num / den
    out.update({"ratio_D_over_S": float(np.mean(wd_pt) / np.mean(ws_pt)),
                "ci95": p3.ci(boots, 0.95), "ci90": p3.ci(boots, 0.90),
                "resamples_not_reached": int(np.isnan(boots).sum()),
                "status": "ok" if out["attained_share"] >= MIN_ATTAINED_SHARE else "descriptive (attainment below 80%)"})
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pair", action="append", required=True, help="WORKLOAD=S_DIR:D_DIR")
    a.add_argument("--set", default="test")
    a.add_argument("--resamples", type=int, default=p3.RESAMPLES)
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    result = {}
    for spec in args.pair:
        name, dirs = spec.split("=", 1)
        s_dir, d_dir = dirs.split(":", 1)
        result[name] = analyze_pair(s_dir, d_dir, args.set, args.resamples)
    text = json.dumps(result, indent=1, default=float)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
