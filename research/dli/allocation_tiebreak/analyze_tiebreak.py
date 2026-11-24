"""Analysis for research/dli/allocation_tiebreak (PREREGISTRATION.md §3).

Reuses the frozen functions of ../policy_replay/analyze_policy_replay.py unchanged, with the
policy pair set to A = `calibrated`, B = `proportional` (the module reads A and B at call time).

Primary: R = mean_t W_cal(t) / mean_t W_prop(t) on the `partb` panel of the DLI-best (E1 +
consolidation rho = 0.25) insert-only snapshots, paired query bootstrap stratified by batch.
Choice (frozen rule): `calibrated` only if decided "calibrated better"; otherwise `proportional`.
S5: `mdn_code` and `calibrated` rows are compared with the Phase 3 E1C run (job 25859945).

    analyze_tiebreak.py --run DIR --set partb [--reference DIR] [--out FILE]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("apr", HERE.parent / "policy_replay" / "analyze_policy_replay.py")
apr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apr)

SEED = 20261125
A, B = "calibrated", "proportional"
LABELS = {"proportional better": "calibrated better", "mdn_code better": "proportional better"}


def choose(decision):
    return "calibrated" if decision == "calibrated better" else "proportional"


def analyze(run_dir, set_name, reference=None, resamples=apr.p3.RESAMPLES, seed=SEED):
    apr.A, apr.B = A, B
    run_a = apr.p3.load_run(run_dir, A, set_name)
    run_b = apr.p3.load_run(run_dir, B, set_name)
    rng = np.random.default_rng(seed)
    out = {"set": set_name, "pair": f"{A} / {B}", "gate_problems": apr.gates(run_a, run_b)}
    pr = apr.work_ratio(run_a, run_b, resamples, rng)
    for key in ("decision", "decision_bonferroni"):
        if key in pr:
            pr[key] = LABELS.get(pr[key], pr[key])
    if "R_prop_over_mdn" in pr:
        pr["R_cal_over_prop"] = pr.pop("R_prop_over_mdn")
    out["primary"] = pr
    out["choice"] = choose(pr.get("decision")) if not out["gate_problems"] else "blocked by gate problems"
    out["S1_delta_recall_at_Bstar"] = apr.recall_diff_at(run_a, run_b, apr.B_STAR, resamples, rng)
    out["S3_S4_levels_at_Bstar"] = apr.level_diagnostics(run_dir, set_name, apr.B_STAR)
    if reference:
        out["S5_reproduction"] = {
            p: apr.reproduction(apr.p3.load_run(run_dir, p, set_name), apr.p3.load_run(reference, p, set_name))
            for p in ("mdn_code", "calibrated")}
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--set", required=True)
    a.add_argument("--reference", type=Path)
    a.add_argument("--resamples", type=int, default=apr.p3.RESAMPLES)
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    result = analyze(args.run, args.set, args.reference, args.resamples)
    text = json.dumps(result, indent=1, default=float)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
