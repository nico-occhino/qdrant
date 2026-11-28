"""CLI selection pilot (CLI_PILOT_PREREGISTRATION.md §4–§5).

    analyze_pilot.py --run W1:500=DIR --run W1:1000=DIR ... [--out FILE]

Each DIR is a `cli_stage_a_plan` output. All curves use G1 recall (`recall30_g1`). The gate-only budget 2,000,000 is
excluded from W@0.90 and used only for the known answer.
* W@0.90 per (workload, b_max, panel, batch): the first measured crossing of mean G1 Recall@30 = 0.90 against mean
  vectors scored, interpolated, never extrapolated (`upper_bound` / `not_attained`).
* Choice: the b_max with the lowest mean W@0.90 on W1 `val` over batches 1–20, over batches where every candidate
  attains. Ties go to the lower total fine-tune seconds on W1.
* Gates per run: completeness (every batch, panel, query and budget); G1 known answer = 1.0 at the exhaustive budget on
  every batch and panel (>= 0.999 mean); monotone mean work along the grid.
* Bracketing, reported per (workload, b_max, panel): batches with `upper_bound` or `not_attained`.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGET, GATE_BUDGET = 0.90, 2_000_000


def w_at(rec, scored):
    if rec[0] >= TARGET:
        return float(scored[0]), "upper_bound"
    for i in range(1, len(rec)):
        if rec[i] >= TARGET:
            return float(scored[i - 1] + (TARGET - rec[i - 1]) / (rec[i] - rec[i - 1]) * (scored[i] - scored[i - 1])), "interpolated"
    return None, "not_attained"


def load(run):
    cells = defaultdict(lambda: defaultdict(list))  # (panel, batch) -> budget -> [(g1, scored)]
    for line in open(Path(run) / "rows.jsonl"):
        r = json.loads(line)
        cells[(r["set"], r["batch"])][r["budget"]].append((r["recall30_g1"] / 30, r["scored"]))
    batches = [json.loads(l) for l in open(Path(run) / "batches.jsonl")]
    return cells, batches


def analyze(runs):
    res = {"runs": {}, "choice": None}
    for key, run in runs.items():
        cells, batches = load(run)
        per, gates, bracket = {}, {"complete": True, "known_answer": True, "monotone": True, "known_answer_min": 1.0}, defaultdict(list)
        nq = {}
        for (panel, t), by in sorted(cells.items()):
            budgets = sorted(b for b in by if b != GATE_BUDGET)
            nq.setdefault(panel, len(by[budgets[0]]))
            if any(len(by[b]) != nq[panel] for b in by) or GATE_BUDGET not in by:
                gates["complete"] = False
            rec = [np.mean([x[0] for x in by[b]]) for b in budgets]
            sc = [np.mean([x[1] for x in by[b]]) for b in budgets]
            if any(b2 < b1 - 1e-9 for b1, b2 in zip(sc, sc[1:])):
                gates["monotone"] = False
            ka = float(np.mean([x[0] for x in by.get(GATE_BUDGET, [(0, 0)])]))
            gates["known_answer_min"] = min(gates["known_answer_min"], ka)
            w, kind = w_at(rec, sc)
            per[f"{panel}/{t}"] = {"w090": w, "kind": kind}
            if kind != "interpolated":
                bracket[panel].append((t, kind))
        gates["known_answer"] = gates["known_answer_min"] >= 0.999
        gates["batches"] = len(batches)
        gates["pass"] = gates["complete"] and gates["known_answer"] and gates["monotone"]
        res["runs"][key] = {"per_batch": per, "gates": gates, "bracketing": dict(bracket),
                            "finetune_seconds": float(sum(b["maintenance"]["train_seconds"] for b in batches[1:])),
                            "wall_seconds": float(sum(b["batch_seconds"] for b in batches))}
    w1 = {k: v for k, v in res["runs"].items() if k.startswith("W1:")}
    if w1:
        common = None
        for k, v in w1.items():
            ok = {int(p.split("/")[1]) for p, x in v["per_batch"].items() if p.startswith("val/") and x["w090"] is not None}
            common = ok if common is None else common & ok
        used = sorted(t for t in common if 1 <= t <= 20)
        mean = {k: float(np.mean([v["per_batch"][f"val/{t}"]["w090"] for t in used])) if used else None for k, v in w1.items()}
        ranked = sorted((k for k in mean if mean[k] is not None), key=lambda k: (mean[k], w1[k]["finetune_seconds"]))
        res["choice"] = {"b_max": int(ranked[0].split(":")[1]) if ranked else None, "mean_w090": mean, "batches_used": used}
    return res


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", action="append", required=True, help="WORKLOAD:BMAX=DIR")
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    res = analyze(dict(x.split("=", 1) for x in args.run))
    text = json.dumps(res, indent=1, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({"choice": res["choice"], "gates": {k: v["gates"] for k, v in res["runs"].items()},
                      "bracketing": {k: v["bracketing"] for k, v in res["runs"].items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
