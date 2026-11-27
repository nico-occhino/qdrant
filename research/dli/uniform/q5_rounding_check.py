"""Q5 gate 4 shortfall classification (Q5_PREREGISTRATION.md §6.4).

    q5_rounding_check.py --plan PLAN_DIR --corpus VECTORS_F16 --run ARM_DIR [--run ARM_DIR ...] --out FILE

The frozen gate is "full-search known answer >= 0.999 on every batch and panel, using plan gold; the tie and
rounding slack from the DLI-best pilot applies, so a shortfall is classified with the tie check before the gate is
judged". `q5_gates.py` reports the raw shortfalls; this script is that classification step, for every batch and panel.
It applies the stored-representation rule of `q4_rounding_check.py` unchanged (same functions, imported): plan gold
is Float64 over the raw Float16 rows, Qdrant searches its stored (AVX+FMA normalised, then Float16) vectors, and a
miss m at a query is explained iff 30 - |gold ∩ T| <= m <= 30 - |gold ∩ T| + max(0, |T| - 30), with T the
stored-representation tie set at rank 30.
Also checked: the full-search returned recall is identical across the given arms, query by query (exact search over
the same live set must not depend on the arm).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q4_rounding_check as Q  # noqa: E402

F, FULL, K = Q.F, Q.FULL, Q.K


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--corpus", type=Path, required=True)
    a.add_argument("--run", type=Path, action="append", required=True)
    a.add_argument("--sets", default="mq,mq_val")
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    plan = json.loads((args.plan / "plan.json").read_text())
    corpus = np.memmap(args.corpus, dtype="<f2", mode="r").reshape(-1, F.DIM)
    cells, ok, arm_equal = {}, True, True
    for t, meta in enumerate(plan["batches"]):
        ids = Q.live_ids(args.plan, t)
        assert len(ids) == meta["live_count"], "live count"
        for set_name in args.sets.split(","):
            s = meta["sets"][set_name]
            queries = np.fromfile(args.plan / s["queries"], "<f4").reshape(-1, F.DIM)[: s["n"]]
            gold = np.fromfile(args.plan / s["gold"], "<u8").reshape(-1, K)
            _, _, ties, offs = F.stored_top_ties(corpus, ids - 1, queries)
            reach = [(len(set(gold[i].tolist()) & set((np.asarray(ties[offs[i]:offs[i + 1]]) + 1).tolist())),
                      int(offs[i + 1] - offs[i])) for i in range(len(queries))]
            per_arm = {}
            for run in args.run:
                rows = [json.loads(l) for l in open(run / f"batch_{t:02d}" / "rows.jsonl")]
                per_arm[run.name] = {r["query"]: r["recall30"] for r in rows
                                     if r["set"] == set_name and r["policy"] == "mdn_code" and r["budget"] == FULL}
            ref = per_arm[args.run[0].name]
            same = all(v == ref for v in per_arm.values())
            arm_equal &= same
            res = []
            for q, got in sorted(ref.items()):
                miss = K - got
                inter, tsize = reach[q]
                lo, hi = K - inter, K - inter + max(0, tsize - K)
                res.append({"query": q, "miss": miss, "unreachable_gold": lo, "tie_slack": max(0, tsize - K),
                            "explained": miss == 0 or lo <= miss <= hi})
            bad = [r for r in res if not r["explained"]]
            ok &= not bad
            cells[f"{t}/{set_name}"] = {"queries": len(res), "recall": 1 - sum(r["miss"] for r in res) / (K * len(res)),
                                        "misses": sum(r["miss"] for r in res),
                                        "unreachable_gold": sum(r["unreachable_gold"] for r in res),
                                        "arms_identical": same, "unexplained": bad}
            print(t, set_name, cells[f"{t}/{set_name}"]["misses"], cells[f"{t}/{set_name}"]["unreachable_gold"],
                  len(bad), same, flush=True)
    out = {"arms": [r.name for r in args.run], "cells": cells, "arms_identical": bool(arm_equal),
           "pass": bool(ok and arm_equal)}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"pass": out["pass"], "arms_identical": out["arms_identical"],
                      "unexplained": sum(len(c["unexplained"]) for c in cells.values())}))


if __name__ == "__main__":
    main()
