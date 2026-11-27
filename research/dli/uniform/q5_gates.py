"""Per-job gates for one Q5 arm run (Q5_PREREGISTRATION.md §6, gate 4).

    q5_gates.py --run DIR --plan PLAN_DIR --sets mq,mq_val --arm E1T [--e1 E1_DIR] --out FILE

* ledger: every batches.jsonl record's live count and live-ID digest equal the plan's;
* completeness: every plan batch has rows for every set, budget and query;
* monotone_work: mean vectors scored never decreases with the budget, per batch and set (violations listed);
* known_answer: mean Recall@30 at the full-search budget (2,000,000) >= 0.999 per batch and set. Shortfalls are
  listed for the tie and rounding classification;
* decisions (E1T only, needs --e1 for the first fits): analyze_q5.check_decisions.
Exit status 0 only if every gate passes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_q5 as Q  # noqa: E402

FULL = 2_000_000
KNOWN = 0.999


def gates(run, plan_dir, sets, arm, e1=None):
    plan = json.loads((Path(plan_dir) / "plan.json").read_text())
    log = [json.loads(line) for line in open(Path(run) / "batches.jsonl")]
    g = {"ledger": len(log) == len(plan["batches"]) and all(
        r["live"] == b["live_count"] and r["live_sha256"] == b["live_sha256"] for r, b in zip(log, plan["batches"]))}
    complete, mono, known, short = True, [], [], []
    for s in sets:
        data = Q.load_rows(run, s)
        complete &= len(data) == len(plan["batches"])
        for t, (budgets, rec, wrk, _) in data.items():
            mw = wrk.mean(axis=0)
            mono += [{"set": s, "batch": t, "knobs": [budgets[i], budgets[i + 1]]}
                     for i in np.flatnonzero(np.diff(mw) < -1e-9)]
            if FULL in budgets:
                r = float(rec[:, budgets.index(FULL)].mean())
                known.append(r)
                if r < KNOWN:
                    short.append({"set": s, "batch": t, "recall": r})
            else:
                complete = False
    g.update({"completeness": bool(complete), "monotone_work": not mono, "non_monotone_points": mono,
              "known_answer": bool(known) and not short, "known_answer_min": min(known) if known else None,
              "known_answer_shortfalls": short})
    if arm == "E1T":
        first = sum(b["trained"] for b in Q.trainer_seconds(e1)) if e1 else None
        chk = Q.check_decisions(Q.decisions(run), Q.trainer_seconds(run), first) if e1 else {"pass": False,
                                                                                          "problems": ["no --e1"]}
        g["decisions"] = chk["pass"]
        g["decision_check"] = chk
    g["pass"] = all(v for k, v in g.items() if isinstance(v, bool))
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--sets", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--e1")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    g = gates(a.run, a.plan, a.sets.split(","), a.arm, a.e1)
    a.out.write_text(json.dumps(g, indent=1) + "\n")
    print(json.dumps({k: v for k, v in g.items() if isinstance(v, bool)}))
    sys.exit(0 if g["pass"] else 1)


if __name__ == "__main__":
    main()
