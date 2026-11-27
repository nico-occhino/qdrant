"""Q4 §5 gate: classify full-search shortfalls against plan gold as stored-representation rounding.

    q4_rounding_check.py --plan PLAN_DIR --corpus VECTORS_1M_F16 --run RUN_DIR [--run RUN_DIR ...] --out FILE

Plan gold is exact Float64 over the *raw* Float16 rows. Qdrant searches its *stored* vectors: cosine-normalized
by its AVX+FMA path, then stored as Float16, modelled bit for bit by `final_benchmark/fb_plan.py`.
For every snapshot and query, T is the stored-representation tie set: live ids with stored score >= s30 − eps.
A full search can return at most min(30, |gold ∩ T|) plan-gold ids. So a full-search miss m is
**rounding-explained** iff m >= 30 − |gold ∩ T| (at least that many gold ids are unreachable by exact stored
search) and m <= 30 − |gold ∩ T| + max(0, |T| − 30) (the excess is within tie slack).
The gate passes if every shortfall is rounding-explained.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fb_plan", HERE.parent / "final_benchmark" / "fb_plan.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
FULL, K, SNAPSHOTS = 2_000_000, 30, (10, 11, 12)


def live_ids(plan_dir, t):
    live = set()
    for b in range(t + 1):
        d = Path(plan_dir) / f"batch_{b:02d}"
        live.update(np.fromfile(d / "insert.u32", "<u4").tolist())
        live.difference_update(np.fromfile(d / "delete.u32", "<u4").tolist())
    return np.array(sorted(live), dtype=np.int64)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--corpus", type=Path, required=True)
    a.add_argument("--run", type=Path, action="append", required=True)
    a.add_argument("--set", default="mq_val")
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    plan = json.loads((args.plan / "plan.json").read_text())
    corpus = np.memmap(args.corpus, dtype="<f2", mode="r").reshape(-1, F.DIM)
    out, ok = {}, True
    for t in SNAPSHOTS:
        meta = plan["batches"][t]
        s = meta["sets"][args.set]
        queries = np.fromfile(args.plan / s["queries"], "<f4").reshape(-1, F.DIM)[: s["n"]]
        gold = np.fromfile(args.plan / s["gold"], "<u8").reshape(-1, K)
        ids = live_ids(args.plan, t)
        assert len(ids) == meta["live_count"], "live count"
        top, s30, ties, offs = F.stored_top_ties(corpus, ids - 1, queries)
        reach = []
        for i in range(len(queries)):
            T = set((np.asarray(ties[offs[i]:offs[i + 1]]) + 1).tolist())
            g = set(gold[i].tolist())
            reach.append((len(g & T), len(T)))
        for run in args.run:
            rows = [json.loads(l) for l in open(run / f"batch_{t:02d}" / "rows.jsonl")]
            full = {r["query"]: r["recall30"] for r in rows
                    if r["set"] == args.set and r["policy"] == "mdn_code" and r["budget"] == FULL}
            res = []
            for q, got in sorted(full.items()):
                miss = K - got
                inter, tsize = reach[q]
                lo, hi = K - inter, K - inter + max(0, tsize - K)
                res.append({"query": q, "miss": miss, "unreachable_gold": lo, "tie_slack": max(0, tsize - K),
                            "explained": miss == 0 or lo <= miss <= hi})
            bad = [r for r in res if not r["explained"]]
            ok &= not bad
            out[f"{run.parent.name}/{t}"] = {"queries": len(res), "misses": sum(r["miss"] for r in res),
                                              "unreachable_gold": sum(r["unreachable_gold"] for r in res),
                                              "unexplained": bad}
    out["pass"] = bool(ok)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v if k == "pass" else {x: v[x] for x in ("misses", "unreachable_gold")} | {"unexplained": len(v["unexplained"])})
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
