"""Offline rank-30 tie check for the DLI-best integration pilot (INTEGRATION_PILOT.md, criteria 2-3).

    python3 -I tie_check.py --corpus F16 --queries OFFICIAL_F32 --plan PLAN_JSON --run RUN_DIR --out FILE

Uniform scenarios only (insert_only, constant_r): the live set of every batch is rebuilt with the
planner's own rules (phase3_plan.py) and checked against the plan's live SHA-256, and the
regenerated val gold is checked against the plan's gold SHA-256. Shift needs the k-means plan and
is checked separately.

Per (batch, query): Float64 scores as in the planner gold; s30, s31; `tie30` = s30 == s31 (an exact
score tie crosses the rank-30 boundary); `tied_gold` / `tied_other` = gold / non-gold live rows with
score == s30. A shortfall of m at a row is *tie-explained* iff tie30 and m <= min(tied_gold,
tied_other): the policy returned other copies of the same score. Checked for (a) every
`proportional_native` row at the full-search budget with R@30 < 30 (also mdn_code and proportional,
for context) and (b) every native-vs-control recall difference, where |difference| plays m.
Tie-excluded known answer: mean native R@30 at full search over queries with tie30 false.

Diagnostic (no gate): `qlike_lost` = gold points missing from a Qdrant-like exact top-30 (vector
normalised in f32 and stored as f16, as Qdrant stores Cosine vectors; f32 query), as in the
static-vs-DLI pilot 2 addendum. A non-tie miss with miss <= qlike_lost is `rounding_reproduced`.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("planner", HERE.parent / "emma_path" / "phase3_plan.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

FULL = 2_000_000
K, DIM = P.K, P.DIM
POLICIES = ("proportional_native", "proportional", "mdn_code")


def uniform_batches(plan):
    """The planner's uniform-scenario batches (inserts, deletes), rows 0-based."""
    n, nb, seed = plan["rows"], len(plan["batches"]) - 1, plan["seed"]
    workload, _, method = plan["scenario"].partition("_")
    ratio = {"insert": 0, "heavy": 10, "medium": 2, "constant": 1}[workload]
    initial = n // 10
    x = (n - initial) // nb
    out = [(np.arange(initial), np.array([], dtype=np.int64))]
    nxt, live = initial, set(range(initial))
    for b in range(nb):
        end = n if b == nb - 1 else nxt + x
        ins = np.arange(nxt, end)
        nxt = end
        live.update(ins.tolist())
        d = 0 if ratio == 0 else len(ins) // ratio
        pool = np.fromiter(live, dtype=np.int64)
        if d == 0:
            dels = np.array([], dtype=np.int64)
        elif method == "o":
            dels = np.sort(pool)[:d]
        else:
            dels = np.sort(np.random.default_rng(seed + b).choice(pool, d, replace=False))
        live.difference_update(dels.tolist())
        out.append((ins, dels))
    return out


def all_scores(corpus, q, chunk=65_536):
    qn = P.normalized(q)
    qf = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    s = np.empty((len(q), len(corpus)))
    sq = np.empty((len(q), len(corpus)), dtype=np.float32)
    for lo in range(0, len(corpus), chunk):
        x = corpus[lo: lo + chunk]
        s[:, lo: lo + chunk] = qn @ P.normalized(x).T
        xf = x.astype(np.float32)
        xq = (xf / np.linalg.norm(xf, axis=1, keepdims=True)).astype(np.float16).astype(np.float32)
        sq[:, lo: lo + chunk] = qf @ xq.T
    return s, sq


def load_rows(run_dir, b):
    rows = {}
    for line in open(Path(run_dir) / f"batch_{b:02d}" / "rows.jsonl"):
        r = json.loads(line)
        if r["set"] == "val" and r["policy"] in POLICIES:
            rows[(r["policy"], r["query"], r["budget"])] = r["recall30"]
    return rows


def explained(m, t):
    return bool(t["tie30"]) and m <= min(t["tied_gold"], t["tied_other"])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--corpus", type=Path, required=True)
    a.add_argument("--queries", type=Path, required=True)
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--analysis", type=Path, help="analysis.json (defaults to RUN/../analysis.json)")
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan["scenario"] != "shift", "uniform scenarios only"
    analysis = json.loads((args.analysis or args.run.parent / "analysis.json").read_text())
    diffs = analysis["native_vs_control"]["recall_differences"]
    assert not analysis["native_vs_control"]["work_mismatches"], "work mismatch: not a tie question"

    n = plan["rows"]
    corpus = np.memmap(args.corpus, dtype="<f2", mode="r").reshape(-1, DIM)[:n]
    q = np.fromfile(args.queries, dtype="<f4").reshape(P.QUERIES, DIM)[900:932]
    scores, qscores = all_scores(corpus, q)
    live = np.zeros(n, dtype=bool)
    per_batch, misses, diff_checks = [], [], []
    for b, (ins, dels) in enumerate(uniform_batches(plan)):
        live[ins] = True
        live[dels] = False
        rows = np.flatnonzero(live)
        meta = plan["batches"][b]
        assert P.live_digest(rows + 1) == (meta["live_count"], meta["live_sha256"]), ("live set", b)
        golds, ties = [], {}
        for i in range(len(q)):
            sc = scores[i, rows]
            top = np.flatnonzero(sc >= -np.partition(-sc, K)[K])  # every row tied with rank 31 kept
            order = top[np.lexsort((rows[top], -sc[top]))]
            s30, s31 = sc[order[K - 1]], sc[order[K]]
            gold = rows[order[:K]]
            golds.append(gold + 1)
            tied = np.flatnonzero(sc == s30)
            g = int(np.isin(rows[tied], gold).sum())
            scq = qscores[i, rows]
            topq = np.argpartition(-scq, K)[:K]
            ties[i] = {"tie30": bool(s30 == s31), "s30": float(s30), "s31": float(s31),
                       "tied_gold": g, "tied_other": int(len(tied) - g),
                       "qlike_lost": int(K - np.isin(rows[topq], gold).sum())}
        gsha = hashlib.sha256(np.stack(golds).astype("<u8").tobytes()).hexdigest()
        assert gsha == plan["files"][meta["sets"]["val"]["gold"]], ("regenerated gold differs", b)
        r = load_rows(args.run, b)
        for (pol, qi, budget), rec in sorted(r.items()):
            if budget == FULL and rec < K:
                misses.append({"batch": b, "policy": pol, "query": qi, "miss": K - rec, **ties[qi],
                               "explained": explained(K - rec, ties[qi]),
                               "rounding_reproduced": K - rec <= ties[qi]["qlike_lost"]})
        for d in diffs:
            if d[0] == b:
                _, s, qi, budget, c, nv = d
                diff_checks.append({"batch": b, "query": qi, "budget": budget, "control": c, "native": nv,
                                    **ties[qi], "explained": explained(abs(c - nv), ties[qi])})
        clean = [qi for qi in range(len(q)) if not ties[qi]["tie30"]]
        nat = [r[("proportional_native", qi, FULL)] / K for qi in clean]
        per_batch.append({"batch": b, "tie30_queries": [qi for qi in range(len(q)) if ties[qi]["tie30"]],
                          "native_raw": float(np.mean([r[("proportional_native", qi, FULL)] / K for qi in range(len(q))])),
                          "native_tie_excluded": float(np.mean(nat)) if nat else None})
        print(json.dumps(per_batch[-1]), flush=True)
    native_misses = [m for m in misses if m["policy"] == "proportional_native"]
    out = {"scenario": plan["scenario"], "run": str(args.run), "gold_regenerated": True,
           "per_batch": per_batch,
           "min_native_tie_excluded": min(p["native_tie_excluded"] for p in per_batch if p["native_tie_excluded"] is not None),
           "full_search_misses": misses,
           "native_misses_unexplained": [m for m in native_misses if not m["explained"]],
           "native_misses_neither_tie_nor_rounding": [m for m in native_misses
                                                      if not m["explained"] and not m["rounding_reproduced"]],
           "recall_differences": diff_checks,
           "recall_differences_unexplained": [d for d in diff_checks if not d["explained"]]}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"min_native_tie_excluded": out["min_native_tie_excluded"],
                      "native_misses": len(native_misses),
                      "native_misses_unexplained_by_tie": len(out["native_misses_unexplained"]),
                      "native_misses_neither_tie_nor_rounding": out["native_misses_neither_tie_nor_rounding"],
                      "recall_differences": len(diff_checks),
                      "recall_differences_unexplained": out["recall_differences_unexplained"]}, indent=1))


if __name__ == "__main__":
    main()
