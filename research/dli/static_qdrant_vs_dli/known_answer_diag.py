"""Known-answer diagnosis for pilot2 shift (job 25875196): for every query that misses at
full search, print the float64 score margin around rank 30, the exact-duplicate group at the
boundary, and whether Qdrant-like f16 re-quantization reorders the boundary.

Usage: python3 -I tie_diag.py <corpus1m.f16> <regen plan dir> <job result dir> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np

DIM, K = 768, 30
corpus_path, plan_dir, job_dir, out_path = map(Path, sys.argv[1:5])
corpus = np.memmap(corpus_path, dtype="<f2", mode="r").reshape(-1, DIM)
plan = json.loads((plan_dir / "plan.json").read_text())

def misses(arm, b):
    rows = [json.loads(l) for l in open(job_dir / arm / f"batch_{b:02d}" / "rows.jsonl")]
    mx = max(r["budget"] for r in rows)
    return {(r["set"], r["query"]): 30 - r["recall30"] for r in rows if r["budget"] == mx and r["recall30"] < 30}

report = []
for b in range(6):
    per_arm = {arm: misses(arm, b) for arm in ("D", "S1", "SG")}
    keys = sorted(set().union(*per_arm.values()))
    if not keys:
        continue
    bat = plan["batches"][b]
    live = np.zeros(len(corpus), bool)
    for i in range(b + 1):
        d = plan_dir / f"batch_{i:02d}"
        live[np.fromfile(d / "insert.u32", "<u4").astype(np.int64) - 1] = True
        live[np.fromfile(d / "delete.u32", "<u4").astype(np.int64) - 1] = False
    rows = np.flatnonzero(live)
    assert len(rows) == bat["live_count"]
    X = corpus[rows].astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # Qdrant-like stored vector: f32 normalize, then cast to f16.
    Xq = corpus[rows].astype(np.float32)
    Xq = (Xq / np.linalg.norm(Xq, axis=1, keepdims=True)).astype(np.float16).astype(np.float32)
    for s, qi in keys:
        meta = bat["sets"][s]
        q = np.fromfile(plan_dir / meta["queries"], "<f4").reshape(-1, DIM)[qi]
        gold = np.fromfile(plan_dir / meta["gold"], "<u8").reshape(-1, K)[qi]
        qn = q.astype(np.float64) / np.linalg.norm(q)
        sc = X @ qn
        order = np.lexsort((rows, -sc))
        top = rows[order[:40]] + 1
        assert np.array_equal(top[:K], gold), (b, s, qi, "regenerated gold differs")
        s30, s31 = sc[order[K - 1]], sc[order[K]]
        # exact-duplicate group of the rank-30 row (identical f16 rows)
        r30 = corpus[rows[order[K - 1]]]
        dup = int(np.all(corpus[rows] == r30, axis=1).sum())
        tie64 = int(np.sum(sc == s30))
        # Qdrant-like scoring, f32 query normalized
        qf = (q / np.linalg.norm(q)).astype(np.float32)
        scq = Xq @ qf
        oq = np.lexsort((rows, -scq))
        topq = set((rows[oq[:K]] + 1).tolist())
        lost_q = sorted(set(gold.tolist()) - topq)
        # width of the f32 band around rank 30 that contains distinct-but-close rows
        band = int(np.sum(np.abs(scq - scq[oq[K - 1]]) <= 2 ** -11))
        report.append({
            "batch": b, "set": s, "query": qi, "miss": {a: per_arm[a].get((s, qi), 0) for a in per_arm},
            "s30_f64": s30, "s31_f64": s31, "margin_f64": s30 - s31, "f64_tie_count_at_s30": tie64,
            "dup_rows_equal_to_rank30": dup, "top40_f64": [round(float(sc[o]), 7) for o in order[:40]],
            "qdrantlike_lost_from_gold": len(lost_q), "f16_band_rows_near_rank30": band,
        })
        print(json.dumps({k: v for k, v in report[-1].items() if k != "top40_f64"}), flush=True)
Path(out_path).write_text(json.dumps(report, indent=1))
