"""Official SISAP23 gold versus exact search over the stored Float16 corpus (10M).

For the first --q queries of a gold directory, computes the exact inner-product top-30 over all
stored Float16 rows (Float32 accumulation, ties -> lower id) and reports how much of the official
top-10/top-30 it contains. This bounds the recall a perfect index over the stored data can reach
against the official oracle (near-tie and precision disagreement), so it is reported, not used to
change any oracle.
"""
import argparse, json, os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--vectors", required=True); ap.add_argument("--gold", required=True)
ap.add_argument("--q", type=int, default=256); ap.add_argument("--out", required=True)
a = ap.parse_args()
D, K = 768, 30
n = os.path.getsize(a.vectors) // (2 * D)
X = np.memmap(a.vectors, dtype=np.float16, mode="r", shape=(n, D))
Q = np.load(Path(a.gold) / "queries.npy")[: a.q]
G = np.load(Path(a.gold) / "gold.npy")[: a.q]
bs = np.full((len(Q), K), -np.inf, np.float32); bi = np.zeros((len(Q), K), np.int64)
for s in range(0, n, 250_000):
    e = min(n, s + 250_000)
    S = Q @ X[s:e].astype(np.float32).T
    p = np.argpartition(-S, K, axis=1)[:, :K]
    ms = np.concatenate([bs, np.take_along_axis(S, p, 1)], 1); mi = np.concatenate([bi, p + s], 1)
    o = np.lexsort((mi, -ms), axis=1)[:, :K]
    bs = np.take_along_axis(ms, o, 1); bi = np.take_along_axis(mi, o, 1)
r10 = np.array([len(set(bi[i, :10]) & set(G[i, :10])) / 10 for i in range(len(Q))])
r30 = np.array([len(set(bi[i]) & set(G[i])) / 30 for i in range(len(Q))])
res = {"queries": len(Q), "gold": a.gold, "exact_f16_recall10_vs_official": float(r10.mean()),
       "exact_f16_recall30_vs_official": float(r30.mean()),
       "queries_r10_below_1": int((r10 < 1).sum()), "queries_r30_below_1": int((r30 < 1).sum())}
Path(a.out).write_text(json.dumps(res, indent=1)); print(res)
