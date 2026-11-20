#!/usr/bin/env python3
"""Query splits and corpus-as-query labels (PREREGISTRATION.md).

    make_inputs.py --queries Q.f32 --corpus corpus.f16le --n N --out DIR
                   [--official-gt GT.i32] [--check-val val.queries.f32] [--pseudo 100000] [--threads 16]

Writes, under DIR:
  T.f32 V.f32 G.f32 C.f32     unit-normalised official queries (rows in splits.json)
  P.f32, P.ids.u64            corpus-as-query vectors (stored-vector semantics) and their 1-based ids
  P.gold.u64                  exact top-10 neighbours of each P point, self excluded (1-based ids)
  {T,V,C}.official.u64        official top-30 (1-based) when --official-gt is given (10M only)
  splits.json                 rows, counts, near-duplicate exclusions, hashes, timings

Stored-vector semantics: Qdrant normalises cosine vectors before storing them as Float16, so a
corpus row x is represented as f16(x / ||x||) and scored by a Float32 dot product. P labels are
computed with that representation by exact (chunked) matrix products.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

D = 768
SPLITS = {
    "T": [(0, 1000), (3000, 4000), (6000, 9000), (9256, 10000)],
    "V": [(1000, 3000)],
    "G": [(4000, 5000)],
    "C": [(5000, 6000)],
}
PSEUDO_SEED = 20261119
NEAR_DUP = 0.95


def rows_of(ranges):
    return np.concatenate([np.arange(a, b) for a, b in ranges])


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def stored(raw_f16):
    """Qdrant cosine storage: normalise in f32, store as f16, score in f32."""
    x = raw_f16.astype(np.float32)
    return unit(x).astype(np.float16).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--official-gt")
    ap.add_argument("--check-val")
    ap.add_argument("--pseudo", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    began = time.time()
    meta = {"queries": a.queries, "queries_sha256": sha(a.queries), "corpus": a.corpus, "n": a.n,
            "splits": {}, "pseudo_seed": PSEUDO_SEED, "near_duplicate_cosine": NEAR_DUP}

    q_all = np.fromfile(a.queries, dtype=np.float32).reshape(-1, D)
    assert q_all.shape[0] == 10_000
    gt = None
    if a.official_gt:
        gt = np.fromfile(a.official_gt, dtype=np.int32).reshape(10_000, -1)
        meta["official_gt_sha256"] = sha(a.official_gt)
    for name, ranges in SPLITS.items():
        rows = rows_of(ranges)
        q = unit(q_all[rows])
        q.tofile(out / f"{name}.f32")
        meta["splits"][name] = {"ranges": ranges, "count": len(rows), "sha256": sha(out / f"{name}.f32")}
        if gt is not None and name in ("T", "V", "C"):
            gt[rows, :30].astype(np.uint64).tofile(out / f"{name}.official.u64")
    if a.check_val:
        ref = np.fromfile(a.check_val, dtype=np.float32).reshape(-1, D)
        mine = np.fromfile(out / "V.f32", dtype=np.float32).reshape(-1, D)
        meta["val_matches_accepted"] = {"bitwise": bool(np.array_equal(ref, mine)),
                                        "max_abs_diff": float(np.abs(ref - mine).max())}
        assert np.abs(ref - mine).max() < 1e-6, "V must equal the accepted Tier 1-3 validation queries"

    # Corpus-as-query sample with a near-duplicate guard against held-out queries.
    corpus = np.memmap(a.corpus, dtype=np.float16, mode="r").reshape(-1, D)[: a.n]
    rng = np.random.default_rng(PSEUDO_SEED)
    ids0 = np.sort(rng.choice(a.n, size=min(a.n, int(a.pseudo * 1.05)), replace=False))
    P = stored(np.asarray(corpus[ids0]))
    held = torch.from_numpy(np.concatenate([np.fromfile(out / f"{s}.f32", dtype=np.float32).reshape(-1, D)
                                            for s in ("V", "G", "C")]))
    maxcos = torch.cat([(torch.from_numpy(P[i:i + 20000]) @ held.T).max(1).values
                        for i in range(0, len(P), 20000)]).numpy()
    keep = maxcos < NEAR_DUP
    excluded = int((~keep).sum())
    ids0, P = ids0[keep][: a.pseudo], P[keep][: a.pseudo]
    P.tofile(out / "P.f32")
    (ids0.astype(np.uint64) + 1).tofile(out / "P.ids.u64")
    meta["pseudo"] = {"requested": a.pseudo, "kept": int(len(ids0)), "excluded_near_duplicate": excluded,
                      "max_cosine_to_heldout_p99": float(np.percentile(maxcos, 99))}

    # Exact top-11 over the stored corpus, then drop the point itself.
    t0 = time.time()
    Pt = torch.from_numpy(P)
    best_s = torch.full((len(P), 11), -np.inf)
    best_i = torch.zeros((len(P), 11), dtype=torch.int64)
    chunk = 250_000
    for s in range(0, a.n, chunk):
        X = torch.from_numpy(stored(np.asarray(corpus[s:min(a.n, s + chunk)])))
        for i in range(0, len(P), 20000):
            S = Pt[i:i + 20000] @ X.T
            v, j = S.topk(11, dim=1)
            ms = torch.cat([best_s[i:i + 20000], v], 1)
            mi = torch.cat([best_i[i:i + 20000], j + s], 1)
            # Order by (-score, id): stable sort on id, then on score.
            o = torch.argsort(mi, dim=1, stable=True)
            ms, mi = torch.gather(ms, 1, o), torch.gather(mi, 1, o)
            o = torch.argsort(-ms, dim=1, stable=True)[:, :11]
            best_s[i:i + 20000], best_i[i:i + 20000] = torch.gather(ms, 1, o), torch.gather(mi, 1, o)
    gold = np.zeros((len(P), 10), dtype=np.uint64)
    self_found = 0
    for r in range(len(P)):
        row = [int(x) for x in best_i[r] if int(x) != int(ids0[r])]
        self_found += int(len(row) == 10)
        gold[r] = np.asarray(row[:10], dtype=np.uint64) + 1
    gold.tofile(out / "P.gold.u64")
    meta["pseudo"].update({"gold_seconds": time.time() - t0, "self_in_top11": self_found,
                           "gold": "exact top-10 over the stored corpus representation, self excluded"})
    for f in ("P.f32", "P.ids.u64", "P.gold.u64"):
        meta["pseudo"][f + ".sha256"] = sha(out / f)
    meta["seconds"] = time.time() - began
    (out / "splits.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "splits"}, indent=1))


if __name__ == "__main__":
    main()
