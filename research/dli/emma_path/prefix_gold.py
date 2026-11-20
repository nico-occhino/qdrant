"""Exact gold for the first N corpus rows (SISAP23 LAION2B CLIP-768, Float16).

Scores are float64 cosine between the stored Float16 rows and the float32 queries; ties
break by the smaller one-based ID. Outputs, in --out:

* ground_truth.i32 - 10,000 x 1,000 int32 one-based IDs in the official layout; only the
  first --collection-queries rows are filled (top-1000), the rest are zero. It feeds the
  harness's collection search pass, which reads only those rows.
* {val,test}.queries.f32 and {val,test}.gold.u64 - preregistered query rows and their
  top-30 IDs (u64, one-based) for the selection-frontier replay.
* manifest.json - inputs, parameters and SHA-256 of every output.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DIM = 768
QUERIES = 10_000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def top(scores, k):
    """Indices of the k best scores, descending, ties by smaller index."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    out = np.empty_like(part)
    for row in range(scores.shape[0]):
        cand = part[row]
        # Include every index tied with the k-th score so the tie rule is exact.
        kth = scores[row, cand].min()
        cand = np.union1d(cand, np.flatnonzero(scores[row] == kth))
        order = np.lexsort((cand, -scores[row, cand]))
        out[row] = cand[order][:k]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--collection-queries", type=int, default=100)
    p.add_argument("--val", default="900:932")
    p.add_argument("--test", default="9000:9256")
    p.add_argument("--chunk", type=int, default=50_000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    corpus = np.memmap(a.corpus, dtype="<f2", mode="r")
    total = corpus.size // DIM
    assert corpus.size == total * DIM and 0 < a.rows <= total, (corpus.size, a.rows)
    corpus = corpus.reshape(total, DIM)[: a.rows]
    queries = np.fromfile(a.queries, dtype="<f4").reshape(QUERIES, DIM)

    slices = {"val": a.val, "test": a.test}
    ranges = {k: tuple(int(x) for x in v.split(":")) for k, v in slices.items()}
    wanted = sorted(set(range(a.collection_queries)).union(
        *[range(lo, hi) for lo, hi in ranges.values()]))
    q = queries[wanted].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    scores = np.empty((len(wanted), a.rows), dtype=np.float64)
    for lo in range(0, a.rows, a.chunk):
        block = np.asarray(corpus[lo: lo + a.chunk], dtype=np.float64)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        scores[:, lo: lo + block.shape[0]] = q @ block.T
    position = {row: i for i, row in enumerate(wanted)}

    gt = np.zeros((QUERIES, 1000), dtype="<i4")
    head = top(scores[[position[r] for r in range(a.collection_queries)]], 1000)
    gt[: a.collection_queries, : head.shape[1]] = head + 1
    gt.tofile(a.out / "ground_truth.i32")
    for name, (lo, hi) in ranges.items():
        queries[lo:hi].astype("<f4").tofile(a.out / f"{name}.queries.f32")
        gold = top(scores[[position[r] for r in range(lo, hi)]], 30) + 1
        gold.astype("<u8").tofile(a.out / f"{name}.gold.u64")

    manifest = {
        "corpus": str(a.corpus), "corpus_rows_used": a.rows, "queries": str(a.queries),
        "score": "float64 cosine of stored Float16 rows vs float32 queries; ties by smaller ID",
        "ids": "one-based row index", "collection_queries": a.collection_queries,
        "slices": slices,
        "outputs": {f.name: sha256(f) for f in sorted(a.out.iterdir()) if f.is_file()},
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["outputs"], indent=1))


if __name__ == "__main__":
    main()
