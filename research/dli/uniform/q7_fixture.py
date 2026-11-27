"""Q7 shared filtered-search fixture (PostgreSQL FILTER_FIXTURE_SPEC.md at 3ec4041, pinned in PINS.md).

    q7_fixture.py --corpus SISAP23_10M.f16le --out DIR

Writes, and checks against every value the spec publishes:
* vectors.f16: zero-based corpus rows [2,600,000, 2,660,000), local ids 1..60,000 in row order (SHA-256 c1f24a8d…);
* filter_rank.u32: little-endian uint32 by ascending local id (SHA-256 3f0c875c…). The rank is the position in the
  order (SplitMix64(id + 20261127), id), and the first five ranked ids are 23712, 11434, 46242, 30672 and 28694;
* queries.f16: corpus rows [2,660,000, 2,660,032), the raw 49,152-byte slice (SHA-256 fb72bd7f…);
* predicates.json: filter_rank <= 60, 600, 6,000 and 30,000 (0.1%, 1%, 10% and 50%), with each predicate's id count.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DIM = 768
CORPUS_SHA = "f5868d6e002a6e73aea47491ee040a70a151406fe9e4552292428e2bfe0616e5"
ROW0, N = 2_600_000, 60_000
Q0, NQ = 2_660_000, 32
SLICE_SHA = "c1f24a8d139ea197c87a06c8b3d184f6ab6248081b013ebce5eb6085772ada0d"
RANK_SHA = "3f0c875cd0fee5d8e4ff70861a418cd8c57ed7125d6720e58e84facada399ea5"
QUERY_SHA = "fb72bd7fa7de94ba87f75b82002e0f332f0e6ef33e8a4b786886d5775ccdb43f"
FIRST_FIVE = [23712, 11434, 46242, 30672, 28694]
PREDICATES = {"0.1%": 60, "1%": 600, "10%": 6_000, "50%": 30_000}
SEED = 20261127
MASK = (1 << 64) - 1


def splitmix64(v):
    x = (v + SEED + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def filter_rank(n=N):
    ids = list(range(1, n + 1))
    order = sorted(ids, key=lambda i: (splitmix64(i), i))
    rank = np.zeros(n, dtype="<u4")
    for r, i in enumerate(order, start=1):
        rank[i - 1] = r
    return order, rank


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--corpus", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--skip-corpus-hash", action="store_true", help="the corpus hash was verified separately")
    args = a.parse_args()
    if not args.skip_corpus_hash:
        h = hashlib.sha256()
        with open(args.corpus, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest() == CORPUS_SHA, "corpus SHA-256"
    args.out.mkdir(parents=True, exist_ok=False)
    row = DIM * 2
    with open(args.corpus, "rb") as f:
        f.seek(ROW0 * row)
        vectors = f.read(N * row)
        f.seek(Q0 * row)
        queries = f.read(NQ * row)
    assert hashlib.sha256(vectors).hexdigest() == SLICE_SHA, "slice SHA-256"
    assert hashlib.sha256(queries).hexdigest() == QUERY_SHA, "query SHA-256"
    order, rank = filter_rank()
    assert order[:5] == FIRST_FIVE, order[:5]
    assert hashlib.sha256(rank.tobytes()).hexdigest() == RANK_SHA, "filter_rank SHA-256"
    (args.out / "vectors.f16").write_bytes(vectors)
    (args.out / "queries.f16").write_bytes(queries)
    (args.out / "filter_rank.u32").write_bytes(rank.tobytes())
    preds = {k: {"max_rank": v, "matching": int((rank <= v).sum())} for k, v in PREDICATES.items()}
    assert [p["matching"] for p in preds.values()] == [60, 600, 6_000, 30_000]
    (args.out / "predicates.json").write_text(json.dumps(
        {"spec": "PostgreSQL FILTER_FIXTURE_SPEC.md @ 3ec4041", "predicates": preds, "k": 30,
         "sha256": {"vectors.f16": SLICE_SHA, "queries.f16": QUERY_SHA, "filter_rank.u32": RANK_SHA}}, indent=1))
    print("q7 fixture verified:", json.dumps(preds))


if __name__ == "__main__":
    main()
