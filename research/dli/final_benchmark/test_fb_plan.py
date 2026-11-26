"""Known-answer tests for fb_plan.py.

1. `stored_top_ties` equals a brute-force full sort on a corpus with planted exact duplicates and
   one-ulp near-duplicates of the rank-30 neighbourhood. The test uses many small chunks, so the
   running prune bound is exercised.
2. An end-to-end tiny plan, corpus panels, constant_r: query rows lie above the floor and are
   never inserted, every tie set contains its stored gold, the CSR offsets are consistent, and the
   live digests replay.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fb_plan as F  # noqa: E402

DIM, K = F.DIM, F.K


def brute(corpus, live_rows, q, eps=F.EPS):
    s = F.stored_queries(q) @ F.stored_rows(corpus[live_rows]).astype(np.float64).T
    tops, s30s, ties = [], [], []
    for i in range(len(q)):
        o = np.lexsort((live_rows, -s[i]))
        tops.append(live_rows[o[:K]])
        s30 = s[i, o[K - 1]]
        s30s.append(s30)
        ties.append(live_rows[o][s[i, o] >= s30 - eps])
    return np.stack(tops), np.array(s30s), ties


rng = np.random.default_rng(7)
base = rng.standard_normal((400, DIM)).astype(np.float16)
q = rng.standard_normal((6, DIM)).astype(np.float32)
# Plant, for query 0, 12 exact copies of a row very close to it, plus near-duplicates of that row
# that differ by one f16 ulp in one coordinate.
near = (q[0] + 0.05 * rng.standard_normal(DIM)).astype(np.float16)
base[:12] = near
for j in range(12, 20):
    v = near.copy()
    v[j] = np.nextafter(v[j], np.float16(np.inf))
    base[j] = v
base = base[rng.permutation(len(base))]
live = np.sort(rng.choice(len(base), 350, replace=False))
for chunk in (37, 64, 1000):
    top, s30, tie, off = F.stored_top_ties(base, live, q, chunk=chunk)
    btop, bs30, bties = brute(base, live, q)
    assert np.array_equal(top, btop), ("top", chunk)
    assert np.array_equal(s30, bs30), ("s30", chunk)
    for i in range(len(q)):
        assert np.array_equal(tie[off[i]: off[i + 1]], bties[i]), ("tie", chunk, i)
        assert set(top[i]) <= set(tie[off[i]: off[i + 1]].tolist())
assert np.diff(off).max() > K, "planted ties not exercised"
print("stored_top_ties vs brute force: OK (max tie set", int(np.diff(off).max()), ")")

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    total, n = 6000, 3000
    corpus = rng.standard_normal((total, DIM)).astype("<f2")
    corpus[4000] = corpus[10]  # a query row duplicating an inserted row: a score-1.0 neighbour
    corpus.tofile(tmp / "c.f16")
    out = tmp / "plan"
    subprocess.run([sys.executable, "-I", str(HERE / "fb_plan.py"), "--corpus", str(tmp / "c.f16"), "--rows", str(n),
                    "--scenario", "constant_r", "--seed", "11", "--batches", "3", "--out", str(out),
                    "--query-floor", str(n), "--test-queries", "20", "--val-queries", "8", "--calib-queries", "10"],
                   check=True, capture_output=True)
    plan = json.loads((out / "plan.json").read_text())
    ids = {s: np.fromfile(out / f"{s}.query_ids.u64", "<u8") for s in ("test", "val", "calib")}
    allq = np.concatenate(list(ids.values()))
    assert len(set(allq.tolist())) == len(allq) and allq.min() >= n + 1, "panel rows"
    inserted = np.concatenate([np.fromfile(out / f"batch_{b:02d}" / "insert.u32", "<u4") for b in range(4)])
    assert not np.isin(allq, inserted).any(), "a query row was inserted"
    for name, rows in ids.items():
        qv = np.fromfile(out / f"{name}.queries.f32", "<f4").reshape(-1, DIM)
        assert np.array_equal(qv, corpus[rows.astype(np.int64) - 1].astype(np.float32)), "query vectors"
    live = np.zeros(n, bool)
    for b, meta in enumerate(plan["batches"]):
        d = out / f"batch_{b:02d}"
        live[np.fromfile(d / "insert.u32", "<u4").astype(np.int64) - 1] = True
        live[np.fromfile(d / "delete.u32", "<u4").astype(np.int64) - 1] = False
        assert F.P.live_digest(np.flatnonzero(live) + 1) == (meta["live_count"], meta["live_sha256"])
        for s, e in meta["sets"].items():
            sg = np.fromfile(out / e["stored_gold"], "<u8").reshape(-1, K)
            tie = np.fromfile(out / e["tie"], "<u8")
            off = np.fromfile(out / e["tie_offsets"], "<u8").astype(np.int64)
            assert len(off) == e["n"] + 1 and off[-1] == len(tie)
            for i in range(e["n"]):
                assert set(sg[i].tolist()) <= set(tie[off[i]: off[i + 1]].tolist())
            assert live[tie.astype(np.int64) - 1].all(), "tie set outside the live set"
    assert plan["epsilon"] == F.EPS and plan["panels"]["kind"] == "corpus"
print("fb_plan end-to-end tiny plan: OK")
