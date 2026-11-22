"""Phase 3 workload planner (PHASE3_PREREGISTRATION.md §2, §7 M5–M7, §10.2).

Generates, for one scenario, every batch's inserts, deletes, expected live set (count and
SHA-256) and the exact top-30 gold of each query set over that live set. The Rust driver
(`dli_phase3_workload`) executes the plan and checks its live set against the ledger. All
arms of a scenario share one plan. Point id = corpus row + 1.

Scenarios: insert_only, {heavy,medium,constant}_{r,o}, shift.
* Uniform scenarios insert rows 0..N-1 in ID order: an initial 10% batch (no deletes), then
  20 batches of x = 0.9·N/20 inserts followed by x/10, x/2 or x deletes; r = random
  (PCG64, seed + batch), o = oldest (smallest live IDs).
* shift: spherical k-means (10 clusters, 25 iterations, seed) on the N rows; clusters chained
  from a seeded random start, each next = nearest unused centroid (cosine). The last chain
  cluster is withheld entirely (OOD queries); 1% of each inserted cluster is held out
  (never inserted). Inserted rows are split into 20 equal batches, cluster by cluster in chain
  order, ID order within a cluster.

Query sets (rows of the official 10,000-query file unless noted):
* uniform: test 9000–9999, val 900–931; insert_only also partb 8000–8999 and calib 1000–2999.
* shift (corpus rows, never inserted): ids (resampled per batch from holdouts of inserted
  clusters, proportional to inserted counts), ood (withheld cluster), mq (holdout of the last
  inserted cluster); each 1,000 test + 32 val.

Gold: exact top-30 by float64 cosine of stored Float16 rows against float32 queries, ties by
smaller ID, over the live set after the batch.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DIM = 768
QUERIES = 10_000
K = 30
BATCHES = 20


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def live_digest(live_ids):
    ids = np.sort(np.asarray(live_ids, dtype="<u4"))
    return len(ids), sha256_bytes(ids.tobytes())


def normalized(rows):
    x = np.asarray(rows, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def exact_top(corpus, live_rows, queries, chunk=65_536):
    """Top-K corpus row indices (of `live_rows`) per query, descending, ties by smaller row.

    Per chunk, every score >= the chunk's K-th best is kept (so no tie at the boundary is
    lost) and merged with the running best by an exact (-score, row) sort.
    """
    q = normalized(queries)
    best_s = [np.empty(0) for _ in range(len(q))]
    best_r = [np.empty(0, dtype=np.int64) for _ in range(len(q))]
    live_rows = np.sort(np.asarray(live_rows, dtype=np.int64))
    for lo in range(0, len(live_rows), chunk):
        rows = live_rows[lo: lo + chunk]
        s = q @ normalized(corpus[rows]).T
        kk = min(K, s.shape[1])
        kth = -np.partition(-s, kk - 1, axis=1)[:, kk - 1]
        for i in range(len(q)):
            keep = np.flatnonzero(s[i] >= kth[i])
            cs = np.concatenate([best_s[i], s[i, keep]])
            cr = np.concatenate([best_r[i], rows[keep]])
            order = np.lexsort((cr, -cs))[:K]
            best_s[i], best_r[i] = cs[order], cr[order]
    return np.stack(best_r)


def spherical_kmeans(x, k, iters, rng):
    xn = normalized(x)
    centers = xn[rng.choice(len(xn), k, replace=False)]
    for _ in range(iters):
        labels = np.argmax(xn @ centers.T, axis=1)
        for c in range(k):
            members = xn[labels == c]
            if len(members):
                m = members.sum(axis=0)
                centers[c] = m / np.linalg.norm(m)
    return np.argmax(xn @ centers.T, axis=1), centers


def chain(centers, rng):
    order = [int(rng.integers(len(centers)))]
    while len(order) < len(centers):
        last = centers[order[-1]]
        rest = [c for c in range(len(centers)) if c not in order]
        order.append(max(rest, key=lambda c: float(centers[c] @ last)))
    return order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--queries", type=Path, help="official 10,000-query f32 file (uniform scenarios)")
    p.add_argument("--rows", type=int, required=True, help="N")
    p.add_argument("--scenario", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--batches", type=int, default=BATCHES)
    p.add_argument("--test-queries", type=int, default=1000)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    corpus = np.memmap(a.corpus, dtype="<f2", mode="r")
    total = corpus.size // DIM
    assert corpus.size == total * DIM and 0 < a.rows <= total
    corpus = corpus.reshape(total, DIM)[: a.rows]
    n, nb = a.rows, a.batches
    sets_static = {}
    batches = []

    def write_queries(name, rows_or_vectors):
        path = a.out / f"{name}.queries.f32"
        np.asarray(rows_or_vectors, dtype="<f4").tofile(path)
        return path

    if a.scenario == "shift":
        rng = np.random.default_rng(a.seed)
        labels, centers = spherical_kmeans(corpus, 10, 25, rng)
        order = chain(centers, rng)
        ood_cluster, inserted_clusters = order[-1], order[:-1]
        holdout, insert_rows = {}, []
        for c in inserted_clusters:
            rows = np.flatnonzero(labels == c)
            need = a.test_queries + 32
            h = max(int(round(0.01 * len(rows))), min(need, len(rows) // 2))
            pick = np.sort(rng.choice(rows, h, replace=False))
            holdout[c] = pick
            insert_rows.append(np.setdiff1d(rows, pick))
        insert_rows = np.concatenate(insert_rows)
        ood_rows = np.flatnonzero(labels == ood_cluster)
        held = np.concatenate(list(holdout.values()))
        assert not np.intersect1d(insert_rows, held).size, "holdout row inserted"
        assert not np.intersect1d(insert_rows, ood_rows).size, "OOD row inserted"
        np.sort(held).astype("<u4").tofile(a.out / "holdout_rows.u32")
        sample = lambda rows, m: np.sort(rng.choice(rows, min(m, len(rows)), replace=False))
        ood_q = sample(ood_rows, a.test_queries + 32)
        mq_q = sample(holdout[inserted_clusters[-1]], a.test_queries + 32)
        tq = a.test_queries
        sets_static.update({"ood": ood_q[:tq], "ood_val": ood_q[tq:], "mq": mq_q[:tq], "mq_val": mq_q[tq:]})
        bounds = np.linspace(0, len(insert_rows), nb + 1).astype(int)
        plan_batches = [(insert_rows[bounds[i]: bounds[i + 1]], np.array([], dtype=np.int64)) for i in range(nb)]
        meta = {"clusters": {int(c): int((labels == c).sum()) for c in range(10)},
                "chain": order, "ood_cluster": int(ood_cluster),
                "holdout_sizes": {int(c): int(len(v)) for c, v in holdout.items()}}
    else:
        assert a.queries is not None
        workload, _, method = a.scenario.partition("_")
        ratio = {"insert": 0, "heavy": 10, "medium": 2, "constant": 1}[workload]
        initial = n // 10
        x = (n - initial) // nb
        plan_batches = [(np.arange(initial), np.array([], dtype=np.int64))]
        qfile = np.fromfile(a.queries, dtype="<f4").reshape(QUERIES, DIM)
        sets_static = {"test": qfile[9000: 9000 + a.test_queries], "val": qfile[900:932]}
        if a.scenario == "insert_only":
            sets_static["partb"] = qfile[8000:9000]
            sets_static["calib"] = qfile[1000:3000]
        meta = {"initial": initial, "per_batch_inserts": x, "ratio": ratio, "method": method or None}
        nxt = initial
        live = set(range(initial))
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
                dels = np.sort(np.random.default_rng(a.seed + b).choice(pool, d, replace=False))
            live.difference_update(dels.tolist())
            plan_batches.append((ins, dels))

    # Static query files: vectors for uniform sets; corpus rows for shift sets.
    static_paths = {}
    for name, v in sets_static.items():
        vectors = corpus[v].astype(np.float32) if a.scenario == "shift" else v
        static_paths[name] = write_queries(name, vectors)

    static_queries = {name: np.fromfile(path, dtype="<f4").reshape(-1, DIM) for name, path in static_paths.items()}
    live = np.zeros(n, dtype=bool)
    inserted_so_far = {}
    for b, (ins, dels) in enumerate(plan_batches):
        d = a.out / f"batch_{b:02d}"
        d.mkdir()
        (ins.astype(np.int64) + 1).astype("<u4").tofile(d / "insert.u32")
        (dels.astype(np.int64) + 1).astype("<u4").tofile(d / "delete.u32")
        assert not live[ins].any(), "re-insert of a live row"
        live[ins] = True
        assert live[dels].all(), "delete of a non-live row"
        live[dels] = False
        live_rows = np.flatnonzero(live)
        count, digest = live_digest(live_rows + 1)
        sets = {}
        queries = dict(static_queries)
        if a.scenario == "shift":
            # IDS: resample from holdouts of clusters inserted so far, proportional to counts.
            for c in np.unique(labels[ins]):
                inserted_so_far[int(c)] = inserted_so_far.get(int(c), 0) + int((labels[ins] == c).sum())
            rng_b = np.random.default_rng(a.seed + 1000 + b)
            weights = np.array([inserted_so_far[c] for c in inserted_so_far], dtype=float)
            take = rng_b.multinomial(a.test_queries + 32, weights / weights.sum())
            ids_rows = np.concatenate([
                rng_b.choice(holdout[c], min(t, len(holdout[c])), replace=False)
                for c, t in zip(inserted_so_far, take)])
            ids_rows = rng_b.permutation(ids_rows)
            vecs = corpus[ids_rows].astype(np.float32)
            queries["ids"], queries["ids_val"] = vecs[: a.test_queries], vecs[a.test_queries:]
            write_queries(f"batch_{b:02d}/ids", queries["ids"])
            write_queries(f"batch_{b:02d}/ids_val", queries["ids_val"])
        for name, q in queries.items():
            gold = exact_top(corpus, live_rows, q) + 1
            gpath = d / f"{name}.gold.u64"
            gold.astype("<u8").tofile(gpath)
            qpath = d / f"{name}.queries.f32" if name in ("ids", "ids_val") else static_paths[name]
            sets[name] = {"queries": str(qpath.relative_to(a.out)), "gold": str(gpath.relative_to(a.out)),
                          "n": int(len(q))}
        batches.append({"index": b, "inserts": int(len(ins)), "deletes": int(len(dels)),
                        "live_count": count, "live_sha256": digest, "sets": sets})
        print(json.dumps({"batch": b, "live": count, "ins": int(len(ins)), "del": int(len(dels))}), flush=True)

    manifest = {"scenario": a.scenario, "rows": n, "seed": a.seed, "batches": batches, "meta": meta,
                "corpus": str(a.corpus), "queries": str(a.queries) if a.queries else None,
                "gold_rule": "float64 cosine, Float16 rows vs float32 queries, ties by smaller id, top-30 over live set",
                "files": {str(f.relative_to(a.out)): sha256_file(f) for f in sorted(a.out.rglob("*")) if f.is_file()}}
    (a.out / "plan.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
