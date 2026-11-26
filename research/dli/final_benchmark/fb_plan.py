"""Final-benchmark workload planner (PREREGISTRATION.md §2, implementation item I1).

    fb_plan.py --corpus F16 --rows N --scenario S --seed SEED --out DIR [--panels corpus|official --queries F32]

The batch streams are those of `emma_path/phase3_plan.py`:
* insert-only and the uniform delete scenarios: 10% initial, then 20 batches;
* shift: spherical k-means chain, 1% holdouts, last cluster withheld.

With `--panels official` and the Phase 3 panel sizes, the planner reproduces a Phase 3 plan byte for byte. That is the
regression check.

What is new:
* **Corpus-row panels** (`--panels corpus`, uniform scenarios). Queries are corpus rows at or above `--query-floor`
  (default 1,000,000), never inserted at N ≤ floor. They are drawn without replacement with `--panel-seed` (default
  20261203): test, then val, then calib. The same panels serve every plan seed.
* **Two golds per (batch, set):**
  - `<set>.gold.u64`, the **source gold** of Phase 3: Float64 cosine of the raw Float16 rows against Float32
    queries, ties by smaller ID;
  - `<set>.sgold.u64`, the **stored gold** (G1): each row normalised in f32 and cast to f16, as Qdrant stores Cosine
    vectors, scored in Float64 against the f32-normalised query, ties by smaller ID.
* **G1 tie sets.** `<set>.s30.f64` holds the stored-gold rank-30 score per query. `<set>.tie.u64` and
  `<set>.tie_off.u64` (CSR offsets, length n + 1) hold every live ID with stored score ≥ s30 − ε, in (−score, id)
  order. ε = 768 · 2⁻²⁴ bounds the f32 accumulation error of a dot product of two unit vectors. Recall@30 under G1 =
  min(30, |returned ∩ tie set|) / 30.

Point id = corpus row + 1 throughout.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p3plan", HERE.parent / "emma_path" / "phase3_plan.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

DIM, K, QUERIES = P.DIM, P.K, P.QUERIES
EPS = DIM * 2.0 ** -24
QUERY_FLOOR = 1_000_000


def stored_rows(rows_f16):
    """Qdrant's Cosine storage of Float16 rows: f32 normalise, cast to f16 (returned as f16)."""
    x = np.asarray(rows_f16, dtype=np.float32)
    n = np.sqrt(np.sum(x * x, axis=1, dtype=np.float32, keepdims=True))
    return (x / n).astype(np.float16)


def stored_queries(q):
    q = np.asarray(q, dtype=np.float32)
    return (q / np.sqrt(np.sum(q * q, axis=1, dtype=np.float32, keepdims=True))).astype(np.float64)


def stored_top_ties(corpus, live_rows, queries, eps=EPS, chunk=65_536):
    """Stored-gold top-K, rank-K score and eps tie set per query, over `live_rows`.

    `live_rows` are 0-based corpus rows, and the returned ids are rows. A running per-query bound,
    max(current K-th best, chunk K-th best) - eps, never exceeds the final s30 - eps, so pruning below it
    loses no tie-set member."""
    q = stored_queries(queries)
    nq = len(q)
    live_rows = np.sort(np.asarray(live_rows, dtype=np.int64))
    best_s = [np.empty(0) for _ in range(nq)]
    best_r = [np.empty(0, dtype=np.int64) for _ in range(nq)]
    thr = np.full(nq, -np.inf)
    for lo in range(0, len(live_rows), chunk):
        rows = live_rows[lo: lo + chunk]
        s = q @ stored_rows(corpus[rows]).astype(np.float64).T
        kk = min(K, s.shape[1])
        kth = -np.partition(-s, kk - 1, axis=1)[:, kk - 1] if s.shape[1] >= K else np.full(nq, -np.inf)
        bound = np.maximum(thr, kth) - eps
        qi, cj = np.nonzero(s >= bound[:, None])
        cuts = np.searchsorted(qi, np.arange(nq + 1))
        for i in range(nq):
            sl = slice(cuts[i], cuts[i + 1])
            if sl.start == sl.stop:
                continue
            cs = np.concatenate([best_s[i], s[i, cj[sl]]])
            cr = np.concatenate([best_r[i], rows[cj[sl]]])
            if len(cs) >= K:
                t = -np.partition(-cs, K - 1)[K - 1]
                thr[i] = t
                keep = cs >= t - eps
                cs, cr = cs[keep], cr[keep]
            best_s[i], best_r[i] = cs, cr
    top, s30, ties, offs = [], np.empty(nq), [], [0]
    for i in range(nq):
        order = np.lexsort((best_r[i], -best_s[i]))
        cs, cr = best_s[i][order], best_r[i][order]
        assert len(cs) >= K, "fewer live rows than K"
        top.append(cr[:K])
        s30[i] = cs[K - 1]
        t = cr[cs >= cs[K - 1] - eps]
        ties.append(t)
        offs.append(offs[-1] + len(t))
    return np.stack(top), s30, np.concatenate(ties), np.asarray(offs, dtype=np.int64)


def uniform_batches(n, nb, seed, scenario):
    workload, _, method = scenario.partition("_")
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
    meta = {"initial": initial, "per_batch_inserts": x, "ratio": ratio, "method": method or None}
    return out, meta


def corpus_panels(full_corpus, floor, seed, sizes):
    """Corpus rows >= floor, without replacement: test, val, calib in that order."""
    total = len(full_corpus)
    need = sum(sizes.values())
    assert total - floor >= need, "not enough rows above the query floor"
    pick = np.random.default_rng(seed).choice(np.arange(floor, total), need, replace=False)
    out, at = {}, 0
    for name, m in sizes.items():
        out[name] = pick[at: at + m]
        at += m
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--queries", type=Path, help="official 10,000-query file (--panels official)")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--scenario", required=True, choices=["insert_only", "constant_r", "constant_o", "medium_r",
                                                          "medium_o", "heavy_r", "heavy_o", "shift"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batches", type=int, default=P.BATCHES)
    p.add_argument("--panels", choices=["corpus", "official"], default="corpus")
    p.add_argument("--panel-seed", type=int, default=20261203)
    p.add_argument("--query-floor", type=int, default=QUERY_FLOOR)
    p.add_argument("--test-queries", type=int, default=1000)
    p.add_argument("--val-queries", type=int, default=64)
    p.add_argument("--calib-queries", type=int, default=2000)
    p.add_argument("--no-stored-gold", action="store_true", help="source gold only (regression against Phase 3)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    mm = np.memmap(a.corpus, dtype="<f2", mode="r")
    total = mm.size // DIM
    assert mm.size == total * DIM and 0 < a.rows <= total
    full = mm.reshape(total, DIM)
    corpus = full[: a.rows]
    n, nb = a.rows, a.batches
    tq, vq = a.test_queries, a.val_queries
    sets_static, query_rows = {}, {}
    batches = []

    def write_queries(name, vectors):
        path = a.out / f"{name}.queries.f32"
        np.asarray(vectors, dtype="<f4").tofile(path)
        return path

    if a.scenario == "shift":
        # Identical to phase3_plan.py, with the validation size as a parameter (Phase 3: 32).
        rng = np.random.default_rng(a.seed)
        labels, centers = P.spherical_kmeans(corpus, 10, 25, rng)
        order = P.chain(centers, rng)
        ood_cluster, inserted_clusters = order[-1], order[:-1]
        holdout, insert_rows = {}, []
        for c in inserted_clusters:
            rows = np.flatnonzero(labels == c)
            need = tq + vq
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
        ood_q = sample(ood_rows, tq + vq)
        mq_q = sample(holdout[inserted_clusters[-1]], tq + vq)
        query_rows.update({"ood": ood_q[:tq], "ood_val": ood_q[tq:], "mq": mq_q[:tq], "mq_val": mq_q[tq:]})
        bounds = np.linspace(0, len(insert_rows), nb + 1).astype(int)
        plan_batches = [(insert_rows[bounds[i]: bounds[i + 1]], np.array([], dtype=np.int64)) for i in range(nb)]
        meta = {"clusters": {int(c): int((labels == c).sum()) for c in range(10)},
                "chain": order, "ood_cluster": int(ood_cluster),
                "holdout_sizes": {int(c): int(len(v)) for c, v in holdout.items()}}
        for name, rows in query_rows.items():
            sets_static[name] = corpus[rows].astype(np.float32)
    else:
        plan_batches, meta = uniform_batches(n, nb, a.seed, a.scenario)
        if a.panels == "official":
            assert a.queries is not None, "--panels official needs --queries"
            qfile = np.fromfile(a.queries, dtype="<f4").reshape(QUERIES, DIM)
            sets_static = {"test": qfile[9000: 9000 + tq], "val": qfile[900: 900 + vq]}
            if a.scenario == "insert_only":
                sets_static["partb"] = qfile[8000:9000]
                sets_static["calib"] = qfile[1000:3000]
        else:
            assert a.query_floor >= n, "query rows must lie above every inserted row"
            sizes = {"test": tq, "val": vq, "calib": a.calib_queries}
            query_rows = corpus_panels(full, a.query_floor, a.panel_seed, sizes)
            sets_static = {name: full[rows].astype(np.float32) for name, rows in query_rows.items()}
        meta["panels"] = a.panels

    static_paths = {name: write_queries(name, v) for name, v in sets_static.items()}
    static_queries = {name: np.fromfile(path, dtype="<f4").reshape(-1, DIM) for name, path in static_paths.items()}
    for name, rows in query_rows.items():
        (rows.astype(np.int64) + 1).astype("<u8").tofile(a.out / f"{name}.query_ids.u64")
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
        count, digest = P.live_digest(live_rows + 1)
        sets = {}
        queries = dict(static_queries)
        if a.scenario == "shift":
            for c in np.unique(labels[ins]):
                inserted_so_far[int(c)] = inserted_so_far.get(int(c), 0) + int((labels[ins] == c).sum())
            rng_b = np.random.default_rng(a.seed + 1000 + b)
            weights = np.array([inserted_so_far[c] for c in inserted_so_far], dtype=float)
            take = rng_b.multinomial(tq + vq, weights / weights.sum())
            ids_rows = np.concatenate([
                rng_b.choice(holdout[c], min(t, len(holdout[c])), replace=False)
                for c, t in zip(inserted_so_far, take)])
            ids_rows = rng_b.permutation(ids_rows)
            vecs = corpus[ids_rows].astype(np.float32)
            queries["ids"], queries["ids_val"] = vecs[:tq], vecs[tq:]
            write_queries(f"batch_{b:02d}/ids", queries["ids"])
            write_queries(f"batch_{b:02d}/ids_val", queries["ids_val"])
            (ids_rows[:tq].astype(np.int64) + 1).astype("<u8").tofile(d / "ids.query_ids.u64")
            (ids_rows[tq:].astype(np.int64) + 1).astype("<u8").tofile(d / "ids_val.query_ids.u64")
        for name, q in queries.items():
            gold = P.exact_top(corpus, live_rows, q) + 1
            gpath = d / f"{name}.gold.u64"
            gold.astype("<u8").tofile(gpath)
            qpath = d / f"{name}.queries.f32" if name in ("ids", "ids_val") else static_paths[name]
            entry = {"queries": str(qpath.relative_to(a.out)), "gold": str(gpath.relative_to(a.out)), "n": int(len(q))}
            if not a.no_stored_gold:
                top, s30, tie, off = stored_top_ties(corpus, live_rows, q)
                for suffix, arr, dt in (("sgold", top + 1, "<u8"), ("s30", s30, "<f8"),
                                        ("tie", tie + 1, "<u8"), ("tie_off", off, "<u8")):
                    np.asarray(arr).astype(dt).tofile(d / f"{name}.{suffix}.{dt[1:].replace('u8', 'u64').replace('f8', 'f64')}")
                entry.update({"stored_gold": f"batch_{b:02d}/{name}.sgold.u64", "s30": f"batch_{b:02d}/{name}.s30.f64",
                              "tie": f"batch_{b:02d}/{name}.tie.u64", "tie_offsets": f"batch_{b:02d}/{name}.tie_off.u64",
                              "tie_max": int(np.diff(off).max())})
            sets[name] = entry
        batches.append({"index": b, "inserts": int(len(ins)), "deletes": int(len(dels)),
                        "live_count": count, "live_sha256": digest, "sets": sets})
        print(json.dumps({"batch": b, "live": count, "ins": int(len(ins)), "del": int(len(dels))}), flush=True)

    manifest = {"scenario": a.scenario, "rows": n, "seed": a.seed, "batches": batches, "meta": meta,
                "corpus": str(a.corpus), "queries": str(a.queries) if a.queries else None,
                "gold_rule": "float64 cosine, Float16 rows vs float32 queries, ties by smaller id, top-30 over live set",
                "files": {str(f.relative_to(a.out)): P.sha256_file(f) for f in sorted(a.out.rglob("*")) if f.is_file()}}
    if a.panels == "corpus" or a.scenario == "shift" or not a.no_stored_gold:
        manifest["planner"] = "final_benchmark/fb_plan.py"
        manifest["panels"] = {"kind": a.panels, "panel_seed": a.panel_seed, "query_floor": a.query_floor,
                              "sizes": {"test": tq, "val": vq, "calib": a.calib_queries}}
    if not a.no_stored_gold:
        manifest["stored_gold_rule"] = ("G1: rows f32-normalised then cast to f16 (Qdrant Cosine storage), float64 dot with "
                                        "the f32-normalised query, ties by smaller id; tie set = live ids with stored score "
                                        f">= s30 - eps, eps = 768*2^-24 = {EPS!r}")
        manifest["epsilon"] = EPS
    (a.out / "plan.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
