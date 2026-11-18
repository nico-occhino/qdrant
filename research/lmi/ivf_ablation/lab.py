"""Offline static-LMI / IVF routing laboratory (numpy only).

Subcommands
  gold   --vectors F16 --n N --queries F32 --rows a:b,... --out DIR [--official GT.i32]
         Write queries.npy (selected rows, normalized), rows.npy, gold.npy (top-30 row ids).
         With --official, gold is the official SISAP top-30 (1-based -> 0-based), else exact
         inner-product top-30 over the first N stored Float16 vectors (Float32 accumulation,
         ties -> lower id).
  run    --vectors F16 --n N --gold DIR --out DIR --B .. --teacher native|converged
         --hidden H --epochs E [--partition mlp,centroid] [--router mlp,centroid]
         Trains/loads the teacher and router (cached under --cache), assigns all N points,
         and writes per-query exact traces for every (partition, router) combination.

The algorithms mirror lib/segment/src/index/lmi_index (teacher: spherical_kmeans.rs as used by
training.rs:335 for cosine: seeded shuffle, first-B sample points as unit centres, Lloyd
iterations with dot-product assignment, means re-normalised, empty or zero-mean clusters keep
their centre; router: 768->H->B ReLU MLP, uniform(+-1/sqrt(fan_in))
init, Adam lr 1e-3, batch 256, cross-entropy on teacher labels over the sample). RNG streams
differ from Rust's StdRng, so models are statistically, not bitwise, equivalent.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

D = 768
K = 30
F32 = np.float32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def vectors(path, n):
    total = os.path.getsize(path) // (2 * D)
    return np.memmap(path, dtype=np.float16, mode="r", shape=(total, D))[:n]


def chunks(n, size):
    for s in range(0, n, size):
        yield s, min(n, s + size)


# ------------------------------------------------------------------ gold
def cmd_gold(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    q_all = np.fromfile(a.queries, dtype=F32).reshape(-1, D)
    rows = np.concatenate([np.arange(int(x.split(":")[0]), int(x.split(":")[1])) for x in a.rows.split(",")])
    Q = q_all[rows]
    norms = np.linalg.norm(Q, axis=1)
    Q = (Q / norms[:, None]).astype(F32)
    if a.official:
        gt = np.fromfile(a.official, dtype=np.int32).reshape(q_all.shape[0], -1)
        gold = gt[rows, :K].astype(np.int64) - 1
        src = "official top-30 (1-based ids -> 0-based)"
    else:
        X = vectors(a.vectors, a.n)
        best_s = np.full((len(Q), K), -np.inf, F32); best_i = np.zeros((len(Q), K), np.int64)
        for s, e in chunks(a.n, 200_000):
            S = Q @ X[s:e].astype(F32).T                       # (nq, chunk)
            part = np.argpartition(-S, K, axis=1)[:, :K]
            cs = np.take_along_axis(S, part, 1); ci = part + s
            ms = np.concatenate([best_s, cs], 1); mi = np.concatenate([best_i, ci], 1)
            # sort by (-score, id) and keep K
            o = np.lexsort((mi, -ms), axis=1)[:, :K]
            best_s = np.take_along_axis(ms, o, 1); best_i = np.take_along_axis(mi, o, 1)
            log("gold", e)
        gold = best_i
        src = f"exact inner-product top-30 over first {a.n} Float16 rows, Float32 accumulation, ties->lower id"
    np.save(out / "queries.npy", Q); np.save(out / "rows.npy", rows); np.save(out / "gold.npy", gold)
    meta = {"rows": a.rows, "n": a.n, "gold_source": src, "query_norm_min": float(norms.min()),
            "query_norm_max": float(norms.max()), "vectors": a.vectors, "queries": a.queries}
    (out / "gold.json").write_text(json.dumps(meta, indent=1))
    log("wrote", out, meta)


# --------------------------------------------------------------- teacher
def sample_rows(n, s, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(s, n), replace=False))


def assign(Xf, C, cn=None):
    """spherical assignment: argmax x.c over unit centres, first index on ties
    (spherical_kmeans.rs::assign_qdrant)."""
    return np.argmax(Xf @ C.T, axis=1)


def unit(C):
    n = np.linalg.norm(C, axis=1, keepdims=True)
    return C / np.where(n == 0, 1.0, n)


def kmeans(Xs, B, iters, seed):
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(Xs))
    C = unit(Xs[order[:B]].astype(np.float64))
    labels = np.full(len(Xs), -1)
    for it in range(iters):
        Cf = C.astype(F32); cn = (Cf * Cf).sum(1)
        new = np.concatenate([assign(Xs[s:e], Cf, cn) for s, e in chunks(len(Xs), 65536)])
        changed = int((new != labels).sum()); labels = new
        sizes = np.bincount(labels, minlength=B)
        nz = sizes > 0
        o = np.argsort(labels, kind="stable")
        starts = np.searchsorted(labels[o], np.nonzero(nz)[0])
        sums = np.add.reduceat(Xs[o], starts, axis=0, dtype=np.float64)   # Float64 sums, as native
        means = sums / sizes[nz, None]
        norms = np.linalg.norm(means, axis=1)
        ok = norms > 0                                   # zero mean has no direction: keep centre
        idx = np.nonzero(nz)[0][ok]
        C[idx] = means[ok] / norms[ok, None]
        log(f"kmeans it={it + 1} changed={changed} empty={int((~nz).sum())}")
        if changed == 0:
            break
    Cf = C.astype(F32); cn = (Cf * Cf).sum(1)
    final = np.concatenate([assign(Xs[s:e], Cf, cn) for s, e in chunks(len(Xs), 65536)])
    return Cf, final


# ------------------------------------------------------------------ MLP
class MLP:
    def __init__(self, B, H, seed):
        rng = np.random.default_rng(seed + 2)
        self.H = H
        def lin(i, o):
            b = 1.0 / np.sqrt(i)
            return rng.uniform(-b, b, (o, i)).astype(F32), rng.uniform(-b, b, o).astype(F32)
        if H:
            self.p = list(lin(D, H)) + list(lin(H, B))
        else:
            self.p = list(lin(D, B))

    def logits(self, x):
        if self.H:
            h = np.maximum(x @ self.p[0].T + self.p[1], 0)
            return h @ self.p[2].T + self.p[3]
        return x @ self.p[0].T + self.p[1]

    def train(self, X, y, epochs, seed, batch=256, lr=1e-3):
        rng = np.random.default_rng(seed + 3)
        m = [np.zeros_like(p) for p in self.p]; v = [np.zeros_like(p) for p in self.p]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        hist = []
        for ep in range(epochs):
            order = rng.permutation(len(X)); tot = 0.0; acc = 0
            for s, e in chunks(len(X), batch):
                idx = order[s:e]; x = X[idx]; yy = y[idx]; n = len(idx)
                if self.H:
                    z1 = x @ self.p[0].T + self.p[1]; h = np.maximum(z1, 0)
                    z = h @ self.p[2].T + self.p[3]
                else:
                    z = x @ self.p[0].T + self.p[1]
                z = z - z.max(1, keepdims=True); ez = np.exp(z); P = ez / ez.sum(1, keepdims=True)
                tot += float(-np.log(P[np.arange(n), yy] + 1e-30).sum()); acc += int((P.argmax(1) == yy).sum())
                g = P; g[np.arange(n), yy] -= 1.0; g /= n
                if self.H:
                    gW2 = g.T @ h; gb2 = g.sum(0); gh = g @ self.p[2]; gh[z1 <= 0] = 0
                    grads = [gh.T @ x, gh.sum(0), gW2, gb2]
                else:
                    grads = [g.T @ x, g.sum(0)]
                t += 1
                for i, gr in enumerate(grads):
                    m[i] = b1 * m[i] + (1 - b1) * gr; v[i] = b2 * v[i] + (1 - b2) * gr * gr
                    mh = m[i] / (1 - b1 ** t); vh = v[i] / (1 - b2 ** t)
                    self.p[i] -= (lr * mh / (np.sqrt(vh) + eps)).astype(F32)
            hist.append((ep + 1, tot / len(X), acc / len(X)))
            log(f"epoch {ep + 1}/{epochs} loss={tot / len(X):.4f} acc={acc / len(X):.4f}")
        return hist

    def macs(self, B):
        return D * self.H + self.H * B if self.H else D * B

    def nbytes(self):
        return int(sum(p.nbytes for p in self.p))


# --------------------------------------------------------------- traces
def budgets_grid(sizes_mean, n):
    return np.unique(np.geomspace(max(1, sizes_mean), 0.25 * n, 64).astype(np.int64))


def nprobe_grid(B):
    return np.unique(np.geomspace(1, min(B, 512), 40).astype(np.int64))


def traces(order, labels, sizes, gold, budgets, probes):
    """order: (nq, B) bucket ids by descending router preference. Returns dict of int arrays."""
    nq, B = order.shape
    inv = np.empty_like(order); inv[np.arange(nq)[:, None], order] = np.arange(B)[None, :]
    gb = labels[gold]                                     # (nq, K) bucket of each gold id
    pos = np.take_along_axis(inv, gb, 1)                  # rank position of each gold's bucket
    cs = np.cumsum(sizes[order], axis=1)                  # cumulative work along the order
    out = {}
    # budget stopping: take buckets until cumulative size >= w
    nsel = np.stack([np.minimum(np.argmax(cs >= w, axis=1) + 1 + (cs[:, -1] < w) * B, B) for w in budgets], 1)
    out["b_work"] = np.take_along_axis(cs, nsel - 1, 1).astype(np.int64)
    out["b_nsel"] = nsel.astype(np.int32)
    out["b_r10"] = (pos[:, None, :10] < nsel[:, :, None]).sum(2).astype(np.int16)
    out["b_r30"] = (pos[:, None, :] < nsel[:, :, None]).sum(2).astype(np.int16)
    P = np.minimum(probes, B)
    out["p_work"] = cs[:, P - 1].astype(np.int64)
    out["p_r10"] = (pos[:, None, :10] < P[None, :, None]).sum(2).astype(np.int16)
    out["p_r30"] = (pos[:, None, :] < P[None, :, None]).sum(2).astype(np.int16)
    # oracle (diagnostic): cost of the distinct gold buckets; router-rank cost to cover gold
    for k in (10, 30):
        oc = np.array([sizes[np.unique(gb[i, :k])].sum() for i in range(nq)])
        deep = pos[:, :k].max(1)
        out[f"oracle_cost_top{k}"] = oc
        out[f"rank_cost_top{k}"] = cs[np.arange(nq), deep]
    return out


def router_order(kind, Q, C, cn, mlp, batch=512):
    orders = []
    for s, e in chunks(len(Q), batch):
        q = Q[s:e]
        score = (q @ C.T) if kind == "centroid" else mlp.logits(q)
        orders.append(np.argsort(-score, axis=1, kind="stable"))
    return np.concatenate(orders)


def bucket_stats(sizes):
    s = sizes.astype(np.float64)
    return {"B": int(len(s)), "empty": int((s == 0).sum()), "max": int(s.max()), "min": int(s.min()),
            "mean": float(s.mean()), "median": float(np.median(s)), "cv": float(s.std() / s.mean()),
            "top10pct_share": float(np.sort(s)[-max(1, len(s) // 10):].sum() / s.sum()),
            # expected work of one random gold-weighted bucket (size-biased mean)
            "size_biased_mean": float((s * s).sum() / s.sum())}


def cmd_run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    X = vectors(a.vectors, a.n)
    S = a.sample or (32768 if a.teacher == "native" else max(32768, 64 * a.B))
    iters = a.iters or (5 if a.teacher == "native" else 20)
    meta = {"argv": sys.argv, "B": a.B, "teacher": a.teacher, "sample": S, "iters": iters, "hidden": a.hidden,
            "epochs": a.epochs, "seed": a.seed, "n": a.n}
    timing = {}
    tkey = hashlib.sha256(json.dumps(["spherical-v1", a.vectors, a.n, a.B, S, iters, a.seed]).encode()).hexdigest()[:16]
    tfile = cache / f"teacher-{tkey}.npz"
    sidx = sample_rows(a.n, S, a.seed)
    Xs = X[sidx].astype(F32)
    if tfile.exists():
        z = np.load(tfile); C, tl = z["C"], z["labels"]; timing["teacher_cached"] = True
    else:
        t0 = time.time(); C, tl = kmeans(Xs, a.B, iters, a.seed); timing["teacher_s"] = time.time() - t0
        np.savez(tfile, C=C, labels=tl)
    cn = (C * C).sum(1)
    mlp = None
    if "mlp" in a.partition or "mlp" in a.router:
        mkey = hashlib.sha256(json.dumps([tkey, a.hidden, a.epochs, a.seed]).encode()).hexdigest()[:16]
        mfile = cache / f"mlp-{mkey}.npz"
        mlp = MLP(a.B, a.hidden, a.seed)
        if mfile.exists():
            z = np.load(mfile); mlp.p = [z[f"p{i}"] for i in range(len(mlp.p))]; timing["mlp_cached"] = True
            meta["train_history"] = json.loads(str(z["hist"]))
        else:
            t0 = time.time(); hist = mlp.train(Xs, tl, a.epochs, a.seed); timing["train_s"] = time.time() - t0
            meta["train_history"] = hist
            np.savez(mfile, hist=json.dumps(hist), **{f"p{i}": p for i, p in enumerate(mlp.p)})
        meta["mlp_bytes"] = mlp.nbytes(); meta["mlp_macs"] = mlp.macs(a.B)
    meta["centroid_bytes"] = int(C.nbytes); meta["centroid_macs"] = D * a.B
    # assign all N points
    parts = {}
    for p in a.partition.split(","):
        t0 = time.time(); lab = np.empty(a.n, np.int32)
        for s, e in chunks(a.n, 100_000):
            x = X[s:e].astype(F32)
            lab[s:e] = assign(x, C, cn) if p == "centroid" else np.argmax(mlp.logits(x), axis=1)
        timing[f"assign_{p}_s"] = time.time() - t0
        parts[p] = lab
        if p == "mlp":   # agreement of MLP-assigned postings with the teacher's Voronoi assignment
            meta["mlp_teacher_agreement_sample"] = float((np.argmax(mlp.logits(Xs), 1) == tl).mean())
    if "mlp" in parts and "centroid" in parts:
        meta["partition_agreement_all"] = float((parts["mlp"] == parts["centroid"]).mean())
    sets = {}
    for g in a.gold.split(","):
        name = Path(g).name
        sets[name] = (np.load(Path(g) / "queries.npy"), np.load(Path(g) / "gold.npy"))
    meta["stats"] = {}
    for p, lab in parts.items():
        sizes = np.bincount(lab, minlength=a.B).astype(np.int64)
        meta["stats"][p] = bucket_stats(sizes)
        budgets = budgets_grid(a.n / a.B, a.n); probes = nprobe_grid(a.B)
        for r in a.router.split(","):
            for name, (Q, gold) in sets.items():
                t0 = time.time()
                order = router_order(r, Q, C, cn, mlp)
                tr = traces(order, lab, sizes, gold, budgets, probes)
                timing[f"route_{p}_{r}_{name}_s"] = time.time() - t0
                np.savez_compressed(out / f"trace-{p}-{r}-{name}.npz", budgets=budgets, probes=probes, **tr)
                log(p, r, name, "R30@budget", np.round(tr["b_r30"].mean(0)[::8] / 30, 3))
    meta["timing"] = timing
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    log("done", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gold")
    for k in ("vectors", "queries", "rows", "out"):
        g.add_argument("--" + k, required=True)
    g.add_argument("--n", type=int, required=True); g.add_argument("--official")
    r = sp.add_parser("run")
    for k in ("vectors", "gold", "out"):
        r.add_argument("--" + k, required=True)
    r.add_argument("--n", type=int, required=True); r.add_argument("--B", type=int, required=True)
    r.add_argument("--teacher", default="native"); r.add_argument("--sample", type=int, default=0)
    r.add_argument("--iters", type=int, default=0); r.add_argument("--hidden", type=int, default=512)
    r.add_argument("--epochs", type=int, default=30); r.add_argument("--seed", type=int, default=42)
    r.add_argument("--partition", default="mlp,centroid"); r.add_argument("--router", default="mlp,centroid")
    r.add_argument("--cache", required=True)
    a = ap.parse_args()
    {"gold": cmd_gold, "run": cmd_run}[a.cmd](a)
