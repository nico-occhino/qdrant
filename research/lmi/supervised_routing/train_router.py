#!/usr/bin/env python3
"""Train and select supervised routers over a fixed native IVF partition (PREREGISTRATION.md).

    train_router.py --export EXPORT --inputs INPUTS --gold GOLD --corpus corpus.f16le --n N --out OUT
                    [--threads 16] [--smoke] [--only CONFIG_NAME ...] [--labels native|official]

EXPORT  harness `export` phase: A0/router.json (+ f32 files), bucket_of_id.u32, occupancy.u32
INPUTS  make_inputs.py output: T.f32, V.f32, P.f32, P.gold.u64 (and *.official.u64 at 10M)
GOLD    native gold from the harness `gold` phase: T.ids.u64, V.ids.u64 (k = 30, 1-based)

Every selected router is written as a native artifact (router.json + raw f32), with linear arms
folded into ONE Linear layer. Selection uses validation (V) only: C95 = mean candidates at the
smallest nprobe in [1, 512] whose mean V coverage of the gold top-10 reaches 0.95, by exact
simulation with the native tie rule (equal logits -> smaller bucket id).
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

D = 768
K = 10
MAX_PROBE = 512
BATCH = 1024
LR = 1e-3
SEED = 7


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_f32(path, cols=D):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(-1, cols))


def load_ids(path, k):
    return np.fromfile(path, dtype=np.uint64).reshape(-1, k)


def load_router(dir_):
    spec = json.loads((Path(dir_) / "router.json").read_text())
    layers = []
    for layer in spec["layers"]:
        if layer["kind"] == "linear":
            w = np.fromfile(Path(dir_) / layer["weights"], dtype=np.float32).reshape(layer["out"], layer["in"])
            b = np.fromfile(Path(dir_) / layer["bias"], dtype=np.float32)
            layers.append(("linear", torch.from_numpy(w.copy()), torch.from_numpy(b.copy())))
        else:
            layers.append(("relu", None, None))
    return layers


def write_artifact(dir_, arm, layers, meta):
    dir_ = Path(dir_)
    dir_.mkdir(parents=True, exist_ok=False)
    out = []
    for i, (kind, w, b) in enumerate(layers):
        if kind == "relu":
            out.append({"kind": "relu"})
            continue
        w = w.detach().to(torch.float32).contiguous().numpy()
        b = b.detach().to(torch.float32).contiguous().numpy()
        assert np.isfinite(w).all() and np.isfinite(b).all(), f"{arm}: non-finite parameters"
        wb, bb = w.tobytes(), b.tobytes()
        (dir_ / f"layer{i}.weights.f32").write_bytes(wb)
        (dir_ / f"layer{i}.bias.f32").write_bytes(bb)
        out.append({"kind": "linear", "in": int(w.shape[1]), "out": int(w.shape[0]),
                    "weights": f"layer{i}.weights.f32", "bias": f"layer{i}.bias.f32",
                    "weights_sha256": sha_bytes(wb), "bias_sha256": sha_bytes(bb)})
    spec = {"arm": arm, "layers": out, **meta}
    (dir_ / "router.json").write_text(json.dumps(spec, indent=1))
    return spec


def forward(layers, Q):
    x = Q
    for kind, w, b in layers:
        x = torch.relu(x) if kind == "relu" else x @ w.T + b
    return x


# ------------------------------------------------------------------ targets and C95

class Targets:
    """Sparse neighbour-bucket counts v_b(q) for the gold top-10."""

    def __init__(self, gold_ids, bucket_of):
        b = bucket_of[(gold_ids[:, :K].astype(np.int64) - 1)]          # (nq, 10)
        self.rows = torch.from_numpy(np.repeat(np.arange(len(b)), K))
        self.cols = torch.from_numpy(b.reshape(-1).astype(np.int64))
        self.n = len(b)

    def dense(self, idx, buckets):
        """t_b(q) = v_b(q)/10 for the rows `idx` (LongTensor)."""
        pos = torch.full((self.n,), -1, dtype=torch.int64)
        pos[idx] = torch.arange(len(idx))
        sel = pos[self.rows] >= 0
        t = torch.zeros((len(idx), buckets))
        t.index_put_((pos[self.rows[sel]], self.cols[sel]), torch.full((int(sel.sum()),), 1.0 / K),
                     accumulate=True)
        return t


def ranking(logits):
    """Native order: descending logit, equal logits -> smaller bucket id (stable sort).
    Truncated to min(MAX_PROBE, buckets) probes."""
    return torch.sort(logits, dim=1, descending=True, stable=True).indices[:, :MAX_PROBE]


def coverage_curves(logits, targets, occupancy, chunk=1000):
    """Mean coverage and mean candidates for nprobe = 1..min(MAX_PROBE, buckets)."""
    probes = min(MAX_PROBE, logits.shape[1])
    cov = torch.zeros(probes, dtype=torch.float64)
    cand = torch.zeros(probes, dtype=torch.float64)
    occ = occupancy.to(torch.float64)
    for s in range(0, logits.shape[0], chunk):
        idx = torch.arange(s, min(s + chunk, logits.shape[0]))
        order = ranking(logits[idx])
        t = targets.dense(idx, logits.shape[1]).to(torch.float64)
        cov += torch.gather(t, 1, order).cumsum(1).sum(0)
        cand += occ[order].cumsum(1).sum(0)
    n = logits.shape[0]
    return (cov / n).numpy(), (cand / n).numpy()


def c95(logits, targets, occupancy):
    cov, cand = coverage_curves(logits, targets, occupancy)
    hit = np.nonzero(cov >= 0.95)[0]
    if len(hit) == 0:
        return {"c95": math.inf, "nprobe95": None, "max_coverage": float(cov[-1])}
    p = int(hit[0])
    return {"c95": float(cand[p]), "nprobe95": p + 1, "coverage_at": float(cov[p]),
            "coverage_curve_head": [float(x) for x in cov[:64]]}


# ------------------------------------------------------------------ A1: LDA head

def lda_head(corpus, n, bucket_of, occupancy, lam):
    """Shared-covariance Gaussian posterior over buckets (closed form, no queries)."""
    B = len(occupancy)
    sums = torch.zeros((B, D), dtype=torch.float64)
    xtx = torch.zeros((D, D), dtype=torch.float64)
    for s in range(0, n, 250_000):
        raw = np.asarray(corpus[s:min(n, s + 250_000)]).astype(np.float32)
        x = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        x = torch.from_numpy(x.astype(np.float16).astype(np.float32)).to(torch.float64)
        sums.index_add_(0, torch.from_numpy(bucket_of[s:s + len(x)].astype(np.int64)), x)
        xtx += x.T @ x
    occ = occupancy.to(torch.float64)
    mu = sums / occ.clamp(min=1)[:, None]
    within = (xtx - (mu * occ[:, None]).T @ mu) / n
    shrunk = (1 - lam) * within + lam * (torch.trace(within) / D) * torch.eye(D, dtype=torch.float64)
    inv = torch.linalg.inv(shrunk)
    W = mu @ inv
    bias = -0.5 * (W * mu).sum(1) + torch.log(occ.clamp(min=1) / n)
    return W.to(torch.float32), bias.to(torch.float32)


# ------------------------------------------------------------------ trainable arms

class LinearRouter(torch.nn.Module):
    def __init__(self, C, c0, kind, rank=0):
        super().__init__()
        self.register_buffer("C", C)
        self.register_buffer("c0", c0)
        self.kind = kind
        B = C.shape[0]
        self.beta = torch.nn.Parameter(torch.zeros(B))
        if kind == "lowrank":
            self.U = torch.nn.Parameter(torch.zeros(B, rank))
            self.V = torch.nn.Parameter(torch.randn(D, rank) * 0.01)
        elif kind == "full":
            self.Delta = torch.nn.Parameter(torch.zeros(B, D))

    def delta(self):
        if self.kind == "lowrank":
            return self.U @ self.V.T
        if self.kind == "full":
            return self.Delta
        return None

    def scores(self, Q):
        s = Q @ self.C.T + self.c0 + self.beta
        if self.kind == "lowrank":
            s = s + (Q @ self.V) @ self.U.T
        elif self.kind == "full":
            s = s + Q @ self.Delta.T
        return s

    def penalty(self):
        B = self.C.shape[0]
        if self.kind == "lowrank":
            return torch.trace((self.U.T @ self.U) @ (self.V.T @ self.V)) / B
        if self.kind == "full":
            return (self.Delta ** 2).sum() / B
        return torch.zeros(())

    def folded(self, tau):
        delta = self.delta()
        W = self.C + (delta if delta is not None else 0)
        return W / tau, (self.c0 + self.beta) / tau

    def params(self):
        return sum(p.numel() for p in self.parameters())


class MlpRouterModel(torch.nn.Module):
    """768 -> 512 -> B, native initialisation (uniform +-1/sqrt(fan_in))."""

    def __init__(self, B, hidden=512):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        def lin(i, o):
            bound = 1 / math.sqrt(i)
            layer = torch.nn.Linear(i, o)
            with torch.no_grad():
                layer.weight.uniform_(-bound, bound, generator=g)
                layer.bias.uniform_(-bound, bound, generator=g)
            return layer
        self.l1, self.l2 = lin(D, hidden), lin(hidden, B)

    def scores(self, Q):
        return self.l2(torch.relu(self.l1(Q)))

    def penalty(self):
        return torch.zeros(())

    def params(self):
        return sum(p.numel() for p in self.parameters())


def grid(smoke):
    configs = []
    for data in ("T", "TP"):
        for tau in (0.02, 0.05):
            configs.append({"arm": "A2", "kind": "bias", "tau": tau, "lam": 0.0, "data": data})
            for r in (8, 32):
                for lam in (0.01, 0.1):
                    configs.append({"arm": "A3", "kind": "lowrank", "rank": r, "tau": tau, "lam": lam, "data": data})
            for lam in (0.01, 0.1, 1.0):
                configs.append({"arm": "A4", "kind": "full", "tau": tau, "lam": lam, "data": data})
        configs.append({"arm": "A5", "kind": "mlp", "tau": 1.0, "lam": 0.0, "data": data})
    for c in configs:
        c["name"] = "_".join(f"{k}{v}" for k, v in c.items() if k not in ("kind",))
    if smoke:  # preflight: one configuration per arm, both data options for A2
        keep, seen = [], set()
        for c in configs:
            key = (c["arm"], c["data"] if c["arm"] == "A2" else "")
            if key not in seen:
                seen.add(key)
                keep.append(c)
        configs = keep
    return configs


def train_one(cfg, C, c0, data, targets_val, Qv, occupancy, smoke):
    torch.manual_seed(SEED)
    B = C.shape[0]
    model = (MlpRouterModel(B) if cfg["kind"] == "mlp" else
             LinearRouter(C, c0, cfg["kind"], cfg.get("rank", 0)))
    Qt, Tt = data
    n = Qt.shape[0]
    epochs = (1 if smoke else (100 if cfg["data"] == "T" else 10))
    steps_per_epoch = math.ceil(n / BATCH)
    total = epochs * steps_per_epoch
    checkpoints = sorted({max(1, round(total * f / 10)) for f in range(1, 11)})
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED)
    tau = cfg["tau"]
    curve, best = [], None
    step, began = 0, time.time()
    while step < total:
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            logits = model.scores(Qt[idx]) / tau
            t = Tt.dense(idx, B)
            loss_rank = -(t * torch.log_softmax(logits, dim=1)).sum(1).mean()
            loss = loss_rank + cfg["lam"] * model.penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step in checkpoints:
                with torch.no_grad():
                    crit = c95(model.scores(Qv), targets_val, occupancy)
                curve.append({"step": step, "loss_rank": float(loss_rank), "loss": float(loss), **{k: crit[k] for k in ("c95", "nprobe95")}})
                if best is None or crit["c95"] < best[0]:
                    best = (crit["c95"], step, {k: v.detach().clone() for k, v in model.state_dict().items()})
            if step >= total:
                break
    model.load_state_dict(best[2])
    return model, {"curve": curve, "best_step": best[1], "steps": total, "seconds": time.time() - began,
                   "params": model.params()}


def layers_of(model, tau, occupancy=None):
    """Native layers: one Linear for linear arms (folded), Linear-ReLU-Linear for the MLP."""
    if isinstance(model, MlpRouterModel):
        b2 = model.l2.bias.detach()
        if occupancy is not None:
            b2 = b2 - torch.log(occupancy.clamp(min=1).to(torch.float32))
        return [("linear", model.l1.weight.detach(), model.l1.bias.detach()), ("relu", None, None),
                ("linear", model.l2.weight.detach(), b2)]
    W, b = model.folded(tau)
    if occupancy is not None:
        b = b - torch.log(occupancy.clamp(min=1).to(torch.float32))
    return [("linear", W.detach(), b.detach())]


def main():
    ap = argparse.ArgumentParser()
    for name in ("export", "inputs", "gold", "corpus", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--labels", choices=["native", "official"], default="native")
    ap.add_argument("--only", nargs="*", help="train only these configuration names (10M retraining)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    began = time.time()
    exp, inp, gold = Path(a.export), Path(a.inputs), Path(a.gold)

    bucket_of = np.fromfile(exp / "bucket_of_id.u32", dtype=np.uint32).astype(np.int64)
    occupancy = torch.from_numpy(np.fromfile(exp / "occupancy.u32", dtype=np.uint32).astype(np.int64))
    a0 = load_router(exp / "A0")
    assert len(a0) == 1 and a0[0][0] == "linear", "A0 must be the single-Linear centroid router"
    C, c0 = a0[0][1], a0[0][2]
    B = C.shape[0]
    assert len(bucket_of) == a.n and len(occupancy) == B

    def labels(split):
        if a.labels == "official" and (inp / f"{split}.official.u64").exists():
            return load_ids(inp / f"{split}.official.u64", 30)
        return load_ids(gold / f"{split}.ids.u64", 30)

    QT, QV = load_f32(inp / "T.f32"), load_f32(inp / "V.f32")
    tgT, tgV = Targets(labels("T"), bucket_of), Targets(labels("V"), bucket_of)
    QP = load_f32(inp / "P.f32")
    QTP = torch.cat([QT, QP])
    gTP = np.concatenate([labels("T")[:, :K], load_ids(inp / "P.gold.u64", 10)])
    tgTP = Targets(gTP, bucket_of)
    datasets = {"T": (QT, tgT), "TP": (QTP, tgTP)}

    report = {"n": a.n, "buckets": B, "labels": a.labels, "smoke": a.smoke,
              "occupancy": {"min": int(occupancy.min()), "max": int(occupancy.max()),
                            "mean": float(occupancy.double().mean()),
                            "cv": float(occupancy.double().std() / occupancy.double().mean()),
                            "empty": int((occupancy == 0).sum())},
              "configs": [], "baselines": {}}

    # Distribution / difficulty of the query sources under A0 (diagnostic).
    with torch.no_grad():
        tgP20 = Targets(load_ids(inp / "P.gold.u64", 10)[:20000], bucket_of)
        for name, (Q, tg) in {"T": (QT, tgT), "V": (QV, tgV), "P20k": (QP[:20000], tgP20)}.items():
            report["baselines"][f"A0_{name}"] = c95(forward(a0, Q), tg, occupancy)

    # Reference arms: A0 (+occupancy), A1 (+occupancy).
    candidates = {}
    with torch.no_grad():
        s0 = forward(a0, QV)
        candidates["A0"] = (c95(s0, tgV, occupancy), [("linear", C, c0)], {"origin": "native centroid router"})
        for tau in (0.02, 0.05):
            crit = c95(s0 / tau - torch.log(occupancy.clamp(min=1).float()), tgV, occupancy)
            report["baselines"][f"A0occ_tau{tau}"] = crit
            if "A0occ" not in candidates or crit["c95"] < candidates["A0occ"][0]["c95"]:
                candidates["A0occ"] = (crit, [("linear", C / tau, c0 / tau - torch.log(occupancy.clamp(min=1).float()))],
                                       {"tau": tau})
        corpus = np.memmap(a.corpus, dtype=np.float16, mode="r").reshape(-1, D)[: a.n]
        for lam in ((0.1,) if a.smoke else (0.01, 0.1, 1.0)):
            t0 = time.time()
            W, b = lda_head(corpus, a.n, bucket_of, occupancy, lam)
            s1 = QV @ W.T + b
            crit = c95(s1, tgV, occupancy)
            report["baselines"][f"A1_lam{lam}"] = {**crit, "seconds": time.time() - t0}
            if "A1" not in candidates or crit["c95"] < candidates["A1"][0]["c95"]:
                candidates["A1"] = (crit, [("linear", W, b)], {"lambda": lam})
            for tau in (0.02, 0.05):
                c = c95(s1 / tau - torch.log(occupancy.clamp(min=1).float()), tgV, occupancy)
                report["baselines"][f"A1occ_lam{lam}_tau{tau}"] = c
                if "A1occ" not in candidates or c["c95"] < candidates["A1occ"][0]["c95"]:
                    candidates["A1occ"] = (c, [("linear", W / tau, b / tau - torch.log(occupancy.clamp(min=1).float()))],
                                           {"lambda": lam, "tau": tau})

    # Trainable arms over the preregistered grid.
    for cfg in grid(a.smoke):
        if a.only and cfg["name"] not in a.only:
            continue
        model, info = train_one(cfg, C, c0, datasets[cfg["data"]], tgV, QV, occupancy, a.smoke)
        with torch.no_grad():
            plain = c95(model.scores(QV), tgV, occupancy)
            occ = c95(model.scores(QV) / cfg["tau"] - torch.log(occupancy.clamp(min=1).float()), tgV, occupancy)
            train = c95(model.scores(QT), tgT, occupancy)
        entry = {**cfg, **info, "val": plain, "val_occ": occ, "train_T": train}
        report["configs"].append(entry)
        print(json.dumps({k: entry[k] for k in ("name", "params", "seconds", "best_step")} |
                         {"val_c95": plain["c95"], "val_occ_c95": occ["c95"], "train_c95": train["c95"]}), flush=True)
        for arm, crit, occ_flag in ((cfg["arm"], plain, False), (cfg["arm"] + "occ", occ, True)):
            if cfg["arm"] == "A5" and occ_flag:
                continue  # occupancy variants are preregistered for A0-A4 only
            key = (crit["c95"], info["params"])
            if arm not in candidates or key < (candidates[arm][0]["c95"], candidates[arm][2].get("params", 0)):
                candidates[arm] = (crit, layers_of(model, cfg["tau"], occupancy if occ_flag else None),
                                   {"config": cfg["name"], "params": info["params"], "tau": cfg["tau"]})

    # Write every selected arm; pick the promotion candidate among learned arms.
    selection = {}
    for arm, (crit, layers, meta) in sorted(candidates.items()):
        write_artifact(out / "routers" / arm, arm, layers, {"selection": crit, **meta})
        selection[arm] = {"val": crit, **meta, "router": str(out / "routers" / arm / "router.json")}
    learned = [arm for arm in selection if arm.startswith(("A2", "A3", "A4", "A5"))]
    promo = min(learned, key=lambda arm: (selection[arm]["val"]["c95"], selection[arm].get("params", 0)))
    report["selection"] = selection
    report["promotion_candidate"] = promo
    report["seconds"] = time.time() - began

    # Offline ranking of the first 200 V queries for native/offline agreement.
    with (out / "consistency_python.jsonl").open("w") as f:
        for arm in selection:
            layers = load_router(out / "routers" / arm)
            with torch.no_grad():
                order = ranking(forward(layers, QV[:200]))[:, :64]
            for q in range(order.shape[0]):
                f.write(json.dumps({"arm": arm, "q": q, "buckets": order[q].tolist()}) + "\n")
    (out / "training.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "arms.txt").write_text("\n".join(selection[arm]["router"] for arm in sorted(selection)) + "\n")
    print("promotion candidate:", promo, json.dumps(selection[promo]["val"]))


if __name__ == "__main__":
    main()
