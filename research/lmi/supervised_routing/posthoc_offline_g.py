#!/usr/bin/env python3
"""POST-HOC DIAGNOSTIC (not part of the preregistered gate): continuous-nprobe coverage on G.

    posthoc_offline_g.py RUN_DIR QUERIES.f32 OUT.json

The gate is decided on native measurements over the preregistered nprobe grid. This script only
describes how much of the difference between arms the grid granularity hides. It replays the
frozen router artifacts of RUN_DIR on the held-out G queries with the exact offline simulation
used for selection (native tie rule; candidates from bucket occupancy; coverage of the native-exact
gold top-10) for every integer nprobe in 1..512, and reports C90/C95/C97: mean candidates at the
smallest nprobe whose mean coverage reaches the target. G queries are rebuilt from the official
query file and must match the hash recorded by the job.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

D = 768
ROWS_G = np.arange(4000, 5000)


def router(dir_):
    spec = json.loads((dir_ / "router.json").read_text())
    layers = []
    for layer in spec["layers"]:
        if layer["kind"] == "relu":
            layers.append(None)
            continue
        w = np.fromfile(dir_ / layer["weights"], dtype=np.float32).reshape(layer["out"], layer["in"])
        b = np.fromfile(dir_ / layer["bias"], dtype=np.float32)
        layers.append((w, b))
    return layers


def forward(layers, q):
    x = q
    for layer in layers:
        x = np.maximum(x, 0) if layer is None else x @ layer[0].T + layer[1]
    return x


def main():
    run, qfile, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    q_all = np.fromfile(qfile, dtype=np.float32).reshape(-1, D)
    G = q_all[ROWS_G]
    G = (G / np.linalg.norm(G, axis=1, keepdims=True)).astype(np.float32)
    splits = json.loads((run / "splits.json").read_text())
    assert hashlib.sha256(G.tobytes()).hexdigest() == splits["splits"]["G"]["sha256"], "G queries must match the job"
    gold = np.fromfile(run / "gold" / "G.ids.u64", dtype=np.uint64).reshape(-1, 30)[:, :10].astype(np.int64)
    bucket_of = np.fromfile(run / "export" / "bucket_of_id.u32", dtype=np.uint32).astype(np.int64)
    occ = np.fromfile(run / "export" / "occupancy.u32", dtype=np.uint32).astype(np.float64)
    B = len(occ)
    targets = np.zeros((len(G), B))
    np.add.at(targets, (np.repeat(np.arange(len(G)), 10), bucket_of[gold.reshape(-1) - 1]), 0.1)
    result = {"note": "post-hoc diagnostic; the preregistered gate uses native grid measurements", "arms": {}}
    probes = min(512, B)
    for dir_ in sorted((run / "train" / "routers").iterdir()):
        logits = forward(router(dir_), G)
        order = np.argsort(-logits, axis=1, kind="stable")[:, :probes]
        cov = np.take_along_axis(targets, order, 1).cumsum(1).mean(0)
        cand = occ[order].cumsum(1).mean(0)
        arm = {}
        for t in (0.90, 0.95, 0.97, 0.98, 0.99):
            hit = np.nonzero(cov >= t - 1e-12)[0]
            arm[str(t)] = None if len(hit) == 0 else {"nprobe": int(hit[0]) + 1, "candidates": float(cand[hit[0]])}
        result["arms"][dir_.name] = arm
    a0 = result["arms"]["A0"]
    for name, arm in result["arms"].items():
        arm["reduction_vs_A0"] = {t: (None if arm[t] is None or a0[t] is None else 1 - arm[t]["candidates"] / a0[t]["candidates"])
                                  for t in ("0.9", "0.95", "0.97")}
    out.write_text(json.dumps(result, indent=1))
    for name, arm in result["arms"].items():
        print(name, {t: (None if arm[t] is None else (arm[t]["nprobe"], round(arm[t]["candidates"]))) for t in ("0.9", "0.95", "0.97")},
              {t: (None if v is None else round(v, 3)) for t, v in arm["reduction_vs_A0"].items()})


if __name__ == "__main__":
    main()
