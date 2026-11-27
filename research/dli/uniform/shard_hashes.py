"""Q11 shard-membership gate (SCALE_SHARED_SPEC.md §2.2).

    shard_hashes.py --plan PLAN_DIR [--out FILE]

The live set is the union of every batch's insert.u32 minus its delete.u32, checked against the plan's final
(live_count, live_sha256). The point id is the u32 value in insert.u32, widened to u64: the id Qdrant stores.
shard(id) = splitmix64(id) mod 4 (wrapping u64). Per shard: size and SHA-256 of the sorted ids as u64 little-endian.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

M64 = (1 << 64) - 1
SHARDS = 4


def splitmix64(x):
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def splitmix64_np(x):
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def live_ids(plan_dir):
    plan = json.loads((plan_dir / "plan.json").read_text())
    live = set()
    for b in plan["batches"]:
        d = plan_dir / f"batch_{b['index']:02d}"
        live.update(np.fromfile(d / "insert.u32", dtype="<u4").tolist())
        live.difference_update(np.fromfile(d / "delete.u32", dtype="<u4").tolist())
    ids = np.array(sorted(live), dtype="<u4")
    last = plan["batches"][-1]
    assert (len(ids), hashlib.sha256(ids.tobytes()).hexdigest()) == (last["live_count"], last["live_sha256"]), \
        "live set does not reproduce the plan's final live_sha256"
    return plan, ids.astype("<u8")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    # Known answers: the reference splitmix64 stream seeded with 0 starts e220a8397b1dcdaf, 6e789e6aa1b965f4.
    assert splitmix64(0) == 0xE220A8397B1DCDAF and splitmix64(0x9E3779B97F4A7C15) == 0x6E789E6AA1B965F4
    probe = np.array([0, 1, 2**32 - 1, 123456789], dtype=np.uint64)
    assert [int(v) for v in splitmix64_np(probe)] == [splitmix64(int(v)) for v in probe]
    plan, ids = live_ids(args.plan)
    shard = splitmix64_np(ids) % np.uint64(SHARDS)
    out = {"plan_dir": args.plan.name, "scenario": plan["scenario"], "seed": plan["seed"],
           "live_count": int(len(ids)), "live_sha256_u32": plan["batches"][-1]["live_sha256"],
           "id_min": int(ids.min()), "id_max": int(ids.max()), "shard_function": "splitmix64(id) mod 4",
           "shards": []}
    for s in range(SHARDS):
        m = np.sort(ids[shard == s]).astype("<u8")
        out["shards"].append({"shard": s, "size": int(len(m)), "sha256_u64le": hashlib.sha256(m.tobytes()).hexdigest()})
    assert sum(x["size"] for x in out["shards"]) == len(ids)
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
