"""Stored-vector oracle gate (PREREGISTRATION.md §2.3 and §6 gate 3, implementation item I5).

    python3 -I oracle_check.py --corpus F16 --run ARM_OUTPUT_DIR [--out FILE]

For each `batch_XX/stored_sample.bin` the driver wrote, the vectors read back from the
collection's storage are compared bit for bit with `fb_plan.stored_rows` of the same corpus
rows. That function also generates the G1 gold. The gate passes iff every sampled vector is
bit-identical.

Reported per batch: sampled count, bit-identical count, the largest per-component difference in
f16 ulps, and an upper bound on the score change of any unit query, max ‖stored − model‖₂, to
compare with the tie tolerance eps.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb_plan as F  # noqa: E402

REC = 8 + F.DIM * 2


def read_sample(path):
    raw = np.fromfile(path, dtype=np.uint8)
    assert raw.size % REC == 0, f"{path}: truncated record"
    rec = raw.reshape(-1, REC)
    ids = rec[:, :8].copy().view("<u8").ravel()
    vec = rec[:, 8:].copy().view("<f2").reshape(-1, F.DIM)
    return ids, vec


def check(corpus, run_dir):
    out = []
    for f in sorted(Path(run_dir).glob("batch_*/stored_sample.bin")):
        ids, got = read_sample(f)
        model = F.stored_rows(corpus[ids.astype(np.int64) - 1])
        same = np.all(got.view("<u2") == model.view("<u2"), axis=1)
        ulps = np.abs(got.view("<i2").astype(np.int32) - model.view("<i2").astype(np.int32))
        delta = np.linalg.norm(got.astype(np.float64) - model.astype(np.float64), axis=1)
        out.append({"batch": f.parent.name, "sampled": int(len(ids)), "identical": int(same.sum()),
                    "max_ulp": int(ulps.max()) if len(ids) else 0,
                    "max_score_bound": float(delta.max()) if len(ids) else 0.0})
    return {"batches": out, "eps": F.EPS,
            "pass": bool(out) and all(b["identical"] == b["sampled"] and b["sampled"] > 0 for b in out)}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--corpus", type=Path, required=True)
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    corpus = np.memmap(args.corpus, dtype="<f2", mode="r").reshape(-1, F.DIM)
    result = check(corpus, args.run)
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
