"""Known-answer test for oracle_check.py: model vectors pass; a one-ulp perturbation fails with
max_ulp 1; a truncated file is rejected."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb_plan as F  # noqa: E402
import oracle_check as O  # noqa: E402

rng = np.random.default_rng(3)
corpus = rng.standard_normal((500, F.DIM)).astype("<f2")
ids = np.array([1, 7, 250, 500], dtype="<u8")


def write(path, vecs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for i, v in zip(ids, vecs):
            f.write(int(i).to_bytes(8, "little"))
            f.write(np.asarray(v, dtype="<f2").tobytes())


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    model = F.stored_rows(corpus[ids.astype(np.int64) - 1])
    write(tmp / "ok" / "batch_00" / "stored_sample.bin", model)
    r = O.check(corpus, tmp / "ok")
    assert r["pass"] and r["batches"][0]["identical"] == 4 and r["batches"][0]["max_ulp"] == 0, r
    bad = model.copy()
    bad[2, 5] = np.nextafter(bad[2, 5], np.float16(np.inf))
    write(tmp / "bad" / "batch_00" / "stored_sample.bin", bad)
    r = O.check(corpus, tmp / "bad")
    assert not r["pass"] and r["batches"][0]["identical"] == 3 and r["batches"][0]["max_ulp"] == 1, r
    assert 0 < r["batches"][0]["max_score_bound"] < F.EPS, r
    assert not O.check(corpus, tmp / "empty")["pass"], "no sample files must not pass"
    (tmp / "trunc" / "batch_00").mkdir(parents=True)
    (tmp / "trunc" / "batch_00" / "stored_sample.bin").write_bytes(b"\x00" * 10)
    try:
        O.check(corpus, tmp / "trunc")
        raise SystemExit("truncated file accepted")
    except AssertionError:
        pass
print("oracle_check known-answer test: OK")
