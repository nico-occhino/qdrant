"""Create a bounded 250k-row experiment sample from the named 10M Lance corpus.

The output is a sample only, never a full-corpus conversion. Qdrant cosine
preprocessing is applied when the Rust clustering benchmark reads the file.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import lance
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=250_000)
    parser.add_argument("--scan-batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("sample output or metadata already exists")
    dataset = lance.dataset(str(args.source))
    count = dataset.count_rows()
    if count != 10_000_000 or not 0 < args.sample_size <= count:
        raise ValueError("expected named 10M corpus and valid sample size")
    rng = np.random.default_rng(args.seed)
    starts = np.arange(args.sample_size, dtype=np.int64) * count // args.sample_size
    ends = (np.arange(args.sample_size, dtype=np.int64) + 1) * count // args.sample_size
    chosen = starts + rng.integers(0, ends - starts)
    assert np.all(chosen[1:] > chosen[:-1])
    started = time.monotonic()
    written = 0
    position = 0
    digest = hashlib.sha256()
    with args.output.open("xb") as sink:
        for batch in dataset.to_batches(columns=["vector"], batch_size=args.scan_batch,
                                        batch_readahead=1, fragment_readahead=1,
                                        scan_in_order=True):
            stop = position + batch.num_rows
            left = np.searchsorted(chosen, position)
            right = np.searchsorted(chosen, stop)
            if right > left:
                block = batch.column(0).values.to_numpy(zero_copy_only=False)
                block = block.reshape(batch.num_rows, 768)
                selected = np.asarray(block[chosen[left:right] - position], dtype="<f4")
                if not np.isfinite(selected).all():
                    raise ValueError(f"non-finite vector in source rows {position}:{stop}")
                encoded = selected.tobytes(order="C")
                sink.write(encoded)
                digest.update(encoded)
                written += len(selected)
            position = stop
    if position != count or written != args.sample_size:
        raise AssertionError((position, written))
    result = {"source": str(args.source.resolve()), "source_rows": count,
              "sample_rows": written, "dimension": 768, "dtype": "little-endian float32",
              "sampling": "one seeded uniform random row per disjoint equal-width stratum",
              "seed": args.seed, "scan_batch": args.scan_batch,
              "sample_bytes": args.output.stat().st_size,
              "sample_sha256": digest.hexdigest(),
              "elapsed_seconds": time.monotonic() - started,
              "client_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
