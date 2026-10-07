#!/usr/bin/env python3
"""Stream the read-only SISAP300K HDF5 inputs into local test files."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = [args.output / name for name in ("vectors.f16", "queries.f32", "gold.i32")]
    if any(path.exists() for path in outputs):
        parser.error("output files already exist; choose a fresh directory")

    with h5py.File(args.dataset_dir / "laion2B-en-clip768v2-n=300K.h5") as source, outputs[0].open("wb") as target:
        vectors = source["emb"]
        assert vectors.shape == (300_000, 768) and vectors.dtype == np.float16
        for start in range(0, 300_000, 1024):
            target.write(np.asarray(vectors[start:start + 1024], dtype="<f2").tobytes())

    with h5py.File(args.dataset_dir / "public-queries-10k-clip768v2.h5") as source, outputs[1].open("wb") as target:
        queries = source["emb"]
        assert queries.shape == (10_000, 768)
        target.write(np.asarray(queries[:100], dtype="<f4").tobytes())

    with h5py.File(args.dataset_dir / "laion2B-en-public-gold-standard-v2-300K-F64-IEEE754.h5") as source, outputs[2].open("wb") as target:
        neighbors = source["knns"]
        assert neighbors.shape == (10_000, 1000)
        target.write(np.asarray(neighbors[:100, :10], dtype="<i4").tobytes())


if __name__ == "__main__":
    main()
