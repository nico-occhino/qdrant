"""Inspect persisted compact LMI postings without copying the corpus."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_u64(handle):
    data = handle.read(8)
    if len(data) != 8:
        raise ValueError("truncated compact posting length")
    return struct.unpack("<Q", data)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, default=10_000_000)
    parser.add_argument("--expected-buckets", type=int, default=3162)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {name: args.segment_index / name for name in
             ("lmi_state.json", "lmi_router.bin", "lmi_postings.bin")}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("native LMI state/router/postings are incomplete")
    with paths["lmi_postings.bin"].open("rb") as handle:
        n_boundaries = read_u64(handle)
        if n_boundaries != args.expected_buckets + 1:
            raise ValueError(f"unexpected boundary count {n_boundaries}")
        boundaries = np.fromfile(handle, dtype="<u8", count=n_boundaries)
        if len(boundaries) != n_boundaries:
            raise ValueError("truncated boundary table")
        n_points = read_u64(handle)
        if n_points != args.expected_points:
            raise ValueError(f"unexpected posting point count {n_points}")
        points = np.fromfile(handle, dtype="<u4", count=n_points)
        if len(points) != n_points or handle.read(1):
            raise ValueError("truncated postings or trailing bytes")
    if boundaries[0] != 0 or boundaries[-1] != n_points or np.any(boundaries[1:] < boundaries[:-1]):
        raise ValueError("invalid compact posting boundaries")
    if int(points.max(initial=0)) >= args.expected_points:
        raise ValueError("posting offset out of range")
    sizes = np.diff(boundaries).astype(np.int64)
    ordered = np.sort(sizes)
    n = len(ordered)
    total = int(ordered.sum())
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(ordered) / (n * total))
    result = {
        "segment_index": str(args.segment_index),
        "point_offsets": int(n_points), "buckets": args.expected_buckets,
        "active_buckets": int(np.count_nonzero(sizes)),
        "empty_buckets": int(np.count_nonzero(sizes == 0)),
        "size_distribution": {
            "min": int(ordered[0]), "p01": float(np.quantile(sizes, .01)),
            "p05": float(np.quantile(sizes, .05)), "median": float(np.median(sizes)),
            "mean": float(np.mean(sizes)), "p95": float(np.quantile(sizes, .95)),
            "p99": float(np.quantile(sizes, .99)), "max": int(ordered[-1]),
            "coefficient_of_variation": float(np.std(sizes) / np.mean(sizes)),
            "gini": gini,
        },
        "files_bytes": {name: path.stat().st_size for name, path in paths.items()},
    }
    result["auxiliary_total_bytes"] = sum(result["files_bytes"].values())
    result["auxiliary_bytes_per_vector"] = result["auxiliary_total_bytes"] / n_points
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
