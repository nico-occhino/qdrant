"""Inspect SISAP HDF5 datasets and emit a compact, reproducible manifest."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def describe(path: Path):
    result = {"file": str(path.resolve()), "bytes": path.stat().st_size, "datasets": {}}
    with h5py.File(path, "r") as handle:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            entry = {"shape": list(obj.shape), "dtype": str(obj.dtype),
                     "chunks": obj.chunks, "compression": obj.compression}
            if obj.ndim:
                first = np.asarray(obj[0])
                entry["first_row_shape"] = list(first.shape)
                entry["first_row_preview"] = first.reshape(-1)[:8].tolist()
            result["datasets"][name] = entry
        handle.visititems(visit)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"format": "SISAP HDF5 structural audit v1",
               "files": [describe(path) for path in args.files]}
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
