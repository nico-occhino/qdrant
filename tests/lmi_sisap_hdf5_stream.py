"""Stream an official SISAP LAION HDF5 embedding matrix into Qdrant.

Point IDs are the official 1-based corpus row numbers. A checkpoint advances
only after an entire upsert block receives a successful response.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import numpy as np


def request_json(method, url, body=None, timeout=180):
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = Request(url, data=encoded, method=method,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:17036")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0, help="zero-based source row")
    parser.add_argument("--end", type=int, required=True, help="exclusive source row")
    parser.add_argument("--scan-batch", type=int, default=256)
    parser.add_argument("--upsert-batch", type=int, default=64)
    parser.add_argument("--min-disk-free-gib", type=float, default=10)
    parser.add_argument("--verify-samples", type=int, default=18)
    parser.add_argument("--expected-total", type=int,
                        help="require this final Qdrant point count (for benchmark gates)")
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source does not exist: {args.source}")
    if args.scan_batch < 1 or args.upsert_batch < 1:
        parser.error("batch sizes must be positive")
    if shutil.disk_usage(args.source.parent).free < args.min_disk_free_gib * 1024**3:
        raise RuntimeError("source drive is below configured free-space guard")
    root = args.url.rstrip("/") + "/collections/" + args.collection
    info = request_json("GET", root)["result"]
    vector_params = info["config"]["params"]["vectors"]
    if vector_params.get("datatype") != "float16" or vector_params["size"] != 768:
        raise RuntimeError(f"collection must be 768-d float16, got {vector_params}")
    with h5py.File(args.source, "r") as source:
        if "emb" not in source:
            raise RuntimeError(f"expected source key emb, found {list(source.keys())}")
        matrix = source["emb"]
        if matrix.ndim != 2 or matrix.shape[1] != 768 or matrix.dtype != np.float16:
            raise RuntimeError(f"expected float16 (N,768) emb, got {matrix.shape} {matrix.dtype}")
        if not 0 <= args.start < args.end <= matrix.shape[0]:
            parser.error(f"row range must satisfy 0 <= start < end <= {matrix.shape[0]}")
        if args.checkpoint.exists():
            saved = json.loads(args.checkpoint.read_text())
            if saved.get("source") != str(args.source.resolve()) or saved.get("end") != args.end:
                raise RuntimeError("checkpoint belongs to a different source or end row")
            cursor = int(saved["next_source_row"])
            if cursor < args.start or cursor > args.end:
                raise RuntimeError("checkpoint row is outside requested import range")
        else:
            cursor = args.start
        collection_count = int(info["points_count"])
        if collection_count and cursor == args.start:
            raise RuntimeError("refusing import into nonempty collection without matching checkpoint")
        began = time.monotonic()
        while cursor < args.end:
            block_end = min(cursor + args.scan_batch, args.end)
            block = np.asarray(matrix[cursor:block_end], dtype=np.float32)
            if not np.isfinite(block).all():
                raise RuntimeError(f"non-finite embedding in source rows [{cursor},{block_end})")
            for batch_start in range(0, len(block), args.upsert_batch):
                batch_end = min(batch_start + args.upsert_batch, len(block))
                points = [{"id": row + 1, "vector": vector.tolist()}
                          for row, vector in zip(range(cursor + batch_start,
                                                       cursor + batch_end),
                                                 block[batch_start:batch_end])]
                response = request_json("PUT", root + "/points?wait=true", {"points": points})
                if response.get("status") != "ok":
                    raise RuntimeError(f"upsert failed in source rows "
                                       f"[{cursor + batch_start},{cursor + batch_end}): {response}")
            cursor = block_end
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            temporary.write_text(json.dumps({"source": str(args.source.resolve()),
                                             "start": args.start, "end": args.end,
                                             "next_source_row": cursor,
                                             "last_completed_id": cursor}, indent=2) + "\n")
            os.replace(temporary, args.checkpoint)
            print(json.dumps({"event": "progress", "next_source_row": cursor,
                              "elapsed_seconds": round(time.monotonic() - began, 1)}), flush=True)
        sample_count = min(max(args.verify_samples, 0), args.end - args.start)
        if sample_count:
            rows = np.linspace(args.start, args.end - 1, sample_count, dtype=np.int64)
            expected_ids = [int(row) + 1 for row in rows]
            retrieved = request_json("POST", root + "/points",
                                     {"ids": expected_ids, "with_payload": False,
                                      "with_vector": True})
            by_id = {int(point["id"]): point for point in retrieved["result"]}
            if set(by_id) != set(expected_ids):
                raise RuntimeError("source verification returned missing or unexpected IDs")
            max_error = 0.0
            for row, point_id in zip(rows, expected_ids):
                original = np.asarray(matrix[int(row)], dtype=np.float32)
                norm = float(np.linalg.norm(original))
                if not np.isfinite(norm) or norm == 0:
                    raise RuntimeError(f"invalid source norm for row {row}")
                original /= norm
                actual = np.asarray(by_id[point_id]["vector"], dtype=np.float32)
                if actual.shape != (768,) or not np.isfinite(actual).all():
                    raise RuntimeError(f"invalid retrieved vector for official ID {point_id}")
                error = float(np.max(np.abs(original - actual)))
                max_error = max(max_error, error)
                if error > 0.002:
                    raise RuntimeError(f"source mismatch for ID {point_id}: max error {error}")
            saved = json.loads(args.checkpoint.read_text())
            saved["source_verification"] = {"checked": sample_count,
                                             "max_abs_error": max_error,
                                             "tolerance": 0.002,
                                             "official_ids_1_based": expected_ids}
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            temporary.write_text(json.dumps(saved, indent=2) + "\n")
            os.replace(temporary, args.checkpoint)
    final = request_json("GET", root)["result"]
    expected_total = args.expected_total
    if expected_total is not None and int(final["points_count"]) != expected_total:
        raise RuntimeError(f"expected exactly {expected_total} points, got {final['points_count']}")
    print(json.dumps({"event": "complete", "points_count": final["points_count"],
                      "source_rows": args.end - args.start,
                      "source_id_first": args.start + 1, "source_id_last": args.end,
                      "source_verification": {"checked": sample_count,
                                               "max_abs_error": max_error if sample_count else None,
                                               "tolerance": 0.002},
                      "checkpoint": str(args.checkpoint)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
