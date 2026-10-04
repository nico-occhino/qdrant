"""Bounded HDF5-to-Qdrant REST importer for an approved remote host.

Run only after source checksum, collection configuration, and storage admission
have been verified. It never loads the full corpus and never creates storage.
"""

import argparse
import json
import resource
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--hdf5-chunk", type=int, default=512)
    parser.add_argument("--api-batch", type=int, default=64)
    parser.add_argument("--start", type=int, default=0, help="zero-based inclusive row")
    parser.add_argument("--end", type=int, help="zero-based exclusive row")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.hdf5_chunk < 1 or args.api_batch < 1 or args.api_batch > args.hdf5_chunk:
        parser.error("require 1 <= api-batch <= hdf5-chunk")
    if not args.collection or "/" in args.collection:
        parser.error("collection must be a simple name")
    endpoint = args.url.rstrip("/") + f"/collections/{args.collection}/points?wait=true"
    started = time.monotonic()
    sent = 0
    source_bytes = 0
    with h5py.File(args.source, "r") as handle:
        if "emb" not in handle:
            raise ValueError("expected /emb dataset; inspect source before ingesting")
        matrix = handle["emb"]
        if len(matrix.shape) != 2 or matrix.shape[1] != 768:
            raise ValueError(f"expected N x 768 embeddings, got {matrix.shape}")
        if matrix.dtype != np.dtype("float16"):
            raise ValueError(f"expected float16 source, got {matrix.dtype}")
        end = matrix.shape[0] if args.end is None else args.end
        if not 0 <= args.start <= end <= matrix.shape[0]:
            parser.error("invalid start/end range")
        print(json.dumps({"source": str(args.source), "shape": matrix.shape,
                          "start": args.start, "end": end, "dry_run": args.dry_run,
                          "hdf5_chunk": args.hdf5_chunk, "api_batch": args.api_batch}),
              flush=True)
        for chunk_start in range(args.start, end, args.hdf5_chunk):
            chunk_end = min(end, chunk_start + args.hdf5_chunk)
            block = np.asarray(matrix[chunk_start:chunk_end], dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError(f"non-finite embedding at chunk {chunk_start}")
            source_bytes += (chunk_end - chunk_start) * 768 * 2
            for batch_start in range(0, len(block), args.api_batch):
                batch_end = min(len(block), batch_start + args.api_batch)
                first_row = chunk_start + batch_start
                points = [
                    {"id": first_row + offset + 1, "vector": row.tolist()}
                    for offset, row in enumerate(block[batch_start:batch_end])
                ]
                payload = json.dumps({"points": points}, separators=(",", ":")).encode()
                if not args.dry_run:
                    request = Request(endpoint, data=payload, method="PUT",
                                      headers={"Content-Type": "application/json"})
                    try:
                        with urlopen(request, timeout=180) as response:
                            result = json.load(response)
                    except (HTTPError, URLError, TimeoutError) as error:
                        raise RuntimeError(f"upsert failed at zero-based row {first_row}") from error
                    if result.get("status") != "ok" or result.get("result", {}).get("status") != "completed":
                        raise RuntimeError(f"upsert not completed at zero-based row {first_row}: {result}")
                sent += batch_end - batch_start
            elapsed = max(time.monotonic() - started, 1e-9)
            print(json.dumps({"next_zero_based_row": chunk_end, "sent": sent,
                              "elapsed_seconds": elapsed, "vectors_per_second": sent / elapsed,
                              "source_bytes_per_second": source_bytes / elapsed,
                              "client_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}), flush=True)


if __name__ == "__main__":
    main()
