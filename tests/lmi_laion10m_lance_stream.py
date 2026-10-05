"""Stream the named LAION 10M Lance corpus into a pre-created Qdrant collection.

Point IDs are zero-based Lance row positions, matching queries.lance neighbors.
No full-corpus materialization or alternate corpus file is created.
"""

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import lance
import numpy as np


def request_json(method, url, body=None, timeout=180):
    payload = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = Request(url, data=payload, method=method,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(4000).decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code} at {url}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"request failed at {url}: {error}") from error


def memory_kib():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemAvailable", "SwapTotal", "SwapFree"}:
            fields[name] = int(value.split()[0])
    return fields


def process_rss_kib(pid):
    if pid is None:
        return None
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError(f"VmRSS unavailable for process {pid}")


def check_resources(args):
    memory = memory_kib()
    available = memory["MemAvailable"]
    swap_used = memory["SwapTotal"] - memory["SwapFree"]
    if available < args.min_available_gib * (1 << 20):
        raise RuntimeError(f"MemAvailable below gate: {available} KiB")
    if swap_used > args.max_swap_used_gib * (1 << 20):
        raise RuntimeError(f"swap use above gate: {swap_used} KiB")
    free_bytes = shutil.disk_usage(args.disk_path).free if args.disk_path else None
    if free_bytes is not None and free_bytes < args.min_disk_free_gib * (1 << 30):
        raise RuntimeError(f"disk free below gate: {free_bytes} bytes")
    return memory, free_bytes


def verify_samples(dataset, url, collection, start, end, count):
    if not count or end <= start:
        return {"checked": 0}
    rng = np.random.default_rng(42)
    ids = sorted(set([start, end - 1] +
                     rng.integers(start, end, size=min(count, end - start)).tolist()))
    expected = dataset.take(ids, columns=["vector"])["vector"].combine_chunks()
    expected = expected.values.to_numpy(zero_copy_only=False).reshape(len(ids), 768)
    found = request_json("POST", f"{url}/collections/{collection}/points",
                         {"ids": ids, "with_vector": True, "with_payload": False})
    if found.get("status") != "ok":
        raise RuntimeError(f"retrieve failed: {found}")
    actual = {int(point["id"]): point["vector"] for point in found["result"]}
    if set(actual) != set(ids):
        raise RuntimeError(f"missing retrieved IDs: {sorted(set(ids) - set(actual))}")
    largest_error = 0.0
    for row_id, row in zip(ids, expected):
        value = np.asarray(actual[row_id], dtype=np.float32)
        if value.shape != (768,):
            raise RuntimeError(f"wrong vector dimension for point {row_id}")
        norm = float(np.linalg.norm(row))
        normalized = row if norm == 0 else row / norm
        error = float(np.max(np.abs(value - normalized)))
        largest_error = max(largest_error, error)
        if error > 2e-5:
            raise RuntimeError(f"stored vector differs at point {row_id}: {error}")
    return {"checked": len(ids), "max_abs_cosine_preprocessed_error": largest_error,
            "ids": ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--scan-batch", type=int, default=512)
    parser.add_argument("--upsert-batch", type=int, default=128)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--disk-path", type=Path)
    parser.add_argument("--min-available-gib", type=float, default=2.0)
    parser.add_argument("--max-swap-used-gib", type=float, default=2.0)
    parser.add_argument("--min-disk-free-gib", type=float, default=40.0)
    parser.add_argument("--verify-samples", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.scan_batch < 1 or not 1 <= args.upsert_batch <= args.scan_batch:
        parser.error("require 1 <= upsert-batch <= scan-batch")
    if not args.collection or "/" in args.collection:
        parser.error("collection must be a simple name")
    if args.disk_path and not args.disk_path.is_dir():
        parser.error("disk-path must be an existing directory")
    dataset = lance.dataset(str(args.source))
    if dataset.schema.names != ["vector"] or dataset.count_rows() != 10_000_000:
        raise ValueError("expected the named 10M single-vector Lance corpus")
    end = dataset.count_rows() if args.end is None else args.end
    if not 0 <= args.start <= end <= 10_000_000:
        parser.error("invalid row range")
    url = args.url.rstrip("/")
    start = args.start
    if args.checkpoint and args.checkpoint.exists():
        prior = json.loads(args.checkpoint.read_text())
        if prior["source"] != str(args.source.resolve()) or prior["collection"] != args.collection:
            raise ValueError("checkpoint source or collection mismatch")
        start = max(start, prior["next_source_row"])
    if start > end:
        parser.error("checkpoint is beyond requested end")
    if not args.dry_run:
        info = request_json("GET", f"{url}/collections/{args.collection}")
        if info.get("status") != "ok":
            raise RuntimeError(f"collection is unavailable: {info}")
    started = time.monotonic()
    sent = 0
    client_peak = 0
    qdrant_peak = 0
    available_low = 1 << 62
    swap_high = 0
    print(json.dumps({"event": "start", "source": str(args.source),
                      "source_rows": 10_000_000, "start": start, "end": end,
                      "scan_batch": args.scan_batch, "upsert_batch": args.upsert_batch,
                      "collection": args.collection, "dry_run": args.dry_run}), flush=True)
    for batch in dataset.to_batches(columns=["vector"], batch_size=args.scan_batch,
                                    batch_readahead=1, fragment_readahead=1,
                                    scan_in_order=True, offset=start, limit=end - start):
        block = batch.column(0).values.to_numpy(zero_copy_only=False)
        block = block.reshape(batch.num_rows, 768)
        if not np.isfinite(block).all():
            raise ValueError(f"non-finite vector near source row {start + sent}")
        chunk_first = start + sent
        for offset in range(0, batch.num_rows, args.upsert_batch):
            rows = block[offset:offset + args.upsert_batch]
            points = [{"id": chunk_first + offset + i, "vector": row.tolist()}
                      for i, row in enumerate(rows)]
            if not args.dry_run:
                response = request_json("PUT", f"{url}/collections/{args.collection}/points?wait=true",
                                        {"points": points})
                if response.get("status") != "ok" or response.get("result", {}).get("status") != "completed":
                    raise RuntimeError(f"upsert did not complete at row {chunk_first + offset}: {response}")
        sent += batch.num_rows
        next_row = start + sent
        memory, free_bytes = check_resources(args)
        available_low = min(available_low, memory["MemAvailable"])
        swap_high = max(swap_high, memory["SwapTotal"] - memory["SwapFree"])
        client_peak = max(client_peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        qdrant_rss = process_rss_kib(args.server_pid)
        if qdrant_rss is not None:
            qdrant_peak = max(qdrant_peak, qdrant_rss)
        if args.checkpoint:
            value = {"source": str(args.source.resolve()), "collection": args.collection,
                     "next_source_row": next_row}
            temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
            temporary.write_text(json.dumps(value))
            os.replace(temporary, args.checkpoint)
        elapsed = max(time.monotonic() - started, 1e-9)
        print(json.dumps({"event": "progress", "next_source_row": next_row,
                          "vectors_sent": sent, "seconds": elapsed,
                          "vectors_per_second": sent / elapsed,
                          "client_peak_rss_kib": client_peak,
                          "qdrant_peak_rss_kib": qdrant_peak or None,
                          "mem_available_low_kib": available_low,
                          "swap_used_high_kib": swap_high,
                          "disk_free_bytes": free_bytes}), flush=True)
    verification = None if args.dry_run else verify_samples(
        dataset, url, args.collection, start, end, args.verify_samples)
    print(json.dumps({"event": "complete", "next_source_row": start + sent,
                      "vectors_sent": sent, "verification": verification}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Lance ingestion stopped: {error}", file=sys.stderr)
        raise
