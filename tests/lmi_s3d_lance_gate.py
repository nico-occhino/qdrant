"""Run one isolated, bounded Lance-to-Qdrant ingestion acceptance gate."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def request_json(method, url, body=None):
    payload = None if body is None else json.dumps(body).encode()
    request = Request(url, data=payload, method=method,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def directory_bytes(path):
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--scan-batch", type=int, default=512)
    parser.add_argument("--upsert-batch", type=int, default=128)
    parser.add_argument("--port", type=int, default=16633)
    args = parser.parse_args()
    if args.count not in (128, 10_000, 100_000):
        parser.error("count must be one of the three planned ingestion gates")
    for port in (args.port, args.port + 1):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    root = args.storage_root.resolve() / f"gate-{args.count}"
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite gate storage at {root}")
    root.mkdir(parents=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = root / "config.yaml"
    config.write_text(f"""storage:
  storage_path: {root}/storage
  snapshots_path: {root}/snapshots
  optimizers:
    default_segment_number: 1
    indexing_threshold_kb: 0
service:
  host: 127.0.0.1
  http_port: {args.port}
  grpc_port: {args.port + 1}
cluster:
  enabled: false
telemetry_disabled: true
""")
    server_log = (args.output / "server.log").open("w")
    process = subprocess.Popen([str(args.binary.resolve()), "--config-path", str(config),
                                "--disable-telemetry"], stdout=server_log,
                               stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{args.port}"
    collection = f"lmi_lance_gate_{args.count}"
    try:
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError(f"Qdrant exited {process.returncode}; see server.log")
            try:
                request_json("GET", base + "/collections")
                break
            except (URLError, TimeoutError):
                time.sleep(0.5)
        else:
            raise RuntimeError("server readiness timeout")
        lmi = {"n_buckets": 3162, "sample_size": 250000,
               "hidden_dim": 512, "epochs": 30, "batch_size": 256,
               "routing_batch_size": 256, "kmeans_iterations": 5,
               "nprobe": 4, "seed": 42}
        create = {"vectors": {"size": 768, "distance": "Cosine",
                              "on_disk": True, "lmi_config": lmi},
                  "hnsw_config": {"m": 0, "payload_m": 0},
                  "optimizers_config": {"indexing_threshold": 0,
                                        "default_segment_number": 1},
                  "shard_number": 1}
        result = request_json("PUT", base + "/collections/" + collection, create)
        if result.get("status") != "ok":
            raise RuntimeError(f"collection creation failed: {result}")
        started = time.monotonic()
        importer = Path(__file__).with_name("lmi_laion10m_lance_stream.py")
        checkpoint = args.output / "checkpoint.json"
        command = [sys.executable, str(importer), "--source", str(args.source),
                   "--url", base, "--collection", collection,
                   "--scan-batch", str(args.scan_batch),
                   "--upsert-batch", str(args.upsert_batch),
                   "--end", str(args.count), "--checkpoint", str(checkpoint),
                   "--server-pid", str(process.pid), "--disk-path", str(root),
                   "--verify-samples", "24"]
        with (args.output / "ingestion.jsonl").open("w") as log:
            ingest = subprocess.run(command, stdout=log, stderr=subprocess.PIPE, text=True)
        if ingest.returncode:
            (args.output / "ingestion-error.log").write_text(ingest.stderr)
            raise RuntimeError(f"ingestion failed with {ingest.returncode}")
        elapsed = time.monotonic() - started
        lines = [json.loads(line) for line in (args.output / "ingestion.jsonl").read_text().splitlines()]
        complete = lines[-1]
        if complete["event"] != "complete" or complete["next_source_row"] != args.count:
            raise AssertionError("ingestion did not complete requested rows")
        for _ in range(120):
            info = request_json("GET", base + "/collections/" + collection)["result"]
            if info["points_count"] == args.count and info["status"] == "green":
                break
            time.sleep(0.5)
        else:
            raise AssertionError(f"collection not green at expected count: {info}")
        config_after = info["config"]
        if config_after["params"]["vectors"]["lmi_config"] != lmi:
            raise AssertionError("collection LMI config changed during ingestion")
        if list((root / "storage").rglob("lmi_state.json")):
            raise AssertionError("LMI unexpectedly built during ingestion gate")
        progresses = [line for line in lines if line["event"] == "progress"]
        summary = {"gate_rows": args.count, "elapsed_seconds": elapsed,
                   "vectors_per_second": args.count / elapsed,
                   "scan_batch": args.scan_batch, "upsert_batch": args.upsert_batch,
                   "client_peak_rss_kib": max(row["client_peak_rss_kib"] for row in progresses),
                   "qdrant_peak_rss_kib": max(row["qdrant_peak_rss_kib"] or 0 for row in progresses),
                   "mem_available_low_kib": min(row["mem_available_low_kib"] for row in progresses),
                   "swap_used_high_kib": max(row["swap_used_high_kib"] for row in progresses),
                   "disk_free_low_bytes": min(row["disk_free_bytes"] for row in progresses),
                   "storage_bytes": directory_bytes(root / "storage"),
                   "points_count": info["points_count"],
                   "segments_count": info["segments_count"],
                   "collection_status": info["status"],
                   "verification": complete["verification"],
                   "optimizer_indexing_threshold": 0,
                   "vector_on_disk": True}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        server_log.close()


if __name__ == "__main__":
    main()
