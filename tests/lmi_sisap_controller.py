"""Manage a fresh isolated Float16 SISAP Qdrant instance and its collection."""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen


def request(method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, method=method,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().split()[2] not in {"Z", "X"}
    except (ProcessLookupError, FileNotFoundError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "stop"])
    parser.add_argument("--binary", type=Path, default=Path("target/release/qdrant"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=["300k", "10m"], default="300k")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pid_file = output / "server.pid"
    base = f"http://127.0.0.1:{args.port}"
    collection = f"sisap2023_{args.mode}_f16_lmi"
    pid = int(pid_file.read_text()) if pid_file.exists() else None
    if args.action == "stop":
        if pid and alive(pid):
            os.kill(pid, signal.SIGTERM)
            for _ in range(120):
                if not alive(pid):
                    break
                time.sleep(0.5)
        print(json.dumps({"pid": pid, "alive": bool(pid and alive(pid))}))
        return
    if args.action == "start" and not (pid and alive(pid)):
        required_environment = {
            "QDRANT_LMI_SAMPLE_BUDGET_BYTES": "800000000",
            "LMI_EXPERIMENTAL_TCH_BUILD_ROUTING": "1",
        }
        if args.mode == "10m":
            missing = {key: value for key, value in required_environment.items()
                       if os.environ.get(key) != value}
            if missing:
                raise RuntimeError(
                    "10m server must be launched with the required LMI environment "
                    f"already set: {sorted(missing)}")
            if os.environ.get("LMI_EXPERIMENTAL_TWO_PASS_POSTINGS"):
                raise RuntimeError("10m server must not enable the two-pass postings path")
        fresh = not root.exists()
        config_path = root / "config.yaml"
        if not fresh and not config_path.is_file():
            raise RuntimeError(f"existing root has no expected config: {root}")
        for port in (args.port, args.port + 1):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"port {port} already has a listener")
        root.mkdir(parents=True, exist_ok=True)
        expected_config = f"""storage:
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
"""
        if fresh:
            config_path.write_text(expected_config)
        elif config_path.read_text() != expected_config:
            raise RuntimeError("existing storage config differs from this SISAP instance")
        log = (output / "server.log").open("a", encoding="utf-8")
        process = subprocess.Popen([str(args.binary.resolve()), "--config-path", str(config_path),
                                    "--disable-telemetry"], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        pid, _ = process.pid, pid_file.write_text(str(process.pid))
        if args.mode == "10m":
            env_items = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            inherited = {}
            for item in env_items:
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                key = key.decode(errors="replace")
                if key in {"QDRANT_LMI_SAMPLE_BUDGET_BYTES",
                           "LMI_EXPERIMENTAL_TCH_BUILD_ROUTING",
                           "LMI_EXPERIMENTAL_TWO_PASS_POSTINGS"}:
                    inherited[key] = value.decode(errors="replace")
            expected = {"QDRANT_LMI_SAMPLE_BUDGET_BYTES": "800000000",
                        "LMI_EXPERIMENTAL_TCH_BUILD_ROUTING": "1"}
            if any(inherited.get(key) != value for key, value in expected.items()) \
                    or inherited.get("LMI_EXPERIMENTAL_TWO_PASS_POSTINGS"):
                process.terminate()
                process.wait(timeout=10)
                raise RuntimeError("Qdrant environment verification failed: " + json.dumps(inherited))
            (output / "server-environment.json").write_text(
                json.dumps(inherited, indent=2) + "\n")
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError(f"Qdrant exited {process.returncode}; inspect {output/'server.log'}")
            try:
                request("GET", base + "/collections")
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise TimeoutError("Qdrant did not become ready")
        if args.mode == "300k":
            lmi = {"n_buckets": 548, "sample_size": 32768, "hidden_dim": 512,
                   "epochs": 30, "batch_size": 256, "routing_batch_size": 256,
                   "kmeans_iterations": 5, "nprobe": 4, "seed": 42}
        else:
            lmi = {"n_buckets": 3162, "sample_size": 250000, "hidden_dim": 512,
                   "epochs": 30, "batch_size": 256, "routing_batch_size": 256,
                   "kmeans_iterations": 5, "nprobe": 4, "seed": 42}
        body = {"vectors": {"size": 768, "distance": "Cosine", "datatype": "float16",
                            "on_disk": True, "lmi_config": lmi},
                "hnsw_config": {"m": 0, "payload_m": 0},
                "optimizers_config": {"indexing_threshold": 0,
                                      "default_segment_number": 1,
                                      "max_optimization_threads": 1},
                "shard_number": 1}
        listed = request("GET", base + "/collections")["result"]["collections"]
        existing = next((item for item in listed if item["name"] == collection), None)
        if existing is None:
            created = request("PUT", base + "/collections/" + collection, body)
            if created.get("status") != "ok":
                raise RuntimeError(f"SISAP collection creation failed: {created}")
            (output / "initial-config.json").write_text(json.dumps(body, indent=2) + "\n")
    if not pid or not alive(pid):
        raise RuntimeError("requested SISAP instance is not running")
    info = request("GET", base + "/collections/" + collection)["result"]
    print(json.dumps({"pid": pid, "port": args.port, "collection": collection,
                      "mode": args.mode, "status": info["status"],
                      "points_count": info["points_count"],
                      "segments_count": info["segments_count"],
                      "indexed_vectors_count": info["indexed_vectors_count"],
                      "root": str(root), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
