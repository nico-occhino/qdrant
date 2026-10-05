"""Start, inspect, or stop the isolated LAION 10M Qdrant scale instance."""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(method, url, body=None):
    payload = None if body is None else json.dumps(body).encode()
    request = Request(url, data=payload, method=method,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def alive(pid):
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return state not in {"Z", "X"}
    except (ProcessLookupError, FileNotFoundError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "stop"])
    parser.add_argument("--binary", type=Path, default=Path("target/release/qdrant"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=17033)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    pid_file = args.output / "server.pid"
    base = f"http://127.0.0.1:{args.port}"
    collection = "laion10m_lmi"
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
        fresh = not root.exists()
        if not fresh and not (root / "config.yaml").is_file():
            raise RuntimeError(f"existing root has no expected config: {root}")
        for port in (args.port, args.port + 1):
            with socket.socket() as probe:
                probe.settimeout(1)
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"port {port} already has a listener")
        if fresh:
            root.mkdir(parents=True)
        config = root / "config.yaml"
        if fresh:
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
        with (args.output / "server.log").open("a") as log:
            process = subprocess.Popen([str(args.binary.resolve()), "--config-path", str(config),
                                        "--disable-telemetry"], stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        pid = process.pid
        pid_file.write_text(str(pid))
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
        config = {"n_buckets": 3162, "sample_size": 250000,
                  "hidden_dim": 512, "epochs": 30, "batch_size": 256,
                  "routing_batch_size": 256, "kmeans_iterations": 5,
                  "nprobe": 4, "seed": 42}
        create = {"vectors": {"size": 768, "distance": "Cosine",
                              "on_disk": True, "lmi_config": config},
                  "hnsw_config": {"m": 0, "payload_m": 0},
                  "optimizers_config": {"indexing_threshold": 0,
                                        "default_segment_number": 1},
                  "shard_number": 1}
        if fresh or collection not in [entry["name"] for entry in request_json("GET", base + "/collections")["result"]["collections"]]:
            result = request_json("PUT", base + "/collections/" + collection, create)
            if result.get("status") != "ok":
                raise RuntimeError(f"collection creation failed: {result}")
            (args.output / "initial-config.json").write_text(json.dumps(create, indent=2))
    if not pid or not alive(pid):
        raise RuntimeError("instance is not running")
    info = request_json("GET", base + "/collections/" + collection)
    result = {"pid": pid, "alive": True, "port": args.port,
              "collection": collection, "points_count": info["result"]["points_count"],
              "segments_count": info["result"]["segments_count"],
              "status": info["result"]["status"],
              "indexing_threshold": info["result"]["config"]["optimizer_config"]["indexing_threshold"],
              "root": str(root)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
