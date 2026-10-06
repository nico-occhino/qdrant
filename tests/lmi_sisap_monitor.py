"""Append Qdrant optimizer/build progress and Linux memory snapshots as JSONL."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    pid = int(args.pid_file.read_text())
    proc = Path(f"/proc/{pid}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        while proc.exists():
            status = {}
            for line in (proc / "status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, value, *_ = line.split()
                    status[key.rstrip(":")] = int(value) * 1024
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith(("MemAvailable:", "SwapFree:")):
                    key, value, *_ = line.split()
                    memory[key.rstrip(":")] = int(value) * 1024
            with urlopen(args.url.rstrip("/") + "/collections/" + args.collection,
                         timeout=30) as response:
                collection = json.load(response)["result"]
            postings = sorted(str(path) for path in args.segments_dir.rglob("lmi_postings.bin"))
            record = {"timestamp": datetime.now().astimezone().isoformat(), "pid": pid,
                      "process_bytes": status, "system_bytes": memory,
                      "points_count": collection.get("points_count"),
                      "segments_count": collection.get("segments_count"),
                      "indexed_vectors_count": collection.get("indexed_vectors_count"),
                      "optimizer_status": collection.get("optimizer_status"),
                      "postings_files": postings}
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            print(json.dumps(record), flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
