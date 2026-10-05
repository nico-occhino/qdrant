"""Watch a long-running Qdrant build and stop it on sustained memory pressure."""

import argparse
import json
import os
import shutil
import signal
import time
from pathlib import Path


def meminfo():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemAvailable", "SwapTotal", "SwapFree"}:
            fields[name] = int(value.split()[0])
    return fields


def rss_kib(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        if any(line.startswith("State:") and "zombie" in line for line in lines):
            return None
        for line in lines:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-disk", type=Path, default=Path("/mnt/c"))
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--min-available-gib", type=float, default=2.0)
    parser.add_argument("--max-swap-used-gib", type=float, default=2.0)
    parser.add_argument("--min-host-free-gib", type=float, default=40.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    unsafe_since = None
    peak_rss = 0
    available_low = 1 << 62
    swap_high = 0
    host_free_low = 1 << 62
    stopped_for_safety = False
    with args.output.open("x") as output:
        while True:
            rss = rss_kib(args.pid)
            if rss is None:
                break
            memory = meminfo()
            available = memory["MemAvailable"]
            swap_used = memory["SwapTotal"] - memory["SwapFree"]
            host_free = shutil.disk_usage(args.host_disk).free
            storage_free = shutil.disk_usage(args.storage).free
            peak_rss = max(peak_rss, rss)
            available_low = min(available_low, available)
            swap_high = max(swap_high, swap_used)
            host_free_low = min(host_free_low, host_free)
            elapsed = time.monotonic() - started
            output.write(json.dumps({"seconds": elapsed, "qdrant_rss_kib": rss,
                                     "mem_available_kib": available,
                                     "swap_used_kib": swap_used,
                                     "host_free_bytes": host_free,
                                     "storage_free_bytes": storage_free}) + "\n")
            output.flush()
            unsafe = (available < args.min_available_gib * (1 << 20) or
                      swap_used > args.max_swap_used_gib * (1 << 20) or
                      host_free < args.min_host_free_gib * (1 << 30))
            if unsafe:
                unsafe_since = elapsed if unsafe_since is None else unsafe_since
                if elapsed - unsafe_since >= 5:
                    os.kill(args.pid, signal.SIGTERM)
                    stopped_for_safety = True
                    break
            else:
                unsafe_since = None
            time.sleep(args.interval)
    summary = {"pid": args.pid, "elapsed_seconds": time.monotonic() - started,
               "qdrant_peak_rss_kib": peak_rss,
               "mem_available_low_kib": available_low,
               "swap_used_high_kib": swap_high,
               "host_free_low_bytes": host_free_low,
               "stopped_for_safety": stopped_for_safety}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if stopped_for_safety:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
