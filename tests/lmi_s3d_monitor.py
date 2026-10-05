"""Run one scale command with RSS, MemAvailable, swap, and disk monitoring."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemAvailable", "SwapTotal", "SwapFree"}:
            values[name] = int(value.split()[0])
    return values


def rss_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--min-available-gib", type=float, default=2.0)
    parser.add_argument("--max-swap-used-gib", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.output.exists():
        parser.error("command required and output must not already exist")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    with (args.output / "command.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        peak_rss = 0
        min_available = 1 << 62
        max_swap = 0
        min_disk = 1 << 62
        unsafe_since = None
        stopped_for_safety = False
        with (args.output / "monitor.jsonl").open("w") as observations:
            while process.poll() is None:
                memory = meminfo()
                swap = memory["SwapTotal"] - memory["SwapFree"]
                available = memory["MemAvailable"]
                disk = shutil.disk_usage(args.disk_path).free
                rss = rss_kib(process.pid)
                peak_rss = max(peak_rss, rss)
                min_available = min(min_available, available)
                max_swap = max(max_swap, swap)
                min_disk = min(min_disk, disk)
                now = time.monotonic()
                observations.write(json.dumps({"seconds": now - started, "rss_kib": rss,
                                               "mem_available_kib": available,
                                               "swap_used_kib": swap,
                                               "disk_free_bytes": disk}) + "\n")
                observations.flush()
                unsafe = (available < args.min_available_gib * (1 << 20) or
                          swap > args.max_swap_used_gib * (1 << 20))
                if unsafe:
                    unsafe_since = now if unsafe_since is None else unsafe_since
                    if now - unsafe_since >= 5:
                        os.killpg(process.pid, signal.SIGTERM)
                        stopped_for_safety = True
                        break
                else:
                    unsafe_since = None
                time.sleep(args.interval)
        exit_code = process.wait(timeout=60)
    result = {"command": command, "exit_code": exit_code,
              "elapsed_seconds": time.monotonic() - started,
              "process_peak_rss_kib": peak_rss,
              "mem_available_low_kib": min_available,
              "swap_used_high_kib": max_swap,
              "disk_free_low_bytes": min_disk,
              "stopped_for_safety": stopped_for_safety}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if exit_code != 0 or stopped_for_safety:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
