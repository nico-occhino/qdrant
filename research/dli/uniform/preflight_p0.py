"""P0 infrastructure preflight (SCALE_SHARED_SPEC.md §P0). Not data.

    preflight_p0.py --scratch DIR --out FILE [--file-gib N]

P0.1 enforcement: the job's cgroup memory.max is read, a file of N GiB (> limit) is written and read back through
     the page cache, then mmap'd and touched (Qdrant's on-disk vector path). The job must survive, and the cgroup's
     memory must stay <= memory.max. (The anonymous over-limit test runs as a separate, hard-killed step in the job
     script, because a swapping child stalls this process too.)
P0.2 counters: memory.current/peak/stat/pressure/events, io.stat and the process's major faults are read.
P0.3 eviction: after reading the file, posix_fadvise(DONTNEED) on it must lower memory.stat `file`.
Every check is recorded as measured; `pass` is the conjunction of the P0.1–P0.3 checks.
"""
import argparse
import json
import mmap
import os
import socket
import sys
import time
from pathlib import Path

GIB = 1 << 30
CHUNK = 64 << 20


def cgroup_dir():
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    v2 = [l for l in lines if l.startswith("0::")]
    if v2:
        return "v2", Path("/sys/fs/cgroup") / v2[0][3:].lstrip("/")
    mem = [l for l in lines if ":memory:" in l]
    return "v1", (Path("/sys/fs/cgroup/memory") / mem[0].split(":", 2)[2].lstrip("/")) if mem else None


def read(p):
    try:
        return Path(p).read_text().strip()
    except OSError as e:
        return f"ERROR {e.__class__.__name__}"


def kv(text):
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


class Cg:
    def __init__(self):
        self.version, self.dir = cgroup_dir()

    def f(self, v2name, v1name=None):
        name = v2name if self.version == "v2" else (v1name or v2name)
        return read(self.dir / name) if self.dir else "ERROR no cgroup"

    def current(self):
        v = self.f("memory.current", "memory.usage_in_bytes")
        return int(v) if v.isdigit() else None

    def limit(self):
        v = self.f("memory.max", "memory.limit_in_bytes")
        return int(v) if v.isdigit() else v

    def stat(self):
        return kv(self.f("memory.stat"))

    def events(self):
        return kv(self.f("memory.events", "memory.oom_control"))

    def snapshot(self):
        return {"current": self.current(), "peak": self.f("memory.peak", "memory.max_usage_in_bytes"),
                "stat_file": self.stat().get("file", self.stat().get("total_cache")),
                "stat_anon": self.stat().get("anon", self.stat().get("total_rss")),
                "stat_file_mapped": self.stat().get("file_mapped", self.stat().get("total_mapped_file")),
                "pressure": self.f("memory.pressure"), "events": self.events(), "io_stat": self.f("io.stat")}


def majflt():
    return int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[9])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--scratch", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--file-gib", type=float, default=12)
    args = a.parse_args()
    cg = Cg()
    rec = {"host": socket.gethostname(), "kernel": os.uname().release, "cgroup_version": cg.version,
           "cgroup_dir": str(cg.dir), "memory_max": cg.limit(), "controllers": read(Path(cg.dir or "/") / "cgroup.controllers"),
           "start": cg.snapshot(), "phases": {}}
    limit = rec["memory_max"] if isinstance(rec["memory_max"], int) else None
    path = args.scratch / "p0-file.bin"
    size = int(args.file_gib * GIB)
    peaks = []

    def sample():
        c = cg.current()
        if c is not None:
            peaks.append(c)

    t = time.time()
    buf = os.urandom(CHUNK)
    with open(path, "wb") as fh:
        for _ in range(size // CHUNK):
            fh.write(buf)
            sample()
        fh.flush()
        os.fsync(fh.fileno())
    print("phase", "write", time.strftime("%T"), file=sys.stderr, flush=True)
    rec["phases"]["write"] = {"seconds": time.time() - t, "after": cg.snapshot()}

    t, f0 = time.time(), majflt()
    with open(path, "rb", buffering=0) as fh:
        while fh.read(CHUNK):
            sample()
    print("phase", "read", time.strftime("%T"), file=sys.stderr, flush=True)
    rec["phases"]["read"] = {"seconds": time.time() - t, "after": cg.snapshot(), "majflt": majflt() - f0}

    fd = os.open(path, os.O_RDONLY)
    before = cg.stat().get("file", cg.stat().get("total_cache"))
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    time.sleep(1)
    after = cg.stat().get("file", cg.stat().get("total_cache"))
    print("phase", "fadvise_dontneed", time.strftime("%T"), file=sys.stderr, flush=True)
    rec["phases"]["fadvise_dontneed"] = {"file_before": before, "file_after": after}
    os.close(fd)

    t, f0 = time.time(), majflt()
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        acc = 0
        for off in range(0, size, 4096):
            acc ^= mm[off]
            if off % CHUNK == 0:
                sample()
    print("phase", "mmap_touch", time.strftime("%T"), file=sys.stderr, flush=True)
    rec["phases"]["mmap_touch"] = {"seconds": time.time() - t, "after": cg.snapshot(), "majflt": majflt() - f0}

    path.unlink()
    rec["peak_sampled"] = max(peaks) if peaks else None
    rec["end"] = cg.snapshot()

    checks = {
        "p0_1_limit_readable": limit is not None,
        "p0_1_file_exceeds_limit": limit is not None and size > limit,
        "p0_1_peak_within_limit": limit is not None and rec["peak_sampled"] is not None and rec["peak_sampled"] <= limit,
        "p0_1_survived_file_and_mmap": True,
        "p0_2_counters": all(not str(rec["end"][k]).startswith("ERROR")
                             for k in ("current", "stat_file", "stat_anon", "pressure", "io_stat")),
        "p0_3_fadvise_lowers_file": (before is not None and after is not None and after < before - GIB // 2),
    }
    rec["checks"] = checks
    rec["pass"] = all(checks.values())
    args.out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({"checks": checks, "pass": rec["pass"], "memory_max": rec["memory_max"],
                      "peak_sampled": rec["peak_sampled"]}, indent=1))


if __name__ == "__main__":
    main()
