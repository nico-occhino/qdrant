"""Summarise the concurrent-service phase (H4, descriptive) from service-batches.jsonl files.

    summarize_service.py ARM=DIR ... [--out FILE]

Per arm: batches; requests issued/ok/timeouts/errors/dropped; achieved QPS (mean over
batches); pooled latency percentiles are NOT recomputed from per-batch percentiles; the
report gives per-batch p50/p95/p99/max and their median and worst; deleted-point
violations; maintenance busy seconds; DLI learned/fallback deltas; settled recall; CPU
seconds and peak RSS over the run (from /proc counters; CLK_TCK 100).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(directory):
    rows = [json.loads(l) for l in open(Path(directory) / "service-batches.jsonl")]
    w = [r["window"] for r in rows]
    lat = {k: [x["latency_ms"][k] for x in w if x["latency_ms"]] for k in ("p50", "p95", "p99", "max")}
    lf = [r["dli_learned_fallback_before_after"] for r in rows]
    learned = fallback = None
    if all(x[0] and x[1] for x in lf):
        learned = sum(b[0] - a[0] for a, b in lf)
        fallback = sum(b[1] - a[1] for a, b in lf)
    c0, c1 = rows[0]["counters_before"], rows[-1]["counters_after"]
    cpu = ((c1["utime_ticks"] or 0) + (c1["stime_ticks"] or 0) - (c0["utime_ticks"] or 0) - (c0["stime_ticks"] or 0)) / 100
    return {
        "arm": rows[0]["arm"], "batches": len(rows),
        "issued": sum(x["issued"] for x in w), "ok": sum(x["ok"] for x in w),
        "timeouts": sum(x["timeouts"] for x in w), "errors": sum(x["errors"] for x in w),
        "dropped_at_cap": sum(x["dropped_at_cap"] for x in w),
        "achieved_qps_mean": float(np.mean([x["achieved_qps"] for x in w])),
        "latency_ms_median_of_batches": {k: float(np.median(v)) for k, v in lat.items() if v},
        "latency_ms_worst_batch": {k: float(np.max(v)) for k, v in lat.items() if v},
        "deleted_point_violations": sum(x["deleted_point_violations"] for x in w),
        "maintenance_busy_seconds": sum(x["maintenance_busy_seconds"] for x in w),
        "dli_learned_requests": learned, "dli_exact_fallback_requests": fallback,
        "settled_recall30_mean": float(np.mean([r["settled"]["recall30"] for r in rows])),
        "settled_latency_p50_median": float(np.median([r["settled"]["latency_ms"]["p50"] for r in rows])),
        "window_seconds_total": sum(x["seconds"] for x in w),
        "settle_seconds_total": sum(x["settle_seconds"] for x in w),
        "cpu_seconds": cpu, "peak_rss_kib": c1["hwm_kib"],
        "write_bytes": (c1["write_bytes"] or 0) - (c0["write_bytes"] or 0),
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument("arms", nargs="+", help="ARM=DIR")
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    out = {name: summarize(d) for name, d in (x.split("=", 1) for x in args.arms)}
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
