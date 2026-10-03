#!/usr/bin/env python3
"""Summarize preserved S.3B1 raw runs; never rewrites source observations."""
import csv
import hashlib
import json
import statistics
import os
from pathlib import Path

ROOT = Path("work/phase_s3")
OUT = Path(os.environ.get("LMI_S3B1_RUN_ROOT", ROOT / "s3b1-continuation"))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def posting_times(path):
    import re

    line = next(x for x in path.read_text().splitlines() if "postings: segment_slots=" in x)
    return {key: float(re.search(key + r"=([0-9.]+)", line).group(1)) for key in (
        "pass1_seconds", "allocation_seconds", "pass2_seconds"
    )}


runs = [read(OUT / f"micro-e-{trial}.json") for trial in (1, 2, 3)]
rows = []
for shape in "ABCDE":
    for k in (1, 16, 64, 256):
        points = [next(x for x in run if x["shape"] == shape and x["k"] == k) for run in runs]
        row = {key: points[0][key] for key in (
            "shape", "d", "h", "b", "rows", "k", "workspace_bytes",
            "native_compute_threads", "process_threads"
        )}
        row.update({
            "rows_per_second_median": statistics.median(x["rows_per_second"] for x in points),
            "microseconds_per_vector_median": statistics.median(x["microseconds_per_vector"] for x in points),
            "rows_per_second_min": min(x["rows_per_second"] for x in points),
            "rows_per_second_max": max(x["rows_per_second"] for x in points),
            "assignment_mismatches": sum(x["assignment_mismatches"] for x in points),
            "rows_compared": sum(x["rows"] for x in points),
        })
        rows.append(row)

baseline = ROOT / "laion-top1"
base_result = read(baseline / "result.json")
laion = []
for k in (1, 16, 256):
    path = OUT / f"laion-k{k}"
    result = read(path / "result.json")
    timings = posting_times(OUT / f"laion-k{k}.log")
    laion.append({
        "k": k,
        "count_seconds": timings["pass1_seconds"],
        "fill_seconds": timings["pass2_seconds"],
        "combined_seconds": timings["pass1_seconds"] + timings["pass2_seconds"],
        "whole_build_seconds": result["total_build_seconds"],
        "router_equal": digest(path / "lmi_router.bin") == digest(baseline / "lmi_router.bin"),
        "postings_equal": digest(path / "lmi_postings.bin") == digest(baseline / "lmi_postings.bin"),
        "query_results_equal": result["query_results"] == base_result["query_results"],
    })

summary = {
    "revision": (OUT / "revision.txt").read_text().strip() if (OUT / "revision.txt").exists() else "313d7de2a",
    "micro": rows,
    "stage_profile_d": read(OUT / "profile-d.json"),
    "laion": laion,
    "source": "three separate CPU-0 perf-profile runs; raw micro-e-1..3.json and laion-k*/result.json preserved",
}
assert not (OUT / "summary.json").exists()
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
assert not (OUT / "micro-summary.csv").exists()
with (OUT / "micro-summary.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"micro_rows": len(rows), "laion": laion}, indent=2))
