"""Gate 2 (Q5_PREREGISTRATION.md §6): two dli_phase3_workload runs are identical except for timing and
segment UUIDs.

    compare_runs.py RUN_A RUN_B [--out FILE]

The comparison covers:
* every row of every `batch_XX/rows.jsonl` (in order), minus `elapsed_ns`;
* every `topology.json` member (level, live, buckets, bucket sizes), in order;
* every `batches.jsonl` record's levels (level, live, buckets, router) and publications (target level,
  expected and published points);
* every `build-events.jsonl` `build_completed` record (points, buckets, reused router, router fingerprint).
Exit status 0 only if all are identical.
"""
import json
import sys
from pathlib import Path

TIMING = {"elapsed_ns"}


def rows(run):
    out = []
    for f in sorted(Path(run).glob("batch_*/rows.jsonl")):
        for line in open(f):
            r = json.loads(line)
            out.append((f.parent.name, {k: v for k, v in r.items() if k not in TIMING}))
    return out


def topology(run):
    out = []
    for f in sorted(Path(run).glob("batch_*/topology.json")):
        t = json.loads(f.read_text())
        out.append((f.parent.name, [(m["level"], m["live"], m["buckets"], m["bucket_sizes"]) for m in t["members"]]))
    return out


def ledger(run):
    out = []
    for line in open(Path(run) / "batches.jsonl"):
        b = json.loads(line)
        out.append((b["batch"], b["live"], b["live_sha256"],
                    [(x["level"], x["live"], x["buckets"], x["router"]) for x in b["levels"]],
                    [(p["target_level"], p["expected_output_points"], p["published_output_points"])
                     for p in b["publications"]]))
    return out


def builds(run):
    out = []
    for line in open(Path(run) / "build-events.jsonl"):
        e = json.loads(line)
        if e.get("stage") == "build_completed":
            out.append((e["physical_count"], e["n_buckets"], e["reused_router"], e["router_fingerprint"]))
    return out


def main():
    a, b = sys.argv[1], sys.argv[2]
    report = {}
    for name, f in (("rows", rows), ("topology", topology), ("ledger", ledger), ("builds", builds)):
        x, y = f(a), f(b)
        first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
        report[name] = {"count_a": len(x), "count_b": len(y), "identical": x == y and len(x) > 0,
                        "first_difference": first}
    report["pass"] = all(v["identical"] for v in report.values())
    text = json.dumps(report, indent=1)
    if len(sys.argv) > 4 and sys.argv[3] == "--out":
        Path(sys.argv[4]).write_text(text + "\n")
    print(text)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
