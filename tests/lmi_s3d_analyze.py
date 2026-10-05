"""Aggregate preserved per-query native 10M LMI observations by nprobe."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def describe(values):
    return {"mean": statistics.mean(values),
            "p50": float(np.quantile(values, .50)),
            "p95": float(np.quantile(values, .95)),
            "p99": float(np.quantile(values, .99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON summary; a CSV with the same stem is also written")
    args = parser.parse_args()
    csv_path = args.output.with_suffix(".csv")
    if args.output.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite preserved results")
    grouped = defaultdict(list)
    with args.raw.open() as source:
        for line in source:
            row = json.loads(line)
            grouped[int(row["nprobe"])].append(row)
    if sorted(grouped) != [1, 2, 4, 8, 16]:
        raise ValueError("expected all five nprobe operating points")
    expected_rows = set(range(35, 4992, 25))
    result = []
    for probe in sorted(grouped):
        rows = grouped[probe]
        if len(rows) != len(expected_rows) or {row["query_row"] for row in rows} != expected_rows:
            raise ValueError(f"missing or duplicate queries at nprobe={probe}")
        summary = {"nprobe": probe, "queries": len(rows),
                   "recall10_tie_aware": statistics.mean(row["recall10_tie_aware"] for row in rows),
                   "recall10_conventional": statistics.mean(row["recall10_conventional"] for row in rows),
                   "candidates": describe([row["candidate_count"] for row in rows]),
                   "candidate_fraction": describe([row["candidate_fraction"] for row in rows])}
        for component in ("route", "gather", "scoring_topk", "total"):
            summary[component + "_ms"] = describe(
                [row[component + "_ns"] / 1e6 for row in rows])
        result.append(summary)
    payload = {"scope": "in-process native Qdrant segment scoring, one trial per operating point, 199 systematic held-out queries",
               "query_selection": {"start": 35, "end": 4992, "stride": 25},
               "methodology": "20 separate warmups; per-query route, posting gather, scoring+topk, total; no HTTP; scores use BatchFilteredSearcher",
               "limitations": "scoring and top-k are not separately timed; results include OS cache effects and nprobe operating-point order",
               "rows": result}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    fields = ["nprobe", "queries", "recall10_tie_aware", "recall10_conventional",
              "candidate_mean", "candidate_p50", "candidate_p95", "candidate_p99",
              "candidate_fraction_mean"]
    for component in ("route", "gather", "scoring_topk", "total"):
        fields += [f"{component}_ms_{stat}" for stat in ("mean", "p50", "p95", "p99")]
    with csv_path.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in result:
            flat = {"nprobe": row["nprobe"], "queries": row["queries"],
                    "recall10_tie_aware": row["recall10_tie_aware"],
                    "recall10_conventional": row["recall10_conventional"],
                    "candidate_fraction_mean": row["candidate_fraction"]["mean"]}
            for stat in ("mean", "p50", "p95", "p99"):
                flat["candidate_" + stat] = row["candidates"][stat]
                for component in ("route", "gather", "scoring_topk", "total"):
                    flat[f"{component}_ms_{stat}"] = row[component + "_ms"][stat]
            writer.writerow(flat)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
