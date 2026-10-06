"""Evaluate an ordinary Qdrant query run against audited SISAP 2023 HDF5 gold."""

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import numpy as np


MARKER = re.compile(r"candidate_source=StaticLearned candidate_count=(\d+)")


def request_json(url, body):
    data = json.dumps(body, separators=(",", ":")).encode()
    request = Request(url, data=data, method="POST",
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def rows(dataset, count):
    value = dataset[:]
    if value.shape[0] == count:
        return value
    if value.ndim == 2 and value.shape[1] == count:
        return value.T
    raise ValueError(f"cannot orient {dataset.name} with shape {value.shape} for {count} queries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-key", required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--knns-key", default="knns")
    parser.add_argument("--dists-key", default="dists")
    parser.add_argument("--url", default="http://127.0.0.1:17037")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-queries", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--expected-points", type=int, default=300000)
    args = parser.parse_args()
    raw_path = args.output
    summary_path = raw_path.with_suffix(".summary.json")
    if raw_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite existing SISAP evaluation output")
    with h5py.File(args.queries, "r") as qfile, h5py.File(args.gold, "r") as gfile:
        queries = qfile[args.query_key]
        count = min(args.limit_queries, queries.shape[0] if queries.shape[0] == 10000 else
                    queries.shape[1] if queries.ndim == 2 and queries.shape[1] == 10000 else -1)
        if count < 1:
            raise ValueError(f"cannot identify 10,000-query axis in {args.query_key}: {queries.shape}")
        query_rows = rows(queries, count)
        gold_ids = rows(gfile[args.knns_key], count)
        gold_dists = rows(gfile[args.dists_key], count)
    if gold_ids.shape != gold_dists.shape:
        raise ValueError(f"gold IDs/distances disagree: {gold_ids.shape} vs {gold_dists.shape}")
    if gold_ids.size and (int(np.min(gold_ids)) < 1
                          or int(np.max(gold_ids)) > args.expected_points):
        raise ValueError(f"gold IDs must be 1-based corpus rows in 1..{args.expected_points}")
    if not np.isfinite(gold_dists).all():
        raise ValueError("gold distances contain non-finite values")
    base = args.url.rstrip("/") + "/collections/" + args.collection
    info = json.load(urlopen(base, timeout=30))["result"]
    if info["points_count"] != args.expected_points:
        raise RuntimeError(f"expected {args.expected_points} vectors, found {info['points_count']}")
    warmup = min(args.warmup, count)
    for query in query_rows[:warmup]:
        request_json(base + "/points/query", {"query": np.asarray(query, dtype=np.float32).tolist(),
                     "limit": 10, "with_payload": False, "with_vector": False})
    log_position = args.server_log.stat().st_size
    observations = []
    measured_start_ns = time.perf_counter_ns()
    with raw_path.open("x", encoding="utf-8") as raw:
        for row in range(warmup, count):
            vector = np.asarray(query_rows[row], dtype=np.float32)
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError(f"invalid query row {row}: {vector.shape}")
            start = time.perf_counter_ns()
            reply = request_json(base + "/points/query", {"query": vector.tolist(), "limit": 10,
                                 "with_payload": False, "with_vector": False})
            elapsed = time.perf_counter_ns() - start
            points = reply["result"]["points"]
            returned = [int(point["id"]) for point in points]
            expected = set(map(int, gold_ids[row][:10]))
            recall = len(set(returned) & expected) / 10
            with args.server_log.open("rb") as log:
                log.seek(log_position)
                new = log.read().decode(errors="replace")
                log_position = log.tell()
            markers = [int(match.group(1)) for line in new.splitlines()
                       if (match := MARKER.search(line))]
            if len(markers) != 1:
                raise RuntimeError(f"query row {row}: expected one learned marker, got {len(markers)}")
            record = {"query_row": row, "latency_http_ns": elapsed,
                      "result_ids_1_based": returned,
                      "result_scores": [float(point["score"]) for point in points],
                      "gold_ids_1_based": list(map(int, gold_ids[row][:10])),
                      "gold_distances": list(map(float, gold_dists[row][:10])),
                      "recall10_id_overlap": recall, "candidate_count": markers[0],
                      "candidate_fraction": markers[0] / args.expected_points}
            observations.append(record)
            raw.write(json.dumps(record, separators=(",", ":")) + "\n")
            raw.flush()
            if len(observations) % 100 == 0:
                print(json.dumps({"event": "progress", "measured": len(observations)}), flush=True)
    measured_sweep_wall_seconds = (time.perf_counter_ns() - measured_start_ns) / 1e9
    lat = [item["latency_http_ns"] / 1e6 for item in observations]
    candidate = [item["candidate_count"] for item in observations]
    summary = {"scope": "Qdrant HTTP end-to-end; not an in-process component timer",
               "queries_total": count, "warmup": warmup, "measured": len(observations),
               "measured_sweep_wall_seconds": measured_sweep_wall_seconds,
               "gold_metric": "conventional top-10 ID overlap; no tie expansion",
               "official_ids": "1-based, passed through unchanged",
               "recall10_mean": statistics.mean(x["recall10_id_overlap"] for x in observations),
               "http_latency_ms": {f"p{p}": float(np.quantile(lat, p / 100))
                                   for p in (50, 95, 99)},
               "candidate_count": {f"p{p}": float(np.quantile(candidate, p / 100))
                                  for p in (50, 95, 99)},
               "candidate_fraction_mean": statistics.mean(candidate) / args.expected_points,
               "router_and_scoring_component_timings": "not observable from HTTP"}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
