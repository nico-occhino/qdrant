"""Evaluate the native 10M LMI through ordinary Qdrant HTTP queries.

This measures end-to-end local HTTP latency. It does not claim to isolate
in-process routing, gathering, scoring, or top-k component timings.
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

import lance
import numpy as np


MARKER = re.compile(r"candidate_source=StaticLearned candidate_count=(\d+)")


def request_json(method, url, body=None):
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = Request(url, data=data, method=method,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def percentile(values, fraction):
    return float(np.quantile(values, fraction, method="linear"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:17033")
    parser.add_argument("--collection", default="laion10m_lmi")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4992)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError("refusing to overwrite preserved observations")
    if not 0 <= args.start < args.end <= 4992:
        parser.error("query range must fit the 4,992 Lance queries")
    if args.stride < 1:
        parser.error("stride must be positive")
    base = args.url.rstrip("/") + "/collections/" + args.collection
    info = request_json("GET", base)["result"]
    if info["points_count"] != 10_000_000:
        raise RuntimeError("collection does not yet contain exactly 10M points")
    dataset = lance.dataset(str(args.queries))
    if dataset.count_rows() != 4992:
        raise RuntimeError("unexpected query set")
    if not 0 <= args.warmup <= 100:
        parser.error("warmup must be in 0..100")
    if args.warmup:
        warmup = dataset.take(list(range(args.warmup)), columns=["vector"])["vector"]
        warmup_vectors = warmup.combine_chunks().values.to_numpy(zero_copy_only=False)
        warmup_vectors = warmup_vectors.reshape(args.warmup, 768)
        for query in warmup_vectors:
            request_json("POST", base + "/points/query",
                         {"query": query.tolist(), "limit": 10,
                          "with_payload": False, "with_vector": False})
    log_position = args.server_log.stat().st_size
    observations = []
    raw = args.output.open("x")
    query_row = args.start
    for batch in dataset.to_batches(columns=["query_id", "vector", "neighbors", "distances"],
                                    batch_size=32, offset=args.start,
                                    limit=args.end - args.start, scan_in_order=True):
        vectors = batch.column(1).values.to_numpy(zero_copy_only=False).reshape(batch.num_rows, 768)
        query_ids = batch.column(0).to_pylist()
        neighbors = batch.column(2).to_pylist()
        distances = batch.column(3).to_pylist()
        for index in range(batch.num_rows):
            current_row = query_row
            query_row += 1
            if (current_row - args.start) % args.stride:
                continue
            query = np.asarray(vectors[index], dtype=np.float32)
            if not np.isfinite(query).all():
                raise RuntimeError("non-finite query vector")
            began = time.perf_counter_ns()
            reply = request_json("POST", base + "/points/query",
                                 {"query": query.tolist(), "limit": 10,
                                  "with_payload": False, "with_vector": False})
            elapsed_ns = time.perf_counter_ns() - began
            if reply.get("status") != "ok":
                raise RuntimeError(f"query failed: {reply}")
            rows = reply["result"]["points"]
            if len(rows) != 10:
                raise RuntimeError(f"query returned {len(rows)} points")
            ids = [int(row["id"]) for row in rows]
            if len(set(ids)) != 10:
                raise RuntimeError("duplicate result ID")
            threshold = distances[index][9]
            tied = {int(point_id) for point_id, distance in zip(neighbors[index], distances[index])
                    if distance <= threshold}
            conventional = set(map(int, neighbors[index][:10]))
            with args.server_log.open("rb") as log:
                log.seek(log_position)
                new_log = log.read().decode(errors="replace")
                log_position = log.tell()
            markers = [int(match.group(1)) for line in new_log.splitlines()
                       if (match := MARKER.search(line))]
            if len(markers) != 1:
                raise RuntimeError(f"expected one learned-path marker for query "
                                   f"{query_ids[index]}, found {len(markers)}")
            observation = {
                "query_row": current_row,
                "query_id": int(query_ids[index]),
                "latency_http_ns": elapsed_ns,
                "result_ids": ids,
                "result_scores": [float(row["score"]) for row in rows],
                "tie_eligible_ground_truth_count": len(tied),
                "recall10_tie_aware": len(set(ids) & tied) / 10,
                "recall10_conventional": len(set(ids) & conventional) / 10,
                "candidate_count": markers[0],
                "candidate_fraction": markers[0] / 10_000_000,
            }
            observations.append(observation)
            raw.write(json.dumps(observation, separators=(",", ":")) + "\n")
            raw.flush()
        print(json.dumps({"event": "progress", "queries": len(observations),
                          "last_query_row": observations[-1]["query_row"]}), flush=True)
    raw.close()
    latencies_ms = [row["latency_http_ns"] / 1e6 for row in observations]
    candidates = [row["candidate_count"] for row in observations]
    summary = {
        "scope": "ordinary Qdrant HTTP query path, nprobe fixed by persisted collection config",
        "queries": len(observations), "warmup_queries": args.warmup,
        "query_selection": {"start": args.start, "end": args.end,
                            "stride": args.stride},
        "points_count": info["points_count"],
        "learned_path_markers": len(observations),
        "recall10_tie_aware": statistics.mean(row["recall10_tie_aware"] for row in observations),
        "recall10_conventional": statistics.mean(row["recall10_conventional"] for row in observations),
        "latency_http_ms": {"mean": statistics.mean(latencies_ms),
                            "p50": percentile(latencies_ms, .50),
                            "p95": percentile(latencies_ms, .95),
                            "p99": percentile(latencies_ms, .99)},
        "candidates": {"mean": statistics.mean(candidates),
                       "p50": percentile(candidates, .50),
                       "p95": percentile(candidates, .95),
                       "p99": percentile(candidates, .99)},
        "candidate_fraction_mean": statistics.mean(candidates) / 10_000_000,
        "component_timings": "not exposed by HTTP; not measured here",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
