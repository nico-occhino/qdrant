"""Capture and verify learned routing across a Qdrant process restart."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import numpy as np


MARKER = re.compile(r"candidate_source=StaticLearned candidate_count=(\d+)")
TRAINING_MARKERS = ("LMI sampling:", "LMI build: training", "LMI spherical teacher:")


def get_json(url):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def query(base, vector):
    body = {"query": vector.tolist(), "limit": 10,
            "with_payload": False, "with_vector": False}
    request = Request(base + "/points/query", data=json.dumps(body).encode(),
                      method="POST", headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        result = json.load(response)["result"]["points"]
    if len(result) != 10:
        raise RuntimeError(f"expected 10 query results, received {len(result)}")
    return {"ids": [int(point["id"]) for point in result],
            "scores": [float(point["score"]) for point in result]}


def state_hashes(directory):
    values = {}
    for name in ("lmi_state.json", "lmi_router.bin", "lmi_postings.bin"):
        for path in sorted(directory.rglob(name)):
            relative = str(path.relative_to(directory))
            values[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not any(path.endswith("lmi_router.bin") for path in values):
        raise RuntimeError("no persisted LMI router found")
    if not any(path.endswith("lmi_postings.bin") for path in values):
        raise RuntimeError("no persisted LMI postings found")
    return values


def dense_query(path, key, row):
    with h5py.File(path, "r") as handle:
        vector = np.asarray(handle[key][row], dtype=np.float32)
    if vector.shape != (768,) or not np.isfinite(vector).all():
        raise RuntimeError("invalid SISAP query vector")
    return vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "verify"])
    parser.add_argument("--url", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-key", required=True)
    parser.add_argument("--query-row", type=int, default=0)
    parser.add_argument("--expected-points", type=int, default=300000)
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    collection_url = base + "/collections/" + args.collection
    info = get_json(collection_url)["result"]
    if info.get("points_count") != args.expected_points:
        raise RuntimeError(f"expected {args.expected_points} points, found {info.get('points_count')}")
    vector = dense_query(args.queries, args.query_key, args.query_row)
    if args.action == "capture":
        if args.checkpoint.exists():
            raise FileExistsError(f"refusing to overwrite restart checkpoint: {args.checkpoint}")
        result = query(collection_url, vector)
        record = {"collection": args.collection, "expected_points": args.expected_points,
                  "query_row": args.query_row, "baseline_result": result,
                  "segment_state_sha256": state_hashes(args.segments_dir),
                  "server_log_bytes_after_baseline": args.server_log.stat().st_size,
                  "collection_status": info.get("status"),
                  "datatype": info.get("config", {}).get("params", {}).get("vectors", {})
                      .get("datatype")}
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return
    if args.evidence is None:
        parser.error("verify requires --evidence")
    if args.evidence.exists():
        raise FileExistsError(f"refusing to overwrite restart evidence: {args.evidence}")
    baseline = json.loads(args.checkpoint.read_text())
    if baseline.get("collection") != args.collection or baseline.get("query_row") != args.query_row:
        raise RuntimeError("restart checkpoint belongs to another query/collection")
    current_hashes = state_hashes(args.segments_dir)
    result = query(collection_url, vector)
    log = args.server_log.read_bytes()
    offset = int(baseline["server_log_bytes_after_baseline"])
    if len(log) < offset:
        raise RuntimeError("server log was truncated across restart")
    after_restart = log[offset:].decode(errors="replace")
    markers = [int(match.group(1)) for line in after_restart.splitlines()
               if (match := MARKER.search(line))]
    gates = {"same_official_query_result_ids": result["ids"] == baseline["baseline_result"]["ids"],
             "same_scores_within_float32_roundoff": all(
                 abs(a - b) <= 1e-6 for a, b in
                 zip(result["scores"], baseline["baseline_result"]["scores"])),
             "persisted_router_and_postings_unchanged": current_hashes
                 == baseline["segment_state_sha256"],
             "learned_route_marker_after_restart": len(markers) == 1,
             "no_retraining_or_sampling_after_restart": not any(
                 marker in after_restart for marker in TRAINING_MARKERS),
             "float16_collection_after_restart": info.get("config", {}).get("params", {})
                 .get("vectors", {}).get("datatype") == "float16"}
    record = {"query_row": args.query_row, "result_after_restart": result,
              "post_restart_candidate_count": markers,
              "segment_state_sha256": current_hashes, "gates": gates}
    if not all(gates.values()):
        print(json.dumps(record, indent=2))
        raise RuntimeError("restart validation failed")
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
