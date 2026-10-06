#!/usr/bin/env python3
"""Enable the single S.3E LMI build only after the one-segment admission gate."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPECTED_LMI = {
    "n_buckets": 3162,
    "sample_size": 250000,
    "hidden_dim": 512,
    "epochs": 30,
    "batch_size": 256,
    "routing_batch_size": 256,
    "kmeans_iterations": 5,
    "nprobe": 4,
    "seed": 42,
}


def request_json(method: str, url: str, body=None):
    payload = None if body is None else json.dumps(body).encode()
    request = Request(
        url,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Qdrant base URL")
    parser.add_argument("--collection", default="laion10m_lmi")
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.evidence.exists():
        parser.error(f"refusing to overwrite evidence: {args.evidence}")

    base = args.url.rstrip("/")
    observed = request_json("GET", f"{base}/collections/{args.collection}")["result"]
    config = observed.get("config", {})
    optimizer = config.get("optimizer_config", {})
    vector_config = config.get("params", {}).get("vectors", {})
    existing_postings = sorted(str(path) for path in args.segments_dir.rglob("lmi_postings.bin"))
    gates = {
        "green": observed.get("status") == "green",
        "exact_points": observed.get("points_count") == 10_000_000,
        "no_indexed_vectors": observed.get("indexed_vectors_count") == 0,
        "one_segment": observed.get("segments_count") == 1,
        "optimizer_idle_ok": observed.get("optimizer_status") == "ok",
        "indexing_disabled": optimizer.get("indexing_threshold") == 0,
        "single_optimizer_thread": optimizer.get("max_optimization_threads") == 1,
        "expected_lmi_config": vector_config.get("lmi_config") == EXPECTED_LMI,
        "no_published_lmi": not existing_postings,
    }
    evidence = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "collection": args.collection,
        "observed": observed,
        "existing_lmi_postings": existing_postings,
        "gates": gates,
        "patch_attempted": False,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    if not all(gates.values()):
        with args.evidence.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2)
            stream.write("\n")
        print(json.dumps(evidence, indent=2))
        return 2

    # Reserve an exclusive durable intent record before the one permitted PATCH.
    evidence["patch_attempted"] = True
    evidence["patch_timestamp"] = datetime.now().astimezone().isoformat()
    descriptor = os.open(args.evidence, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    response = request_json(
        "PATCH",
        f"{base}/collections/{args.collection}",
        {"optimizers_config": {"indexing_threshold": 1, "max_optimization_threads": 1}},
    )
    evidence["patch_response"] = response
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"LMI build admission guard failed: {error}") from error
