#!/usr/bin/env python3
"""At the fixed ingestion point, disable optimizer threads without enabling indexing."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen


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
    parser.add_argument("--url", required=True)
    parser.add_argument("--collection", default="laion10m_lmi")
    parser.add_argument("--minimum-points", type=int, default=5_278_528)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.evidence.exists():
        parser.error(f"refusing to overwrite evidence: {args.evidence}")

    endpoint = args.url.rstrip("/") + "/collections/" + args.collection
    while True:
        response = request_json("GET", endpoint)
        state = response["result"]
        config = state["config"]["optimizer_config"]
        points = state["points_count"]
        if points >= args.minimum_points:
            evidence = {
                "observed_at": datetime.now().astimezone().isoformat(),
                "observed": state,
                "patch_attempted": False,
            }
            if (
                points > 10_000_000
                or state.get("indexed_vectors_count") != 0
                or config.get("indexing_threshold") != 0
                or state.get("optimizer_status") == "error"
            ):
                evidence["refused"] = "ingestion consolidation pause preconditions failed"
                args.evidence.parent.mkdir(parents=True, exist_ok=True)
                with args.evidence.open("x", encoding="utf-8") as stream:
                    json.dump(evidence, stream, indent=2)
                    stream.write("\n")
                print(json.dumps(evidence, indent=2))
                return 2

            evidence["patch_attempted"] = True
            evidence["patch_at"] = datetime.now().astimezone().isoformat()
            args.evidence.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(args.evidence, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(evidence, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            patch = {"optimizers_config": {
                "indexing_threshold": 0,
                "max_segment_size": 67_108_864,
                "max_optimization_threads": 0,
            }}
            evidence["patch_response"] = request_json("PATCH", endpoint, patch)
            evidence["patch"] = patch
            args.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(evidence, indent=2))
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
