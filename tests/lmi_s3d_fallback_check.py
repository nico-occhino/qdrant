"""Verify exact and filtered Qdrant requests avoid learned candidate routing."""

import argparse
import json
import struct
import time
from pathlib import Path
from urllib.request import Request, urlopen


def query_vector(path):
    with path.open("rb") as source:
        if source.read(8) != b"LMIQ10M1":
            raise ValueError("unexpected query fixture")
        count, dim = struct.unpack("<II", source.read(8))
        if (count, dim) != (4992, 768):
            raise ValueError("unexpected query fixture shape")
        source.read(8)  # query ID
        return struct.unpack("<768f", source.read(768 * 4))


def send(url, body):
    request = Request(url, data=json.dumps(body).encode(), method="POST",
                      headers={"Content-Type": "application/json"})
    start = time.perf_counter_ns()
    with urlopen(request, timeout=300) as response:
        reply = json.load(response)
    return reply, time.perf_counter_ns() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-input", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:17033")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    url = args.url.rstrip("/") + "/collections/laion10m_lmi/points/query"
    vector = query_vector(args.query_input)
    base = {"query": vector, "limit": 10, "with_payload": False,
            "with_vector": False}
    cases = {
        "exact": {**base, "params": {"exact": True}},
        "filtered": {**base, "filter": {"must": [{"has_id": [0]}]}},
    }
    results = {}
    for name, body in cases.items():
        offset = args.server_log.stat().st_size
        # A Plain exact scan over 10M on-disk vectors exceeds the default
        # 60-second search deadline on this host; ReadParams accepts seconds.
        request_url = url + ("?timeout=300" if name == "exact" else "")
        reply, duration = send(request_url, body)
        if reply.get("status") != "ok":
            raise RuntimeError(f"{name} query failed: {reply}")
        rows = reply["result"]["points"]
        with args.server_log.open("rb") as source:
            source.seek(offset)
            new_log = source.read().decode(errors="replace")
        if "candidate_source=StaticLearned" in new_log:
            raise AssertionError(f"{name} request used learned candidates")
        ids = [int(row["id"]) for row in rows]
        if name == "exact" and len(ids) != 10:
            raise AssertionError("exact request did not return ten points")
        if name == "filtered" and ids != [0]:
            raise AssertionError(f"filtered request expected only point 0, got {ids}")
        results[name] = {"result_ids": ids, "duration_ns": duration,
                         "learned_marker_present": False,
                         "lmi_search_log_present": "[LMI-DUMMY] search" in new_log}
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
