"""Enable exactly one 300K LMI build after the verified consolidation gate."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen


CONFIG = {"n_buckets": 548, "sample_size": 32768, "hidden_dim": 512,
          "epochs": 30, "batch_size": 256, "routing_batch_size": 256,
          "kmeans_iterations": 5, "nprobe": 4, "seed": 42}


def request(method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, method=method,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=45) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:17036")
    parser.add_argument("--collection", default="sisap2023_300k_f16_lmi")
    parser.add_argument("--segments-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.evidence.exists():
        parser.error(f"refusing to overwrite admission evidence: {args.evidence}")
    base = args.url.rstrip("/")
    observed = request("GET", f"{base}/collections/{args.collection}")["result"]
    vector = observed.get("config", {}).get("params", {}).get("vectors", {})
    optimizer = observed.get("config", {}).get("optimizer_config", {})
    checkpoint = json.loads(args.checkpoint.read_text())
    verification = checkpoint.get("source_verification", {})
    existing = sorted(str(p) for p in args.segments_dir.rglob("lmi_postings.bin"))
    gates = {"green": observed.get("status") == "green",
             "exact_300k_points": observed.get("points_count") == 300000,
             "zero_indexed_vectors": observed.get("indexed_vectors_count") == 0,
             "one_segment": observed.get("segments_count") == 1,
             "optimizer_idle": observed.get("optimizer_status") == "ok",
             "indexing_disabled": optimizer.get("indexing_threshold") == 0,
             "completed_source_and_roundtrip_verification": checkpoint.get("start") == 0
                 and checkpoint.get("end") == 300000
                 and checkpoint.get("next_source_row") == 300000
                 and verification.get("checked", 0) >= 1
                 and verification.get("tolerance") == 0.002,
             "float16_cosine_and_scaled_lmi_config": vector.get("datatype") == "float16"
                 and vector.get("distance") == "Cosine" and vector.get("lmi_config") == CONFIG,
             "no_existing_lmi_postings": not existing}
    record = {"timestamp": datetime.now().astimezone().isoformat(),
              "observed": observed, "gates": gates, "existing_postings": existing,
              "source_checkpoint": checkpoint,
              "patch_attempted": all(gates.values())}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.evidence, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not all(gates.values()):
        print(json.dumps(record, indent=2))
        raise SystemExit(2)
    record["patch_response"] = request("PATCH", f"{base}/collections/{args.collection}",
            {"optimizers_config": {"indexing_threshold": 1,
                                   "max_optimization_threads": 1}})
    record["patch_timestamp"] = datetime.now().astimezone().isoformat()
    args.evidence.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
