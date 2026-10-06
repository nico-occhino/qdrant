"""Guarded SISAP10M state transitions; this script does not auto-start a run."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen


CONFIG = {"n_buckets": 3162, "sample_size": 250000, "hidden_dim": 512,
          "epochs": 30, "batch_size": 256, "routing_batch_size": 256,
          "kmeans_iterations": 5, "nprobe": 4, "seed": 42}
OFFICIAL_BASE_ROWS = 10_120_191
CONSOLIDATION_MAX_SEGMENT_SIZE_KB = 32_000_000


def request(method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, method=method,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def snapshot(base, collection):
    return request("GET", f"{base}/collections/{collection}")["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "adopt-empty", "status", "verify-ingest",
                                             "consolidate", "admit-build"])
    parser.add_argument("--url", default="http://127.0.0.1:17036")
    parser.add_argument("--collection", default="sisap2023_10m_f16_lmi")
    parser.add_argument("--segments-dir", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=21600)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    collection_url = f"{base}/collections/{args.collection}"
    if args.action in {"create", "adopt-empty"}:
        collections = request("GET", base + "/collections")["result"]["collections"]
        exists = any(row["name"] == args.collection for row in collections)
        body = {"vectors": {"size": 768, "distance": "Cosine", "datatype": "float16",
                            "on_disk": True, "lmi_config": CONFIG},
                "hnsw_config": {"m": 0, "payload_m": 0},
                "optimizers_config": {"indexing_threshold": 0,
                                      "default_segment_number": 1,
                                      "max_optimization_threads": 1},
                "shard_number": 1}
        if args.action == "create":
            if exists:
                raise RuntimeError("refusing to reuse an existing collection; inspect it manually")
            result = request("PUT", collection_url, body)
            if result.get("status") != "ok":
                raise RuntimeError(f"collection creation failed: {result}")
        else:
            if not exists:
                raise RuntimeError("cannot adopt: collection does not exist")
            current = snapshot(base, args.collection)
            vector = current.get("config", {}).get("params", {}).get("vectors", {})
            optimizer = current.get("config", {}).get("optimizer_config", {})
            if not (current.get("points_count") == 0
                    and vector.get("datatype") == "float16"
                    and vector.get("distance") == "Cosine"
                    and vector.get("lmi_config") == CONFIG
                    and optimizer.get("indexing_threshold") == 0):
                raise RuntimeError("refusing adoption: collection is not empty/configured as expected")
        args.state.parent.mkdir(parents=True, exist_ok=True)
        state = {"phase": "created_ingest_disabled", "collection": args.collection,
                 "created": datetime.now().astimezone().isoformat(), "config": body,
                 "action": args.action}
        args.state.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({"state": "created_ingest_disabled", "config": body}, indent=2))
        return
    observed = snapshot(base, args.collection)
    if args.action == "status":
        print(json.dumps(observed, indent=2))
        return
    if not args.state.is_file():
        raise RuntimeError("missing run state; create the guarded collection first")
    state = json.loads(args.state.read_text())
    if state.get("collection") != args.collection:
        raise RuntimeError("state file belongs to a different collection")
    if args.action == "verify-ingest":
        if state.get("phase") != "created_ingest_disabled":
            raise RuntimeError("ingest verification is only allowed after guarded creation")
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise RuntimeError("--checkpoint must point to the completed SISAP importer checkpoint")
        checkpoint = json.loads(args.checkpoint.read_text())
        source_verified = checkpoint.get("source_verification", {})
        gates = {
            "official_sisap10m_source_path": checkpoint.get("source")
                == "/mnt/c/datasets/sisap2023/laion2B-en-clip768v2-n=10M.h5",
            "complete_official_source_range": checkpoint.get("start") == 0
                and checkpoint.get("end") == OFFICIAL_BASE_ROWS
                and checkpoint.get("next_source_row") == OFFICIAL_BASE_ROWS,
            "sampled_source_roundtrip": source_verified.get("checked", 0) >= 1
                and source_verified.get("tolerance") == 0.002,
            "collection_exact_official_file_count": observed.get("points_count")
                == OFFICIAL_BASE_ROWS,
            "indexing_still_disabled": observed.get("config", {}).get("optimizer_config", {})
                .get("indexing_threshold") == 0,
        }
        record = {"timestamp": datetime.now().astimezone().isoformat(),
                  "collection": args.collection, "gates": gates,
                  "checkpoint": checkpoint, "observed": observed}
        if not all(gates.values()):
            print(json.dumps(record, indent=2))
            raise SystemExit(2)
        state["phase"] = "ingest_verified"
        state["ingest_verification"] = record
        args.state.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return
    if args.action == "consolidate":
        if state.get("phase") not in {"ingest_verified", "consolidating"}:
            raise RuntimeError("consolidation is allowed only after verified complete ingestion")
        observed = snapshot(base, args.collection)
        optimizer = observed.get("config", {}).get("optimizer_config", {})
        vector = observed.get("config", {}).get("params", {}).get("vectors", {})
        if state.get("phase") == "ingest_verified":
            preconditions = {
                "green": observed.get("status") == "green",
                "exact_official_file_point_count": observed.get("points_count") == OFFICIAL_BASE_ROWS,
                "zero_indexed": observed.get("indexed_vectors_count") == 0,
                "indexing_disabled": optimizer.get("indexing_threshold") == 0,
                "float16_cosine_lmi_config": vector.get("datatype") == "float16"
                    and vector.get("distance") == "Cosine" and vector.get("lmi_config") == CONFIG,
            }
            if not all(preconditions.values()):
                raise RuntimeError("refusing consolidation transition: " + json.dumps(preconditions))
            state["phase"] = "consolidating"
            state["consolidation"] = {
                "started_at": datetime.now().astimezone().isoformat(),
                "preconditions": preconditions,
                "pre_patch_observed": observed,
                "merge_ceiling_kb": CONSOLIDATION_MAX_SEGMENT_SIZE_KB,
            }
            args.state.write_text(json.dumps(state, indent=2) + "\n")
        if (optimizer.get("max_segment_size") is None
                or optimizer.get("max_segment_size", 0) < CONSOLIDATION_MAX_SEGMENT_SIZE_KB):
            patch_body = {"optimizers_config": {
                "max_segment_size": CONSOLIDATION_MAX_SEGMENT_SIZE_KB}}
            patch_response = request("PATCH", collection_url, patch_body)
            state["consolidation"]["merge_ceiling_patch"] = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "body": patch_body,
                "response": patch_response,
            }
            args.state.write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps({"phase": "consolidation_requested",
                              "merge_ceiling_kb": CONSOLIDATION_MAX_SEGMENT_SIZE_KB,
                              "response": patch_response}), flush=True)
        deadline = time.monotonic() + args.wait_seconds
        last = None
        while True:
            observed = snapshot(base, args.collection)
            optimizer = observed.get("config", {}).get("optimizer_config", {})
            gates = {"green": observed.get("status") == "green",
            "exact_official_file_point_count": observed.get("points_count") == OFFICIAL_BASE_ROWS,
                     "zero_indexed": observed.get("indexed_vectors_count") == 0,
                     "one_segment": observed.get("segments_count") == 1,
                     "optimizer_idle": observed.get("optimizer_status") == "ok",
                     "indexing_disabled": optimizer.get("indexing_threshold") == 0,
                     "merge_ceiling_configured": optimizer.get("max_segment_size")
                         == CONSOLIDATION_MAX_SEGMENT_SIZE_KB}
            current = {"phase": "consolidating", "gates": gates,
                       "points_count": observed.get("points_count"),
                       "segments_count": observed.get("segments_count"),
                       "status": observed.get("status"),
                       "optimizer_status": observed.get("optimizer_status")}
            if current != last:
                print(json.dumps(current), flush=True)
                last = current
            if all(gates.values()):
                state["phase"] = "consolidated"
                state["consolidation"]["completed_at"] = datetime.now().astimezone().isoformat()
                state["consolidation"]["observed"] = observed
                args.state.write_text(json.dumps(state, indent=2) + "\n")
                print(json.dumps({"phase": "consolidated", "observed": observed}), flush=True)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("collection did not reach the exact one-segment build gate")
            time.sleep(10)
    if args.evidence.exists():
        raise RuntimeError(f"refusing duplicate build-admission evidence: {args.evidence}")
    if state.get("phase") != "consolidated":
        raise RuntimeError("build admission requires the guarded consolidated state")
    if args.segments_dir is None:
        parser.error("admit-build requires --segments-dir")
    vector = observed.get("config", {}).get("params", {}).get("vectors", {})
    optimizer = observed.get("config", {}).get("optimizer_config", {})
    postings = sorted(str(path) for path in args.segments_dir.rglob("lmi_postings.bin"))
    gates = {
        "green": observed.get("status") == "green",
        "exact_official_file_point_count": observed.get("points_count") == OFFICIAL_BASE_ROWS,
        "zero_indexed": observed.get("indexed_vectors_count") == 0,
        "one_segment": observed.get("segments_count") == 1,
        "optimizer_idle": observed.get("optimizer_status") == "ok",
        "indexing_disabled_before_gate": optimizer.get("indexing_threshold") == 0,
        "single_optimizer_thread": optimizer.get("max_optimization_threads") == 1,
        "float16_cosine_lmi_config": vector.get("datatype") == "float16"
            and vector.get("distance") == "Cosine" and vector.get("lmi_config") == CONFIG,
        "no_prior_postings": not postings,
    }
    record = {"timestamp": datetime.now().astimezone().isoformat(),
              "collection": args.collection, "observed": observed,
              "existing_postings": postings, "gates": gates,
              "patch_attempted": all(gates.values())}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    # Record the decision even on failure; never overwrite a prior decision.
    fd = os.open(args.evidence, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not all(gates.values()):
        print(json.dumps(record, indent=2))
        raise SystemExit(2)
    # This is the only automated LMI-enabling PATCH in this state machine.
    record["patch_response"] = request(
        "PATCH", collection_url,
        {"optimizers_config": {"indexing_threshold": 1,
                               "max_optimization_threads": 1}})
    record["patch_timestamp"] = datetime.now().astimezone().isoformat()
    args.evidence.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    state["phase"] = "build_enabled_once"
    state["build_admission_evidence"] = str(args.evidence.resolve())
    args.state.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
