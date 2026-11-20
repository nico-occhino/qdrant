"""Validate a bounded exact-Plain panel measured from an isolated PBS scratch copy."""
import hashlib
import json
import math
from pathlib import Path
import statistics

POINTS = 10_120_191
COMMIT = "9b12d583b1bf9df802b2cc420244bd3c24732d86"
PRODUCER = "69216d5ada898feef7066c72da0514aeae2ead89"
INTERFACE = "direct_segment_indices_sequential_merge"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_records(warmup, repetitions, expected_rows, gold_by_row, plain_by_row,
                     segment_count, points=POINTS):
    if len(warmup) != 20:
        raise ValueError("Plain official warmup must contain all 20 rows")
    for order, row in enumerate(warmup):
        if (row.get("warmup_order") != order or row.get("query_row") != order
                or row.get("parameter") != "plain-exact"):
            raise ValueError("Plain warmup order or setting differs")
        validate_record(row, segment_count, points, require_recall=False)

    if len(repetitions) != 3:
        raise ValueError("Plain panel must contain exactly three repetitions")
    all_rows = []
    for repetition, records in enumerate(repetitions, 1):
        if [row.get("query_row") for row in records] != list(expected_rows):
            raise ValueError(f"Plain panel coverage/order differs in repetition {repetition}")
        for row in records:
            if row.get("repetition") != repetition or row.get("parameter") != "plain-exact":
                raise ValueError("Plain panel repetition or setting differs")
            validate_record(row, segment_count, points, require_recall=True)
            query_row = row["query_row"]
            ids = list(map(int, row["returned_ids_1_based"]))
            gold = list(map(int, gold_by_row[query_row]))
            reference = list(map(int, plain_by_row[query_row]))
            recall = len(set(ids) & set(gold)) / 10
            if abs(float(row["recall_at_10"]) - recall) > 1e-12 or recall != 1.0:
                raise ValueError(f"exact Plain result disagrees with ground truth at query row {query_row}")
            if ids != gold:
                raise ValueError(f"scratch Plain ranking differs from ground truth at row {query_row}")
            if ids != reference:
                raise ValueError(f"scratch Plain ranking differs from preserved Plain result at row {query_row}")
        all_rows.extend(records)

    times = [float(row["native_search_ms"]) for row in all_rows]
    warmup_times = [float(row["native_search_ms"]) for row in warmup]
    return {
        "query_count_per_repetition": len(expected_rows),
        "repetitions": 3,
        "measured_query_count": len(all_rows),
        "mean_ms": statistics.fmean(times),
        "p50_ms": _quantile(times, 0.50),
        "p95_ms": _quantile(times, 0.95),
        "p99_ms": _quantile(times, 0.99),
        "warmup_mean_ms": statistics.fmean(warmup_times),
        "warmup_max_ms": max(warmup_times),
        "mean_recall_at_10": 1.0,
    }


def validate_record(row, segment_count, points, require_recall):
    elapsed = row.get("native_search_ms")
    ids = list(map(int, row.get("returned_ids_1_based", [])))
    if (not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0
            or row.get("candidate_count") != points or row.get("segment_count") != segment_count
            or row.get("search_interface") != INTERFACE
            or row.get("fallback_observation") != "not_applicable_direct_segment_index"
            or len(ids) != 10 or len(set(ids)) != 10
            or any(not 1 <= point_id <= points for point_id in ids)):
        raise ValueError("Plain query record is incomplete or invalid")
    if require_recall and row.get("recall_at_10") != 1.0:
        raise ValueError("exact Plain record does not report full exact recall")


def _quantile(values, quantile):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def validate_files(campaign, result_root, source_generation, expected_generation_sha,
                   pbs_job_id, inventory_before_sha, inventory_after_sha):
    import numpy as np

    campaign = Path(campaign)
    result_root = Path(result_root)
    source_generation = Path(source_generation)
    marker_path = result_root / "plain-ssd-panel.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    generation = json.loads((source_generation / "generation.json").read_text(encoding="utf-8"))
    manifest = json.loads((campaign / "manifest.json").read_text(encoding="utf-8"))
    if (marker.get("format") != "sisap23-plain-ssd-panel-v1"
            or marker.get("test_executed") is not True
            or marker.get("source_commit") != COMMIT
            or marker.get("source_generation") != str(source_generation)
            or marker.get("source_generation_sha256") != expected_generation_sha
            or marker.get("point_count") != POINTS or marker.get("repetitions") != 3
            or marker.get("warmup_rows") != 20
            or marker.get("evaluation_storage") != "isolated_pbs_scratch_local_ssd"
            or generation.get("accepted") is not True or generation.get("kind") != "data"
            or generation.get("producer_commit") != PRODUCER or generation.get("points") != POINTS
            or manifest.get("source_commit") != COMMIT
            or digest(source_generation / "generation.json") != expected_generation_sha):
        raise ValueError("Plain scratch panel provenance differs")
    segment_count = marker.get("segment_count")
    if not isinstance(segment_count, int) or segment_count < 1:
        raise ValueError("Plain scratch panel has no segment topology")
    if inventory_before_sha != inventory_after_sha:
        raise ValueError("Plain scratch collection inventory changed during the query run")

    inputs = {Path(item["path"]).name: campaign / item["path"]
              for item in manifest["prepared"].values()}
    for item in manifest["prepared"].values():
        path = campaign / item["path"]
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"prepared input hash mismatch: {path.name}")
    panel = np.fromfile(inputs["panel_rows.u32le"], dtype="<u4").astype(int).tolist()
    warmup = read_jsonl(result_root / "warmup-rows.jsonl")
    repetitions = [read_jsonl(result_root / f"panel-rep-{i}.jsonl") for i in range(1, 4)]
    gold = np.fromfile(inputs["gold_20_9999_top10.i32le"], dtype="<i4").reshape(9980, 10)
    plain = read_jsonl(campaign / "runs/plain/full/full/rows.jsonl")
    if len(plain) != 9980 or [row["query_row"] for row in plain] != list(range(20, 10000)):
        raise ValueError("preserved Plain exact results are incomplete")
    gold_by_row = {row: gold[row - 20].tolist() for row in panel}
    plain_by_row = {row["query_row"]: row["returned_ids_1_based"]
                    for row in plain if row["query_row"] in set(panel)}
    if set(plain_by_row) != set(panel):
        raise ValueError("preserved Plain result rows do not cover the paired panel")
    summary = validate_records(warmup, repetitions, panel, gold_by_row, plain_by_row, segment_count)
    report = {
        "format": "sisap23-plain-ssd-panel-validation-v1",
        "validation": "passed",
        "pbs_job_id": pbs_job_id,
        "source_commit": COMMIT,
        "source_generation": str(source_generation),
        "source_generation_sha256": expected_generation_sha,
        "query_manifest_sha256": digest(campaign / "manifest.json"),
        "panel_rows_sha256": digest(inputs["panel_rows.u32le"]),
        "ground_truth_sha256": digest(inputs["gold_20_9999_top10.i32le"]),
        "inventory_before_sha256": inventory_before_sha,
        "inventory_after_sha256": inventory_after_sha,
        "evaluation_storage": "isolated_pbs_scratch_local_ssd",
        "cache_advice": marker.get("cache_advice"),
        "segment_count": segment_count,
        "panel_rows": panel,
        "summary": summary,
    }
    out = result_root / "validation.json"
    if out.exists():
        if json.loads(out.read_text(encoding="utf-8")) != report:
            raise ValueError("existing Plain scratch validation report differs")
    else:
        with out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return report
