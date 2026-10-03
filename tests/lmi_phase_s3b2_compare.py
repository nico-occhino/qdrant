#!/usr/bin/env python3
"""Compare two completed S.3 build outputs without modifying either one."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def assignments(path: Path) -> list[int]:
    raw = path.read_bytes()
    (boundary_count,) = struct.unpack_from("<Q", raw)
    boundaries = struct.unpack_from(f"<{boundary_count}Q", raw, 8)
    cursor = 8 + 8 * boundary_count
    (point_count,) = struct.unpack_from("<Q", raw, cursor)
    points = struct.unpack_from(f"<{point_count}I", raw, cursor + 8)
    assert cursor + 8 + 4 * point_count == len(raw)
    assert boundaries[0] == 0 and boundaries[-1] == point_count
    result = [-1] * point_count
    for bucket in range(boundary_count - 1):
        for offset in points[boundaries[bucket] : boundaries[bucket + 1]]:
            assert 0 <= offset < point_count and result[offset] == -1
            result[offset] = bucket
    assert all(bucket >= 0 for bucket in result)
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path)
    parser.add_argument("tch", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), "refusing to overwrite a prior comparison"
    native = json.loads((args.native / "result.json").read_text())
    tch = json.loads((args.tch / "result.json").read_text())
    a = assignments(args.native / "lmi_postings.bin")
    b = assignments(args.tch / "lmi_postings.bin")
    assert len(a) == len(b)
    differing = [
        {"offset": i, "native_bucket": x, "tch_bucket": y}
        for i, (x, y) in enumerate(zip(a, b))
        if x != y
    ]
    result = {
        "count": len(a),
        "assignment_mismatches": len(differing),
        "mismatch_rate": len(differing) / len(a),
        "mismatch_examples": differing[:32],
        "query_results_equal": native["query_results"] == tch["query_results"],
        "build_seconds": {
            "native": native["total_build_seconds"],
            "tch": tch["total_build_seconds"],
        },
        "reopen_seconds": {
            "native": native["reopen_seconds"],
            "tch": tch["reopen_seconds"],
        },
        "file_sha256": {
            name: {"native": digest(args.native / name), "tch": digest(args.tch / name)}
            for name in ["lmi_state.json", "lmi_router.bin", "lmi_postings.bin"]
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
