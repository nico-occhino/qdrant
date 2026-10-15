"""Create a distinct matched run rooted in a verified recovered Plain generation."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from preserve_generation import inventory

ORIGINAL_COMMIT = "42a163701a51078ec24983b3a93f96ab7d661489"
N = 10_120_191

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--generation", type=Path, required=True)
    p.add_argument("--new-root", type=Path, required=True)
    p.add_argument("--commit", required=True)
    a = p.parse_args()
    if not (a.original.is_absolute() and a.generation.is_absolute() and a.new_root.is_absolute()):
        raise ValueError("all roots must be absolute")
    if a.new_root == a.original or a.new_root.is_relative_to(a.original):
        raise ValueError("new root overlaps frozen original")
    original_bytes = (a.original / "manifest.json").read_bytes()
    original = json.loads(original_bytes)
    if original["source_commit"] != ORIGINAL_COMMIT:
        raise ValueError("wrong frozen input manifest")
    generation_path = a.generation / "generation.json"
    generation_bytes = generation_path.read_bytes()
    generation = json.loads(generation_bytes)
    if generation.get("accepted") is not True or generation.get("point_count") != N:
        raise ValueError("recovered Plain generation is not accepted")
    if generation.get("preservation_commit") != a.commit:
        raise ValueError("reviewed commit differs from preservation implementation")
    if generation.get("original_source_commit") != ORIGINAL_COMMIT:
        raise ValueError("wrong original data provenance")
    if inventory(a.generation / "collection") != generation["collection_files"]:
        raise ValueError("recovered collection inventory changed")
    segments = list((a.generation / "collection").rglob("segment.json"))
    if len(segments) != 1:
        raise ValueError("recovered Plain must contain one segment")
    expected = dict(original)
    expected["source_commit"] = a.commit
    expected["recovered_plain"] = {
        "generation_path": str(a.generation),
        "generation_json_sha256": hashlib.sha256(generation_bytes).hexdigest(),
        "original_manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "data_origin_commit": ORIGINAL_COMMIT}
    if a.new_root.exists():
        saved = json.loads((a.new_root / "manifest.json").read_text())
        if saved != expected:
            raise ValueError("existing matched manifest differs")
    else:
        a.new_root.mkdir()
        save_new(a.new_root / "original-manifest.json", original)
        for key, item in original["prepared"].items():
            source = a.original / item["path"]
            target = a.new_root / item["path"]
            if source.is_symlink() or digest(source) != item["sha256"] or source.stat().st_size != item["bytes"]:
                raise ValueError(f"bad prepared input {key}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != item["sha256"]:
                raise ValueError(f"prepared input copy changed: {key}")
        save_new(a.new_root / "manifest.json", expected)
        (a.new_root / "runs").mkdir()
        plain_run = a.new_root / "runs" / "plain"
        plain_run.mkdir()
        shutil.copytree(a.generation / "collection", plain_run / "collection")
        if inventory(plain_run / "collection") != generation["collection_files"]:
            raise ValueError("matched Plain copy differs from accepted generation")
        imported_segments = list((plain_run / "collection").rglob("segment.json"))
        if len(imported_segments) != 1:
            raise ValueError("copied Plain must contain one segment")
        save_new(a.new_root / "runs/plain/imported.generation.json", {
            "format": "sisap23-imported-plain-v1", "variant": "plain",
            "source_commit": ORIGINAL_COMMIT, "verification_commit": a.commit,
            "point_count": N, "segment_count": 1, "index_kind": "plain",
            "segment_path": str(imported_segments[0].parent),
            "generation_path": str(a.generation),
            "generation_json_sha256": hashlib.sha256(generation_bytes).hexdigest(),
            "construction_time": None,
            "construction_time_comparable": False})
    for key, item in original["prepared"].items():
        target = a.new_root / item["path"]
        if digest(target) != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise ValueError(f"new run prepared input changed: {key}")
    imported = json.loads((a.new_root / "runs/plain/imported.generation.json").read_text())
    if imported["generation_json_sha256"] != hashlib.sha256(generation_bytes).hexdigest():
        raise ValueError("import marker changed")
    if inventory(a.new_root / "runs/plain/collection") != generation["collection_files"]:
        raise ValueError("matched Plain collection changed")
    copied_segments = list((a.new_root / "runs/plain/collection").rglob("segment.json"))
    if len(copied_segments) != 1 or str(copied_segments[0].parent) != imported["segment_path"]:
        raise ValueError("imported segment path differs from verified copy")

if __name__ == "__main__":
    main()
