"""Create-new SHA-256 inventories and provenance for WAL-restored Plain data."""
import argparse, hashlib, json, os
from pathlib import Path

N = 10_120_191
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def inventory(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"unsafe collection root: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"unsafe collection entry: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": digest(path)}
    if not files:
        raise ValueError("empty collection")
    return files

def write_new(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())

def report(path, expected):
    value = json.loads(path.read_text())
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"{path}: {key} is not {wanted!r}")
    return value

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["inventory", "verify", "finalize"])
    for key in ("scratch", "staging", "final", "report", "wal_report", "original"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--job-id", required=True)
    a = p.parse_args()
    if a.staging.parent.name != ".staging" or a.final.parent != a.staging.parent.parent or a.staging.name != a.final.name:
        raise ValueError("unsafe staging/final generation layout")
    staged = a.staging / "collection"
    wal = report(a.wal_report, {"accepted": True, "completion_marker_created": False})
    if wal["persisted_after_restart"]["total_vectors"] != N:
        raise ValueError("wrong recovered scratch vector count")
    if a.action == "inventory":
        if a.final.exists():
            raise FileExistsError(a.final)
        copied = inventory(a.scratch)
        if inventory(staged) != copied:
            raise ValueError("scratch/staging inventory or SHA-256 differs")
        write_new(a.staging / "copied-inventory.json",
                  {"format": "sisap23-copied-inventory-v1", "files": copied})
        return
    copied = json.loads((a.staging / "copied-inventory.json").read_text())["files"]
    if inventory(a.scratch) != copied:
        raise ValueError("validated scratch generation changed")
    if a.action == "verify":
        if inventory(staged) != copied:
            raise ValueError("staging changed before Qdrant reopen")
        return
    check = report(a.report, {"accepted": True, "recovered_vectors_compared": 1679,
                              "index_files_unchanged": True})
    for name in ("first_reopen", "second_reopen"):
        if check[name]["complete_and_valid"] is not True or check[name]["total_vectors"] != N:
            raise ValueError(f"{name} failed")
    after = inventory(staged)
    changed = sorted(k for k in set(copied) | set(after) if copied.get(k) != after.get(k))
    source = inventory(a.original)
    if a.final.exists():
        raise FileExistsError(a.final)
    provenance = {
        "format": "sisap23-recovered-plain-generation-v1", "accepted": True,
        "point_count": N, "index_kind": "plain",
        "original_source_commit": "42a163701a51078ec24983b3a93f96ab7d661489",
        "wal_replay_commit": "f654cfc96fa7b1819f5a5bfcfa0f5e965cb0ca13",
        "preservation_commit": a.commit, "pbs_job_id": a.job_id,
        "hostname": os.uname().nodename, "original_collection": str(a.original),
        "original_collection_inventory_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        "scratch_replay_report": str(a.wal_report), "scratch_replay_report_sha256": digest(a.wal_report),
        "persistent_verification_report": str(a.report), "persistent_verification_report_sha256": digest(a.report),
        "copy_inventory_sha256": digest(a.staging / "copied-inventory.json"),
        "post_reopen_changes": changed, "collection_files": after,
        "data_correctness": "verified after two normal Qdrant collection reopens",
        "construction_time_comparable": False}
    write_new(a.staging / "generation.json", provenance)
    os.rename(a.staging, a.final)
    if inventory(a.final / "collection") != after:
        raise ValueError("published collection differs from accepted staging inventory")
    print(a.final)

if __name__ == "__main__":
    main()
