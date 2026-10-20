"""Review retained staging and publish a no-overwrite descriptor. No Qdrant startup."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from preserve_generation import digest, inventory, write_new

FORMAT = "sisap23-publication-v1"

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(path.read_text())

def canonical(path):
    path = Path(os.path.abspath(path))
    require(path == path.resolve(), f"symlink or noncanonical path refused: {path}")
    return path

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def event(phase, **fields):
    print(json.dumps({"phase":phase, "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      **fields}, sort_keys=True), flush=True)

def audit_report(data, expected):
    require(data.get("accepted") is True and data.get("points") == expected, "incomplete vector audit")
    require((data.get("dimension"), data.get("datatype"), data.get("distance")) ==
            (768, "float16", "cosine"), "vector configuration differs")
    for key in ("missing_ids", "duplicate_ids", "invalid_mappings", "absent_vectors"):
        require(data.get(key) == 0, f"invalid vector audit: {key}")
    fingerprint = data.get("vector_fingerprint_xor_sha256", "")
    require(len(fingerprint) == 64 and all(c in "0123456789abcdef" for c in fingerprint),
            "invalid vector fingerprint")
    return fingerprint

def review_staging(staging, producer, job):
    """Metadata only: no collection scan, fsync, mkdir, or mutation."""
    from recovery_stages import N, BASE, check_generation_metadata
    staging = canonical(staging)
    require(staging.parent.name == ".staging" and staging.is_dir(), "expected retained staging directory")
    names = ("generation.json", "scratch.json", "persistent.json", "scratch.exit.json",
             "persistent.exit.json", "source-inventory.json", "copied-inventory.json")
    hashes = {}
    values = {}
    for name in names:
        path = staging / name
        require(path.is_file() and not path.is_symlink(), f"missing/unsafe report: {name}")
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        values[name] = json.loads(raw)
    meta = values["generation.json"]
    check_generation_metadata(meta, producer)
    require(meta["job_id"] == job, "producer job mismatch")
    require(meta.get("data_origin_commit") == BASE and
            len(meta.get("original_manifest_sha256", "")) == 64 and
            Path(meta.get("original_root", "")).is_absolute() and
            Path(meta.get("source_path", "")).is_absolute(), "missing original-data provenance")
    if meta["kind"] == "data":
        require(Path(meta["source_path"]).is_relative_to(Path(meta["original_root"])),
                "recovered data source outside original root")
    require(values["scratch.exit.json"].get("exit_code") == 0 and
            values["persistent.exit.json"].get("exit_code") == 0, "native stage failed")
    scratch, persistent = values["scratch.json"], values["persistent.json"]
    require(meta["report"] == scratch and meta["persistent_report"] == persistent,
            "embedded reports differ from saved reports")
    require(persistent.get("format") == "sisap23-persistent-verified-v2" and
            persistent.get("index_files_unchanged") is True and
            persistent.get("ingested_points") == 0, "persistent proof incomplete")
    fingerprint = audit_report(persistent["data"], N)
    if meta["kind"] == "data":
        require(scratch.get("format") == "sisap23-recovered-data-v2", "wrong scratch report")
        require(audit_report(scratch["first"], N) == audit_report(scratch["second"], N) == fingerprint,
                "restart fingerprints differ")
        require(scratch["first"].get("corpus_vectors_compared") == N and
                0 <= scratch["first"].get("corpus_max_abs_error", 1) <= 0.002,
                "corpus verification incomplete")
        require(scratch["first"].get("physical_index_kinds") and
                all(k == "plain" for k in scratch["first"]["physical_index_kinds"]),
                "unexpected indexed data source")
    else:
        require(scratch.get("format") == "sisap23-index-only-v2" and
                audit_report(scratch["data"], N) == fingerprint and
                scratch.get("index", {}).get("complete") is True, "index proof incomplete")
    require(meta["source_inventory_sha256"] == hashes["source-inventory.json"],
            "source inventory provenance differs")
    files = meta["collection_files"]
    require(files and values["source-inventory.json"] and values["copied-inventory.json"],
            "empty file inventory")
    for entries in (files, values["source-inventory.json"], values["copied-inventory.json"]):
        for name, entry in entries.items():
            path = Path(name)
            require(name and not path.is_absolute() and ".." not in path.parts and
                    isinstance(entry["bytes"], int) and entry["bytes"] >= 0 and
                    len(entry["sha256"]) == 64 and
                    all(c in "0123456789abcdef" for c in entry["sha256"]), "invalid inventory entry")
    review = {"staging":str(staging), "producer_commit":producer, "job_id":job,
              "report_hashes":hashes}
    token = hashlib.sha256(json.dumps(review, sort_keys=True).encode()).hexdigest()
    return meta, {"review_token":token, **review, "kind":meta["kind"],
        "variant":meta["variant"], "points":N, "vector_fingerprint":fingerprint,
        "collection_files":len(files), "collection_bytes":sum(v["bytes"] for v in files.values()),
        "metadata_review":"passed", "current_collection_hashes":"NOT CHECKED",
        "publication":"NOT PERFORMED",
        "prior_final_exists":os.path.lexists(staging.parent.parent / staging.name),
        "original_root":meta.get("original_root"), "source_path":meta.get("source_path"),
        "original_manifest_sha256":meta.get("original_manifest_sha256"),
        "data_origin_commit":meta.get("data_origin_commit"),
        "producer_hardware_fingerprint":meta.get("hardware_fingerprint"),
        "scratch_report":scratch, "persistent_report":persistent}

def storage_path(generation):
    """Resolve only complete descriptors; raw staging is never an accepted handle."""
    generation = canonical(generation)
    require(".staging" not in generation.parts, "raw staging is not a published generation")
    marker = generation / "publication.json"
    if not marker.exists():
        require(not marker.is_symlink() and (generation / "collection").is_dir(),
                "incomplete publication: no publication.json or legacy collection")
        return generation
    require(not marker.is_symlink(), "symlink publication marker")
    publication = read(marker)
    require(publication.get("format") == FORMAT and publication.get("accepted") is True and
            publication.get("collection_integrity") == "sha256-verified" and
            publication.get("staging_retained") is True and
            len(publication.get("review_token", "")) == 64 and
            len(publication.get("publisher_commit", "")) == 40,
            "invalid publication marker")
    relative = Path(publication["storage_relative"])
    require(len(relative.parts) == 2 and relative.parts[0] == ".staging" and
            relative.parts[1] not in (".", ".."), "invalid retained storage path")
    storage = canonical(generation.parent / relative)
    require(storage.is_dir(), "retained storage is missing")
    require(not (storage / "generation.json").is_symlink() and
            not (generation / "generation.json").is_symlink(), "symlink metadata refused")
    require(digest(storage / "generation.json") == digest(generation / "generation.json") ==
            publication["generation_sha256"], "generation metadata changed")
    require(read(storage / "generation.json")["producer_commit"] == publication["producer_commit"],
            "publication producer differs")
    return storage

def install_descriptor(staging, final, review, publisher_commit):
    """Reserve a fresh namespace; publish the complete record using link(), never rename()."""
    staging, final = canonical(staging), canonical(final)
    require(staging.parent.name == ".staging" and staging.parent.parent == final.parent and
            final.name not in (".staging", "", ".", ".."), "invalid publication layout")
    final.mkdir()  # Atomic reservation, EEXIST for files, dirs and dangling symlinks.
    # Never remove a partial final directory: it is evidence of an interrupted attempt.
    metadata = (staging / "generation.json").read_bytes()
    require(hashlib.sha256(metadata).hexdigest() == review["report_hashes"]["generation.json"],
            "reviewed metadata changed")
    with (final / "generation.json").open("xb") as out:
        out.write(metadata)
        out.flush()
        os.fsync(out.fileno())
    publication = {"format":FORMAT, "accepted":True,
        "storage_relative":str(staging.relative_to(final.parent)),
        "generation_sha256":review["report_hashes"]["generation.json"],
        "producer_commit":review["producer_commit"], "publisher_commit":publisher_commit,
        "producer_job_id":review["job_id"], "review_token":review["review_token"],
        "collection_integrity":"sha256-verified", "staging_retained":True}
    temp = final / ".publication.prepared.json"
    write_new(temp, publication)
    sync_dir(final)
    sync_dir(final.parent)
    try:
        os.link(temp, final / "publication.json", follow_symlinks=False)
    except OSError as error:
        raise OSError(error.errno, "publication link failed; outcome may be uncertain on NFS; "
                      "inspect the destination, never overwrite/retry in place", str(final)) from error
    try:
        sync_dir(final)
        sync_dir(final.parent)
    except OSError as error:
        raise OSError(error.errno, "commit marker exists but directory sync failed; "
                      "durability uncertain; inspect before reuse", str(final)) from error
    # Retain the tiny prepared file: no unlink/cleanup after the commit point.
    return publication

def publish_reviewed(staging, final, producer, job, token, publisher, verified_files=None):
    meta, review = review_staging(staging, producer, job)
    require(token == review["review_token"], "review token differs; inspect reports again")
    require(not os.path.lexists(final), "publication destination already exists; preserve it")
    started = time.monotonic()
    event("publication_inventory", status="start", staging=str(staging),
          expected_bytes=review["collection_bytes"], reused_current_inventory=verified_files is not None)
    current = inventory(staging / "collection") if verified_files is None else verified_files
    require(current == meta["collection_files"], "current collection differs from verified generation")
    # Detect metadata changes across the full hash pass.
    _, after = review_staging(staging, producer, job)
    require(after["review_token"] == token, "reports changed during inventory")
    event("publication_inventory", status="passed", elapsed_s=time.monotonic()-started)
    result = install_descriptor(staging, final, review, publisher)
    event("publication", status="complete", final=str(final), staging_retained=str(staging))
    return result

def capability(parent):
    """Tiny, explicitly requested test; keeps its unique probe directory for inspection."""
    parent = canonical(parent)
    require(".staging" not in parent.parts and "collection" not in parent.parts and
            not any((ancestor / "generation.json").exists() or (ancestor / "manifest.json").exists()
                    for ancestor in (parent, *parent.parents)),
            "probe must be outside generations, collections and experiment roots")
    probe = Path(tempfile.mkdtemp(prefix="sisap23-fs-probe-", dir=parent))
    result = {"probe_directory":str(probe), "production_artifacts_touched":False}
    try:
        source = probe / "source"
        with source.open("xb") as out:
            out.write(b"source")
            out.flush()
            os.fsync(out.fileno())
        destination = probe / "linked"
        os.link(source, destination)
        collision = probe / "collision"
        collision.write_bytes(b"existing")
        try:
            os.link(source, collision)
        except FileExistsError:
            pass
        else:
            raise AssertionError("hard link replaced an existing destination")
        require(collision.read_bytes() == b"existing", "collision content changed")
        sync_dir(probe)
        sync_dir(parent)
        result["descriptor_primitives"] = "passed"
        old, new = probe / "rename-source", probe / "rename-target"
        old.mkdir()
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            result["renameat2"] = "libc symbol unavailable"
        else:
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            code = rename(-100, os.fsencode(old), -100, os.fsencode(new), 1)
            result["renameat2"] = {"return_code":code, "errno":ctypes.get_errno() if code else 0,
                                  "source_exists":old.exists(), "target_exists":new.exists()}
    except Exception as error:
        result["descriptor_primitives"] = "FAILED"
        result["error"] = repr(error)
        event("capability", **result)
        raise
    event("capability", **result)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["inspect", "publish", "capability"])
    p.add_argument("--staging", type=Path)
    p.add_argument("--expected-producer")
    p.add_argument("--expected-job")
    p.add_argument("--final", type=Path)
    p.add_argument("--approve-publication", help="review token from the inspected reports")
    p.add_argument("--publisher-commit")
    p.add_argument("--confirm-prior-sync-completed", action="store_true")
    p.add_argument("--parent", type=Path)
    p.add_argument("--allow-create-probe", action="store_true")
    args = p.parse_args()
    if args.action == "capability":
        require(args.allow_create_probe and args.parent is not None, "explicit probe creation approval required")
        capability(args.parent)
        return
    require(args.staging and args.expected_producer and args.expected_job, "staging/producer/job required")
    meta, review = review_staging(args.staging, args.expected_producer, args.expected_job)
    if args.action == "inspect":
        print(json.dumps(review, indent=2, sort_keys=True))
        return
    require(args.final and args.approve_publication and args.publisher_commit and
            args.confirm_prior_sync_completed, "manual report approval and prior sync confirmation required")
    require(os.environ.get("PBS_JOBID"), "full inventory must run in a compute allocation, not the frontend")
    from recovery_stages import verify_checkout
    verify_checkout(args.publisher_commit)
    capability(args.final.parent)  # Fail before the large read-only hash pass.
    publish_reviewed(args.staging, args.final, args.expected_producer, args.expected_job,
                     args.approve_publication, args.publisher_commit)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        event("failure", error=repr(error), destructive_cleanup=False)
        raise
