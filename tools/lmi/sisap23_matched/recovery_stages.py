"""Opt-in SISAP23 recovery/build/evaluation stages. Run on PBS compute nodes only."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import uuid

from preserve_generation import digest, inventory, write_new

BASE = "59f9550854db66d5c4ed5e5464a8c9132c07b6ce"
BRANCH = "thesis/sisap23-lmi-memory-recovery"
HARDWARE = "deimos11.meta.zcu.cz|AMD EPYC 9555 64-Core Processor|nfs"
N = 10_120_191
LMI = dict(n_buckets=3162, sample_size=250000, hidden_dim=512, epochs=30,
           batch_size=256, routing_batch_size=256, kmeans_iterations=5, nprobe=4, seed=42)
REPO = Path(__file__).resolve().parents[3]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    return json.loads(path.read_text())

def checked_relative(raw):
    path = Path(raw)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe relative path")
    return path

def fingerprint(root):
    cpu = subprocess.check_output(["lscpu"], text=True, env={**os.environ, "LC_ALL": "C"})
    model = next(line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("Model name:"))
    fs = subprocess.check_output(["stat", "-f", "-c", "%T", str(root)], text=True).strip()
    return f"{socket.gethostname()}|{model}|{fs}"

def verify_checkout(commit):
    require(len(commit) == 40, "full reviewed commit SHA required")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    require(actual == commit and branch == BRANCH and not dirty, "source commit/branch/cleanliness mismatch")

def original_manifest(root):
    manifest = read(root / "manifest.json")
    state = dict(line.split("=", 1) for line in (root / "suite.state").read_text().splitlines())
    require(manifest["source_commit"] == BASE == state["source_commit"], "wrong incident source commit")
    require(state["manifest_sha256"] == digest(root / "manifest.json"), "incident manifest changed")
    require(state["hardware_fingerprint"] == HARDWARE, "incident hardware fingerprint differs")
    return manifest

def check_inputs(root, manifest):
    for key, item in manifest["prepared"].items():
        path = root / checked_relative(item["path"])
        require(not path.is_symlink() and path.stat().st_size == item["bytes"]
                and digest(path) == item["sha256"], f"prepared input changed: {key}")

def checked_generation(path, commit, kind=None, variant=None):
    require(path.is_absolute() and ".staging" not in path.parts and not path.is_symlink(),
            "accepted generation required, not staging or symlink")
    meta = read(path / "generation.json")
    require(meta.get("format") == "sisap23-generation-v2" and meta.get("accepted") is True,
            "not an accepted recovery generation")
    require(meta["producer_commit"] == commit, "generation source commit differs")
    require(meta["points"] == N, "generation count differs")
    require(meta["persistent_report"].get("accepted") is True
            and meta["persistent_report"].get("normal_restarts") == 2
            and data_digest(meta["persistent_report"]) == meta["vector_fingerprint"],
            "persistent restart audit incomplete")
    require(meta["report"].get("accepted") is True
            and meta["report"].get("ingested_points") == 0
            and data_digest(meta["report"]) == meta["vector_fingerprint"],
            "scratch audit incomplete")
    if meta["kind"] == "data":
        require(meta["report"].get("normal_restarts") == 2
                and meta["report"]["first"].get("corpus_vectors_compared") == N,
                "exhaustive corpus validation missing")
    elif meta["variant"] == "lmi":
        require(meta["report"].get("lmi") == LMI, "fixed LMI configuration changed")
    else:
        require(meta["report"].get("hnsw_params") == {"hnsw-a":[16,100], "hnsw-b":[32,200]}.get(meta["variant"]),
                "fixed HNSW configuration changed")
    if kind:
        require(meta["kind"] == kind, "wrong generation kind")
    if variant:
        require(meta["variant"] == variant, "wrong generation variant")
    require(inventory(path / "collection") == meta["collection_files"], "generation file inventory changed")
    return meta

def sync_tree(path):
    for file in path.rglob("*"):
        if file.is_file():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
    directories = [p for p in path.rglob("*") if p.is_dir()] + [path]
    for directory in directories:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

def publish_new(staging, final):
    """Linux renameat2(RENAME_NOREPLACE), fail closed when unavailable."""
    require(staging.parent.name == ".staging" and staging.parent.parent == final.parent,
            "publication must stay on one filesystem")
    sync_tree(staging)
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    require(rename is not None, "renameat2 is required for atomic no-replace publication")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if rename(-100, os.fsencode(staging), -100, os.fsencode(final), 1) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(final))
    fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def release_scratch_cache(path, scratch):
    """Only advise clean pages of our verified scratch copy; never drop host caches."""
    require(path.resolve() != scratch.resolve() and path.resolve().is_relative_to(scratch.resolve()),
            "cache advice outside a private scratch subdirectory refused")
    advised = 0
    for file in path.rglob("*"):
        require(not file.is_symlink(), "scratch symlink refused")
        if file.is_file():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            advised += file.stat().st_size
    return advised

def rust(name, env, attempt, label):
    log = attempt / f"{label}.log"
    resources = attempt / f"{label}.resources.txt"
    report = attempt / f"{label}.json"
    env = {**os.environ, **env, "SISAP23_RECOVERY_REPORT": str(report)}
    # Compile separately: native CPU/RSS measurements must exclude rustc/linker.
    built = subprocess.run(["cargo", "test", "-p", "collection", "--release",
        "--features", "lmi-training", "--locked", "--lib", "--no-run", "--message-format=json"],
        cwd=REPO, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (attempt / f"{label}.compile.log").write_text(built.stdout)
    require(built.returncode == 0, "compilation failed; see compile log; no native test executed")
    binaries = []
    for line in built.stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if (item.get("reason") == "compiler-artifact" and item.get("executable")
                and item.get("target", {}).get("name") == "collection"
                and item.get("profile", {}).get("test")):
            binaries.append(item["executable"])
    require(len(set(binaries)) == 1, "unique collection test executable not found")
    with log.open("x") as out:
        result = subprocess.run(["/usr/bin/time", "-v", "-o", str(resources),
            binaries[0], name, "--ignored", "--nocapture", "--test-threads=1"],
            cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
    write_new(attempt / f"{label}.exit.json", {"exit_code": result.returncode})
    require(result.returncode == 0, f"{name} failed; inspect {log}; no automatic retry")
    if name == "sisap23_matched_evaluate":
        require(f"sisap23_native_evaluation_complete variant={env['SISAP23_VARIANT']}" in log.read_text(),
                "evaluator did not execute")
        return None
    value = read(report)
    require(value.get("accepted") is True, f"missing acceptance: {report}")
    return value

def data_digest(report):
    data = report.get("data", report.get("second", {}))
    require(data.get("accepted") is True and data.get("points") == N, "incomplete vector audit")
    return data["vector_fingerprint_xor_sha256"]

def construct(args):
    manifest = original_manifest(args.original)
    check_inputs(args.original, manifest)
    output = args.output.resolve()
    require(not output.is_relative_to(args.original.resolve()), "recovery output overlaps incident")
    output.mkdir(parents=True, exist_ok=True)
    (output / ".staging").mkdir(exist_ok=True)
    token = f"{args.action}-{args.variant}-{os.environ['PBS_JOBID']}-{uuid.uuid4().hex[:12]}"
    attempt = output / ".staging" / token
    attempt.mkdir()
    scratch = Path(os.environ["SCRATCHDIR"]).resolve()
    work = scratch / token
    work.mkdir()
    copied = work / "collection"
    source_meta = None
    if args.action == "recover":
        source = args.original / "runs" / args.source_variant / "collection"
        if args.source_variant == "lmi":
            require(not (source.parent / "build.complete.json").exists(), "LMI is already marked complete")
    else:
        source_meta = checked_generation(args.data_generation, args.commit, kind="data")
        source = args.data_generation / "collection"
    before = inventory(source)
    write_new(attempt / "source-inventory.json", before)
    copied_started = time.monotonic()
    shutil.copytree(source, copied)
    require(inventory(copied) == before, "scratch copy differs from source")
    require(inventory(source) == before, "source changed during copying")
    copy_seconds = time.monotonic() - copied_started
    advised = release_scratch_cache(copied, scratch)
    env = {
        "SISAP23_BENCH_ROOT": str(args.original), "SISAP23_RECOVERY_ROOT": str(output),
        "SISAP23_RECOVERY_COLLECTION": str(copied), "SISAP23_VARIANT": args.variant,
        "SISAP23_REVIEWED_COMMIT": args.commit,
        "QDRANT_LMI_SAMPLE_BUDGET_BYTES": "768000000",
    }
    if args.action == "recover":
        corpus = work / "corpus.f16le"
        shutil.copy2(args.original / "corpus.f16le", corpus)
        require(digest(corpus) == manifest["prepared"]["corpus"]["sha256"], "scratch corpus hash differs")
        env["SISAP23_CORPUS_PATH"] = str(corpus)
        name, kind, variant = "sisap23_recovery_vectors", "data", "data"
    else:
        name, kind, variant = "sisap23_recovery_index", "index", args.variant
    try:
        report = rust(name, env, attempt, "scratch")
        if source_meta:
            require(data_digest(report) == source_meta["vector_fingerprint"], "index changed input vectors")
        scratch_files = inventory(copied)
        shutil.copytree(copied, attempt / "collection")
        require(inventory(attempt / "collection") == scratch_files, "persistent staging copy differs")
        write_new(attempt / "copied-inventory.json", scratch_files)
        env["SISAP23_RECOVERY_COLLECTION"] = str(attempt / "collection")
        env["SISAP23_VERIFY_VARIANT"] = variant
        verified = rust("sisap23_recovery_verify", env, attempt, "persistent")
        require(data_digest(verified) == data_digest(report), "persistent vector fingerprint differs")
        require(inventory(source) == before, "source changed; publication refused")
        files = inventory(attempt / "collection")
        meta = {
            "format": "sisap23-generation-v2", "accepted": True, "kind": kind, "variant": variant,
            "producer_commit": args.commit, "data_origin_commit": BASE, "points": N,
            "original_root": str(args.original), "original_manifest_sha256": digest(args.original / "manifest.json"),
            "source_path": str(source), "source_generation": str(args.data_generation) if source_meta else None,
            "source_inventory_sha256": digest(attempt / "source-inventory.json"),
            "source_generation_sha256": digest(args.data_generation / "generation.json") if source_meta else None,
            "vector_fingerprint": data_digest(verified), "collection_files": files,
            "report": report, "persistent_report": verified, "job_id": os.environ["PBS_JOBID"],
            "hardware_fingerprint": fingerprint(output), "scratch_copy_seconds": copy_seconds,
            "scratch_cache_bytes_advised": advised,
            "index_build_seconds": report.get("index_build_seconds"),
            "ingest_seconds": None, "ingested_points": 0,
            "historical_ingest_seconds_observation": 22401.5 if args.action == "recover" and args.source_variant == "lmi" else None,
        }
        write_new(attempt / "generation.json", meta)
        final = output / token
        publish_new(attempt, final)
        print(f"ACCEPTED_GENERATION={final}", flush=True)
    finally:
        require(inventory(source) == before, "original/source inventory changed")

def validate_variant(root, variant):
    import math
    import numpy as np
    from validate import check_ids, read_jsonl
    gold = np.fromfile(root / "gold_20_9999_top10.i32le", dtype="<i4").reshape(9980, 10)
    plain = read_jsonl(root / "runs/plain/full/full/rows.jsonl")
    require([r["query_row"] for r in plain] == list(range(20, 10000)), "Plain reference incomplete")
    panel = np.fromfile(root / "panel_rows.u32le", dtype="<u4").astype(int).tolist()
    require(len(panel) == 1000 and len(set(panel)) == 1000, "invalid panel input")
    settings = (["full"] if variant == "plain" else
        [f"nprobe-{p}" for p in (1,4,8,16,30,40,64,100,160,256)] if variant == "lmi" else
        [f"hnsw-ef-{p}" for p in (16,32,64,128,256,512,1024)])
    for setting in settings:
        conditions = [(f"full/{setting}", list(range(20,10000)))] + [
            (f"panel/rep-{rep}/{setting}", panel) for rep in (1,2,3)]
        for folder, ids in conditions:
            path = root / "runs" / variant / folder
            marker = read(path / "condition.complete.json")
            require(marker.get("complete") is True and marker["query_count"] == len(ids)
                    and marker["setting"] == setting, "condition completion differs")
            rows = read_jsonl(path / "rows.jsonl")
            require([r["query_row"] for r in rows] == ids, "condition rows differ")
            require(all(r["parameter"] == setting and math.isfinite(r["native_search_ms"])
                        and r["native_search_ms"] > 0 for r in rows), "invalid timing or setting")
            require(all(all(1 <= int(i) <= N for i in r["returned_ids_1_based"]) for r in rows),
                    "returned ID outside corpus")
            if variant == "plain":
                require(all(r["candidate_count"] == N for r in rows), "Plain full-scan evidence differs")
            elif variant == "lmi":
                require(all(isinstance(r["candidate_count"], int) and 10 <= r["candidate_count"] <= N
                            for r in rows), "invalid learned candidate count")
            else:
                require(all(r["hnsw_ef"] == int(setting.rsplit("-",1)[1]) for r in rows), "HNSW ef differs")
            check_ids(rows, path, gold, plain)
            chunks = sorted((path / "chunks").glob("rows-*.jsonl"))
            require(len(chunks) == marker["chunks"], "chunk count differs")
            require([r for chunk in chunks for r in read_jsonl(chunk)] == rows,
                    "saved chunks do not reproduce assembled results")

def validate_plain(root, manifest):
    validate_variant(root, "plain")

def init_campaign(args):
    manifest = original_manifest(args.original)
    check_inputs(args.original, manifest)
    validate_plain(args.original, manifest)
    require(not args.campaign.exists(), "campaign exists; no overwrite or repair in place")
    require(not args.campaign.resolve().is_relative_to(args.original.resolve()), "campaign overlaps incident")
    args.campaign.mkdir()
    for item in manifest["prepared"].values():
        relative = checked_relative(item["path"])
        target = args.campaign / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.original / relative, target)
    check_inputs(args.campaign, manifest)
    updated = dict(manifest)
    updated.update(source_commit=args.commit, format="sisap23-matched-v1",
        protocol="independent-recovery-v2", hardware_fingerprint=HARDWARE,
        evaluation_sources={"plain": BASE, "lmi": args.commit, "hnsw-a": args.commit, "hnsw-b": args.commit},
        original_manifest_sha256=digest(args.original / "manifest.json"),
        original_root=str(args.original))
    write_new(args.campaign / "manifest.json", updated)
    run = args.campaign / "runs/plain"
    run.mkdir(parents=True)
    result_inventories = {}
    for folder in ("full", "panel"):
        source = args.original / "runs/plain" / folder
        inv = inventory(source)
        shutil.copytree(source, run / folder)
        require(inventory(run / folder) == inv and inventory(source) == inv, "Plain results copy differs")
        result_inventories[folder] = inv
    validate_plain(args.campaign, updated)
    marker = {"format":"sisap23-evaluation-generation-v2","variant":"plain","origin":"imported-plain",
        "source_commit":BASE,"producer_commit":None,"point_count":N,"segment_count":1,"index_kind":"plain",
        "hardware_fingerprint":HARDWARE,"original_root":str(args.original),
        "result_inventories":result_inventories,"historical_ingest_seconds":None}
    write_new(run / "evaluation.generation.json", marker)
    write_new(run / "evaluation.complete.json", {"accepted":True,"hardware_fingerprint":HARDWARE,
        "source_commit":BASE,"manifest_sha256":digest(args.campaign / "manifest.json"),
        "marker_sha256":digest(run / "evaluation.generation.json"),"origin":"imported-plain"})
    print(f"CAMPAIGN={args.campaign}")

def evaluate(args):
    # One timed variant per campaign; concurrent benchmark jobs would contaminate latency.
    import fcntl
    require(args.campaign.is_absolute()
            and not args.campaign.resolve().is_relative_to(args.original.resolve()),
            "evaluation campaign overlaps incident")
    with (args.campaign / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return evaluate_locked(args)

def evaluate_locked(args):
    manifest = read(args.campaign / "manifest.json")
    require(manifest.get("protocol") == "independent-recovery-v2" and manifest["source_commit"] == args.commit,
            "campaign source commit differs")
    require(fingerprint(args.campaign) == manifest["hardware_fingerprint"] == HARDWARE,
            "query host/CPU/filesystem differs from deimos11; do not mix latency")
    check_inputs(args.campaign, manifest)
    require(args.variant != "plain", "Plain measurements already preserved; no implicit rerun")
    meta = checked_generation(args.index_generation, args.commit, kind="index", variant=args.variant)
    run = args.campaign / "runs" / args.variant
    generation_sha = digest(args.index_generation / "generation.json")
    if run.exists():
        marker = read(run / "evaluation.generation.json")
        require(marker["generation_sha256"] == generation_sha and marker["source_commit"] == args.commit,
                "cannot resume with a different index generation or source")
    else:
        run.mkdir()
        shutil.copytree(args.index_generation / "collection", run / "collection")
        require(inventory(run / "collection") == meta["collection_files"], "evaluation copy differs")
        segment = run / "collection" / checked_relative(meta["report"]["index"]["segment_relative_path"])
        require((segment / "segment.json").is_file(), "verified nonempty segment missing")
        m, ef = {"lmi":(None,None), "hnsw-a":(16,100), "hnsw-b":(32,200)}[args.variant]
        marker = {"format":"sisap23-evaluation-generation-v2","variant":args.variant,"origin":"index-generation",
            "source_commit":args.commit,"producer_commit":meta["producer_commit"],
            "point_count":N,"segment_count":1,"index_kind":"lmi" if args.variant == "lmi" else "hnsw",
            "hnsw_m":m,"ef_construct":ef,"lmi":LMI if args.variant == "lmi" else None,
            "hardware_fingerprint":HARDWARE,"generation_path":str(args.index_generation),
            "generation_sha256":generation_sha,"segment_path":str(segment)}
        write_new(run / "evaluation.generation.json", marker)
    require(inventory(run / "collection") == meta["collection_files"], "evaluation collection changed")
    complete = run / "evaluation.complete.json"
    if complete.exists():
        saved = read(complete)
        require(saved["accepted"] is True and saved["marker_sha256"] == digest(run / "evaluation.generation.json")
                and saved["manifest_sha256"] == digest(args.campaign / "manifest.json")
                and saved["hardware_fingerprint"] == HARDWARE, "saved evaluation completion differs")
        require(saved["result_inventories"] == {f:inventory(run / f) for f in ("full","panel")},
                "accepted query results changed")
        print("evaluation already accepted")
        return
    attempt = run / f"attempt-{os.environ['PBS_JOBID']}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir()
    rust("sisap23_matched_evaluate", {"SISAP23_BENCH_ROOT":str(args.campaign),
        "SISAP23_VARIANT":args.variant,"SISAP23_REVIEWED_COMMIT":args.commit}, attempt, "evaluate")
    require(inventory(run / "collection") == meta["collection_files"], "query evaluation changed collection")
    require(fingerprint(args.campaign) == HARDWARE, "hardware changed")
    validate_variant(args.campaign, args.variant)
    write_new(complete, {"accepted":True,"source_commit":args.commit,"hardware_fingerprint":HARDWARE,
        "result_inventories":{f:inventory(run / f) for f in ("full","panel")},
        "marker_sha256":digest(run / "evaluation.generation.json"),
        "manifest_sha256":digest(args.campaign / "manifest.json")})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["probe","recover","build","init","evaluate","validate"])
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--variant", choices=["lmi","hnsw-a","hnsw-b"], default="lmi")
    p.add_argument("--source-variant", choices=["lmi","plain"], default="lmi")
    p.add_argument("--data-generation", type=Path)
    p.add_argument("--index-generation", type=Path)
    p.add_argument("--campaign", type=Path)
    args = p.parse_args()
    require(os.environ.get("PBS_JOBID") and os.environ.get("SCRATCHDIR"), "PBS compute allocation required")
    require(args.original.is_absolute() and args.output.is_absolute(), "absolute paths required")
    verify_checkout(args.commit)
    if args.action == "probe":
        require(not args.output.resolve().is_relative_to(args.original.resolve()), "probe output overlaps incident")
        attempt = args.output / "probes" / f"{os.environ['PBS_JOBID']}-{uuid.uuid4().hex[:12]}"
        attempt.mkdir(parents=True)
        rust("sisap23_recovery_memory_probe", {"QDRANT_LMI_SAMPLE_BUDGET_BYTES":"768000000"},
             attempt, "admission")
        print(f"ACCEPTED_PROBE={attempt}")
    elif args.action in ("recover","build"):
        require(args.action != "build" or args.data_generation is not None, "data generation required")
        construct(args)
    elif args.action == "init":
        require(args.campaign is not None, "campaign required")
        init_campaign(args)
    elif args.action == "evaluate":
        require(args.campaign is not None and args.index_generation is not None, "campaign and index required")
        evaluate(args)
    else:
        require(args.campaign is not None, "campaign required")
        subprocess.run(["python", str(REPO / "tools/lmi/sisap23_matched/validate.py"),
                        "--root", str(args.campaign)], check=True)

if __name__ == "__main__":
    main()
