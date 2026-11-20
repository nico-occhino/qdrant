"""PBS-only, no-rebuild recovery of a retained, constructed SISAP23 LMI index."""

import argparse
import os
from pathlib import Path
import shutil
import uuid

from preserve_generation import digest, inventory, write_new
from recovery_stages import (
    BASE, N, check_inputs, checked_generation, data_digest, fingerprint,
    original_manifest, publish_new, read, require, rust, timed,
)

BUILDER = "9b12d583b1bf9df802b2cc420244bd3c24732d86"
PRODUCER = "69216d5ada898feef7066c72da0514aeae2ead89"
MANIFEST_SHA256 = "5ab566ba0f641b177248d7fe42b83540a13b6f779da2c455ec815ea155d2ec96"


def canonical_existing(path):
    require(path.is_absolute() and path.exists() and path == path.resolve(),
            f"noncanonical or missing input: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--data-generation", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold-nfs-summary", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    args = parser.parse_args()

    job = os.environ["PBS_JOBID"]
    scratch = canonical_existing(Path(os.environ["SCRATCHDIR"]))
    require(job in Path("/proc/self/cgroup").read_text(), "not inside the PBS job cgroup")
    stage_source = canonical_existing(args.source_stage)
    original = canonical_existing(args.original)
    data_generation = canonical_existing(args.data_generation)
    cold_summary = canonical_existing(args.cold_nfs_summary)
    candidate_report = canonical_existing(args.candidate_report)
    coverage = read(candidate_report)
    expected_ids = [6706733, 8866671, 9621721, 2522962, 228421,
                    6583526, 2372432, 3413607, 8759799, 5619845]
    require(coverage.get("format") == "sisap23-row20-candidate-coverage-v1"
            and coverage.get("query_row") == 20 and coverage.get("nprobe") == 4
            and coverage.get("active_points") == N
            and coverage.get("scored_in_descending_order") is True
            and len(coverage.get("ground_truth_coverage", [])) == 10
            and [row["id_1_based"] for row in coverage.get("ranked_candidate_top10", [])] == expected_ids,
            "bounded candidate-coverage diagnostic is incomplete or disagrees with ordinary search")
    require(stage_source.parent.name == ".staging" and "24900597" in stage_source.name,
            "unexpected retained construction staging source")
    require(not (stage_source / "generation.json").exists(),
            "retained failed stage unexpectedly contains an acceptance descriptor")
    require(digest(data_generation / "generation.json") == MANIFEST_SHA256,
            "accepted Plain generation manifest changed")
    data_meta = checked_generation(data_generation, PRODUCER, "data", "data")
    manifest = original_manifest(original)
    timed("prepared_input_integrity", check_inputs, original, manifest)
    require(BUILDER == os.environ["SISAP23_REVIEWED_COMMIT"], "wrong reviewed source revision")

    output = args.output
    require(output.is_absolute() and output == output.resolve() and not output.exists(),
            "new canonical output root required")
    require(not output.is_relative_to(stage_source) and not output.is_relative_to(data_generation)
            and not stage_source.is_relative_to(output), "output overlaps preserved input")
    output.mkdir()
    from generation_publication import capability, event
    timed("publication_capability", capability, output)
    (output / ".staging").mkdir()
    token = f"recover-built-lmi-{job}-{uuid.uuid4().hex[:12]}"
    attempt = output / ".staging" / token
    attempt.mkdir()
    work = scratch / token
    work.mkdir()
    proof_collection = work / "collection"
    source_collection = stage_source / "collection"
    persistent_collection = attempt / "collection"
    try:
        old_report = read(stage_source / "scratch.json")
        require(old_report.get("format") == "sisap23-index-only-v2"
                and old_report.get("accepted") is True
                and old_report.get("variant") == "lmi"
                and old_report.get("ingested_points") == 0
                and old_report.get("index", {}).get("complete") is True,
                "historical native index report incomplete")
        require(read(stage_source / "scratch.exit.json").get("exit_code") == 0,
                "historical native index stage did not pass")
        require(data_digest(old_report) == data_meta["vector_fingerprint"],
                "built index vector fingerprint differs from accepted Plain source")
        original_inventory = read(stage_source / "copied-inventory.json")
        require(timed("retained_source_inventory", inventory, source_collection) == original_inventory,
                "retained LMI collection differs from its construction inventory")
        write_new(attempt / "source-inventory.json", original_inventory)
        shutil.copy2(stage_source / "scratch.json", attempt / "scratch.json")
        shutil.copy2(stage_source / "scratch.exit.json", attempt / "scratch.exit.json")

        timed("persistent_isolated_copy", shutil.copytree, source_collection, persistent_collection)
        require(timed("persistent_copy_inventory", inventory, persistent_collection) == original_inventory,
                "new persistent copy differs from retained construction")
        write_new(attempt / "copied-inventory.json", original_inventory)
        timed("scratch_query_copy", shutil.copytree, persistent_collection, proof_collection)
        require(timed("scratch_query_copy_inventory", inventory, proof_collection) == original_inventory,
                "query proof copy is not byte-identical to persistent collection")
        write_new(attempt / "query-proof-inventory.json", original_inventory)

        env = {
            "SISAP23_BENCH_ROOT": str(original),
            "SISAP23_RECOVERY_ROOT": str(output),
            "SISAP23_RECOVERY_COLLECTION": str(persistent_collection),
            "SISAP23_QUERY_PROOF_COLLECTION": str(proof_collection),
            "SISAP23_VERIFY_VARIANT": "lmi",
            "SISAP23_REVIEWED_COMMIT": BUILDER,
            "QDRANT_LMI_SAMPLE_BUDGET_BYTES": "768000000",
        }
        verified = rust("sisap23_recovery_verify", env, attempt, "persistent")
        require(data_digest(verified) == data_digest(old_report)
                and verified.get("index") == old_report.get("index")
                and verified.get("query_proof_storage") == "isolated_pbs_scratch"
                and verified.get("query_proof_collection") == str(proof_collection),
                "persistent structure, fingerprint, or isolated learned-query proof differs")
        files = timed("persistent_inventory_after_reopen", inventory, persistent_collection)
        require(files == original_inventory, "persistent collection changed during verification")
        require(timed("query_copy_inventory_after_search", inventory, proof_collection) == files,
                "query proof copy changed during ordinary search")
        require(timed("retained_source_inventory_after", inventory, source_collection) == original_inventory,
                "retained failed construction source changed")
        require(digest(data_generation / "generation.json") == MANIFEST_SHA256,
                "accepted Plain manifest changed during recovery")

        cold = read(cold_summary)
        require(cold.get("warm_query_path_passed") is True
                and cold.get("cold", {}).get("query_error") is not None,
                "cold NFS operational observation is incomplete")
        metadata = {
            "format": "sisap23-generation-v2", "accepted": True,
            "kind": "index", "variant": "lmi", "recovery_mode": "reuse-built-index",
            "producer_commit": BUILDER, "data_origin_commit": BASE,
            "data_producer_commit": PRODUCER, "points": N,
            "original_root": str(original),
            "original_manifest_sha256": digest(original / "manifest.json"),
            "source_path": str(source_collection),
            "source_generation": str(data_generation),
            "source_generation_sha256": MANIFEST_SHA256,
            "source_inventory_sha256": digest(attempt / "source-inventory.json"),
            "source_build_stage": str(stage_source),
            "source_build_job_id": "24900597.pbs-m1.metacentrum.cz",
            "source_build_report_sha256": digest(stage_source / "scratch.json"),
            "query_proof_scratchdir": str(scratch),
            "query_proof_collection": str(proof_collection),
            "query_proof_inventory_sha256": digest(attempt / "query-proof-inventory.json"),
            "cold_nfs_observation": {"source": str(cold_summary),
                                      "sha256": digest(cold_summary),
                                      "interpretation": "operational timeout; not an acceptance query proof"},
            "candidate_coverage_observation": {"source": str(candidate_report),
                                               "sha256": digest(candidate_report),
                                               "ground_truth_candidates_selected": coverage["ground_truth_candidates_selected"]},
            "vector_fingerprint": data_digest(verified),
            "collection_files": files, "report": old_report,
            "persistent_report": verified, "job_id": job,
            "hardware_fingerprint": fingerprint(output),
            "index_build_seconds": old_report.get("index_build_seconds"),
            "ingest_seconds": None, "ingested_points": 0,
        }
        write_new(attempt / "generation.json", metadata)
        final = output / token
        publish_new(attempt, final, verified_files=files)
        event("recovered_generation_accepted", final=str(final), stage_retained=str(attempt))
    except BaseException as error:
        event("recovery_validation_failed", error=repr(error), retained_staging=str(attempt),
              source_unchanged_by_design=True, automatic_retry=False)
        raise


if __name__ == "__main__":
    main()
