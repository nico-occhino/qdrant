"""Small, local recovery state-machine tests. No cluster or SISAP data."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import recovery_stages as stages
from preserve_generation import inventory, digest
from validate import check_recovery_provenance

COMMIT = "a" * 40

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.data = {"accepted":True, "points":stages.N,
                     "vector_fingerprint_xor_sha256":"fingerprint"}
        self.gen = self.root / "generation"
        (self.gen / "collection").mkdir(parents=True)
        (self.gen / "collection/vectors").write_bytes(b"synthetic vector data")
        self.meta = {"format":"sisap23-generation-v2","accepted":True,"kind":"data",
            "variant":"data","producer_commit":COMMIT,"points":stages.N,
            "vector_fingerprint":"fingerprint",
            "report":{"accepted":True,"ingested_points":0,"normal_restarts":2,
                      "first":{"corpus_vectors_compared":stages.N},"second":self.data},
            "persistent_report":{"accepted":True,"normal_restarts":2,"data":self.data},
            "collection_files":inventory(self.gen / "collection")}
        save(self.gen / "generation.json", self.meta)

    def test_complete_generation_accepted(self):
        self.assertEqual(stages.checked_generation(self.gen, COMMIT, "data"), self.meta)

    def test_changed_vectors_rejected(self):
        (self.gen / "collection/vectors").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            stages.checked_generation(self.gen, COMMIT)

    def test_partial_restart_or_wrong_source_rejected(self):
        with self.assertRaises(ValueError):
            stages.checked_generation(self.gen, "b" * 40)
        self.meta["persistent_report"]["normal_restarts"] = 1
        save(self.gen / "generation.json", self.meta)
        with self.assertRaises(ValueError):
            stages.checked_generation(self.gen, COMMIT)

    def test_missing_corpus_audit_rejected(self):
        self.meta["report"]["first"]["corpus_vectors_compared"] = stages.N - 1
        save(self.gen / "generation.json", self.meta)
        with self.assertRaises(ValueError):
            stages.checked_generation(self.gen, COMMIT)

    def test_staging_never_accepted(self):
        with self.assertRaises(ValueError):
            stages.checked_generation(self.root / ".staging/g", COMMIT)

    def test_unverified_stage_cannot_be_published(self):
        stage = self.root / ".staging/g"
        stage.mkdir(parents=True)
        (stage / "report").write_text("looks successful")
        target = self.root / "published"
        with self.assertRaises(FileNotFoundError):
            stages.publish_new(stage, target)
        self.assertTrue(stage.exists())
        self.assertFalse(target.exists())

    def test_cache_advice_restricted_to_own_scratch(self):
        scratch = self.root / "scratch"
        scratch.mkdir()
        with self.assertRaises(ValueError):
            stages.release_scratch_cache(self.gen, scratch)
        (scratch / "link").symlink_to(self.gen)
        with self.assertRaises(ValueError):
            stages.release_scratch_cache(scratch, scratch)

    def test_native_failure_does_not_accept_or_retry(self):
        attempt = self.root / "attempt"
        attempt.mkdir()
        with patch.object(stages.subprocess, "run") as run:
            run.return_value.returncode = 101
            run.return_value.stdout = "synthetic compilation failure"
            with self.assertRaises(ValueError):
                stages.rust("sisap23_recovery_index", {}, attempt, "scratch")
            self.assertEqual(run.call_count, 1)
        self.assertFalse((attempt / "generation.json").exists())

    def test_hnsw_is_independent_of_lmi(self):
        self.meta.update(kind="index", variant="hnsw-b")
        self.meta["report"].update(data=self.data, hnsw_params=[32,200])
        save(self.gen / "generation.json", self.meta)
        # No LMI path or marker is present.
        stages.checked_generation(self.gen, COMMIT, "index", "hnsw-b")
        with self.assertRaises(ValueError):
            stages.checked_generation(self.gen, COMMIT, "index", "hnsw-a")

    def test_lmi_requires_ordinary_search_path_proof(self):
        self.meta.update(kind="index", variant="lmi")
        self.meta["report"].update(data=self.data, lmi=stages.LMI)
        save(self.gen / "generation.json", self.meta)
        with self.assertRaisesRegex(ValueError, "ordinary learned search proof"):
            stages.checked_generation(self.gen, COMMIT, "index", "lmi")
        self.meta["persistent_report"]["ordinary_lmi_query_path"] = {
            "learned_delta":1, "plain_fallback_delta":0,
            "returned_ids_1_based":list(range(1, 11))}
        save(self.gen / "generation.json", self.meta)
        stages.checked_generation(self.gen, COMMIT, "index", "lmi")

    def test_evaluation_rejects_wrong_hardware_before_search(self):
        campaign = self.root / "campaign"
        save(campaign / "manifest.json", {"protocol":"independent-recovery-v2",
            "source_commit":COMMIT,"hardware_fingerprint":stages.HARDWARE})
        from types import SimpleNamespace
        with patch.object(stages, "fingerprint", return_value="other-host"):
            with self.assertRaisesRegex(ValueError, "host/CPU/filesystem"):
                stages.evaluate(SimpleNamespace(campaign=campaign, commit=COMMIT, original=self.root / "original"))

    def test_plain_provenance_and_inventory_are_retained(self):
        campaign = self.root / "campaign"
        run = campaign / "runs/plain"
        for folder in ("full", "panel"):
            (run / folder).mkdir(parents=True)
            (run / folder / "rows").write_text("historical")
        manifest = {"source_commit":COMMIT, "hardware_fingerprint":stages.HARDWARE,
                    "evaluation_sources":{"plain":stages.BASE}}
        save(campaign / "manifest.json", manifest)
        marker = {"format":"sisap23-evaluation-generation-v2","variant":"plain",
            "source_commit":stages.BASE,"origin":"imported-plain",
            "hardware_fingerprint":stages.HARDWARE,
            "result_inventories":{f:inventory(run / f) for f in ("full","panel")}}
        save(run / "evaluation.generation.json", marker)
        save(run / "evaluation.complete.json", {"accepted":True,"source_commit":stages.BASE,
            "hardware_fingerprint":stages.HARDWARE,
            "manifest_sha256":digest(campaign / "manifest.json"),
            "marker_sha256":digest(run / "evaluation.generation.json")})
        check_recovery_provenance(campaign, manifest, "plain")
        (run / "full/rows").write_text("changed")
        with self.assertRaises(ValueError):
            check_recovery_provenance(campaign, manifest, "plain")

    def test_pbs_environment_refuses_frontend_execution(self):
        import subprocess
        script = Path(__file__).parent / "pbs/recovery-env.sh"
        env = dict(os.environ)
        env.pop("PBS_JOBID", None)
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PBS compute allocation required", result.stderr)

    def test_checkout_requires_exact_clean_branch(self):
        with patch.object(stages.subprocess, "check_output",
                          side_effect=[COMMIT, stages.BRANCH, " M source"]):
            with self.assertRaises(ValueError):
                stages.verify_checkout(COMMIT)
        with patch.object(stages.subprocess, "check_output",
                          side_effect=[COMMIT, stages.BRANCH, ""]):
            stages.verify_checkout(COMMIT)

if __name__ == "__main__":
    unittest.main()
