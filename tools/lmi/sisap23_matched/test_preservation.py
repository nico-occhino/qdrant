"""Synthetic, small-file tests of generation acceptance and provenance."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original = self.root / "original"
        self.scratch = self.root / "scratch"
        self.stage = self.root / "generations/.staging/g1"
        self.final = self.root / "generations/g1"
        for collection in (self.original, self.scratch, self.stage / "collection"):
            collection.mkdir(parents=True)
            (collection / "collection.json").write_bytes(b"all vectors")
        self.wal = self.root / "wal.json"
        save(self.wal, {"accepted": True, "completion_marker_created": False,
                        "persisted_after_restart": {"total_vectors": 10120191}})
        self.report = self.root / "verified.json"
        save(self.report, {"accepted": True, "recovered_vectors_compared": 1679,
                           "index_files_unchanged": True,
                           "first_reopen": {"complete_and_valid": True, "total_vectors": 10120191},
                           "second_reopen": {"complete_and_valid": True, "total_vectors": 10120191}})
        self.args = [sys.executable, str(HERE / "preserve_generation.py"),
                     "--scratch", str(self.scratch), "--staging", str(self.stage),
                     "--final", str(self.final), "--report", str(self.report),
                     "--wal-report", str(self.wal), "--commit", "reviewed",
                     "--job-id", "1.pbs", "--original", str(self.original)]

    def run_action(self, action, success=True):
        result = subprocess.run(self.args[:2] + [action] + self.args[2:],
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_copy_verified_then_reopen_and_publish(self):
        self.run_action("inventory")
        self.run_action("verify")
        self.run_action("finalize")
        info = json.loads((self.final / "generation.json").read_text())
        self.assertTrue(info["accepted"])
        self.assertEqual(info["collection_files"]["collection.json"]["sha256"], sha(b"all vectors"))
        self.assertFalse(self.stage.exists())
        self.assertTrue(self.original.joinpath("collection.json").exists())

    def test_changed_copy_fails_closed(self):
        self.run_action("inventory")
        (self.stage / "collection/collection.json").write_bytes(b"changed")
        self.run_action("verify", False)
        self.assertFalse(self.final.exists())

    def test_rejected_wal_cannot_publish(self):
        save(self.wal, {"accepted": False, "completion_marker_created": False,
                        "persisted_after_restart": {"total_vectors": 10120191}})
        self.run_action("inventory", False)
        self.assertFalse(self.final.exists())

    def test_imported_plain_keeps_original_build_provenance(self):
        prepared = self.root / "old-run"
        prepared.mkdir()
        (prepared / "corpus.f16le").write_bytes(b"synthetic")
        save(prepared / "manifest.json", {
            "format": "sisap23-matched-v1",
            "source_commit": "42a163701a51078ec24983b3a93f96ab7d661489",
            "prepared": {"corpus": {"path": "corpus.f16le", "bytes": 9,
                                     "sha256": sha(b"synthetic")}}})
        generation = self.root / "accepted"
        (generation / "collection/0/segments/id").mkdir(parents=True)
        (generation / "collection/0/segments/id/segment.json").write_bytes(b"segment")
        from tools.lmi.sisap23_matched.preserve_generation import inventory
        save(generation / "generation.json", {
            "accepted": True, "point_count": 10120191,
            "preservation_commit": "reviewed",
            "original_source_commit": "42a163701a51078ec24983b3a93f96ab7d661489",
            "collection_files": inventory(generation / "collection")})
        new = self.root / "new-run"
        cmd = [sys.executable, str(HERE / "bootstrap_recovered_suite.py"),
               "--original", str(prepared), "--generation", str(generation),
               "--new-root", str(new), "--commit", "reviewed"]
        first = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        marker = json.loads((new / "runs/plain/imported.generation.json").read_text())
        self.assertEqual(marker["source_commit"], "42a163701a51078ec24983b3a93f96ab7d661489")
        self.assertFalse((new / "runs/plain/build.complete.json").exists())
        self.assertEqual(json.loads((new / "manifest.json").read_text())["source_commit"], "reviewed")
        self.assertTrue((new / "runs/plain/collection/0/segments/id/segment.json").is_file())
        (new / "runs/plain/collection/0/segments/id/segment.json").write_bytes(b"changed")
        mutated = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(mutated.returncode, 0, "changed matched Plain copy must fail closed")

if __name__ == "__main__":
    unittest.main()
