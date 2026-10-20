"""Publication regressions on tiny synthetic collections, without Qdrant or cluster access."""
import contextlib
import ctypes
import errno
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generation_publication as pub
import recovery_stages as stages
from preserve_generation import digest, inventory

PRODUCER = "a" * 40
PUBLISHER = "b" * 40
JOB = "123.pbs"
FINGERPRINT = "c" * 64

def save(path, value):
    path.write_text(json.dumps(value))

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stage = self.root / ".staging/g"
        (self.stage / "collection").mkdir(parents=True)
        (self.stage / "collection/vectors").write_bytes(b"small test payload")
        self.final = self.root / "published-g"
        data = {"accepted":True, "points":stages.N, "dimension":768,
                "datatype":"float16", "distance":"cosine", "missing_ids":0,
                "duplicate_ids":0, "invalid_mappings":0, "absent_vectors":0,
                "vector_fingerprint_xor_sha256":FINGERPRINT,
                "corpus_vectors_compared":stages.N, "corpus_max_abs_error":0.0,
                "physical_index_kinds":["plain"]}
        scratch = {"format":"sisap23-recovered-data-v2", "accepted":True,
                   "first":dict(data), "second":dict(data), "normal_restarts":2, "ingested_points":0}
        persistent = {"format":"sisap23-persistent-verified-v2", "accepted":True,
                      "data":dict(data), "normal_restarts":2, "ingested_points":0,
                      "index_files_unchanged":True}
        files = inventory(self.stage / "collection")
        for name in ("source-inventory.json", "copied-inventory.json"):
            save(self.stage / name, files)
        for name in ("scratch.exit.json", "persistent.exit.json"):
            save(self.stage / name, {"exit_code":0})
        save(self.stage / "scratch.json", scratch)
        save(self.stage / "persistent.json", persistent)
        self.meta = {"format":"sisap23-generation-v2", "accepted":True, "kind":"data",
            "variant":"data", "producer_commit":PRODUCER, "points":stages.N, "job_id":JOB,
            "data_origin_commit":stages.BASE, "original_manifest_sha256":"d"*64,
            "original_root":str(self.root / "original"),
            "source_path":str(self.root / "original/runs/lmi/collection"),
            "vector_fingerprint":FINGERPRINT, "report":scratch, "persistent_report":persistent,
            "source_inventory_sha256":digest(self.stage / "source-inventory.json"),
            "collection_files":files}
        self.persist()

    def persist(self):
        save(self.stage / "generation.json", self.meta)

    def review(self):
        return pub.review_staging(self.stage, PRODUCER, JOB)[1]

    def publish(self, **kwargs):
        token = kwargs.pop("token", self.review()["review_token"])
        with contextlib.redirect_stdout(io.StringIO()):
            return pub.publish_reviewed(self.stage, self.final, PRODUCER, JOB, token, PUBLISHER, **kwargs)

    def test_review_is_metadata_only_and_does_not_write(self):
        before = inventory(self.stage)
        with patch.object(pub, "inventory", side_effect=AssertionError("collection scan")):
            result = self.review()
        self.assertEqual(result["current_collection_hashes"], "NOT CHECKED")
        self.assertEqual(before, inventory(self.stage))
        self.assertFalse(self.final.exists())

    def test_cli_inspect_is_read_only(self):
        args = ["publication", "inspect", "--staging", str(self.stage),
                "--expected-producer", PRODUCER, "--expected-job", JOB]
        before = inventory(self.stage)
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()) as out:
            pub.main()
        self.assertEqual(json.loads(out.getvalue())["publication"], "NOT PERFORMED")
        self.assertEqual(inventory(self.stage), before)
        self.assertFalse(self.final.exists())

    def test_cli_missing_manual_approval_cannot_publish(self):
        args = ["publication", "publish", "--staging", str(self.stage), "--final", str(self.final),
                "--expected-producer", PRODUCER, "--expected-job", JOB]
        with patch("sys.argv", args):
            with self.assertRaisesRegex(ValueError, "manual report approval"):
                pub.main()
        self.assertFalse(self.final.exists())

    def test_wrong_job_or_producer_is_rejected(self):
        with self.assertRaises(ValueError):
            pub.review_staging(self.stage, PRODUCER, "other.pbs")
        with self.assertRaises(ValueError):
            pub.review_staging(self.stage, PUBLISHER, JOB)

    def test_publish_retains_stage_and_preserves_producer(self):
        before = inventory(self.stage)
        self.publish()
        self.assertEqual(inventory(self.stage), before)
        self.assertEqual(pub.storage_path(self.final), self.stage)
        self.assertEqual(stages.checked_generation(self.final, PRODUCER, "data"), self.meta)
        with self.assertRaises(ValueError):
            stages.checked_generation(self.final, PUBLISHER, "data")
        self.assertEqual(json.loads((self.final / "publication.json").read_text())["publisher_commit"], PUBLISHER)

    def test_directory_collision_never_replaced(self):
        self.final.mkdir()
        (self.final / "keep").write_bytes(b"existing")
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual((self.final / "keep").read_bytes(), b"existing")

    def test_file_and_dangling_symlink_collisions(self):
        self.final.write_bytes(b"existing")
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.final.read_bytes(), b"existing")
        self.final.unlink()  # Synthetic fixture only.
        self.final.symlink_to(self.root / "absent")
        with self.assertRaises(ValueError):
            self.publish()
        self.assertTrue(self.final.is_symlink())

    def test_changed_payload_rejected_before_reservation(self):
        (self.stage / "collection/vectors").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse(self.final.exists())

    def test_review_token_must_match_current_reports(self):
        token = self.review()["review_token"]
        self.meta["new_provenance_field"] = "changed"
        self.persist()
        with self.assertRaises(ValueError):
            self.publish(token=token)
        self.assertFalse(self.final.exists())

    def test_mismatched_saved_report_is_rejected(self):
        report = json.loads((self.stage / "persistent.json").read_text())
        report["data"]["points"] -= 1
        save(self.stage / "persistent.json", report)
        with self.assertRaises(ValueError):
            self.review()

    def test_partial_reports_and_native_failure_rejected(self):
        save(self.stage / "persistent.exit.json", {"exit_code":101})
        with self.assertRaises(ValueError):
            self.review()
        (self.stage / "persistent.exit.json").unlink()
        with self.assertRaises(ValueError):
            self.review()

    def test_incomplete_first_audit_cannot_hide_behind_second(self):
        self.meta["report"]["first"]["missing_ids"] = 1
        save(self.stage / "scratch.json", self.meta["report"])
        self.persist()
        with self.assertRaises(ValueError):
            self.review()

    def test_interrupt_before_commit_marker_is_not_accepted(self):
        with patch.object(pub.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "unsupported")):
            with self.assertRaises(OSError):
                self.publish()
        self.assertTrue(self.stage.is_dir())
        self.assertTrue(self.final.is_dir())
        self.assertFalse((self.final / "publication.json").exists())
        with self.assertRaises(ValueError):
            pub.storage_path(self.final)
        with self.assertRaises(ValueError):
            self.publish()

    def test_directory_sync_failure_before_marker_leaves_rejected_partial(self):
        with patch.object(pub, "sync_dir", side_effect=OSError(errno.EINVAL, "unsupported dir sync")):
            with self.assertRaises(OSError):
                self.publish()
        self.assertFalse((self.final / "publication.json").exists())
        with self.assertRaises(ValueError):
            pub.storage_path(self.final)

    def test_directory_sync_failure_after_marker_reports_uncertain_durability(self):
        with patch.object(pub, "sync_dir", side_effect=[None, None, OSError(errno.EIO, "sync")]):
            with self.assertRaisesRegex(OSError, "durability uncertain"):
                self.publish()
        self.assertTrue((self.final / "publication.json").exists())
        self.assertTrue(self.stage.exists())

    def test_capability_refuses_staging_parent(self):
        with self.assertRaises(ValueError):
            pub.capability(self.stage)
        self.assertFalse(list(self.stage.glob("sisap23-fs-probe-*")))

    def test_rename_is_not_needed_even_when_unsupported(self):
        with patch.object(pub.ctypes, "CDLL", side_effect=OSError(errno.EINVAL, "unsupported")),              patch.object(pub.os, "rename", side_effect=AssertionError("unsafe rename")),              patch.object(pub.os, "replace", side_effect=AssertionError("unsafe replace")):
            self.publish()
        self.assertEqual(pub.storage_path(self.final), self.stage)

    def test_nfs_link_error_after_server_commit_is_reported_uncertain(self):
        real_link = os.link
        def uncertain(source, destination, **kwargs):
            real_link(source, destination, **kwargs)
            raise OSError(errno.EIO, "server acknowledgement lost")
        with patch.object(pub.os, "link", side_effect=uncertain):
            with self.assertRaisesRegex(OSError, "uncertain"):
                self.publish()
        self.assertTrue((self.final / "publication.json").is_file())
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(pub.storage_path(self.final), self.stage)

    def test_marker_link_collision_preserves_existing_record(self):
        def collide(source, destination, **kwargs):
            Path(destination).write_bytes(b"existing")
            raise FileExistsError(errno.EEXIST, "exists")
        with patch.object(pub.os, "link", side_effect=collide):
            with self.assertRaises(OSError):
                self.publish()
        self.assertEqual((self.final / "publication.json").read_bytes(), b"existing")
        self.assertTrue(self.stage.is_dir())

    def test_descriptor_metadata_change_or_path_escape_rejected(self):
        self.publish()
        record = json.loads((self.final / "publication.json").read_text())
        record["storage_relative"] = "../outside"
        save(self.final / "publication.json", record)
        with self.assertRaises(ValueError):
            pub.storage_path(self.final)

    def test_cleanup_cannot_target_persistent_stage_or_scratch_root(self):
        scratch = self.root / "scratch"
        scratch.mkdir()
        with self.assertRaises(ValueError):
            stages.cleanup_owned_scratch(self.stage, scratch)
        with self.assertRaises(ValueError):
            stages.cleanup_owned_scratch(scratch, scratch)
        owned = scratch / "owned"
        owned.mkdir()
        (owned / "test").write_bytes(b"test")
        stages.cleanup_owned_scratch(owned, scratch)
        self.assertFalse(owned.exists())
        self.assertTrue(self.stage.exists())

    def test_capability_records_einval_without_requiring_rename(self):
        class Rename:
            def __call__(self, *args):
                ctypes.set_errno(errno.EINVAL)
                return -1
        class Lib:
            renameat2 = Rename()
        with patch.object(pub.ctypes, "CDLL", return_value=Lib()), contextlib.redirect_stdout(io.StringIO()):
            result = pub.capability(self.root)
        self.assertEqual(result["descriptor_primitives"], "passed")
        self.assertEqual(result["renameat2"]["errno"], errno.EINVAL)
        self.assertTrue(Path(result["probe_directory"]).is_dir())

if __name__ == "__main__":
    unittest.main()
