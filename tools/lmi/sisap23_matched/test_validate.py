"""Regression tests for panel query-row reference alignment."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import check_ids
from recovery_stages import N, validate_recovered_segment_marker


class QueryRowAlignmentTests(unittest.TestCase):
    def test_nonconsecutive_panel_ids_address_both_references_by_query_row(self):
        gold = [[1,2,3,4,5,6,7,8,9,10] for _ in range(40)]
        plain = [{"returned_ids_1_based": list(range(101,111))} for _ in range(40)]
        gold[17] = list(range(11,21))
        plain[17] = {"returned_ids_1_based": list(range(11,21))}
        gold[35] = list(range(31,41))
        plain[35] = {"returned_ids_1_based": list(range(41,51))}
        data = [
            {"query_row":37,"returned_ids_1_based":list(range(11,21)),"recall_at_10":1.0},
            {"query_row":55,"returned_ids_1_based":list(range(31,41)),"recall_at_10":1.0},
        ]
        official, reference = check_ids(data, "synthetic panel", gold, plain)
        self.assertEqual(official, [1.0, 1.0])
        self.assertEqual(reference, [1.0, 0.0])

    def test_recovered_two_segment_index_marker_matches_complete_topology(self):
        half = N // 2
        index = {"complete":True,"segment_count":2,
            "segments":[
                {"points":half,"segment_relative_path":"0/segments/a"},
                {"points":N-half,"segment_relative_path":"0/segments/b"}],
            "segment_relative_paths":["0/segments/a","0/segments/b"]}
        marker = {"segment_count":2,"segment_relative_paths":[
            "0/segments/a","0/segments/b"],"index_kind":"lmi",
            "evaluation_storage":"isolated_pbs_scratch_local_ssd",
            "cache_advice":"POSIX_FADV_DONTNEED_before_each_parameter",
            "warmup_protocol":"official_queries_0_through_19_logged_per_parameter"}
        validate_recovered_segment_marker(index, marker, "lmi")
        marker["segment_count"] = 1
        with self.assertRaisesRegex(ValueError, "topology differs"):
            validate_recovered_segment_marker(index, marker, "lmi")

    def test_recovered_segment_marker_rejects_wrong_index_kind(self):
        index = {"complete":True,"segment_count":1,
            "segments":[{"points":N,"segment_relative_path":"0/segments/a"}],
            "segment_relative_paths":["0/segments/a"]}
        marker = {"segment_count":1,"segment_relative_paths":["0/segments/a"],
            "index_kind":"plain","evaluation_storage":"isolated_pbs_scratch_local_ssd",
            "cache_advice":"POSIX_FADV_DONTNEED_before_each_parameter",
            "warmup_protocol":"official_queries_0_through_19_logged_per_parameter"}
        with self.assertRaisesRegex(ValueError, "topology differs"):
            validate_recovered_segment_marker(index, marker, "hnsw-a")


if __name__ == "__main__":
    unittest.main()
