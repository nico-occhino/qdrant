"""Regression tests for panel query-row reference alignment."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import check_ids


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


if __name__ == "__main__":
    unittest.main()
