import unittest

from validate_plain_ssd_panel import validate_records


def record(row, repetition, ids, recall=1.0):
    return {
        "query_row":row,"repetition":repetition,"parameter":"plain-exact",
        "native_search_ms":2.0,"recall_at_10":recall,"candidate_count":100,
        "segment_count":2,
        "search_interface":"direct_segment_indices_sequential_merge",
        "fallback_observation":"not_applicable_direct_segment_index",
        "returned_ids_1_based":ids,
    }

def warmup_records(ids):
    return [{
        "warmup_order":i,"query_row":i,"parameter":"plain-exact",
        "native_search_ms":1.0,"candidate_count":100,"segment_count":2,
        "search_interface":"direct_segment_indices_sequential_merge",
        "fallback_observation":"not_applicable_direct_segment_index",
        "returned_ids_1_based":ids,
    } for i in range(20)]


class PlainSsdPanelTests(unittest.TestCase):
    def test_three_repeated_exact_panels_validate_against_plain_and_ground_truth(self):
        ids=list(range(1,11))
        warmup=warmup_records(ids)
        reps=[[record(23,rep,ids)] for rep in range(1,4)]
        summary=validate_records(warmup,reps,[23],{23:ids},{23:ids},2,points=100)
        self.assertEqual(summary["measured_query_count"],3)
        self.assertEqual(summary["mean_recall_at_10"],1.0)

    def test_panel_rejects_missing_exact_neighbor_even_when_recall_field_lies(self):
        ids=list(range(1,11)); warmup=warmup_records(ids)
        reps=[[record(23,rep,ids[:-1]+[11])] for rep in range(1,4)]
        with self.assertRaisesRegex(ValueError,"ground truth"):
            validate_records(warmup,reps,[23],{23:ids},{23:ids},2,points=100)

    def test_panel_rejects_incorrect_exact_neighbor_order(self):
        ids=list(range(1,11)); warmup=warmup_records(ids)
        reordered=ids.copy(); reordered[0],reordered[1]=reordered[1],reordered[0]
        reps=[[record(23,rep,reordered)] for rep in range(1,4)]
        with self.assertRaisesRegex(ValueError,"ranking"):
            validate_records(warmup,reps,[23],{23:ids},{23:ids},2,points=100)

    def test_panel_rejects_unobserved_collection_fallback_claim(self):
        ids=list(range(1,11)); warmup=warmup_records(ids)
        reps=[[record(23,rep,ids)] for rep in range(1,4)]
        reps[1][0]["fallback_observation"]="plain_fallback_segments=0"
        with self.assertRaisesRegex(ValueError,"incomplete or invalid"):
            validate_records(warmup,reps,[23],{23:ids},{23:ids},2,points=100)


if __name__ == "__main__":
    unittest.main()
