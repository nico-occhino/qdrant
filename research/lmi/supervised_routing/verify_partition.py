#!/usr/bin/env python3
"""Exit non-zero unless a rebuilt IVF partition reproduces a reference run query by query.

    verify_partition.py NEW.jsonl REFERENCE.jsonl [--offset Q] [--max-mismatch-rate R]

NEW is the harness `evaluate` output for arm A0 (rep 1 rows carry ids); REFERENCE is an accepted
`ann_matched_v120` queries.jsonl (rows with "setting", "q", "candidates", "ids"). Query q of NEW is
compared with query q + offset of REFERENCE, at every setting both contain. Candidate counts and the
returned ids (the shorter prefix of the two top lists) must be identical, except that a mismatch
rate up to --max-mismatch-rate (default 0) is tolerated when the reference predates a numerically
relevant router change (see PREREGISTRATION.md, Amendment 1). Mismatch counts are always reported.
"""
import json
import sys


def main():
    new_path, ref_path = sys.argv[1], sys.argv[2]
    offset = int(sys.argv[sys.argv.index("--offset") + 1]) if "--offset" in sys.argv else 0
    tolerance = (float(sys.argv[sys.argv.index("--max-mismatch-rate") + 1])
                 if "--max-mismatch-rate" in sys.argv else 0.0)
    new = {}
    for line in open(new_path):
        r = json.loads(line)
        if r["arm"] == "A0" and r["rep"] == 1:
            new[(r["nprobe"], r["q"] + offset)] = (r["candidates"], r["ids"])
    checked = mismatched_cand = mismatched_ids = 0
    for line in open(ref_path):
        r = json.loads(line)
        if "ids" not in r:
            continue
        key = (r["setting"], r["q"])
        if key not in new:
            continue
        cand, ids = new[key]
        m = min(len(ids), len(r["ids"]))
        checked += 1
        mismatched_cand += int(cand != r["candidates"])
        mismatched_ids += int(ids[:m] != r["ids"][:m])
    result = {"checked": checked, "candidate_mismatches": mismatched_cand, "id_mismatches": mismatched_ids,
              "settings": sorted({k[0] for k in new}), "max_mismatch_rate": tolerance}
    print(json.dumps(result))
    ok = (checked > 0 and mismatched_cand <= tolerance * checked and mismatched_ids <= tolerance * checked)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
