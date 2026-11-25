"""Analysis for the DLI-best integration pilot (INTEGRATION_PILOT.md).

    analyze_integration.py --run DIR --reference E1_DIR --scenario S --sets a,b --out FILE

Criteria: (1) ledger/completeness from batches.jsonl; (2) native == control: identical
scored_total in every row, recall differences listed for the offline tie check; (3) known
answer at the full-search budget per (batch, panel); (4) mechanisms: vacuum publications and
dead fraction (deletes), consolidation shape of the final levels. Descriptive: W@0.90 of
DLI-best (`proportional_native`) over Phase 3 E1 (`mdn_code`), paired by query.
"""
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("apr", HERE.parent / "policy_replay" / "analyze_policy_replay.py")
apr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apr)

SEED = 20261126
FULL = 2_000_000
KNOWN_ANSWER = 0.995
RHO = 0.25
MAX_DEAD = 0.20


def native_vs_control(run_dir):
    rows = 0
    work_mismatch, recall_diff = [], []
    for f in sorted(Path(run_dir).glob("batch_*/rows.jsonl")):
        b = int(f.parent.name.split("_")[1])
        by = defaultdict(dict)
        for line in open(f):
            r = json.loads(line)
            if r["policy"] in ("proportional", "proportional_native"):
                by[r["policy"]][(r["set"], r["query"], r["budget"])] = r
        for key, c in by["proportional"].items():
            n = by["proportional_native"].get(key)
            rows += 1
            if n is None or n["scored_total"] != c["scored_total"]:
                work_mismatch.append([b, *key])
            elif n["recall30"] != c["recall30"]:
                recall_diff.append([b, *key, c["recall30"], n["recall30"]])
        if len(by["proportional_native"]) != len(by["proportional"]):
            work_mismatch.append([b, "row-count", len(by["proportional"]), len(by["proportional_native"])])
    return {"rows": rows, "work_mismatches": work_mismatch, "recall_differences": recall_diff,
            "recall_difference_queries": sorted({(d[0], d[1], d[2]) for d in recall_diff})}


def known_answer(run_dir, sets):
    out = {}
    for s in sets:
        data = apr.p3.load_run(run_dir, "proportional_native", s)
        per = {t: float(v[FULL][0].mean()) for t, v in data.items() if FULL in v}
        out[s] = {"min": min(per.values()) if per else None, "per_batch": per,
                  "pass_raw": bool(per) and min(per.values()) >= KNOWN_ANSWER}
    return out


def mechanisms(run_dir, scenario):
    batches = [json.loads(l) for l in open(Path(run_dir) / "batches.jsonl")]
    worst_dead, vacuums = 0.0, 0
    # A learned segment is any DLI publication destination (the buffer never is). A vacuum
    # rewrites one learned segment alone: a single source that an earlier publication produced.
    learned = set()
    for x in batches:
        for p in x["publications"]:
            if len(p["source_uuids"]) == 1 and p["source_uuids"][0] in learned:
                vacuums += 1
            learned.add(p["destination_uuid"])
        for s in x["segments"]:
            if s["appendable"]:
                continue
            total = s["live"] + s["deleted"]
            if total:
                worst_dead = max(worst_dead, s["deleted"] / total)
    last = sorted(x["levels"], key=lambda v: v["level"]) if (x := batches[-1]) else []
    deepest = last[-1]["live"] if last else 0
    above = [v["live"] for v in last[:-1]]
    return {"batches": len(batches), "single_source_rewrites_of_learned_levels": vacuums,
            "worst_dead_fraction_after_settle": worst_dead,
            "dead_gate": worst_dead <= MAX_DEAD + 1e-12,
            "final_levels": [(v["level"], v["live"], v["buckets"]) for v in last],
            "consolidation_shape": all(a <= RHO * deepest for a in above) if last else None,
            "scenario": scenario}


def versus_e1(run_dir, reference, sets, resamples):
    out = {}
    for s in sets:
        a = apr.p3.load_run(run_dir, "proportional_native", s)
        b = apr.p3.load_run(reference, "mdn_code", s)
        if not a or not b:
            out[s] = {"status": "missing"}
            continue
        r = apr.work_ratio(a, b, resamples, np.random.default_rng(SEED))
        out[s] = {k: r[k] for k in ("R_prop_over_mdn", "ci95", "attained_both", "status") if k in r}
        if "R_prop_over_mdn" in out[s]:
            out[s]["R_dlibest_over_e1"] = out[s].pop("R_prop_over_mdn")
            out[s]["stop_for_discussion"] = out[s]["R_dlibest_over_e1"] > 1.05
    return out


def analyze(run_dir, reference, scenario, sets, resamples=apr.p3.RESAMPLES):
    return {"scenario": scenario, "sets": sets,
            "native_vs_control": native_vs_control(run_dir),
            "known_answer": known_answer(run_dir, sets),
            "mechanisms": mechanisms(run_dir, scenario),
            "versus_e1_descriptive": versus_e1(run_dir, reference, sets, resamples) if reference else None}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--reference", type=Path)
    a.add_argument("--scenario", required=True)
    a.add_argument("--sets", required=True)
    a.add_argument("--resamples", type=int, default=apr.p3.RESAMPLES)
    a.add_argument("--out", type=Path)
    args = a.parse_args()
    result = analyze(args.run, args.reference, args.scenario, args.sets.split(","), args.resamples)
    text = json.dumps(result, indent=1, default=float)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
