"""Known-answer test for analyze_phase3.py (PHASE3_PREREGISTRATION.md §7 M9).

Builds synthetic run directories with a planted effect and checks the decisions:
* Part A: arm A has recall +3 pp on every query at every budget  -> Δ = 3.0, H6 supported;
  identical arms -> H7 equivalent; a +3 pp contrast against a 1 pp margin -> not equivalent.
* Part B: arm A scores 0.6× the work of B at the same recall curve -> ratio 0.6, supported.
Run: python3 -I test_analyze_phase3.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUDGETS = [1000, 2000, 3500, 5000, 7000, 10000]


def write_run(root, name, set_name, recall_shift=0.0, work_scale=1.0, batches=4, queries=400, seed=0):
    rng = np.random.default_rng(seed)
    d = root / name
    d.mkdir()
    # Curves cross 0.90 between budgets; no clipping at the primary budget (index 3).
    base = rng.uniform(0.5, 0.75, size=queries)
    with open(d / "batches.jsonl", "w") as log:
        for b in range(batches):
            bd = d / f"batch_{b:02d}"
            bd.mkdir()
            with open(bd / "rows.jsonl", "w") as f:
                for i, budget in enumerate(BUDGETS):
                    for q in range(queries):
                        r = min(1.0, base[q] + 0.06 * i + recall_shift)
                        f.write(json.dumps({"set": set_name, "query": q, "policy": "mdn_code", "budget": budget,
                                            "recall30": r * 30, "scored_total": budget * work_scale,
                                            "dead_offsets": 0, "unique_offsets": budget}) + "\n")
            log.write(json.dumps({"batch": b, "publications": [], "levels": [], "ingest_seconds": 1.0}) + "\n")
    return d


def run(args):
    out = subprocess.run([sys.executable, "-I", str(HERE / "analyze_phase3.py"), *args],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for scenario in ["constant_r", "constant_o", "insert_only"]:
            (root / f"plan_{scenario}").mkdir()
            (root / f"plan_{scenario}" / "plan.json").write_text(json.dumps({"rows": 1_000_000}))
        e1 = write_run(root, "cr_e1", "test")
        e1g = write_run(root, "cr_e1g", "test", recall_shift=0.03)
        o1 = write_run(root, "co_e1", "test")
        o1g = write_run(root, "co_e1g", "test")
        b1 = write_run(root, "io_e1", "partb")
        b1c = write_run(root, "io_e1c", "partb", work_scale=0.6)
        res = run(["--run", f"a=constant_r:E1:{e1}", "--run", f"b=constant_r:E1G:{e1g}",
                   "--run", f"c=constant_o:E1:{o1}", "--run", f"d=constant_o:E1G:{o1g}",
                   "--run", f"e=insert_only:E1:{b1}", "--run", f"f=insert_only:E1C:{b1c}",
                   "--plans", f"constant_r={root / 'plan_constant_r'}",
                   "--plans", f"constant_o={root / 'plan_constant_o'}",
                   "--plans", f"insert_only={root / 'plan_insert_only'}"])["hypotheses"]
        h6, h7, h9 = res["H6"], res["H7"], res["H9"]
        assert h6["budget"] == 5000, h6
        assert abs(h6["delta_pp"] - 3.0) < 0.05, h6
        assert h6["outcome"] == "supported", h6
        assert h7["delta_pp"] == 0.0 and h7["outcome"] == "supported (equivalent)", h7
        assert abs(h9["point"] - 0.6) < 1e-9 and h9["outcome"] == "supported", h9
        assert res["H5"] == {"status": "missing runs"}
        # Same planted +3 pp contrast judged as equivalence against ±1 pp must fail.
        res2 = run(["--run", f"c=constant_o:E1:{e1}", "--run", f"d=constant_o:E1G:{e1g}",
                    "--plans", f"constant_o={root / 'plan_constant_o'}"])["hypotheses"]
        assert res2["H7"]["outcome"] == "not supported", res2["H7"]
        print("analyze_phase3 known-answer test: OK",
              {"H6": round(h6["delta_pp"], 3), "H7": h7["outcome"], "H9": h9["point"]})


if __name__ == "__main__":
    main()
