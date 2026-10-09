"""Compare candidate-budget probing with the fixed-P frontier at matched work."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap10m/g2_119")
ANCHORS = [4, 13, 32, 64, 128]
N = 10_120_191
SEED = 20261009
EXPECTED = [
    "09c88623def82515b50ddf4f1c85d82a1188e5a29bafbc4fb00ca729dbf8d81f",
    "6789b1aace656c3c475675041433ccdb97bd13190d15edc8ff2b7c0bf8708a94",
    "adcf9f104ef60899e9b922e0bbb155061af3a61532fdcf11705ed55340a7eade",
]


def stats(a: np.ndarray) -> dict:
    return {
        "mean": float(a.mean()),
        "p5": float(np.quantile(a, 0.05)),
        "p50": float(np.quantile(a, 0.50)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    result = []
    for p in ANCHORS:
        budget_dir = ROOT / "candidate_budget" / f"target_p{p:04}"
        fixed_dir = ROOT / "fixed" / f"p{p:04}"
        assert json.loads((budget_dir / "hashes_before.json").read_text()) == EXPECTED
        assert json.loads((budget_dir / "hashes_after.json").read_text()) == EXPECTED
        budget = [json.loads(line) for line in (budget_dir / "rows.jsonl").open()]
        fixed = [json.loads(line) for line in (fixed_dir / "rows.jsonl").open()]
        assert len(budget) == len(fixed) == 9_980
        assert [row["query_row"] for row in budget] == [row["query_row"] for row in fixed]
        br = np.array([row["recall_at_10"] for row in budget])
        bc = np.array([row["candidate_count"] for row in budget])
        bp = np.array([row["probes_used"] for row in budget])
        bg = np.array([row["candidate_gold_coverage"] for row in budget])
        fr = np.array([row["recall_at_10"] for row in fixed])
        fc = np.array([row["candidate_count"] for row in fixed])
        fg = np.array([row["candidate_gold_coverage"] for row in fixed])
        delta_r = br - fr
        delta_c = bc - fc
        boot_r = np.empty(2_000)
        boot_c = np.empty(2_000)
        for i in range(2_000):
            sample = rng.integers(0, len(delta_r), len(delta_r))
            boot_r[i] = delta_r[sample].mean()
            boot_c[i] = delta_c[sample].mean()
        summary = json.loads((budget_dir / "summary.json").read_text())
        result.append(
            {
                "target_fixed_p": p,
                "budget_threshold": summary["budget_threshold"],
                "fixed_recall": float(fr.mean()),
                "budget_recall": float(br.mean()),
                "paired_recall_delta": float(delta_r.mean()),
                "paired_recall_delta_ci95": [float(v) for v in np.quantile(boot_r, [0.025, 0.975])],
                "fixed_candidates": stats(fc),
                "budget_candidates": stats(bc),
                "candidate_fraction_fixed": float(fc.mean() / N),
                "candidate_fraction_budget": float(bc.mean() / N),
                "paired_candidate_delta": float(delta_c.mean()),
                "paired_candidate_delta_ci95": [float(v) for v in np.quantile(boot_c, [0.025, 0.975])],
                "fixed_coverage": float(fg.mean()),
                "budget_coverage": float(bg.mean()),
                "probes_used": stats(bp),
                "query_count": len(budget),
            }
        )
    output = {"seed": SEED, "results": result}
    (ROOT / "candidate_budget_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    fixed = json.loads((ROOT / "fixed_analysis.json").read_text())["primary_table"]
    fig, ax = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
    ax.plot(
        [row["candidate_fraction"] * 100 for row in fixed],
        [row["recall"]["mean"] for row in fixed],
        marker="o",
        label="Fixed P",
    )
    ax.scatter(
        [row["candidate_fraction_budget"] * 100 for row in result],
        [row["budget_recall"] for row in result],
        marker="D",
        label="Candidate budget",
    )
    for row in result:
        ax.annotate(
            f"budget near P={row['target_fixed_p']}",
            (row["candidate_fraction_budget"] * 100, row["budget_recall"]),
            fontsize=7,
        )
    ax.set_xlabel("Mean candidates (% of corpus)")
    ax.set_ylabel("Returned Recall@10")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(ROOT / "figures" / "10_fixed_vs_candidate_budget.png", dpi=180)
    plt.close(fig)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
