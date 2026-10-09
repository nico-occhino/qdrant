"""Analyze controlled native latency repetitions and join them to fixed-P effectiveness."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap10m/g2_119")
ALL_P = [1, 2, 4, 8, 10, 13, 16, 30, 32, 40, 64, 100, 128, 160, 256]
SEED = 20261009
EXPECTED = [
    "09c88623def82515b50ddf4f1c85d82a1188e5a29bafbc4fb00ca729dbf8d81f",
    "6789b1aace656c3c475675041433ccdb97bd13190d15edc8ff2b7c0bf8708a94",
    "adcf9f104ef60899e9b922e0bbb155061af3a61532fdcf11705ed55340a7eade",
]


def stats(values):
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def save_svg_clean(fig, path: Path) -> None:
    fig.savefig(path)
    path.write_text(chr(10).join(line.rstrip() for line in path.read_text().splitlines()) + chr(10))


def main() -> None:
    plan = json.loads((ROOT / "latency_panel_design.json").read_text())
    assert plan["seed"] == SEED
    query_rows = plan["query_rows"]
    assert len(query_rows) == 1_000 and len(set(query_rows)) == 1_000
    warm_suffix_start = plan["warm_suffix_start"]
    assert warm_suffix_start == 200
    fixed = json.loads((ROOT / "fixed_analysis.json").read_text())
    P = [p for p in ALL_P if all((ROOT / "latency_panel" / f"rep{rep}_p{p:04}" / "summary.json").exists() for rep in (1, 2, 3))]
    assert P and all(p in fixed["probe_grid"] for p in P)
    rng = np.random.default_rng(SEED)
    per_p = {}
    all_c = []
    all_l = []
    for p in P:
        fixed_raw = [json.loads(line) for line in (ROOT / "fixed" / f"p{p:04}" / "rows.jsonl").open()]
        fixed_candidates = {row["query_row"]: row["candidate_count"] for row in fixed_raw}
        arrays = []
        candidate_arrays = []
        repetitions = []
        for rep in (1, 2, 3):
            folder = ROOT / "latency_panel" / f"rep{rep}_p{p:04}"
            assert json.loads((folder / "hashes_before.json").read_text()) == EXPECTED
            assert json.loads((folder / "hashes_after.json").read_text()) == EXPECTED
            raw = [json.loads(line) for line in (folder / "rows.jsonl").open()]
            assert len(raw) == 1_000
            assert [row["query_row"] for row in raw] == query_rows
            assert all(row["nprobe"] == p and row["repetition"] == rep for row in raw)
            assert all(row["candidate_count"] == fixed_candidates[row["query_row"]] for row in raw)
            latency = np.array([row["native_search_ms"] for row in raw])
            candidates = np.array([row["candidate_count"] for row in raw])
            arrays.append(latency[warm_suffix_start:])
            candidate_arrays.append(candidates)
            repetitions.append({
                "full_panel": stats(latency),
                "warm_suffix": stats(latency[warm_suffix_start:]),
            })
            all_c.extend(candidates[warm_suffix_start:])
            all_l.extend(latency[warm_suffix_start:])
        per_query_median = np.median(np.stack(arrays), axis=0)
        per_p[str(p)] = {
            "repetitions": repetitions,
            "median_of_run_means_ms": float(np.median([run["warm_suffix"]["mean"] for run in repetitions])),
            "median_of_run_p50_ms": float(np.median([run["warm_suffix"]["p50"] for run in repetitions])),
            "median_of_run_p95_ms": float(np.median([run["warm_suffix"]["p95"] for run in repetitions])),
            "median_of_run_p99_ms": float(np.median([run["warm_suffix"]["p99"] for run in repetitions])),
            "run_mean_range_ms": [
                float(np.min([run["warm_suffix"]["mean"] for run in repetitions])),
                float(np.max([run["warm_suffix"]["mean"] for run in repetitions])),
            ],
            "per_query_median_ms": per_query_median.tolist(),
            "mean_candidates_panel": float(candidate_arrays[0].mean()),
            "candidate_p95_panel": float(np.quantile(candidate_arrays[0], 0.95)),
            "mean_candidates_warm_suffix": float(candidate_arrays[0][warm_suffix_start:].mean()),
            "candidate_p95_warm_suffix": float(np.quantile(candidate_arrays[0][warm_suffix_start:], 0.95)),
        }

    all_c = np.asarray(all_c, dtype=float)
    all_l = np.asarray(all_l, dtype=float)
    slope, intercept = np.polyfit(all_c, all_l, 1)
    predicted = intercept + slope * all_c
    r2 = 1 - float(((all_l - predicted) ** 2).sum() / ((all_l - all_l.mean()) ** 2).sum())
    residual = all_l - predicted
    model = {
        "intercept_ms": float(intercept),
        "slope_ms_per_candidate": float(slope),
        "slope_ms_per_10000_candidates": float(slope * 10_000),
        "r_squared": r2,
        "residual_p50_ms": float(np.quantile(residual, 0.50)),
        "residual_p95_ms": float(np.quantile(residual, 0.95)),
        "residual_p99_ms": float(np.quantile(residual, 0.99)),
        "observations": len(all_c),
    }

    model["point_mean_residual_ms"] = {
        str(p): float(
            per_p[str(p)]["median_of_run_means_ms"]
            - (intercept + slope * per_p[str(p)]["mean_candidates_warm_suffix"])
        )
        for p in P
    }

    paired_latency = {}
    for a, b in [(4, 8), (8, 16), (16, 30), (30, 40), (40, 100), (100, 160), (160, 256)]:
        if a not in P or b not in P:
            continue
        delta = np.array(per_p[str(b)]["per_query_median_ms"]) - np.array(
            per_p[str(a)]["per_query_median_ms"]
        )
        boot = np.empty(2_000)
        for i in range(len(boot)):
            boot[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
        paired_latency[f"{a}_to_{b}"] = {
            "mean_delta_ms": float(delta.mean()),
            "ci95_ms": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        }

    points = [row for row in fixed["primary_table"] if row["p"] in P]
    joined = [
        {
            "p": row["p"],
            "recall": row["recall"]["mean"],
            "candidate_fraction": row["candidate_fraction"],
            "warm_median_mean_ms": per_p[str(row["p"])]["median_of_run_means_ms"],
            "warm_median_p50_ms": per_p[str(row["p"])]["median_of_run_p50_ms"],
            "warm_median_p95_ms": per_p[str(row["p"])]["median_of_run_p95_ms"],
        }
        for row in points
    ]
    pareto = []
    for point in joined:
        if not any(
            other["recall"] >= point["recall"]
            and other["warm_median_mean_ms"] <= point["warm_median_mean_ms"]
            and (
                other["recall"] > point["recall"]
                or other["warm_median_mean_ms"] < point["warm_median_mean_ms"]
            )
            for other in joined
        ):
            pareto.append(point["p"])
    x = np.array([point["warm_median_mean_ms"] for point in joined])
    y = np.array([point["recall"] for point in joined])
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    knee = joined[int(np.argmax(yn - xn))]["p"]

    marginal_latency = []
    for previous, current in zip(joined, joined[1:]):
        delta_recall = current["recall"] - previous["recall"]
        delta_ms = current["warm_median_mean_ms"] - previous["warm_median_mean_ms"]
        marginal_latency.append({
            "from_p": previous["p"],
            "to_p": current["p"],
            "delta_recall": delta_recall,
            "delta_warm_mean_ms": delta_ms,
            "recall_points_per_added_ms": None if delta_ms <= 0 else delta_recall * 100 / delta_ms,
        })

    result = {
        "design_seed": SEED,
        "panel_query_count": len(query_rows),
        "completed_probe_grid": P,
        "warm_suffix_start": warm_suffix_start,
        "warm_suffix_query_count": len(query_rows) - warm_suffix_start,
        "probe_orders_by_repetition": plan["probe_orders_by_repetition"],
        "per_p": per_p,
        "paired_latency_bootstrap": paired_latency,
        "linear_candidate_latency_model": model,
        "joined_frontier": joined,
        "marginal_latency": marginal_latency,
        "pareto_recall_vs_warm_native_mean_latency": pareto,
        "knee_recall_vs_warm_native_mean_latency_chord": knee,
    }
    (ROOT / "latency_panel_analysis.json").write_text(json.dumps(result, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
    ax.plot(x, y, marker="o", markersize=4)
    for point in joined:
        ax.annotate(str(point["p"]), (point["warm_median_mean_ms"], point["recall"]), fontsize=7)
    ax.set_xlabel("Controlled warm native mean latency (ms)")
    ax.set_ylabel("Returned Recall@10")
    ax.grid(alpha=0.25)
    fig.savefig(ROOT / "figures" / "04_recall_warm_native_latency.png", dpi=300)
    save_svg_clean(fig, ROOT / "figures" / "04_recall_warm_native_latency.svg")
    plt.close(fig)
    print(json.dumps({"joined": joined, "model": model, "pareto": pareto, "knee": knee}, indent=2))


if __name__ == "__main__":
    main()
