"""Reconstruct fixed-P G2 statistics and figures from completed raw JSONL outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/nicoo/work/qdrant-upstream-sync/work/upstream_sync/sisap10m/g2_119")
PRIMARY_P = [1, 2, 4, 8, 10, 13, 16, 30, 32, 40, 64, 100, 128, 160]
P = PRIMARY_P + ([256] if (ROOT / "fixed/p0256/summary.json").exists() else [])
N = 10_120_191
B = 3_162
ROWS = 9_980
SEED = 20261009
EXPECTED = [
    "09c88623def82515b50ddf4f1c85d82a1188e5a29bafbc4fb00ca729dbf8d81f",
    "6789b1aace656c3c475675041433ccdb97bd13190d15edc8ff2b7c0bf8708a94",
    "adcf9f104ef60899e9b922e0bbb155061af3a61532fdcf11705ed55340a7eade",
]


def values(a: np.ndarray) -> dict:
    return {
        "mean": float(a.mean()),
        "min": float(a.min()),
        "p5": float(np.quantile(a, 0.05)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }


def bootstrap_delta(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    delta = b - a
    means = np.empty(2_000)
    for i in range(len(means)):
        means[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {
        "mean": float(delta.mean()),
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
    }


def pareto(rows: list[dict], cost: str) -> list[int]:
    keep = []
    for current in rows:
        dominated = any(
            other["recall"]["mean"] >= current["recall"]["mean"]
            and other[cost] <= current[cost]
            and (
                other["recall"]["mean"] > current["recall"]["mean"]
                or other[cost] < current[cost]
            )
            for other in rows
        )
        if not dominated:
            keep.append(current["p"])
    return keep


def chord_knee(rows: list[dict], cost: str) -> dict:
    x = np.array([row[cost] for row in rows], dtype=float)
    y = np.array([row["recall"]["mean"] for row in rows], dtype=float)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    distance = y - x  # Distance to the endpoint chord up to a constant factor.
    idx = int(np.argmax(distance))
    return {"p": rows[idx]["p"], "normalized_distance": float(distance[idx])}


def figure(x, ys, labels, xlabel, ylabel, name, *, logx=False):
    fig, ax = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
    for y, label in zip(ys, labels):
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=label)
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(labels) > 1:
        ax.legend(frameon=False)
    fig.savefig(ROOT / "figures" / name, dpi=300)
    save_svg_clean(fig, ROOT / "figures" / name.replace(".png", ".svg"))
    plt.close(fig)


def save_svg_clean(fig, path: Path) -> None:
    fig.savefig(path)
    path.write_text(chr(10).join(line.rstrip() for line in path.read_text().splitlines()) + chr(10))


def main() -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    per_p = {}
    for p in P:
        directory = ROOT / "fixed" / f"p{p:04}"
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["state_hashes_unchanged"]
        assert json.loads((directory / "hashes_before.json").read_text()) == EXPECTED
        assert json.loads((directory / "hashes_after.json").read_text()) == EXPECTED
        raw = [json.loads(line) for line in (directory / "rows.jsonl").open()]
        assert len(raw) == ROWS
        assert [row["query_row"] for row in raw] == list(range(20, 10_000))
        assert all(row["nprobe"] == p for row in raw)
        log_path = ROOT / ("fixed_p0256.log" if p == 256 else f"p{p:04}.log")
        log = log_path.read_text()
        assert "test result: ok. 1 passed; 0 failed" in log
        assert f"G2 P={p} completed:" in log
        for field, raw_field in [
            ("mean_recall", "recall_at_10"),
            ("mean_candidates", "candidate_count"),
            ("mean_candidate_gold_coverage", "candidate_gold_coverage"),
            ("mean_oracle_coverage", "oracle_coverage"),
        ]:
            assert np.isclose(summary[field], np.mean([row[raw_field] for row in raw]), atol=1e-9)
        per_p[p] = raw

    recall = np.array([[row["recall_at_10"] for row in per_p[p]] for p in P])
    candidate = np.array([[row["candidate_count"] for row in per_p[p]] for p in P])
    coverage = np.array(
        [[row["candidate_gold_coverage"] for row in per_p[p]] for p in P]
    )
    oracle = np.array([[row["oracle_coverage"] for row in per_p[p]] for p in P])
    latency = np.array([[row["latency_ms"] for row in per_p[p]] for p in P])
    assert np.all(candidate[1:] >= candidate[:-1]), "candidate prefix nesting failed"
    assert np.all(coverage[1:] >= coverage[:-1]), "coverage prefix nesting failed"
    assert all(
        row["gold_bucket_ranks_1_based"] == per_p[4][i]["gold_bucket_ranks_1_based"]
        for p in P
        for i, row in enumerate(per_p[p])
    ), "router ordering changed across probe conditions"
    assert np.all(oracle >= coverage - 1e-12)
    assert np.all(coverage >= recall - 1e-12)
    assert np.all(oracle[1:] >= oracle[:-1] - 1e-12)

    main_rows = []
    for i, p in enumerate(P):
        rows = per_p[p]
        r, c, g, o, l = recall[i], candidate[i], coverage[i], oracle[i], latency[i]
        main_rows.append(
            {
                "p": p,
                "probe_fraction": p / B,
                "recall": values(r),
                "recall_histogram_0_to_10": [int(np.sum(r == v / 10)) for v in range(11)],
                "candidate": values(c),
                "candidate_fraction": float(c.mean() / N),
                "candidate_gold_coverage": values(g),
                "oracle_coverage": values(o),
                "partition_loss": float(1 - o.mean()),
                "router_loss": float(o.mean() - g.mean()),
                "output_loss": float(g.mean() - r.mean()),
                "native_full_sequence_latency_ms": values(l),
                "native_suffix_rows_1000_to_9999_latency_ms": values(l[980:]),
                "elapsed_seconds": float(json.loads((ROOT / "fixed" / f"p{p:04}" / "summary.json").read_text())["elapsed_seconds"]),
                "per_query_recall_decreases_from_previous": int(np.sum(r < recall[i - 1])) if i else 0,
                "per_query_recall_increases_from_previous": int(np.sum(r > recall[i - 1])) if i else 0,
            }
        )

    rng = np.random.default_rng(SEED)
    paired = {}
    for a, b in [(4, 8), (8, 16), (16, 30), (16, 32), (30, 40), (32, 64), (40, 100), (64, 128), (100, 160), (160, 256)]:
        if b not in P:
            continue
        ia, ib = P.index(a), P.index(b)
        paired[f"{a}_to_{b}"] = {
            "recall": bootstrap_delta(recall[ia], recall[ib], rng),
            "candidates": bootstrap_delta(candidate[ia], candidate[ib], rng),
        }

    targets = {}
    for t in [0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99]:
        found = next((row for row in main_rows if row["recall"]["mean"] >= t), None)
        targets[str(t)] = (
            None
            if found is None
            else {
                "p": found["p"],
                "probe_fraction": found["probe_fraction"],
                "mean_candidates": found["candidate"]["mean"],
                "candidate_fraction": found["candidate_fraction"],
            }
        )

    marginal = []
    for prev, curr in zip(main_rows, main_rows[1:]):
        dr = curr["recall"]["mean"] - prev["recall"]["mean"]
        dc = curr["candidate"]["mean"] - prev["candidate"]["mean"]
        marginal.append(
            {
                "from_p": prev["p"],
                "to_p": curr["p"],
                "delta_recall": dr,
                "delta_candidates": dc,
                "recall_points_per_10000_candidates": dr * 100 * 10_000 / dc,
            }
        )

    baseline = per_p[4]
    ranks = np.array([row["gold_bucket_ranks_1_based"] for row in baseline])
    depth = {}
    for k in [1, 5, 8, 9, 10]:
        needed = np.sort(ranks, axis=1)[:, k - 1]
        depth[str(k)] = values(needed)
    fragments = np.array([row["distinct_gold_buckets"] for row in baseline])
    depth10 = np.max(ranks, axis=1)
    groups = {
        "p4_recall_le_0_3": recall[P.index(4)] <= 0.3,
        "p4_recall_0_4_to_0_7": (recall[P.index(4)] >= 0.4)
        & (recall[P.index(4)] <= 0.7),
        "p4_recall_ge_0_8": recall[P.index(4)] >= 0.8,
    }
    difficulty = {
        name: {
            "query_count": int(mask.sum()),
            "mean_recall_by_p": {
                str(p): float(recall[i, mask].mean()) for i, p in enumerate(P)
            },
        }
        for name, mask in groups.items()
    }
    fragmentation = {
        "distinct_buckets": values(fragments),
        "pearson_vs_p4_recall": float(pearsonr(fragments, recall[P.index(4)]).statistic),
        "spearman_vs_p4_recall": float(spearmanr(fragments, recall[P.index(4)]).statistic),
        "pearson_vs_depth10": float(pearsonr(fragments, depth10).statistic),
        "spearman_vs_depth10": float(spearmanr(fragments, depth10).statistic),
        "pearson_depth10_vs_p4_recall": float(pearsonr(depth10, recall[P.index(4)]).statistic),
        "spearman_depth10_vs_p4_recall": float(spearmanr(depth10, recall[P.index(4)]).statistic),
        "pearson_vs_p4_candidates": float(pearsonr(fragments, candidate[P.index(4)]).statistic),
        "spearman_vs_p4_candidates": float(spearmanr(fragments, candidate[P.index(4)]).statistic),
        "spearman_vs_recall_by_p": {
            str(p): float(spearmanr(fragments, recall[i]).statistic)
            for i, p in enumerate(P)
        },
        "pearson_vs_recall_by_p": {
            str(p): float(pearsonr(fragments, recall[i]).statistic)
            for i, p in enumerate(P)
        },
    }

    recall_decrease_details = []
    for i in range(1, len(P)):
        for old, new in zip(per_p[P[i - 1]], per_p[P[i]]):
            if new["recall_at_10"] < old["recall_at_10"]:
                recall_decrease_details.append({
                    "from_p": P[i - 1],
                    "to_p": P[i],
                    "query_row": new["query_row"],
                    "recall_delta": new["recall_at_10"] - old["recall_at_10"],
                    "coverage_before": old["candidate_gold_coverage"],
                    "coverage_after": new["candidate_gold_coverage"],
                })
    high_p_failure_modes = {}
    for p in [v for v in [160, 256] if v in P]:
        low = [row for row in per_p[p] if row["recall_at_10"] < 1.0]
        router_dominant = output_dominant = equal_loss = 0
        for row in low:
            router_loss = row["oracle_coverage"] - row["candidate_gold_coverage"]
            output_loss = row["candidate_gold_coverage"] - row["recall_at_10"]
            if router_loss > output_loss + 1e-12:
                router_dominant += 1
            elif output_loss > router_loss + 1e-12:
                output_dominant += 1
            else:
                equal_loss += 1
        high_p_failure_modes[str(p)] = {
            "imperfect_query_count": len(low),
            "router_dominant": router_dominant,
            "output_dominant": output_dominant,
            "equal_loss": equal_loss,
            "recall_zero_count": sum(row["recall_at_10"] == 0 for row in low),
            "recall_le_0_3_count": sum(row["recall_at_10"] <= 0.3 for row in low),
        }

    result = {
        "state_sha256": EXPECTED,
        "full_query_count": ROWS,
        "probe_grid": P,
        "primary_table": main_rows,
        "targets": targets,
        "marginal": marginal,
        "paired_bootstrap_seed": SEED,
        "paired_bootstrap_replicates": 2_000,
        "paired": paired,
        "pareto_recall_vs_candidate_fraction": pareto(main_rows, "candidate_fraction"),
        "knee_recall_vs_candidate_fraction_chord": chord_knee(main_rows, "candidate_fraction"),
        "gold_bucket_depth": depth,
        "fragmentation": fragmentation,
        "difficulty": difficulty,
        "candidate_gold_coverage_monotonic": True,
        "candidate_count_nested_monotonic": True,
        "primary_probe_grid": PRIMARY_P,
        "recall_decrease_details": recall_decrease_details,
        "high_p_failure_modes": high_p_failure_modes,
    }
    historical_path = Path("/home/nicoo/work/qdrant/work/phase_g/g2_frontier/analytic-frontier.json")
    if historical_path.exists():
        historical = json.loads(historical_path.read_text())
        old = {row["nprobe"]: row for row in historical["frontier"]}
        result["historical_phase_g_analytic_comparison"] = [
            {
                "p": p,
                "old_candidate_coverage": old[p]["router_gold_coverage"]["mean"],
                "new_candidate_coverage": main_rows[P.index(p)]["candidate_gold_coverage"]["mean"],
                "old_candidates": old[p]["candidate_count"]["mean"],
                "new_candidates": main_rows[P.index(p)]["candidate"]["mean"],
                "old_oracle": old[p]["oracle_recall"]["mean"],
                "new_oracle": main_rows[P.index(p)]["oracle_coverage"]["mean"],
            }
            for p in [1, 2, 4, 8, 16, 32]
        ]
    (ROOT / "fixed_analysis.json").write_text(json.dumps(result, indent=2) + "\n")

    x = np.array(P)
    means = lambda key: [row[key]["mean"] for row in main_rows]
    figure(x, [means("recall")], ["Returned Recall@10"], "nprobe", "Recall@10", "01_recall_nprobe.png", logx=True)
    figure(x / B * 100, [means("recall")], ["Returned Recall@10"], "Probed buckets (%)", "Recall@10", "02_recall_probe_fraction.png")
    figure([row["candidate_fraction"] * 100 for row in main_rows], [means("recall")], ["Returned Recall@10"], "Candidates (% of corpus)", "Recall@10", "03_recall_candidate_fraction.png")
    figure(x, [means("candidate")], ["Candidates"], "nprobe", "Mean candidates", "05_candidates_nprobe.png", logx=True)
    figure(x, [means("oracle_coverage"), means("candidate_gold_coverage"), means("recall")], ["Oracle", "Router candidate coverage", "Returned recall"], "nprobe", "Top-10 fraction", "06_loss_decomposition.png", logx=True)
    figure([row["to_p"] for row in marginal], [[row["recall_points_per_10000_candidates"] for row in marginal]], ["Marginal gain"], "nprobe reached", "Recall points / +10k candidates", "07_marginal_gain.png", logx=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.3), layout="constrained")
    ax.hist(np.max(ranks, axis=1), bins=np.geomspace(1, B + 1, 35), edgecolor="none")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Router rank needed to include all 10 gold neighbors")
    ax.set_ylabel("Queries")
    ax.grid(alpha=0.25)
    fig.savefig(ROOT / "figures" / "08_gold_router_depth.png", dpi=300)
    save_svg_clean(fig, ROOT / "figures" / "08_gold_router_depth.svg")
    plt.close(fig)
    figure(x, [[difficulty[name]["mean_recall_by_p"][str(p)] for p in P] for name in groups], list(groups), "nprobe", "Mean Recall@10", "09_difficulty_groups.png", logx=True)
    print(json.dumps({"primary": [{"p": r["p"], "recall": r["recall"]["mean"], "candidates": r["candidate"]["mean"]} for r in main_rows], "targets": targets}, indent=2))


if __name__ == "__main__":
    main()
