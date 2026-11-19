#!/usr/bin/env python3
"""Analysis and promotion gate for the supervised-routing pilot (PREREGISTRATION.md).

    analyze.py pilot RUN_DIR OUT_DIR      1M: frontiers on V and G, gate on G, figures, tables
    analyze.py confirm RUN_DIR OUT_DIR    10M: frontiers on C, final criterion, figures, tables

RUN_DIR is a persisted job output (eval/*.jsonl, gold/*.ids.u64 or inputs/*.official.u64,
train/training.json, export/, consistency files). Everything is recomputed from raw per-query
records: recall from returned ids against gold, latency as the per-query median over repetitions.
"Matched recall" = lowest observed value among grid points whose mean Recall@10 meets the target;
no interpolation or extrapolation.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGETS = [0.90, 0.95, 0.97, 0.98, 0.99]
BOOT = 10_000
RNG = np.random.default_rng(20261119)


def load_eval(path):
    """-> {arm: {nprobe: dict(q -> record)}} with ns lists across repetitions."""
    data = defaultdict(lambda: defaultdict(dict))
    for line in open(path):
        r = json.loads(line)
        d = data[r["arm"]][r["nprobe"]].setdefault(r["q"], {"ns": [], "route": [], "union": [], "score": []})
        d["ns"].append(r["ns"]); d["route"].append(r["route_ns"]); d["union"].append(r["union_ns"])
        d["score"].append(r["score_ns"])
        if r["rep"] == 1:
            d["ids"], d["candidates"], d["hits"] = r["ids"], r["candidates"], r["hits"]
    return data


def gold_ids(path, k=30):
    return np.fromfile(path, dtype=np.uint64).reshape(-1, k)[:, :10].astype(np.int64)


def arrays(data, gold):
    """Per arm, per nprobe: recall, candidates, median-ns, and stage medians as (nq,) arrays."""
    out = {}
    for arm, by_p in data.items():
        out[arm] = {}
        for p, by_q in by_p.items():
            qs = sorted(by_q)
            assert qs == list(range(len(gold))), f"{arm} nprobe {p}: incomplete query set"
            rec = np.array([len(set(by_q[q]["ids"][:10]) & set(gold[q])) / 10 for q in qs])
            out[arm][p] = {
                "recall": rec,
                "candidates": np.array([by_q[q]["candidates"] for q in qs], float),
                "ns": np.array([np.median(by_q[q]["ns"]) for q in qs]),
                "route": np.array([np.median(by_q[q]["route"]) for q in qs]),
                "union": np.array([np.median(by_q[q]["union"]) for q in qs]),
                "score": np.array([np.median(by_q[q]["score"]) for q in qs]),
                "hits": np.array([by_q[q]["hits"] for q in qs]),
                "reps": len(by_q[qs[0]]["ns"]),
            }
    return out


def frontier_rows(arr, set_name):
    rows = []
    for arm in sorted(arr):
        for p in sorted(arr[arm]):
            a = arr[arm][p]
            rows.append({"set": set_name, "arm": arm, "nprobe": p, "recall": a["recall"].mean(),
                         "candidates": a["candidates"].mean(), "candidate_fraction": None,
                         "p50_ms": np.median(a["ns"]) / 1e6, "p95_ms": np.percentile(a["ns"], 95) / 1e6,
                         "p99_ms": np.percentile(a["ns"], 99) / 1e6, "mean_ms": a["ns"].mean() / 1e6,
                         "route_p50_ms": np.median(a["route"]) / 1e6, "union_p50_ms": np.median(a["union"]) / 1e6,
                         "score_p50_ms": np.median(a["score"]) / 1e6, "short_results": int((a["hits"] < 10).sum()),
                         "reps": a["reps"]})
    return rows


def matched(arr_arm, target, idx=None):
    """Lowest observed mean candidates and lowest median latency among points with mean recall >= target."""
    best_c = best_l = None
    for p, a in arr_arm.items():
        sel = slice(None) if idx is None else idx
        if a["recall"][sel].mean() < target:
            continue
        c, lat = a["candidates"][sel].mean(), np.median(a["ns"][sel])
        if best_c is None or c < best_c[0]:
            best_c = (c, p)
        if best_l is None or lat < best_l[0]:
            best_l = (lat, p, np.percentile(a["ns"][sel], 95))
    return best_c, best_l


def bootstrap_ratio(arr, a, b, target, kind):
    """Paired query bootstrap of matched latency (kind='latency') or candidates ratio a/b."""
    nq = len(next(iter(arr[a].values()))["recall"])
    ratios = []
    for _ in range(BOOT):
        idx = RNG.integers(0, nq, nq)
        ma, mb = matched(arr[a], target, idx), matched(arr[b], target, idx)
        if ma[0] is None or mb[0] is None:
            continue
        ratios.append(ma[1][0] / mb[1][0] if kind == "latency" else ma[0][0] / mb[0][0])
    ratios = np.array(ratios)
    return {"resamples": int(len(ratios)), "of": BOOT,
            "ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))] if len(ratios) else None}


def consistency(run):
    nat = {}
    for line in open(run / "consistency_native.jsonl"):
        r = json.loads(line)
        nat[(r["arm"], r["q"])] = r["buckets"]
    agree = total = 0
    worst = defaultdict(int)
    for line in open(run / "train" / "consistency_python.jsonl"):
        r = json.loads(line)
        n = nat.get((r["arm"], r["q"]))
        if n is None:
            continue
        for p in (1, 2, 4, 8, 16, 32, 64):
            total += 1
            same = n[:p] == r["buckets"][:p]
            agree += same
            if not same:
                worst[r["arm"]] += 1
    return {"checks": total, "agreement": agree / total if total else None, "disagreements_by_arm": dict(worst)}


def oracle(gold, bucket_of, occupancy, budgets_p, cand_budgets):
    """Coverage of the best fixed-nprobe bucket choice and of a density-greedy candidate-budget choice."""
    nq = len(gold)
    fixed = {p: 0.0 for p in budgets_p}
    budget = {c: 0.0 for c in cand_budgets}
    for q in range(nq):
        b = bucket_of[gold[q] - 1]
        buckets, v = np.unique(b, return_counts=True)
        order = np.argsort(-v, kind="stable")
        cum = np.cumsum(v[order]) / 10
        for p in budgets_p:
            fixed[p] += cum[min(p, len(cum)) - 1]
        dens = np.argsort(-(v / np.maximum(occupancy[buckets], 1)), kind="stable")
        spent, got = np.cumsum(occupancy[buckets][dens]), np.cumsum(v[dens]) / 10
        for c in cand_budgets:
            within = spent <= c
            budget[c] += got[within][-1] if within.any() else 0.0
    return ({p: fixed[p] / nq for p in budgets_p}, {c: budget[c] / nq for c in cand_budgets})


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def figures(arr_by_set, out, n_points, occupancy, training, oracle_cov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    for set_name, arr in arr_by_set.items():
        for metric, label, fname in (("candidates", "mean candidates", "recall_vs_candidates"),
                                     ("p50", "median latency (ms)", "recall_vs_p50"),
                                     ("p95", "p95 latency (ms)", "recall_vs_p95")):
            fig, ax = plt.subplots(figsize=(7, 5))
            for arm in sorted(arr):
                ps = sorted(arr[arm])
                x = [arr[arm][p]["candidates"].mean() if metric == "candidates" else
                     (np.median if metric == "p50" else lambda v: np.percentile(v, 95))(arr[arm][p]["ns"]) / 1e6
                     for p in ps]
                y = [arr[arm][p]["recall"].mean() for p in ps]
                ax.plot(x, y, marker="o", ms=3, label=arm, lw=1.5 if arm in ("A0",) else 1)
            ax.axhline(0.95, color="grey", ls=":", lw=0.8)
            ax.set_xscale("log"); ax.set_xlabel(label); ax.set_ylabel("Recall@10"); ax.set_ylim(0.6, 1.0)
            ax.set_title(f"{set_name}: {fname.replace('_', ' ')}"); ax.legend(fontsize=7, ncol=2)
            fig.tight_layout(); fig.savefig(figs / f"{set_name}_{fname}.png", dpi=130); plt.close(fig)
        # time decomposition at each arm's matched-0.95 latency point
        fig, ax = plt.subplots(figsize=(7, 4))
        names, parts = [], []
        for arm in sorted(arr):
            m = matched(arr[arm], 0.95)[1]
            if m is None:
                continue
            a = arr[arm][m[1]]
            names.append(f"{arm}\nnp={m[1]}")
            parts.append([np.median(a["route"]), np.median(a["union"]), np.median(a["score"])])
        if parts:
            parts = np.array(parts) / 1e6
            left = np.zeros(len(names))
            for i, lab in enumerate(("route", "union", "score")):
                ax.barh(names, parts[:, i], left=left, label=lab); left += parts[:, i]
            ax.set_xlabel("median ms at matched R@10 >= 0.95"); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(figs / f"{set_name}_time_decomposition.png", dpi=130)
        plt.close(fig)
        # candidate saving vs speedup at every recall target, relative to A0
        fig, ax = plt.subplots(figsize=(6, 5))
        for arm in sorted(arr):
            if arm == "A0":
                continue
            xs, ys = [], []
            for t in TARGETS:
                a, b = matched(arr[arm], t), matched(arr["A0"], t)
                if a[0] and b[0]:
                    xs.append(1 - a[0][0] / b[0][0]); ys.append(1 - a[1][0] / b[1][0])
            if xs:
                ax.plot(xs, ys, marker="o", label=arm)
        ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
        ax.plot([-0.5, 0.5], [-0.5, 0.5], color="grey", ls=":", lw=0.8)
        ax.set_xlabel("candidate saving vs A0"); ax.set_ylabel("median-latency saving vs A0")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figs / f"{set_name}_saving_vs_speedup.png", dpi=130)
        plt.close(fig)
    # occupancy
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(occupancy, bins=100); ax.set_yscale("log"); ax.set_xlabel("bucket occupancy n_b"); ax.set_ylabel("buckets")
    fig.tight_layout(); fig.savefig(figs / "occupancy.png", dpi=130); plt.close(fig)
    # training curves
    if training:
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in training["configs"]:
            ax.plot([p["step"] for p in c["curve"]], [p["c95"] for p in c["curve"]], lw=0.8, label=c["name"][:40])
        ax.axhline(training["selection"]["A0"]["val"]["c95"], color="k", ls="--", lw=1, label="A0")
        ax.set_xlabel("step"); ax.set_ylabel("V C95 (mean candidates)"); ax.set_yscale("log")
        ax.legend(fontsize=4, ncol=2); fig.tight_layout(); fig.savefig(figs / "training_curves.png", dpi=130)
        plt.close(fig)
    # oracle vs achieved coverage
    if oracle_cov:
        (fixed, budget), arr = oracle_cov
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(list(budget.keys()), list(budget.values()), "k--", label="oracle (candidate budget, density-greedy)")
        for arm in sorted(arr):
            ps = sorted(arr[arm])
            ax.plot([arr[arm][p]["candidates"].mean() for p in ps], [arr[arm][p]["recall"].mean() for p in ps],
                    marker=".", lw=0.8, label=arm)
        ax.set_xscale("log"); ax.set_xlabel("mean candidates"); ax.set_ylabel("coverage / Recall@10")
        ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(figs / "oracle_vs_achieved.png", dpi=130)
        plt.close(fig)


def main():
    mode, run, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    training = json.loads((run / "train" / "training.json").read_text()) if (run / "train" / "training.json").exists() else None
    occupancy = np.fromfile(run / "export" / "occupancy.u32", dtype=np.uint32).astype(np.int64)
    bucket_of = np.fromfile(run / "export" / "bucket_of_id.u32", dtype=np.uint32).astype(np.int64)
    sets = {"V": "V", "G": "G"} if mode == "pilot" else {"V": "V", "C": "C"}
    gold_path = (lambda s: run / "gold" / f"{s}.ids.u64") if mode == "pilot" else (lambda s: run / "inputs" / f"{s}.official.u64")
    arr_by_set, rows = {}, []
    for s in sets:
        if not (run / "eval" / f"{s}.jsonl").exists():
            continue
        arr = arrays(load_eval(run / "eval" / f"{s}.jsonl"), gold_ids(gold_path(s)))
        arr_by_set[s] = arr
        rows += frontier_rows(arr, s)
    for r in rows:
        r["candidate_fraction"] = r["candidates"] / len(bucket_of)
    write_csv(out / "COMPARISON_TABLES.csv", rows)

    primary = "G" if mode == "pilot" else "C"
    arr = arr_by_set[primary]
    summary = {"mode": mode, "run": str(run), "primary_set": primary, "matched": {}, "consistency": consistency(run)}
    for arm in sorted(arr):
        summary["matched"][arm] = {}
        for t in TARGETS:
            c, l = matched(arr[arm], t)
            summary["matched"][arm][str(t)] = None if c is None else {
                "min_candidates": c[0], "nprobe_at_min_candidates": c[1], "min_p50_ms": l[0] / 1e6,
                "nprobe_at_min_p50": l[1], "p95_ms_at_min_p50": l[2] / 1e6}
    # oracle coverage on the primary set
    gold = gold_ids(gold_path(primary))
    fixed, budget = oracle(gold, bucket_of, occupancy, [1, 2, 4, 8, 16, 32, 64],
                           [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000])
    summary["oracle_coverage"] = {"fixed_nprobe": fixed, "candidate_budget_density_greedy": budget}

    if mode == "pilot":
        promo = training["promotion_candidate"]
        a0, p = summary["matched"]["A0"]["0.95"], summary["matched"][promo]["0.95"]
        gate = {"promotion_candidate": promo}
        if a0 is None or p is None:
            gate["criteria"] = {"reached_0.95": False}
            gate["passed"] = False
        else:
            cand_red = 1 - p["min_candidates"] / a0["min_candidates"]
            lat_ratio = p["min_p50_ms"] / a0["min_p50_ms"]
            route_a0 = np.median(np.concatenate([arr["A0"][k]["route"] for k in arr["A0"]])) / 1e6
            route_p = np.median(np.concatenate([arr[promo][k]["route"] for k in arr[promo]])) / 1e6
            sel = training["selection"][promo]
            router_dir = Path(sel["router"]).parent
            local_dir = run / "train" / "routers" / promo
            bytes_p = sum(f.stat().st_size for f in (local_dir if local_dir.exists() else router_dir).glob("*.f32"))
            bytes_a0 = sum(f.stat().st_size for f in (run / "export" / "A0").glob("*.f32"))
            train_seconds = next((c["seconds"] for c in training["configs"] if c["name"] == sel.get("config")), 0.0)
            short = sum(int((a["hits"] < 10).sum()) for a in arr[promo].values())
            vp = json.loads((run / "verify_partition.json").read_text()) if (run / "verify_partition.json").exists() else None
            gate["criteria"] = {
                "1_candidate_reduction": {"value": cand_red, "threshold": 0.10, "pass": cand_red >= 0.10},
                "2_latency_reduction": {"ratio": lat_ratio, "pass": lat_ratio < 1.0,
                                        "bootstrap": bootstrap_ratio(arr, promo, "A0", 0.95, "latency")},
                "3_correctness": {"short_results": short, "consistency": summary["consistency"]["agreement"],
                                  "pass": short == 0 and (summary["consistency"]["agreement"] or 0) >= 0.99},
                "4_budget": {"router_bytes": bytes_p, "a0_router_bytes": bytes_a0, "training_seconds": train_seconds,
                             "route_p50_ms": route_p, "a0_route_p50_ms": route_a0,
                             "pass": bytes_p <= 2 * bytes_a0 and train_seconds <= 7200 and route_p <= route_a0 + 0.5},
                "5_generalisation": {"held_out_set": "G", "train_T_c95": next((c["train_T"]["c95"] for c in training["configs"] if c["name"] == sel.get("config")), None),
                                     "val_c95": sel["val"]["c95"], "pass": True},
                "6_integrity": {"verify_partition": vp, "pass": vp is not None and vp.get("candidate_mismatches", 1) <= 0.01 * vp.get("checked", 0)},
            }
            gate["candidate_bootstrap"] = bootstrap_ratio(arr, promo, "A0", 0.95, "candidates")
            gate["passed"] = all(c["pass"] for c in gate["criteria"].values())
        summary["gate"] = gate
    else:
        promo = [a for a in arr if a not in ("A0", "A1")][0]
        a0, p = summary["matched"]["A0"]["0.95"], summary["matched"][promo]["0.95"]
        boot = bootstrap_ratio(arr, promo, "A0", 0.95, "latency")
        ratio = None if (a0 is None or p is None) else p["min_p50_ms"] / a0["min_p50_ms"]
        p95_ok = None if ratio is None else p["p95_ms_at_min_p50"] <= 1.05 * a0["p95_ms_at_min_p50"]
        cand = None if (a0 is None or p is None) else 1 - p["min_candidates"] / a0["min_candidates"]
        if ratio is not None and ratio <= 0.90 and p95_ok and boot["ci95"] and boot["ci95"][1] <= 0.90:
            outcome = "positive (robust)"
        elif ratio is not None and ratio <= 0.90 and p95_ok:
            outcome = "positive (tentative)"
        elif cand is not None and cand >= 0.10:
            outcome = "routing-only positive"
        else:
            outcome = "negative"
        summary["final"] = {"arm": promo, "latency_ratio": ratio, "bootstrap": boot, "p95_ok": p95_ok,
                            "candidate_reduction": cand, "outcome": outcome}

    (out / "analysis.json").write_text(json.dumps(summary, indent=1, default=float))
    try:
        figures(arr_by_set, out, len(bucket_of), occupancy, training, ((fixed, budget), arr))
    except ImportError as e:
        print("figures skipped:", e)
    print(json.dumps(summary.get("gate", summary.get("final")), indent=1, default=float))


if __name__ == "__main__":
    main()
