"""CLI centroid-routing control (CLI_SHARED_SPEC.md §8; descriptive, validation panels).

    analyze_control.py --mlp RUN_DIR --centroid RUN_DIR [--out FILE] [--resamples 10000]

Each RUN_DIR is a `cli_stage_a_plan` output (rows.jsonl, batches.jsonl) on the same plan and panel.
Per batch and routing:
* W@0.90: the first measured crossing of mean Recall@30 = 0.90 against mean vectors scored, interpolated, never
  extrapolated (`upper_bound` / `not_attained` explicit);
* R@30 at the first budget with mean scored >= 50,000.
The contrast is R = mean_t W_centroid(t) / mean_t W_mlp(t), over batches where both attain, from a batch-stratified
paired query bootstrap (10,000 resamples, seed 20261128) with the crossing recomputed per resample. The maintenance
records (path, splits, reclusters, buckets, L_route, train seconds) are summarized per routing.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGET, NOMINAL, SEED = 0.90, 50_000, 20261128


def load(run):
    rec = defaultdict(lambda: defaultdict(dict))  # batch -> budget -> query -> (recall, scored)
    for line in open(Path(run) / "rows.jsonl"):
        r = json.loads(line)
        rec[r["batch"]][r["budget"]][r["query"]] = (r["recall30"] / 30, r["scored"])
    batches = [json.loads(l) for l in open(Path(run) / "batches.jsonl")]
    out = {}
    for t, by_budget in rec.items():
        budgets = sorted(by_budget)
        queries = sorted(by_budget[budgets[0]])
        recall = np.array([[by_budget[b][q][0] for q in queries] for b in budgets])
        scored = np.array([[by_budget[b][q][1] for q in queries] for b in budgets], dtype=float)
        out[t] = (budgets, queries, recall, scored)
    return out, batches


def w_at(recall_mean, scored_mean):
    if recall_mean[0] >= TARGET:
        return float(scored_mean[0]), "upper_bound"
    for i in range(1, len(recall_mean)):
        if recall_mean[i] >= TARGET:
            r0, r1, w0, w1 = recall_mean[i - 1], recall_mean[i], scored_mean[i - 1], scored_mean[i]
            return float(w0 + (TARGET - r0) / (r1 - r0) * (w1 - w0)), "interpolated"
    return None, "not_attained"


def at_50k(recall_mean, scored_mean):
    for r, w in zip(recall_mean, scored_mean):
        if w >= NOMINAL:
            return float(r)
    return None


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--mlp", required=True)
    a.add_argument("--centroid", required=True)
    a.add_argument("--out", type=Path)
    a.add_argument("--resamples", type=int, default=10_000)
    args = a.parse_args()
    runs = {"mlp": load(args.mlp), "centroid": load(args.centroid)}
    common = sorted(set(runs["mlp"][0]) & set(runs["centroid"][0]))
    per_batch, used = {}, []
    for t in common:
        row = {}
        for name, (data, _) in runs.items():
            budgets, queries, recall, scored = data[t]
            w, kind = w_at(recall.mean(1), scored.mean(1))
            row[name] = {"w090": w, "kind": kind, "recall_50k": at_50k(recall.mean(1), scored.mean(1))}
        assert runs["mlp"][0][t][1] == runs["centroid"][0][t][1], f"batch {t}: query sets differ"
        per_batch[t] = row
        if row["mlp"]["w090"] is not None and row["centroid"]["w090"] is not None:
            used.append(t)
    point = (np.mean([per_batch[t]["centroid"]["w090"] for t in used]) /
             np.mean([per_batch[t]["mlp"]["w090"] for t in used])) if used else None
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(args.resamples if used else 0):
        sums = {"mlp": [], "centroid": []}
        ok = True
        for t in used:
            nq = len(runs["mlp"][0][t][1])
            idx = rng.integers(0, nq, nq)
            for name in sums:
                _, _, recall, scored = runs[name][0][t]
                w, _ = w_at(recall[:, idx].mean(1), scored[:, idx].mean(1))
                if w is None:
                    ok = False
                sums[name].append(w)
        if ok:
            boots.append(np.mean(sums["centroid"]) / np.mean(sums["mlp"]))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None
    maint = {}
    for name, (_, batches) in runs.items():
        ms = [b["maintenance"] for b in batches[1:]]
        lr = [m["l_route"] for m in ms if m.get("l_route") is not None]
        maint[name] = {"splits": sum(m["split_buckets"] for m in ms), "reclusters": sum(m["path"] == "recluster" for m in ms),
                       "triggers": [m["trigger"] for m in ms], "final_buckets": batches[-1]["buckets"],
                       "train_seconds": float(sum(m["train_seconds"] for m in ms)),
                       "build_train_seconds": batches[0]["maintenance"]["train_seconds"],
                       "l_route_mean": float(np.mean(lr)) if lr else None, "l_route_max": float(max(lr)) if lr else None,
                       "max_over_mean_before": [round(m["max_over_mean_before"], 2) for m in ms]}
    res = {"batches_used": used, "R_centroid_over_mlp": point, "ci95": ci, "valid_resamples": len(boots),
           "per_batch": per_batch, "maintenance": maint}
    text = json.dumps(res, indent=1, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({k: res[k] for k in ("batches_used", "R_centroid_over_mlp", "ci95", "valid_resamples")}, default=float))
    for t in common:
        r = per_batch[t]
        print(t, {n: (None if r[n]["w090"] is None else round(r[n]["w090"]), r[n]["kind"][0],
                      None if r[n]["recall_50k"] is None else round(r[n]["recall_50k"], 3)) for n in r})
    for n, m in maint.items():
        print(n, {k: v for k, v in m.items() if k not in ("triggers", "max_over_mean_before")})


if __name__ == "__main__":
    main()
