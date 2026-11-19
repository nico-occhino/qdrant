"""Tier 3 analysis (TIER3_PREREGISTRATION.md + Amendments A and B; 3A and 3B jobs).

Usage: python3 -I analyze_tier3.py <job-results-dir> <gold-10m-dir> <out-prefix>

Operating points are locked on validation (smallest setting with R@10 >= 0.95, and separately
R@30 >= 0.90); test recall/latency/throughput are reported at those settings. Recall is
returned-ID recall against the official SISAP23 top-30 gold. Plain runs on the first 256
queries of each split.
"""
import json
import sys
from pathlib import Path

import numpy as np

ORDER = ["plain", "hnsw", "lmi-native", "ivf-3162", "ivf-8192"]
# Paired comparisons (a over b): Q1 in 3A, Q4 in 3B. Run when both systems are present.
PAIRS = [("ivf-3162", "lmi-native"), ("ivf-8192", "ivf-3162")]
TARGETS = {"r10": 0.95, "r30": 0.90}
B = 10_000
rng = np.random.default_rng(20261119)


def load_queries(path):
    """Per setting: per-query ids/candidates/route_ns from the recall pass (records with ids)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "ids" not in r:
            continue
        d = out.setdefault(r["setting"], {"q": [], "ids": [], "cand": [], "route": []})
        # ann_matched_v120 stores row i as point id i+1; gold is 0-based rows.
        d["q"].append(r["q"]); d["ids"].append([i - 1 for i in r["ids"]]); d["cand"].append(r["candidates"]); d["route"].append(r["route_ns"])
    for d in out.values():
        order = np.argsort(d["q"])
        for k in d:
            d[k] = np.asarray(d[k])[order]
        assert np.array_equal(d["q"], np.arange(len(d["q"])))
    return out


def recalls(ids, gold):
    n = len(ids)
    r10 = np.array([len(set(ids[i][:10]) & set(gold[i][:10])) / 10 for i in range(n)])
    r30 = np.array([len(set(ids[i][:30]) & set(gold[i][:30])) / 30 for i in range(n)])
    return r10, r30


def ci(x):
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def ratio_ci(a, b):
    """Paired bootstrap CI of mean(a)/mean(b) over the same queries."""
    idx = rng.integers(0, len(a), size=(B, len(a)))
    r = a[idx].mean(axis=1) / b[idx].mean(axis=1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def main():
    res, gold_dir, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    gold = {t: np.load(gold_dir / t / "gold.npy") for t in ("val", "test")}
    systems = [s for s in ORDER if (res / s / "build.json").exists()]
    report = {"job": res.name, "systems": {}}
    per_query = {}
    for s in systems:
        build = json.load(open(res / s / "build.json"))
        events = []
        ev = res / s / "build-events.jsonl"
        if ev.exists():
            events = [json.loads(l) for l in open(ev)]
        sysrep = {"n_buckets": build["spec"].get("lmi", {}).get("n_buckets"),
                  "build": {k: build[k] for k in ("seconds_total", "seconds_index", "threads", "index_bytes", "segment_bytes", "peak_rss_kib")},
                  "build_events": {e["stage"]: {k: v for k, v in e.items() if k.endswith("seconds") or k == "posting_pass_seconds"} for e in events},
                  "curves": {}}
        for t in ("val", "test"):
            summ = json.load(open(res / s / f"summary-{t}.json"))
            q = load_queries(res / s / f"queries-{t}.jsonl")
            curve = []
            for setting in summ["settings"]:
                d = q[setting]
                g = gold[t][: len(d["q"])]
                r10, r30 = recalls(d["ids"], g)
                per_query[(s, t, setting)] = {"r10": r10, "r30": r30, "cand": d["cand"].astype(float), "route": d["route"].astype(float)}
                lat = [l for l in summ["latency"] if l["setting"] == setting]
                thr = {x["threads"]: x["qps"] for x in summ["throughput"] if x["setting"] == setting}
                curve.append({
                    "setting": setting, "queries": len(d["q"]),
                    "r10": float(r10.mean()), "r30": float(r30.mean()),
                    "result_len_min": int(min(len(x) for x in d["ids"])),
                    "candidates_mean": float(d["cand"].mean()),
                    "route_p50_ms": float(np.median(d["route"]) / 1e6),
                    "p50_ms": float(np.median([l["p50_ns"] for l in lat]) / 1e6),
                    "p95_ms": float(np.median([l["p95_ns"] for l in lat]) / 1e6),
                    "p99_ms": float(np.median([l["p99_ns"] for l in lat]) / 1e6),
                    "p50_ms_reps": [l["p50_ns"] / 1e6 for l in lat],
                    "qps": thr,
                })
            sysrep["curves"][t] = curve
        locked = {}
        for key, target in TARGETS.items():
            ok = [c["setting"] for c in sysrep["curves"]["val"] if c[key] >= target]
            if not ok:
                locked[key] = None
                continue
            setting = min(ok)
            tc = next(c for c in sysrep["curves"]["test"] if c["setting"] == setting)
            pq = per_query[(s, "test", setting)]
            locked[key] = {"setting": setting, "val": next(c[key] for c in sysrep["curves"]["val"] if c["setting"] == setting),
                           "test": tc[key], "test_ci": ci(pq[key]), **{k: tc[k] for k in ("candidates_mean", "route_p50_ms", "p50_ms", "p95_ms", "p99_ms", "qps")}}
        sysrep["locked"] = locked
        report["systems"][s] = sysrep

    # Plain = exact search over stored Float16: agreement with the official gold.
    if "plain" in report["systems"]:
        report["plain_vs_official"] = {t: {"r10": c["r10"], "r30": c["r30"], "queries": c["queries"]}
                                       for t in ("val", "test") for c in report["systems"]["plain"]["curves"][t]}

    # Paired comparisons on the same test queries at each one's validation-locked setting,
    # plus same-setting comparisons when both use the same grid unit (same B).
    report["pairs"] = {}
    for a_name, b_name in PAIRS:
        if a_name not in report["systems"] or b_name not in report["systems"]:
            continue
        sa, sb = report["systems"][a_name], report["systems"][b_name]
        locked = {}
        for key in TARGETS:
            a, b = sa["locked"][key], sb["locked"][key]
            if a and b:
                pa, pb = per_query[(a_name, "test", a["setting"])], per_query[(b_name, "test", b["setting"])]
                locked[key] = {"a_setting": a["setting"], "b_setting": b["setting"],
                               "candidates_ratio": a["candidates_mean"] / b["candidates_mean"], "candidates_ratio_ci": ratio_ci(pa["cand"], pb["cand"]),
                               "p50_ratio": a["p50_ms"] / b["p50_ms"], "recall_diff": a["test"] - b["test"], "recall_diff_ci": ci(pa[key] - pb[key])}
        same = []
        if sa["n_buckets"] is not None and sa["n_buckets"] == sb["n_buckets"]:
            bset = {c["setting"]: c for c in sb["curves"]["test"]}
            for c in sa["curves"]["test"]:
                st = c["setting"]
                if st not in bset:
                    continue
                pa, pb = per_query[(a_name, "test", st)], per_query[(b_name, "test", st)]
                lc = bset[st]
                same.append({"nprobe": st, "r10_a": c["r10"], "r10_b": lc["r10"], "r10_diff_ci": ci(pa["r10"] - pb["r10"]),
                             "cand_ratio": c["candidates_mean"] / lc["candidates_mean"], "p50_ratio": c["p50_ms"] / lc["p50_ms"],
                             "route_ms_a": c["route_p50_ms"], "route_ms_b": lc["route_p50_ms"]})
        report["pairs"][f"{a_name}/{b_name}"] = {"locked": locked, "same_setting": same}

    Path(prefix + ".json").write_text(json.dumps(report, indent=1))
    Path(prefix + ".txt").write_text(render(report))
    print(render(report))


def render(r):
    L = [f"Tier 3 — {r['job']}", ""]
    L.append("Build (index-only):")
    for s, x in r["systems"].items():
        b = x["build"]
        L.append(f"  {s:11s} {b['seconds_index']:8.0f} s  threads={b['threads']:2d}  index={b['index_bytes']/1e6:8.1f} MB  rss={b['peak_rss_kib']/1e6:5.1f} GB")
        for st, e in x["build_events"].items():
            L.append(f"      {st}: " + ", ".join(f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items() if k != "posting_pass_seconds"))
    for t in ("val", "test"):
        L += ["", f"Curves ({t}):", f"  {'system':11s} {'set':>5s} {'R@10':>6s} {'R@30':>6s} {'cand':>9s} {'route':>7s} {'p50':>8s} {'p99':>8s}  qps(1/16)"]
        for s, x in r["systems"].items():
            for c in x["curves"][t]:
                q = c["qps"]
                L.append(f"  {s:11s} {c['setting']:5d} {c['r10']:6.3f} {c['r30']:6.3f} {c['candidates_mean']:9.0f} {c['route_p50_ms']:7.2f} {c['p50_ms']:8.2f} {c['p99_ms']:8.2f}  "
                         f"{q.get(1, float('nan')):.0f}/{q.get(16, float('nan')):.0f}")
    for key, target in TARGETS.items():
        L += ["", f"Validation-locked at {key.upper()} >= {target} (test values):"]
        for s, x in r["systems"].items():
            lk = x["locked"][key]
            if lk is None:
                L.append(f"  {s:11s} not reached on validation"); continue
            L.append(f"  {s:11s} set={lk['setting']:5d} val={lk['val']:.3f} test={lk['test']:.3f} [{lk['test_ci'][0]:.3f},{lk['test_ci'][1]:.3f}] "
                     f"cand={lk['candidates_mean']:.0f} p50={lk['p50_ms']:.2f} p95={lk['p95_ms']:.2f} p99={lk['p99_ms']:.2f} ms "
                     f"qps1={lk['qps'].get(1, float('nan')):.0f} qps16={lk['qps'].get(16, float('nan')):.0f}")
    if "plain_vs_official" in r:
        L += ["", "Plain (exact Float16) vs official gold:"] + [f"  {t}: R@10={v['r10']:.4f} R@30={v['r30']:.4f} (n={v['queries']})" for t, v in r["plain_vs_official"].items()]
    for pair, p in r["pairs"].items():
        L += ["", f"{pair} at locked points (ratios a/b):"]
        for key, v in p["locked"].items():
            L.append(f"  {key}: setting {v['a_setting']} vs {v['b_setting']}: candidates x{v['candidates_ratio']:.3f} [{v['candidates_ratio_ci'][0]:.3f},{v['candidates_ratio_ci'][1]:.3f}], "
                     f"p50 x{v['p50_ratio']:.3f}, recall diff {v['recall_diff']:+.3f} [{v['recall_diff_ci'][0]:+.3f},{v['recall_diff_ci'][1]:+.3f}]")
        if p["same_setting"]:
            L.append(f"  {pair} same nprobe (test):")
        for v in p["same_setting"]:
            L.append(f"  nprobe {v['nprobe']:4d}: R@10 {v['r10_a']:.3f} vs {v['r10_b']:.3f} diff CI [{v['r10_diff_ci'][0]:+.3f},{v['r10_diff_ci'][1]:+.3f}]  "
                     f"cand x{v['cand_ratio']:.3f}  p50 x{v['p50_ratio']:.3f}  route {v['route_ms_a']:.2f}/{v['route_ms_b']:.2f} ms")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
