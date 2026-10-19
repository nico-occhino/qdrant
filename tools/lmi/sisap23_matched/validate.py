"""Independent raw-result and input validation for SISAP23 matched benchmark."""
import argparse, hashlib, json, statistics
from pathlib import Path


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def summarize(rows, np):
    ms = [float(r["native_search_ms"]) for r in rows]
    return {"mean_ms": statistics.fmean(ms), "p50_ms": float(np.quantile(ms, .5)),
            "p95_ms": float(np.quantile(ms, .95)), "p99_ms": float(np.quantile(ms, .99))}


def check_ids(data, path, gold, plain):
    """Compare each record with reference rows addressed by global SISAP query_row."""
    recalls, ref = [], []
    for i, r in enumerate(data):
        ids = list(map(int, r["returned_ids_1_based"]))
        if len(ids) != 10 or len(set(ids)) != 10:
            raise ValueError(f"invalid top10 at {path}:{i}")
        query_index = int(r["query_row"]) - 20
        if not 0 <= query_index < len(gold) or query_index >= len(plain):
            raise ValueError(f"{path}: query_row outside reference arrays: {r['query_row']}")
        actual = len(set(ids) & set(map(int, gold[query_index]))) / 10
        if abs(float(r["recall_at_10"]) - actual) > 1e-12:
            raise ValueError(f"official recall mismatch: {path}: query_row {r['query_row']}")
        recalls.append(actual)
        ref.append(len(set(ids) & set(map(int, plain[query_index]["returned_ids_1_based"]))) / 10)
    return recalls, ref



def check_recovery_provenance(root, manifest, variant):
    from preserve_generation import inventory
    from recovery_stages import HARDWARE, BASE, checked_generation
    run = root / "runs" / variant
    marker = json.loads((run / "evaluation.generation.json").read_text())
    complete = json.loads((run / "evaluation.complete.json").read_text())
    expected_source = BASE if variant == "plain" else manifest["source_commit"]
    if (marker.get("format") != "sisap23-evaluation-generation-v2"
            or marker.get("variant") != variant
            or marker.get("source_commit") != expected_source
            or manifest["evaluation_sources"].get(variant) != expected_source
            or complete.get("source_commit") != expected_source
            or complete.get("accepted") is not True
            or marker.get("hardware_fingerprint") != HARDWARE
            or complete.get("hardware_fingerprint") != HARDWARE
            or manifest.get("hardware_fingerprint") != HARDWARE
            or complete.get("marker_sha256") != digest(run / "evaluation.generation.json")
            or complete.get("manifest_sha256") != digest(root / "manifest.json")):
        raise ValueError(f"recovery provenance mismatch: {variant}")
    if variant == "plain":
        if marker.get("origin") != "imported-plain":
            raise ValueError("Plain must retain its historical evaluator provenance")
        for folder in ("full", "panel"):
            if inventory(run / folder) != marker["result_inventories"][folder]:
                raise ValueError("preserved Plain results changed")
    else:
        if complete.get("result_inventories") != {f:inventory(run / f) for f in ("full","panel")}:
            raise ValueError("accepted query results changed")
        generation = Path(marker["generation_path"])
        if digest(generation / "generation.json") != marker["generation_sha256"]:
            raise ValueError("index generation marker changed")
        meta = checked_generation(generation, expected_source, "index", variant)
        if inventory(run / "collection") != meta["collection_files"]:
            raise ValueError("evaluated index differs from accepted generation")
    return marker

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    root = ap.parse_args().root
    import numpy as np
    manifest = json.loads((root / "manifest.json").read_text())
    for key, entry in manifest["prepared"].items():
        p = root / entry["path"]
        if not p.is_file() or p.stat().st_size != entry["bytes"] or digest(p) != entry["sha256"]:
            raise SystemExit(f"prepared input hash/size mismatch: {key}")
    expected_panel = np.fromfile(root / "panel_rows.u32le", dtype="<u4").astype(int).tolist()
    gold = np.fromfile(root / "gold_20_9999_top10.i32le", dtype="<i4").reshape(9980, 10)
    plain = read_jsonl(root / "runs/plain/full/full/rows.jsonl")
    if len(plain) != 9980:
        raise SystemExit(f"Plain exact reference has {len(plain)} rows")
    grids = {"plain": ["full"], "lmi": [f"nprobe-{p}" for p in (1, 4, 8, 16, 30, 40, 64, 100, 160, 256)],
             "hnsw-a": [f"hnsw-ef-{p}" for p in (16, 32, 64, 128, 256, 512, 1024)],
             "hnsw-b": [f"hnsw-ef-{p}" for p in (16, 32, 64, 128, 256, 512, 1024)]}
    results, panels = {}, {}
    expected = {"plain": ("plain", None, None), "lmi": ("lmi", None, None),
                "hnsw-a": ("hnsw", 16, 100), "hnsw-b": ("hnsw", 32, 200)}
    for variant, settings in grids.items():
        run = root / "runs" / variant
        imported = variant == "plain" and (run / "imported.generation.json").exists()
        recovered = manifest.get("protocol") == "independent-recovery-v2"
        marker = (check_recovery_provenance(root, manifest, variant) if recovered else
                  json.loads((run / ("imported.generation.json" if imported else "build.complete.json")).read_text()))
        kind, m, ef = expected[variant]
        if marker["index_kind"] != kind or marker["point_count"] != 10120191 or marker.get("segment_count") != 1:
            raise SystemExit(f"wrong build marker: {variant}")
        if recovered:
            pass  # Per-variant source and immutable generation checked above.
        elif imported:
            generation_path = Path(marker["generation_path"])
            generation_file = generation_path / "generation.json"
            generation = json.loads(generation_file.read_text())
            if (marker.get("format") != "sisap23-imported-plain-v1"
                    or marker.get("source_commit") != "42a163701a51078ec24983b3a93f96ab7d661489"
                    or generation.get("preservation_commit") != manifest.get("source_commit")
                    or generation.get("accepted") is not True
                    or digest(generation_file) != marker.get("generation_json_sha256")
                    or str(run / "collection") not in marker.get("segment_path", "")):
                raise SystemExit("imported Plain provenance mismatch")
        elif marker.get("source_commit") != manifest.get("source_commit"):
            raise SystemExit(f"build source commit differs from input manifest: {variant}")
        if m is not None and (marker["hnsw_m"] != m or marker["ef_construct"] != ef):
            raise SystemExit(f"wrong HNSW config: {variant}")
        if variant == "lmi" and marker.get("lmi") != {"n_buckets":3162,"sample_size":250000,"hidden_dim":512,"epochs":30,"batch_size":256,"routing_batch_size":256,"kmeans_iterations":5,"nprobe":4,"seed":42}:
            raise SystemExit("wrong LMI config")
        for setting in settings:
            path = run / "full" / setting / "rows.jsonl"
            data = read_jsonl(path)
            if len(data) != 9980 or [r["query_row"] for r in data] != list(range(20, 10000)):
                raise SystemExit(f"invalid full query coverage: {path}")
            try:
                recall, ref = check_ids(data, path, gold, plain)
            except ValueError as e:
                raise SystemExit(f"{path}: {e}") from e
            ms = [float(r["native_search_ms"]) for r in data]
            if any(r.get("parameter") != setting for r in data):
                raise SystemExit(f"setting label mismatch: {path}")
            if variant == "plain" and any(r.get("candidate_count") != 10120191 for r in data):
                raise SystemExit(f"Plain full-scan count mismatch: {path}")
            if variant == "lmi" and any(r.get("candidate_count", 0) > 10120191 for r in data):
                raise SystemExit(f"LMI candidate count exceeds corpus: {path}")
            if variant.startswith("hnsw") and any(r.get("hnsw_ef") != int(setting.rsplit("-", 1)[1]) for r in data):
                raise SystemExit(f"HNSW ef mismatch: {path}")
            if variant == "lmi" and any(r.get("candidate_count") is None for r in data):
                raise SystemExit(f"LMI candidate count missing: {path}")
            if variant.startswith("hnsw") and any(r.get("hnsw_ef") is None for r in data):
                raise SystemExit(f"HNSW effort missing: {path}")
            results[f"{variant}/{setting}"] = {"n":len(data),"mean_recall_official":statistics.fmean(recall),
                "mean_recall_float16_plain_reference":statistics.fmean(ref), **summarize(data,np),
                "queries_per_second":1000/statistics.fmean(ms)}
            rep_stats = []
            for rep in (1,2,3):
                p = run / "panel" / f"rep-{rep}" / setting / "rows.jsonl"
                panel = read_jsonl(p)
                if len(panel) != 1000:
                    raise SystemExit(f"missing paired panel rows: {p}")
                rows = [r["query_row"] for r in panel]
                if rows != expected_panel:
                    raise SystemExit(f"paired panel IDs differ from manifest input: {p}")
                try:
                    pr, rf = check_ids(panel, p, gold, plain)
                except ValueError as e:
                    raise SystemExit(f"{p}: {e}") from e
                if any(r.get("parameter") != setting for r in panel):
                    raise SystemExit(f"panel setting label mismatch: {p}")
                if variant.startswith("hnsw") and any(r.get("hnsw_ef") != int(setting.rsplit("-",1)[1]) for r in panel):
                    raise SystemExit(f"panel HNSW ef mismatch: {p}")
                rep_stats.append({"repetition":rep,"mean_recall_official":statistics.fmean(pr),
                    "mean_recall_float16_plain_reference":statistics.fmean(rf), **summarize(panel,np)})
            for metric in ("mean_ms","p50_ms","p95_ms","p99_ms"):
                vals = [x[metric] for x in rep_stats]
                panels[f"{variant}/{setting}"] = {"repetitions":rep_stats,
                    "median_"+metric:float(np.median(vals)), "range_"+metric:[min(vals),max(vals)]}
    report = {"manifest_sha256":digest(root/"manifest.json"),"source_commit":manifest["source_commit"],
              "full_sweeps":results,"paired_warm_panels":panels,"validation":"passed"}
    if manifest.get("protocol") == "independent-recovery-v2":
        report["evaluation_sources"] = manifest["evaluation_sources"]
        report["hardware_fingerprint"] = manifest["hardware_fingerprint"]
        report["plain_measurements"] = "preserved historical measurements; not remeasured"
    report_path = root / "validation.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing != report:
            raise SystemExit("saved validation report differs from current inputs or results")
    else:
        with open(report_path,"x",encoding="utf-8") as f:
            json.dump(report,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
