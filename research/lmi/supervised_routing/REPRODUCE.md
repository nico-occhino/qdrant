# Reproduce

## 1. Stage inputs (local → MetaCentrum)

The input directory `static-supervised-routing-20261119/native-<commit12>/` holds:
- `source.bundle` (`git bundle` of the branch) and `native.commit`;
- `common.sh` and `sr-1m.pbs`;
- `queries.f32` (official 10K queries, sha256 `03ef80fa…`) and `ground_truth.i32` (official 10M gold, `e0c33df2…`);
- `val.queries.f32` (accepted Tier 1–3 validation queries, `60f1ac05…`);
- `profile-ivf8192-queries.jsonl` (job 25718749) and `tier3b-ivf8192-queries-test.jsonl` (job 25743351);
- symbolic links to the vendored crates (`dli-v120-20261117/vendor.tar`, `cargo-config.toml`);
- `inputs.sha256` over all of the above.

The corpus is read from the fixed path in `pbs/common.sh` (sha256 `f5868d6e…`).

## 2. Jobs

```bash
cd /storage/budejovice1/home/nico/lmi-metacentrum/static-supervised-routing-20261119/native-d76aea7b245b
```
```bash
qsub -N sr-preflight -v MODE=preflight -l walltime=2:00:00 sr-1m.pbs
```
```bash
qsub -N sr-pilot -v MODE=pilot -l walltime=12:00:00 -W depend=afterok:<preflight-id> sr-1m.pbs
```

The job verifies the inputs, builds the harness (`cargo test --profile perf -p segment --features testing,lmi-training
--test supervised_routing_v120 --no-run`), then runs these phases:
- source, build;
- export (A0, partition fingerprint);
- verify_partition (pilot only);
- inputs (`make_inputs.py`), native gold (T, V, G);
- train (`train_router.py`);
- consistency, evaluate;
- persist (`SHA256SUMS`, `COMPLETE.json`).

## 3. Analysis (local)

Run from the repository root.

```bash
uv run --no-project --with numpy --with matplotlib python -I research/lmi/supervised_routing/analyze.py pilot <job-output-dir> <analysis-dir>
```
```bash
uv run --no-project --with numpy python -I research/lmi/supervised_routing/posthoc_offline_g.py <job-output-dir> <official queries.f32> <out.json>
```

`analyze.py` recomputes recall from the returned IDs against gold, computes per-query medians over repetitions,
matched-recall points (lowest observed, no interpolation), the gate with a paired query bootstrap (10,000 resamples,
seed 20261119), native/offline agreement, oracle coverage, `COMPARISON_TABLES.csv` and the figures.

## Environment

- Rust 1.98.0 and protoc 3.18.0, from `sisap23-build-environment-v1.sh`.
- PyTorch 2.5.1+cpu and numpy 1.26.4, from `envs/py312-torch251`.
- AMD node with 16 CPUs, 96 GB RAM and 200 GB local SSD scratch.
- Training is CPU-only and deterministic (seed 7). The data-shuffling and initialisation generators are seeded.
