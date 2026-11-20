# Reproducing the Emma-path DLI study

Everything runs from one committed source revision; the job verifies it (`native.commit`,
clean tree) and records the binary hash. See PREREGISTRATION.md for the protocol and
IMPLEMENTATION.md for the code changes.

## Inputs (remote input directory `IN`)

| File | Content |
|---|---|
| `source.bundle`, `native.commit` | `git bundle` of the branch and the exact commit to build |
| `vendor.tar`, `cargo-config.toml` | `cargo vendor --locked` output for that `Cargo.lock`; config with `__VENDOR_DIR__` placeholder |
| `queries.f32` | SISAP23 public queries, 10,000 × 768 f32 |
| `ground_truth.i32` | official top-1000 (used at 10M) |
| `{val,test}.{queries.f32,gold.u64}` | rows 900–931 and 9000–9255 and their official top-30 (same files as H-10M) |
| `inputs.sha256` | hashes of all of the above, checked first by the job |

The corpus is the verified 10,120,191-row Float16 file (sha256 `f5868d6e…`) already on
MetaCentrum storage; the job copies it to scratch and re-hashes it.

## Jobs

```bash
qsub -l walltime=03:00:00 -v MODE=preflight,IN=$IN research/dli/emma_path/pbs/emma.pbs
qsub -l walltime=08:00:00 -v MODE=pilot,IN=$IN    research/dli/emma_path/pbs/emma.pbs
qsub -l walltime=<from pilot projection> -v MODE=10m,IN=$IN research/dli/emma_path/pbs/emma.pbs
```

Each job writes `results/<mode>-<jobid>/` with per-arm `summary.json`, build/native/query
events, `dli_hierarchy.json`, `frontier/{rows,reference_costs}.jsonl`, `topology.json`,
`analysis.{json,txt}`, `SHA256SUMS` and `COMPLETE.json` (or `FAILED.json` with the phase).

## Analysis only

```bash
python -I research/dli/emma_path/analyze.py pilot   <out>/G0 <out>/E1 <out>/E1R --out pilot.json
python -I research/dli/emma_path/analyze.py confirm <out>/G0 <out>/E1 --out confirm.json
```
