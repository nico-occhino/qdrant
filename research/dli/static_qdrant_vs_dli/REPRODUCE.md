# Reproduce: static Qdrant LMI lifecycle vs DLI

**Branch and environment.** Branch `research/static-vs-dli-20261122`; environment as in `research/BUILD_ENVIRONMENT.md`
(conda env `lmi-rust-inspect`, `LIBTORCH_USE_PYTORCH=1`).

## Local checks
```bash
cargo test --offline -p collection --lib --features testing,lmi-training,dli-research -- dli_ lmi
python3 -I research/dli/static_qdrant_vs_dli/test_analyze_static_vs_dli.py
python3 -I research/dli/emma_path/test_analyze_phase3.py
```

## One workload locally (synthetic or small)
1. Build a plan: `python3 -I research/dli/emma_path/phase3_plan.py --corpus C --queries Q --rows N --scenario insert_only --seed 20261301 --out PLAN`
2. Arm D: `DLI_P3_SYSTEM=dli DLI_ARM=E1 DLI_FRONTIER_POLICIES=mdn_code DLI_FRONTIER_BUDGETS=... DLI_P3_PLAN=PLAN DLI_P3_CORPUS=C DLI_P3_OUTPUT=OUT_D DLI_P3_SCRATCH=TMP DLI_P3_SETS=val QDRANT_LMI_SAMPLE_BUDGET_BYTES=2000000000 cargo test ... -- --ignored --exact tests::lmi_lifecycle::dli_sisap_scaling_benchmark::dli_phase3_workload --nocapture`
3. Arm S: the same command with `DLI_P3_SYSTEM=static DLI_P3_STATIC_SEGMENTS=1 DLI_P3_STATIC_BUCKETS=1000 DLI_P3_STATIC_INDEXING_KB=20000 DLI_P3_STATIC_NPROBES=1,2,4,...`
4. Analyse: `python3 -I research/dli/static_qdrant_vs_dli/analyze_static_vs_dli.py --pair W=OUT_S:OUT_D --set val`

## MetaCentrum
Stage a source bundle and the verified inputs as for the Emma path (`research/dli/emma_path/REPRODUCE.md`), then:
`qsub -l walltime=12:00:00 -v MODE=pilot,SCENARIO=<insert_only|shift|constant_r|constant_o>,IN=<dir> research/dli/static_qdrant_vs_dli/pbs/static_vs_dli.pbs`.
Results go to `dli-emma-path-20261120/results/static-vs-dli-<mode>-<scenario>-<job>/`. A directory is never reused.
