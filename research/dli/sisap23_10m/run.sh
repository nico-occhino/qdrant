#!/usr/bin/env bash
set -euo pipefail
cd ${WORKTREES:?}/qdrant-dli-cross-segment
campaign=${ADMIN:?}/experiments/2026-11-12-dli-architecture/continuation-25
export CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=${ADMIN:?}/experiments/2026-11-12-dli-architecture/cargo-target
export LIBTORCH_USE_PYTORCH=1 PATH=${CONDA_ENV:?}/lmi-rust-inspect/bin:$PATH
export LD_LIBRARY_PATH=${CONDA_ENV:?}/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export QDRANT_MEASURE_SCORING=1
export DLI_BENCH_HEAD="$(git rev-parse HEAD)"
system=${1:-geometric}
case "$system" in geometric|size-tiered|static) ;; *) exit 2 ;; esac
# Isolated processes. Gate each larger population on previous success.
for population in ${DLI_RUN_POINTS:-1000000 3000000 10120191}; do
 export DLI_BENCH_POINTS="$population" DLI_ORACLE_POPULATION="$population"
 export DLI_SISAP_INPUTS="$campaign/data"
 if [[ "$population" != 10120191 ]]; then
   export DLI_SISAP_INPUTS="$campaign/oracle-$population"
   if [[ ! -f "$DLI_SISAP_INPUTS/oracle.json" ]]; then
     OPENBLAS_NUM_THREADS=2 python research/dli/sisap23_10m/checkpoint_oracle.py "$campaign/data" "$DLI_SISAP_INPUTS" "$population"
   fi
 fi
 # First progressive track: geometric. Other policies follow a validated 10M run.
 export DLI_SINGLE_SYSTEM="$system" DLI_BENCH_POLICY="$system"
 export DLI_SISAP_OUTPUT="$campaign/$system-$population"
 mkdir -p "$DLI_SISAP_OUTPUT"
 test ! -e "$DLI_SISAP_OUTPUT/run.log"
 export QDRANT_OPTIMIZATION_EVENTS="$DLI_SISAP_OUTPUT/native-events.jsonl"
 export QDRANT_LMI_BUILD_EVENTS="$DLI_SISAP_OUTPUT/build-events.jsonl"
 export DLI_QUERY_EVENTS="$DLI_SISAP_OUTPUT/query-events.jsonl"
 cargo test --release --locked --offline -p collection --lib --features 'testing,dli-research' -- --ignored --exact tests::lmi_lifecycle::dli_sisap_scaling_benchmark::sisap23_scaling_static_lmi_vs_temporal_dli --nocapture > "$DLI_SISAP_OUTPUT/run.log" 2>&1
 echo "complete population=$population"
done
