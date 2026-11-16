#!/usr/bin/env bash
set -euo pipefail
cd /home/nicoo/worktrees/qdrant-dli-cross-segment
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/cargo-target
export LIBTORCH_USE_PYTORCH=1
export PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export QDRANT_MEASURE_SCORING=1
export DLI_BENCH_HEAD="$(git rev-parse HEAD)"
study_root=${DLI_STUDY_ROOT:-/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-24}
for run in pilot-geometric full-size-tiered full-geometric; do
  export DLI_SISAP_OUTPUT="$study_root/$run"
  mkdir -p "$DLI_SISAP_OUTPUT"
  if [[ -e "$DLI_SISAP_OUTPUT/run.log" || -e "$DLI_SISAP_OUTPUT/native-events.jsonl" ]]; then
    echo "Existing run preserved: choose a new DLI_STUDY_ROOT" >&2
    exit 1
  fi
  export QDRANT_OPTIMIZATION_EVENTS="$DLI_SISAP_OUTPUT/native-events.jsonl"
  export DLI_BENCH_POLICY=geometric
  export DLI_BENCH_POINTS=300000
  export DLI_SISAP_INPUTS=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-22/benchmark-300k
  if [[ "$run" == pilot-geometric ]]; then
    export DLI_BENCH_POINTS=100000
    export DLI_SISAP_INPUTS=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-24/pilot-inputs
  fi
  if [[ "$run" == full-size-tiered ]]; then export DLI_BENCH_POLICY=size-tiered; fi
  env | sort | rg '^(DLI_|QDRANT_MEASURE|QDRANT_OPTIMIZATION|CARGO_|LIBTORCH)' > "$DLI_SISAP_OUTPUT/environment.txt"
  cargo test --release --locked --offline -p collection --lib --features 'testing,dli-research' -- --ignored --exact tests::lmi_lifecycle::dli_sisap300k_benchmark::sisap23_300k_static_lmi_vs_temporal_dli --nocapture > "$DLI_SISAP_OUTPUT/run.log" 2>&1
  echo "completed $run"
done
