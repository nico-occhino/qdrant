#!/bin/bash
# Bounded frozen-topology control only. Never ingests or trains a model.
set -eu
output=${1:?output directory required}
mode=${2:-compare}
count=${3:-3}
start=${4:-0}
test ! -e "$output/rounds.json"
mkdir -p "$output"
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-${ADMIN:?}/experiments/2026-11-12-dli-architecture/cargo-target}
export LIBTORCH_USE_PYTORCH=1
export PATH=${CONDA_ENV:?}/lmi-rust-inspect/bin:$PATH
export LD_LIBRARY_PATH=${CONDA_ENV:?}/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export DLI_REOPEN_COLLECTION_ROOT=${ADMIN:?}/experiments/2026-11-12-dli-architecture/continuation-25/geometric-10120191/collection
export DLI_SISAP_INPUTS=${ADMIN:?}/experiments/2026-11-12-dli-architecture/continuation-25/data
export DLI_SISAP_OUTPUT="$output"
export DLI_ORACLE_POPULATION=10120191
export DLI_PROFILE_QUERY_COUNT="$count"
export DLI_PROFILE_QUERY_START="$start"
export DLI_PROFILE_TOPK=0 DLI_PROFILE_RESERVED=0 DLI_PROFILE_COMPARE_TOPK=0
case "$mode" in
    full) ;;
    topk) export DLI_PROFILE_TOPK=1 ;;
    reserved) export DLI_PROFILE_TOPK=1 DLI_PROFILE_RESERVED=1 ;;
    compare) export DLI_PROFILE_COMPARE_TOPK=1 ;;
    *) echo 'mode must be full, topk, reserved or compare' >&2; exit 2 ;;
esac
export QDRANT_MEASURE_SCORING=1
export DLI_NATIVE_ACCESS_EVENTS="$output/access-events.jsonl"
export DLI_QUERY_EVENTS="$output/query-events.jsonl"
git rev-parse HEAD > "$output/source-head.txt"
git diff --binary > "$output/source.patch"
rustc --version > "$output/rustc.txt"
getconf CLK_TCK > "$output/clock-ticks-per-second.txt"
# Debug mechanism control. Do not report this as optimized benchmark latency.
timeout 600s cargo test --locked --offline -p collection --lib --features 'testing,lmi-training,dli-research' dli_frozen_scaling_reopen_profile -- --ignored --nocapture > "$output/run.log" 2>&1
