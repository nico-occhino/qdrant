#!/usr/bin/env bash
# Build-only native S.3 benchmark; data is an existing corpus.f32 plus dataset.json.
# Required: LMI_S3_DATA, LMI_S3_RUN_ROOT (new directory), LibTorch build environment.
set -euo pipefail
: "${LMI_S3_DATA:?existing dataset directory}"
: "${LMI_S3_RUN_ROOT:?new output directory under work/}"
test ! -e "$LMI_S3_RUN_ROOT"
mkdir -p "$LMI_S3_RUN_ROOT"
export LMI_S3_RUN_ROOT=$(realpath "$LMI_S3_RUN_ROOT")
export LMI_S3_DATA=$(realpath "$LMI_S3_DATA")
git rev-parse HEAD >"$LMI_S3_RUN_ROOT/revision.txt"
git diff >"$LMI_S3_RUN_ROOT/working-tree.patch"
rustc -Vv >"$LMI_S3_RUN_ROOT/rustc.txt"
uname -a >"$LMI_S3_RUN_ROOT/system.txt"
sha256sum "$LMI_S3_DATA/corpus.f32" "$LMI_S3_DATA/dataset.json" >"$LMI_S3_RUN_ROOT/input-hashes.txt"
if [ -z "${LMI_S3_BINARY:-}" ]; then
  cargo test -p segment --profile perf --features lmi-training --locked --lib phase_s3_build_benchmark --no-run --message-format=json >"$LMI_S3_RUN_ROOT/compiler.jsonl" 2>"$LMI_S3_RUN_ROOT/compiler.log"
  LMI_S3_BINARY=$(python3 -c 'import json,sys; a=[json.loads(x) for x in open(sys.argv[1])]; print(next(x["executable"] for x in reversed(a) if x.get("reason")=="compiler-artifact" and x.get("executable") and x["target"]["name"]=="segment"))' "$LMI_S3_RUN_ROOT/compiler.jsonl")
fi
export RUST_LOG=segment::index::lmi_index::build=info
for trial in $(seq 1 "${LMI_S3_TRIALS:-2}"); do
  export LMI_S3_OUTPUT="$LMI_S3_RUN_ROOT/trial-$trial"
  /usr/bin/time -v -o "$LMI_S3_RUN_ROOT/trial-$trial-time.txt" taskset -c "${LMI_S3_CPU:-0}" "$LMI_S3_BINARY" phase_s3_build_benchmark --ignored --nocapture --test-threads=1 >"$LMI_S3_RUN_ROOT/trial-$trial.log" 2>&1
done
