#!/bin/bash
set -euo pipefail
verify_benchmark_checkout() {
  local expected actual branch
  expected=${SISAP23_REVIEWED_COMMIT:?pass the reviewed benchmark commit SHA}
  branch=$(git -C "$QDRANT_DIR" branch --show-current)
  actual=$(git -C "$QDRANT_DIR" rev-parse HEAD)
  test "$branch" = "thesis/sisap23-matched-bench" || { echo "wrong source branch: $branch" >&2; exit 2; }
  test "$actual" = "$expected" || { echo "source commit mismatch: expected $expected, found $actual" >&2; exit 2; }
}
setup_benchmark_build_env() {
  local tool_log="$SISAP23_BENCH_ROOT/provenance/toolchain-${PBS_JOBID:-manual}.txt"
  source "$WORKSPACE/envs/py312-torch251/bin/activate"
  source /storage/brno2/home/nico/.cargo/env
  export PATH="/storage/brno2/home/nico/.cargo/bin:$PATH"
  export TMPDIR="${SCRATCHDIR:?PBS scratch allocation required}/tmp"
  export CARGO_TARGET_DIR="${SCRATCHDIR:?PBS scratch allocation required}/cargo-target"
  mkdir -p "$TMPDIR" "$CARGO_TARGET_DIR" "$(dirname "$tool_log")"
  local rustv pytorch torch_root protocv cxx_path
  rustv=$(rustc --version)
  [[ "$rustv" == "rustc 1.98.0 "* ]] || { echo "expected Rust 1.98.0, found $rustv" >&2; exit 2; }
  python -c "import sys; assert sys.version_info[:2] == (3, 12), f'expected Python 3.12, found {sys.version}'" || exit 2
  python -c "import torch; assert torch.__version__.split('+')[0] == '2.5.1', f'expected PyTorch 2.5.1, found {torch.__version__}'" || exit 2
  pytorch=$(python -c "import torch; print(torch.__version__)")
  torch_root=$(python -c "import pathlib,torch; print(pathlib.Path(torch.__file__).parent)")
  export LIBTORCH="$torch_root"
  export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
  export CXX=clang++ CXXFLAGS=-g0
  cxx_path=$(command -v "$CXX") || { echo "clang++ is required by the selected torch-sys setup" >&2; exit 2; }
  protocv=$(protoc --version) || { echo "protoc is required; load the site protobuf module" >&2; exit 2; }
  "$CXX" --version > "$TMPDIR/cxx-version.txt"
  printf 'int main() { return 0; }\n' | "$CXX" -x c++ - -o "$TMPDIR/cxx-probe"
  "$TMPDIR/cxx-probe"
  {
    printf 'reviewed_commit=%s\n' "$SISAP23_REVIEWED_COMMIT"
    printf 'rust=%s\n' "$rustv"
    printf 'cargo=%s\n' "$(cargo --version)"
    printf 'python=%s\n' "$(python --version 2>&1)"
    printf 'pytorch=%s\n' "$pytorch"
    printf 'libtorch=%s\n' "$LIBTORCH"
    printf 'protoc=%s\n' "$protocv"
    printf 'cxx=%s (%s)\n' "$CXX" "$cxx_path"
    cat "$TMPDIR/cxx-version.txt"
    printf 'cargo_target_dir=%s\ntmpdir=%s\n' "$CARGO_TARGET_DIR" "$TMPDIR"
  } > "$tool_log"
}
