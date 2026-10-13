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
  local metacentrum_rust_home=/storage/brno2/home/nico
  ENV_DIR="$WORKSPACE/envs/py312-torch251"
  test -x "$ENV_DIR/bin/python" || {
      echo "Missing Python executable: $ENV_DIR/bin/python" >&2
      exit 2
  }
  export PATH="$ENV_DIR/bin:$PATH"
  export RUSTUP_HOME="$metacentrum_rust_home/.rustup"
  export CARGO_HOME="$metacentrum_rust_home/.cargo"
  export PATH="$CARGO_HOME/bin:$PATH"
  export TMPDIR="${SCRATCHDIR:?PBS scratch allocation required}/tmp"
  export CARGO_TARGET_DIR="${SCRATCHDIR:?PBS scratch allocation required}/cargo-target"
  mkdir -p "$TMPDIR" "$CARGO_TARGET_DIR" "$(dirname "$tool_log")"
  local rustv pytorch torch_root torch_lib_dir protocv cxx_path
  rustv=$(rustc --version)
  [[ "$rustv" == "rustc 1.98.0 "* ]] || { echo "expected Rust 1.98.0, found $rustv" >&2; exit 2; }
  python -c "import sys; assert sys.version_info[:2] == (3, 12), f'expected Python 3.12, found {sys.version}'" || exit 2
  python -c "import torch; assert torch.__version__.split('+')[0] == '2.5.1' and torch.version.cuda is None, f'expected PyTorch 2.5.1 CPU, found {torch.__version__}'" || exit 2
  pytorch=$(python -c "import torch; print(torch.__version__)")
  torch_root=$(python -c "import pathlib,torch; print(pathlib.Path(torch.__file__).parent)")
  torch_lib_dir="$torch_root/lib"
  test -d "$torch_lib_dir" || { echo "PyTorch library directory missing: $torch_lib_dir" >&2; exit 2; }
  unset LIBTORCH LIBTORCH_BYPASS_VERSION_CHECK
  export LIBTORCH_USE_PYTORCH=1
  export LD_LIBRARY_PATH="$torch_lib_dir:${LD_LIBRARY_PATH:-}"
  export CXX=g++ CXXFLAGS=-g0
  cxx_path=$(command -v "$CXX") || { echo "g++ is required by the MetaCentrum build setup" >&2; exit 2; }
  protocv=$(protoc --version) || { echo "protoc is required; load the site protobuf module" >&2; exit 2; }
  "$CXX" --version > "$TMPDIR/cxx-version.txt"
  printf 'int main() { return 0; }\n' | "$CXX" -x c++ - -o "$TMPDIR/cxx-probe"
  "$TMPDIR/cxx-probe"
  {
    printf 'reviewed_commit=%s\n' "$SISAP23_REVIEWED_COMMIT"
    printf 'rustup_home=%s\ncargo_home=%s\n' "$RUSTUP_HOME" "$CARGO_HOME"
    printf 'rust=%s\n' "$rustv"
    printf 'cargo=%s\n' "$(cargo --version)"
    printf 'python=%s\n' "$(python --version 2>&1)"
    printf 'pytorch=%s\n' "$pytorch"
    printf 'libtorch_use_pytorch=%s\ntorch_lib_dir=%s\n' "$LIBTORCH_USE_PYTORCH" "$torch_lib_dir"
    printf 'protoc=%s\n' "$protocv"
    printf 'cxx=%s (%s)\n' "$CXX" "$cxx_path"
    cat "$TMPDIR/cxx-version.txt"
    printf 'cargo_target_dir=%s\ntmpdir=%s\n' "$CARGO_TARGET_DIR" "$TMPDIR"
  } > "$tool_log"
}
