#!/usr/bin/env bash
set -euo pipefail
cd /home/nicoo/work/qdrant
export PATH=/home/nicoo/.cargo/bin:/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export CXX=clang++ CXXFLAGS=-g0 CARGO_BUILD_JOBS=2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LMI_PHASE_F4_DIR=/home/nicoo/work/lmi-phase-f4-data
stage=/mnt/c/Users/nicoo/Documents/Codex/2026-09-21/referenced-chatgpt-conversation-this-is-an/work/phase_f4
python "$stage/retry_selection.py"
cp "$stage/evaluation_f4.rs" lib/segment/src/index/lmi_index/evaluation_f4.rs
cp "$stage/analyze.py" tests/lmi_phase_f4_analyze.py
mkdir -p "$LMI_PHASE_F4_DIR/verification"
rustfmt --edition 2024 --config skip_children=true lib/segment/src/index/lmi_index/evaluation_f4.rs
cargo test --release -p segment --features lmi-training --locked --lib phase_f4_oracle_and_objective_control -- --ignored --nocapture --test-threads=1 2>&1 | tee "$LMI_PHASE_F4_DIR/verification/native-run.log"
python tests/lmi_phase_f4_analyze.py > "$LMI_PHASE_F4_DIR/verification/analysis.log"
cp "$LMI_PHASE_F4_DIR/Phase_F4_Report.md" docs/lmi-phase-f4.md
git diff --check
