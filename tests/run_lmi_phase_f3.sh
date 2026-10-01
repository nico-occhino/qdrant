#!/usr/bin/env bash
# Reproduction only: explicitly choose a NEW output directory.
set -euo pipefail
: "${LMI_PHASE_F3_DIR:?Set a new, nonexistent F.3 output directory}"
cd /home/nicoo/work/qdrant
export PATH=/home/nicoo/.cargo/bin:/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export CXX=clang++ CXXFLAGS=-g0 CARGO_BUILD_JOBS=2
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tests/lmi_phase_f3_prepare.py
mkdir -p "$LMI_PHASE_F3_DIR/verification"
cargo test --release -p segment --features lmi-training --locked --lib phase_f3_study -- --ignored --nocapture --test-threads=1 2>&1 | tee "$LMI_PHASE_F3_DIR/verification/native-run.log"
python tests/lmi_phase_f3_analyze.py
git diff --check
