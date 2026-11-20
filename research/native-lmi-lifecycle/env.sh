#!/usr/bin/env bash
# Source from the isolated worktree; two concurrent compiler jobs.
export PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH
export LIBTORCH_USE_PYTORCH=1 CXX=g++ CXXFLAGS=-g0
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=/home/nicoo/work/qdrant-native-lmi-lifecycle-20261109-target
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
