#!/usr/bin/env bash
set -euo pipefail
cd /home/nicoo/work/qdrant
root=${LMI_S3B1_RUN_ROOT:-work/phase_s3/s3b1-continuation}
binary=$(python3 -c 'import json,sys; a=[json.loads(x) for x in open(sys.argv[1])]; print(next(x["executable"] for x in reversed(a) if x.get("reason")=="compiler-artifact" and x.get("executable") and x["target"]["name"]=="segment"))' "$root/compiler-e.jsonl")
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch/lib
export TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
for trial in 1 2 3; do
  export LMI_S3B1_MICRO_OUTPUT=$(realpath "$root")/micro-e-$trial.json
  test ! -e "$LMI_S3B1_MICRO_OUTPUT"
  /usr/bin/time -v -o "$root/micro-e-$trial-time.txt" taskset -c 0 "$binary" phase_s3b1_router_microbenchmark --ignored --nocapture --test-threads=1 > "$root/micro-e-$trial.log" 2>&1
done
export LMI_S3B1_PROFILE_OUTPUT=$(realpath "$root")/profile-d.json
test ! -e "$LMI_S3B1_PROFILE_OUTPUT"
/usr/bin/time -v -o "$root/profile-d-time.txt" taskset -c 0 "$binary" phase_s3b1_large_b_stage_profile --ignored --nocapture --test-threads=1 > "$root/profile-d.log" 2>&1
