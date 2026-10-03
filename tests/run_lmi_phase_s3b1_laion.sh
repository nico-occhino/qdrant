#!/usr/bin/env bash
set -euo pipefail
cd /home/nicoo/work/qdrant
root=${LMI_S3B1_RUN_ROOT:-work/phase_s3/s3b1-continuation}
binary=$(python3 -c 'import json,sys; a=[json.loads(x) for x in open(sys.argv[1])]; print(next(x["executable"] for x in reversed(a) if x.get("reason")=="compiler-artifact" and x.get("executable") and x["target"]["name"]=="segment"))' "$root/compiler-e.jsonl")
export LMI_S3_DATA=/home/nicoo/work/lmi-phase-f-data
export RUST_LOG=segment::index::lmi_index::build=info
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch/lib
export TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
for k in 1 16 256; do
  config="$root/laion-k${k}-config.json"
  test ! -e "$config"
  sed "s/\"routing_batch_size\": 16/\"routing_batch_size\": $k/" tests/lmi_phase_s3b1_config.json > "$config"
  export LMI_S3_CONFIG=$(realpath "$config")
  export LMI_S3_OUTPUT=$(realpath "$root")/laion-k${k}
  test ! -e "$LMI_S3_OUTPUT"
  /usr/bin/time -v -o "$root/laion-k${k}-time.txt" taskset -c 0 "$binary" phase_s3_build_benchmark --ignored --nocapture --test-threads=1 > "$root/laion-k${k}.log" 2>&1
done
