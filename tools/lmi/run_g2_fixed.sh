#!/usr/bin/env bash
set -euo pipefail

cd /home/nicoo/work/qdrant-upstream-sync
root=work/upstream_sync/sisap10m/g2_119
binary=target/release/deps/collection-b836457245be0da5
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-starterpack/lib/python3.12/site-packages/torch/lib

test -f "$root/fixed/p0004/summary.json"

for p in 1 2 8 10 13 16 30 32 40 64 100 128 160; do
    label=$(printf '%04d' "$p")
    test ! -e "$root/fixed/p$label"
    echo "G2 fixed condition P=$p begins $(date -u --iso-8601=seconds)"
    LMI_G2_P="$p" "$binary" g2_fixed_persisted_state --ignored --nocapture \
        2>&1 | tee "$root/p$label.log"
    test -f "$root/fixed/p$label/summary.json"
    echo "G2 fixed condition P=$p ends $(date -u --iso-8601=seconds)"
done
