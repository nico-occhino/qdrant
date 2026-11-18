#!/bin/bash
# Submit Tier 2 jobs through the existing ControlMaster (no new SSH master is created).
# Usage: submit.sh preflight        -> submits the 1M preflight only
#        submit.sh tier2            -> submits T2-F1, T2-F23, T2-F4 (only after preflight COMPLETE)
# Records every qsub output and the qstat line under the local experiment directory.
set -euo pipefail
SOCK="$HOME/.ssh/metacentrum-control"; TARGET=nico@perian.metacentrum.cz
ssh -F /dev/null -S "$SOCK" -O check "$TARGET"
S=(ssh -F /dev/null -S "$SOCK" -o BatchMode=yes "$TARGET")
ROOT=/storage/budejovice1/home/nico/lmi-metacentrum/static-lmi-ivf-20261118
E=/home/nicoo/research-admin/experiments/2026-11-18-static-lmi-ivf; mkdir -p "$E/hpc-submissions"
sub() { # name ncpus mem walltime N lab-args
  local name=$1 ncpu=$2 mem=$3 wall=$4 n=$5 args=$6
  local cmd="cd $ROOT && qsub -N $name -l select=1:ncpus=$ncpu:mem=${mem}gb:scratch_ssd=40gb:cpu_vendor=amd -l walltime=$wall -v CFG_NAME=$name,N=$n,LAB_ARGS='$args' lab-10m.pbs"
  local id; id=$("${S[@]}" "$cmd")
  echo "$(date -Is) $id $name ncpus=$ncpu mem=${mem}gb walltime=$wall N=$n args=[$args]" | tee -a "$E/hpc-submissions/submitted.log"
  "${S[@]}" "qstat -x $id" | tee -a "$E/hpc-submissions/submitted.log"
}
case "${1:?preflight|tier2}" in
  preflight)
    sub T1-preflight 8 16 00:45:00 1000000 "--B 256 --teacher native --epochs 2" ;;
  tier2)
    ok=$("${S[@]}" "ls $ROOT/results/T1-preflight-*/COMPLETE.json 2>/dev/null | head -1")
    [[ -n "$ok" ]] || { echo "preflight has not completed; refusing" >&2; exit 1; }
    sub T2-F1 16 48 04:00:00 10120191 "--B 3162 --teacher native --sample 250000 --iters 5 --hidden 512 --epochs 30"
    sub T2-F23 16 48 08:00:00 10120191 "--B 8192 --teacher converged --sample 524288 --iters 20 --hidden 512 --epochs 30"
    sub T2-F4 16 48 04:00:00 10120191 "--B 8192 --teacher native --sample 262144 --iters 5 --hidden 0 --epochs 30 --partition mlp" ;;  # router default mlp,centroid; commas break qsub -v
esac
