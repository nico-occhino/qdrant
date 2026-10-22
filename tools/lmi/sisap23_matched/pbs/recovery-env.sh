#!/bin/bash
# Sourced by the independent opt-in recovery jobs. Never opens the incident collection.
set -euo pipefail
: "${PBS_JOBID:?PBS compute allocation required}"
: "${SCRATCHDIR:?PBS scratch required}"
export WORKSPACE="${WORKSPACE:-/storage/budejovice1/home/nico/lmi-metacentrum}"
export QDRANT_DIR="${QDRANT_DIR:-$WORKSPACE/qdrant-sisap23-index-topology-diagnostics}"
: "${SISAP23_REVIEWED_COMMIT:?exact reviewed commit required}"
: "${SISAP23_RECOVERY_ROOT:?new persistent recovery output required}"
export SISAP23_ORIGINAL_ROOT="${SISAP23_ORIGINAL_ROOT:-$WORKSPACE/results/sisap23-matched-recovered-20261016-01}"
test "$(git -C "$QDRANT_DIR" rev-parse HEAD)" = "$SISAP23_REVIEWED_COMMIT"
test "$(git -C "$QDRANT_DIR" branch --show-current)" = thesis/sisap23-index-topology-diagnostics
test -z "$(git -C "$QDRANT_DIR" status --porcelain)"
original_real=$(realpath -m "$SISAP23_ORIGINAL_ROOT")
recovery_real=$(realpath -m "$SISAP23_RECOVERY_ROOT")
case "$recovery_real/" in "$original_real/"*) echo 'Output overlaps incident' >&2; exit 2;; esac
export SISAP23_BENCH_ROOT="$SISAP23_RECOVERY_ROOT"
# All stdout/stderr, native resource logs and acceptance records are persistent.
mkdir -p "$SISAP23_RECOVERY_ROOT/job-logs"
set -o noclobber
exec > "$SISAP23_RECOVERY_ROOT/job-logs/$SISAP23_STAGE-$PBS_JOBID.log" 2>&1
module load "${PROTOBUF_MODULE:-protobuf}"
source "$QDRANT_DIR/tools/lmi/sisap23_matched/pbs/common-env.sh"
setup_benchmark_build_env
export QDRANT_LMI_SAMPLE_BUDGET_BYTES=768000000
export RUST_BACKTRACE=1 RUST_LOG=info
cd "$QDRANT_DIR"
args=(--original "$SISAP23_ORIGINAL_ROOT" --output "$SISAP23_RECOVERY_ROOT" --commit "$SISAP23_REVIEWED_COMMIT")
case "$SISAP23_STAGE" in
  recover) args+=(--source-variant "${SISAP23_SOURCE_VARIANT:-lmi}");;
  build) args+=(--variant "${SISAP23_VARIANT:?select lmi, hnsw-a or hnsw-b}" --data-generation "${SISAP23_DATA_GENERATION:?accepted data generation required}" --data-producer-commit "${SISAP23_DATA_PRODUCER_COMMIT:-$SISAP23_REVIEWED_COMMIT}");;
  evaluate) args+=(--variant "${SISAP23_VARIANT:?variant required}" --index-generation "${SISAP23_INDEX_GENERATION:?accepted index generation required}" --campaign "${SISAP23_CAMPAIGN:?campaign required}");;
  init|validate) args+=(--campaign "${SISAP23_CAMPAIGN:?campaign required}");;
  probe) ;;
  *) echo 'Unsupported recovery stage' >&2; exit 2;;
esac
export PYTHONUNBUFFERED=1
python tools/lmi/sisap23_matched/recovery_stages.py "$SISAP23_STAGE" "${args[@]}"
