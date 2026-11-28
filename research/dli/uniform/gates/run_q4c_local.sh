#!/bin/bash
# Q4 arms (c) and (c′) (Q4_PREREGISTRATION.md §7, Amendment A1): replays of (b) and (a′) whose snapshots 10–12 are
# also evaluated with PostgreSQL's level routers, into batch_<t>/import. Serial, one machine, one binary.
#   run_q4c_local.sh BIN PLAN_DIR CORPUS EXPORT_DIR OUT_DIR
set -euo pipefail
BIN=$1 PLAN=$2 CORPUS=$3 EXPORT=$4 OUT=$5
BUDGETS=1000,2000,3500,5000,7000,10000,14000,20000,28000,40000,50000,56000,80000,113000,160000,226000,320000,2000000
( cd "$EXPORT" && sha256sum -c --quiet SHA256SUMS )
mkdir "$OUT"   # never reuse
sha256sum "$BIN" > "$OUT/binary.sha256"
for ARM in b a1; do
  R="$OUT/$ARM"; mkdir -p "$R/scratch" "$R/out"
  if [[ $ARM = b ]]; then ALIGN=(DLI_P3_DRAIN_ALIGN=global); SCHED=postgres_schedule; else ALIGN=(); SCHED=qdrant_schedule; fi
  env "${ALIGN[@]}" DLI_P3_IMPORT_ROUTERS="$EXPORT/$SCHED" \
      DLI_ARM=E1 DLI_P3_VARIANT=none DLI_P3_SYSTEM=dli DLI_P3_PLAN="$PLAN" DLI_P3_CORPUS="$CORPUS" \
      DLI_P3_OUTPUT="$R/out" DLI_P3_SCRATCH="$R/scratch" DLI_P3_SETS=mq_val DLI_P3_STATIC_BATCHES= \
      DLI_P3_MAX_BATCHES=13 DLI_P3_EVAL=10,11,12 DLI_FRONTIER_BUDGETS=$BUDGETS DLI_FRONTIER_POLICIES=mdn_code \
      DLI_BENCH_HEAD=q4c-local QDRANT_DLI_BUILD_CPUS=8 DLI_BENCH_PUBLICATION_TIMEOUT_S=14400 \
      QDRANT_LMI_SAMPLE_BUDGET_BYTES=2000000000 \
      QDRANT_OPTIMIZATION_EVENTS="$R/out/native-events.jsonl" QDRANT_LMI_BUILD_EVENTS="$R/out/build-events.jsonl" \
      "$BIN" --ignored --exact tests::lmi_lifecycle::dli_sisap_scaling_benchmark::dli_phase3_workload --nocapture \
      > "$R/run.log" 2>&1
  rm -rf "$R/scratch"
  echo "ARM_DONE $ARM $(date -u +%FT%TZ)"
done
