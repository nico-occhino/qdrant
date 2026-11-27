#!/bin/bash
# Q5 gates 2 and 3 (Q5_PREREGISTRATION.md §6) on one machine.
#   run_local_gates.sh BIN_BASE BIN_NEW PLAN_DIR CORPUS OUT_DIR [BATCHES]
# Gate 2: E1 with the base binary and with the new binary, on the first BATCHES (default 3) batches of the plan,
# the mq_val panel and the 17-point grid. Compared by compare_runs.py.
# Gate 3: E1T with the new binary on the same prefix. Its decisions are checked against its builds
# (analyze_q5.check_decisions, with the E1 run's first fits).
set -euo pipefail
BASE=$1 NEW=$2 PLAN=$3 CORPUS=$4 OUT=$5 NB=${6:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
BUDGETS=1000,2000,3500,5000,7000,10000,14000,20000,28000,40000,50000,56000,80000,113000,160000,226000,320000
mkdir "$OUT"   # never reuse
run() {  # label binary arm
  local R="$OUT/$1"; mkdir -p "$R/scratch"
  env DLI_ARM=$3 DLI_P3_VARIANT=none DLI_P3_SYSTEM=dli DLI_P3_PLAN="$PLAN" DLI_P3_CORPUS="$CORPUS" \
      DLI_P3_OUTPUT="$R/out" DLI_P3_SCRATCH="$R/scratch" DLI_P3_SETS=mq_val DLI_P3_STATIC_BATCHES= \
      DLI_P3_MAX_BATCHES=$NB DLI_FRONTIER_BUDGETS=$BUDGETS DLI_FRONTIER_POLICIES=mdn_code \
      DLI_BENCH_HEAD=gate QDRANT_DLI_BUILD_CPUS=8 DLI_BENCH_PUBLICATION_TIMEOUT_S=14400 \
      QDRANT_LMI_SAMPLE_BUDGET_BYTES=2000000000 \
      QDRANT_OPTIMIZATION_EVENTS="$R/out/native-events.jsonl" QDRANT_LMI_BUILD_EVENTS="$R/out/build-events.jsonl" \
      "$2" --ignored --exact tests::lmi_lifecycle::dli_sisap_scaling_benchmark::dli_phase3_workload --nocapture \
      > "$R/run.log" 2>&1
  rm -rf "$R/scratch"
}
mkdir -p "$OUT"/{e1-base,e1-new,e1t-new}/out
run e1-base "$BASE" E1
run e1-new "$NEW" E1
python3 -I "$HERE/compare_runs.py" "$OUT/e1-base/out" "$OUT/e1-new/out" --out "$OUT/gate2.json" || echo "GATE2 FAILED"
run e1t-new "$NEW" E1T
python3 -I - "$HERE/.." "$OUT" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
import analyze_q5 as Q
out = sys.argv[2]
first = sum(b["trained"] for b in Q.trainer_seconds(f"{out}/e1-new/out"))
decs = Q.decisions(f"{out}/e1t-new/out")
chk = Q.check_decisions(decs, Q.trainer_seconds(f"{out}/e1t-new/out"), first)
chk["max_ratio_eligible"] = max((d["max_over_mean"] for d in decs if d["target_level"] >= 2), default=None)
chk["fired"] = [d for d in decs if d["retrain"]]
json.dump(chk, open(f"{out}/gate3.json", "w"), indent=1)
print(json.dumps({k: chk[k] for k in ("decisions", "retrain_decisions", "trained_builds", "first_fits",
                                       "max_ratio_eligible", "problems", "pass")}))
PY
