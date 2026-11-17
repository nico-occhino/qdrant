#!/bin/bash
# Retained10M queries, then one bounded private64K training diagnostic. No ingestion.
set -eu
out=${1:?fresh output directory}
plan=${2:?validation-locked native plan}
teacher=${3:?prepared private subset directory}
test ! -e "$out"; mkdir -p "$out"
export PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH
export CARGO_TARGET_DIR=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/cargo-target
export CARGO_BUILD_JOBS=2 LIBTORCH_USE_PYTORCH=1
unset LMI_TEACHER_DIAGNOSTIC
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export DLI_REOPEN_COLLECTION_ROOT=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-25/geometric-10120191/collection
export DLI_SISAP_INPUTS=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-25/data
export DLI_SISAP_OUTPUT="$out" DLI_FRONTIER_PLAN="$plan"
export DLI_FRONTIER_QUERY_COUNT=${DLI_FRONTIER_QUERY_COUNT:-12}
export DLI_QUERY_EVENTS="$out/query-events.jsonl" DLI_ROUTING_EVENTS="$out/routing-events.jsonl"
export DLI_NATIVE_ACCESS_EVENTS="$out/access-events.jsonl" DLI_DETAILED_QUERY_STATS=1 QDRANT_MEASURE_SCORING=1
export QDRANT_OPTIMIZATION_EVENTS="$out/unexpected-optimizations.jsonl"
git rev-parse HEAD > "$out/source-head.txt"; git diff --binary > "$out/source.patch"
getconf CLK_TCK > "$out/clock-ticks-per-second.txt"; rustc --version > "$out/rustc.txt"
sha256sum "$DLI_REOPEN_COLLECTION_ROOT/storage/0/dli_hierarchy.json" "$DLI_REOPEN_COLLECTION_ROOT/storage/config.json" > "$out/topology-before.sha256"
python - "$out" <<'PY'
import os,json,hashlib,h5py,numpy as np,sys,torch
from pathlib import Path
out=Path(sys.argv[1]);base=Path(os.environ['DLI_SISAP_INPUTS']);checks={}
for name,key,raw,dtype,shape in [('public-queries-10k-clip768v2.h5','emb','queries.f32','<f4',(10000,768)),('laion2B-en-public-gold-standard-v2-10M-F64-IEEE754.h5','knns','ground_truth.i32','<i4',(10000,1000))]:
 source=Path('/mnt/c/datasets/sisap2023')/name;data=np.memmap(base/raw,mode='r',dtype=dtype,shape=shape)
 with h5py.File(source) as f:assert np.array_equal(f[key][400:700],data[400:700])
 checks[name]={'source_md5':hashlib.md5(source.read_bytes()).hexdigest(),'raw_sha256':hashlib.sha256((base/raw).read_bytes()).hexdigest(),'rows400_700_equal':True}
checks.update(torch=torch.__version__,profile='release',launcher_pid_in_namespace=os.getppid(),pid_namespace=os.readlink('/proc/self/ns/pid'),uid=os.getuid(),independent_persistence_verified=False,persistence_limitation='sandbox denies user-systemd bus; managed foreground only',native_plan_sha256=hashlib.sha256(Path(os.environ['DLI_FRONTIER_PLAN']).read_bytes()).hexdigest(),query_count=int(os.environ['DLI_FRONTIER_QUERY_COUNT']))
(out/'manifest.json').write_text(json.dumps(checks,indent=2))
PY
env -u DLI_QUERY_EVENTS -u DLI_ROUTING_EVENTS -u DLI_NATIVE_ACCESS_EVENTS -u QDRANT_OPTIMIZATION_EVENTS -u DLI_DETAILED_QUERY_STATS cargo test --release --locked --offline -p collection --lib --features 'testing,lmi-training,dli-research' dli_ -- --nocapture > "$out/release-safety.log" 2>&1
timeout 1800s cargo test --release --locked --offline -p collection --lib --features 'testing,lmi-training,dli-research' tests::lmi_lifecycle::dli_sisap_scaling_benchmark::dli_frozen_recall_frontier -- --ignored --exact --nocapture > "$out/run.log" 2>&1 &
native_pid=$!
printf '{"launcher_pid":%s,"native_timeout_pid":%s,"namespace":"sandbox local PID values; not host PIDs","independent_persistence_verified":false}\n' "$$" "$native_pid" > "$out/process.json"
ps -o pid,ppid,sid,pgid,stat,args -p "$$","$native_pid" > "$out/process-ownership.txt"
wait "$native_pid"
test -s "$out/frontier.json"; sha256sum -c "$out/topology-before.sha256" > "$out/topology-after.txt"; test ! -s "$out/unexpected-optimizations.jsonl"
python research/dli/search_optimization/analyze_recall_frontier.py "$out" > "$out/analysis.log" 2>&1
# The private training process runs only after query measurements finish.
export DLI_TEACHER_CONTROL="$teacher" LMI_TEACHER_DIAGNOSTIC="$teacher/teacher.json"
unset DLI_QUERY_EVENTS DLI_ROUTING_EVENTS DLI_NATIVE_ACCESS_EVENTS DLI_DETAILED_QUERY_STATS QDRANT_OPTIMIZATION_EVENTS
export LMI_BUILD_EVENTS="$teacher/build-events.jsonl"
timeout 600s cargo test --release --locked --offline -p collection --lib --features 'testing,lmi-training,dli-research' tests::lmi_lifecycle::dli_sisap_scaling_benchmark::dli_small_teacher_router_control -- --ignored --exact --nocapture > "$teacher/build.log" 2>&1
test -s "$teacher/index.json"
python research/dli/search_optimization/analyze_teacher_control.py "$teacher" > "$teacher/analysis.log" 2>&1

/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python research/dli/search_optimization/plot_recall_frontier.py "$(dirname "$plan")" "$out" "$teacher"
