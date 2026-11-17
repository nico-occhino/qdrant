#!/bin/bash
# No ingestion/training. Run from the existing DLI worktree.
set -eu
out=${1:?fresh output directory required}
test ! -e "$out"
mkdir -p "$out"
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/cargo-target
export LIBTORCH_USE_PYTORCH=1
export PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/bin:$PATH
export LD_LIBRARY_PATH=/home/nicoo/miniconda3/envs/lmi-rust-inspect/lib/python3.11/site-packages/torch/lib
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export DLI_REOPEN_COLLECTION_ROOT=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-25/geometric-10120191/collection
export DLI_SISAP_INPUTS=/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-25/data
export DLI_SISAP_OUTPUT="$out"
export DLI_STUDY_VALIDATION_COUNT=${DLI_STUDY_VALIDATION_COUNT:-12}
export DLI_STUDY_TEST_COUNT=${DLI_STUDY_TEST_COUNT:-24}
export DLI_QUERY_EVENTS="$out/query-events.jsonl"
export DLI_ROUTING_EVENTS="$out/routing-events.jsonl"
export DLI_NATIVE_ACCESS_EVENTS="$out/access-events.jsonl"
export DLI_DETAILED_QUERY_STATS=1 QDRANT_MEASURE_SCORING=1
export QDRANT_OPTIMIZATION_EVENTS="$out/unexpected-optimizations.jsonl"
git rev-parse HEAD > "$out/source-head.txt"
git diff --binary > "$out/source.patch"
rustc --version > "$out/rustc.txt"
getconf CLK_TCK > "$out/clock-ticks-per-second.txt"
uname -a > "$out/uname.txt"
cp "$DLI_SISAP_INPUTS/manifest.json" "$out/dataset-manifest.json"
sha256sum "$DLI_REOPEN_COLLECTION_ROOT/storage/0/dli_hierarchy.json" "$DLI_REOPEN_COLLECTION_ROOT/storage/config.json" > "$out/topology-before.sha256"
python - "$out" <<'PY'
import hashlib,h5py,json,numpy as np,pathlib,sys,torch
out=pathlib.Path(sys.argv[1]);base=pathlib.Path('/home/nicoo/research-admin/experiments/2026-11-12-dli-architecture/continuation-25/data')
check={}
for name,key,raw,dtype,shape in [('public-queries-10k-clip768v2.h5','emb','queries.f32','<f4',(10000,768)),('laion2B-en-public-gold-standard-v2-10M-F64-IEEE754.h5','knns','ground_truth.i32','<i4',(10000,1000))]:
 source=pathlib.Path('/mnt/c/datasets/sisap2023')/name
 arr=np.memmap(base/raw,mode='r',dtype=dtype,shape=shape)
 with h5py.File(source) as f:assert np.array_equal(f[key][100:300],arr[100:300])
 check[name]={'rows':[100,300],'raw_source_equal':True,'source_md5':hashlib.md5(source.read_bytes()).hexdigest(),'raw_sha256':hashlib.sha256((base/raw).read_bytes()).hexdigest()}
check['torch']=torch.__version__;check['profile']='release';check['validation_count']=int(__import__('os').environ['DLI_STUDY_VALIDATION_COUNT']);check['test_count']=int(__import__('os').environ['DLI_STUDY_TEST_COUNT']);check['independent_survival']='not guaranteed; sandbox denies user-systemd bus; foreground managed execution'
(out/'manifest.json').write_text(json.dumps(check,indent=2))
PY
env -u DLI_QUERY_EVENTS -u DLI_ROUTING_EVENTS -u DLI_NATIVE_ACCESS_EVENTS -u QDRANT_OPTIMIZATION_EVENTS -u DLI_DETAILED_QUERY_STATS cargo test --release --locked --offline -p collection --lib --features 'testing,lmi-training,dli-research' dli_ -- --nocapture > "$out/release-safety.log" 2>&1
timeout 1800s cargo test --release --locked --offline -p collection --lib --features 'testing,lmi-training,dli-research' tests::lmi_lifecycle::dli_sisap_scaling_benchmark::dli_frozen_matched_work_study -- --ignored --exact --nocapture > "$out/run.log" 2>&1
test -f "$out/study.json"
sha256sum -c "$out/topology-before.sha256" > "$out/topology-after-check.txt"
test ! -s "$out/unexpected-optimizations.jsonl"
python research/dli/search_optimization/analyze_matched_study.py "$out" > "$out/analysis.log" 2>&1
/home/nicoo/miniconda3/envs/lmi-starterpack/bin/python research/dli/search_optimization/plot_matched_study.py "$out"
