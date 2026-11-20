# Shared staging for supervised-routing PBS jobs (sourced). Requires: OUT, IN, SCRATCHDIR, PBS_JOBID.
set -euo pipefail
CAMPAIGN=/storage/budejovice1/home/nico/lmi-metacentrum/static-supervised-routing-20261119
CORPUS=/storage/budejovice1/home/nico/lmi-metacentrum/results/sisap23-matched-final-20261028-01/corpus.f16le
CORPUS_SHA=f5868d6e002a6e73aea47491ee040a70a151406fe9e4552292428e2bfe0616e5
EXPECTED_COMMIT=$(cat "$IN/native.commit")
W="$SCRATCHDIR/work"

finish() {
  rc=$?
  if [[ $rc -ne 0 ]]; then
    printf '{"job":"%s","phase":"%s","exit":%d,"utc":"%s"}\n' "$PBS_JOBID" "${phase:-?}" "$rc" "$(date -u +%FT%TZ)" > "$OUT/FAILED.json"
    for f in "$W"/*.log "$W"/sr/*/*.json; do [[ -f "$f" ]] && cp -n "$f" "$OUT/partial-$(basename "$f")" || true; done
  fi
  rm -rf "$W"
  exit $rc
}
trap finish EXIT

stage() {
  echo "START $(date -u +%FT%TZ) host=$(hostname) job=$PBS_JOBID commit=$EXPECTED_COMMIT"
  { lscpu; free -g; df -h "$SCRATCHDIR"; } > "$OUT/hardware.txt"
  phase=verify_inputs
  ( cd "$IN" && sha256sum -c inputs.sha256 )
  phase=environment
  source /storage/budejovice1/home/nico/lmi-metacentrum/development-search-timeout-24900597-20261026/sisap23-build-environment-v1.sh
  sisap23_export_build_environment
  sisap23_load_build_modules
  sisap23_check_build_environment | tee "$OUT/toolchain.txt"
  PY="$SISAP23_PYTHON_ENV/bin/python"
  "$PY" -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)" | tee -a "$OUT/toolchain.txt"
  phase=stage
  git() { command git -c filter.lfs.smudge= -c filter.lfs.clean= -c filter.lfs.process= -c filter.lfs.required=false "$@"; }
  export GIT_LFS_SKIP_SMUDGE=1
  mkdir -p "$W/data" "$W/sr"
  git clone --quiet --no-checkout "$IN/source.bundle" "$W/qdrant"
  git -C "$W/qdrant" checkout --quiet --detach "$EXPECTED_COMMIT"
  test "$(git -C "$W/qdrant" rev-parse HEAD)" = "$EXPECTED_COMMIT"
  test -z "$(git -C "$W/qdrant" status --porcelain)"
  tar -C "$W" -xf "$IN/vendor.tar"
  mkdir -p "$W/qdrant/.cargo"
  sed "s|__VENDOR_DIR__|$W/vendor|" "$IN/cargo-config.toml" > "$W/qdrant/.cargo/config.toml"
  /usr/bin/time -v cp "$CORPUS" "$W/data/corpus.f16" 2> "$OUT/stage-corpus.time"
  echo "$CORPUS_SHA  $W/data/corpus.f16" | sha256sum -c -
  cp "$IN"/queries.f32 "$IN"/ground_truth.i32 "$IN"/val.queries.f32 "$W/data/"
  SRC="$W/qdrant/research/lmi/supervised_routing"
  phase=build_binary
  cd "$W/qdrant"
  export CARGO_TARGET_DIR="$W/target" CARGO_BUILD_JOBS=$NCPUS
  /usr/bin/time -v cargo test --locked --offline --profile perf -p segment --features testing,lmi-training \
      --test supervised_routing_v120 --no-run > "$OUT/build.stdout" 2> "$OUT/build.stderr"
  BIN=$(ls -t "$W/target/perf/deps"/supervised_routing_v120-* | grep -v '\.d$' | head -1)
  sha256sum "$BIN" | tee "$OUT/binary.sha256"
  export SR_ROOT="$W/sr"
  # Native large-sample opt-in (build_plan.rs): 2 GB covers S=524,288 x 768 x 4 B.
  export QDRANT_LMI_SAMPLE_BUDGET_BYTES=2000000000
}

run() { /usr/bin/time -v "$BIN" --ignored --exact supervised_routing_v120 --nocapture; }

persist() {
  phase=persist
  ( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
  ( cd "$OUT" && sha256sum -c --quiet SHA256SUMS )
  printf '{"job":"%s","commit":"%s","binary_sha256":"%s","utc":"%s","status":"complete"}\n' \
    "$PBS_JOBID" "$EXPECTED_COMMIT" "$(cut -d' ' -f1 "$OUT/binary.sha256")" "$(date -u +%FT%TZ)" > "$OUT/COMPLETE.json"
  phase=done
  echo "COMPLETE $(date -u +%FT%TZ)"
}
