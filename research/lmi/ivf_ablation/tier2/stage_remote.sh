#!/bin/bash
# Stage frozen code + small inputs for Tier 2 to MetaCentrum through the existing ControlMaster.
# Usage: stage_remote.sh <commit>
# Never overwrites a remote file: refuses if ROOT/code.commit exists with a different commit.
set -euo pipefail
COMMIT=${1:?commit}
SOCK="$HOME/.ssh/metacentrum-control"; TARGET=nico@perian.metacentrum.cz
S=(ssh -F /dev/null -S "$SOCK" -o BatchMode=yes "$TARGET")
ssh -F /dev/null -S "$SOCK" -O check "$TARGET"
ROOT=/storage/budejovice1/home/nico/lmi-metacentrum/static-lmi-ivf-20261118
E=${ADMIN:?}/experiments/2026-11-18-static-lmi-ivf
WT=${WORKTREES:?}/qdrant-static-lmi-ivf
L=$E/remote-staging-$COMMIT; mkdir -p "$L"
git -C "$WT" archive --format=tar "$COMMIT" research/lmi/ivf_ablation > "$L/code.tar"
echo "$COMMIT" > "$L/code.commit"
( cd "$L" && sha256sum code.tar > code.tar.sha256 )
cp -r "$E/gold-10m" "$E/gold-1m" "$L/"
cp "$WT/research/lmi/ivf_ablation/tier2/lab-10m.pbs" "$L/"
( cd "$L" && find code.tar gold-10m gold-1m lab-10m.pbs -type f | sort | xargs sha256sum > inputs.sha256 )
existing=$("${S[@]}" "cat $ROOT/code.commit 2>/dev/null || true")
if [[ -n "$existing" && "$existing" != "$COMMIT" ]]; then
  echo "remote ROOT already holds commit $existing; use a new ROOT for a different revision" >&2; exit 1
fi
"${S[@]}" "mkdir -p $ROOT/results"
tar -C "$L" -cf - . | "${S[@]}" "tar -C $ROOT -xf - && cd $ROOT && sha256sum -c --quiet inputs.sha256 && echo REMOTE_STAGED_OK"
