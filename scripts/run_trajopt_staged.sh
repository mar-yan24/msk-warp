#!/usr/bin/env bash
# Staged trajectory optimisation: several short processes instead of one long one.
#
# This machine sits at ~24.8 GB committed of a 31.3 GB limit before any Python starts, and three
# single-process attempts at the motor control were OOM-killed mid-run. Each stage exits and
# releases everything, and Adam's moments ride along in the checkpoint, so the staged run is
# equivalent to one long one rather than a sequence of restarts.
#
#   scripts/run_trajopt_staged.sh <cfg> <cycle> <worlds> <total-iters> <stage-iters> <out.json>
set -u
PY=.venv/Scripts/python.exe
CFG=$1; CYCLE=$2; WORLDS=$3; TOTAL=$4; STAGE=$5; OUT=$6
SCALES=docs/research/phase4-capability/gait_motor_seed2.json
PARAMS="${OUT%.json}_params.npz"

done_iters=0
while [ "$done_iters" -lt "$TOTAL" ]; do
  remaining=$((TOTAL - done_iters))
  n=$([ "$remaining" -lt "$STAGE" ] && echo "$remaining" || echo "$STAGE")
  last=$([ "$n" -eq "$remaining" ] && echo yes || echo no)

  args=(--cfg "$CFG" --scales "$SCALES" --cycle "$CYCLE" --worlds "$WORLDS" \
        --iters "$n" --log-every 25 --out "$OUT")
  [ -f "$PARAMS" ] && args+=(--resume "$PARAMS")
  [ "$last" = no ] && args+=(--skip-verify)

  echo "=== stage: $n iterations (${done_iters}/${TOTAL} done, final=$last) ==="
  if ! $PY scripts/trajopt_hopper.py "${args[@]}"; then
    echo "!! stage failed or was killed at ${done_iters}; checkpoint is intact, rerun to continue"
    exit 1
  fi
  done_iters=$((done_iters + n))
done
echo "=== all $TOTAL iterations complete ==="
