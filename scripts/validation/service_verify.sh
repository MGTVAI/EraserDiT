#!/usr/bin/env bash
# Bring up the erase service and run the end-to-end acceptance against it.
#
#   scripts/validation/service_verify.sh [gpu] [min-free-mib]
#
# With no GPU argument it waits for a card to free up, because the resident
# pipeline needs ~60 GiB and this host is shared.  Everything is torn down on
# exit, including a service that fails to become healthy.
#
# This exists because the M2 acceptance in commit 6c714f2 ran before the
# cross-attention fix (26c5a6b): its endpoint checks passed, but the video it
# produced came from the broken revision.  Re-run it after any change to the
# model path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

MODEL="${ERASERDIT_MODEL:-/root/.cache/huggingface/hub/models--jieeliu--EraserDiT/snapshots/904fb412da76235085dbbccaefdbde4979fa3d29}"
PORT="${SERVICE_PORT:-30000}"
MIN_FREE="${2:-60000}"
TASK_ROOT="${TASK_ROOT:-/tmp/mgerase_tasks}"
LOG=/tmp/service_verify_server.log
GPU="${1:-}"

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"
}

if [ -z "$GPU" ]; then
  while :; do
    while read -r index free; do
      if [ "$free" -ge "$MIN_FREE" ]; then GPU="$index"; break; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    [ -n "$GPU" ] && break
    echo "no card with ${MIN_FREE} MiB free yet, waiting..."
    sleep 120
  done
fi
echo "using gpu=$GPU free=$(free_mib "$GPU") MiB port=$PORT"

server_pid=""
cleanup() {
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    sleep 3
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "$TASK_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./inference_server.sh --pipeline-name EraserDiTErasePipeline \
    --model-path "$MODEL" --task-root "$TASK_ROOT" \
    --input-allowed-root "$PWD/data" --port "$PORT" > "$LOG" 2>&1 &
server_pid=$!
echo "server pid=$server_pid, log=$LOG"

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "service healthy after ${_}"
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited during startup:"; tail -30 "$LOG"; exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null || {
  echo "service never became healthy"; tail -30 "$LOG"; exit 1; }

python3 scripts/validation/service_smoke.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --video data/10268234.mp4 --mask data/10268234_mask.mp4 \
  --prompt "There is a bridge over the lake."
