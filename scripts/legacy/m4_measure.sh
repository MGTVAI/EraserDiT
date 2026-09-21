#!/usr/bin/env bash
# M4 acceptance measurements for the two plan clips (plan §6.1).
#
#   scripts/legacy/m4_measure.sh <clip> <N|A> <repeats> [gpu]
#
# N = new architecture, unaccelerated (sdpa).  A = the recommended accelerated
# configuration from docs/performance.md#single-gpu.  B (the frozen baseline) is a different
# codebase and runs through EraserDiT-baseline/baseline_runner.py instead; see
# docs/performance.md#single-gpu.
#
# Results land in results/m4/<clip>_<config>.jsonl via m3_measure.sh, which
# already owns the single-process guard, the GPU pick and the bookkeeping.
set -euo pipefail

CLIP="${1:?clip id required}"
CONFIG="${2:?config N or A required}"
REPEATS="${3:?repeat count required}"
GPU="${4:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

export ERASERDIT_DETERMINISTIC=0
export M3_VIDEO="data/${CLIP}.mp4"
export M3_MASK="data/${CLIP}_mask.mp4"
export M3_OUT_DIR="results/m4"
mkdir -p "$M3_OUT_DIR"
[ -n "$GPU" ] && export M3_GPU="$GPU"

PROMPT_FILE="data/${CLIP}.prompt"
if [ -f "$PROMPT_FILE" ]; then
  export M3_PROMPT="$(cat "$PROMPT_FILE")"
elif [ "$CLIP" = "113000356" ]; then
  export M3_PROMPT="There is a rooftop terrace overlooking the city at sunset."
else
  export M3_PROMPT="There is a bridge over the lake."
fi

case "$CONFIG" in
  N) exec "$HERE/m3_measure.sh" "${CLIP}_N" "$REPEATS" --attention-backend sdpa ;;
  A) exec "$HERE/m3_measure.sh" "${CLIP}_A" "$REPEATS" \
       --attention-backend sage_attn --enable-torch-compile --warmup ;;
  *) echo "config must be N or A, got $CONFIG" >&2; exit 2 ;;
esac
