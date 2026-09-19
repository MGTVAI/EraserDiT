#!/usr/bin/env bash
# Second M3 stream: screening run on its own card, alongside m3_sweep.sh.
#
#   scripts/m3_probe.sh <gpu> [repeats]      # default 2
#
# Two questions the main sweep does not answer:
#   1. the fusion x attention-backend *combinations* the plan asks for, and
#   2. whether a CUDA-graphs compile mode beats max-autotune-no-cudagraphs.
#
# Everything is measured against a reference row taken on the *same* card, so
# the host contention from the parallel stream cancels out of the deltas.
# Results land in results/m3-probe/.
set -euo pipefail

GPU="${1:?gpu index required}"
REPEATS="${2:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

export ERASERDIT_DETERMINISTIC=0
export M3_GPU="$GPU"
export M3_NO_GUARD=1
export M3_OUT_DIR="results/m3-probe"
mkdir -p "$M3_OUT_DIR"

run() { echo "### $1 ($REPEATS repeats)"; "$HERE/m3_measure.sh" "$1" "$REPEATS" "${@:2}"; }

run ref_sdpa          --attention-backend sdpa
run fusion_qk_sage    --attention-backend sage_attn --operator-fusion-backend triton --operator-fusion-ops qk_rmsnorm_rope
run fusion_all_sage   --attention-backend sage_attn --operator-fusion-backend triton
echo "### compile_reduce_overhead ($REPEATS repeats)"
MGERASE_TORCH_COMPILE_MODE=reduce-overhead \
  "$HERE/m3_measure.sh" compile_reduce_overhead "$REPEATS" \
  --attention-backend sdpa --enable-torch-compile
echo "### sage_compile_reduce_overhead ($REPEATS repeats)"
MGERASE_TORCH_COMPILE_MODE=reduce-overhead \
  "$HERE/m3_measure.sh" sage_compile_reduce_overhead "$REPEATS" \
  --attention-backend sage_attn --enable-torch-compile
