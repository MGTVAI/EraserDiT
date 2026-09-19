#!/usr/bin/env bash
# Re-measure the configs the main sweep ran while a second stream was live.
#
#   scripts/m3_cleanup.sh <gpu> [candidate-repeats] [reference-repeats]
#
# `fusion_all` and `compile_default` only ever ran with the probe stream on the
# other card, so their absolute timings carry host contention.  This replays
# them together with a `ref_sdpa` row taken in the same window: comparing that
# reference against the uncontended `sdpa` in results/m3 quantifies the
# contention, and the candidates are judged against the contemporaneous
# reference rather than the original one.  Results land in results/m3-clean/.
set -euo pipefail

GPU="${1:?gpu index required}"
REPEATS="${2:-5}"
REF_REPEATS="${3:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

export ERASERDIT_DETERMINISTIC=0
export M3_GPU="$GPU"
export M3_NO_GUARD=1
export M3_OUT_DIR="results/m3-clean"
mkdir -p "$M3_OUT_DIR"

echo "### ref_sdpa ($REF_REPEATS repeats, contention calibration)"
"$HERE/m3_measure.sh" ref_sdpa "$REF_REPEATS" --attention-backend sdpa
echo "### fusion_all ($REPEATS repeats)"
"$HERE/m3_measure.sh" fusion_all "$REPEATS" --attention-backend sdpa --operator-fusion-backend triton
echo "### compile_default ($REPEATS repeats)"
MGERASE_TORCH_COMPILE_MODE=default \
  "$HERE/m3_measure.sh" compile_default "$REPEATS" \
  --attention-backend sdpa --enable-torch-compile
