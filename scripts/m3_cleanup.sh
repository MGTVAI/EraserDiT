#!/usr/bin/env bash
# Confirm the leading M3 candidate on its own window.
#
#   scripts/m3_cleanup.sh <gpu> [candidate-repeats] [reference-repeats]
#
# The probe stream screened the combinations; `sage_attn + 全融合` came out on
# top but only had two repeats on a contended card.  This gives it the plan's
# five, next to a `ref_sdpa` row measured in the same window so the pair is
# comparable even if the host load has drifted since the main sweep.
#
# The parallel-contention question is already settled: the probe's `ref_sdpa`
# came in 0.9% above the main sweep's lone `sdpa`, so the main sweep's numbers
# stand as measured.
#
# Results land in results/m3-clean/.
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

echo "### ref_sdpa ($REF_REPEATS repeats, paired reference)"
"$HERE/m3_measure.sh" ref_sdpa "$REF_REPEATS" --attention-backend sdpa

echo "### fusion_all_sage ($REPEATS repeats)"
"$HERE/m3_measure.sh" fusion_all_sage "$REPEATS" \
  --attention-backend sage_attn --operator-fusion-backend triton

echo "### sage_compile_default ($REPEATS repeats, screening)"
MGERASE_TORCH_COMPILE_MODE=default \
  "$HERE/m3_measure.sh" sage_compile_default "$REPEATS" \
  --attention-backend sage_attn --enable-torch-compile
