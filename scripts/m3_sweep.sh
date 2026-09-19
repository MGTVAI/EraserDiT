#!/usr/bin/env bash
# M3 measurement sweep: every acceleration configuration in the plan, repeated.
#
#   scripts/m3_sweep.sh [repeats]        # default 5
#
# Delegates to m3_measure.sh, which owns the single-inference guard, the GPU
# pick and the JSONL bookkeeping.  Results land in results/m3/<config>.jsonl.
#
# Plan §6.4: equivalence runs use the deterministic profile, performance runs
# use the fast one, so the sweep turns determinism off.  `sdpa` doubles as the
# unaccelerated reference N.
set -euo pipefail

REPEATS="${1:-5}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

export ERASERDIT_DETERMINISTIC=0

run() { echo "### $1 ($REPEATS repeats)"; "$HERE/m3_measure.sh" "$1" "$REPEATS" "${@:2}"; }

run sdpa          --attention-backend sdpa
run flash_attn    --attention-backend flash_attn
run sage_attn     --attention-backend sage_attn
run auto          --attention-backend auto
run compile       --attention-backend sdpa --enable-torch-compile
run sage_compile  --attention-backend sage_attn --enable-torch-compile
run fusion_qk     --attention-backend sdpa --operator-fusion-backend triton --operator-fusion-ops qk_rmsnorm_rope
run fusion_all    --attention-backend sdpa --operator-fusion-backend triton

# Exploratory: the plan fixes max-autotune-no-cudagraphs, which pays a long
# autotune up front; `default` is the cheap alternative worth a row.
echo "### compile_default ($REPEATS repeats)"
MGERASE_TORCH_COMPILE_MODE=default \
  "$HERE/m3_measure.sh" compile_default "$REPEATS" \
  --attention-backend sdpa --enable-torch-compile
