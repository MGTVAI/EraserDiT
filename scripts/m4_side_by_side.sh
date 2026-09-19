#!/usr/bin/env bash
# Side-by-side review videos for the M4 delivery (plan §M4).
#
#   scripts/m4_side_by_side.sh <clip> [source] [B] [N] [A]
#
# Defaults take the clip from data/ and the three pipeline outputs from
# results/m4/.  Panels that do not exist are dropped, so the script works with
# whatever has been measured so far.  Output: results/m4/<clip>_compare.mp4
#
# Frames are aligned by index (`setpts=N/(25*TB)`), not by timestamp: the
# source clips and the outputs disagree on frame-rate metadata (113000356 is
# 24000/1001 while its mask is 1199/50) and resampling would silently drop or
# duplicate frames.
set -euo pipefail

CLIP="${1:?clip id required}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

SOURCE="${2:-data/${CLIP}.mp4}"
B="${3:-results/m4/${CLIP}_B.mp4}"
N="${4:-results/m4/${CLIP}_N_run1.mp4}"
A="${5:-results/m4/${CLIP}_A_run1.mp4}"
OUT="results/m4/${CLIP}_compare.mp4"
HEIGHT="${M4_PANEL_HEIGHT:-720}"

inputs=(); filters=(); index=0
for pair in "source:$SOURCE" "B:$B" "N:$N" "A:$A"; do
  label="${pair%%:*}"; path="${pair#*:}"
  if [ -f "$path" ]; then
    inputs+=(-i "$path")
    chain="[${index}:v]scale=-2:${HEIGHT},setpts=N/(25*TB)"
    chain="${chain},drawtext=text='${label}':x=12:y=12:fontsize=30:fontcolor=white"
    chain="${chain}:box=1:boxcolor=black@0.55:boxborderw=6[p${index}]"
    filters+=("$chain")
    index=$((index + 1))
  else
    echo "skip panel $label: $path not found"
  fi
done

if [ "$index" -lt 2 ]; then
  echo "need at least two panels to compare, have $index" >&2
  exit 1
fi

stack=""
for i in $(seq 0 $((index - 1))); do stack="${stack}[p${i}]"; done
# Each panel is its own filterchain, so they join with ';' not ','.
filter=""
for chain in "${filters[@]}"; do filter="${filter}${chain};"; done
filter="${filter%;}"

mkdir -p results/m4
ffmpeg -v error -y "${inputs[@]}" \
  -filter_complex "${filter};${stack}hstack=inputs=${index}[out]" -map "[out]" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 "$OUT"
echo "wrote $OUT (${index} panels)"
