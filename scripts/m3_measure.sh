#!/usr/bin/env bash
# M3 measurement harness: run one configuration N times and append the timings.
#
#   scripts/m3_measure.sh <config-name> <repeats> [extra inference_cli.sh args...]
#
# Results go to results/m3/<config>.jsonl, one JSON object per repeat with the
# end-to-end seconds and the pure-inference stage breakdown.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:?config name required}"; shift
REPEATS="${1:?repeat count required}"; shift

MODEL="${ERASERDIT_MODEL:-/root/.cache/huggingface/hub/models--jieeliu--EraserDiT/snapshots/904fb412da76235085dbbccaefdbde4979fa3d29}"
VIDEO="${M3_VIDEO:-data/10268234.mp4}"
MASK="${M3_MASK:-data/10268234_mask.mp4}"
PROMPT="${M3_PROMPT:-There is a bridge over the lake.}"
OUT_DIR="${M3_OUT_DIR:-results/m3}"
# A second stream on another card must skip the guard and pin its GPU; the
# guard stays on by default because two processes on one card OOM.
NO_GUARD="${M3_NO_GUARD:-0}"
mkdir -p "$OUT_DIR"

if [ "$NO_GUARD" != "1" ]; then
  # Never stack two inference processes: they each need ~45 GiB and would OOM.
  while pgrep -f "^/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python -m entrypoints" >/dev/null; do
    echo "waiting for the running inference process to finish..."
    sleep 30
  done
fi

GPU="${M3_GPU:-$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
      awk -F, '{g=$2-$3; if (g+0>m+0){m=g; idx=$1}} END {print idx}')}"
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
echo "config=$CONFIG repeats=$REPEATS gpu=$GPU free=${FREE}MiB args: $*"

for i in $(seq 1 "$REPEATS"); do
  LOG="$OUT_DIR/${CONFIG}_run${i}.log"
  START=$(date +%s)
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./inference_cli.sh --model-path "$MODEL" \
      --video-input "$VIDEO" --mask-input "$MASK" \
      --output-path "$OUT_DIR/${CONFIG}_run${i}.mp4" --prompt "$PROMPT" \
      "$@" > "$LOG" 2>&1 || {
        echo "  run $i FAILED (see $LOG)";
        python3 - "$CONFIG" "$i" "$LOG" "$OUT_DIR/${CONFIG}.jsonl" <<'PY'
import json, re, sys
config, run, log, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
text = open(log, errors="replace").read()
err = next((l for l in text.splitlines() if "Error" in l or "error" in l), "")
with open(out, "a") as fh:
    fh.write(json.dumps({"config": config, "run": int(run), "failed": True,
                         "error": err[:300]}) + "\n")
PY
        continue;
      }
  END=$(date +%s)
  python3 - "$CONFIG" "$i" "$LOG" "$OUT_DIR/${CONFIG}.jsonl" "$((END-START))" <<'PY'
import json, sys
config, run, log, out, wall = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
text = open(log, errors="replace").read()
start = text.rfind("{")
try:
    payload = json.loads(text[text.index('{\n  "tasks"'):]) if '"tasks"' in text else json.loads(text[start:])
    task = payload["tasks"][0]
except Exception:
    with open(out, "a") as fh:
        fh.write(json.dumps({"config": config, "run": int(run), "failed": True,
                             "error": "no json payload"}) + "\n")
    sys.exit(0)
timing = task.get("timing", {})
extra = timing.get("extra") or {}
record = {
    "config": config,
    "run": int(run),
    "failed": False,
    "wall_seconds": wall,
    # Older records predate the warmup split and only carry elapsed_seconds.
    "e2e_seconds": task.get("e2e_seconds_excluding_warmup", task.get("elapsed_seconds")),
    "e2e_seconds_including_warmup": task.get("elapsed_seconds"),
    "warmup_seconds": (task.get("warmup") or {}).get("duration_seconds"),
    "pure_inference_seconds": timing.get("pure_inference_seconds"),
    "stages_ms": timing.get("pure_inference_stage_breakdown_ms"),
    "step_median_ms": timing.get("step_median_ms"),
    "step_times_ms": timing.get("step_times_ms"),
    "load_seconds": extra.get("load_seconds"),
    "peak_allocated_gib": extra.get("peak_allocated_gib"),
    "peak_reserved_gib": extra.get("peak_reserved_gib"),
    "torch_compile": extra.get("torch_compile"),
    "attention_backend": extra.get("attention_backend"),
    "operator_fusion": extra.get("operator_fusion"),
    "runtime_timing_seconds": extra.get("runtime_timing_seconds"),
}
with open(out, "a") as fh:
    fh.write(json.dumps(record) + "\n")
print(f"  run {run}: e2e={record['e2e_seconds']}s pure={record['pure_inference_seconds']}s "
      f"denoise={(record['stages_ms'] or {}).get('EraserDiTEraseDenoisingStage')}ms")
PY
done
