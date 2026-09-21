#!/usr/bin/env bash
# Four-card validation is authorized only after ALL selected cards are idle.
# Pass --wait first to wait for availability; otherwise exit 75 when occupied.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ERASERDIT_PYTHON:-/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python}"
WAIT_FOR_GPUS=0
if [[ "${1:-}" == "--wait" ]]; then
  WAIT_FOR_GPUS=1
  shift
fi
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=2,3,6,7
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
while true; do
  if "$PYTHON" - <<'PY'
import subprocess, sys, time
for sample in range(3):
    output = subprocess.check_output([
        'nvidia-smi', '-i', '2,3,6,7', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True)
    rows = [list(map(int, line.split(','))) for line in output.strip().splitlines()]
    if len(rows) != 4 or any(memory > 64 or utilization != 0 for _, memory, utilization in rows):
        print('Four-GPU test deferred; selected cards are occupied:\n' + output, flush=True)
        sys.exit(75)
    if sample < 2:
        time.sleep(1)
PY
  then
    break
  elif [[ "$WAIT_FOR_GPUS" == 0 ]]; then
    exit 75
  fi
  sleep 30
done
exec "$PYTHON" scripts/benchmarks/parallel_benchmark.py \
  --matrix-configs serial cfg2_sp2 sp4 spatial_vae4 cfg2_sp2_spatial_vae4 sp4_spatial_vae4 "$@"
