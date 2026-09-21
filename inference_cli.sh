#!/usr/bin/env bash
# EraserDiT CLI launcher.
#
# 单条素材（M3 推荐加速配置，GPU 2）：
#
#   cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT
#   SNAP=/root/.cache/huggingface/hub/models--jieeliu--EraserDiT/snapshots/904fb412da76235085dbbccaefdbde4979fa3d29
#
#   CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh \
#     --model-path "$SNAP" \
#     --video-input data/10268234.mp4 \
#     --mask-input  data/10268234_mask.mp4 \
#     --output-path results/out_10268234.mp4 \
#     --prompt "There is a bridge over the lake." \
#     --attention-backend sage_attn --enable-torch-compile --warmup
#
# 第二组素材（横屏、2 窗口；提示词不同）：
#
#   CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh \
#     --model-path "$SNAP" \
#     --video-input data/113000356.mp4 \
#     --mask-input  data/113000356_mask.mp4 \
#     --output-path results/out_113000356.mp4 \
#     --prompt "There is a rooftop terrace overlooking the city at sunset." \
#     --attention-backend sage_attn --enable-torch-compile --warmup
#
# 多任务共用常驻 pipeline（预热只付一次，两种画幅可混用）：
#
#   CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh --model-path "$SNAP" \
#     --task-file tasks.json \
#     --attention-backend sage_attn --enable-torch-compile --warmup
#
# 不加速的对照（矩阵里的 N）：
#
#   CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh --model-path "$SNAP" \
#     --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
#     --output-path results/out_N.mp4 --prompt "There is a bridge over the lake." \
#     --attention-backend sdpa
#
# 说明：
#   - HF_HUB_OFFLINE 与 PYTORCH_CUDA_ALLOC_CONF（expandable_segments:True）下面已默认
#     置好，不必手动 export；要换值直接 export 同名变量，脚本不覆盖既有值
#   - --warmup 是加速达标的必要条件，不是可选优化：不预热的话 Inductor 自动调优
#     落进正式任务的首次前向，去噪收益从 17% 掉到 10%，过不了 15% 门槛
#   - 换解释器用 ERASERDIT_PYTHON=<path>，默认写死 conda 环境路径
#   - 该卡有常驻租户（约 18.5 GiB）。脚本启动时打印空闲显存，低于 60 GiB 告警：
#     VAE 编码要一整块 7.5 GiB 连续显存，余量不足时会先 OOM
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ERASERDIT_PYTHON:-/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python}"

# Loading the local snapshot must not fall back to huggingface.co; the direct
# connection stalls silently (see vibe/environment.md).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# The VAE encoder up-casts a whole 121-frame window to float32 for GroupNorm --
# on 1080x1920 that is a single contiguous 7.5 GiB request.  Without
# expandable_segments the allocator strands it behind blocks it already holds:
# 2026-09-20 the single-clip run OOMed with 61 GiB free, while the harnessed
# entry points (scripts/validation/acceptance.py, scripts/legacy/m3_measure.sh) pass with this set.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ~60 GiB is the working floor for a 121-frame 1080x1920 window.  Warn rather
# than refuse: this launcher also carries exploratory runs.
GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"
GPU_INDEX="${GPU_INDEX%%,*}"
if [ -n "${GPU_INDEX}" ]; then
  GPU_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    -i "${GPU_INDEX}" 2>/dev/null || true)"
  if [ -n "${GPU_FREE_MIB}" ]; then
    echo "gpu ${GPU_INDEX}: ${GPU_FREE_MIB} MiB free"
    if [ "${GPU_FREE_MIB}" -lt 60000 ] 2>/dev/null; then
      echo "warning: below the ~60000 MiB floor -- OOM inside the VAE encode is likely" >&2
    fi
  fi
fi

exec "${PYTHON}" -m entrypoints.cli.erase_eraserdit "$@"
