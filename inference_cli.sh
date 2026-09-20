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
#     --task-file tasks/acceptance_tasks.json \
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
#   - HF_HUB_OFFLINE 下面已默认置 1，不必手动 export
#   - --warmup 是加速达标的必要条件，不是可选优化：不预热的话 Inductor 自动调优
#     落进正式任务的首次前向，去噪收益从 17% 掉到 10%，过不了 15% 门槛
#   - 换解释器用 ERASERDIT_PYTHON=<path>，默认写死 conda 环境路径
#   - 该卡有常驻租户，跑前确认空闲显存 >= 60 GiB
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ERASERDIT_PYTHON:-/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python}"

# Loading the local snapshot must not fall back to huggingface.co; the direct
# connection stalls silently (see vibe/environment.md).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -m entrypoints.cli.erase_eraserdit "$@"
