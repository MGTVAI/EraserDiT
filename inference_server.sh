#!/usr/bin/env bash
# EraserDiT / LTX095 erase service launcher.
#
# 启动（GPU 2）：
#
#   cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT
#   SNAP=/root/.cache/huggingface/hub/models--jieeliu--EraserDiT/snapshots/904fb412da76235085dbbccaefdbde4979fa3d29
#
#   CUDA_VISIBLE_DEVICES=2 ./inference_server.sh \
#     --pipeline-name EraserDiTErasePipeline \
#     --model-path "$SNAP" \
#     --task-root /tmp/mgerase_tasks \
#     --input-allowed-root "$PWD/data" \
#     --port 30000 \
#     --attention-backend sage_attn --enable-torch-compile --warmup
#
# 另开终端验收（全部端点 + 严格契约 + 任务生命周期 + 结果下载与删除）：
#
#   python3 scripts/validation/service_smoke.py --base-url http://127.0.0.1:30000 \
#     --video data/10268234.mp4 --mask data/10268234_mask.mp4 \
#     --prompt "There is a bridge over the lake."
#
# 或一键起服务 + 验收（自动等一张空闲卡，无论成败都拆干净）：
#
#   scripts/validation/service_verify.sh 2
#
# 说明：
#   - --pipeline-name 不能省，服务端靠它选管线
#   - --input-allowed-root 是输入白名单，必须是绝对路径，否则请求被拒
#   - HF_HUB_OFFLINE 与 PYTORCH_CUDA_ALLOC_CONF（expandable_segments:True）下面已默认
#     置好，不必手动 export；要换值直接 export 同名变量
#   - 服务是长驻进程，跑完记得停；该卡有常驻租户，需空闲显存 >= 60 GiB
#   - 模型路径有改动后必须重跑验收：M2 的首次验收跑在 cross-attention 修复之前，
#     20 项端点检查全过，但它下载到的 mp4 是损坏版本产出的（端点检查不查画面）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ERASERDIT_PYTHON:-/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python}"

# Loading the local snapshot must not fall back to huggingface.co.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Match the frozen baseline's encoder settings (see vibe/environment.md).
export MGERASE_FFMPEG_THREADS="${MGERASE_FFMPEG_THREADS:-auto}"
# Same reason as inference_cli.sh: the VAE encode needs one contiguous 7.5 GiB
# block, which the default allocator can strand behind its own free blocks.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -m entrypoints.server.serve "$@"
