#!/usr/bin/env bash
# EraserDiT / LTX095 erase service launcher.
#
# Usage:
#   ./inference_server.sh --pipeline-name EraserDiTErasePipeline \
#       --model-path <snapshot> --task-root /tmp/mgerase_tasks \
#       --input-allowed-root "$PWD/data"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ERASERDIT_PYTHON:-/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python}"

# Loading the local snapshot must not fall back to huggingface.co.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Match the frozen baseline's encoder settings (see vibe/environment.md).
export MGERASE_FFMPEG_THREADS="${MGERASE_FFMPEG_THREADS:-auto}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -m entrypoints.server.serve "$@"
