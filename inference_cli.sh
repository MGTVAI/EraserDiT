#!/usr/bin/env bash
# EraserDiT CLI launcher.
#
# Usage:
#   ./inference_cli.sh --model-path <snapshot> --video-input <vid> --mask-input <mask> \
#       --output-path results/out.mp4 --prompt "..."
#   ./inference_cli.sh --model-path <snapshot> --task-file tasks.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ERASERDIT_PYTHON:-/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python}"

# Loading the local snapshot must not fall back to huggingface.co; the direct
# connection stalls silently (see vibe/environment.md).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -m entrypoints.cli.erase_eraserdit "$@"
