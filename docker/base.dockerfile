# syntax=docker/dockerfile:1
# EraserDiT runtime image.  Build context: repository root.
#
#   docker build -f docker/base.dockerfile -t erasedit:cu126 .
#
# The model snapshot is not baked in; mount it and pass --model-path.  The
# launchers already default HF_HUB_OFFLINE=1, so a missing path fails loudly
# instead of stalling on huggingface.co.
#
#   docker run --gpus all --rm -it \
#       -v /path/to/snapshot:/models/eraserdit:ro \
#       -v "$PWD/data:/workspace/EraserDiT/data:ro" \
#       -v "$PWD/results:/workspace/EraserDiT/results" \
#       erasedit:cu126 ./inference_cli.sh \
#           --model-path /models/eraserdit \
#           --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
#           --output-path results/out.mp4 \
#           --prompt "There is a bridge over the lake." \
#           --attention-backend sage_attn --enable-torch-compile --warmup
#
#   docker run --gpus all --rm -p 30000:30000 \
#       -v /path/to/snapshot:/models/eraserdit:ro \
#       -v "$PWD/data:/workspace/EraserDiT/data:ro" \
#       -v /tmp/mgerase_tasks:/tmp/mgerase_tasks \
#       erasedit:cu126 ./inference_server.sh \
#           --pipeline-name EraserDiTErasePipeline --model-path /models/eraserdit \
#           --task-root /tmp/mgerase_tasks --input-allowed-root /workspace/EraserDiT/data
ARG PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
# ffmpeg carries the x264 encoder the output contract and the acceptance
# metrics depend on; libgl1/libglib2.0-0 satisfy opencv-python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/EraserDiT

# Both launchers fall back to a site-specific conda interpreter when
# ERASERDIT_PYTHON is unset, which does not exist here.
ENV ERASERDIT_PYTHON=/usr/local/bin/python

COPY requirements.txt ./
# flash-attn and sageattention compile CUDA kernels against the installed
# Torch, so the build must see it: --no-build-isolation plus the toolchain from
# the devel base image.  requirements.txt pins torch==2.6.0+cu126, which the
# base image already provides.
RUN python -m pip install --upgrade pip \
    && CUDA_HOME=/usr/local/cuda \
        PATH=/usr/local/cuda/bin:$PATH \
        MAX_JOBS="${MAX_JOBS:-8}" \
        PIP_NO_CACHE_DIR=1 \
        python -m pip install --no-build-isolation -r requirements.txt

COPY . .
ENV PYTHONPATH=/workspace/EraserDiT \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Fail the build early if the entry points cannot even be resolved.
RUN ./inference_cli.sh --help > /dev/null \
    && ./inference_server.sh --help > /dev/null \
    && python -c "import torch, diffusers, flash_attn, sageattention, triton"

CMD ["/bin/bash"]
