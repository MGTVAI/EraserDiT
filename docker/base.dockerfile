# syntax=docker/dockerfile:1
# Build context: repository root
# docker build -f docker/base.dockerfile -t erasedit:ltx095-cu126 .

ARG PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/EraserDiT

# Keep the project scripts unchanged: they invoke .venv/bin/torchrun.  The
# virtual environment shares the PyTorch supplied by the validated base image.
RUN python -m venv --system-site-packages /opt/erasedit-venv \
    && printf '%s\n' '#!/usr/bin/env bash' 'exec python -m torch.distributed.run "$@"' \
        > /opt/erasedit-venv/bin/torchrun \
    && chmod +x /opt/erasedit-venv/bin/torchrun
ENV VIRTUAL_ENV=/opt/erasedit-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        torch==2.6.0+cu126 \
        torchvision==0.21.0+cu126 \
        triton==3.2.0 \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
    && CUDA_HOME=/usr/local/cuda \
        PATH=/usr/local/cuda/bin:$PATH \
        MAX_JOBS=8 \
        python -m pip install --no-build-isolation -r requirements.txt

COPY . .
RUN ln -s /opt/erasedit-venv .venv \
    && mkdir -p Acceptance/result_videos Acceptance/4k_result_videos

ENV PYTHONPATH=/workspace/EraserDiT
ENTRYPOINT ["/bin/bash"]
