# syntax=docker/dockerfile:1
# Build context: repository root
# docker build -f docker/base.dockerfile -t mgerase:ltx095-cu126 .

ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
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

WORKDIR /workspace/MGErase

# Keep the project scripts unchanged: they invoke .venv/bin/torchrun.  The
# virtual environment shares the PyTorch supplied by the validated base image.
RUN python -m venv --system-site-packages /opt/mgerase-venv \
    && printf '%s\n' '#!/usr/bin/env bash' 'exec python -m torch.distributed.run "$@"' \
        > /opt/mgerase-venv/bin/torchrun \
    && chmod +x /opt/mgerase-venv/bin/torchrun
ENV VIRTUAL_ENV=/opt/mgerase-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt requirements-optional.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install \
        fastapi==0.140.0 \
        uvicorn==0.51.0 \
        pydantic==2.13.4 \
        boto3

COPY . .
RUN ln -s /opt/mgerase-venv .venv \
    && mkdir -p Acceptance/result_videos Acceptance/4k_result_videos

ENV PYTHONPATH=/workspace/MGErase
ENTRYPOINT ["/bin/bash"]
