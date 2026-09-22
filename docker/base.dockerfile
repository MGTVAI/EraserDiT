# syntax=docker/dockerfile:1
# EraserDiT runtime image.  Build context: repository root.
#
#   docker build -f docker/base.dockerfile -t eraserdit:cu126 .
#
# Mount a complete model and inputs at runtime; see docs/setup.md.
# Default backend is SDPA; optional attention packages are not required.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
FROM ${PYTORCH_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

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

COPY requirements.txt ./
RUN uv venv --python 3.10 .venv \
    && uv pip install --python .venv/bin/python --no-cache --index-strategy unsafe-best-match -r requirements.txt \
    && uv pip check --python .venv/bin/python

COPY . .
ENV PYTHONPATH=/workspace/EraserDiT \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Fail the build early if the entry points cannot even be resolved.
RUN uv run --no-project python -m entrypoints.cli.erase_eraserdit --help > /dev/null \
    && uv run --no-project python -m entrypoints.server.serve --help > /dev/null \
    && uv run --no-project python -c "import torch, diffusers, triton"

CMD ["/bin/bash"]
