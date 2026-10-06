<h1 align="center">
  <span style="color:#2196f3;"><b>EraserDiT</b></span>: Fast Video Inpainting with Diffusion Transformer Model
</h1>

<p align="center">
  <a href="https://huggingface.co/jieeliu/EraserDiT"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/JieLiu95/EraserDiT"><img alt="Github" src="https://img.shields.io/badge/EraserDiT-github-black"></a>
  <a href="https://arxiv.org/abs/2506.12853"><img alt="arXiv" src="https://img.shields.io/badge/EraserDiT-arXiv-b31b1b"></a>
  <a href="https://jieliu95.github.io/EraserDiT_demo/"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</p>

## 原始算法：[EraserDiT](https://github.com/JieLiu95/EraserDiT)

可根据指定区域擦除视频中的物体，并恢复背景内容与时序一致性。

| 输入视频 | 擦除结果 |
| :---: | :---: |
| ![输入视频：天台上的人物](docs/113000356_input.gif) | ![擦除人物后的天台背景](docs/113000356_erased.gif) |

## 本仓库：面向算法的推理 infra

本仓库基于 [原始 EraserDiT 算法](https://github.com/JieLiu95/EraserDiT)，提供**更低显存、更快推理和可部署的视频擦除服务**：输入视频、掩码与背景提示词，即可通过 CLI、HTTP API、WebUI完成擦除。

实现参考 SGLang 的 `python/sglang/multimodal_gen`，可理解为面向 EraserDiT 的 **mini SGLang 多模态推理运行时**。所需能力在仓库内实现，便于在**算法环境中维护依赖兼容性**，无需安装 SGLang；模块划分清晰，便于**阅读、调试、二次开发**。

## 推理加速后的性能数据

固定使用 `data/113000356.mp4` 及对应 mask：1920×1080、145 帧、24000/1001 fps；prompt 为 `There is a bridge over the lake.`，seed 42、CFG 3、50 步×strength 0.8（每窗口实际 40 步）、121 帧窗口、9 帧重叠、完整尾窗填充。芯片为NVIDIA L40s。


| 测试 | GPU | VAE | 配置 | 状态 | 1-step 预热 | 请求耗时 | 纯模型推理 | 文本编码 | VAE 编码 | DiT 去噪 | DiT 加速（B1=1.00×） | VAE 解码 | 纯推理加速（B1=1.00×） |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 1 | 1 | 原始算法，全驻留 BF16/SDPA | OOM（正式请求，不再复测） | 28.07 | OutOfMemoryError（request） | — | — | — | — | — | — | — |
| B1 | 1 | 1 | 原始算法，CPU offload + VAE tiling | 2026-10-06 复测 | 38.18 | 1814.12 | 467.59 | 0.15 | 16.51 | 440.21 | 1.00× | 10.72 | 1.00× |
| S0 | 1 | 1 | 当前源码，preload + 逐层卸载，关闭融合 | 2026-10-06 复测 | 70.10 | 485.02 | 453.72 | 1.13 | 17.11 | 424.50 | 1.04× | 10.97 | 1.03× |
| S1 | 1 | 1 | S0 + 窗口流式 I/O + uint8 帧缓存 | 2026-10-06 复测 | 70.47 | 482.12 | 452.88 | 1.32 | 16.98 | 423.65 | 1.04× | 10.93 | 1.03× |
| S2 | 1 | 1 | S1 改为 DiT 组件卸载 | 2026-10-06 复测 | 75.58 | 480.09 | 450.94 | 1.05 | 16.32 | 422.54 | 1.04× | 11.03 | 1.04× |
| S3 | 1 | 1 | S2 + 精确 QK RoPE / AdaLN 融合 | 2026-10-06 复测 | 73.93 | 422.12 | 392.82 | 1.04 | 16.81 | 363.78 | 1.21× | 11.18 | 1.19× |
| SP2 | 2 | 2 | NCCL SP2 + VAE2 | 2026-10-06 复测 | 56.24 | 248.84 | 220.88 | 0.30 | 11.17 | 203.37 | 2.16× | 6.02 | 2.12× |
| CFG2 | 2 | 2 | NCCL CFG2 + VAE2 | 2026-10-06 复测 | 55.13 | 216.34 | 188.91 | 0.53 | 10.52 | 171.17 | 2.57× | 6.68 | 2.48× |
| SP4 | 4 | 4 | NCCL SP4 + VAE4 | 2026-10-06 复测 | 52.96 | 164.75 | 137.79 | 0.29 | 10.09 | 122.53 | 3.59× | 4.87 | 3.39× |
| CFG2+SP2 | 4 | 4 | NCCL CFG2+SP2 + VAE4 | 2026-10-06 复测 | 52.56 | 150.59 | 122.46 | 0.50 | 9.11 | 107.15 | 4.11× | 5.71 | 3.82× |
| C2 | 1 | 1 | S3 + TeaCache 0.1 | 2026-10-06 复测 | 73.79 | 258.36 | 230.82 | 1.06 | 16.66 | 202.22 | 2.18× | 10.87 | 2.03× |
| C5 | 1 | 1 | S3 + CacheDiT 0.1 | 2026-10-06 复测 | 73.60 | 268.18 | 240.64 | 1.05 | 16.33 | 212.78 | 2.07× | 10.47 | 1.94× |
| C7 | 4 | 4 | CFG2+SP2 + TeaCache 0.1 | 2026-10-06 复测 | 54.59 | 106.95 | 79.21 | 0.30 | 9.14 | 64.07 | 6.87× | 5.68 | 5.90× |
| C8 | 4 | 4 | CFG2+SP2 + CacheDiT 0.1 | 2026-10-06 复测 | 56.09 | 111.30 | 82.23 | 0.81 | 9.00 | 66.37 | 6.63× | 6.04 | 5.69× |
| Q1 | 1 | 1 | INT8 FFN + SDPA + 精确融合 | 2026-10-06 复测 | 72.00 | 407.39 | 380.09 | 1.04 | 16.32 | 352.22 | 1.25× | 10.50 | 1.23× |
| Q2 | 1 | 1 | 动态 FP8 FFN + SDPA + 精确融合 | 2026-10-06 复测 | 73.56 | 430.73 | 402.66 | 1.08 | 16.30 | 374.86 | 1.17× | 10.40 | 1.16× |
| Q3 | 1 | 1 | 静态 FP8 FFN + SDPA + 精确融合 | 2026-10-06 复测 | 70.79 | 408.50 | 379.17 | 1.29 | 17.01 | 350.41 | 1.26× | 10.44 | 1.23× |
| Q4 | 1 | 1 | Q3 + Sage FP8 attention | 2026-10-06 复测 | 70.38 | 348.48 | 319.22 | 1.19 | 17.33 | 289.54 | 1.52× | 11.14 | 1.46× |
| Q5 | 1 | 1 | Q4 + 快速 QK RoPE | 2026-10-06 复测 | 68.70 | 313.51 | 284.88 | 1.11 | 16.39 | 256.44 | 1.72× | 10.95 | 1.64× |
| Q6 | 1 | 1 | Q5 + 快速 AdaLN / gated residual | 2026-10-06 复测 | 69.42 | 283.96 | 255.69 | 1.07 | 16.95 | 226.47 | 1.94× | 11.20 | 1.83× |
| Q7 | 4 | 4 | CFG2+SP2 + Q6 完整量化／融合组合（关闭缓存） | 2026-10-06 复测 | 53.97 | 127.11 | 97.71 | 0.33 | 8.78 | 81.68 | 5.39× | 6.92 | 4.79× |
| Q7_cumulative | 4 | 4 | C7（四卡胜者）+ Q6 量化／快速融合，保留 TeaCache 0.1 | 2026-10-06 复测 | 51.18 | 93.68 | 65.41 | 0.54 | 8.68 | 49.45 | 8.90× | 6.74 | 7.15× |


## 快速开始

### 1. 安装依赖与下载模型

需要 Linux、NVIDIA 驱动、CUDA 12.6 Toolkit（含 `nvcc`）和 uv。环境要求与排错见 [安装说明](docs/setup.md)。

```bash
git clone https://github.com/MGTVAI/EraserDiT.git
cd EraserDiT
sudo apt-get install -y git curl build-essential ffmpeg libgl1 libglib2.0-0
nvcc --version
uv venv --python 3.10
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  torch==2.6.0+cu126 pip setuptools wheel packaging ninja psutil
MAX_JOBS=4 uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --no-build-isolation-package flash-attn -r requirements.txt
uv pip check --python .venv/bin/python
HF_HUB_OFFLINE=0 uv run --no-project hf download jieeliu/EraserDiT \
  --revision 904fb412da76235085dbbccaefdbde4979fa3d29 \
  --local-dir data/model --exclude ".DS_Store"
```

### 2. 设置共同运行参数

```bash
mkdir -p outputs
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MGERASE_NCCL_PACKING=direct MGERASE_DIT_BOUNDARY_TRANSPORT=cpu
export MGERASE_POSTPROCESS_CHUNKED_FP32=1 MGERASE_VAE_INPLACE_ACTIVATIONS=1
export MGERASE_VAE_OFFLOAD_MODE=cached MGERASE_FFMPEG_THREADS=auto
export MGERASE_ULYSSES_HEAD_CHUNKS=1 MGERASE_ULYSSES_OUTPUT_OVERLAP=0

RUNTIME=(
  --model-path data/model
  --no-dit-layerwise-offload --text-encoder-cpu-offload --vae-cpu-offload
  --vae-low-memory --cuda-memory-limit-gib 22
  --runtime-mode windowed_streaming --streaming-cache-dtype uint8
  --infer-len 121 --overlap 9 --no-compact-tail-padding
  --attention-backend sdpa
  --operator-fusion-backend triton --operator-fusion-ops qk_rmsnorm_rope,rmsnorm_adaln
  --num-inference-steps 50 --strength .8 --guidance-scale 3 --seed 42
  --transformer-cache-mode off --no-cache-text-projections
  --warmup --warmup-steps 1
)
VIDEO=(
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4
  --prompt "There is a bridge over the lake."
)
```

### 3. 选择单卡、双卡或四卡

单卡使用 DiT 组件卸载：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  "${RUNTIME[@]}" "${VIDEO[@]}" --dit-cpu-offload --output-path outputs/l40s-1.mp4
```

双卡使用独立 NCCL CFG2：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  "${RUNTIME[@]}" "${VIDEO[@]}" --no-dit-cpu-offload \
  --dit-parallel-backend nccl --cfg-degree 2 --output-path outputs/l40s-2.mp4
```

四卡每个 CFG 分支用两卡做序列并行。 保留已验证的数值行为；输出通信重叠环境变量只作用于此命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MGERASE_ULYSSES_HEAD_CHUNKS=4 MGERASE_ULYSSES_OUTPUT_OVERLAP=1 \
uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  "${RUNTIME[@]}" "${VIDEO[@]}" --no-dit-cpu-offload \
  --dit-parallel-backend nccl --cfg-degree 2 --sp-degree 2 --sp-linear-mode reference \
  --output-path outputs/l40s-4.mp4
```

### 4. WebUI 与 HTTP 服务

沿用前面的环境设置，单卡启动：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m entrypoints.server.serve \
  --model-path data/model --task-root outputs/service --host 127.0.0.1 --port 30000 \
  --no-dit-layerwise-offload --dit-cpu-offload --text-encoder-cpu-offload --vae-cpu-offload \
  --vae-low-memory --cuda-memory-limit-gib 22 --runtime-mode windowed_streaming \
  --attention-backend sdpa --operator-fusion-backend triton \
  --operator-fusion-ops qk_rmsnorm_rope,rmsnorm_adaln --warmup --warmup-steps 2
```

打开 `http://127.0.0.1:30000/ui`，上传视频和mask，或使用画笔绘制静态擦除区域，填写背景提示词后提交。页面支持预览、进度、取消与结果下载；静态画笔不跟踪移动物体。
HTTP 请求、允许读取的本地路径及任务管理见 [服务API](docs/service_api.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [L40S验收记录](docs/l40s_validation_20261004.md) | 本轮完整条件、质量筛选、性能与显存结果 |
| [安装部署](docs/setup.md) | 依赖、模型、可选扩展、容器与排错 |
| [CLI说明](docs/cli.md) / [服务API](docs/service_api.md) | 参数、任务文件、服务接口 |
| [性能配置](docs/performance.md) | 卸载、attention、融合、缓存、量化与并行边界 |
| [架构](docs/architecture.md) / [配置](config/README.md) | 代码组织、执行流程与配置职责 |
| [文档索引](docs/README.md) / [路线图](docs/roadmap.md) | 历史实验与后续方向 |

## 参考仓库

- [SGLang multimodal_gen](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)：多模态推理运行时、服务与逐层卸载设计参考。

## 本仓库开发人员（按贡献度排名）

- [ZhiHeng66](https://github.com/ZhiHeng66)
- [balbalabal](https://github.com/balbalabal)
- [Alwaysssssss](https://github.com/Alwaysssssss)

## 📜 Citation

If you find our work helpful, please consider giving a star 🌟 and citation 📝

```
@article{liu2025eraserdit,
  title={EraserDiT: Fast Video Inpainting with Diffusion Transformer Model},
  author={Liu, Jie and Hui, Zheng},
  journal={arXiv preprint arXiv:2506.12853},
  year={2025}
}
```
