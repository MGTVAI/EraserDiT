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

本仓库基于 [原始 EraserDiT 算法](https://github.com/JieLiu95/EraserDiT)，提供**更低显存、更快推理和可部署的视频擦除服务**：输入视频、掩码与背景提示词，即可通过 CLI、批量任务或 HTTP API 完成擦除。

实现参考 SGLang 的 `python/sglang/multimodal_gen`，可理解为面向 EraserDiT 的 **mini SGLang 多模态推理运行时**。所需能力在仓库内实现，便于在**算法环境中维护依赖兼容性**，无需安装 SGLang；模块划分清晰，便于**阅读、调试、二次开发**。

### 已实现的优化

| 方向 | 能力 |
| --- | --- |
| **降低显存** | DiT 逐层预取与卸载、T5 FSDP CPU offload、VAE 组件卸载；窗口调度、帧缓存释放和可选 VAE 分块 |
| **加速单卡计算** | SDPA / FlashAttention / SageAttention；Triton QK RoPE、AdaLN 与残差融合；FFN、完整 DiT 及 T5/VAE 编译 |
| **减少重复计算** | 文本编码、RoPE 与文本投影复用；TeaCache / CacheDiT 残差缓存；可选尾窗口减填充 |
| **多卡单任务** | peer CFG/SP 与独立 NCCL DiT 后端，支持 Ulysses、Ring/USP、TP、FSDP/HSDP 及组合；保留 VAE 编解码并行 |
| **多卡多任务** | DP dispatcher 将独立视频分配给不同 GPU worker 组 |
| **实验性量化** | DiT Linear INT8 W8A8，可选择 blocks 或 FFN 范围 |

默认使用 **BF16 + SDPA + 单卡权重卸载**。编译、融合、残差缓存、量化、VAE tiling 和尾窗减填充均需显式开启。

### 实测性能

以下完整视频测试使用仓库示例 `data/113000356.mp4` 及对应掩码，
环境为 **A100 80GB、PyTorch 2.6.0+cu126、1920×1080 / 145 帧**，
seed=42、50 个配置步、strength=0.8、窗口 121 帧 / 重叠 9 帧，每窗口实际去噪 40 步。
**请求耗时包含视频处理与输出，不含模型加载；去噪是请求中的一个阶段。**
显存为进程的 PyTorch peak allocated，另行标明 reserved 或参数存储的除外；逐卡峰值不能相加当作同时总峰值。
各表对应不同实验轮次，单次筛选不能视为稳定排名。

#### 单卡显存与融合（2026-09-27）

SDPA、DiT 逐层卸载、T5/VAE CPU offload，关闭残差缓存、编译和尾窗减填充。

| 配置 | 请求 s ↓ | 去噪 s ↓ | allocated GiB ↓ | reserved GiB ↓ |
| --- | ---: | ---: | ---: | ---: |
| 显存生命周期优化前 | 425.59 | — | 34.882 | 56.502 |
| 显存生命周期优化后 | 412.22 | — | 30.653 | 49.240 |
| 再加静态条件复用 | 415.33 | 366.80 | 30.653 | 48.387 |
| 再加 QK RoPE / AdaLN 融合 | **391.85** | **340.45** | **30.653** | **48.387** |

显存峰值降低约 **12.1%**。显存优化后相对优化前的整段 RGB SSIM 为 **0.996333**；
静态条件复用未引入额外像素差异，融合前后输出逐像素一致。各项仅测一次，部分测量存在共享主机争用。
真实形状 DiT forward 的五次交替测量中，融合使常驻 / 逐层卸载中位耗时分别减少 **7.63% / 7.70%**。

#### 单卡注意力与 FFN 编译（2026-09-27）

沿用单卡卸载与融合，开启尾窗减填充。Sage 为独立构建的 **2.2.0**，不是依赖清单中的 1.0.6。

| 配置 | 请求 s ↓ | 去噪 s ↓ | allocated GiB ↓ | RGB SSIM ↑ |
| --- | ---: | ---: | ---: | ---: |
| SDPA | 240.915 | 201.709 | 30.653 | 0.992833 |
| SageAttention 2 | 235.614 | 197.404 | 30.653 | 0.982647 |
| FlashAttention 2 | 238.248 | 200.219 | 30.653 | 0.982683 |
| FlashAttention 2 + Inductor FFN | 241.725 | 202.793 | 30.653 | 0.982654 |
| SageAttention 2 + Inductor FFN | 250.095 | 205.719 | 30.653 | 0.982664 |

每项单次筛选；两项 FFN 编译请求分别包含 **4.699 / 6.509 s** 的形状预热，未观察到完整请求净加速。
本表及后续有 SSIM 的表格，除另有说明外，均以未减填充的 SDPA 融合输出为参考；整段均值不代表每帧质量。

#### 单卡 TeaCache / CacheDiT（2026-09-29）

DiT 常驻、T5/VAE 卸载，开启尾窗减填充；使用 SDPA，关闭编译、融合、量化和文本投影缓存。
两种缓存阈值均为 0.3，最多连续复用 1 步；CacheDiT 前 1 / 后 0 个 block 实算。

| 配置 | 请求 s ↓ | 去噪 s ↓ | allocated GiB ↓ | 复用分支步 |
| --- | ---: | ---: | ---: | ---: |
| 无缓存 | 246.282 | 217.439 | 34.029 | 0 / 160 |
| TeaCache | 146.685 | 119.898 | 34.029 | 72 / 160 |
| CacheDiT | 150.558 | 123.286 | 34.029 | 72 / 160 |

本轮请求耗时分别减少 **40.44% / 38.87%**，缓存持有张量峰值分别为 **765 / 1020 MiB**。
每项一次，无缓存为会话首个请求；检查了全部 145 帧的缩放区域图，未见明显人物残留，未做原分辨率实时播放。
本轮未设 SSIM 门槛，缓存命中率不等同于擦除质量保证。

#### 双卡 / 四卡 peer 并行、缓存与 INT8（2026-09-27）

DiT 常驻、T5/VAE 主卡卸载，开启融合和尾窗减填充，不开启编译。
此处缓存使用更激进的设置：阈值 0.3、最多连续复用 **3 步**；CacheDiT 前 1 / 后 1 个 block 实算。

| 配置 | GPU 数 | 请求 s ↓ | 去噪 s ↓ | 各卡 allocated GiB ↓ | RGB SSIM ↑ |
| --- | ---: | ---: | ---: | --- | ---: |
| CFG · SDPA | 2 | 141.052 | 102.288 | 34.045 / 5.760 | 0.992833 |
| CFG · SageAttention 2 | 2 | 140.505 | 102.228 | 34.045 / 5.792 | 0.982647 |
| CFG · TeaCache | 2 | 79.206 | 36.682 | 34.045 / 6.323 | 0.979842 |
| CFG · CacheDiT | 2 | 79.690 | 41.129 | 34.045 / 6.697 | 0.981373 |
| INT8 blocks · 无缓存 | 1 | 243.027 | 203.746 | 32.502 | 0.981043 |
| CFG · TeaCache · INT8 blocks | 2 | 75.322 | 37.113 | 32.502 / 5.323 | 0.978960 |
| CFG2×SP2 · 原通信 | 4 | 98.365 | 59.176 | — | 0.992833 |
| CFG2×SP2 · 大张量直写 | 4 | 96.151 | 57.761 | 34.045 / 5.170 / 5.170 / 5.170 | 0.992833 |
| CFG2×SP2 · TeaCache，首轮 | 4 | **59.611** | **20.504** | 34.045 / 5.421 / 5.421 / 5.421 | 0.979842 |
| CFG2×SP2 · TeaCache，复跑 | 4 | **58.810** | **20.479** | 同上 | 0.979842 |

双卡 SDPA 相对前轮单卡 240.915 s 的观测加速约 **1.71×**；四卡无缓存相对双卡约 **1.47×**。
缓存复用 **108 / 160（67.5%）** 分支步。INT8 使选中 Linear 的权重存储约减半，
但没有证明独立的去噪提速；TeaCache + INT8 的请求差异主要来自其他阶段。
除四卡 TeaCache 两次外均为单次筛选；缓存与量化以基本擦除效果验收，SSIM 仅诊断，已抽查六帧，未做逐帧播放。

#### 完整 DiT 与 VAE 解码器编译（2026-09-28）

DiT 常驻、T5/VAE 卸载、SDPA、尾窗减填充；编译配置关闭手工融合和 Transformer 缓存。
首次请求包含编译成本，预热后请求另列。

| 配置 | GPU 数 | 请求 s ↓ | 去噪 s ↓ | 测量方式 |
| --- | ---: | ---: | ---: | --- |
| 完整 DiT 编译 | 1 | 310.927 | 253.962 | 首次请求 |
| 完整 DiT 编译 | 2 | 253.567 | 208.461 | CFG2，首次请求 |
| eager，无手工融合 | 2 | 162.781 | 112.644 | 同会话两次中位数 |
| 完整 DiT 编译 | 2 | **143.360** | **92.115** | 预热后两次中位数，请求 142.994 / 143.726 s |
| eager，手工融合 | 2 | 150.910 | 103.583 | 独立进程，预热后单次对照 |
| 完整 DiT + VAE decoder 编译 | 1 | 301.703 | 217.318 | 另一会话，首次请求 |
| 完整 DiT + VAE decoder 编译 | 1 | 224.084 | 175.062 | 同会话重复请求 |
| 仅完整 DiT 编译 | 1 | 224.128 | 175.116 | 同会话重复请求对照 |

双卡完整编译相对同会话 eager 请求中位耗时减少 **11.93%**，相对手工融合单次对照减少 **5.00%**；
编译后主 / 副卡 allocated 约 **34.108 / 5.829 GiB**。单卡 / 双卡完整编译的整段 SSIM 分别为 **0.982325 / 0.982309**。
追加 decoder 编译将解码阶段从 **7.368 降至 6.330 s**，但完整请求仅差 **0.044 s**，未证明端到端收益；
组合 SSIM 为 **0.982273**，重复请求 allocated / reserved 为 **34.046 / 62.381 GiB**。

#### NCCL 并行与通信优化（2026-09-28 / 29）

BF16、SDPA、DiT 常驻或 FSDP 分片、T5/VAE 卸载、尾窗减填充，关闭编译、融合、量化和全部 Transformer 缓存。
下表均为同轮次、同条件的双卡 Ulysses2 前后对照；两轮的“优化前”不是同一版本。

| 指标 | 9 月 28 日优化前 → 后 | 9 月 29 日优化前 → 后 |
| --- | --- | --- |
| 优化内容 | 通信组复用、QKV 打包、序列分片计算 | owner 输出汇聚、worker 内 CFG 合并 |
| 请求测量次数 | 各 5 次，首个单列，后 4 次统计 | 各 3 次，首个单列，后 2 次统计 |
| 首个请求 s | 203.13 → 168.25 | 173.95 → 170.76 |
| 后续请求中位数 s | **203.05 → 175.34** | 174.14 → 170.20 |
| 后续请求范围 s | 202.53–205.74 → 171.10–176.12 | 171.83–176.45 → 167.89–172.51 |
| 去噪中位数 s | **149.81 → 122.39** | 122.331 → 122.192 |
| worker 最大 allocated GiB | 5.734 → 5.360 | 5.360 → 5.360 |
| worker 最大 reserved GiB | 9.736 → 10.049 | 10.049 → 10.068 |
| owner 最大 allocated GiB | 30.446 → 30.446 | 30.446 → 30.446 |
| 整片返回预测张量 MiB | — | **1673.44 → 836.72** |
| 整段 RGB SSIM | 0.992833050 | 0.992833050 |

9 月 28 日组合优化的请求中位耗时减少 **13.65%**，去噪减少 **18.31%**。
9 月 29 日返回张量字节数减少 **50%**、前后输出文件一致；请求范围重叠，去噪仅差 0.139 s，尚未证明稳定端到端提速。

其他 NCCL 四卡组合的单次整片结果：

| 配置 | 请求 s ↓ | RGB SSIM ↑ |
| --- | ---: | ---: |
| CFG2×SP2 sharded | 122.50 | 0.992833050 |
| USP4 streaming × FSDP4 + SP sharded | 204.19 | 0.992944427 |
| TP2 aligned × SP2 sharded | 291.95 | 0.992833050 |

分片后的实际参数存储如下，仅统计 DiT 参数，不含激活、T5/VAE 或通信缓冲区：

| 配置 | 每 rank 的 DiT 参数 GiB ↓ |
| --- | ---: |
| CFG2，完整 DiT 副本 | 3.583 |
| TP2 aligned | 1.800 |
| FSDP2 / HSDP shard2 | 1.832 |
| USP4 streaming + FSDP4 | 0.956 |

NCCL 已验收的 CFG、Ulysses、Ring reference/streaming、TP reference/aligned、FSDP/HSDP、DP 及组合均达到整片 **SSIM ≥0.985**；
Ring online 与纯 sharded TP 候选未达标。多卡能力与单次耗时不代表所有策略都能提速。
完整配置、测量日志说明与历史实验统一收录在[文档索引](docs/README.md)。

## 快速开始

### 1. 安装

需要 Linux、NVIDIA GPU、CUDA 12.6 开发工具链（含 `nvcc`）和已安装的 `uv`。
统一依赖包含 FlashAttention、SageAttention 和接口测试依赖；环境准备见 [安装说明](docs/setup.md)。

```bash
sudo apt-get install -y git curl build-essential ffmpeg libgl1 libglib2.0-0

git clone https://github.com/MGTVAI/EraserDiT.git
cd EraserDiT
uv venv --python 3.10
# FlashAttention 构建需要环境中已有 Torch 和构建工具
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  torch==2.6.0+cu126 pip setuptools wheel packaging ninja psutil
MAX_JOBS=4 uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --no-build-isolation-package flash-attn -r requirements.txt
uv pip check --python .venv/bin/python
```

### 2. 下载模型

从 [Hugging Face](https://huggingface.co/jieeliu/EraserDiT) 下载完整模型到 `data/model/`：

```bash
HF_HUB_OFFLINE=0 uv run --no-project hf download jieeliu/EraserDiT \
  --revision 904fb412da76235085dbbccaefdbde4979fa3d29 \
  --local-dir data/model \
  --exclude ".DS_Store"
```

### 3. 运行

以下命令使用仓库中的示例视频和掩码，结果写入 `outputs/result.mp4`：

```bash
mkdir -p outputs
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/result.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --attention-backend sdpa \
  --dit-layerwise-offload --text-encoder-cpu-offload --vae-cpu-offload
```

## 常用优化配置

双卡 peer CFG 全驻留示例（SageAttention 和 TeaCache 均需按素材检查质量）：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/result_cfg2.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --attention-backend sage_attn --no-dit-layerwise-offload --no-text-encoder-cpu-offload --no-vae-cpu-offload \
  --cfg-degree 2 --enable-torch-compile \
  --transformer-cache-mode teacache --teacache-threshold 0.3 --cache-text-projections
```

## 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.server.serve \
  --pipeline-name EraserDiTErasePipeline --model-path data/model \
  --task-root outputs/service --input-allowed-root "$PWD/data" \
  --host 127.0.0.1 --port 30000
```

## 文档

完整使用指南与按日期归档的验证记录见[文档索引](docs/README.md)。

| 文档 | 内容 |
| --- | --- |
| [安装与部署](docs/setup.md) | 统一依赖安装、模型下载、容器和排错 |
| [命令行推理](docs/cli.md) | 单视频、批量任务、参数与多卡入口 |
| [服务 API](docs/service_api.md) | 服务配置、请求、任务和结果 |
| [性能配置](docs/performance.md) | 注意力、编译、卸载、缓存、并行与量化 |
| [配置说明](config/README.md) | 模型、服务与运行配置 |
| [代码架构](docs/architecture.md) | 模块职责、执行流程与依赖边界 |
| [测量与验证](docs/validation.md) | 环境检查、CPU 回归、完整视频质量与性能对照 |
| [逐层卸载](docs/layerwise_offload.md) | SGLang 源码迁移、预取和组件管理 |

## 下一步工作

消费级显卡适配、WebUI / ComfyUI、单卡与多卡性能、缓存质量调优和 FP8 量化。具体计划见 [开发路线](docs/roadmap.md)。

## 参考仓库

- [EraserDiT](https://github.com/JieLiu95/EraserDiT)：模型与视频擦除算法。
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
