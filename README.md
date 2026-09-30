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
| **多卡单任务** | peer CFG/SP 与独立 NCCL DiT 后端，支持 Ulysses、Ring/USP、TP、FSDP/HSDP 及组合；VAE 编解码支持高度分片与逐层边界交换 |
| **多卡多任务** | DP dispatcher 将独立视频分配给不同 GPU worker 组 |
| **实验性量化** | DiT Linear INT8 W8A8，可选择 blocks 或 FFN 范围 |

默认使用 **BF16 + SDPA + 单卡权重卸载**。编译、融合、残差缓存、量化、VAE tiling 和尾窗减填充均需显式开启。

### 实测性能

使用 **A100 80GB** 处理同一段 **1920×1080、145 帧**视频，不降低输出分辨率、帧数或配置采样步数。
下表按逐步增加优化的路线整理实测结果。**耗时是从读取视频到保存结果的完整时间，不含模型加载；越小越好。**

| 步骤 | 增加了什么优化 | 通俗解释 | GPU 数 | 整片耗时 |
| --- | --- | --- | ---: | ---: |
| 1 | 单卡起点：按需加载权重 | 暂时不用的模型权重放回 CPU | 1 | 425.59 秒 |
| 2 | 优化中间数据的释放 | 用完就释放，减少显存里同时存放的数据 | 1 | 412.22 秒 |
| 3 | 复用固定信息、合并小算子 | 相同的信息少算几遍，连续的小计算一起做 | 1 | 391.85 秒 |
| 4 | 减少尾窗口填充 | 最后一段视频较短时，少计算补出来的帧 | 1 | 240.92 秒 |
| 5 | 增加双卡 CFG 并行 | 两张卡同时计算去噪的两个分支；DiT 权重改为常驻显存 | 2 | 141.05 秒 |
| 6 | 再加 TeaCache | 相邻步骤变化较小时，复用之前的计算结果 | 2 | 79.21 秒 |
| 7 | 扩展到四卡，保留 TeaCache | 每个分支再由两张卡分摊视频计算 | 4 | **58.81 秒** |

#### SP：DiT 去噪加速

SP（序列并行）让每张卡处理一部分视频特征。Ulysses 通过卡间交换数据分摊注意力计算，
Ring 让各卡轮流传递所需数据，USP 则把两种方式组合起来。
`CFG2×SP2` 表示两个去噪分支各用两张卡，共四张卡。

**加速倍数 = 对照的 DiT 去噪耗时 ÷ 优化后的 DiT 去噪耗时**。
这里只统计两个视频窗口的 DiT 去噪阶段，包含该阶段的并行通信，不含模型加载、VAE 编解码和视频前后处理。
例如 1.77× 表示去噪耗时减少约 44%；每行的对照配置单独列出。

| SP 方案 | 对照配置 | DiT 去噪耗时变化 | DiT 加速比 | 优化后 DiT 峰值显存（GiB） |
| --- | --- | --- | --- | --- |
| Ulysses：四卡 CFG2×SP2，peer 后端 | 双卡 CFG2，无缓存 | 102.288 → 57.761 秒 | **1.77×，耗时减少 43.5%** | 未单独统计 |
| Ulysses：四卡 CFG2×SP2，保留 TeaCache | 双卡 CFG2，同样使用 TeaCache | 36.682 → 20.479–20.504 秒 | **约 1.79×，耗时减少 44.1%–44.2%** | 未单独统计 |
| Ulysses：双卡 NCCL，优化通信和分片计算 | 同一双卡 Ulysses 的旧实现 | 149.81 → 122.39 秒 | **1.22×，耗时减少 18.3%** | 单个 DiT worker 最大 5.36 |
| Ring：双卡，收齐数据后计算（reference） | 缺少同条件性能对照 | 待测 | **尚不能给出可靠比例** | 未完整统计 |
| Ring：双卡，边传递边计算（streaming） | 缺少同条件性能对照 | 待测 | **尚不能给出可靠比例** | 未完整统计 |
| USP：四卡，Ulysses2×Ring2 | 缺少同条件性能对照 | 待测 | **尚不能给出可靠比例** | 未完整统计 |

前两行是增加 SP 和 GPU 数量后的观察值，来自不同轮次的同素材测试；多数只测一次，四卡缓存测了两次。
第三行取优化前后各四次预热后请求的 DiT 去噪中位数，表示**同一 SP 方案的实现优化收益**，不是双卡相对单卡的加速。
这些数据不能用于认定 Ulysses、Ring、USP 之间的速度排名。

显存统计为实际分配峰值（allocated）。peer 记录只有完整流程峰值，不能作为 DiT 阶段峰值填入表中；
NCCL 行统计 DiT worker 进程，不含负责 T5/VAE 的主进程，不代表整卡峰值。
上述无缓存 Ulysses 输出的整片 SSIM 为 0.99283，缓存组合为 0.97984；缓存会改变画面细节。
Ring reference、streaming 和 USP 已通过整片正确性验证，性能比例仍需补测；Ring online 未通过质量门槛。
原始条件见 [peer SP 测量](docs/four_gpu_optimization_20260927.md)和 [NCCL SP 验证](docs/distributed_parallel_20260928.md)。

#### VAE 并行加速

同一段完整视频，两组均使用 **双卡 CFG、全部模型权重常驻**，仅将 VAE 从单卡执行改为双卡按高度分片。
关闭缓存、融合、编译、量化、VAE tiling 和尾窗减填充；下表为同一会话中相邻两次预热后请求，阶段耗时为两个窗口之和。

| 阶段 | 单卡 VAE | 双卡 VAE | 变化 |
| --- | ---: | ---: | --- |
| VAE 编码 | 10.036 秒 | **5.587 秒** | 加速 **1.80×** |
| VAE 解码 | 5.886 秒 | **3.229 秒** | 加速 **1.82×** |
| DiT 去噪 | 187.739 秒 | 187.895 秒 | 基本不变 |
| 整片耗时 | 218.618 秒 | **211.730 秒** | 耗时减少 **3.15%** |

VAE 编解码合计节省约 **7.11 秒**；主要耗时仍在 DiT，因此局部约 1.8× 的加速没有等比例转化为整片加速。

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
