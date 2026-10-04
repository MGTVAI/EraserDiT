<h1 align="center">EraserDiT：面向 L40S 的视频擦除推理与服务</h1>

<p align="center">
  <a href="https://huggingface.co/jieeliu/EraserDiT">模型权重</a> ·
  <a href="https://github.com/JieLiu95/EraserDiT">原始算法</a> ·
  <a href="https://arxiv.org/abs/2506.12853">论文</a> ·
  <a href="https://jieliu95.github.io/EraserDiT_demo/">算法演示</a>
</p>

输入视频、对应掩码和背景提示词，擦除指定区域并生成背景。本仓库基于原始 EraserDiT，
提供 CLI、批量调度、HTTP API 与浏览器 WebUI；推理运行时参考 SGLang multimodal_gen，在仓库内实现，无需安装 SGLang。

| 输入视频 | 擦除结果 |
| :---: | :---: |
| ![天台上的人物](docs/113000356_input.gif) | ![擦除人物后的天台背景](docs/113000356_erased.gif) |

## L40S 如何选配置

- **一张卡**：BF16/SDPA、DiT 组件卸载，适合单机起步。
- **两张卡处理一个视频**：NCCL CFG2，同时计算两个去噪分支，降低单请求延迟。
- **四张卡处理一个视频**：CFG2×SP2 reference，加输出通信重叠，本轮单视频延迟最低。
- **多个独立视频**：使用 DP。四路单卡 worker 吞吐更高；两路双卡 CFG2 worker 的单请求延迟更低。

以下推荐命令均保留 BF16、原始分辨率和帧数，关闭近似缓存、量化及尾窗减填充。
通用 CLI 默认仍使用逐层卸载；下方 L40S 配置显式选择本轮验证过的组件卸载、融合和多卡模式。

## 实测性能

2026-10-04，同一台机器的 NVIDIA L40S，PyTorch 2.6.0+cu126。
输入为 `data/113000356.mp4`：1920×1080、145帧、24000/1001 fps；seed 42，CFG 3，
50步×strength 0.8（每窗口实际40步），121帧窗口、9帧重叠、完整尾窗填充。

### 单个视频延迟

每种配置连续测五次。请求耗时覆盖读取、推理与保存，**不含模型加载和预热**；
显存为加载至退出期间，同一物理卡上全部任务进程 NVML 占用之和的采样峰值。

| GPU数 | 配置 | 请求中位数 | 请求五次范围 | DiT去噪中位数 | DiT相对单卡加速 | 整片相对单卡加速 | 最高单卡任务显存 |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | BF16组件卸载 | 420.23 s | 418.91–420.54 s | 364.10 s | 1.00× | 1.00× | 20.682 GiB |
| 2 | NCCL CFG2 | 241.17 s | 240.02–242.36 s | 187.67 s | 1.94× | 1.74× | 21.523 GiB |
| 4 | CFG2×SP2 reference | 207.82 s | 204.69–208.74 s | 154.40 s | 2.36× | 2.02× | 21.746 GiB |
| 4 | 上一配置 + 输出通信重叠，head分块4 | **195.93 s** | **195.27–196.31 s** | **142.03 s** | **2.56×** | **2.14×** | **21.746 GiB** |

DiT耗时为每次请求两个窗口的去噪阶段之和，再取五次中位数，包含该阶段的多卡通信，
不含模型加载、预热、文本编码、VAE编解码及视频读写。单卡、双卡、推荐四卡的DiT五次范围分别为
363.91–364.40 s、187.34–187.92 s、141.44–142.33 s。
DiT加速比 = 单卡DiT去噪中位数 ÷ 对应配置DiT去噪中位数，使用未四舍五入的实测值计算。

加载 / 首次预热分别为：单卡33.88 / 79.61 s，双卡48.55 / 68.30 s，
推荐四卡55.97 / 65.47 s。首次使用新的环境可能还有不同的编译缓存准备成本。
输出通信重叠使本轮四卡整片中位数缩短5.72%，去噪阶段缩短8.02%；这两个比例的统计范围不同。

### 批量吞吐

相同输入和计算配置，任务在 worker 间均衡分配。冷批次吞吐包含加载、预热和退出；
热态估计以最慢 worker 的请求耗时之和计算，排除加载、预热、调度与退出，**不是持续到达负载测试**。

| GPU数 | worker配置 | 总请求数 | 请求中位数 | 冷批次吞吐（条/分钟） | 热态估计（条/分钟） | 最高单卡任务显存 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | DP2：两路单卡 | 6 | 421.57 s | 0.257 | 0.283 | 20.682 GiB |
| 4 | DP4：四路单卡 | 8 | 429.85 s | **0.472** | **0.552** | 20.682 GiB |
| 4 | CFG2×DP2：两路双卡 | 6 | **246.33 s** | 0.405 | 0.480 | 21.523 GiB |

### 质量与内存边界

七组正式矩阵共40条输出均与单卡 BF16 参考文件逐字节一致。
另有两个原始素材、seed 42/7 的质量验证，双卡 CFG2 与四卡 reference 均与各自 BF16 参考一致。
这证明已测路径的结果一致性，不保证任意素材的擦除效果或无闪烁。

`--cuda-memory-limit-gib 22` 是每进程 PyTorch 分配器预算；24 GiB 验收依据同卡任务进程合计实测，
包括 CUDA/NCCL 占用。离散采样不是连续峰值的数学上界。以上是 L40S 实测，尚不等同于24 GB消费卡的兼容性验证。
CPU 内存、输入缓存和输出空间随分辨率、帧数与 worker 数增加；CPU 卸载仍需要足够的主机内存。

本轮 Sage/FP8/快速融合、sharded SP 与 TeaCache/CacheDiT 候选未通过既定完整素材质量门槛，
保留为实验入口，推荐关闭。FFN compile 的候选收益不足以支持默认开启。
详细参数、质量分数、阶段耗时、测试范围和历史筛选见 [L40S验收记录](docs/l40s_validation_20261004.md)。

## 快速开始

以下命令从仓库根目录执行，使用 **Bash**；运行配置使用数组，请在同一终端中执行。

### 1. 安装依赖与下载模型

需要 Linux、NVIDIA 驱动、CUDA 12.6 Toolkit（含 `nvcc`）和 uv。
完整依赖包含需要构建的 FlashAttention；推荐 BF16/SDPA 无需额外构建 Sage FP8 扩展。
环境要求与排错见 [安装说明](docs/setup.md)。

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

模型完整下载后可离线运行。示例掩码中白色表示擦除、黑色表示保留；
自有视频掩码需匹配输入尺寸、帧率与帧数。提示词描述擦除后的背景。

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
  --warmup --warmup-steps 2
)
VIDEO=(
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4
  --prompt "There is a rooftop terrace overlooking the city at sunset."
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

四卡每个 CFG 分支用两卡做序列并行。`reference` 保留已验证的数值行为；
输出通信重叠环境变量只作用于此命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MGERASE_ULYSSES_HEAD_CHUNKS=4 MGERASE_ULYSSES_OUTPUT_OVERLAP=1 \
uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  "${RUNTIME[@]}" "${VIDEO[@]}" --no-dit-cpu-offload \
  --dit-parallel-backend nccl --cfg-degree 2 --sp-degree 2 --sp-linear-mode reference \
  --output-path outputs/l40s-4.mp4
```

### 4. 多视频批量处理

准备 `tasks.json`，每项包含 `video`、`mask`、`output`、`prompt`，可指定 `seed`。
输出路径必须互不相同，任务数至少等于 worker 数；完整 JSON 示例见 [CLI说明](docs/cli.md)。
沿用上面的 `RUNTIME` 数组与环境设置：

```bash
# 两卡，每卡独立处理视频
CUDA_VISIBLE_DEVICES=0,1 uv run --no-project python -m entrypoints.cli.erase_parallel \
  "${RUNTIME[@]}" --dit-cpu-offload --task-file tasks.json \
  --dp-degree 2 --parallel-run-dir outputs/dp2

# 四卡，四路独立单卡worker，优先批量吞吐
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-project python -m entrypoints.cli.erase_parallel \
  "${RUNTIME[@]}" --dit-cpu-offload --task-file tasks.json \
  --dp-degree 4 --parallel-run-dir outputs/dp4

# 四卡，两路双卡CFG2，优先降低每个视频的处理时间
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-project python -m entrypoints.cli.erase_parallel \
  "${RUNTIME[@]}" --no-dit-cpu-offload --dit-parallel-backend nccl --cfg-degree 2 \
  --task-file tasks.json --dp-degree 2 --parallel-run-dir outputs/cfg2-dp2
```

## WebUI 与 HTTP 服务

沿用前面的环境设置，单卡启动：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m entrypoints.server.serve \
  --model-path data/model --task-root outputs/service --host 127.0.0.1 --port 30000 \
  --no-dit-layerwise-offload --dit-cpu-offload --text-encoder-cpu-offload --vae-cpu-offload \
  --vae-low-memory --cuda-memory-limit-gib 22 --runtime-mode windowed_streaming \
  --attention-backend sdpa --operator-fusion-backend triton \
  --operator-fusion-ops qk_rmsnorm_rope,rmsnorm_adaln --warmup --warmup-steps 2
```

打开 `http://127.0.0.1:30000/ui`，上传视频和mask，或使用画笔绘制静态擦除区域，
填写背景提示词后提交。页面支持预览、进度、取消与结果下载；静态画笔不跟踪移动物体。
HTTP 请求、允许读取的本地路径及任务管理见 [服务API](docs/service_api.md)。
浏览器流程已验证；真实 L40S 的取消后恢复测试使用320×240、12帧输入，不作为1080p服务性能结论。

## 验证与复现

本轮 CPU 回归253项中173通过、80按环境条件跳过；另完成 GPU 卸载、量化搬运、通信流水与故障恢复专项验证。
正式性能测试冻结推理源码，记录输入/模型指纹、环境、逐卡进程采样、完整输出和 worker 分配。
七组40条正式请求的输出、视频规格、显存、退出状态与源码一致性核验均通过。

可用统一入口复测；每次选择新的结果目录，其他 profile 与 DP 参数见 [验证说明](docs/validation.md)：

```bash
uv run --no-project python -m entrypoints.cli.benchmark_l40s \
  --profile bf16_fused --devices 0 --repeats 5 --run-dir results/my-l40s-1
```

| 文档 | 内容 |
| --- | --- |
| [L40S验收记录](docs/l40s_validation_20261004.md) | 本轮完整条件、质量筛选、性能与显存结果 |
| [安装部署](docs/setup.md) | 依赖、模型、可选扩展、容器与排错 |
| [CLI说明](docs/cli.md) / [服务API](docs/service_api.md) | 参数、任务文件、服务接口 |
| [性能配置](docs/performance.md) | 卸载、attention、融合、缓存、量化与并行边界 |
| [架构](docs/architecture.md) / [配置](config/README.md) | 代码组织、执行流程与配置职责 |
| [文档索引](docs/README.md) / [路线图](docs/roadmap.md) | 历史实验与后续方向 |

## 来源、贡献者与许可

模型与视频擦除算法来自 [EraserDiT](https://github.com/JieLiu95/EraserDiT)。
运行时、服务和卸载设计参考 [SGLang multimodal_gen](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)。
本仓库许可证为 [Apache-2.0](LICENSE)。

本仓库开发人员（按贡献度排名）：[ZhiHeng66](https://github.com/ZhiHeng66)、
[balbalabal](https://github.com/balbalabal)、[Alwaysssssss](https://github.com/Alwaysssssss)。

```bibtex
@article{liu2025eraserdit,
  title={EraserDiT: Fast Video Inpainting with Diffusion Transformer Model},
  author={Liu, Jie and Hui, Zheng},
  journal={arXiv preprint arXiv:2506.12853},
  year={2025}
}
```
