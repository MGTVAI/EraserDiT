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
- **两张卡处理一个视频**：NCCL CFG2，同时计算两个去噪分支。
- **四张卡处理一个视频**：最新 SP4（aligned + 原生归约融合 + 自动分块）在本素材单次实测中请求耗时最低，为 178.03 秒；配置与复现见[最新 SP 验收](docs/sp_finish_20261006.md)。

以上推荐均使用 BF16、SDPA、原始分辨率和帧数，关闭近似缓存、量化和尾窗减填充。

## 性能数据与复测结果

固定使用 `data/113000356.mp4` 及对应 mask：1920×1080、145 帧、24000/1001 fps；
prompt 为 `There is a bridge over the lake.`，seed 42、CFG 3、50 步×strength 0.8
（每窗口实际 40 步）、121 帧窗口、9 帧重叠、完整尾窗填充。

每个测试使用独立常驻会话：模型加载后显式预热 1 次，预热中的 DiT 只运行 1 步，
随后正式运行 1 次。请求耗时包含读取、预处理、推理、后处理和保存，不含模型加载与显式预热。
下表耗时单位为秒，已有数据均为单次实测，不是平均值或中位数；各阶段数据来自同一次正式请求，
不能用分项之和反推请求时间。完整保留预热、请求、纯模型推理、文本编码、VAE 编码、DiT 去噪和 VAE 解码耗时。
“DiT 加速”和“纯推理加速”分别以 B1 的对应耗时除以本项耗时计算，B1 均为 1.00×。

本轮并行仅保留 SP2、CFG2、SP4、CFG2+SP2，双卡统一 VAE2、四卡统一 VAE4；
TeaCache / CacheDiT 测试阈值统一为 0.1，各保留单卡并新增 CFG2+SP2 组合。
这些是本表的测试配置约定，不代表 CLI / 服务默认值已修改。
S0–S3 和单卡缓存保留旧值供参考，待同一份最新源码复测；新并行配置不沿用旧配置耗时。
单卡量化 Q1–Q6 继续保留，四卡量化只新增当前单卡实测最快的 Q6 组合。

| 测试 | GPU | VAE | 配置 | 状态 | 1-step 预热 | 请求耗时 | 纯模型推理 | 文本编码 | VAE 编码 | DiT 去噪 | DiT 加速（B1=1.00×） | VAE 解码 | 纯推理加速（B1=1.00×） |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 1 | 1 | 原始算法，全驻留 BF16/SDPA | OOM（正式请求，不再复测） | 28.07 | OutOfMemoryError（request） | — | — | — | — | — | — | — |
| B1 | 1 | 1 | 原始算法，CPU offload + VAE tiling | 既有实测 | 42.18 | 1809.25 | 472.16 | 0.16 | 16.51 | 444.82 | 1.00× | 10.68 | 1.00× |
| S0 | 1 | 1 | 当前源码，preload + 逐层卸载，关闭融合 | 待复测（旧值） | 68.68 | 482.56 | 452.93 | 1.13 | 16.46 | 424.45 | 1.05× | 10.88 | 1.04× |
| S1 | 1 | 1 | S0 + 窗口流式 I/O + uint8 帧缓存 | 待复测（旧值） | 72.73 | 482.26 | 453.22 | 1.22 | 17.30 | 424.17 | 1.05× | 10.53 | 1.04× |
| S2 | 1 | 1 | S1 改为 DiT 组件卸载 | 待复测（旧值） | 73.78 | 479.79 | 450.74 | 1.06 | 16.35 | 422.03 | 1.05× | 11.30 | 1.05× |
| S3 | 1 | 1 | S2 + 精确 QK RoPE / AdaLN 融合 | 待复测（旧值） | 73.01 | 421.00 | 391.84 | 1.06 | 16.30 | 364.07 | 1.22× | 10.41 | 1.20× |
| SP2 | 2 | 2 | NCCL SP2 + VAE2 | 待复测 | — | — | — | — | — | — | — | — | — |
| CFG2 | 2 | 2 | NCCL CFG2 + VAE2 | 待复测 | — | — | — | — | — | — | — | — | — |
| SP4 | 4 | 4 | NCCL SP4 + VAE4 | 待复测 | — | — | — | — | — | — | — | — | — |
| CFG2+SP2 | 4 | 4 | NCCL CFG2+SP2 + VAE4 | 待复测 | — | — | — | — | — | — | — | — | — |
| C2 | 1 | 1 | TeaCache 0.1 | 待复测（旧值） | 74.85 | 258.56 | 230.79 | 1.07 | 16.53 | 202.76 | 2.19× | 10.43 | 2.05× |
| C5 | 1 | 1 | CacheDiT 0.1 | 待复测（旧值） | 73.31 | 268.69 | 240.96 | 1.09 | 16.33 | 213.08 | 2.09× | 10.47 | 1.96× |
| C7 | 4 | 4 | CFG2+SP2 + TeaCache 0.1 | 待测 | — | — | — | — | — | — | — | — | — |
| C8 | 4 | 4 | CFG2+SP2 + CacheDiT 0.1 | 待测 | — | — | — | — | — | — | — | — | — |
| Q1 | 1 | 1 | INT8 FFN + SDPA + 精确融合 | 待复测（旧值） | 73.50 | 410.05 | 381.41 | 1.07 | 16.66 | 352.38 | 1.26× | 11.30 | 1.24× |
| Q2 | 1 | 1 | 动态 FP8 FFN + SDPA + 精确融合 | 待复测（旧值） | 72.81 | 432.50 | 403.54 | 1.43 | 16.78 | 374.81 | 1.19× | 10.51 | 1.17× |
| Q3 | 1 | 1 | 静态 FP8 FFN + SDPA + 精确融合 | 待复测（旧值） | 71.96 | 407.28 | 378.70 | 1.05 | 16.31 | 350.86 | 1.27× | 10.47 | 1.25× |
| Q4 | 1 | 1 | Q3 + Sage FP8 attention | 待复测（旧值） | 71.16 | 344.46 | 317.46 | 1.04 | 16.38 | 289.57 | 1.54× | 10.46 | 1.49× |
| Q5 | 1 | 1 | Q4 + 快速 QK RoPE | 待复测（旧值） | 70.75 | 313.27 | 285.13 | 1.34 | 16.72 | 256.57 | 1.73× | 10.49 | 1.66× |
| Q6 | 1 | 1 | Q5 + 快速 AdaLN / gated residual | 待复测（旧值） | 68.49 | 285.25 | 256.04 | 1.33 | 16.62 | 226.59 | 1.96× | 11.50 | 1.84× |
| Q7 | 4 | 4 | CFG2+SP2 + Q6 完整量化／快速融合组合 | 待支持后测试 | — | — | — | — | — | — | — | — | — |

B0 原始全驻留 BF16/SDPA 在正式请求中发生 OOM，保留此结论，不再复测。
S3 同时作为单卡无缓存／未量化对照；四卡缓存和量化共用 CFG2+SP2 无缓存／未量化对照，
不再重复列 C0/Q0。原测试编号保留，已移出的阈值扫描和并行消融不再占主表行。

并行复测统一当前优化实现，SP 路径采用 aligned、原生归约融合、direct 打包、自动 head 分块和
输出通信重叠，并核验各组合实际生效的配置。多卡 VAE 关闭组件卸载和 VAE tiling，保留 VAE low-memory；
共享主卡的 DiT rank 0 沿用窗口间卸载、窗口内驻留策略。
缓存组统一预热保护、连续复用上限和文本投影缓存策略；Q7 关闭缓存，仅叠加 Q6 的静态 FP8 FFN、
Sage FP8 attention、快速 QK RoPE / AdaLN / gated residual。
当前 NCCL 路径尚不支持这套量化、attention 和快速融合组合，需补齐执行支持并验证后再测，不能直接套用单卡数据。
质量差异独立记录，不作为性能测试去留条件。

CFG2+SP2 是本轮缓存与量化的四卡组合基线，其是否最快需同条件复测确认。
历史 VAE1 配置中，最新 SP4 请求为 178.03 秒，CFG2+SP2 加通信重叠为 196.58 秒；
这些成绩不代表本表 VAE4 配置的性能。

本表沿用单次正式请求口径，替代[旧复测方案](docs/performance_retest_plan_20261004.md)中的五次重复口径；
测试范围以本表为准。已有单卡数据、原始配置、资源采样、日志和输出保留在
`results/readme_retest_20261004/`。历史 VAE 对照与修复结果见
[失败项修复记录](docs/readme_failure_repair_20261005.md)，机器可读汇总及输出核验位于
`results/readme_repair_20261005/readme_results.json` 和 `completeness.json`。
历史 SP2 / SP4 阶段数据及源码、输出、显存核验位于 `outputs/sp_finish_20261006/`，
复现与验证见[最新 SP 验收](docs/sp_finish_20261006.md)。旧记录保留，不计作新组合的复测结果。

## 快速开始

以下命令从仓库根目录执行，使用 **Bash**；运行配置使用数组，请在同一终端中执行。
通用 CLI 默认使用逐层卸载；以下命令显式选择已验证的组件卸载、融合和多卡模式。

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

四卡每个 CFG 分支用两卡做序列并行。 保留已验证的数值行为；
输出通信重叠环境变量只作用于此命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MGERASE_ULYSSES_HEAD_CHUNKS=4 MGERASE_ULYSSES_OUTPUT_OVERLAP=1 \
uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  "${RUNTIME[@]}" "${VIDEO[@]}" --no-dit-cpu-offload \
  --dit-parallel-backend nccl --cfg-degree 2 --sp-degree 2 --sp-linear-mode reference \
  --output-path outputs/l40s-4.mp4
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
四组单视频正式配置共 20 条请求的输出、视频规格、显存、退出状态与源码一致性核验均通过。

可用统一入口复测；每次选择新的结果目录，其他 profile 参数见 [验证说明](docs/validation.md)：

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
