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
- **两张卡处理一个视频**：NCCL CFG2 + VAE2，本轮单次请求 216.34 秒。
- **四张卡处理一个视频**：NCCL CFG2+SP2 + VAE4（aligned + 原生归约融合 + 自动分块），本轮无缓存／未量化配置中最快，单次请求 150.59 秒。

以上推荐均使用 BF16、SDPA、原始分辨率和帧数，关闭近似缓存、量化和尾窗减填充。
本轮多卡 VAE 测试采用每进程 44 GiB 分配器上限、单卡任务合计 46 GiB 验收目标；
实测最高约 30.51 GiB，不能沿用历史 VAE1 的 24 GiB 结论。

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
后续复测采用逐阶段累积选优：下一阶段必须继承上一阶段实测最快的完整配置，再叠加新候选；
候选未加速时保留上一阶段最佳配置。按 GPU 数分别维护最佳配置，以同条件完整请求耗时选优，
质量独立记录。基础优化、并行/VAE、缓存、量化依次推进；缓存阶段的胜出配置也必须带入量化阶段。
每次保存上一阶段配置及结果来源、本次改动、完整 CLI/环境参数和实际生效配置。
组合不兼容时先补齐执行支持并验证，再测试累积组合，不静默关闭已有优化。
下表中的固定组合用于既有结果与候选对照，后续累积组合另行标明继承来源和完整配置。
S0–S3、缓存、并行及量化均使用本轮冻结源码重新实测；B1 同时重新测量。
单卡量化 Q1–Q6 保留无缓存固定对照，四卡 Q7 使用本轮单卡实测最快量化配置。

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

B0 原始全驻留 BF16/SDPA 在正式请求中发生 OOM，保留此结论，不再复测。
S3 同时作为单卡无缓存／未量化固定对照；四卡缓存和量化共用 CFG2+SP2 无缓存／未量化对照，
不再重复列 C0/Q0。原测试编号保留，已移出的阈值扫描和并行消融不再占主表行。

并行复测统一当前优化实现，SP 路径采用 aligned、原生归约融合、direct 打包、自动 head 分块和
输出通信重叠，并核验各组合实际生效的配置。多卡 VAE 关闭组件卸载和 VAE tiling，保留 VAE low-memory；
共享主卡的 DiT rank 0 沿用窗口间卸载、窗口内驻留策略。
缓存组统一预热保护、连续复用上限和文本投影缓存策略；Q7 关闭缓存，仅叠加 Q6 的静态 FP8 FFN、
Sage FP8 attention、快速 QK RoPE / AdaLN / gated residual。
本轮补齐 NCCL 静态 FP8 FFN、Sage FP8 attention 和快速融合执行；各 rank 从 BF16 权重独立量化，实际调用与配置见本轮审计记录。
质量差异独立记录，不作为性能测试去留条件。

CFG2+SP2 是表中既定的四卡组合对照；后续缓存与量化累积复测必须使用四卡上一阶段的实测最佳配置，
不预先认定 CFG2+SP2 最快。
历史 VAE1 配置中，最新 SP4 请求为 178.03 秒，CFG2+SP2 加通信重叠为 196.58 秒；
这些成绩不代表本表 VAE4 配置的性能。

本表沿用单次正式请求口径，替代[旧复测方案](docs/performance_retest_plan_20261004.md)中的五次重复口径；
测试范围以本表为准。已有单卡数据、原始配置、资源采样、日志和输出保留在
`results/readme_retest_20261004/`。历史 VAE 对照与修复结果见
[失败项修复记录](docs/readme_failure_repair_20261005.md)，机器可读汇总及输出核验位于
`results/readme_repair_20261005/readme_results.json` 和 `completeness.json`。
历史 SP2 / SP4 阶段数据及源码、输出、显存核验位于 `outputs/sp_finish_20261006/`，
复现与验证见[最新 SP 验收](docs/sp_finish_20261006.md)。旧记录保留，不计作新组合的复测结果。

本轮完整记录位于 `results/readme_retest_20261006/`：`readme_results.json` 保存耗时，
`selection.json` 保存各阶段胜者，`audit.json` 保存配置、输出规格、资源核验和独立质量比较。
所有新值均为一次正式请求；质量指标为每 12 帧抽样、缩至 320×180 的解码 RGB 对比，
不是全分辨率质量验收。B0 保留历史 OOM，未重测。

按请求耗时选优：单卡基础 S3，双卡 CFG2，四卡 CFG2+SP2；单卡缓存阶段 C2，四卡缓存阶段 C7。

| 累积候选 | GPU / VAE | 继承来源与改动 | 1-step 预热 | 请求耗时 | 纯模型推理 | 文本编码 | VAE 编码 | DiT 去噪 | DiT 加速 | VAE 解码 | 纯推理加速 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1_cumulative | 1 / 1 | C2 + Q1，保留 TeaCache 0.1 | 72.04 | 251.72 | 224.29 | 1.04 | 16.44 | 195.92 | 2.25× | 10.87 | 2.08× |
| Q2_cumulative | 1 / 1 | C2 + Q2，保留 TeaCache 0.1 | 73.12 | 265.78 | 237.65 | 1.17 | 16.95 | 208.40 | 2.11× | 11.12 | 1.97× |
| Q3_cumulative | 1 / 1 | C2 + Q3，保留 TeaCache 0.1 | 73.31 | 253.91 | 224.18 | 1.05 | 16.94 | 195.24 | 2.25× | 10.94 | 2.09× |
| Q4_cumulative | 1 / 1 | C2 + Q4，保留 TeaCache 0.1 | 70.45 | 220.40 | 190.99 | 1.48 | 17.13 | 161.91 | 2.72× | 10.44 | 2.45× |
| Q5_cumulative | 1 / 1 | C2 + Q5，保留 TeaCache 0.1 | 71.51 | 201.45 | 171.99 | 1.33 | 16.64 | 143.48 | 3.07× | 10.54 | 2.72× |
| Q6_cumulative | 1 / 1 | C2 + Q6，保留 TeaCache 0.1 | 70.10 | 183.99 | 155.48 | 1.23 | 16.58 | 127.26 | 3.46× | 10.40 | 3.01× |
| Q7_cumulative | 4 / 4 | C7（四卡胜者）+ Q6 量化／快速融合，保留 TeaCache 0.1 | 51.18 | 93.68 | 65.41 | 0.54 | 8.68 | 49.45 | 8.90× | 6.74 | 7.15× |

完整 CLI/环境参数及生效配置见每项 `manifest.json` / `report.json`；较慢候选不替换上一阶段胜者。
单卡最终胜者为 Q6_cumulative（183.99 秒），四卡最终胜者为 Q7_cumulative（93.68 秒）；
二者均保留 TeaCache 0.1，并使用静态 FP8 FFN、Sage FP8 attention 和快速融合。
S0–S3 输出逐字节一致；新多卡 VAE、缓存、量化输出与 S3 有差异，独立质量记录见 `audit.json`。

## 快速开始

以下命令从仓库根目录执行，使用 **Bash**；运行配置使用数组，请在同一终端中执行。
通用 CLI 默认使用逐层卸载；以下命令显式选择已验证的组件卸载、融合和多卡模式。
下面快速开始保留 VAE1 和 22 GiB 每进程预算，与本轮主表的 VAE2/VAE4 配置不同；
主表复现使用各项 `manifest.json` 记录的完整命令和环境。

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

历史 L40S 验收的 CPU 回归253项中173通过、80按环境条件跳过；另完成 GPU 卸载、量化搬运、通信流水与故障恢复专项验证。
本轮 NCCL 扩展相关回归57项中22通过、35按环境条件跳过；真实四卡短视频以及正式 Q7/Q7_cumulative 均核验各 rank 的量化、attention 和快速融合实际调用，无回退。
正式性能测试冻结推理源码，记录输入/模型指纹、环境、逐卡进程采样、完整输出和 worker 分配。
历史四组单视频配置共 20 条请求的验收记录保留。此次完成 27 项正式复测，输出均为 1920×1080、145 帧、24000/1001 fps；当前运行时各项冻结源码一致，资源采样、退出状态与完整输出核验通过。

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
