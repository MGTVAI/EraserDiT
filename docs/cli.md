# 命令行推理

[服务 API](service_api.md) · [性能与验收](performance.md) · [安装与部署](setup.md)

以下命令从仓库根目录执行，先安装 [README](../README.md) 中的依赖。
`--model-path` 统一指向 `$PWD/data/model`；首次使用按 [模型准备说明](setup.md#模型与素材) 下载或复制完整权重。
FFmpeg 的 `ffmpeg` 和 `ffprobe` 必须在 `PATH` 中。

## 单视频

```bash
MODEL_DIR="$PWD/data/model"
VIDEO="$PWD/data/113000356.mp4"
MASK="$PWD/data/113000356_mask.mp4"
CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path "$MODEL_DIR" --video-input "$VIDEO" --mask-input "$MASK" \
  --output-path outputs/result.mp4 \
  --prompt "Describe the background after removal." --attention-backend sdpa
```

掩码视频需与源视频的帧数、尺寸和帧率对应；掩码标出待擦除区域，prompt 描述擦除后的背景。
通过 `uv run --no-project` 使用仓库 `.venv`，相对路径以仓库根目录为准。
如需离线加载，在命令前设置 `HF_HUB_OFFLINE=1`；如需调整显存分配器，设置
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。这些变量不由包装脚本隐式注入。
GPU 编号相对于 `CUDA_VISIBLE_DEVICES`；例如物理卡 2、3 对应 `cuda:0`、`cuda:1`。
原尺寸 1080×1920、121 帧窗口曾测得约 60 GiB 空闲显存需求，不能作为所有配置的固定门槛。

48 GB 单卡处理此示例时，可使用 `--no-dit-layerwise-offload --no-text-encoder-cpu-offload --no-vae-cpu-offload --vae-tiling`
降低 VAE 激活显存；只卸载权重不能避免整块 VAE 编码的显存不足。
单卡 tiling 可与卸载组合；属于近似路径，默认关闭。256/224 小块配置本轮未过画质门槛，
使用前需按素材验证，见 [迁移验证](sglang_memory_validation_20260924.md)。

```bash
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/result.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --attention-backend sdpa --no-dit-layerwise-offload --no-text-encoder-cpu-offload --no-vae-cpu-offload --vae-tiling
```

查看完整参数：

```bash
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-project python -m entrypoints.cli.erase_eraserdit --help
```

## 单卡精确融合

在原有单卡 BF16、SDPA、逐层卸载命令上增加 `--operator-fusion-backend auto`，
默认启用 QK RoPE 与 AdaLN 两个位置。额外的 gated residual 融合需显式选择：

```bash
--operator-fusion-backend auto \
--operator-fusion-ops qk_rmsnorm_rope,rmsnorm_adaln,gated_residual
```

`gated_residual` 保留乘法后的 BF16 舍入，合并乘法和残差加法；收益与输入形状有关，
暂不加入默认算子集合。`auto` 对不支持的布局或精度回退，`triton` 强制执行契约检查。
局部 FFN compile 可与上述图外融合及单卡逐层卸载组合，CUDA graphs 关闭。
SP1 的 INT8 常驻 DiT 也可使用图外融合；INT8 DiT 权重卸载、多卡 DiT 权重卸载仍不支持。

增加 `--enable-torch-compile` 启用 FFN 编译。默认保留原生 Linear 数值边界；
允许近似计算时可为进程设置 `MGERASE_COMPILE_LINEAR_BACKEND=inductor`，
让 Inductor 优化 Linear/激活组合。该选项不保证任意素材达到质量门槛。
日志 `torch_compile.preparation_seconds_total` 为当前编译模型生命周期累计的新形状预热时间，
`preparation_history` 按窗口形状列出耗时；请求耗时仍包含本次发生的预热。

`--cache-text-projections` 可在 `--transformer-cache-mode off` 时单独开启，
但缓存命中不代表端到端提速，需计入它保存的 K/V 显存并实测。
验证记录见 [单卡方案实施](single_gpu_optimization_20260927.md)。

## 尾窗口减填充

需要减少部分尾窗口的计算时，可显式增加 `--compact-tail-padding`。
它保留所有真实输入帧、重叠区和采样步数，只减少非首窗口的多余镜像填充；
首窗口和完整窗口不变。属于近似选项，默认关闭，要求 `infer_len` 和 `overlap` 均为 `8k+1`。
API 请求使用 `"compact_tail_padding": true`，任务 JSON 可按任务覆盖该字段。
画质与尾窗口长度相关，整段 SSIM 达标不等于每帧达标；见 [99% 质量目标验证](quality99_optimization_20260927.md)。

## 批量任务

默认使用 `data/113000356.mp4` 和 `data/113000356_mask.mp4`，下面以两个 seed 演示批量任务。创建 `tasks.json`，顶层为非空数组。任务内字段覆盖公共 CLI 采样参数；每项使用独立的 `id` 和输出路径。
相对路径以运行时工作目录为准。

```json
[
  {
    "id": "terrace_seed42",
    "video": "data/113000356.mp4",
    "mask": "data/113000356_mask.mp4",
    "output": "outputs/113000356_seed42.mp4",
    "prompt": "There is a rooftop terrace overlooking the city at sunset.",
    "seed": 42
  },
  {
    "id": "terrace_seed43",
    "video": "data/113000356.mp4",
    "mask": "data/113000356_mask.mp4",
    "output": "outputs/113000356_seed43.mp4",
    "prompt": "There is a rooftop terrace overlooking the city at sunset.",
    "seed": 43
  }
]
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m entrypoints.cli.erase_eraserdit --model-path "$MODEL_DIR" \
  --task-file tasks.json --attention-backend sdpa
```

任务顺序共用一个常驻 session。`--warmup` 仅为首个任务执行预热，不保证覆盖后续所有输入形状。
标准输出末尾包含 `{"tasks": [...]}` 报告，记录输出路径、加载、预热、请求耗时及内存等指标。

## 常用参数

| 参数 | 默认值 / 说明 |
| --- | --- |
| `--seed` | `42` |
| `--num-inference-steps` / `--strength` | `50` / `0.8`，通常每窗口执行 40 个有效去噪步 |
| `--guidance-scale` | `3.0` |
| `--infer-len` / `--overlap` | `121` / `9`，窗口长度与重叠帧数 |
| `--dtype` | `bf16` |
| `--dit-layerwise-offload` | 默认开启，使用 SGLang 原生循环预取 |
| `--text-encoder-cpu-offload` / `--vae-cpu-offload` | 默认开启，分别使用 T5 FSDP 和 VAE 组件搬运 |
| `--pin-cpu-memory` | 默认开启；无 DiT 字节预算参数 |
| `--dit-offload-prefetch-size` | 默认 0，代表一层；[0,1) 为层数比例，≥1 为整数层数 |
| `--attention-backend` | `sdpa`；可选项以 `--help` 为准 |
| `--transformer-cache-mode` | `off`；可选 `teacache` / `cache_dit` |
| `--teacache-threshold` / `--cache-dit-residual-diff-threshold` | 均为 `0.3`，仅对应缓存模式启用时生效 |
| `--cache-text-projections` / `--no-cache-text-projections` | 默认 auto：残差缓存开启时复用文本投影，off 时不启用 |
| `--transformer-quantization` | `none`；实验选项 `int8_w8a8_native` |

全部参数运行 `uv run --no-project python -m entrypoints.cli.erase_eraserdit --help`；任务 JSON 使用下划线命名，例如 `infer_len`。
模型加载、设备、编译和量化等进程配置通过 CLI 设置。

## 加速与多卡

单卡全驻留可启用注意力与局部 FFN 编译；近似后端仍需按素材验收：

```bash
--no-dit-layerwise-offload --no-text-encoder-cpu-offload --no-vae-cpu-offload \
--attention-backend sage_attn --enable-torch-compile --warmup
```

预热和编译有一次性成本，冷启动不保证净加速。卸载、缓存、并行、量化的组合限制与测量条件见
[性能说明](performance.md)。

两卡单任务：设置 `CUDA_VISIBLE_DEVICES=6,7`，关闭卸载后使用局部编译和缓存：

```bash
--no-dit-layerwise-offload --no-text-encoder-cpu-offload --no-vae-cpu-offload --cfg-degree 2 --enable-torch-compile \
--attention-backend sage_attn --transformer-cache-mode teacache
```

无近似缓存使用 `--attention-backend sdpa --transformer-cache-mode off --no-cache-text-projections`。
SP 使用 `--sp-degree 2 --cfg-degree 1 --sp-linear-mode sharded`；
迁移前的历史验收数据见 [可组合加速验收](composable_acceleration_validation_20260923.md)。
服务入口也支持同名的 SP/CFG、VAE 与量化进程参数。

独立 NCCL DiT 进程池使用 `--dit-parallel-backend nccl`，新增 TP、Ulysses×Ring、
FSDP/HSDP 正交分组。参数组合、限制和整片 SSIM 验收状态见
[NCCL 并行实施记录](distributed_parallel_20260928.md)。新路径目前要求 SDPA、常驻 DiT、
关闭编译/量化/融合/缓存；T5/VAE CPU offload 可以保留。

多视频分配到不同 GPU 使用 DP dispatcher；`tasks.json` 沿用上面的格式：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_parallel \
  --model-path "$MODEL_DIR" --task-file tasks.json \
  --dp-degree 2 --parallel-run-dir results/dp_run \
  --transformer-cache-mode off --no-cache-text-projections
```

`parallel-run-dir` 必须尚不存在。DP 提升多任务吞吐，单任务并行参数及设备分组见
[并行说明](performance.md#parallel)。


## 完整 DiT 编译（实验性）

`--enable-torch-compile --torch-compile-scope transformer` 将一次完整 DiT forward
交给 Inductor：输入/文本/时间投影、全部 attention/FFN/norm/调制/残差与输出投影。
原有 `--torch-compile-scope ffn` 仍为默认值。

在现有单卡命令上使用：

```bash
--enable-torch-compile --torch-compile-scope transformer \
--no-dit-layerwise-offload --no-dit-cpu-offload \
--attention-backend sdpa --operator-fusion-backend disabled \
--transformer-cache-mode off --no-cache-text-projections
```

支持 CFG1/SP1 和 CFG2/SP1，要求未量化且常驻的 DiT；T5/VAE CPU offload 可以保留。
双卡在上述参数上增加 `--cfg-degree 2 --sp-degree 1 --parallel-devices 0,1`，
并通过 `CUDA_VISIBLE_DEVICES` 选择两张物理卡。每卡编译完整 DiT，CFG 合成与设备搬运保持 eager。
原片测量与冷/热启动收益见 [双卡完整编译验证](cfg_compile_20260928.md)。
暂不支持 SP>1、旧式 `cfg_parallel_device`、FA/Sage、自定义融合、DiT 卸载、残差/文本投影缓存。
使用 `fullgraph=True`：断图或编译失败直接报错，不静默回退。
CUDA Graph 仍关闭；视频 I/O、VAE、scheduler、CFG 合成和窗口外 RoPE 准备保持 eager。
`MGERASE_COMPILE_LINEAR_BACKEND` 只影响 FFN 模式，完整模式使用 Inductor Linear。

首次执行每种形状会发生编译；`torch_compile.first_call_history` 记录首次调用耗时
（包含计算与编译），`successful_forwards` 记录成功执行次数。
CFG2 下 `rank_compile` 分别记录两卡的首次调用和成功次数，顶层成功次数为两卡合计；
顶层 `first_call_history` 仍表示主卡。每个窗口第一步依次执行两卡，避免同时首次追踪，
后续步骤并发运行；编译入口跨窗口和请求保留。
`applied` 代表已注册编译入口，并不意味着已经执行；实际成功要结合该计数判断。
首次成本计入请求/去噪时间，不能直接视为稳态性能；更改窗口形状可能重新编译。
编译可能改变 BF16 舍入，完整视频质量需单独验收。

## 编译 T5 与 VAE 组件（实验性）

使用 `--compile-components` 单独选择辅助组件，与 DiT 的 `--enable-torch-compile`
及 `--torch-compile-scope` 相互独立，默认均不启用。
[组合验证](component_compile_20260928.md)中三组件同时开启的 smoke SSIM 低于 0.98，
以下全部开启仍为实验用法；仅 `vae_decoder` 与完整 DiT 编译的组合已通过
[原片整段 0.98 检查](decoder_full_validation_20260928.md)：

```bash
# 优先试验仅 VAE 解码器，保留现有 DiT/T5 设置和 VAE CPU offload
--compile-components vae_decoder

# 实验性：VAE 编码与解码
--compile-components vae_encoder,vae_decoder

# 同时编译 T5 与 VAE，需要 T5 常驻 GPU
--compile-components text_encoder,vae_encoder,vae_decoder --no-text-encoder-cpu-offload
```

可以只选 `vae_encoder` 或 `vae_decoder`，也可叠加完整 DiT 编译。
VAE 编译的是原始 encoder/decoder forward；后验分布、采样、切片/分块调度以及
阶段间 CPU/GPU 搬运留在图外。单卡 VAE CPU offload 可保留，多卡 VAE 暂不支持。
T5 仍保留 tokenizer 与请求内静态缓存；整图 T5 编译暂不支持 FSDP CPU offload。

各组件使用 `fullgraph=True, dynamic=False`、关闭 CUDA Graph。编译错误直接抛出。
沿用 `MGERASE_TORCH_COMPILE_MODE`，支持 `default` 和 `max-autotune-no-cudagraphs`。
当前 Torch 2.6 下 T5 的精度转换模拟触发编译器错误，所以仅 T5 使用标准 Inductor
精度语义；VAE 保留精度转换模拟。组合输出仍需独立质量验收。

CLI 的 `timing.extra.component_compile` 和服务任务的 `metrics.component_compile`
包含组件级执行次数与首次输入形状
调用耗时，后者包含编译和执行，已计入请求耗时。T5 一次请求通常只执行两次，
且 VAE 每窗口只调用少数次，首次编译成本未必能在一个视频中摊薄。
