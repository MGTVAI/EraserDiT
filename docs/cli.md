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
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。单视频入口不会隐式注入这些变量；DP dispatcher 会为子进程设置 `HF_HUB_OFFLINE=1`。
GPU 编号相对于 `CUDA_VISIBLE_DEVICES`；例如物理卡 2、3 对应 `cuda:0`、`cuda:1`。
原尺寸 1080×1920、121 帧窗口曾测得约 60 GiB 空闲显存需求，不能作为所有配置的固定门槛。

48 GB 单卡处理此示例时，可使用 `--no-dit-layerwise-offload --no-text-encoder-cpu-offload --no-vae-cpu-offload --vae-tiling`
降低 VAE 激活显存；只卸载权重不能避免整块 VAE 编码的显存不足。
单卡 tiling 可与卸载组合；属于近似路径，默认关闭。256/224 小块配置本轮未过画质门槛，
使用前需按素材验证，见 [迁移验证](sglang_memory_validation_20260924.md)。

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
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
SP1 的 INT8/FP8 DiT 也可使用图外融合；单卡支持量化与 DiT 卸载，多卡逐层卸载仍不支持。

增加 `--enable-torch-compile` 启用 FFN 编译。默认保留原生 Linear 数值边界；
允许近似计算时可为进程设置 `MGERASE_COMPILE_LINEAR_BACKEND=inductor`，
让 Inductor 优化 Linear/激活组合。该选项不保证任意素材达到质量门槛。
日志 `torch_compile.preparation_seconds_total` 为当前编译模型生命周期累计的新形状预热时间，
`preparation_history` 按窗口形状列出耗时；请求耗时仍包含本次发生的预热。

`--cache-text-projections` 可在 `--transformer-cache-mode off` 时单独开启，
但缓存命中不代表端到端提速，需计入它保存的 K/V 显存并实测。
验证记录见 [单卡方案实施](single_gpu_optimization_20260927.md)。

## 单卡 Attention 性能路径

L40S 长序列可将 Sage FP8 Attention、静态 FP8 FFN 与完整 Q/K RMSNorm + RoPE 融合组合：

```bash
--attention-backend sage_fp8 \
--transformer-quantization fp8_w8a8_static --quantization-scope ffn \
--operator-fusion-backend triton --operator-fusion-ops qk_rmsnorm_rope_fast \
--no-dit-layerwise-offload --no-dit-cpu-offload
```

`qk_rmsnorm_rope_fast` 是显式近似选项，使用 FP32 归约并保留归一化、权重乘法的 BF16 边界；
不保证逐元素一致，不能与 `qk_rmsnorm_rope` 同选。仅验证单卡推理，要求 BF16、宽度 2048、
Diffusers RMSNorm、eps=1e-5 和连续布局；`auto` 遇到不支持的契约时回退，`triton` 报错。
Sage FP8 需要支持该内核的 SM89 扩展，环境与完整视频结果见
[Attention 优化](attention_optimization_20261004.md)。短序列收益需单独测量。

继续融合非仿射 RMSNorm、AdaLN 和门控残差时，将算子列表改为：

```bash
--operator-fusion-ops qk_rmsnorm_rope_fast,rmsnorm_adaln_fast,gated_residual
```

`rmsnorm_adaln_fast` 接收原始 hidden states，使用一次 kernel 完成归一化与调制，
保留各 BF16 舍入边界，但允许 FP32 归约顺序变化；不能与 `rmsnorm_adaln` 同选。
要求单 batch、宽度 2048、BF16、非仿射 Diffusers RMSNorm、eps=1e-6；默认不启用。
整请求测量和 Q/K/V 合并筛选见 [AdaLN 与残差优化](adaln_optimization_20261004.md)。

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

## 单窗口性能基准

从示例视频和 mask 各无损截取前 121 帧，固定一个完整窗口、50 配置步数、
strength=0.8。每种配置的请求共用一个常驻 session，先用 2 步请求预热，再执行 5 次正式请求。
汇总纯推理与 DiT 去噪耗时的中位数和范围，不含加载、预热或视频读写。
输出目录必须不存在；按实际分配的 GPU 设置 `--devices`。

```bash
uv run --no-project python -m entrypoints.cli.benchmark_window \
  --run-dir outputs/window_reference --devices 1,2,3,6

# 同一输入和参数验证实验性 Ulysses 打包优化
uv run --no-project python -m entrypoints.cli.benchmark_window \
  --run-dir outputs/window_packed --devices 1,2,3,6 \
  --configs sp2,sp4,cfg2_sp2 --packing packed
```

默认对比 SP1、SP2、CFG2、SP4、CFG2×SP2；可用 `--configs` 选择子集。
`--prepare-only` 仅生成素材、任务及命令清单，不启动 GPU。
产物含 `manifest.json`、各配置日志、视频和 `summary.json`。
脚本检查每次正式推理只有一个去噪窗口且输出 121 帧。

`--text-cache-ab` 在同一常驻 session 内按 off/on、on/off 交替比较文本缓存，
`--repeats 5` 表示每种模式各五次；汇总单列纯推理、去噪、扣除预热的完整请求耗时和输出 SHA256。
manifest 保存当前 Python 源码指纹，包括未提交文件。

`--profile-step 2` 为每个窗口的第二个去噪步输出各 rank 的 CPU/CUDA trace 和算子汇总，
产物位于 run-dir 下的 `profiles/`。它仅用于瓶颈诊断，采集耗时不能用于速度对照。
普通 CLI 可设置 `MGERASE_DIT_PROFILE_DIR=/absolute/path` 与 `MGERASE_DIT_PROFILE_STEP=2`；
未设置目录时不启动 profiler、不安装模块计时 hooks。计数从 1 开始，短于目标步的窗口不采集。
trace 标记投影、norm、FFN、attention 和 Ulysses 打包/交换/重排；父子区间重叠，rank 并行，不能相加。

实验开关 `MGERASE_NCCL_PACKING=packed` 减少 Ulysses 发送缓冲区打包拷贝，
默认 `reference` 保持原路径。两者通信内容与顺序相同，不等长输出分片保留原打包路径。
已通过 CPU/真实 NCCL 精确检查，三个配置的单窗口 A/B 输出文件一致。
本轮纯 SP 观察到小幅提速，CFG2×SP2 未证明稳定收益，见[单窗口验证](window_optimization_20261002.md)。

## 常用参数

| 参数 | 默认值 / 说明 |
| --- | --- |
| `--seed` | `42` |
| `--num-inference-steps` / `--strength` | `50` / `0.8`，通常每窗口执行 40 个有效去噪步 |
| `--guidance-scale` | `3.0` |
| `--infer-len` / `--overlap` | `121` / `9`，窗口长度与重叠帧数 |
| `--dtype` | `bf16` |
| `--dit-layerwise-offload` | 默认开启，使用 SGLang 原生循环预取 |
| `--vae-low-memory` | 默认关闭；VAE 归一化、卷积分批并保留完整邻域，搭配默认卸载用于 [24 GiB 显存预算](memory24_20260930.md) |
| `--cuda-memory-limit-gib` | 默认不设置；限制每进程执行设备的 PyTorch 分配器，并在阶段边界回收缓存。NCCL CFG/Ulysses 共用主卡的 rank 0 仅在去噪窗口内驻留权重；整卡占用仍需 NVML 验收 |
| `--text-encoder-cpu-offload` / `--vae-cpu-offload` | 默认开启，分别使用 T5 FSDP 和 VAE 组件搬运 |
| `--pin-cpu-memory` | 默认开启；无 DiT 字节预算参数 |
| `--dit-offload-prefetch-size` | 默认 0，代表一层；[0,1) 为层数比例，≥1 为整数层数 |
| `--attention-backend` | `sdpa`；可选项以 `--help` 为准 |
| `--transformer-cache-mode` | `off`；可选 `teacache` / `cache_dit` |
| `--teacache-threshold` / `--cache-dit-residual-diff-threshold` | 均为 `0.3`，仅对应缓存模式启用时生效 |
| `--cache-text-projections` / `--no-cache-text-projections` | 默认 auto：残差缓存开启时复用文本投影，off 时不启用 |
| `--transformer-quantization` | `none`；实验选项 `int8_w8a8_native`、`fp8_w8a8_native`、`fp8_w8a8_tensorwise`、`fp8_w8a8_static`（固定激活 scale，快速累加） |
| `--quantization-scope` | `blocks`；可选 `ffn`、`ffn_up`（仅 FFN 升维层） |
| `--sage-fp8-accum-dtype` | `fp32+fp32`；实验选项 `fp32+fp16`，后者需兼容的 SageAttention2++ 构建 |
| `--sage-fp8-qk-quant-gran` | `per_thread`；可选 `per_warp`；非默认选项要求显式 `--attention-backend sage_fp8` |

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

两卡单任务：设置 `CUDA_VISIBLE_DEVICES=0,1`，关闭卸载后使用局部编译和缓存：

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
[NCCL 并行实施记录](distributed_parallel_20260928.md)。新路径目前要求 BF16、SDPA、常驻或 FSDP 分片的 DiT、
关闭编译/量化；T5/VAE CPU offload 可以保留。常驻 CFG/Ulysses 已支持限定融合和残差缓存，见本页末尾实验选项。
常驻 CFG/Ulysses 可增加 `--cache-text-projections`，每窗/CFG 分支独立保存文本投影与 K/V，
窗口结束清理；正常权重版本或条件变化会使缓存失效。TP、Ring、FSDP 暂不支持此缓存组合。

双卡 NCCL Ulysses 示例（保留默认 T5/VAE 卸载）：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/result_nccl_sp2.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --dit-parallel-backend nccl --sp-degree 2 --ulysses-degree 2 --sp-linear-mode sharded \
  --no-dit-layerwise-offload --no-dit-cpu-offload \
  --attention-backend sdpa --transformer-cache-mode off --no-cache-text-projections
```

该命令使用默认完整尾窗；历史整片验收使用 `--compact-tail-padding`，复现时需同时核对窗口配置。

多视频分配到不同 GPU 使用 DP dispatcher；`tasks.json` 沿用上面的格式：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_parallel \
  --model-path "$MODEL_DIR" --task-file tasks.json \
  --dp-degree 2 --parallel-run-dir results/dp_run \
  --transformer-cache-mode off --no-cache-text-projections
```

`parallel-run-dir` 必须尚不存在，DP 度数不能超过任务数，每项输出路径必须唯一。
DP worker 各自维护独立会话与请求缓存；缓存仍需满足该 worker 的卸载/compile/拓扑限制。
子进程强制离线加载。
同四卡 DP 的吞吐、单条延迟及内存取舍见[实测](dp_topology_20261002.md)。单任务并行参数及设备分组见
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

## VAE 阶段卸载（实验性）

启用 `--vae-cpu-offload` 时，可设置 `MGERASE_VAE_OFFLOAD_MODE`：

- `full`：默认，编码和解码阶段均搬运整个 VAE。
- `split`：仅搬运当前 encoder/decoder 和根参数/buffer。
- `cached`：在 split 基础上复用 CPU 权重存储，未修改的参数不回拷。

例如：`MGERASE_VAE_OFFLOAD_MODE=cached uv run --no-project python -m entrypoints.cli.erase_eraserdit ...`。
缓存遵循 CPU pin_memory 设置，仅支持 eval 推理；阶段内直接通过 `.data` 修改参数不受支持。
完整单窗口对照使用 `entrypoints.cli.benchmark_window --vae-offload-mode full|split|cached`。
内存口径、适用范围与结果见[内存优化验证](memory_optimization_20261002.md)。

CLI 擦除只消费输出文件，现会跳过 preload 路径的整窗 FP32 返回张量。
Python `EraseSession.run` 默认仍保留原返回行为；保存文件且不需要内存输出时，可传
`request_extra={"return_output_tensor": False}`。`save_output=False` 时仍返回张量。
详见[帧转换与输出缓存优化](frame_copy_optimization_20261002.md)。

### 2026-10-02 实验选项

- 常驻 NCCL CFG/Ulysses SP1/2/4 可组合 `--operator-fusion-backend triton` 与
  `MGERASE_NCCL_PACKING=direct`。支持 `qk_rmsnorm_rope,rmsnorm_adaln` 及显式选择的
  `qk_rmsnorm_rope_native,rmsnorm_adaln_native`，不含 gated residual 和近似 `*_fast`。
  后者保留原生归约、进一步融合逐元素计算；使用 `sp2_native_rms` / `sp4_native_rms`
  profile 复现，详见[原生归约融合](sp_native_rms_20261005.md)。
- `MGERASE_POSTPROCESS_CHUNKED_FP32=1` 按颜色校正块转 FP32，减少后处理临时激活。
- `MGERASE_VAE_INPLACE_ACTIVATIONS=1` 配合 `--vae-low-memory` 复用 VAE 临时激活，并分块计算下采样残差；保持原归约与舍入。
- `--vae-low-memory --vae-chunk-elements 16777216` 设置 VAE 分块目标元素预算；完整帧/邻域为下限。
- `--runtime-mode windowed_streaming --streaming-cache-dtype uint8` 保留源帧/提交帧的 uint8 精度，并与 preload 使用相同的 RGB mask 阈值；软 mask 需有界预扫描。
- `--cache-probe-metric mask_frame_max` 为 TeaCache/CacheDiT 增加逐帧 mask/边缘变化约束；默认 global。

默认值保持原行为。实测范围和验收状态见[本轮记录](fusion_memory_optimization_20261002.md)。

### NCCL GPU 边界传输（实验性）

在创建会话前设置 `MGERASE_DIT_BOUNDARY_TRANSPORT=cuda_ipc`，可避免 owner 与 DiT worker
之间每步预测的 CPU 中转；默认值 `cpu` 保持原路径。输入 GPU 必须与 rank 0 相同。
该实现仍有接收端 GPU 拷贝与同步，改变环境变量不会切换已有进程池。
验证范围、性能和生命周期约束见[边界传输记录](cuda_ipc_boundary_20261003.md)。

### SP2 / SP4 选择性 GEMM 保护（实验性）

`--sp-linear-mode aligned` 在已筛查形状上仅保护敏感投影，让其他 BF16 投影按本地 token
计算；默认仍为 `reference`。仅支持常驻 NCCL Ulysses SP2/4，可组合 CFG2，不支持
TP、Ring、FSDP、编译或量化。CLI 和服务启动参数均支持该模式。

当前加速范围为 L40S、PyTorch 2.6.0 / CUDA 12.6、确定性模式、EraserDiT 2048 宽度、
batch 1、32640 或 10200 token。其他环境、模型、形状或激活 dtype 回退为完整 reference
计算。首尾投影保留全长；SP4 的 10200-token FFN 下投影也保留全长。
每个 rank 的 `sp_linear_policy` 报告实际执行模式、回退原因和本地/保护调用次数；计数是
最近一次分支 forward 的包装调用次数，不是整个请求的 GEMM 数。

```bash
# 双卡完整视频基准；四卡改为 sp4_aligned 和 0,1,2,3。
MGERASE_ULYSSES_HEAD_CHUNKS=4 MGERASE_ULYSSES_OUTPUT_OVERLAP=1 \
uv run --no-project python -m entrypoints.cli.benchmark_l40s \
  --run-dir outputs/sp2_aligned --profile sp2_aligned --devices 0,1 --repeats 5
```

基准自动设置 direct 打包；直接调用推理 CLI 时还需 `MGERASE_NCCL_PACKING=direct`，并显式
设置 `--cfg-degree 1 --sp-degree 2`（四卡纯 SP 使用 `--sp-degree 4`）。长短序列应分别
筛选通信分块；微基准可用 `--variants fusion_direct,aligned,aligned_heads2_output,aligned_heads4_output`
交替测量并逐元素比较。

### Ulysses 通信计算重叠（实验性）

历史仅输入重叠候选在 L40S 整片筛选中变慢；输入/输出联合重叠需按拓扑和序列长度分别测量。

在常驻 NCCL Ulysses SP2/4 的命令前设置：

```bash
export MGERASE_NCCL_PACKING=direct
export MGERASE_ULYSSES_HEAD_CHUNKS=4
```

需要在创建进程池前设置；运行中的池不会动态切换。
`HEAD_CHUNKS` 可选 `auto` 或 1/2/4，默认 1 为原路径。`auto` 配合 direct 和输出重叠，
在已测 L40S 形状上为 SP4 短序列选两块，其他已测 SP2/SP4 形状选四块，未覆盖形状用一块。
详见[分块与补零优化](sp_finish_20261006.md)。分块需要 heads 能被 `SP × chunks` 整除，
不支持 Ring、TP、FSDP；当前仍要求 BF16/SDPA、关闭编译和量化。
默认只重叠输入交换。可额外设置 `MGERASE_ULYSSES_OUTPUT_OVERLAP=1`，
将每块输出交换与下一块 attention 重叠；需要实际选择的块数为 2 或 4。
输出重叠不作为通用默认配置。输入方案历史实测见[验收记录](ulysses_overlap_20261003.md)，
当前已验证配置见[性能说明](performance.md)。
