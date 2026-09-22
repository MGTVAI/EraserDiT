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

48 GB 单卡处理此示例时，可使用 `--resource-policy fullgpu --vae-tiling`
降低 VAE 激活显存；`dynamic_offload` 只卸载权重，不能避免整块 VAE 编码的显存不足。
当前分块模式要求 `fullgpu`，保留原始分辨率和帧数，但与整块 VAE 的数值不完全相同。

```bash
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/result.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --attention-backend sdpa --resource-policy fullgpu --vae-tiling
```

查看完整参数：

```bash
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --no-project python -m entrypoints.cli.erase_eraserdit --help
```

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
| `--resource-policy` | `fullgpu`；也可选整组件或动态卸载 |
| `--attention-backend` | `sdpa`；可选项以 `--help` 为准 |
| `--transformer-cache-mode` | `off`；可选 `teacache` / `cache_dit` |
| `--cache-text-projections` / `--no-cache-text-projections` | 默认 auto：残差缓存开启时复用文本投影，off 时不启用 |
| `--transformer-quantization` | `none`；实验选项 `int8_w8a8_native` |

全部参数运行 `uv run --no-project python -m entrypoints.cli.erase_eraserdit --help`；任务 JSON 使用下划线命名，例如 `infer_len`。
模型加载、设备、编译和量化等进程配置通过 CLI 设置。

## 加速与多卡

单卡已测配置可将 `--attention-backend sdpa` 替换为：

```bash
--attention-backend sage_attn --enable-torch-compile --warmup
```

预热和编译有一次性成本，冷启动不保证净加速。卸载、缓存、并行、量化的组合限制与测量条件见
[性能说明](performance.md)。

两卡单任务：设置 `CUDA_VISIBLE_DEVICES=0,1`，添加 `--cfg-degree 2`，关闭编译及缓存：

```bash
--cfg-degree 2 --transformer-cache-mode off --no-cache-text-projections
```

多视频分配到不同 GPU 使用 DP dispatcher；`tasks.json` 沿用上面的格式：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_parallel \
  --model-path "$MODEL_DIR" --task-file tasks.json \
  --dp-degree 2 --parallel-run-dir results/dp_run \
  --transformer-cache-mode off --no-cache-text-projections
```

`parallel-run-dir` 必须尚不存在。DP 提升多任务吞吐，单任务并行参数及设备分组见
[并行说明](performance.md#parallel)。
