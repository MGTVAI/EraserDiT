# 命令行推理

[服务 API](service_api.md) · [性能与验收](performance.md) · [辅助脚本](../scripts/README.md)

以下命令从仓库根目录执行，先安装 [README](../README.md) 中的依赖。
`--model-path` 指向完整本地模型目录。本机已校验快照为 `results/cache_prediction_model`，
对应 `jieeliu/EraserDiT@904fb412da76235085dbbccaefdbde4979fa3d29`；其他机器需准备自己的权重目录。

## 单视频

```bash
export ERASERDIT_PYTHON="$(command -v python)"
SNAP="$PWD/results/cache_prediction_model"
CUDA_VISIBLE_DEVICES=0 ./inference_cli.sh \
  --model-path "$SNAP" \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/output.mp4 \
  --prompt "There is a bridge over the lake." \
  --attention-backend sdpa
```

`ERASERDIT_PYTHON` 可覆盖启动脚本内的本机 conda 解释器路径。
脚本默认设置 `HF_HUB_OFFLINE=1` 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
并保留已有环境变量。GPU 编号相对于 `CUDA_VISIBLE_DEVICES`；例如物理卡 2、3 对应 `cuda:0`、`cuda:1`。
原尺寸 1080×1920、121 帧窗口通常需约 60 GiB 空闲显存，实际峰值随配置与素材变化。

## 批量任务

创建 `tasks.json`，顶层为非空数组。任务内字段覆盖公共 CLI 采样参数；每项使用独立的 `id` 和输出路径。
相对路径以运行时工作目录为准。

```json
[
  {
    "id": "bridge",
    "video": "data/10268234.mp4",
    "mask": "data/10268234_mask.mp4",
    "output": "results/bridge.mp4",
    "prompt": "There is a bridge over the lake.",
    "seed": 42
  },
  {
    "id": "terrace",
    "video": "data/113000356.mp4",
    "mask": "data/113000356_mask.mp4",
    "output": "results/terrace.mp4",
    "prompt": "There is a rooftop terrace overlooking the city at sunset.",
    "seed": 42
  }
]
```

```bash
CUDA_VISIBLE_DEVICES=0 ./inference_cli.sh --model-path "$SNAP" \
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

全部参数运行 `./inference_cli.sh --help`；任务 JSON 使用下划线命名，例如 `infer_len`。
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
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 python -m entrypoints.cli.erase_parallel \
  --model-path "$SNAP" --task-file tasks.json \
  --dp-degree 2 --parallel-run-dir results/dp_run \
  --transformer-cache-mode off --no-cache-text-projections
```

`parallel-run-dir` 必须尚不存在。DP 提升多任务吞吐，单任务并行参数及设备分组见
[并行说明](performance.md#parallel)。
