<h1 align="center">
  <span style="color:#2196f3;"><b>EraserDiT</b></span>: Fast Video Inpainting with Diffusion Transformer Model
</h1>

<p align="center">
  <a href="https://huggingface.co/jieeliu/EraserDiT"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/JieLiu95/EraserDiT"><img alt="Github" src="https://img.shields.io/badge/EraserDiT-github-black"></a>
  <a href="https://arxiv.org/abs/2506.12853"><img alt="arXiv" src="https://img.shields.io/badge/EraserDiT-arXiv-b31b1b"></a>
  <a href="https://jieliu95.github.io/EraserDiT_demo/"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</p>

---

## 🗺️ Open-Source Roadmap

### 🛠️ In Progress
- [ ] Gradio demo
- [ ] Multi-GPU inference support

### ✅ Completed
- [x] Single-GPU inference
- [x] Model weights release
- [x] Paper publication

---

## 🚀 Overview

**EraserDiT:** Interactively removes specified objects and automatically generates the corresponding prompts. It processes a 2K‑resolution video (2160×2100, 97 frames) in only 65 seconds on a single NVIDIA H800 GPU without any acceleration. Experiments show strong performance in content fidelity, texture restoration, and temporal consistency.


---
## 🎯 Install dependencies

```
pip install -r requirements.txt
```

The pinned environment uses Python 3.10, Torch 2.6.0/cu126, and
FlashAttention 2.8.3. Create a fresh environment and install Torch first so
FlashAttention can build against the already-installed Torch:

```
conda create -n EraserDiT python=3.10 -y
conda activate EraserDiT
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 triton==3.2.0 \
  --extra-index-url https://download.pytorch.org/whl/cu126
CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH \
  MAX_JOBS=8 pip install --no-build-isolation -r requirements.txt
```
---
## 🧸 Inference

EraserDiT requires >60GB GPU memory for a 2K‑resolution video.
An experimental dual-GPU CFG CLI path is available; see
[parallel validation and commands](docs/performance_cfg_parallel.md).
With `CUDA_VISIBLE_DEVICES=2,3`, add `--cfg-parallel-device cuda:1`
to compute the two CFG branches concurrently. Keep `--transformer-cache-mode off`
and `--no-cache-text-projections`; this version requires `fullgpu` without torch.compile.

单卡 INT8 W8A8 实验入口：`--transformer-quantization int8_w8a8_native`，见 [量化说明](docs/performance_quantization.md)。

CFG/SP/VAE/DP 组合入口与测试矩阵见 [并行说明](docs/performance_parallel.md)。
`--cfg-degree 2` 或 `--sp-degree 2` 使用两卡去噪；`--vae-degree 2`
在 VAE 阶段复用两卡并交换卷积边界。SP 默认保留原矩阵形状以控制 BF16 数值漂移；
额外启用 `--vae-tiling` 会切换为近似分块，需单独验收质量。
多视频用 `python -m entrypoints.cli.erase_parallel --dp-degree 2 ...`。
四卡测试脚本 `scripts/parallel_four_gpu.sh` 仅在 2、3、6、7 号卡全部空闲时启动。

本仓库已迁移到 MGErase 组合式架构（stages + 常驻 pipeline + 服务端骨架），运行入口是
`inference_cli.sh` 与 `inference_server.sh`。原版单体入口 `inference.py` 及其专用的
加载、预处理、后处理和工具代码已删除。

目录职责：`pipelines/runtime/` 管理视频窗口、调度与读写，`pipelines/session.py` 管理常驻会话；
`entrypoints/server/` 管理 HTTP API、任务队列与 worker；`config/service_contract.py` 和
`config/service_contracts/` 定义公共及各模型的服务请求契约。

后续 CLI、服务和性能复测统一使用以下已校验模型路径：

```bash
SNAP=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model
```

该目录的权重通过软链接指向共享存储，16 个模型文件（约 29.24 GB）已与官方版本
`jieeliu/EraserDiT@904fb412da76235085dbbccaefdbde4979fa3d29` 校验一致。
校验记录见 [availability_verified.json](results/cache_prediction_model/availability_verified.json)。

两个启动脚本内部已默认 `HF_HUB_OFFLINE=1`（加载本地快照时直连 huggingface.co 会静默卡住）与
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（VAE 编码要一整块 7.5 GiB 连续显存，默认
分配器会把它卡在自有空闲块后面）。两者都可用同名环境变量覆盖。

### CLI

单条素材（推荐加速配置，去噪 1.22×、端到端 1.17–1.44×，两组素材均过门禁）：

```
CUDA_VISIBLE_DEVICES=6 ./inference_cli.sh \
  --model-path "$SNAP" \
  --video-input data/10268234.mp4 \
  --mask-input  data/10268234_mask.mp4 \
  --output-path results/out_10268234.mp4 \
  --prompt "There is a bridge over the lake." \
  --attention-backend sage_attn --enable-torch-compile --warmup
```

第二组素材（横屏、2 窗口，提示词不同）：

```
CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh \
  --model-path "$SNAP" \
  --video-input data/113000356.mp4 \
  --mask-input  data/113000356_mask.mp4 \
  --output-path results/out_113000356.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --attention-backend sage_attn --enable-torch-compile --warmup
```

多任务共用常驻 pipeline（预热只付一次，两种画幅可混用）：

```
CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh --model-path "$SNAP" \
  --task-file tasks/acceptance_tasks.json \
  --attention-backend sage_attn --enable-torch-compile --warmup
```

不加速的对照（矩阵里的 N）：

```
CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh --model-path "$SNAP" \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/out_N.mp4 --prompt "There is a bridge over the lake." \
  --attention-backend sdpa
```

### 服务端

```
CUDA_VISIBLE_DEVICES=2 ./inference_server.sh \
  --pipeline-name EraserDiTErasePipeline \
  --model-path "$SNAP" \
  --task-root /tmp/mgerase_tasks \
  --input-allowed-root "$PWD/data" \
  --port 30000 \
  --attention-backend sage_attn --enable-torch-compile --warmup
```

另开终端验收（全部端点 + 严格契约 + 任务生命周期 + 结果下载与删除）：

```
python3 scripts/service_smoke.py --base-url http://127.0.0.1:30000 \
  --video data/10268234.mp4 --mask data/10268234_mask.mp4 \
  --prompt "There is a bridge over the lake."
```

或一键起服务 + 验收（自动等一张空闲卡，无论成败都拆干净）：

```
scripts/service_verify.sh 2
```

### 注意

- **`--warmup` 是加速达标的必要条件**，不是可选优化：不预热则 Inductor 自动调优落进正式
  任务的首次前向，去噪收益从 17% 掉到 10%，过不了 15% 门槛。代价是每进程一次性付出，
  单任务冷启动不保证净收益（取决于素材形状数）
- `--pipeline-name` 不能省，服务端靠它选管线；`--input-allowed-root` 是输入白名单，
  必须是绝对路径
- 该卡有常驻租户（约 18.5 GiB），跑前确认空闲显存 ≥60 GiB；`inference_cli.sh` 启动时打印
  空闲显存并在低于 60 GiB 时告警。峰值在 VAE 编码（一整块 7.5 GiB），余量不足先在这里 OOM
- 换解释器用 `ERASERDIT_PYTHON=<path>`，脚本默认写死 conda 环境路径

### 文档

| 文件 | 内容 |
| --- | --- |
| `docs/m3_report.md` | 单卡加速测量矩阵、推荐配置、组合限制 |
| `docs/m4_report.md` | 验收矩阵、质量指标、连续任务验收、条件与未验证项 |
| `docs/service_api.md` | 服务端接口契约与验收方式 |
| `docker/base.dockerfile` | 运行镜像（本机无 docker，构建与启动未验证） |

---
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

### 后续性能优化与整组件卸载

开发顺序与验收标准见 [性能优化计划](vibe/performance_plan.md)：内存卸载 → TeaCache / cache-dit → 量化。

EraserDiT CLI 和通用服务入口可设置 `--resource-policy component_offload`：权重从 CPU 加载，
仅在文本编码、VAE 编解码、完整去噪阶段使用时搬到 GPU，阶段退出（包括异常）后返回 CPU。
默认仍为 `fullgpu`。整组件卸载不降低 VAE 激活本身的峰值，且会增加传输耗时。
当前请关闭 `--enable-torch-compile`。EraserDiT 也已接入 `dynamic_offload`：通过
`--max-weight-usage` 指定管理权重的预算（字节，默认 5 GiB），预算足够时整组件驻留，
不足时按块异步搬运。预算不包含激活和少量未包装权重；每个块保留 pinned CPU 镜像。
`--pin-memory` 可额外固定小层权重。结果 JSON 的 `memory_runtime` 提供实际驻留与搬运统计。

```bash
CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/offload.mp4 --prompt "There is a bridge over the lake." \
  --resource-policy component_offload
```
