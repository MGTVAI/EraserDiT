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

### ✅ Completed
- [x] Experimental multi-GPU inference (CFG / SP / VAE / DP)
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

运行入口为 `inference_cli.sh` 和 `inference_server.sh`，采用 stages + 常驻 pipeline 架构。
先准备完整本地模型目录；以下命令从仓库根目录执行：

```bash
ERASERDIT_PYTHON="$(command -v python)" CUDA_VISIBLE_DEVICES=0 ./inference_cli.sh \
  --model-path results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/output.mp4 --prompt "There is a bridge over the lake." \
  --attention-backend sdpa
```

`results/cache_prediction_model` 是本机已校验的官方快照路径，其他机器需替换。
原尺寸推理通常需要约 60 GiB 空闲显存；可按性能文档选择动态卸载、编译或多卡配置。
启动脚本默认启用离线加载与 expandable segments，可通过同名环境变量覆盖。

| 文档 | 内容 |
| --- | --- |
| [CLI](docs/cli.md) | 单视频、批量任务、参数与多卡入口 |
| [服务 API](docs/service_api.md) | 部署、请求契约、任务生命周期与验收 |
| [性能与验收](docs/performance.md) | 配置限制、统一质量标准、实测结论与产物索引 |
| [辅助脚本](scripts/README.md) | 基准测量、功能验证、历史实验 |
| [回归测试](tests/README.md) | CPU / GPU 测试范围与运行方式 |

目录职责：`pipelines/runtime/` 管理视频窗口、调度与读写，`pipelines/session.py` 管理常驻会话；
`entrypoints/server/` 管理 HTTP API、任务队列与 worker；`config/service_contracts/` 定义模型请求契约。
容器入口见 `docker/base.dockerfile`，历史记录中尚未验证镜像构建与启动。

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
