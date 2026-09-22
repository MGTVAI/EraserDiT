<h1 align="center">
  <span style="color:#2196f3;"><b>EraserDiT</b></span>: Fast Video Inpainting with Diffusion Transformer Model
</h1>

<p align="center">
  <a href="https://huggingface.co/jieeliu/EraserDiT"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/JieLiu95/EraserDiT"><img alt="Github" src="https://img.shields.io/badge/EraserDiT-github-black"></a>
  <a href="https://arxiv.org/abs/2506.12853"><img alt="arXiv" src="https://img.shields.io/badge/EraserDiT-arXiv-b31b1b"></a>
  <a href="https://jieliu95.github.io/EraserDiT_demo/"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</p>

## 简介

EraserDiT 根据视频、掩码和背景提示词擦除指定区域，支持单视频、批量任务和异步 HTTP 服务。
可按需求启用注意力加速、编译、权重卸载、Transformer 缓存及多 GPU 并行。

## 快速开始

需要 Linux x86_64、NVIDIA GPU 和支持 CUDA 12.6 的驱动。先确认 `nvidia-smi` 正常。
Python 3.10 和依赖由 [uv](https://docs.astral.sh/uv/) 管理，默认使用无需编译扩展的 SDPA。

### 1. 安装

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/MGTVAI/EraserDiT.git
cd EraserDiT
uv venv --python 3.10
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
uv pip check --python .venv/bin/python
```

已有代码时直接进入仓库目录执行 uv 命令。私有仓库需要先配置 GitHub 访问权限。
依赖同时使用 PyPI 和 PyTorch CUDA 源，安装命令按版本约束从两个源选择包。

### 2. 下载模型

从 [Hugging Face](https://huggingface.co/jieeliu/EraserDiT) 下载完整模型到 `data/model/`：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
uv run --no-project hf download jieeliu/EraserDiT \
  --revision 904fb412da76235085dbbccaefdbde4979fa3d29 \
  --local-dir data/model \
  --exclude ".DS_Store"
```

模型约 30 GB，另需为依赖、临时帧缓存和输出预留空间。下载中断后可重新执行同一命令。

### 3. 运行

以下命令使用仓库中的示例视频和掩码，结果写入 `outputs/result.mp4`：

```bash
mkdir -p outputs
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/result.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --attention-backend sdpa \
  --resource-policy dynamic_offload
```

替换视频和掩码即可处理自己的素材；掩码标出待擦除区域，prompt 描述擦除后的背景。
所有命令从仓库根目录执行，`uv run --no-project` 使用 `.venv`，无需手动激活。
显存需求随分辨率、窗口及配置变化；原尺寸视频建议使用大显存 GPU，卸载配置见[性能说明](docs/performance.md#offload)。

## 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.server.serve \
  --pipeline-name EraserDiTErasePipeline --model-path data/model \
  --task-root outputs/service --input-allowed-root "$PWD/data" \
  --host 127.0.0.1 --port 30000
```

启动后访问 `http://127.0.0.1:30000/docs` 查看交互式 API。
上传、查询进度、下载和取消任务见[服务 API](docs/service_api.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [安装与部署](docs/setup.md) | 新机器安装、模型下载、可选依赖、容器和排错 |
| [命令行推理](docs/cli.md) | 单视频、批量任务、参数与多卡入口 |
| [服务 API](docs/service_api.md) | 服务配置、请求、任务和结果 |
| [性能配置](docs/performance.md) | 注意力、编译、卸载、缓存、并行与量化 |
| [测量与验证](docs/validation.md) | 输出质量、性能测量和异常恢复 |
| [回归测试](tests/README.md) | CPU / GPU 测试范围与运行方式 |
| [代码架构](docs/architecture.md) | 模块职责、依赖方向与执行流程 |
| [配置说明](config/README.md) | 模型、服务与运行配置 |

## 参考仓库

- [EraserDiT](https://github.com/JieLiu95/EraserDiT)：模型与视频擦除算法。
- [SGLang](https://github.com/sgl-project/sglang)：推理框架与多模态服务设计参考。
- [Hugging Face 模型](https://huggingface.co/jieeliu/EraserDiT) · [论文](https://arxiv.org/abs/2506.12853) · [演示](https://jieliu95.github.io/EraserDiT_demo/)

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
