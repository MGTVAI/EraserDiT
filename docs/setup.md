# 安装与部署

新机器使用 [README 快速开始](../README.md#快速开始) 完成安装、下载和首次推理。
本文补充环境要求、可选配置与容器运行。所有命令从仓库根目录执行。

## 环境要求

- Linux x86_64、NVIDIA GPU，`nvidia-smi` 可正常显示设备。
- Python 3.10，由 uv 自动获取；PyTorch 2.6.0 / CUDA 12.6，由依赖文件安装。
- FFmpeg 与 FFprobe 在 `PATH` 中，FFmpeg 支持 `libx264`。
- 模型约 30 GB；磁盘还需容纳 Python 依赖、解码缓存和输出，CPU 内存及显存随素材尺寸变化。

默认 SDPA 使用 PyTorch 自带实现，无需安装 CUDA Toolkit 或注意力扩展。
系统需安装兼容 CUDA 12.6 的 NVIDIA 驱动；可用下列命令检查 Python 环境：

```bash
uv run --no-project python -c 'import torch; print(torch.__version__, torch.version.cuda); print("CUDA available:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())'
ffmpeg -version
ffprobe -version
uv run --no-project python -m entrypoints.cli.erase_eraserdit --help
uv run --no-project python -m entrypoints.server.serve --help
```

uv 的安装与虚拟环境说明见[官方安装文档](https://docs.astral.sh/uv/getting-started/installation/)和[环境文档](https://docs.astral.sh/uv/pip/environments/)。
项目使用 `requirements.txt` 管理依赖，`uv run --no-project` 从 `.venv` 执行命令。

## 模型与素材

从 [Hugging Face 模型仓库](https://huggingface.co/jieeliu/EraserDiT) 拉取固定版本的完整快照：

```bash
HF_HUB_OFFLINE=0 uv run --no-project hf download jieeliu/EraserDiT \
  --revision 904fb412da76235085dbbccaefdbde4979fa3d29 \
  --local-dir data/model
```

下载中断后重新执行即可利用已下载文件；`--local-dir` 保留目录结构。
访问需要身份认证的仓库时，先运行 `uv run --no-project hf auth login`。
命令说明见 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)。

本地目录应包含 `model_index.json`、`tokenizer/`、`text_encoder/`、`vae/`、`transformer/`、
`scheduler/`，以及分片索引引用的全部权重。模型完整后可设置 `HF_HUB_OFFLINE=1` 离线运行。
`data/model/` 不随 Git 或 Docker 镜像分发。

已有模型可直接复制到新机器的 `data/model/`，复制时需要包含软链接实际指向的文件。
更换机器时重新创建 `.venv` 并安装依赖；模型和输入素材可以复用，编译缓存应在目标 GPU 上生成。

示例视频是 `data/113000356.mp4`，掩码是 `data/113000356_mask.mp4`。
自有掩码需与视频逐帧对应，尺寸和帧率匹配，白色表示待擦除区域、黑色表示保留区域。
prompt 描述擦除后的背景，无需额外下载 caption 模型。

## 可选依赖

默认安装只包含 SDPA 推理和 HTTP 服务所需依赖。测试依赖单独安装：

```bash
uv pip install --python .venv/bin/python -r requirements-test.txt
```

FlashAttention / SageAttention 按需安装，需要与 Torch 匹配的 CUDA 开发工具链和 C++ 编译器，
`nvcc` 应在 `PATH` 中；无法自动定位时将 `CUDA_HOME` 设为本机 Toolkit 目录：

```bash
uv pip install --python .venv/bin/python pip setuptools wheel packaging ninja
uv pip install --python .venv/bin/python --no-build-isolation -r requirements-attention.txt
```

依赖文件包含版本范围。需要复用同平台的确切环境时，可以导出实际安装版本：

```bash
uv pip freeze --python .venv/bin/python > environment.freeze.txt
```

在另一台同平台机器的新环境中安装：

```bash
uv venv --python 3.10
uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match -r environment.freeze.txt
```

含可选 CUDA 扩展的环境应先安装基础依赖，再按上面的步骤构建扩展。

## 容器

主机需要 NVIDIA 驱动和支持 GPU 的容器运行时。镜像通过 uv 安装依赖并提供 FFmpeg，
默认使用 SDPA；模型和素材通过挂载提供。

```bash
docker build -f docker/base.dockerfile -t eraserdit:cu126 .
mkdir -p outputs
docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -v "$PWD/data/model:/workspace/EraserDiT/data/model:ro" \
  -v "$PWD/data:/inputs:ro" -v "$PWD/outputs:/outputs" \
  eraserdit:cu126 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model --video-input /inputs/113000356.mp4 \
  --mask-input /inputs/113000356_mask.mp4 --output-path /outputs/result.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." --attention-backend sdpa
```

挂载源文件必须存在，输出目录需可写。构建时会检查 CLI、服务入口和核心包能否导入。
部署后按[验证步骤](validation.md)检查实际 GPU 推理和输出。

## 常见问题

| 问题 | 处理 |
| --- | --- |
| `uv: command not found` | 执行 `export PATH="$HOME/.local/bin:$PATH"`，或重新打开终端 |
| `CUDA available: False` | 检查驱动、GPU 可见性及容器 GPU 支持 |
| 找不到 `ffmpeg` / `ffprobe` | 安装系统 FFmpeg 并确认命令在 `PATH` 中 |
| 模型文件缺失 | 关闭离线模式，重新执行完整模型下载命令 |
| 显存不足 | 使用较小素材验证，或启用[权重卸载](performance.md#offload)；卸载不能消除激活显存 |
| 本机 HTTP 请求受代理影响 | 设置 `NO_PROXY=127.0.0.1,localhost` |

服务调用见[服务 API](service_api.md)，批量和多卡运行见[CLI](cli.md)。
