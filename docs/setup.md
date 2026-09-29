# 安装与部署

新机器使用 [README 快速开始](../README.md#快速开始) 完成安装、下载和首次推理。
本文说明统一依赖安装、环境检查与容器运行。所有命令从仓库根目录执行。

## 环境要求

- Linux x86_64、NVIDIA GPU，`nvidia-smi` 可正常显示设备。
- Python 3.10，由 uv 自动获取；PyTorch 2.6.0 / CUDA 12.6，由依赖文件安装。
- FFmpeg 与 FFprobe 在 `PATH` 中，FFmpeg 支持 `libx264`。
- 模型约 30 GB；磁盘还需容纳 Python 依赖、解码缓存和输出，CPU 内存及显存随素材尺寸变化。

`requirements.txt` 统一包含推理、HTTP 服务、FlashAttention / SageAttention 和测试依赖。
SDPA 本身使用 PyTorch 内置实现，但完整安装包含 FlashAttention，需准备 CUDA 12.6 Toolkit、
C++ 编译器及兼容驱动。`nvcc --version` 检查的是 Toolkit；`nvidia-smi` 中的 CUDA 版本不能替代它。
Toolkit 无法自动定位时，将 `CUDA_HOME` 设为实际安装目录并把其 `bin` 加入 `PATH`。

## 安装统一依赖

先按 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/)安装 uv，
然后在仓库根目录执行：

```bash
sudo apt-get install -y git curl build-essential ffmpeg libgl1 libglib2.0-0
nvcc --version
uv venv --python 3.10
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  torch==2.6.0+cu126 pip setuptools wheel packaging ninja psutil
MAX_JOBS=4 uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --no-build-isolation-package flash-attn -r requirements.txt
uv pip check --python .venv/bin/python
```

第一步预装 Torch 和构建工具，第二步从唯一依赖清单安装全部包。
FlashAttention 的构建脚本需要导入 Torch，仅将它关闭构建隔离；其余包保留 uv 默认行为。
`MAX_JOBS=4` 限制扩展编译并发，可按 CPU 内存调整。
参见 [FlashAttention 2.8.3 安装要求](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3)
和 [uv 构建隔离说明](https://docs.astral.sh/uv/pip/compatibility/#pep-517-build-isolation)。
安装后仍默认使用 SDPA，安装扩展不会自动切换注意力后端。

## 环境检查

```bash
uv run --no-project python -c 'import flash_attn, sageattention, httpx; print("attention/test imports OK")'
```

检查设备和入口：

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

## 复用环境

依赖清单包含固定版本和版本范围，并非完整锁文件。可导出实际安装版本：

```bash
uv pip freeze --python .venv/bin/python > environment.freeze.txt
```

另一台同平台机器先按上面的流程准备 Toolkit、创建环境并预装 Torch 与构建工具，
再用快照安装全部依赖：

```bash
MAX_JOBS=4 uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match \
  --no-build-isolation-package flash-attn -r environment.freeze.txt
uv pip check --python .venv/bin/python
```

历史实验若使用 SageAttention 2 或本机编译的扩展，以对应记录为准；
统一清单保留 `sageattention==1.0.6`，不能据此直接复现其他扩展版本的性能。

## 容器

主机需要 NVIDIA 驱动和支持 GPU 的容器运行时。镜像通过 uv 安装依赖并提供 FFmpeg，
使用 CUDA devel 基础镜像构建统一清单中的注意力扩展，推理默认使用 SDPA；模型和素材通过挂载提供。

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
| 构建 FlashAttention 时缺少 `torch` / `nvcc` | 按安装顺序预装 Torch，检查 CUDA Toolkit 与 `CUDA_HOME`；使用上述关闭单包构建隔离的命令 |
| 扩展编译被系统杀死 | 降低 `MAX_JOBS`，检查 CPU 内存与磁盘空间 |
| 显存不足 | 使用较小素材验证，或启用[权重卸载](performance.md#offload)；卸载不能消除激活显存 |
| 任务开始后 CPU 忙、GPU 空闲，停在掩码读取 | 入口默认设置 `NUMPY_MADVISE_HUGEPAGE=0`，避免共享主机上大数组分配触发透明大页整理停顿；直接调用 Python API 时，在导入 NumPy/Torch 前设置此环境变量。显式设置为 `1` 可恢复 NumPy 默认行为 |
| 本机 HTTP 请求受代理影响 | 设置 `NO_PROXY=127.0.0.1,localhost` |

服务调用见[服务 API](service_api.md)，批量和多卡运行见[CLI](cli.md)。
