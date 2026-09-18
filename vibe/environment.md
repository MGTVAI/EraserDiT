# 当前环境与输入资源

以 EraserDiT 已有资源为迁移起点。以下为 2026-09-18 本机实测结果，区分"已实测通过"与"尚未验证"。路径存在及包可导入不代表完整推理已经通过。

## 仓库与输入视频

仓库根目录：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT`。

下表路径均相对仓库根目录，四个文件已存在；宽高、帧数和帧率来自 `ffprobe` 元数据。

| 文件 | 宽 × 高 | 帧数 | 帧率 |
| --- | --- | --- | --- |
| `data/10268234.mp4` | 1080 × 1920 | 120 | 2997/100 |
| `data/10268234_mask.mp4` | 1080 × 1920 | 120 | 2997/100 |
| `data/113000356.mp4` | 1920 × 1080 | 145 | 24000/1001 |
| `data/113000356_mask.mp4` | 1920 × 1080 | 145 | 1199/50 |

第二组视频与 mask 帧数相同，但帧率元数据略有差异。迁移时保留原版帧对应与输出帧率处理方式，不擅自重采样。

第一组提示词为 `There is a bridge over the lake.`；第二组提示词待定，生成基线前需固定并记录。两组输入均未超过原入口启用高分辨率裁剪的面积阈值（1088 × 1920），当前可使用默认 `bbox_path=None`。

## 模型路径

- 原代码默认模型标识：`jieeliu/EraserDiT`。
- 加载位置：`utils/inference_utils.py` 中 `init(..., pre_dir=...)`。
- 本地快照（后续显式模型路径）：

```text
/root/.cache/huggingface/hub/models--jieeliu--EraserDiT/snapshots/904fb412da76235085dbbccaefdbde4979fa3d29
```

快照实测完整，共约 28 GB，无断链，各组件齐备：

| 组件 | 文件 | 大小 |
| --- | --- | --- |
| `transformer/` | `diffusion_pytorch_model.safetensors` + `config.json` | 7.69 GB |
| `text_encoder/` | 四个分片 + 索引 + `config.json` | 19.05 GB |
| `vae/` | `diffusion_pytorch_model.safetensors` + `config.json` | 2.49 GB |
| `tokenizer/` | `spiece.model`、`tokenizer_config.json` 等 | — |
| `scheduler/` | `scheduler_config.json` | — |

该快照没有根目录 `model_index.json`。原版按上述五个子目录分别加载；新加载器需通过模型配置显式声明组件，不应要求现有快照必须包含该文件。

模型配置要点（供后续组件设计与维度检查）：

- transformer：`LTXVideoTransformer3DModel`，28 层、32 头、每头 64 维，`cross_attention_dim=2048`，`caption_channels=4096`，`in_channels=257`，`out_channels=128`，`qk_norm=rms_norm_across_heads`。
- vae：`AutoencoderKLLTXVideo`，空间压缩比 32、时间压缩比 8，latent 128 通道，`encoder_causal=true`，`patch_size=4`。
- scheduler：`FlowMatchEulerDiscreteScheduler`，`shift=1.0`，指数时间平移。
- text_encoder：`google/t5-v1_1-xxl`，配置声明 `torch_dtype=float32`，对应上表 19 GB。

README 使用 `HF_ENDPOINT=https://hf-mirror.com`；本次检查时该变量未设置。迁移验收优先使用固定本地快照，避免远端版本变化。

## Conda 环境

`EraserDiT` 为验收基准环境，也是本机唯一与仓库 `requirements.txt` 锁定版本一致的解释器。

```text
环境名：EraserDiT
环境路径：/mnt/shanhai-ai/envs/conda/envs/EraserDiT
Python：/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python
```

进入环境：

```bash
conda activate /mnt/shanhai-ai/envs/conda/envs/EraserDiT
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT
```

检查时交互 shell 实际位于 `/opt/conda` 的 `base` 环境，不能将默认 `python` 当作上述推理环境。

实测包版本：

| 包 | 版本 |
| --- | --- |
| Python | 3.10.21 |
| torch | 2.6.0+cu126 |
| torchvision | 0.21.0+cu126 |
| triton | 3.2.0 |
| flash_attn | 2.8.3 |
| sageattention | 1.0.6 |
| diffusers | 0.40.0 |
| transformers | 5.3.0 |
| accelerate | 1.13.0 |
| bitsandbytes | 0.50.2 |
| numpy | 2.2.6 |
| opencv-python | 4.13.0.92 |
| fastapi / uvicorn / pydantic | 0.140.0 / 0.51.0 / 2.13.4 |

以下为实测通过的检查项：

- `import torch` 6.3 秒返回；`torch.cuda.is_available()` 为真，可见 8 个设备。
- CUDA 版本 12.6，cuDNN 9.5.1，设备算力 `(8, 0)`。
- bf16 矩阵运算正常。
- 原版 `inference.py`、`utils/`、`models/`、`pipelines/` 的全部 import 均可解析。
- 注意力与编译路径的实际前向均跑通：SDPA、`flash_attn_func`、`sageattn`、`torch.compile`。

> 早期记录过"尝试导入 `torch` 长时间未返回"。该现象在本次检查中未复现——环境于 09-17 17:35 重建过，问题应已随重建消失。

本机另存在 `eraze_dit` 环境（Python 3.14.7 + torch 2.14.0+cu126），与 `requirements.txt` 锁定的版本集不是一套，不得用于本项目的开发或验收。README 已改为指向 `EraserDiT`。

## 环境能力边界

A100 为 sm_80，以下路径在当前环境不可用，不得作为已验证能力交付：

- FP8 全系：`torch._scaled_mm` 实测报错，仅支持 sm_90 / sm_89 / ROCm MI300+。对应 MGErase 的 `sage_fp8`、`fp8_w8a8`、`fp8_w8a8_triton_selective` 及相关 `--fp8-*` 参数。
- INT8 路径（SageAttention、`int8_w8a8_viditq`）的可用性以 sm_80 为准，需在接入时逐项实测，不能按导入成功判定。

## 容器与构建

- `docker/base.dockerfile` 基于 `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`，尚未构建过，本机无相关镜像。
- 构建所需 `nvcc` 可用：`/usr/local/cuda-12.6`。
- 容器定位为可复现交付物；加速与基线数字以 conda 环境为准。

## 硬件与运行约定

- 当前机器可见 8 张 NVIDIA A100-SXM4-80GB。
- 实测时 8 张卡均有其他任务占用，空闲显存上限为 62673 MiB（GPU 2 与 GPU 3），GPU 0 仅剩 717 MiB。原版 2K 推理要求 >60 GB，余量不足 1.5 GB。
- 正式性能测量需在协调出的独占（或至少同卡无大任务）窗口内进行；开发期间在共享卡上的相对对比需标注当时的占用状态。
- 系统已提供 `ffmpeg` 和 `ffprobe`。
- 原版使用 CUDA、BF16、50 个配置采样步、`strength=0.8`，分段长度配置为 121；seed 默认未固定。
- 原版结果默认写入根目录 `results/` 下带时间戳的子目录。现有 `results/` 内容为 09-15 与 09-17 的试跑产物，不作为基线。
- 磁盘：`/` 可用约 518 GB，`/mnt/shanhai-ai` 可用约 17 TB。

## 已有产出与尚未验证

`results/` 下有三份完整产出（`2026-09-15T20-57-32`、`2026-09-17T20-09-53`、`2026-09-17T20-27-31`），均为第一组素材，规格与输入一致（1080 × 1920、120 帧、2997/100 fps），`ffmpeg` 全解码无错误。同期另外九个目录为空，属失败或中断的试跑。

因此原版端到端推理在本机第一组素材上已经成立。这些产出使用默认 `seed=None`，互相之间是不同随机样本，**不构成可复现基线**。

尚未验证：

- 第二组素材（1920 × 1080）从未运行过。
- 完整模型加载与单次前向未被单独检查。
- 两组素材的可复现基线与加速对照。

基线与性能报告需记录实际 GPU、环境版本、模型快照、提示词、种子及所有采样参数。
