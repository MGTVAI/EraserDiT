# 当前环境与输入资源

以 EraserDiT 已有资源为迁移起点。以下为 2026-09-17 本机检查结果，路径存在及包版本检查不代表完整推理已经通过。

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

原 README 为第一组提供的提示词为 `There is a bridge over the lake.`；第二组提示词尚未单独约定，生成基线前需固定并记录。两组输入均未超过原入口启用高分辨率裁剪的面积阈值，当前可使用默认 `bbox_path=None`。

## 模型路径

- 原代码默认模型标识：`jieeliu/EraserDiT`。
- 加载位置：`utils/inference_utils.py` 中 `init(..., pre_dir=...)`。
- 当前本地缓存：`/root/.cache/huggingface/hub/models--jieeliu--EraserDiT`。
- 已发现的本地快照，可作为后续显式模型路径：

```text
/root/.cache/huggingface/hub/models--jieeliu--EraserDiT/snapshots/904fb412da76235085dbbccaefdbde4979fa3d29
```

快照中已检查到有效文件：`scheduler/` 配置、`tokenizer/` 文件、`text_encoder/` 配置及四个权重分片和索引、`transformer/` 配置及权重、`vae/` 配置及权重。尚未执行完整模型加载或权重校验。

该快照没有根目录 `model_index.json`。原版按上述五个子目录分别加载；新加载器需通过模型配置显式声明组件，不应要求现有快照必须包含该文件。

README 使用 `HF_ENDPOINT=https://hf-mirror.com`；本次检查时该变量未设置。迁移验收优先使用固定本地快照，避免远端版本变化。

## Conda 环境

优先沿用 README 指定且本机已存在的环境：

```text
环境名：eraze_dit
环境路径：/mnt/shanhai-ai/envs/conda/envs/eraze_dit
Python：/mnt/shanhai-ai/envs/conda/envs/eraze_dit/bin/python
```

进入环境：

```bash
conda activate /mnt/shanhai-ai/envs/conda/envs/eraze_dit
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT
```

检查时交互 shell 实际位于 `/opt/conda` 的 `base` 环境，不能将默认 `python` 当作上述推理环境。

以下为 `eraze_dit` 解释器及已安装包元数据的实测版本：


这些版本与仓库 `requirements.txt`、MGErase 的验证环境并不完全一致。开发时先验证现有环境的原版执行能力，再确定需要补充或调整的依赖，不直接套用 MGErase 的版本清单。

本次额外尝试导入 `torch`，长时间未返回后已中止，未完成 CUDA 可用性检查；原因尚未定位，不能据包元数据认定运行环境已通过验证。

## 硬件与运行约定

- 当前机器可见 8 张 NVIDIA A100-SXM4-80GB；GPU 存在其他任务占用，可用显存需在运行前重新检查。
- 系统已提供 `ffmpeg` 和 `ffprobe`，本次已使用 `ffprobe` 读取输入元数据。
- 原版使用 CUDA、BF16、50 个配置采样步、`strength=0.8`，分段长度配置为 121；seed 默认未固定。
- 原版结果默认写入根目录 `results/` 下带时间戳的子目录。
- 本文记录已有环境，不表示根目录 `inference_cli.sh`、`inference_server.sh` 已实现；新入口与常驻验收按开发文档推进。

后续基线与性能报告需记录实际 GPU、环境版本、模型快照、提示词、种子及所有采样参数。完整推理、扩展后端兼容性和视频对齐仍待开发阶段验证。
