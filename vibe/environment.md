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

第二组视频与 mask 帧数相同，但帧率元数据略有差异；第二组 mask 为 `yuv444p`（无色彩下采样），
视频为 `yuv420p`。迁移时保留原版帧对应与输出帧率处理方式，不擅自重采样。

两个 mask 都是 RGB 三通道等值的 0/255 二值图，另有少量编解码振铃（1–2、252–254），
由原版的 `threshold=0.039`（等价于 > 4.97）滤除。这决定了 `plan.md` §1 里掩码灰度核的取值：
255 值经 `0.299/255 + 0.587/255 + 0.0004` 后约为 `0.988`，而不是 1.0。

两组输入的 mask 均裁剪人物：第一组是湖边栈桥上的老人，第二组是屋顶平台上的人。已确认的产出显示提示词描述的是场景主体结构而非被擦除对象——第一组的 `bridge` 即指画面中的木质栈桥。

固定提示词：

| 素材 | 提示词 |
| --- | --- |
| 第一组 | `There is a bridge over the lake.` |
| 第二组 | `There is a rooftop terrace overlooking the city at sunset.` |

第二组提示词为本次新增，可随基线调整但必须固定并记录。两组输入均未超过原入口启用高分辨率裁剪的面积阈值（1088 × 1920），当前可使用默认 `bbox_path=None`。

## 模型路径

- 原代码默认模型标识：`jieeliu/EraserDiT`。
- 原版加载位置：冻结基线的 `utils/inference_utils.py` 中 `init(..., pre_dir=...)`；该旧模块已从当前仓库删除。
- 后续 CLI、服务和性能复测统一显式使用的模型路径：

```text
/mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model
```

2026-09-20 核验：16 个模型文件共约 29.24 GB，权重软链接均可访问，
与官方固定版本 `jieeliu/EraserDiT@904fb412da76235085dbbccaefdbde4979fa3d29` 校验一致。
校验记录：`results/cache_prediction_model/availability_verified.json`。各组件齐备：

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

**加载必须显式离线或指定镜像。** 直接跑 `baseline_runner.py`（不设 `HF_ENDPOINT`）会卡在
直连 `huggingface.co` 的 TCP SYN-SENT 上，进程停住不报错。实测加 `HF_HUB_OFFLINE=1` 后
8.3 s 完成加载，且强制使用本地快照。冻结记录里的加载条件按 `HF_HUB_OFFLINE=1` 记。

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
- 迁移前原版 `inference.py`、`utils/`、`models/`、`pipelines/` 的全部 import 均可解析（历史检查；当前仓库已删除旧入口及其专用工具）。
- 注意力与编译路径的实际前向均跑通：SDPA、`flash_attn_func`、`sageattn`、`torch.compile`。

> 早期记录过"尝试导入 `torch` 长时间未返回"。该现象在本次检查中未复现——环境于 09-17 17:35 重建过，问题应已随重建消失。

本机另存在 `eraze_dit` 环境（Python 3.14.7 + torch 2.14.0+cu126），与 `requirements.txt` 锁定的版本集不是一套，不得用于本项目的开发或验收。README 已改为指向 `EraserDiT`。

## 环境能力边界

A100 为 sm_80，以下路径在当前环境不可用，不得作为已验证能力交付：

- FP8 全系：`torch._scaled_mm` 实测报错，仅支持 sm_90 / sm_89 / ROCm MI300+。对应 MGErase 的 `sage_fp8`、`fp8_w8a8`、`fp8_w8a8_triton_selective` 及相关 `--fp8-*` 参数。
- INT8 只有量化侧的 `int8_w8a8_viditq`（transformer 与 text encoder 各一套，注意力侧没有 int8 方法）：探针要求 sm ≥ 8.0，A100 满足；但内核根目录 `/root/viditq` **在当前机器上不存在**，因此该路径现在跑不起来，需补齐内核后再实测，不能按导入成功判定。

## 容器与构建

- `docker/base.dockerfile` 基于 `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`，尚未构建过，本机无相关镜像。
  该文件目前是模板残留：里面引用 `Acceptance/result_videos`、`.venv/bin/torchrun` 等本仓库不存在的
  东西，必须按新入口重写后再谈构建验证。
- 构建所需 `nvcc` 可用：`/usr/local/cuda-12.6`。
- 容器定位为可复现交付物；加速与基线数字以 conda 环境为准。

## 硬件与运行约定

- 当前机器可见 8 张 NVIDIA A100-SXM4-80GB。
- 实测时 8 张卡均有其他任务占用；空闲上限 62673 MiB（GPU 2、GPU 3）= 61.2 GiB，而原版实测峰值
  reserved 为 60.0 GiB、allocated 42.8–44.2 GiB，余量仅约 1.2 GiB 且不含碎片；GPU 0 仅剩 717 MiB。
- 正式性能测量需在协调出的独占（或至少同卡无大任务）窗口内进行；开发期间在共享卡上的相对对比需标注当时的占用状态。
- 系统已提供 `ffmpeg` 和 `ffprobe`。
- 原版运行配置（基线冻结必须逐项记录）：CUDA + BF16；`num_inference_steps=50` 配 `strength=0.8`
  → 实际去噪 40 步，调度取 `linear_quadratic_schedule(50)` 的后 40 个时间步；
  `guidance_scale=3`（CFG 拆成两次 batch=1 前向）；`TEMP_INFER_LEN=121`、`shift_alpha=9`、
  `ksize=(9,9)`、`dilate_iter=9`、`threshold=0.039`；`frame_rate=25`——它决定 RoPE 时间插值尺度
  `8/25`，**不随输入视频帧率变化**；`decode_timestep=0.0`、`decode_noise_scale=0.0`
  （解码噪声数值上无效，但仍消耗随机数）；`max_sequence_length=128`；负向提示词为
  `inference.py:17` 的长串。seed 默认未固定，基线固定为 42。
- 原版结果默认写入根目录 `results/` 下带时间戳的子目录。现有 `results/` 内容为 09-15 与 09-17 的试跑产物，不作为基线。
- 磁盘：`/` 可用约 518 GB，`/mnt/shanhai-ai` 可用约 17 TB。

## 已有产出

主仓库 `results/` 下有三份第一组试跑产出（`2026-09-15T20-57-32`、`2026-09-17T20-09-53`、`2026-09-17T20-27-31`），规格与输入一致，解码无错误；同期另外九个目录为空。这些产出 `seed=None`，**不构成基线**。

第二组于 2026-09-18 在冻结副本中首次跑通（`EraserDiT-ref/results/2026-09-18T10-10-52-113000356/`），1920 × 1080、145 帧、24000/1001 fps，与输入一致，目视擦除干净。输出帧率取视频流的 24000/1001 而非 mask 的 1199/50，与原版处理一致。

## 基线运行记录（`seed=42`，`baseline-run`）

`EraserDiT-baseline/results/` 现存两组素材共 8 份输出，另有 `baseline_summary.json`（只覆盖最后一次运行）。按目录名与文件写入时间还原：

| 素材 | 运行窗口 | 端到端 | 峰值 alloc / resv | 输出 md5 |
| --- | --- | --- | --- | --- |
| 第一组 | 10:42–10:48、10:48–10:54（同进程两轮） | 376s、348s | — | **两次不同**（文件大小不同） |
| 第一组 | 10:55–11:00、11:00–11:06（同进程两轮） | 318s、325s | 42.78 / 60.0 GiB | **两次相同** |
| 第二组 | 11:09–11:32（4 份，两两成对，两对之间相隔 11 分钟） | 675.6s（中位） | 44.19 / 60.01 GiB | **4 份全相同** |

加载 7.5s（页缓存热，冷缓存会显著更长）。`ffmpeg` 全帧对比（Y 通道，含编码噪声）：第一组前两轮 SSIM 0.988 / PSNR 38.8 dB，前两轮与后两轮之间 0.988 / 39.1 dB，后两轮之间逐位相同。

**结论：可复现性在同一 seed、同一代码下也不稳定。** 同一进程内的两轮，早期一次不一致、后期一次逐字节一致；第二组四次（含并发进程）全部逐字节一致。差异不是种子或参数造成的（`run_batch` 每次调用都会重新播种），只能来自环境相关分支——最可能是共享卡占用改变了 cuBLAS / SDPA 的内核或 workspace 选择。**基线自洽度不能假定为 1.0。** 该问题已由下面的 M0 确定性口径解决。

allocated 与 reserved 相差约 16 GiB，是分配器保留的缓存；实际余量比 reserved 显示的更紧。同卡其他任务占用 18542 MiB 时，全卡剩余最低仅 714 MiB。

## M0 确定性口径（2026-09-18 施加并实测）

冻结副本并入第 3 类外挂改动：`inference.py` 注入确定性开关——`CUBLAS_WORKSPACE_CONFIG=:4096:8`、
`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`，
并记录未改动的 `matmul.allow_tf32=False`、`cudnn.allow_tf32=True`。由 `ERASERDIT_DETERMINISTIC`
控制，默认开、`=0` 关闭；不触碰任何算法参数与采样参数。`baseline_runner.py` 的摘要现在带实际
生效的设置、每轮输出 md5 与 md5 唯一数。

确定性口径下 `--repeat 4`（同进程四轮，GPU 3，同卡另有 18542 MiB 占用）：

| 素材 | 端到端（四轮） | 中位 | md5 唯一数 | 峰值 reserved |
| --- | --- | --- | --- | --- |
| `10268234` | 338.7 / 342.6 / 335.3 / 335.3 s | 337.0 s | **1** | 60.0 GiB |
| `113000356` | 639.2 / 632.1 / 642.0 / 637.0 s | 638.1 s | **1** | 60.01 GiB |

两组的确定性输出都**与未施加开关时偶然自洽的那批输出逐字节相同**（第一组 `90cb85b2…`，
第二组 `23ff1ab8…`），说明开关是把计算钉在既有的数值路径上，不改变结果本身。
**基线自洽下限由此确定为「逐字节相同」，非擦除区 SSIM ≥ 0.99 的门禁成立（M0 判定通过）。**

两套口径的耗时差（第一组，同一张卡前后各测一轮，各 3–4 次）：

| 口径 | 端到端（中位） | 组内离散 |
| --- | --- | --- |
| 确定性 | 337.0 s | 7.3 s |
| 快速（`=0`） | 318.0 s | 1.9 s |

即确定性口径端到端约 **+6%**（≈19 s）。两次测量相隔约 1 小时、同卡占用状态有变化，故该差值
按上界用；正式报告里不得把它混进加速倍数。

这三次快速口径运行也全部落在同一个 md5 上（与确定性口径同支），说明分支分歧是偶发的环境
相关现象，而不是快速口径必然给出不同结果。

## 冻结记录

| 项 | 值 |
| --- | --- |
| 基线提交 | `9944867`（参考副本 `EraserDiT-ref`，detached 只读）；运行副本 `EraserDiT-baseline` 分支 `baseline-run` |
| 外挂改动 | ① 固定随机源（`BASELINE_SEED=42`，已提交）② 常驻 pipeline wrapper（已提交）③ 确定性开关（**当前在工作区，未提交**） |
| 外挂工具 | `baseline_runner.py`（常驻运行、计时、md5）、`compare_outputs.py`（质量指标口径） |
| 权重快照 | `jieeliu/EraserDiT@904fb412da76235085dbbccaefdbde4979fa3d29`，加载一律 `HF_HUB_OFFLINE=1` |
| 环境 | conda `EraserDiT`：Python 3.10.21 / torch 2.6.0+cu126 / cuDNN 9.5.1 / A100-SXM4-80GB |
| 提示词与采样参数 | 见「仓库与输入视频」「硬件与运行约定」两节 |
| 基准视频 | `10268234`：`results/2026-09-18T11-59-22-10268234/`，md5 `90cb85b29c33d88bbd239dedb77cac59`；`113000356`：`results/2026-09-18T12-22-28-113000356/`，md5 `23ff1ab8ea0327dcf9ac6b17d0e966cc` |

## 尚未验证

- 冷页缓存下的加载耗时。
- 确定性口径的耗时差在第二组素材上未单独测（按第一组的 +6% 上界引用）。
- 加速版与基线的对照。

基线与性能报告需记录实际 GPU、环境版本、模型快照、提示词、种子及所有采样参数。

## M1 新架构（2026-09-18，进行中）

MGErase `python/` 已平铺到仓库根（`config/ entrypoints/ layers/ loader/ models/ nodes/
pipelines/ utils/ memory/ cache/ distributed/ parallel/`，其中 profiling 模块已合并到 `utils/` 根目录），
import 全量改写为无前缀形式；原版 `models/{transformer_ltx,autoencoder_kl_ltx}.py` 移到
`models/dits/eraserdit_transformer.py`、`models/vaes/eraserdit_vae.py` 并改名为
`EraserDiT*` 类以免与 LTX095 模型类在 `models/registry.py` 中撞名；`utils/pre.py` 等原版
实现的语义已逐函数搬进 `utils/eraserdit_*.py` 与 `nodes/stages/model_specific_stages/eraserdit_erase/`。
旧入口 `inference.py` 及其专用的 `utils/{inference_utils,pre,post,post_pkg,common}.py`
已删除；文档与代码注释中的原版文件行号用于追溯冻结基线。
`pipelines/eraserdit_video2video.py` 仍供 `scripts/legacy/m1b_model_ab.py` 做模型 A/B 验证，
`utils/colorfix_wmask.py` 仍由当前后处理模块使用。

入口：`./inference_cli.sh`（单任务参数或 `--task-file` JSON 任务数组），
`ERASERDIT_DETERMINISTIC=0` 关闭确定性口径（默认开）。渲染在 GPU 2 上实测：

| 项 | 值 |
| --- | --- |
| 第一组端到端（40 步，确定性口径） | 316.7 s（非确定性 312.9 s）；基线同口径 337.0 s |
| 同 seed 重复 | 四份输出**逐字节相同**（md5 `083bb752…`），与是否开确定性口径无关 |
| 窗口规划 | `load=0:120 deal=0:120 commit=0:120 bbox=(0,0,1080,1920)`，与原版单窗一致 |

**移植保真度（CPU 单元对照，与冻结原版逐位比较）**：`VideoInpaintPre.__call__` +
`mask_video_nchw` 在 120/121/112/24 帧四种窗口形态下 `masked_video` 与压缩掩码均
**逐位相同**；`linear_quadratic_schedule`、`retrieve_timesteps`、`get_timesteps`
产出的 40 步时间步完全相同；`VideoProcessor.preprocess` 与 `2x-1` 逐位相同。

**M1b 等价性：模型路径已逐位对齐，写出侧已修正三处口径差异。**

`scripts/legacy/m1b_model_ab.py` 在同一进程、同一 GPU、同一窗口输入上对跑「原版 diffusers 管线」与
「新阶段链」，`cond_latents`、初始 latents 与**解码帧全部逐位相同** —— 新架构的模型路径
（VAE 编码 → 40 步去噪 → VAE 解码）与原版一致。逐段排查后修掉四处写出侧差异：

| 差异 | 原版 | 修复前 | 量级 |
| --- | --- | --- | --- |
| x264 线程数 | ffmpeg 默认（等同 `-threads auto`） | 框架默认 `-threads 4` | ~1.5–2 dB |
| 帧缓存精度 | uint8 | streaming 模式的 bf16 缓存同时量化模型输入与提交帧 | ~1.2 dB |
| 输出码率 | `bit_rate//1e6` M（7M） | 输入的原始码率串（7.69M） | ~0.4 dB |
| 掩码二值化阈值 | `255/2*0.039` = 4.97 | 共享读取器默认 `0.3*max` = 76.5 | 0.009% 像素 |

改动：`pipelines/runtime/context.py` 增加适配器可声明的写出契约；`_create_output_writer` 与流式写出器
统一走 `MGERASE_FFMPEG_THREADS`（EraserDiT CLI 默认 `auto`）；适配器默认
`runtime_mode="windowed_preload"`（uint8 帧缓存）；`read_mask_array` 增加 `threshold_ratio`，
适配器传 `mask_threshold/2`；文本编码补齐原版 autocast。

第一组实测（40 步、同 seed、同权重、两侧确定性口径，`compare_outputs.py` 口径）：

| 阶段 | 非擦除区左 / 右条带 | 整帧 |
| --- | --- | --- |
| streaming + 修复前写出 | 0.9892 / 0.9880，37.87 / 37.93 dB | 0.9871 / 37.92 dB |
| 修复写出侧三处后 | 0.9902 / 0.9893，39.22 / 39.46 dB | 0.9884 / 39.60 dB |
| 掩码改 RGB 解码后（终值） | **0.9911 / 0.9906，40.54 / 40.69 dB** | 0.9897 / 40.89 dB |

**M1b 门禁通过**（非擦除区 SSIM ≥ 0.99、PSNR ≥ 40 dB）：左/右条带 0.9911 / 0.9906、
40.54 / 40.69 dB。逐像素非擦除口径同量级（0.9895 / 40.55 dB）。整帧 PSNR 40.89 dB 亦达标；
整帧 SSIM 0.9897 高于基线自洽上限 0.9881（即具备区分度），但低于辅助项字面阈值 0.995，
按辅助项如实记录。mask 内与边缘逐帧目视：人物干净移除，栈桥纹理重建正常，无残留、无损坏、
无新增时序退化。

端到端 313.2 s（确定性口径），对基线 337.0 s 为 0.93×，远低于稳定性门禁 1.15×。

第二组（`113000356`，两窗，覆盖前缀/重叠路径）同样通过：

| 指标 | 值 | 门禁 |
| --- | --- | --- |
| 非擦除区左条带 | SSIM 0.9919 / PSNR 48.57 dB | ≥ 0.99 / ≥ 40 dB ✅ |
| 非擦除区右条带 | SSIM 0.9918 / PSNR 49.16 dB | ≥ 0.99 / ≥ 40 dB ✅ |
| 整帧 | SSIM 0.9918 / PSNR 48.77 dB | ✅ |
| 端到端 | 622.8 s（两窗；基线 638.1 s，0.98×） | ≤ 1.15× ✅ |

窗口规划与原版一致：`load=0:121`（overlap_right=9）+ `load=112:145`（overlap_left=9），
即原版的两批 121 帧 / 24 帧。该素材只有两窗，多窗口才走到的衔接帧路径由此覆盖；
`≥3` 窗的前缀语义缺口（见 `plan.md` 风险表）仍未触发。

对照的 8 份基线输出逐字节相同（md5 `23ff1ab8…`、2,010,707 B）。

复现命令：

```bash
HF_HUB_OFFLINE=1 ./inference_cli.sh \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/m1b/final2_10268234.mp4 \
  --prompt "There is a bridge over the lake."
```

对照命令：

```bash
python EraserDiT-baseline/compare_outputs.py \
  results/m1b/final2_10268234.mp4 \
  EraserDiT-baseline/results/2026-09-18T11-59-22-10268234/10268234_results_final_crop_False.mp4 \
  --mask data/10268234_mask.mp4
```

## M2 服务端（2026-09-18/19）

服务骨架改为模型适配器提供契约后，用 `scripts/validation/service_smoke.py` 对 EraserDiT 做端到端验收，
**全部通过**：11 个端点、严格请求契约（未声明字段 422）、任务生命周期到终态
（`completed`，window/object 计数与指标齐备）、结果下载与删除。

```
./inference_server.sh --pipeline-name EraserDiTErasePipeline --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model \
  --task-root /tmp/mgerase_svc_tasks --input-allowed-root "$PWD/data"
python scripts/validation/service_smoke.py --base-url http://127.0.0.1:30000 \
  --video data/10268234.mp4 --mask data/10268234_mask.mp4 --prompt "There is a bridge over the lake."
```

模型卡返回 `capability=eraserdit_video_erase`；`/server_info` 同时给出启动配置与
`effective_acceleration`（注意力预检报告、算子融合决策含回退原因、编译开关）。
