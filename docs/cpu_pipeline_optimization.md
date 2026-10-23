# S4：CPU读取与缓存复用

S4 = S3 + CPU读取与缓存复用。本配置减少视频读取、数组复制、窗口组装、CPU/GPU 搬运及重复文本编码，沿用 BF16/SDPA 和原 DiT/VAE 算法。优化显式启用，库默认路径保留。完整验证范围为单卡、单对象 `windowed_streaming`、uint8 帧缓存和固定 145 帧 S3 素材；不将结果外推到多卡、多对象或任意视频。

## 启用

按[安装说明](setup.md)准备 Linux、CUDA Toolkit、uv、`.venv`、FFmpeg、模型和素材，然后从仓库根目录执行：

```bash
source scripts/activate_cpu_pipeline.sh
```

再运行 [README](../README.md#快速开始) 的单卡 S3 命令。脚本只设置本配置的环境变量，使用 CPU 解码，不选择 GPU、不安装环境、不加载模型，也不改变当前目录。库默认关闭这些可选优化；两个读取线程需要显式选择 `2`。

以下文件随仓库分发，不依赖 `outputs/setup/` 或用户目录。已有实验结果、模型、视频和环境仍保存在本机。

S4 的 15 个设置读取及核验清单集中在 `scripts/cpu_pipeline_config.py`。脚本适配入口通过
`scripts.benchmark_l40s.main(extra_env=...)` 传入需要记录的设置；通用 benchmark 不包含 S4 开关名。
准备和推理均走该适配入口，保证冻结记录、实际子进程环境和汇总核验一致。
直接使用通用 CLI 时不会自动登记这些 S4 设置；S4 复测请使用本页脚本入口。

## 保留功能与开关

| 开关 | 启用值 | 行为与边界 |
|---|---:|---|
| `MGERASE_VIDEO_UINT8_GPU_NORMALIZE` | 1 | CPU 视频保持 uint8，上传后转 FP32 并归一化；使用设备张量除数保持原舍入。仅对象流式 uint8 输入路径。 |
| `MGERASE_VIDEO_UINT8_GPU_CLEAR_CACHE` | 1 | 每窗口视频转换后清理一次可回收 CUDA 缓存，包括预热；主要控制保留显存，不保证提速。 |
| `MGERASE_PIPE_READ_THREADS` | 2 | 同时推进视频/掩码两个读取管道，核对帧位置和帧数；原串行值为 1。 |
| `MGERASE_MASK_PIPE_R_ONLY` | 1 | 相同 FFmpeg RGB 转换后提取 R 通道再传输，不改为灰度转换。 |
| `MGERASE_STREAMING_COPY_OPT` | 1 | readinto 直接填独立数组；缓存接管数组；淘汰仅保留必要尾部；重叠窗直接填最终缓冲。 |
| `MGERASE_MASK_WINDOW_BUFFER_OPT` | 1 | uint8 掩码直接转换进最终 FP32 窗口，原地归一化并处理零前缀/非活动区。 |
| `MGERASE_OUTPUT_UINT8_GPU_CONVERT` | 1 | GPU 按原 clamp/mul/round 顺序转 uint8、整理 THWC，再回传 CPU；CPU 输入保持原路径。 |
| `MGERASE_WRITER_HANDOFF_OPT` | 1 | 完整输出块转交 writer，截断块只复制缓存保留尾部；队列深度不变，写完释放引用。 |
| `MGERASE_MASK_THRESHOLD_INPLACE` | 1 | 复用新读取的可写 uint8 掩码，原地比较并转 0/255；保留阈值和边界语义，不适用输入回退。 |
| `MGERASE_STYLE_UINT8_GPU_UPLOAD` | 1 | 颜色参考帧以 uint8 上传后转 FP32，原颜色统计、截断量化和运算顺序不变。 |
| `MGERASE_MASK_DILATE_CAST_OPT` | 1 | 布尔比较直接转 FP16/uint8，省去 int64 中间数组；原核、迭代和近似二值化时机不变。 |
| `MGERASE_WINDOW_UINT8_VALIDATION_OPT` | 1 | uint8 视频跳过 NaN 扫描；维度检查、浮点/复数及浮点掩码验证保留。 |
| `MGERASE_TEXT_SESSION_CACHE` | 1 | Session 内有界缓存一组 CPU 提示词嵌入；既有窗口缓存优先，完整键失效、训练模式或 Session 关闭时不复用。 |
| `MGERASE_MASK_SCAN_REUSE` | 1 | 保留最多首 16 帧，最大值确定后继续同一 reader；未确定时完成原扫描并重开，保留软/暗掩码语义。 |
| `MGERASE_FIRST_WINDOW_BORROW` | 1 | 单对象首窗可写、连续、精确首块时借用源缓冲；提交前独立保留原始尾部，其他场景复制。 |

以上为 14 项改进、15 个环境设置。完整块交接省去的是交接复制，编码器仍需读取像素；首窗借用仍保留必要原始尾部，不能称为整个请求零复制。

仅回退某项可将其开关设为 `0`，读取线程设为 `1`。性能复测继续保持 `MGERASE_VIDEO_UINT8_GPU_CLEAR_CACHE=1`，并核对实际生效。未启用下一窗预取、年轻代 GC、GPU 解码或新增 NVTX 标签；原窗口 GC、CUDA 同步与缓存释放流程保留。

## 完整 S4 复测（2026-10-14）

GPU0、NVIDIA L40S；两个独立进程串行运行，各自加载和预热，同一冻结源码、输入、模型、参数与环境设置。代码基点为 `c9f7e6d4259ec937e59ca97febc3ae821dd4b1b5` 加本次工作树优化。

- 输入 `data/113000356.mp4` / `data/113000356_mask.mp4`：1920×1080、145 帧、24000/1001 fps。
- prompt：`There is a bridge over the lake.`；负提示词沿用 CLI 默认；seed42、CFG3、50 步、strength0.8，每窗口实际 40 步。
- infer_len121、overlap9、compact-tail-padding 关闭；尾窗仍按原模型长度填充。
- BF16、显式 SDPA、精确 Triton `qk_rmsnorm_rope,rmsnorm_adaln`；量化、Transformer 缓存及编译关闭。
- 文本/VAE/DiT 组件 CPU offload、VAE low-memory/cached offload；allocator 上限 22 GiB，任务峰值目标 24 GiB。
- Python3.10、Torch2.6.0+cu126、CUDA12.6、NumPy2.2.6；OMP/MKL8；确定性模式开启，`CUBLAS_WORKSPACE_CONFIG=:4096:8` 在导入 Torch 前设置。

| 指标 | 第一次 | 第二次 | 均值 |
|---|---:|---:|---:|
| 请求秒 | 401.039532 | 401.090283 | **401.064908** |
| 模型阶段合计秒 | 391.323801 | 391.866372 | 391.595086 |
| 文本秒 | 0.005745 | 0.006735 | 0.006240 |
| VAE 编码秒 | 16.420606 | 16.348112 | 16.384359 |
| DiT 去噪秒 | 364.408721 | 364.478438 | 364.443580 |
| VAE 解码秒 | 10.482696 | 11.027034 | 10.754865 |
| 任务 NVML 峰值 GiB | 21.658203 | 21.658203 | 21.658203 |
| PyTorch allocated 峰值 GiB | 18.178776 | 18.178776 | 18.178776 |
| PyTorch reserved 峰值 GiB | 20.812500 | 20.812500 | 20.812500 |
| 进程树 RSS 峰值 GiB | 38.552986 | 39.570377 | 39.061682 |

均值只使用这两次新测量；两次相差 0.050751 秒，样本标准差 0.035886 秒。请求时间排除加载和预热，内存峰值覆盖启动到退出；峰值均值只是各轮峰值的平均。模型阶段合计包含文本、VAE、DiT，不是纯 GPU 时间。

两份完整 145 帧输出与原 S3 逐字节一致，SHA256：

```text
e5c1f45a46dc75eccd093c3df763903478b6347d604877d85b55ae28a8060ba5
```

预热和正式窗口上传后清理均实际生效，首窗借用生效，正式首窗文本 Session 缓存命中。文本收益属于相同提示词热复用；新提示词和新 Session 仍需编码。该复测没有增加 CUDA 同步、Event、GC、empty_cache、NVTX 或 Profiler。

本机原始结果目录标识为 `all_opt_repeat_20261014_091719`，保存 CLI/env、279 个冻结源码文件哈希、输入哈希、run.log、report、summary、memory.jsonl、实际生效记录和完整输出；不随 Git 分发。

## 局部收益与历史证据

各行来自对应轮次，不能相加为累计收益。

| 子项 | 原路径 → 候选 | 口径 |
|---|---|---|
| 121 帧视频输入转换 | 1.054 → 0.108 秒 | 5 次局部中位数；视频输入逻辑载荷减少 75% |
| 两路读取 | 5.533 → 2.827 秒 | 3 次真实输入中位数；流式掩码管道载荷减少 2/3 |
| 读取/缓存/组装/淘汰 | 4.125 → 2.875 秒 | 3 次局部合计中位数；相关正式显式复制 6.551 → 2.134 GiB |
| 掩码窗口构造 | 0.629 → 0.290 秒 | 5 次局部中位数 |
| GPU 输出转换 | 2.273 → 0.318 秒 | 一组完整请求中的转换子项 |
| writer 缓存交接 | 0.569 → 0.000023 秒 | 一组完整请求；相关交接复制 0.840 → 0 GiB |
| 掩码阈值 | 0.730 → 0.260 秒 | 一组完整请求中的阈值子项 |
| 颜色参考上传 | 3.045 → 0.308 秒 | 一组完整请求的 API 主机 wall，可能含 GPU 等待 |
| 掩码膨胀 | 29.585 → 22.655 ms | 8 帧局部中位数；增量 allocated 574 → 240 MiB |
| 窗口验证 | 0.473 → 0.139 秒 | 一组完整请求的 stage |
| 文本热复用 | 1.300 → 0.006 秒 | 一组完整请求；正式 T5 调用 2 → 0 |
| 掩码预扫描 | 0.750 → 0.446 秒 | 一组完整请求；管道累计读出 161 → 145 帧，非 FFmpeg 内部计数 |
| 首窗组装 | 0.438 → 0.000808 秒 | 前后基线均值与单次候选；省去 717.847 MiB 整窗复制 |
| 每窗口上传后清理 | 21.365 → 19.451 GiB | 专门配对的任务 NVML 峰值，活跃分配不变 |

早期三项组合三轮请求均值 419.346 → 410.103 秒（2.20%），其缓存清理为 0，且不含后续改进，不能当作当前全配置的累计加速。当前没有全组合相对原始配置的同条件重复对照。两次复测样本较少，不证明所有素材或负载下的稳定性。

每窗口清理发生在视频上传后，后续颜色处理/输出仍可能抬高 reserved 峰值；组合任务峰值约 21.658 GiB，当前样例低于 24 GiB。主机 wall 可能承担 GPU 等待，子项包含在父项中，并行阶段重叠；逻辑字节量不是 PCIe/DRAM 硬件计数。

## 复现入口

只冻结源码、参数和输入哈希，不执行推理：

```bash
bash scripts/run_cpu_pipeline_benchmark.sh --gpu 0 --prepare-only
```

在已分配的空闲 GPU 上运行两次独立完整请求并核对原 S3 输出摘要：

```bash
bash scripts/run_cpu_pipeline_benchmark.sh --gpu 0 --runs 2 \
  --reference-sha256 e5c1f45a46dc75eccd093c3df763903478b6347d604877d85b55ae28a8060ba5
```

可通过 `--run-dir` 指定新目录，默认写入被忽略的 `results/`；已存在目录不会覆盖。默认 GPU0，请根据实际分配修改。入口复用 `benchmark_l40s --profile s3 --cache off --repeats 1`，每轮一个进程、独立加载和预热，共用同一冻结源码。通过元数据观察器核对清理、借用和文本命中，不添加额外 GPU 操作。

每轮 `run_N/` 保存原始日志、报告、参数、内存采样、冻结源码和视频，`run_N-runtime.json` 保存实际生效信息；`cpu_pipeline_summary.json` 给出本批均值和核验项，全部通过后生成 `ALL_COMPLETE`。没有传入参考 SHA256 时，只检查各轮输出互相一致，不能称为与原基线质量一致。

## 回归

无需模型、隐藏 GPU 的针对性回归见 [tests README](../tests/README.md#cpu-数据路径优化回归)。GPU 数值检查有显式 CUDA 跳过条件；完整模型质量和性能验证仍以本节复现方式为准。发布用脚本、文档整理没有改变已测的运行时源码。
