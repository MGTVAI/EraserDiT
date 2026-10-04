# L40S INT8 / FP8 量化对照（2026-10-03）

默认仍为 BF16。提供实验性 `fp8_w8a8_native`，并优化 INT8 的 FFN 扩展层。
小分辨率时量化启动和缩放开销可能超过 GEMM 收益；不能按位宽推算加速比。

## 实现

- INT8：权重按输出通道、激活按 token 对称量化。token 数 ≥1024 且输出宽度 ≥2×输入宽度时，
  Triton GEMM 融合缩放、bias 和 BF16 写回，避免完整 INT32 中间结果；其他形状使用 `torch._int_mm`。
- FP8：E4M3，权重按输出通道、激活按 token 动态缩放。L40S + PyTorch 2.6 的原生逐行缩放
  GEMM 不可用，因此使用单位标量缩放的 `torch._scaled_mm`，FP32 输出后单独应用行/通道缩放。
  `use_fast_accum=False`。激活使用直接 FP32→E4M3 PTX 转换，避免经 FP16 的两次舍入。
- `blocks` 量化 224 个 Linear；`ffn` 仅量化 56 个 Linear。输入输出投影、条件处理、
  LayerNorm、Softmax、T5 和 VAE 保持原精度。无 BF16 权重副本或 BF16 GEMM 回退。
- 当前不支持与完整 DiT compile、NCCL DiT 路径、DiT 权重卸载组合。FP8 要求 SM89 或更新架构；
  本轮实际验证 L40S。量化模式、转换耗时、层数和调用次数在 `quantization` 中报告。

INT8 是均匀间隔舍入，异常值可能拉大间隔；FP8 的范围更灵活，但有效尾数较少。
哪种误差更小取决于激活分布。单个 Linear 的随机输入误差不能代表多步去噪后的画质。
SSIM 衡量与 BF16 输出的一致性，不是“模型准确率”；擦除区域和时间变化需单独检查。

## 测量口径

L40S 48 GB、SM89、PyTorch 2.6.0+cu126。真实 `data/model/` 权重、示例视频 `113000356`、
seed 42、50 个配置采样步、strength 0.8（实际去噪 40 步）、CFG 3.0、SDPA。
关闭缓存、文本投影缓存、compile 和手工算子融合；DiT 常驻，T5/VAE CPU 卸载，VAE low-memory。
所有视频模式都使用相同 2 步预热；表中请求耗时排除加载和预热。

- 短片：前 33 帧，视频 Lanczos 缩小到 640×360、mask 最近邻缩放，infer_len=33。
  每个模式在同一常驻 session 中重复五次。初始 blocks 轮在 GPU 0；
  后续 FFN 筛选与另一张 GPU 上的 1080p 验证重叠，不作为隔离性能结论。
- 原分辨率：前 121 帧、1920×1080、infer_len=121，在 GPU 2 上顺序执行。
  每个模式一次正式请求，用于质量及初步速度筛选，不能视为稳定的五次中位数。
- SSIM：解码 RGB 0–255，11×11 Gaussian、sigma=1.5、reflect 边界；mask 为原始 mask >127。
  空 mask 帧不参与 mask/edge 指标，另记录有效帧数；edge 是 11×11 膨胀减腐蚀的边界带。
  时间误差为相邻帧 `(候选−BF16)` 差值的 MAE，
  不能直接当作感知闪烁评分。视频编码也计入最终像素差异。

## 初始短片 blocks 筛选

下表使用融合优化前的 INT8，与 FP8、BF16 比较；耗时为五次中位数。

| 模式 | 去噪秒 | 端到端秒 | 峰值 allocated GiB | 平均 SSIM | mask 内 SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 9.335 | 12.212 | 6.624 | 1 | 1 |
| INT8 blocks | 14.687 | 17.609 | 5.077 | 0.98338 | 0.96598 |
| FP8 blocks | 15.455 | 18.383 | 5.077 | 0.98608 | 0.97842 |

两种 blocks 量化都没有短片加速收益。FP8 更接近 BF16，但该短窗口下 BF16 自身也存在擦除残留，
因此不能凭这些相似度宣称任务质量通过；需要原始分辨率和标准窗口验证。

## 原分辨率 FFN 对照

同一 GPU 顺序执行，每种一次正式请求；这是筛选结果，未证明稳定的端到端加速。

| 模式 | 去噪秒 | 纯推理秒 | 端到端秒 | allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 214.881 | 229.378 | 253.660 | 24.441 / 43.105 |
| INT8 FFN | 214.101 | 228.595 | 251.543 | 23.566 / 43.230 |
| FP8 FFN | 224.645 | 239.122 | 262.157 | 23.566 / 43.230 |

| 模式 | 平均 SSIM | 最低帧 SSIM | mask SSIM | 最低 mask 帧 SSIM | RGB MAE（0–255） | 时间差异 MAE（0–255） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 FFN | 0.98721 | 0.98263 | 0.98762 | 0.97979 | 0.891 | 0.983 |
| FP8 FFN | 0.98883 | 0.98545 | 0.98953 | 0.98316 | 0.714 | 0.889 |

mask 指标覆盖 103 个非空帧，整帧指标覆盖全部 121 帧。MAE 列数值使用 0–255 像素标度，
不把 SSIM 的差值解释成准确率损失。抽查首/中/末帧及误差较大的 mask 区域；未见明显画面崩坏，
这仍是单素材、单 seed 验证，不保证其他视频或多窗口衔接。

两种 FFN 量化将选中层权重及 bias/scale 缓冲从约 1.751 GiB 减至 0.877 GiB；
峰值 allocated 降低约 0.87 GiB，reserved 未同步下降。保留 BF16 默认；量化目前更适合作为
显存与质量折中选项，不能宣称已有稳定的完整视频加速。

## Linear 局部性能

使用第 0 层真实权重，输入为固定 seed 的 BF16 正态随机数。包含激活量化、GEMM 和缩放；
三次预热后，五组每组十次，交替模式顺序，CUDA event 计时，包含 host 调度导致的设备空闲。
以下为最终实现，同一轮中位数（毫秒），不代表端到端速度。

| 层与 M×K×N | BF16 | INT8 | FP8 |
| --- | ---: | ---: | ---: |
| Q，1280×2048×2048 | 0.065 | 0.250 | 0.271 |
| FFN 扩展，1280×2048×8192 | 0.206 | 0.209 | 0.278 |
| FFN 扩展，10240×2048×8192 | 1.637 | 0.993 | 1.894 |
| FFN 扩展，32640×2048×8192 | 5.535 | 3.287 | 6.184 |
| FFN 收缩，32640×8192×2048 | 4.792 | 4.335 | 4.514 |

INT8 扩展层优化前在两个较大 M 上分别为 1.709 / 5.406 ms，优化后 0.993 / 3.287 ms；
前后两轮的 BF16 计时也有波动，不能将这个比例当作稳定的完整模型收益。
融合内核在已测形状上与原 INT8 GEMM+epilogue 逐元素一致，未通过降低累加精度换速度。
FP8 在这里没有明显优势，仍保留为质量与显存对照。

## 复现与验证

Linear 测试（输出文件不能已存在）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 uv run --no-project python \
  -m entrypoints.cli.benchmark_quantization \
  --model-path data/model --output results/quantization_new/linear.json
```

视频在正式 CLI 上分别设置以下参数；其他配置必须一致：

```text
--transformer-quantization none
--transformer-quantization int8_w8a8_native --quantization-scope ffn
--transformer-quantization fp8_w8a8_native --quantization-scope ffn
```

本地完整命令、tasks、日志、逐帧指标、对比图和视频在 `results/quantization_20261003/`，
原分辨率在其 `full121/` 子目录；`run_video.py`、`run_full.py`、`quality.py`、`summarize.py`
保留实际实验步骤。`manifest.json` 记录代码哈希、依赖和测量限制。实验产物不入 Git。

CPU 回归：210 项，跳过 61 项，零失败。GPU 量化回归 7 项通过，覆盖真实 INT8/FP8 GEMM、
FP8 舍入、空输入/零输入、bias、非整块行数、融合数值一致性、转换幂等及实际层执行。
装配/compile 等关联测试也通过。不同素材、seed、多窗口衔接和其他 GPU 未完成质量验收。

## FP8 变慢的原因复核

追加拆分测试使用同一 L40S、固定随机 BF16 权重/激活、五组每组二十次；
CUDA Graph 列用于减少主机调度空隙，不能直接当作当前 pipeline 的速度。
普通调用仍保留 Python/Triton 调度开销。原来三次预热、十次调用的微测量不适合被解释成纯 GPU kernel 时间。

| M×K×N / 路径 | 普通调用 ms | CUDA Graph ms |
| --- | ---: | ---: |
| 1200×2048×8192，BF16 Linear | 0.201 | 0.204 |
| 同形状，当前完整 FP8 Linear | 0.275 | 0.149 |
| 32640×2048×8192，BF16 Linear | 5.411 | 5.358 |
| 同形状，当前完整 FP8 Linear | 6.119 | 5.972 |

大扩展层独立测量：激活量化 0.323 ms、FP8 GEMM（FP32 输出）4.287 ms、
缩放/bias/BF16 写回 2.386 ms。独立阶段与完整调用的缓存、温度等条件不同，不能直接相加。
仅把 GEMM 输出改为 BF16 的诊断测量为 3.394 ms，但这没有包括行/通道缩放，
也没有进行完整视频质量验收，不能作为已实现的替代方案。

当前 FP8 路径为：动态量化 → FP8 GEMM → FP32 中间矩阵 → 独立缩放/bias → BF16。
大扩展层的 FP32 中间矩阵有 32640×8192 个元素，约 1.07 GB（十进制）；额外写入和读取约 2.14 GB。
L40S 官方带宽为 864 GB/s，理想条件下这一额外流量就对应约 2.48 ms。
这是大输出矩阵的额外显存往返，不是说 FP32 寄存器累加本身不该使用。
硬件参数见 [NVIDIA L40S](https://www.nvidia.com/en-us/data-center/l40s/)。

另一项使用真实权重、同尺寸合成输入的单次 BF16 DiT forward profiler 显示：
28 个 self-attention 内部计算约 939 ms，28 个 FFN 扩展 Linear 约 196 ms、收缩约 202 ms；
这些是 profiler 的算子设备时间，不是请求耗时。FFN 量化并未优化占时更大的 self-attention 内部计算。
因此局部 GEMM 提速也不能按相同比例外推到整个去噪过程。

结论：当前 FP8 实现尚不是高效部署路径，主要问题是显存往返和额外调用；
不能据此得出“L40S 不适合 FP8”或“FP8 天生更慢”。后续应优先将缩放融合进 GEMM epilogue、
让量化与相邻算子融合、减少主机调度，并在精度验证后选择累加策略。
简单改成纯 PyTorch tensorwise 动态量化仍会生成大型临时张量，本轮探针未带来通用收益。
诊断脚本与原始数据保存在 `results/quantization_diagnosis_20261003/`。
