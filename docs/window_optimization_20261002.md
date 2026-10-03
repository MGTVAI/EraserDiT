# 单窗口并行配置与 Ulysses 打包验证

2026-10-02，L40S 上的 121 帧单窗口测试。本轮同卡对照中，双卡 CFG2 比 SP2
纯推理耗时减少 **21.5%**；四卡 CFG2×SP2 比 SP4 减少 **7.3%**。
这是并行配置选择带来的收益。Ulysses 打包代码优化另作同配置 A/B，默认仍为原路径。

## 条件与计时

- 示例视频和 mask 各截取前 121 帧，1920×1080，保留帧索引；无损 H.264 截取后逐帧 RGB 校验一致。
- BF16、SDPA、seed 42、guidance 3、50 配置步数、strength 0.8，每次恰好执行 40 个去噪步。
- NCCL、Ulysses、sharded linear；DiT 常驻，T5/VAE CPU offload、VAE low-memory。
  关闭缓存、编译、算子融合、量化、VAE tiling 和尾窗口减填充。
- 每种配置独立加载，先执行 2 步请求预热，再在同一 session 中执行 5 次正式请求。
  每次请求只有一个窗口；共 8 组、40 次正式请求。
- 纯推理取 `timing.pure_inference_seconds`，去噪取 `EraserDiTEraseDenoisingStage`。
  不统计模型加载、预热、视频读写和窗口提交。
- PyTorch 2.6.0+cu126。单卡使用 GPU 6，双卡使用 4、5，四卡使用 0–3。
  不同设备组并发运行，同一组内依次测原版配置和打包优化版；每组 A/B 固定同一组卡。
  功耗上限均为 350W，未锁频。GPU 占用、频率和进程每 15 秒采样。

## 测量结果

单位为秒；中位数后列出五次请求的范围。

| 配置 | 打包路径 | 纯推理中位数（范围） | 去噪中位数（范围） |
| --- | --- | ---: | ---: |
| 单卡 SP1 | 原版 | 227.75（226.80–228.01） | 213.58（213.01–213.97） |
| 双卡 SP2 | 原版 | 156.24（151.64–159.48） | 142.27（137.88–145.67） |
| 双卡 CFG2 | 原版 | 122.59（122.06–122.72） | 108.86（107.96–109.00） |
| 四卡 SP4 | 原版 | 90.78（90.25–91.34） | 76.99（76.51–77.41） |
| 四卡 CFG2×SP2 | 原版 | 84.19（83.95–84.27） | 70.43（70.34–70.68） |
| 双卡 SP2 | 优化 | 149.70（147.07–151.02） | 135.88（133.05–137.27） |
| 四卡 SP4 | 优化 | 88.19（88.03–90.45） | 74.58（74.45–76.45） |
| 四卡 CFG2×SP2 | 优化 | 83.40（82.55–84.33） | 69.72（68.92–70.67） |

双卡优先 CFG2，四卡优先 CFG2×SP2：这两组同卡配置对照的纯推理与去噪范围均不重叠。
非去噪推理约 14 秒，没有随 SP 度数同等缩短。

打包优化的纯推理中位数变化为 SP2 **−4.2%**、SP4 **−2.9%**、CFG2×SP2 **−0.9%**。
CFG2×SP2 的范围重叠，不视为已证明稳定提速。SP2 原版自身波动较大，SP4 两种路径的
纯推理范围略有重叠；数值属于本轮观察，不能推广到其他素材或硬件。
不同设备组共享 CPU/内存，未交替 A/B、未锁频；单卡与多卡也使用不同物理卡，
不把跨组加速比解释为严格隔离的扩展效率。保留 `reference` 为默认，`packed` 显式开启。

## 实现与正确性

`models/adapters/eraserdit/nccl_sequence.py` 将按目标 head 分片逐个 contiguous、再 cat
改成维度重排后一次 contiguous。等长输出分片也走一次打包；不等长输出分片保留原路径。
collective 的数据、顺序及收发长度不变；没有更改 attention 或 GEMM 精度。
四路输入打包的 CPU 算子检查为 4 次 copy 加 1 次 cat → 1 次 copy；这不是 GPU 加速比例。
rank report 的 `ulysses_packing` 记录实际路径。

- CPU 布局检查覆盖 SP2/SP4、FP32/BF16、batch 1/3、非连续数据、等长和不等长分片。
- 真实 NCCL 小模型检查覆盖 SP2、SP4、CFG2×SP2；新旧打包输出逐元素一致。
- 全部 40 次正式请求均为 121 帧、一个去噪窗口、40 步；每组 5 个输出文件完全一致。
- 三组打包 A/B 的全部 15 个优化输出文件与各自原版 SHA256 完全一致。
- 本片段上 CFG2 与 SP1 文件一致，CFG2×SP2 与 SP2 文件一致。
  不同 SP 度数仍可能改变 BF16 数值，本轮未新增跨拓扑质量门槛。
- CPU 专项 4 项通过；NCCL 打包专项 1 项通过，内部执行上述 3 种拓扑。

## 复现

从仓库根目录运行，选择已经分配的设备，输出目录必须不存在：

```bash
uv run --no-project python -m entrypoints.cli.benchmark_window \
  --run-dir outputs/window_reference --devices 0,1,2,3

uv run --no-project python -m entrypoints.cli.benchmark_window \
  --run-dir outputs/window_packed --devices 0,1,2,3 \
  --configs sp2,sp4,cfg2_sp2 --packing packed

CUDA_VISIBLE_DEVICES=0,1,2,3 ERASERDIT_TEST_DIT_NCCL=1 OMP_NUM_THREADS=1 \
  uv run --no-project python -m unittest \
  tests.test_nccl_dit.DistributedDiTTests.test_ulysses_packing_matches_reference -v
```

默认每组重复 5 次；`--prepare-only` 只准备素材与命令。CLI 擦除入口也可通过
`MGERASE_NCCL_PACKING=packed` 显式启用打包优化。
本轮本机产物在 `outputs/window_optimization_20261002/`：`summary.json` 为完整计时与文件核对，
各子目录保存 manifest、日志和输出，`gpu_samples.jsonl` 为资源采样。这些产物不随 Git 分发。
