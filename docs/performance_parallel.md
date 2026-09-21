# EraserDiT CFG / SP / VAE / DP 并行

最新验收规则见 [优化验收标准](optimization_acceptance.md)：非 cache/量化优化采用 SSIM ≥ 0.985、MSE ≤ 36、MAE ≤ 6；cache/量化按视觉大致一致验收。下文历史字节一致结果保留，不再作为必须条件。

2026-09-21 更新：已按用户本轮指定在物理 2、3、6、7 卡完成四卡组合测试。
原尺寸 CFG=2×SP=2 为 87.682 s，相对本轮单卡 210.302 s 加速 2.398×，输出字节一致。
补测纯 SP=4：209.114 → 118.665 s，1.762×，输出字节一致。
完整结果与常驻占用说明见 [四卡报告](performance_parallel_four_gpu.md)。

以下为 2026-09-20 实现说明。用户指定不实现 TP、PP；TeaCache、Cache-DiT 和文本投影缓存关闭。
先用物理 2、3 号卡测试；物理 6、7 号卡有占用时不启动四卡测试。

## 设备分组与实现边界

所有 GPU 编号在进程内都是 `CUDA_VISIBLE_DEVICES` 映射后的本地编号。
`--sp-degree S --cfg-degree C` 的 Transformer 使用 `S*C` 张卡：连续 S 张卡为一个 CFG 分支，
每个分支内部采用 Ulysses 的序列 / 注意力头交换。通信通过单进程多线程的 GPU 间 tensor copy，
尚非 NCCL 多进程实现。每层交换前后同步生产者和消费者，异常会中止同组 barrier 并等待工作线程退出。
不实现 TP 或 PP，每张参与卡持有完整 Transformer 权重。

`--vae-degree V` 在编码、解码阶段复用同一设备池。每个任务组的卡数为 `max(S*C,V)`，
目前各个 degree 支持 1/2/4，CFG 只支持 1/2，最多四卡。未声明设备顺序时从本地 0 开始。
主卡保持唯一的 scheduler、随机数发生器、后验采样、窗口衔接和视频写出。
模型副本属于窗口 / VAE 阶段，正常结束与异常退出均释放；副本创建计入端到端耗时。

当前入口要求 `fullgpu`，拒绝 torch.compile 和 Transformer 缓存组合。并行参数由 CLI
`entrypoints.cli.erase_eraserdit` 提供；未扩展 HTTP 服务的多 worker 调度。

### SP 的数值兼容模式

`--sp-linear-mode reference` 为默认值。保留输入投影的完整序列长度；在分片的线性层前填充
其他分片的零行，使 GEMM 形状及有效行位置与串行一致，输出只保留本分片。注意力仍按头分配，
分片间交换 Q/K/V 和注意力输出。这种模式有冗余线性计算和临时显存，不能声称线性层获得 S 倍
加速或整体显存按 S 等分。

`--sp-linear-mode sharded` 直接按局部 token 执行线性层，属于实验选项。
实测 BF16 GEMM 因 M 维变化产生小量舍入差异，经去噪放大；首版小尺寸 40 步输出
PSNR 26.17 dB / SSIM 0.9332，未过既有质量线，因此不作为推荐模式。
该失败数据来自输入投影改为完整长度之前的初版；最终配置仍需各自重新验收。
reference 模式在修复后的双窗口真实模型对照中与串行逐字节相同。
SageAttention 还会对 K 做序列均值归约；为保持归约形状，SP 先收集完整 K 并做中心化，
再选择本卡的注意力头，调用 Sage 时关闭重复中心化。该路径增加一次 K 交换，
已在 30,600 token 的独立注意力对照中逐位一致。

### VAE 空间分片与可选近似分块

`--vae-degree 2` 默认按高度分片激活，在每个卷积层交换相邻边界，保留完整时间轴和空间感受野。
保留原卷积形状，并对 RMSNorm 和下采样残差的 channel-group mean 保留全局空间归约布局；
其他卡的非边界内部行填零，计算后只保留本卡输出行。该兼容实现重复卷积和归一化 FLOPs，
收益需通过独立的显存 / 耗时测量判断，不能声称卷积计算已均分。支持空间像素重排式上下采样，
拒绝全局 GroupNorm、非单位高度卷积 stride/dilation、随机 decoder noise 等未适配配置。

显式加 `--vae-tiling` 才切换为近似分块，默认 tile=512、stride=448，均为 32 的倍数。
执行与本地 VAE `tiled_encode` / `tiled_decode` 相同的空间 tile、重叠混合顺序和裁剪。
每块保留完整时间轴，不切断因果时间上下文；先合并编码 moments，再在主卡采样 posterior。
每卡最多在途一块，不复制文本编码器。少于 V 张卡的 tile 数会降低 effective_degree 并报告。

分块本身会改变未分块 VAE 的结果。应先比较单卡 tiled 与多卡 tiled，再独立评估与 untiled 的质量差异；
并行一致性通过不能代替 tiling 质量验收。启用了随机 decoder noise 的 VAE 配置会明确拒绝此路径。
小尺寸强制 tile=256 / stride=192 的测试相对未分块基线仅 PSNR 19.82 dB、SSIM 0.7917，
未通过质量线，因此该设置只用于验证并行实现，不推荐用于输出质量。输入能放进一个 tile 时，
自动使用单卡未分块路径并报告 `input_fits_one_tile`；不会为了凑并行度强行切块。

### DP 独立任务并行

`python -m entrypoints.cli.erase_parallel` 将 task-file 的独立任务轮流分配到 D 个常驻子进程，
每个子进程处理自己的任务列表。GPU 分组互不重叠，每组内部可以启用 CFG / SP / VAE。
总卡数必须为 `D * max(S*C,V)`。要求显式设置 CUDA_VISIBLE_DEVICES，输出路径必须互不相同。
任一 worker 失败会终止本调度器创建的其他 worker，保存退出码与日志。
这是静态任务分配，尚未实现根据任务耗时动态窃取任务。
同一视频的相邻生成窗口继续串行，避免破坏上一窗口尾帧依赖。

## 两卡命令

所有推理使用已校验本地权重。保持 `--transformer-cache-mode off --no-cache-text-projections`。

```bash
CUDA_VISIBLE_DEVICES=2,3 ./inference_cli.sh \
  --model-path results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/parallel_sp.mp4 --prompt "There is a bridge over the lake." \
  --attention-backend sage_attn --sp-degree 2 \
  --transformer-cache-mode off --no-cache-text-projections
```

- CFG=2：将 `--sp-degree 2` 换成 `--cfg-degree 2`。
- VAE=2：加 `--vae-degree 2`；可与上述任一种两卡方式组合。近似 tile 模式另加 `--vae-tiling`。
- 原 `--cfg-parallel-device cuda:1` 保持兼容，但不要与新的 CFG/SP mesh 参数混用。

矩阵测试工具沿用 CLI 输入参数，`--output-path` 改为 JSON 报告路径：

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python scripts/parallel_benchmark.py \
  --model-path results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/parallel_matrix/full.json --prompt "There is a bridge over the lake." \
  --attention-backend sage_attn --matrix-configs serial cfg2 sp2 spatial_vae2 cfg2_spatial_vae2 sp2_spatial_vae2 \
  --transformer-cache-mode off --no-cache-text-projections
```

DP=2 的任务文件需至少两个任务，且每个任务声明独立 `output`：

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python -m entrypoints.cli.erase_parallel \
  --dp-degree 2 --parallel-run-dir results/dp_new_run --task-file tasks.json \
  --model-path results/cache_prediction_model \
  --transformer-cache-mode off --no-cache-text-projections
```

每个 task 自带 `video`、`mask`、`output`、`prompt`；也可以像普通 CLI 一样给公共输入参数。
`parallel-run-dir` 必须不存在，避免覆盖已有 worker 日志。

## 四卡组合

确认 2、3、6、7 全部空闲后，设置 `CUDA_VISIBLE_DEVICES=2,3,6,7`：

| 组合 | 参数 | 分组 |
| --- | --- | --- |
| CFG × SP | `--cfg-degree 2 --sp-degree 2` | 物理 2/3 正分支，6/7 负分支 |
| SP=4 | `--sp-degree 4` | 四卡协作，正负分支依次执行 |
| CFG × SP + VAE | 前者加 `--vae-degree 4` | VAE 阶段复用四卡 |
| DP × CFG | dispatcher `--dp-degree 2 --cfg-degree 2` | 2/3 一项任务，6/7 另一项 |
| DP × SP | dispatcher `--dp-degree 2 --sp-degree 2` | 两组分别处理独立任务 |
| DP × CFG/SP + VAE | 上两行加 `--vae-degree 2` | 每个任务组内部复用两卡 |

`bash scripts/parallel_four_gpu.sh <普通矩阵测试的输入参数>` 连续三次检查四卡显存占用 ≤64 MiB、
利用率为零后才启动四卡矩阵，否则退出 75。首参数 `--wait` 可每 30 秒继续等待；
默认不后台等待、不占卡。该检查不能替代集群调度器的独占资源分配。

## 原尺寸探索测量

素材 `10268234`，1080×1920、120 帧、50 个采样步 / 40 个有效去噪步，SageAttention。
这是单次探索测量，尚非至少五次重复的稳态统计。

| 模式 | 端到端耗时 | 对各自串行加速 | 输出 |
| --- | ---: | ---: | --- |
| 既有 CFG=2 | 118.259 s | 1.774×（串行 209.788 s） | 字节一致 |
| SP=2 reference | 150.021 s | 1.400×（串行 209.975 s） | 字节一致 |
| CFG=2 + VAE=2 空间分片 | 121.353 s | 1.730×（串行 209.975 s） | 字节一致 |

SP 主卡 / 辅卡峰值 allocated 为 47.115 / 5.501 GiB，主卡峰值没有下降。
CFG+VAE 组合为 48.778 / 29.509 GiB；相对 CFG 单独没有显示额外加速或主卡显存收益。
因此当前双卡优先 CFG=2，VAE 空间并行保留为可组合的实验路径。
组合原尺寸编码、解码也已验证，见 `full_cfg_spatial_vae.json`。
最终 SP 产物为 `full_sage_centering.json`；`full_reference.json` 的 SP 项来自修复 K 中心化之前，
不能当作最终数值结果。修复后输出 SHA256 为
`0b50b4c9176930c1adbb289298b5ea67c70525c352b58138c6b0ca786b876e0b`。

## 本轮验证

- 两卡真实模型：CFG=2 与未分块单卡输出逐字节一致。
- SP reference：192×320、25 帧、双窗口、5 个有效步，与未分块串行逐字节一致。
- VAE=2、CFG=2+VAE=2、SP=2 reference+VAE=2：与相同 tile=256/stride=192 的串行逐字节一致。
- 原生 VAE `tiled_encode` / `tiled_decode` 与两卡实现的编码 moments、解码 tensor 逐位一致。
- 不分块的 VAE 空间并行：原生编码 moments / 解码 tensor 逐位一致；
  192×320、25 帧、双窗口、50 采样步下，serial / VAE=2 / CFG=2+VAE=2 / SP=2+VAE=2
  四份 MP4 逐字节一致，见 `spatial_vae_reference.json`。
  单次耗时分别为 12.375 / 12.439 / 8.029 / 20.556 s；小尺寸 VAE 单独并行没有加速，
  SP 通信开销超过收益。`spatial_vae.json`、`spatial_vae_reduction.json`、`spatial_vae50.json`
  均来自修复 RMSNorm 布局之前，不作为最终结果。
- DP=2：两卡各处理两个连续任务，四份输出都与独立单卡基线逐字节一致。
- 三窗口、33 帧、两轮共 14 次请求：serial / CFG=2 / SP=2 逐字节一致；
  tiled / VAE=2 / CFG=2+VAE=2 / SP=2+VAE=2 四种模式逐字节一致。
  此处 tile=256/stride=192，仅验证并行一致性。结果见 `three_windows.json`。
- 两卡实际模型、异常中断与下一窗口恢复通过；CPU 四分片交换测试覆盖非整除 token 数与多轮 barrier。
- 四个逻辑 rank 的 CFG=2×SP=2 用真实小型 CPU Transformer 验证分组和预测，
  但这不代表四张物理 GPU 的数值或性能验收。
- 全量 53 项测试全部通过，包含显式启用的真实双卡 Transformer / VAE 测试；见 `regression_final.log`。
- 四卡性能 / 输出组合测试已于 2026-09-21 完成，见上方四卡报告；7 号卡有其他常驻占用，性能按单次探索数据报告。

产物目录 `results/parallel_matrix/`，最终结果索引为 `validation_summary.json`。`small.json` 是初版 sharded SP；
`reference_small.json` 为修复后的 reference 模式；不要混合两版性能和质量结论。
`dp_run/report.json` 记录 worker 卡号与退出码。

测试命令：

```bash
CUDA_VISIBLE_DEVICES=2,3 ERASERDIT_TEST_TWO_GPU=1 \
ERASERDIT_TEST_MODEL="$PWD/results/cache_prediction_model" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python -m unittest discover -s tests -v
```
