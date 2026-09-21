# EraserDiT 缓存优化

最新完整视频、指定参考输出的正确性复测见 [cache_correctness.md](cache_correctness.md)。

参考本地 sglang 的 `runtime/cache/cache_dit_integration.py`：其 cache-dit 接口支持
TaylorSeer 一阶预测；MGErase 的 `runtime/cache/cache_dit.py` 则保留分支独立的前段探针、
中段残差和前后段完整计算。当前实现沿用这些缓存边界，并为 EraserDiT 增加以下改进。

- `cache_text_projections=true` 缓存固定文本的 caption projection 及每层 cross-attention K/V。
  不跳去噪步骤，可单独与 `transformer_cache_mode=off` 使用，也可与两种残差缓存组合。
  CFG/窗口独立；提示词 tensor identity、版本或 autocast 配置变化时失效。没有 mutation counter
  的 inference tensor 保守重算。权重在窗口内保持固定，不向模型挂载持久缓存。
- 残差在 FP32 中做减法、保存和恢复，再转换回 block 的输入精度。原 bf16 减法会再次舍入，
  即使输入完全不变，也可能无法还原上次完整输出；新实现避免 bf16 级别的额外舍入。
- `cache_residual_predictor=linear` 用最近两次**完整计算**的残差及真实步距估计一阶变化率。
  复用步不更新变化率，外推距离不超过最近完整计算之间的步距。不同 CFG 分支独立预测；
  布局改变或步序中断后重新建立历史。warmup、末步保护、连续复用上限保持生效。
- 正常退出和异常退出均释放预测张量；窗口指标增加 `residual_predictor`、`residual_dtype`、
  `predicted_steps`，缓存内存峰值包含预测张量。

文本缓存默认 `auto`（请求字段为 `null`）：开启 TeaCache/cache-dit 时自动复用文本投影，
残差模式为 `off` 时默认不启用；可用 `--no-cache-text-projections` 显式关闭。
预测开关默认 `none`，残差缓存模式默认 `off`。`none` 使用 FP32 残差复用；`linear` 额外保留一份
FP32 变化率，增加显存占用。LTX095 的缓存数值策略保持原样。TeaCache 的第一层调制输入指标仍未校准，
一阶预测不是 TeaCache 系数拟合，也不能保证任意阈值下的生成质量。

CLI、task-file、JSON 和 multipart 请求均可传入新参数。优先使用无损档：

```bash
--transformer-cache-mode off --cache-text-projections
```

需要更多加速时使用保守组合档；会引入额外视频误差：

```bash
--transformer-cache-mode cache_dit --cache-text-projections \
--cache-dit-residual-diff-threshold 0.03 --cache-residual-predictor none
```

一阶预测、更高阈值及更多后段完整层没有在本轮扫描中稳定胜出，**不推荐默认开启预测或提高阈值**。
两类缓存均不支持与整个 Transformer 的 `torch.compile` 组合。

## 复测

以下脚本在同一个常驻 session 中先完成一次完整请求预热，再交错重复运行所选配置，默认每组五次。
输出包含任务、原始日志、逐窗口缓存统计、耗时中位数、重复输出一致性，以及整帧、擦除区、
非擦除区、边缘的 SSIM/PSNR 和帧间变化误差。加载和预热不计入样本。

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 \
  /mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python scripts/cache_benchmark.py \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model --video results/dynamic_offload_smoke/video_33.mp4 \
  --mask results/dynamic_offload_smoke/mask_33.mp4 \
  --prompt "There is a bridge over the lake." --infer-len 17 \
  --directory results/cache_retest --repeats 5 --configs text dbc_none dbc_text
```

本次完整视频正确性复测使用 GPU 2、3；下方 GPU 0、1、6、7 的结果属于此前短视频实验。
单卡运行由 `CUDA_VISIBLE_DEVICES` 指定。脚本默认 fullgpu，可显式选择卸载策略；
`--tea-threshold`、`--dbc-threshold`、`--back-blocks`、`--configs` 可用于扫描。
数值一致性指标以 off 输出为参考，不等同于擦除质量的人工验收。

模型为 `jieeliu/EraserDiT@904fb412da76235085dbbccaefdbde4979fa3d29`。本地权重逐文件核对了
官方 SHA256。后续统一使用模型路径
`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model`；
`results/cache_prediction_model/availability_verified.json` 记录本次全部 16 个文件的校验结果，
`results/cache_prediction_model/verified_manifest.json` 保留最初的权重来源记录。
比较器使用 `-vsync 0` 逐帧解码，避免裁切视频的时间戳导致 ffmpeg 补帧、错配 mask。


## 实测结果（2026-09-20）

统一设置：SDPA、bf16、seed=42、fullgpu；50 调度步、strength=0.8，每窗口实际 40 步；
infer_len=17、overlap=9，33 帧共三个窗口、240 次 CFG 前向。所有 SSIM/PSNR 都相对无缓存输出。

GPU 6 小尺寸 192×320：完整请求预热后，每组五次交错运行。下表为端到端中位数，已排除加载和预热。

| 配置 | 秒 | 相对 off 耗时减少 | 整帧 SSIM-Y | 擦除区 SSIM |
|---|---:|---:|---:|---:|
| off | 17.15 | — | 1.00000 | 1.00000 |
| 仅文本缓存 | 16.21 | 5.5% | 1.00000 | 1.00000 |
| FP32 cache-dit，关闭文本缓存 | 15.49 | 9.7% | 0.96982 | 0.94491 |
| FP32 cache-dit + 文本缓存 | 13.78 | 19.6% | 0.96982 | 0.94491 |

组合档中位加速比 **1.244×**；去噪中位耗时 15.69 → 12.42 秒。off 的五次范围为 16.75–18.32 秒，
组合档为 13.62–14.81 秒。各档重复输出 SHA256 均一致；文本缓存输出与对应未开启文本缓存的档位逐字节一致。
组合档每请求复用 38/240 次中段、减少约 15.3% 的 block 执行。小尺寸缓存保留峰值约 63.2 MiB，
原分辨率第一组约 202.4 MiB；统计包含借用的提示词 storage，跨 CFG/view 去重，不等于 allocator 峰值。

在独立旧代码 checkout `e82f3b4` 上复测同参数 cache-dit 0.03：整帧 SSIM/PSNR 为 **0.95923 / 28.95 dB**，
擦除区为 **0.92376 / 25.93 dB**；新实现为 **0.96982 / 30.97 dB**、**0.94491 / 28.71 dB**。
旧/新 off 解码像素一致。小尺寸擦除区 SSIM 仍低于 0.95；要求完全一致时应使用仅文本缓存档。

两组原分辨率素材的前 33 帧，以下每档仅一次，不作为稳定大尺寸加速结论：

| 素材 / GPU | off 秒 | 组合档秒 | 整帧 SSIM / PSNR dB | 擦除区 SSIM |
|---|---:|---:|---:|---:|
| 10268234，1080×1920 / GPU 7 | 72.35 | 60.36 | 0.98998 / 38.52 | 0.98487 |
| 113000356，1920×1080 / GPU 6 | 71.04 | 61.33 | 0.98873 / 44.62 | 0.97484 |

两组的仅文本缓存输出也与 off 逐字节一致。原分辨率第一轮 GPU 0，以及早期 GPU 1 重复组受到外部任务
占用变化影响，保留质量结果但不采用其耗时结论；上述重测保存了 GPU 6/7 负载采样。
第 16 帧并排检查显示 cache 与 off 接近，但 off 本身存在明显填充伪影/残留。本轮是相对误差验证，
不是整体擦除质量验收；尚未覆盖完整 120/145 帧视频及原默认 infer_len=121 的配置。

38 项测试通过，包含 GPU bf16 等价、CFG/布局隔离、提示词原地修改失效、缓存 storage 去重、
CLI/JSON/multipart 和异常清理。真实模型在 GPU 7 通过 dynamic_offload 2 GiB、连续请求、
首个完整 CFG 步后异常注入及恢复一致性验证；缓存 tensor 引用释放、权重回 CPU、预算/事件队列归零。

汇总：`results/cache_optimization_summary.json`；五次重复与负载：`results/cache_optimized_stable/`；
原尺寸：`results/cache_optimized_full_a_retry/`、`results/cache_optimized_full_b/`；
故障恢复：`results/cache_optimized_recovery/report.json`；分层/预测扫描：`results/cache_prediction_sweep*/`；
帧对比：`results/cache_prediction/visual_comparison.jpg`。所有运行均保留 tasks、runs、quality 和日志。
