# INT8 FFN 融合与分块优化（2026-10-03）

性能优先继续优化已有 INT8 路径：合并 BF16 GELU 与下一层的逐 token 激活量化，
减少一次大尺寸 BF16 张量写回和读取。权重量化、激活 scale、GEMM 和输出舍入保持原样。

## 实现与范围

`int8_w8a8_native` 的 `ffn` / `blocks` 范围自动启用；`ffn_up` 和 FP8 路径不变。
仅匹配原生 `GELU(approximate='tanh')`，中间必须为 Identity 或概率为 0 的 Dropout。
其他激活和非零 Dropout 保持原路径，无需新增 CLI 参数。

转换时在执行 GPU 上生成涵盖全部 BF16 位模式的 GELU 查找表，保留当前 PyTorch CUDA
GELU 的舍入。每层注册独立的 128 KiB buffer，28 层共 3.5 MiB；可随权重卸载和搬运，
不共享 Tensor，避免逐层卸载时 buffer 名称去重及驻留状态相互影响。
报告增加 `fused_gelu_count`、`fused_gelu_call_count` 和 `gelu_lut_bytes_per_module`。

## 算子与模型筛选

L40S、PyTorch 2.6.0+cu126，block 0 真实权重、固定 seed 的合成 BF16 激活，
完整 FFN 包含两个 INT8 Linear、GELU、激活量化及 epilogue。
预热后 5 组 × 20 次，交替顺序，CUDA events；普通调用与 CUDA Graph 分开。

| token 数 | 原 FFN eager ms | 融合 FFN eager ms | 原 FFN Graph ms | 融合 FFN Graph ms |
| --- | ---: | ---: | ---: | ---: |
| 1,200 | 0.515 | 0.491 | 0.283 | 0.285 |
| 32,640 | 7.978 | 6.370 | 7.977 | 6.376 |

两列均使用最终 GEMM 分块，仅比较 GELU 融合。大 FFN 普通调用耗时减少 **20.15%**；小形状 Graph 没有收益。
Graph 只用于算子诊断，不意味着视频入口启用了 Graph。

真实完整 DiT、32,640 token、Sage FP8 per_thread Attention，5 组交替顺序：
原 INT8 FFN 中位数 1.9994 s，GELU 融合加新分块后 1.9483 s，减少 **2.55%**。
FFN 和完整 DiT 输出均与原 INT8 路径逐元素一致。这些是合成输入结果，不能替代视频测量。

同时筛选了 15 组降维 INT8 GEMM 分块，计入动态激活量化；最好的融合候选约 4.76 ms，
仍慢于原生路径的 3.95 ms，因此保留原生降维 GEMM。

升维层 BM=256 通过后续 7 组 × 30 次同卡交替复测：M=32640、K=2048、N=8192 的完整
Linear 从 3.201 ms 降至 2.923 ms，减少 **8.70%**。M=8192/8193/16320 未见稳定收益。
因此仅在 SM89 的精确形状 M=32640、K=2048、N=8192 使用 BM/BN/BK=256/128/64，
8 warps、3 stages；其他形状保留原分块。新旧分块和原生 INT8 epilogue 输出精确一致。

## 视频协议

GPU 2 串行运行原 INT8 FFN 和最终组合，各两次。沿用上一轮 121 帧、1920×1080 输入与 mask，
seed=42、CFG=3、50 步配置、strength=0.8（实际 40 去噪步）。Attention 为 Sage FP8、
per_warp、FP32+FP32；关闭缓存和手工融合，DiT 常驻、T5/VAE 卸载，首任务预热 2 步。
请求时间排除加载和预热，不跨轮引用历史时间计算本轮收益。

`full121/separate.log` 为原路径；复现包装器关闭 GELU 融合和新分块。
`full121/fused.log` 使用最终实现，记录了最终 GEMM 源码哈希。
一次未完成的候选启动不纳入统计，保留 `aborted_fused.log` 和原因记录于 `manifest.json`。

## 视频结果

| 配置 | 去噪 s | 纯推理 s | 请求 s |
| --- | ---: | ---: | ---: |
| 原 INT8，第一次 | 180.966 | 195.239 | 214.441 |
| 原 INT8，第二次 | 181.454 | 195.795 | 215.190 |
| GELU 融合＋新分块，第一次 | 177.154 | 191.462 | 214.893 |
| GELU 融合＋新分块，第二次 | 177.110 | 191.437 | 212.858 |
| 原 INT8 中位数 | 181.210 | 195.517 | 214.815 |
| 最终组合中位数 | 177.132 | 191.449 | 213.876 |

去噪减少 **2.25%**（4.078 s），纯推理减少 **2.08%**。
请求只减少 **0.44%**（0.940 s），小于本轮请求样本波动，尚不足以确认稳定的整请求提速。
请求中非纯推理部分的中位数从 19.298 s 增至 22.426 s；不将 FFN 的约 20% 收益视为视频加速。

峰值 allocated 从 23.5662 增至 23.5696 GiB，增加 3.5 MiB；reserved 从 43.2324 增至
43.2344 GiB。本轮减少了 FFN 激活读写，但没有降低整请求峰值显存。
两个候选均实际执行 56 个量化 Linear 和 28 个融合 GELU；累计 GELU 调用 2352/4592
（含首任务预热），Attention 无回退和失败。

四个视频及上轮 INT8 视频的 SHA256 均为：
`de3ebd16ba5f3a54501262bad9413c6799f2e259dadf5735904af9ce9d283224`。
本轮没有增加视频输出误差；相对原始 BF16 的既有量化误差仍见
[上一轮质量结果](quantization_optimization_20261003.md)。

保留两项精确输出优化，INT8 `ffn` / `blocks` 自动生效；默认量化模式不变。
请求阶段收益较小，后续应继续分析 FFN 之外的计算和输入输出耗时。

## 复现与验证

```sh
CUDA_VISIBLE_DEVICES=0 uv run --no-project python \
  -m entrypoints.cli.benchmark_quantized_ffn \
  --output results/quantized_ffn_measure.json

CUDA_VISIBLE_DEVICES=0 ERASERDIT_TEST_INT8=1 OMP_NUM_THREADS=1 \
  uv run --no-project python -m unittest tests.test_quantization -v
```

量化专项 12 项通过，覆盖所有有限 BF16 位模式、零输入、非连续张量、不整齐行数、
模型转换、融合跳过条件，以及 CPU 转换后的逐层卸载与局部编译精确输出。
默认回归 219 项检查通过，其中 67 项按环境条件跳过；GPU 跳过不代表该环境完成了验证。

原始结果与脚本：`results/quant_perf_20261003/`，包括 `ffn_final.json`、`forward_final.json`、
`tuning.json`、`expansion.json`、`tests_final.log` 和 `regression_final.log`。
视频命令、日志、逐任务耗时、显存、调用计数和哈希见 `full121/summary.json`。
