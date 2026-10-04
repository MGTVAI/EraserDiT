# L40S FP16 与 FP8 单独对照（2026-10-03）

本轮使用真正的 FP16 输入和输出，不以 BF16 代替 FP16。仅测算子，不运行视频流水线。

## 条件

- 单张空闲 L40S（GPU 0），PyTorch 2.6.0+cu126；固定 seed 42。
- 使用 DiT 第 0 层真实权重，转为 FP16；激活为 FP16 正态随机数。
- 五组、每组二十次，预热后交替正反测试顺序，报告 CUDA event 中位数。
- FP16 使用 PyTorch 默认 `allow_fp16_reduced_precision_reduction=True`；FP8 E4M3，`use_fast_accum=False`。
- GEMM：FP8 权重、激活事先量化，标量 scale 在 GEMM 内应用，两者无 bias、输出均为 FP16。
- 完整 Linear：FP8 权重按通道预量化，激活逐 token 动态量化，FP32 中间输出后缩放加 bias，最终 FP16；FP16 对照为 `F.linear`。
- 两种 FP8 测试的缩放粒度不同，因此不能将它们的耗时差解释为量化阶段的独立耗时。
- CUDA Graph 减少主机调度空隙；普通调用保留调度和分配开销。两者均不代表完整视频速度。

## 纯 GEMM（CUDA Graph）

| M×K×N | FP16 ms | FP8 ms | FP16/FP8 加速比 |
| --- | ---: | ---: | ---: |
| 1200×2048×2048 | 0.072 | 0.038 | 1.91× |
| 32640×2048×2048 | 1.437 | 0.790 | 1.82× |
| 1200×2048×8192 | 0.240 | 0.122 | 1.97× |
| 32640×2048×8192 | 6.244 | 3.443 | 1.81× |
| 1200×8192×2048 | 0.184 | 0.159 | 1.16× |
| 32640×8192×2048 | 6.166 | 3.184 | 1.94× |

## 完整 Linear

| M×K×N | FP16 Graph ms | FP8 Graph ms | FP16 普通调用 ms | FP8 普通调用 ms |
| --- | ---: | ---: | ---: | ---: |
| 1200×2048×2048 | 0.063 | 0.047 | 0.062 | 0.230 |
| 32640×2048×2048 | 1.428 | 1.794 | 1.479 | 1.805 |
| 1200×2048×8192 | 0.235 | 0.164 | 0.236 | 0.234 |
| 32640×2048×8192 | 5.980 | 6.229 | 6.210 | 6.260 |
| 1200×8192×2048 | 0.194 | 0.178 | 0.196 | 0.231 |
| 32640×8192×2048 | 6.139 | 4.688 | 6.220 | 4.756 |

FP8 纯 GEMM 有明确加速收益；当前完整量化路径仍有额外开销，加速取决于形状。
本轮 FP8 相对 FP16 的输出 relative RMSE 约 3.6%–3.8%，所有输出有限。
这是随机激活上的算子误差，不是视频质量损失或准确率；没有启用 fast accumulation 换速度。
另测关闭 FP16 reduced-precision reduction 的保守配置，保留在 `measure.json`，主要趋势一致。
原始文件含每组数据及 min/max，较大 FP16 GEMM 存在一定波动，不应把中位数当作固定硬件峰值。

## 复现

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-project python \
  -m entrypoints.cli.benchmark_fp16_fp8 \
  --output results/fp16_fp8_new/measure.json
```

附加 `--no-fp16-reduced-precision-reduction` 可测保守的 FP16 累加配置。
原始数据：`results/fp16_fp8_20261003/default_fp16.json`；原始保守配置：同目录 `measure.json`。
本轮只增加独立 benchmark，未改变模型量化或推理 dtype。

INT8 已加入同一 benchmark，最新三方同轮结果见 [FP16 / FP8 / INT8 对照](fp16_fp8_int8_benchmark_20261003.md)。
