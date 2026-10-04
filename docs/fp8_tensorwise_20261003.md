# L40S 动态 FP8 执行路径修正（2026-10-03）

旧 `fp8_w8a8_native` 的 per-token/per-channel scale 无法在当前 PyTorch 2.6 的 SM89
原生 GEMM epilogue 中直接使用，因此先输出 FP32，再执行缩放和 bias。
对 M=32640、N=8192 的升维层，FP32 中间结果写入再读取约 2.14 GB，抵消了 GEMM 收益。

新增实验模式：

```sh
--transformer-quantization fp8_w8a8_tensorwise --quantization-scope ffn
```

权重按整张量量化一次；激活每次通过 GPU 分块 absmax、最终归约、转换三个 kernel
动态量化，不同步回 CPU。cuBLAS GEMM 接收两个 scalar scale，完成缩放和 bias，直接输出 BF16。
E4M3 转换保留直接 FP32→FP8 的舍入，关闭 fast accumulation。没有保留 BF16 权重或回退 GEMM。
旧模式保留，默认仍不量化；整张量缩放属于不同数值方案，需要独立质量验证。

## 完整 Linear 测量

L40S，PyTorch 2.6.0+cu126，模型 block 0 真实权重、合成正态 BF16 激活。
包含动态量化、GEMM、缩放和 bias；预热后 5 组、每组 20 次，交替顺序，CUDA events。
下表为 eager 中位数，单位 ms；输入 M=32640，K/N 分别为输入/输出宽度。

| K→N | BF16 | INT8 | 旧 FP8 | 新 tensorwise FP8 |
|---|---:|---:|---:|---:|
| 2048→2048 | 1.192 | 1.588 | 1.759 | 1.222 |
| 2048→8192 | 5.665 | 3.548 | 6.263 | 3.618 |
| 8192→2048 | 5.263 | 4.186 | 4.501 | 4.594 |

新 FP8 升维层相对 BF16 为 1.57×，但不是所有层都获益。
M=1200 时，新 FP8 升维层 eager 0.299 ms，BF16 0.212 ms；CUDA Graph 下为
0.115 vs 0.209 ms，说明小层启动开销显著。Graph 仅用于算子诊断，视频入口未因此启用 Graph。
这些误差与耗时不是生成质量或端到端加速的替代品。

## 视频对照

固定 121 帧原始 1920×1080 视频与 mask、seed=42、50 步配置、strength=0.8
（实际 40 个去噪步）、CFG=3，SDPA、无缓存、关闭手工融合，FFN 范围 56 层。
GPU 2 串行运行新 FP8 和重新测量的未量化 BF16，各一次，预热 2 步；不包含加载和预热时间。
单次结果不代表稳定加速比。FP8 报告 56 层均实际执行，4704 次调用包含预热，无 GEMM 回退。

新 FP8 去噪 214.659 s、纯推理 229.403 s、请求总耗时 252.621 s，峰值 allocated 23.565 GiB。
旧 FP8 历史单次去噪 224.645 s，说明修正减少了退化；历史 BF16 为 214.881 s。
同卡重测 BF16 去噪 215.072 s、纯推理 229.590 s、请求总耗时 252.943 s，
峰值 allocated 24.441 GiB。新 FP8 去噪仅快 0.19%，请求仅快 0.13%，
单次测试下应视为持平，不构成有效加速。allocated 减少约 0.875 GiB，但 reserved 未减少。

重测 BF16 与历史 BF16 视频 SHA256 完全一致（`f6baceed78a900464db10037ac5c6dd4facbee080cb5c8a30ce43567be91c1ec`）。
与 BF16 输出比较：平均 SSIM 0.989291、最低帧 0.985653；103 个非空 mask 帧的
平均区域 SSIM 0.989364、最低区域帧 0.983427；MAE 0.665086（0–255），
时序差分 MAE 0.820128。已查看首、中、末帧的并排图，未见明显整体失真。
这只验证当前片段的接近程度，不等同于所有素材的质量保证。

当前结论：算子修正有效，但尚无足够端到端收益支持默认启用量化。
要显著加速，应按整个 DiT forward 的热点占比继续优化；FFN Linear 之外的 Attention、
归一化、激活及访存仍未被这些量化模式加速。

## 复现与验证

- 算子完整记录：`results/fp8_fused_20261003/real_weights.json`，包含 eager/Graph 及误差。
- 视频命令、任务、日志：`results/fp8_fused_20261003/full121/`。
- GPU 数值及模型转换：`CUDA_VISIBLE_DEVICES=0 ERASERDIT_TEST_INT8=1 uv run --no-project python -m unittest tests.test_quantization`。
- 算子入口：`uv run --no-project python -m entrypoints.cli.benchmark_quantization --output results/fp8_linear.json`。

验证：8 项 GPU 量化测试通过；全套 211 项测试通过（跳过 40 项）；CLI 算子入口 smoke 通过。
