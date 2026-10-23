# Q6a / Q6b / Q6c

三档均继承原 Q6 和 S4 的 15 项 CPU/数据路径设置，保持原 Sage 主核
`per_thread / fp32+fp32`、BF16 Cross-attention、原 VAE、121/9 窗口、完整尾窗、
50 步 × strength 0.8、CFG 3，关闭文本投影/KV 缓存、残差缓存和 compile。
通用 CLI 默认行为不变；新扩展默认关闭。预设由 `scripts/q6_benchmark.py` 显式启用。
通用基准入口为 `scripts/benchmark_l40s.py`（也支持 `python -m scripts.benchmark_l40s`）；
原 `entrypoints.cli.benchmark_l40s` 保持原始实现；新增基准接口与 Q6 适配位于 `scripts`，两者不互相转发。

| 预设 | 累计新增内容 | 数值性质 |
| --- | --- | --- |
| q6 | 原 Q6 + S4，作为本轮基线 | 既有量化方案 |
| q6a | Cross-Q RMSNorm；Sage K−mean/INT8 量化融合 | 目标保持既有 Q6 输出 |
| q6b | q6a + 经审计的 DiT/Sage 输出分配与静态 FP8 GEMM 定点跳过填充 | 目标保持 q6a 输出 |
| q6c | q6b + 28 层 Self-attention QKV FP8；FFN 第一层 GEMM 输出端 GELU + FP8 量化 | 近似方案，质量单独判断 |

三档均不启用 AdaLN→FP8、Self-attention 输出投影 FP8 或额外 VAE 优化。
FFN 输出融合保留原 BF16 输入及静态 FP8 输入量化，因此不依赖 AdaLN→FP8。
Q6c 改变第一层 FFN GEMM/GELU 的舍入；不能称为精确融合。
不同优化的历史收益不能相加，组合数据以本轮完整请求为准。

## 未纳入的实验

[未纳入配置的实验记录](q6_experiments_not_adopted.md) 汇总 AdaLN→FP8、
Cross 残差/Norm 融合、输入 dtype、文字/KV 缓存、Sage 配置与主核分工、
纯 FFN GEMM 筛选、VAE Norm、Self-attention 输出投影 FP8 的结果及未采用原因。
文档区分完整视频、局部原型和只读分析，并列出原始证据目录；历史收益不代表当前组合的增量。

## 使用

从仓库根目录运行，使用已配置的 SageAttention `eb615cf6cf4d221338033340ee2de1c37fbdba4a`
及其已验证 SM89 构建。Sage 适配器核对源码、包装层和原生库指纹；不修改安装目录。
目前仅覆盖单进程、主线程、SM89 eager 推理。拒绝 NCCL/SP/TP/CFG 并行、compile、
FSDP 和逐层卸载；不同预设可各自在一张独立 GPU 上运行。

```bash
uv run --no-project python scripts/q6_benchmark.py \
  --profile q6b --devices 0 --cache off --repeats 1 \
  --sage-source /absolute/path/to/validated/SageAttention \
  --run-dir results/my_q6b
```

Q6c 额外需要 CUDA 12.6 头文件、g++ 和当前环境的 cuBLASLt。
仅提交 C++ 源码与构建脚本，不提交 `.so`；库路径和 SHA256 会进入运行清单。

```bash
uv run --no-project python scripts/build_ffn_epilogue.py \
  --cuda-home /absolute/path/to/cuda-12.6 \
  --output results/q6_native/ffn_epilogue.so
export MGERASE_Q6_FFN_LIBRARY="$PWD/results/q6_native/ffn_epilogue.so"
uv run --no-project python scripts/q6_benchmark.py \
  --profile q6c --devices 0 --cache off --repeats 1 \
  --sage-source /absolute/path/to/validated/SageAttention \
  --run-dir results/my_q6c
```

显式环境开关为 `MGERASE_Q6_SAGE_PERIPHERAL`、`MGERASE_Q6_TARGETED_FILL`、
`MGERASE_Q6_QKV_CALIBRATION`、`MGERASE_Q6_FFN_EPILOGUE`。
三个布尔开关默认 `0`，校准路径默认空；不通过预设时应自行完整设置原 Q6 与 S4。
Cross-Q 使用已有 `--operator-fusion-ops` 中的 `cross_q_rmsnorm_native`，不改变默认算子集合。
进程内配置首次读取后固定，切换预设应启动新进程。

## 实现与适用范围

- Cross-Q 保留原 FP32 均值与 BF16 舍入边界，支持自动回退，强制启用时拒绝不支持的输入。
- Sage 仅在连续 BF16 `[1,32640,32,64]` 上融合 K−mean/量化，保留原 Q 量化、V 准备及主核。
  其他形状走原 Sage 路径并计入回退；`smooth_v=False` 时不再分配从未读取的均值缓冲。
- 填充优化只改明确的分配/GEMM 调用点，不替换 `torch.empty` 或 Sage 函数。
  PyTorch 填充标志仍是进程全局状态，每次调用均恢复原值；只适用于独占串行推理 worker，
  不支持并发线程服务。确定性算法开关不改变。
- QKV 一次共享激活量化、三次 FP8 GEMM，权重独立缩放；注册缓冲参与原组件卸载，
  不保留 BF16 权重副本。Self/Cross 输出投影及 Cross-Q/K/V 不做 FP8 量化。
- 默认校准文件来自 `113000356`、seed 42 的 28 层统计，偶数步校准、奇数步留出、5% 余量。
  这是固定输入的校准证据，不是跨视频范围保证。更换模型、LoRA、计算路径或输入分布，
  需要另行校准/质量验证；本轮组合也必须单独验证。
- FFN 输出融合仅支持每次 32640 行、2048→8192→2048、静态比例 `.125`、零 dropout。
  使用一个共享 32 MiB workspace，卸载重载导致指针变化时重建计划；cuBLASLt 算法不符则报错。
  历史完整实验显存增量约 1.54 GiB，不能仅用 workspace 大小估计峰值。

运行清单冻结 Python 源码、校准 JSON 和 C++ 源码，记录完整参数、S4、生效开关、
依赖指纹、输出哈希、阶段耗时和进程树 NVML 峰值。量化报告包含扩展调用数及填充恢复状态。
一次分卡测试只作整合验证，不据此判定亚百分比收益或宣布质量验收通过。

## 测试

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --no-project python -m unittest \
  tests.test_architecture tests.test_quantization tests.test_sage_fp8_options \
  tests.test_operator_fusion_precision tests.test_native_rms_fusion tests.test_q6_extensions -v

CUDA_VISIBLE_DEVICES=0 ERASERDIT_TEST_Q6=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH=.:/absolute/path/to/validated/SageAttention \
  uv run --no-project python -m unittest tests.test_q6_extensions -v
```

GPU 检查覆盖 Cross-Q 原生对照、Sage 填充/零值/NaN 毒化对照、QKV 显式 FP8 对照、
FFN BF16 入口对照，以及卸载重载后的输出和计划重建；需要先构建并设置 FFN 库路径。
跳过的 CUDA 用例不代表通过。

## 2026-10-23 最终串行结果

最终性能表统一使用 GPU0 串行完整请求；每档一次成功样本，不与分卡并行结果混用。
Q6c 沿用先前已完成的同卡串行样本，Q6a/Q6b 本轮依次重测。

| 配置 | GPU | 请求 / s | DiT / s | NVML 峰值 / GiB |
| --- | ---: | ---: | ---: | ---: |
| Q6a | 0 | 255.923 | 218.840 | 21.189 |
| Q6b | 0 | 247.145 | 209.053 | 21.189 |
| Q6c | 0 | 235.792 | 197.907 | 22.400 |

Q6a/Q6b 与原 Q6 编码输出逐字节一致，Q6c 与已验证的 Q6c 编码输出逐字节一致。
Q6c 相对 Q6b 的 PSNR 为 43.487 dB，半分辨率亮度 SSIM
为 0.990313，时序差分 MAE 为 0.882898。
质量指标复用相同输出哈希的逐帧分析；无真值或验收阈值，Q6c 保持显式启用。
一次采样不证明统计稳定性。最终报告与原始数据位于
`results/q6_crossq_rmsnorm_20261021_01/integration_20261023_01/`。
