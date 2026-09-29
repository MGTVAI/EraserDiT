# 单卡优化：整段 RGB SSIM ≥ 0.985（2026-09-27）

用户将本轮整段 RGB SSIM 门槛从 0.99 调整到 **0.985**。分辨率、输出帧数、
帧率和采样步数保持不变。仍报告 MSE、MAE、最差帧和时序误差；整段平均通过
不表示每帧达到 0.985。组合优化与原有未减填充 SDPA 参考输出比较，避免逐项误差累积被掩盖。

## 已有结果重新判定

以下数值来自此前实验，不是本轮重新测速。原始 0.99 报告保留原判定，见
[此前实验](quality99_optimization_20260927.md)。

| 候选 | 整段 SSIM | 达到新门槛 |
|---|---:|---|
| 原片尾窗口减填充 | 0.992833 | 是 |
| 第二段素材，双方均使用 89 帧窗口，尾窗口减填充 | 0.987735 | 是 |
| 原片 TeaCache 0.01 | 0.983136 | 否 |
| 原片 TeaCache 0.02 | 0.983144 | 否 |
| 原片 TeaCache 0.05 | 0.982354 | 否 |
| 原片 TeaCache 0.05 + 线性预测 | 0.982787 | 否 |

第二段素材默认 121 帧窗口只有首窗口，不触发减填充。89 帧配置的结果不能当作默认配置收益。
尾窗口开关继续默认关闭，其他素材和近似计算组合需要各自验证。

## 新版 SageAttention 隔离验证

环境原有 SageAttention 1.0.6 保留；[官方 v2.2.0 源码](https://github.com/thu-ml/SageAttention/tree/v2.2.0) 在忽略的实验目录内编译，
仅候选进程通过 `PYTHONPATH` 使用，不覆盖已安装包。源码提交：
`eb615cf6cf4d221338033340ee2de1c37fbdba4a`。

A100 的公共 dispatcher 路由为 `sageattn_qk_int8_pv_fp16_cuda`，QK INT8、PV FP16、
FP32 累加；不是 FP8 路径。构建指定 CUDA 12.6、`TORCH_CUDA_ARCH_LIST=8.0`。
不引入 `sglang` 包。

实验脚本、构建日志与候选视频存于 `results/quality985_20260927/`。
### 注意力微基准

物理 GPU 3，BF16，`B=1,H=32,D=64`，NHD 输入；每条路径预热 3 次，
五组交替顺序，每组 3 次，以 CUDA Event 计时，包含公共调用的量化、平滑与 V 转换。
随机输入仅用于测速与算子误差观测，不能代替视频 SSIM。验证调用后 Q/K/V 未被修改。

| tokens | SDPA ms | Sage 2 默认 FP32 累加 ms | 变化 |
|---|---:|---:|---:|
| 32640 | 47.012 | 45.563 | 减少 3.1% |
| 10200 | 4.578 | 4.868 | 增加 6.3% |

另一次五组筛选：长窗口 SDPA 47.277 ms、Sage FP16 累加 46.133 ms、
混合 FP16+FP32 累加 46.754 ms；短窗口分别为 4.617、4.916、5.029 ms。
默认路径相对 SDPA 的随机输入输出 L2 相对误差约 0.0113；FP16 累加长窗口约 0.0147。
降低累加精度未显示更好的收益，不进入完整视频测试。

完整视频使用默认 dispatcher；数值误差以以下视频对照评估。

### 同卡完整视频对照

物理 GPU 2，原片 145 帧，参数与此前正式尾窗口试验一致。加载单列，未显式请求预热。
本轮 SDPA + 减填充复跑：请求 **240.915 s**、去噪 **201.709 s**，
峰值 allocated **30.653 GiB**、reserved **48.387 GiB**。
完整 RGB SHA256 与此前 240.978 s 的正式 CLI 输出相同：
`d04a28190a3a57c394d9abf375b0e488dcbf82543dcecec2c6d1add70532165b`。

| 同卡顺序运行 | 请求 s | 去噪 s | allocated GiB | reserved GiB |
|---|---:|---:|---:|---:|
| SDPA + 尾窗口减填充 | 240.915 | 201.709 | 30.653 | 48.387 |
| Sage 2 默认 + 尾窗口减填充 | 235.614 | 197.404 | 30.653 | 48.387 |

候选请求减少 **2.2%**、去噪减少 **2.1%**，显存峰值一致。每种完整视频只运行一次，
属于筛选结果，不是五组稳态统计。两次加载约 44 s，不计入表中请求时间。
日志确认 self-attention 实际使用 `sage_attn`，无回退；cross-attention 仍为 SDPA。

相对 `results/performance_20260927/fused50.mp4`（未减填充 SDPA 参考）：

- 整段 RGB SSIM **0.982647 < 0.985**，组合候选不合格。
- MSE **2.496150**，MAE **1.030777**；最低帧 SSIM **0.960997**。
- 相邻帧误差变化 MAE **1.041296**；完整解码 145 帧，尺寸、帧率一致。

这是 Sage 2 与减填充的组合结果，不能据此断言单独 Sage 2 的视频 SSIM。
由于组合已低于数值门槛，无需通过人工画面验收来改变结论；不纳入推荐配置。
本轮未改动运行后端默认值或安装环境。继续使用 SDPA + 已验收的尾窗口选项。

## 后续顺序

先针对当前逐层卸载路径评估 FFN/逐点算子融合和局部编译，分别记录编译成本、
稳态推理和显存峰值；不直接移除 compile + offload 的兼容性保护。
旧的激进编译记录 SSIM 约 0.9838，仍不满足 0.985，不能自动转为合格方案。
更激进的缓存或量化只有在组合输出达到新门槛时才继续扩大测试。

## 复现关键命令

从实验目录的 SageAttention 源码构建扩展（使用项目环境的 Python）：

```bash
CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 EXT_PARALLEL=1 \
  python setup.py build_ext --inplace
```

候选沿用 [此前完整 CLI 命令](quality99_optimization_20260927.md#使用)，
仅把 `--attention-backend sdpa` 改为 `sage_attn`，并为该进程设置
`PYTHONPATH=/绝对项目路径/results/quality985_20260927/SageAttention`。
输出路径必须另设，保留参考文件；`--compact-tail-padding` 保持开启。

```bash
python results/quality985_20260927/quality.py \
  --reference results/performance_20260927/fused50.mp4 \
  --candidate results/quality985_20260927/sage2_tail.mp4 \
  --report results/quality985_20260927/sage2_tail_quality.json --lossy
```

实验脚本 `--lossy` 的进程退出码不表示达标，判定读取报告中的
`threshold_pass`；本候选为 `false`，汇总 `summary.json` 中 `qualified=false`。
