# L40S Attention 量化验证（2026-10-03）

## 配置与范围

本轮只替换 Self-Attention 的内部 QKᵀ/PV 计算，所有 Linear 保持 BF16，Cross-Attention 保持 SDPA。
使用现有 `sage_fp8` 后端：QK INT8 per-thread、PV FP8、`fp32+fp32` 累加、smooth K、无 smooth V。
显式后端不可用时直接报错；不把自动回退当作量化结果。

固定 SageAttention v2.2.0，commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`。
在实验目录内使用 NVIDIA CUDA 12.6.3 编译器、当前 PyTorch CUDA 开发头文件、Ninja 编译 SM89；
不覆盖项目环境的依赖。扩展源文件未修改。
Torch 2.6.0+cu126，L40S，视频实验单卡 GPU 2 串行执行，GPU 7 未使用。

## 完整 Attention 算子

合成 BF16 Q/K/V，batch=1、32 heads、head_dim=64；与模型视频形状一致。
调用项目后端 wrapper，计入动态量化、K 平滑和布局处理。
预热后交替测量 5 组，每组 20 次，CUDA events，eager；没有使用 CUDA Graph。

| tokens | SDPA 中位数 ms | Sage FP8 中位数 ms | SDPA/Sage |
|---|---:|---:|---:|
| 1200 | 0.08745 | 0.42010 | 0.21× |
| 32640 | 41.97268 | 25.31891 | 1.66× |

长序列相对 RMSE 0.03868（合成输入），输出有限；这不是视频质量指标。
小序列受预处理和启动开销影响，量化反而更慢，不能将长序列结果泛化到所有窗口。

## 视频验证协议

输入为现有视频前 121 帧、原始 1920×1080 分辨率和对应 mask；seed=42、CFG=3、
50 步配置、strength=0.8（实际去噪 40 步）、infer_len=121、overlap=9。
关闭 transformer cache、text projection cache、手工融合和 DiT 卸载；保留 T5/VAE 卸载与 VAE low-memory。
预热 2 步，耗时不含模型加载和预热。
SDPA 重测一次，Sage 两次（同进程两个任务，仅首任务预热）；与此前同配置 BF16 结果交叉核对。
新 BF16 视频和前一轮视频 SHA256 完全一致。

质量使用逐帧 RGB SSIM（11×11 Gaussian，sigma=1.5）、MAE、mask/edge 和时序差分 MAE；
空 mask 帧不进入 mask 统计。性能和质量数据只代表这一段素材。

## 视频性能结果

| 模式 | 去噪 s | 纯推理 s | 请求总耗时 s |
|---|---:|---:|---:|
| BF16 SDPA | 214.684 | 229.165 | 251.397 |
| Sage FP8 第一次 | 183.725 | 198.067 | 220.485 |
| Sage FP8 第二次 | 184.345 | 198.745 | 221.624 |
| Sage FP8 中位数 | 184.035 | 198.406 | 221.055 |

相对本轮基线，去噪为 **1.167×**（耗时减少 **14.28%**），请求为 **1.137×**
（耗时减少 **12.07%**，约省 **30.34 秒**）。请求包含视频处理与输出，排除模型加载和预热。
两次 Sage 结果接近；基线另有前两轮同配置去噪 214.881/215.072 s 作为一致性核对。
这不是 2× 整体加速，但已获得可重复的端到端收益。

峰值 allocated 均为 24.441 GiB、reserved 均为 43.105 GiB；本次没有降低请求峰值显存。
日志确认 `effective=sage_fp8`、`fallback_count=0`、`failure_count=0`。
首 block 的累计调用计数分别为 84/164（包含首任务 4 次 warmup forward；每个正式请求 80 次 CFG forward）。

两次 Sage 输出 SHA256 完全一致：`5fcafe3e83d4aae6a72fd0a3d9238ee7234c65aeec31cd5ac72e5c378c355ecf`。
因此下列第一遍逐帧质量统计同样适用于第二遍。

## 视频质量

与同配置 BF16 SDPA 比较：

| 指标 | Sage FP8 第一次 |
|---|---:|
| 平均 SSIM | 0.991730 |
| 最低整帧 SSIM | 0.988131 |
| 平均 mask SSIM | 0.992219 |
| 最低 mask 帧 SSIM | 0.986839 |
| 平均 edge SSIM | 0.992580 |
| MAE（0–255） | 0.449348 |
| 时序差分 MAE | 0.652464 |

121 帧全部参与整帧统计，103 帧存在非空 mask。查看首、中、末帧对比，以及
mask SSIM 最低的第 100 帧区域裁剪，未见明显新增失真或残影。
SSIM 是与参考输出的接近程度，不能替代多素材的生成质量验证。

## 使用和复现

当前扩展已构建在实验目录。保留原 CLI 参数，仅切换后端并关闭 Linear 量化：

```sh
PYTHONPATH="$PWD/results/attention_quant_20261003/SageAttention" \
  uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model --task-file YOUR_TASKS.json \
  --attention-backend sage_fp8 --transformer-quantization none
```

该片段说明后端启用方式；完整测试的卸载、缓存、采样等参数以保存的 `*_command.json` 为准。
不能仅安装 requirements 中的旧 SageAttention 1.0.6 就假定具备本后端。

算子复现：

```sh
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD/results/attention_quant_20261003/SageAttention" \
  uv run --no-project python -m entrypoints.cli.benchmark_attention \
  --output results/attention_check.json
```

完整视频复现脚本：`results/attention_quant_20261003/run_video.py`，输出目录内包含任务和完整命令。
再次运行前应改输出目录以保留已有结果。

记录位置：`results/attention_quant_20261003/` 的 `manifest.json`、`attention_benchmark.json`、
`full121/attention_summary.json`、`full121/quality.json`，以及构建日志和扩展依赖指纹。

## 当时的切片结论与后续修订

本轮 121 帧切片曾支持优先尝试 `sage_fp8`、Linear 保持 BF16。
后续完整素材与原生融合对照未通过质量门槛，该建议不再作为通用推荐；
默认保持 BF16/SDPA，见 [最新验收记录](l40s_validation_20261004.md)。
不要将其收益泛化到短序列：1200 token 算子实测反而变慢。
本轮未叠加 Linear 量化或缓存，12.07% 的请求耗时下降来自 Attention 后端切换。
生产代码已有该后端，本轮完成依赖构建与真实视频验证，并新增可复用的 Attention benchmark 入口。
