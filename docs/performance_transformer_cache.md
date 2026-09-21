# P2：EraserDiT Transformer 缓存

最新验收规则见 [优化验收标准](optimization_acceptance.md)：非 cache/量化优化采用 SSIM ≥ 0.985、MSE ≤ 36、MAE ≤ 6；cache/量化按视觉大致一致验收。下文历史字节一致结果保留，不再作为必须条件。

最新完整视频、指定参考输出的正确性复测见 [cache_correctness.md](cache_correctness.md)。

## 接入与默认行为

`transformer_cache_mode=off | teacache | cache_dit`，默认 `off`，同一请求只能选择一种。
CLI、task-file、HTTP 本地文件请求和 multipart parameters 使用相同请求字段；task-file 显式字段覆盖 CLI。
当前两种模式均为实验配置，不能与整个 Transformer 的 `torch.compile` 组合。
可以与 P1 的 `dynamic_offload` 组合，缓存路径仍通过 block 卸载包装器执行。

TeaCache 比较第一层 `norm1(hidden_states) * (1 + scale_msa) + shift_msa`，在投影后缓存整个 block 栈的残差。
探针通过第一层 forward 的卸载包装器计算，确保动态卸载时权重在正确设备上。
**未使用 LTX095 的拟合系数**：策略 `eraserdit_modulated_input_relative_l1_experimental` 为恒等多项式，
直接累积时间调制后视频特征的相对 L1 距离。这是未校准的实验策略；`calibration_size=0`、
`coefficient_calibrated=false`，阈值不能照搬其他模型。

cache-dit 使用项目已有 DBCache 控制器：每步计算前 F 层，比较前段残差与最近一次完整计算时的前段残差，
满足阈值时复用中段残差，随后执行后 B 层。没有新增第三方 cache-dit 包依赖。

两路 CFG 独立保存状态。每窗口创建控制器，正常退出或异常都释放全部缓存张量；不向模型挂载持久缓存。
布局签名包含帧数、高宽、RoPE scale、tensor shape/dtype/device，不能只按 token 总数复用。
前四个有效去噪步和最后一步完整计算；默认最多连续复用一次。这里的步数是 strength 截取后的实际步数。

## 请求参数

| 字段 | 默认 | 含义 |
|---|---:|---|
| `transformer_cache_mode` | `off` | 缓存模式 |
| `transformer_cache_force_compute` | `false` | 保持缓存接入但禁止复用，检查等价性 |
| `teacache_threshold` | `0.005` | 第一层调制输入距离的累计阈值，未校准 |
| `max_teacache_consecutive_skip` | `1` | TeaCache 连续跳步上限 |
| `teacache_warmup_steps` | `4` | TeaCache 前段完整计算步数 |
| `cache_dit_front_blocks` | `1` | 始终执行的前段层数 |
| `cache_dit_back_blocks` | `0` | 始终执行的后段层数 |
| `cache_dit_warmup_steps` | `4` | cache-dit 前段完整计算步数 |
| `cache_dit_residual_diff_threshold` | `0.03` | 前段残差相对 L1 阈值 |
| `cache_dit_max_consecutive_cached_steps` | `1` | 中段连续复用上限 |
| `cache_end_guard_steps` | `1` | 最后完整计算步数，至少 1 |

上述阈值只作为默认 50 调度步的小尺寸实验起点；短步数、其他尺寸仍需单独测量。

CLI 参数将下划线替换为连字符，例如 `--transformer-cache-mode cache_dit`。
TeaCache 的强制全计算内部复用通用控制器的 `calibrate` 标志，但不会训练或拟合系数。
cache-dit 的强制全计算将实际 warmup 扩大至窗口总步数，报告中的 resolved_params 会显示实际值。

HTTP JSON 示例（与已有 video_path/mask_path 等字段一起提交）：

```json
{
  "transformer_cache_mode": "cache_dit",
  "cache_dit_residual_diff_threshold": 0.03,
  "cache_dit_front_blocks": 1,
  "cache_dit_back_blocks": 0,
  "cache_dit_max_consecutive_cached_steps": 1
}
```

CLI 结果和 HTTP `task.metrics.transformer_cache_history` 按窗口记录实际完整计算/复用步数、
CFG 分支统计、布局/系数身份、强制计算原因、生命周期状态。
`peak_retained_tensor_bytes` 是单窗口前向间保留的缓存 tensor storage 峰值，不包含计算中的临时激活，
也不等于 CUDA allocator reserved。每请求 allocator 峰值独立重置。
server_info 的缓存项是可用能力与默认值，实际生效情况以任务报告为准。

## 复现

只允许使用物理 GPU 2、3。下例使用 GPU 2；不要在空闲显存不足时执行原尺寸输入。

```bash
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
PY=/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python
MODEL=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model

$PY entrypoints/cli/erase_eraserdit.py --model-path "$MODEL" \
  --video-input results/dynamic_offload_smoke/video_33.mp4 \
  --mask-input results/dynamic_offload_smoke/mask_33.mp4 \
  --output-path results/transformer_cache_smoke/example.mp4 \
  --resource-policy dynamic_offload --max-weight-usage 2147483648 \
  --attention-backend sdpa --num-inference-steps 50 --infer-len 17 --overlap 9 \
  --transformer-cache-mode cache_dit

$PY scripts/cache_compare.py --directory results/transformer_cache_smoke \
  --mask results/dynamic_offload_smoke/mask_33.mp4 \
  --names off tea_force dbc_force tea_003 dbc_010 tea_010 dbc_020 tea_0005 tea_001 dbc_003 dbc_005 tea_repeat dbc_repeat
```

连续请求/故障恢复使用 `scripts/offload_verify.py`：同样的 CLI 参数，添加 `--repeat 2 --inject-failure`，
`--output-path` 指向 JSON。故障注入在首个完整 CFG 步后触发，检查缓存张量释放、CPU 空闲权重和下一请求恢复。

## 验证范围

单元测试覆盖关闭模式、非法参数、compile 冲突、CLI 覆盖、CFG/窗口隔离、末步保护、连续跳步上限、
同 token 数不同几何布局失效、后段层始终计算、错误 pending step、异常清理，
以及真实小型 Transformer 的强制全计算逐元素等价。

本轮实测数据见下文。正式推荐仍需两组原尺寸素材完整步数、至少五次重复，以及 mask 内、边缘、
非擦除区和跨窗口时序的目视验收。小尺寸通过不等于原尺寸质量验收通过。

## 本轮小尺寸结果（2026-09-20）

物理 GPU 2；192×320、33 帧、3 窗口，infer_len=17/overlap=9；20 调度步、strength=0.8，
每窗 16 个有效步，两路 CFG 共 96 次前向。SDPA、bf16、seed=42、dynamic_offload 2 GiB，
无 compile/融合/预热。指标取相同参数无缓存输出为参照：整帧使用 ffmpeg Y-plane SSIM/PSNR，
mask 内/外及 9×9 形态学边缘环另保存 Gaussian SSIM 和 PSNR（不能混作同一种 SSIM）。

| 配置 | 复用次数 / 前向次数 | 整帧 SSIM-Y | PSNR-Y dB | 单次端到端秒 |
|---|---:|---:|---:|---:|
| off | 0/96 | 1.0000 | ∞ | 57.98 |
| tea_force | 0/96 | 1.0000 | ∞ | 24.19 |
| dbc_force | 0/96 | 1.0000 | ∞ | 57.64 |
| tea_0005 | 0/96 | 1.0000 | ∞ | 36.81 |
| tea_001 | 12/96 | 0.8924 | 23.53 | 30.57 |
| tea_003 | 24/96 | 0.8387 | 21.27 | 19.80 |
| tea_010 | 36/96 | 0.8279 | 20.90 | 23.78 |
| dbc_003 | 6/96 | 0.9014 | 23.65 | 23.01 |
| dbc_005 | 12/96 | 0.9109 | 24.67 | 18.55 |
| dbc_010 | 16/96 | 0.9189 | 25.39 | 47.81 |
| dbc_020 | 32/96 | 0.8481 | 21.56 | 22.74 |

两种 force_compute 视频与 off 的 SHA256 均为
`135eb26c9378fd85edb70bb6d08aefd7d21574192e95eacf569764178074ee21`。
TeaCache 0.03 和 cache-dit 0.10 在不同模式任务交错后重复请求，分别与自身前次输出逐字节一致。

本组有命中的配置**均未通过** SSIM ≥0.95 / PSNR ≥28 dB；TeaCache 0.005 无命中，输出一致，
不构成加速验证。阈值与最终视频误差不是单调关系，不能把更低阈值视为质量保证。
当前 active 模式参数只是实验起点，默认 `off`；不推荐将以上有损档位投入使用。

共享 GPU 占用变化很大：同配置 TeaCache 0.03 两次耗时 19.80 / 40.54 秒，cache-dit 0.10 为
47.81 / 32.81 秒。表中单次耗时不能解释为加速比；未满足独占/稳定负载、五次重复的性能门禁。
请求 allocated 峰值均约 1.850 GiB。保守阈值运行测得单窗口缓存保留 tensor 峰值：
TeaCache 1,523,712 bytes，cache-dit 2,949,120 bytes；不包含临时激活。

本地结果：`results/transformer_cache_smoke/{runs.json,conservative_runs.json,quality.json}`；
相应 MP4、任务列表和完整日志在同目录。质量 JSON 同时记录 mask 内、非擦除区和边缘误差。

功能回归：原有卸载测试 22 项通过，新增缓存测试 10 项通过（包括本地 JSON 和 multipart 契约往返）。
两个模式均通过真实模型的三窗口连续两次请求、首个完整 CFG 步后注入异常、恢复后输出一致检查。
故障时缓存报告 `closed=true, aborted=true`，全部缓存 tensor 引用清除，权重回到 CPU，
动态预算状态/事件队列恢复为零。报告位于 `tea_recovery.json`、`dbc_recovery.json`。
这组故障用例使用 4 调度步（3 个有效步），只验证生命周期；缓存命中由上述 20 步矩阵另验。

### 默认 50 调度步复核

同一小尺寸三窗口素材，50 调度步 / strength=0.8，每窗实际 40 步，两路 CFG 总共 240 次前向。
以下结果只与本组独立的 50 步 off 基准比较；其余设置同上。

| 配置 | 复用次数 / 前向次数 | 整帧 SSIM-Y | PSNR-Y dB | 单次端到端秒 |
|---|---:|---:|---:|---:|
| off | 0/240 | 1.0000 | ∞ | 66.40 |
| tea_0005 | 36/240 | 0.9540 | 28.22 | 66.30 |
| dbc_003 | 37/240 | 0.9592 | 28.95 | 57.06 |

结果与视频：`results/transformer_cache_smoke/normal_steps/`。

本组两种低阈值候选通过了整帧数值门槛：TeaCache 0.005 为 SSIM-Y 0.9540 / PSNR-Y 28.22 dB，
cache-dit 0.03 为 0.9592 / 28.95 dB。显式启用时的实验默认阈值据此收紧为 0.005 / 0.03，
全局模式仍为 off。20 步历史表的 0.03 / 0.10 是初始扫描值，不是最终 active-mode 默认值。

这不等于最终质量通过：擦除区 Gaussian SSIM 约 0.922 / 0.924，边缘约 0.928 / 0.926，
尚未进行原尺寸残留、损坏和跨窗时序目视验收。TeaCache 仍未拟合校准系数。
本组单次耗时为 66.40 → 66.30 / 57.06 秒；共享负载与不足五次重复使其不能作为正式速度结论。

P2 工程接入和小尺寸功能验证完成；正式验收待两组原尺寸完整步数、分区域/时序质量检查、
每配置至少五次稳定负载测量。接下来优先对上述候选做原尺寸验证和 TeaCache 系数校准，
再推进 P3 量化；不将尚未验收的缓存默认启用。

### 增大阈值复测（2026-09-20）

按用户要求，保持上组 192×320 / 33 帧 / 三窗口、50 调度步（实际 40 步）、SDPA、bf16、
seed=42、dynamic_offload 2 GiB。仅改变请求阈值，保持前四步/末一步完整计算、最多连续复用一次；
TeaCache 测 0.01/0.03/0.10，cache-dit 测 0.05/0.10/0.20。
全部在物理 GPU 2 串行执行，GPU 3 未启动任务。前后各执行一次 off，共 8 次请求。

| 配置 | 复用率 | 端到端秒 | 去噪秒 | 相对首个 off 耗时减少 | 整帧 SSIM-Y | PSNR-Y dB |
|---|---:|---:|---:|---:|---:|---:|
| off | 0.0% | 67.13 | 60.77 | +0.0% | 1.0000 | ∞ |
| off_repeat | 0.0% | 86.56 | 81.06 | -28.9% | 1.0000 | ∞ |
| tea_001 | 20.0% | 58.11 | 53.52 | +13.4% | 0.9266 | 25.76 |
| tea_003 | 45.0% | 67.40 | 60.80 | -0.4% | 0.9015 | 24.28 |
| tea_010 | 45.0% | 40.71 | 36.20 | +39.4% | 0.9015 | 24.28 |
| dbc_005 | 23.3% | 58.90 | 54.42 | +12.3% | 0.9156 | 25.03 |
| dbc_010 | 45.0% | 57.28 | 52.80 | +14.7% | 0.8950 | 24.16 |
| dbc_020 | 45.0% | 61.07 | 56.26 | +9.0% | 0.8950 | 24.16 |

**六个增大阈值配置均未通过整帧 SSIM ≥0.95 / PSNR ≥28 dB 的质量线。**
两次 off 输出逐字节一致，并与上一组 50 步 off 一致，保证质量比较使用相同基准。

TeaCache 0.03 与 0.10 都复用 108/240 次，视频逐字节一致；cache-dit 0.10 与 0.20 也都复用
108/240 次，视频逐字节一致。当前 warmup/end-guard/max-consecutive=1 下，复用率已经达到
45% 上限；继续增加这些阈值没有新增跳步收益。cache-dit 复用中段 27/28 层，对应总 block
执行数减少约 43.4%，不是整个前向完全跳过。

**表中耗时变化不能解释为稳定加速比。** 前后 off 为 67.13 / 86.56 秒，相差约 28.9%；
相同输出、相同复用次数的 TeaCache 0.03/0.10 也分别耗时 67.40 / 40.71 秒。
因此 TeaCache 0.10 的单次耗时下降 39.4% 是本轮观测值，不能归因于它比 0.03 多跳步。
cache-dit 最快观测值为阈值 0.10、57.28 秒，相对首个 off 减少 14.7%。
每档仅一次，尚不满足稳定负载、多次重复的性能门禁；本轮不提升默认阈值。

可复现任务、输出、逐窗口命中、分区域质量和 GPU 2 采样：
`results/transformer_cache_smoke/higher_thresholds/` 下的 `tasks.json`、`runs.json`、
`quality.json`、`summary.json`、`gpu2.csv`、MP4 与完整日志。推理和监控进程均已退出。
沿用本报告 CLI 命令，将 `--output-path` 与缓存模式替换为
`--task-file results/transformer_cache_smoke/higher_thresholds/tasks.json`，保持 50 调度步；
比较脚本的 `--directory` 改为该目录即可。

### 后续优化

FP32 残差、无损文本投影/KV 缓存及本轮 GPU 0、1、6、7 实测见
[缓存优化报告](cache_optimization.md)。当前活动缓存模式默认自动复用文本投影；
可用 `--no-cache-text-projections` 关闭，或在 `off` 模式用 `--cache-text-projections` 单独启用无损档。
