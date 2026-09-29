# 允许近似计算后的单卡优化（2026-09-27）

本轮将用户的“99%”暂按**整段 RGB SSIM ≥ 0.99**解释，另报告逐帧最低值和时序误差。
SSIM 不是正确像素比例，整段通过也不表示每帧通过。没有降低输入分辨率或采样步数。

## 已实现

### 可选尾窗口减填充

新增 `--compact-tail-padding`，CLI、任务 JSON 和服务请求均可使用，默认关闭。
非首窗口保留全部真实帧和原有 overlap，仅把新帧填充长度缩到最近的 8 帧倍数。
首窗口、完整窗口以及 overlap 不足的窗口保留原路径。
例如 145 帧视频的窗口为 121 帧和 33 帧；此前第二个窗口也镜像填充到 121 帧，
开启后第二个窗口只计算 33 帧，latent token 从 32640 降到 10200。
这会改变模型看到的时间上下文和随机张量形状，因此属于近似选项。

要求 `infer_len`、`overlap` 均为 `8k+1`，且 `time_sample=8`；不符合时提前拒绝。
窗口调度、输出帧数、overlap 提交规则和采样步数保持不变，模型实际帧数由已有元数据传递。
没有多余尾部填充的任务不应期待同等收益。

### TeaCache 与逐层卸载协同

为 SGLang 来源的层管理器增加限定层执行计划：探针仅执行 block 0，未命中时执行全部 block；
预取范围限制在计划内，末端不回绕，缓存命中时不会加载未执行的后续层。
复制流分配的权重使用 `record_stream` 记录计算流使用，避免在计算结束前回收 storage。
退出计划和异常路径释放全部受管层，再恢复正常循环模式。
CFG 分支和窗口继续使用原有独立缓存生命周期；没有导入 `sglang` 包。

只放开 TeaCache + 单卡逐层卸载。CacheDiT + 逐层卸载、compile + 卸载、量化 + 卸载等
既有限制继续保留。支持组合不等于质量达标：下面的缓存候选均不作为 0.99 配置推荐。

## 原片筛选

A100 80GB，`113000356.mp4`，1920×1080、145 帧、24000/1001 FPS；seed=42，
配置 50 步、strength=0.8，每窗口实际 40 步，窗口 121、overlap 9。
SDPA、BF16、确定性模式、QK RoPE + AdaLN 融合，DiT 逐层卸载，T5/VAE CPU offload，
预取一层，文本投影缓存关闭。基线为上一轮 `results/performance_20260927/fused50.mp4`。

| 配置 | 请求 s | 去噪 s | 跳过分支步 / 160 | 整段 RGB SSIM | 达到 0.99 |
|---|---:|---:|---:|---:|---|
| 上一轮无缓存基线 | 391.85 | 340.45 | 0 | 1.000000 | 是 |
| TeaCache 0.01 | 350.88 | 298.88 | 20 | 0.983136 | 否 |
| TeaCache 0.02 | 328.35 | 282.92 | 28 | 0.983144 | 否 |
| TeaCache 0.05 | 269.32 | 215.75 | 60 | 0.982354 | 否 |
| TeaCache 0.05 + 线性预测 | 269.36 | 219.08 | 58 | 0.982787 | 否 |
| 尾窗口减填充原型，无缓存 | 243.01 | 200.97 | 0 | 0.992833 | 是，整段 |
| 正式 CLI 开关复跑，无缓存 | 240.98 | 201.11 | 0 | 0.992833 | 是，整段 |

正式开关复跑与原型的完整解码 RGB SHA256 相同：
`d04a28190a3a57c394d9abf375b0e488dcbf82543dcecec2c6d1add70532165b`。
相对此前基线，正式开关单次耗时减少约 38.5%，约 1.63×；不是五次稳态加速比。

这些是筛选请求，不是五组交替稳态统计。不同候选在物理 GPU 2、3、6 上运行，
存在其他主机作业；加载时间单列，不含于请求时间。原片基线来自此前运行。
不能将缓存和减填充的单项收益相乘，也没有验证两者组合达到 0.99。

所有上述候选的 peak allocated 均为 30.653 GiB，仍由首窗口 VAE 编码决定。
无缓存减填充 reserved 为 48.387 GiB；缓存候选约 49.783 GiB。
普通残差缓存 retained tensor 峰值约 765 MiB，线性预测约 1275 MiB，
额外缓存内存没有超过此次首窗口 VAE 峰值，不代表它不占显存。
请求结束受管 DiT 权重驻留均为零。

## 质量边界

尾窗口原型相对当前基线：整段 SSIM **0.992833**，MSE **1.093645**，MAE **0.416477**。
相对更早的显存优化前原始基线 `before50.mp4`：整段 SSIM **0.992835**，同样通过 0.99。
视频完整解码 145 帧，尺寸与帧率保持一致。

但最差帧 SSIM 为 **0.961088**，最后 24 帧平均约 **0.977860**。
145 帧中有 53 帧低于逐帧 0.99 门槛。
抽查对照图可见水花纹理变化，不能将整段分数描述成“每帧 99% 一致”。
相邻帧误差变化 MAE 为 0.46658（0–255 RGB 尺度），只是差异观测，不是无闪烁证明；
本轮没有进行整段人工播放验收。

指标采用 RGB 三通道均值、Gaussian SSIM 11×11、sigma=1.5、reflect border，
逐帧等权平均。时序误差为相邻两帧 `(candidate-reference)` 差值的绝对均值。
改变素材、窗口长度或叠加其他近似优化后，需要重新验收。

## 第二段素材的反例

`10268234.mp4` 为 1080×1920、120 帧、29.97 FPS，prompt 固定为
`A clear view of the background.`。该素材默认 121 帧配置只有首窗口，按设计不触发减填充。
为实际覆盖尾窗口，对照与候选均设 `infer_len=89`、overlap=9，其余参数同上：
首窗口 89 帧，尾窗口 40 个真实帧（含 9 帧 overlap），候选只将尾部填充到 41 帧。

| 同一 89 帧窗口配置 | 请求 s | 去噪 s | peak allocated GiB | 整段 SSIM |
|---|---:|---:|---:|---:|
| 原填充，GPU 3 | 259.70 | 219.23 | 23.276 | 1.000000 |
| 减填充，GPU 2 | 187.21 | 150.35 | 23.276 | 0.987735 |

完整解码均为 120 帧、尺寸和帧率一致，但质量**未达到 0.99**，因此不推荐该配置。
另存的默认 121 帧无缓存参考请求为 211.05 s，窗口设置不同，不能用它作为上表的质量对照。
两组不同设备的单次耗时仅作筛选记录，不构成稳态统计。
这条反例说明开关不是通用的“99% 精度模式”，保持默认关闭。

## 使用

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
  python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/compact-tail.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 --infer-len 121 --overlap 9 \
  --dit-layerwise-offload --text-encoder-cpu-offload --vae-cpu-offload \
  --dit-offload-prefetch-size 0 --attention-backend sdpa \
  --operator-fusion-backend auto --transformer-cache-mode off \
  --no-cache-text-projections --compact-tail-padding
```

默认关闭；`--no-compact-tail-padding` 恢复原始填充策略。
任务 JSON / API 字段为 `compact_tail_padding`，每个请求独立设置。

## 验证与产物

CPU 最终全量 117 项：95 通过、22 按设备或 opt-in 条件跳过。
GPU 最终专项 47 项全部通过；`git diff --check` 通过。
测试覆盖新参数的 CLI/API 传递和校验、尾窗口 latent/mask 帧数对齐、单帧尾部、完整窗口、
限定层预取、异常后恢复、CFG 分支隔离和跨窗口重置。
真实权重、32640 token 的强制计算探测中，计划卸载与原循环卸载逐元素一致；
说明上述缓存候选的误差来自跳步近似，而非调度本身。

`results/quality99_20260927/` 为本地忽略目录，保存所有命令日志、JSON、输出视频、
逐帧质量指标、`quality.py` / `collect.py`、`tail_review.jpg` 对照图和回归日志。
`tail_probe.py` 仅用于原型实验；正式入口使用上述 CLI 开关，无需加载该脚本。
