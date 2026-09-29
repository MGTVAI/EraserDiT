# 双卡并行、缓存与量化推进（2026-09-27）

顺序按用户要求：完成单卡编译/注意力筛选后，先推进多卡，再测试 TeaCache、CacheDiT 和量化。
单卡结果见 [SSIM ≥ 0.98 验证](quality98_optimization_20260927.md)。
并行继续按整段 SSIM ≥ 0.98 验证；后三项不要求与参考对齐，改以基本擦除效果、
输出完整性和异常画面检查为主，SSIM 仅作诊断。始终不降低分辨率、输出帧数或配置采样步数。

## 双卡 CFG

使用物理 GPU 2、3，两张 A100 80GB。正、负条件分支各占一张卡，DiT 副本常驻并跨窗口复用。
新增支持保留主卡 T5 FSDP CPU offload 与 VAE CPU offload；它们不参与 DiT 的线程式并行。
多卡 DiT 权重卸载、VAE 多卡卸载继续拒绝。没有导入 `sglang` 包。

各 rank 的 RoPE 改为每窗口计算一次、在该 rank 的分支/步数之间复用，窗口退出释放，
避免跨形状复用；新增窗口隔离和数值一致性测试。

| 配置 | 请求 s | 去噪 s | 整段 SSIM | 各卡 peak allocated GiB |
|---|---:|---:|---:|---|
| 单卡 SDPA + 尾窗口减填充，前轮 | 240.915 | 201.709 | 0.992833 | 30.653 |
| 双卡 CFG SDPA + 尾窗口减填充 | 141.052 | 102.288 | 0.992833 | 34.045 / 5.760 |
| 双卡 CFG Sage 2 + 首步保护 + 减填充 | 140.505 | 102.228 | 0.982647 | 34.045 / 5.792 |

双卡请求减少 **41.5%**，约 **1.71×**；去噪减少 **49.3%**。
主卡增加的显存主要来自常驻 DiT 权重，副卡只承载 DiT 分支。各卡峰值不能当作同一时刻的总峰值。
参考视频为 `results/performance_20260927/fused50.mp4`，参数与单卡报告一致。
逐帧数值与单卡减填充结果的指标相同；抽查第 24/72 帧，人物已被擦除。
这是单次完整请求筛选，不是五次稳态统计。模型加载约 42.32 s，单列于请求外。

### 并发初始化边界

首次双卡 Sage 2 全驻留试验在尾窗口切换时 SIGSEGV，没有生成视频，不计作成功或性能结果。
系统记录落在 libstdc++，未取得该次原生栈，尚不能确定根因。
后续增加每窗口首个 CFG step 顺序执行，再并行后续 steps：初始化懒加载内核/设备状态时
避免两个分支同时首次进入；这些首步结果直接用于采样，没有额外假步或缓存更新。
该策略只用于 CFG2/SP1，SP collectives 不能逐 rank 串行运行。
上表 SDPA 141.052 s 测量在加入串行首步保护前；后续结果另列。

CPU 配置与 mesh 专项通过，覆盖主卡组件卸载、CFG 首步真实输出、RoPE 跨步复用和窗口释放。
Sage 2 的带保护重试完整运行通过，两个窗口均结束，整段 SSIM 0.982647。
这证明该配置可以完成本次任务，尚不能把首步保护描述为已证实的根因修复。
相对单卡 Sage 2 的 235.614 s，双卡为 140.505 s，减少 **40.4%**、约 **1.68×**。

## 缓存与量化的验收口径

用户明确允许 TeaCache、CacheDiT、量化不对齐参考，只要大致完成擦除。
固定输入、mask、prompt、seed 和输出几何，核查人物仍可辨识的残留、严重块状或黑帧伪影、
窗口拼接异常及完整帧数；SSIM 不再作为后三项的拒绝门槛。
视觉抽查包含站立、后仰下落和尾部水花；不把高 SSIM 等同于擦除成功。

本轮候选均显式启用，默认关闭：

- TeaCache：双卡 CFG，threshold=0.3，连续跳步最多 3，warmup=4，末尾保护 1。
- CacheDiT：双卡 CFG，前 1 / 后 1 个 block 实算，threshold=0.3，连续缓存最多 3，warmup=4，末尾保护 1。
- INT8：单卡 `int8_w8a8_native`，blocks 范围的指定 Linear；权重按输出通道、激活按 token 对称量化，
  使用原生 INT8 GEMM，无浮点 GEMM 静默回退。保持主卡 T5/VAE CPU 卸载，DiT 常驻。

新允许 INT8 常驻 DiT 与 T5/VAE CPU 卸载，以及 SP1 上图外 QK/AdaLN 融合组合；
多卡 SP 的 INT8+融合限制、DiT INT8 权重卸载限制继续保留。
真实 INT8 算子/模块转换与配置专项 7 项通过。实验产物：`results/cache_quant_20260927/`。

### 实测与画面检查

| 配置 | 请求 s | 去噪 s | 各卡 allocated GiB | SSIM（仅诊断） |
|---|---:|---:|---|---:|
| 双卡 CFG + TeaCache | 79.206 | 36.682 | 34.045 / 6.323 | 0.979842 |
| 双卡 CFG + CacheDiT | 79.690 | 41.129 | 34.045 / 6.697 | 0.981373 |
| 单卡 INT8 blocks，无缓存 | 243.027 | 203.746 | 32.502 | 0.981043 |
| 双卡 CFG + TeaCache + INT8 blocks | 75.322 | 37.113 | 32.502 / 5.323 | 0.978960 |

两种缓存均在 160 个逻辑分支步中复用 108 步（67.5%）。CacheDiT 复用中间 block，
前/后 block 仍计算，因此其去噪时间略长。两者总耗时差距不足 0.5 s，不宣称存在稳定排名；
本轮都约 80 s。相对双卡无缓存 SDPA 141.052 s，TeaCache 减少 43.8%，
相对 391.849 s 未减填充单卡基线减少 79.8%，约 4.95×。

三项均完整解码 145 帧、1920×1080、帧率不变。抽查 **0、24、48、72、96、132** 帧，
站立/下落人物已基本擦除，没有看到明显人形残留或黑帧；背景与水花细节有变化。
这属于抽样画面检查，没有做逐帧人工播放或多素材泛化验收。
TeaCache 的 0.979842 低于此前 0.98，但用户已对缓存取消对齐门槛，因此不据此拒绝。
对照图：`tea_review.jpg`、`cachedit_int8_review.jpg`。

TeaCache+INT8 也完成相同六帧抽查与全视频解码，人物基本擦除，对照图为 `tea_int8_review.jpg`。
其 SSIM 0.978960 仅作诊断，不适用 0.98 拒绝门槛。主/副卡相对纯 TeaCache 分别减少
约 **1.54 / 1.00 GiB allocated**；reserved 主卡为 52.201 GiB。
组合去噪 **37.113 s** 比纯 TeaCache **36.682 s** 略长，所以请求 **75.322 s** 对
**79.206 s** 的差异不能归因于 INT8 内核提速，差异主要落在其他阶段。
当前把两者视为 **约 75–80 s** 的候选；需要更少权重驻留时选 INT8 组合。
本次最快 75.322 s 相对 391.849 s 减少约 **80.8%**，约 **5.20×**，仍属单次筛选结果。

INT8 实际转换并执行 **224** 个 Linear，主模型累计 **35840** 次调用，浮点回退为零、
无 BF16 重复权重；选中 Linear 的存储从 **3289595904 → 1647951872 bytes**。
但单卡总耗时没有优于当前单卡方案，因此不把它作为独立提速配置推荐。
其 DiT 常驻，不能把 32.502 GiB 与逐层卸载的 30.653 GiB 描述成进一步降显存。

同卡五组 INT8 Linear 微基准（BF16 / INT8，长窗口 ms）：
2048→2048 为 **1.074 / 1.298**，2048→8192 为 **4.415 / 4.713**，
8192→2048 为 **4.073 / 3.736**。量化输入、INT32 中间结果和独立 epilogue 的成本
抵消了部分 GEMM 收益；后续量化提速应优先研究融合 GEMM/反量化输出，而非扩大替换范围。

## 复现缓存候选

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/tea-cfg2.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 --infer-len 121 --overlap 9 \
  --cfg-degree 2 --parallel-devices 0,1 \
  --no-dit-layerwise-offload --no-dit-cpu-offload \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --attention-backend sdpa --operator-fusion-backend auto \
  --compact-tail-padding --no-cache-text-projections \
  --transformer-cache-mode teacache --teacache-threshold 0.3 \
  --max-teacache-consecutive-skip 3 --teacache-warmup-steps 4 --cache-end-guard-steps 1
```

CacheDiT 替换最后两行的 TeaCache 参数为：

```bash
--transformer-cache-mode cache_dit --cache-dit-front-blocks 1 --cache-dit-back-blocks 1 \
--cache-dit-residual-diff-threshold 0.3 --cache-dit-max-consecutive-cached-steps 3 \
--cache-dit-warmup-steps 4 --cache-end-guard-steps 1
```

两种残差缓存二选一。近似候选都保持显式启用，默认关闭。
在 TeaCache 命令上增加下面参数可复现显存更低的组合，并另设输出文件：

```bash
--transformer-quantization int8_w8a8_native --quantization-scope blocks
```

最终 CPU 全量 122 项：98 通过、24 条件跳过；`git diff --check` 通过。

## 使用双卡基线

沿用单卡命令，改用 `CUDA_VISIBLE_DEVICES=2,3` 并设置：

```bash
--cfg-degree 2 --parallel-devices 0,1 \
--no-dit-layerwise-offload --no-dit-cpu-offload \
--text-encoder-cpu-offload --vae-cpu-offload \
--attention-backend sdpa --operator-fusion-backend auto \
--compact-tail-padding --transformer-cache-mode off --no-cache-text-projections
```

逻辑设备编号从可见 GPU 列表重新编号，`0,1` 对应物理 `2,3`。
实验产物：`results/multi_gpu_20260927/`。
