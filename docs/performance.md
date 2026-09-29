# 性能配置

[NCCL 输出汇聚与进程边界优化](nccl_boundary_optimization_20260929.md)：最终输出仅向必要 rank 汇聚，
CFG 在 worker 内合并，返回预测张量字节数减半；新增逐窗口边界耗时和各 rank GPU 区间统计。

[双卡 CFG + 完整 DiT 编译](cfg_compile_20260928.md)：预热后两次为 142.994 / 143.726 秒，
相比同会话无融合 eager 请求减少约 11.9%；相比补测的手工融合预热结果减少约 5.0%。
整段 SSIM 0.982309，三次编译输出 RGB 一致；首次请求 253.567 秒，SP>1 仍不支持完整编译。

[完整 DiT + VAE decoder 原片验证](decoder_full_validation_20260928.md)：
组合整段 SSIM 0.982273；重复请求 224.084 秒，对照仅 DiT 为 224.128 秒，
新增 decoder 编译没有证明有效端到端收益。

新增 [T5 与 VAE 组件编译](component_compile_20260928.md)，可独立选择编译目标。

新增 [完整 DiT 编译](full_transformer_compile_20260928.md)：显式 transformer 范围，
真实权重 forward 耗时减少约 17%–30%，原片整段 SSIM 0.982325；
首次请求包含编译成本，为 310.927 秒，未优于此前单卡方案。限制与详细记录见报告。

最新 [四卡 CFG×SP 验证](four_gpu_optimization_20260927.md)：无缓存约 96 秒，TeaCache 两次约 59 秒；
后续见 [参考 SGLang 的优化方案](next_optimization_plan_20260927.md)。

当前整段质量目标更新为 [RGB SSIM ≥ 0.98](quality98_optimization_20260927.md)，
记录编译与单卡卸载适配、FlashAttention / SageAttention 的组合筛选。
此前 [0.985 验证](quality985_optimization_20260927.md) 保留当时判定。
后续 [双卡 CFG、TeaCache、CacheDiT 与 INT8](multi_gpu_cache_quant_20260927.md)：
并行仍验证 0.98；按用户后续要求，缓存与量化以基本擦除效果验收，SSIM 仅作诊断。

此前 [SSIM ≥ 0.99 的单卡优化验证](quality99_optimization_20260927.md) 接入可选尾窗口减填充和
TeaCache 的限定层卸载计划。前者在示例整段达标，后者所测阈值未达标，均不默认开启。

最新显存优化与静态条件复用见 [2026-09-27 验证](memory_optimization_20260927.md)。

后续 [DiT 融合优化](performance_optimization_20260927.md) 修正了 QK/RoPE 数值边界。
单卡 BF16 eager + 当前逐层卸载可增加 `--operator-fusion-backend auto` 启用；
真实形状逐层 DiT forward 五次交替测量中位数减少 7.7%，原片 50 步相对本轮基线逐像素一致。
默认仍关闭融合。局部 FFN compile 已支持单卡逐层卸载和图外融合，INT8 组合限制保持原样。

下一阶段见 [单卡优化方案](single_gpu_optimization_plan.md)：先验证文本投影缓存和精确融合，
再按实测推进局部编译；残差缓存与卸载协同作为独立的可选路线。

[本轮实施与筛选](single_gpu_optimization_20260927.md) 新增显式 `gated_residual` 融合，
并验证文本投影缓存、局部 FFN 编译；当前结果不支持将这些实验候选加入默认配置。

[CLI](cli.md) · [服务 API](service_api.md) · [测量步骤](validation.md) · [测试](../tests/README.md)

当前内存管理已迁移为 SGLang 源码方案，配置见 [内存管理](sglang_memory.md)。
下面 2026-09-22 / 23 的数据属于迁移前实现；其中旧参数和卸载组合不能直接复跑。

## 实测性能总结（2026-09-22）

使用现有 EraserDiT 环境、A100 80GB、`data/model` 权重，原片为
`data/113000356.mp4` 及对应 mask（1920×1080、145 帧、24000/1001 FPS）。
固定 BF16、seed=42、50 个配置步、strength=0.8，共两个窗口，每窗口实际去噪 40 步。
以下耗时为完整任务耗时，不含模型加载及显式预热；显存为主卡 PyTorch peak allocated，
不代表多卡合计、reserved 或整张卡占用。未特别说明时使用 SDPA、fullgpu、关闭量化。

### 原片性能与质量

| 配置 | 耗时 s | 相对基线加速 | 主卡峰值 GiB | RGB SSIM | 结果说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| 单卡无缓存基线 | 400.37 | 1.00× | 47.14 | 1.000000 | 历史对照 |
| 双卡 CFG=2 | 225.45 | 1.78× | 47.14 | 1.000000 | 整段 RGB 与基线一致 |
| 双卡 SP=2 reference | 285.62 | 1.40× | 47.14 | 1.000000 | 整段 RGB 与基线一致 |
| 单卡 TeaCache 0.3 | 248.20 | 1.61× | 47.14 | 0.982012 | 允许有损，复用率 45% |
| 单卡 CacheDiT 0.3 | 252.65 | 1.58× | 47.14 | 0.981961 | 允许有损，中段复用率 45% |
| 单卡动态卸载，2 GiB 权重预算 | 409.31 | 0.98× | 33.64 | 1.000000 | 整段 RGB 一致，峰值降低约 29% |
| 单卡 TeaCache 0.02 + 动态卸载 | 348.14 | 1.15× | 33.64 | 0.983139 | 历史有损组合，复用率 17.5% |
| 单卡 SageAttention + 两种算子融合 | 357.38 | 1.12× | 47.14 | 0.983132 | 未达到非缓存、非量化 SSIM 门槛 |

在本次原片测试中，双卡 CFG 的耗时最短；SP=2 reference 同样保持像素一致。
单卡允许缓存损失时，阈值 `0.3` 的 TeaCache / CacheDiT 均获得实际复用收益。
优先降低显存时，动态卸载将主卡峰值从 47.14 GiB 降到 33.64 GiB，耗时略增。
该历史轮次未测试双卡叠加残差缓存/卸载，不能把表内单项加速比直接相乘。

两种 `0.3` 缓存均开启文本投影缓存，warmup=4、末步保护=1、最多连续复用一步。
每项两个窗口合计 160 个 CFG 分支步，计算 88 步、复用 72 步；CacheDiT 复用时仍计算探针。
TeaCache / CacheDiT 的纯推理分别为 215.83 / 219.00 s，MSE 为 3.138675 / 3.131396，
MAE 为 1.265181 / 1.255964。两段视频均通过完整解码、帧数、尺寸和帧率检查。
缓存与量化允许有损，这些质量指标只记录差异，不用非缓存、非量化门槛判定失败。
历史 TeaCache `0.005` 原片耗时 408.64 s、复用率为 0；该值已不再是默认阈值。

上述原片配置各测一次，加速比使用历史基线，不是五次重复的稳态统计。
基线未启用文本投影缓存，缓存行包含其收益；SP=2 使用 GPU 6、7，
其中 GPU 7 测试前已有 12253 MiB 显存占用，设备占用条件并不完全一致。
本次未进行整段人工播放验收。

### 其他优化的覆盖范围

初始矩阵与 SP=2 追加测试共覆盖 44 个配置、58 个完整解码输出；随后追加上述两项
`0.3` 缓存原片测试。短片为 512×288、33 帧，其耗时不能直接外推到原片。

| 短片配置 | 正式推理 s | 次数 | 观察 |
| --- | ---: | ---: | --- |
| SDPA 基线 | 8.727 | 5 | 中位数 |
| SageAttention | 8.925 | 5 | 中位数，本素材未提速 |
| SageAttention + compile | 6.541 | 5 | 中位数，SSIM 0.983674 |
| SDPA + compile | 6.505 | 2 | 中位数，SSIM 0.983813 |

编译耗时不含加载和显式预热；上述编译组合有数值差异，未达到非缓存、非量化 SSIM 门槛。
文本投影缓存、单独 RMSNorm+AdaLN 融合、非分块双卡 VAE、DP 和卸载类配置已完成短片验证，
对应输出与基线一致；QK/RoPE 融合、SP sharded、窗口流式处理和近似 VAE 分块存在数值差异。
完整参数、耗时和质量见[44 项矩阵](optimization_validation_20260922.md#完整配置矩阵)。
四卡组合未实测；A100 不支持的 `sage_fp8` 未运行；权重量化未纳入本次性能测试。

### 复跑与原始记录

在仓库根目录执行，脚本使用已有环境并包含完整输入、采样、GPU 和优化参数：

```bash
# 双卡 CFG=2
bash results/optimization_validation_20260922/full_cfg2/command.sh
# 双卡 SP=2 reference
bash results/optimization_validation_20260922/full_sp2_reference/command.sh
# 单卡动态卸载，2 GiB 权重预算
bash results/optimization_validation_20260922/full_dynamic_2g/command.sh
# 顺序运行 TeaCache 0.3 和 CacheDiT 0.3 的完整原片
bash results/cache_threshold_03_20260922/command.sh
```

旧缓存脚本若未显式指定阈值，复现旧结果时需补上当时的 TeaCache `0.005` / CacheDiT `0.03`；
当前两者默认均为 `0.3`。历史性能记录保留，不用新阈值结果覆盖旧数据。
详见[非量化优化验证](optimization_validation_20260922.md)与[阈值 0.3 验证](cache_threshold_03_validation_20260922.md)。
原始汇总为 [44 项 JSON](../results/optimization_validation_20260922/summary_with_sp2.json)和
[缓存 0.3 JSON](../results/cache_threshold_03_20260922/summary.json)；`results/` 产物保留在本机，不随 Git 分发。

## 当前配置

CLI/服务默认开启单卡 DiT 逐层卸载、T5 FSDP CPU offload 和 VAE 组件卸载。
无字节预算。预取参数 0 表示一层；完整说明见 [SGLang 内存管理](sglang_memory.md)。

<a id="offload"></a>
卸载支持局部 FFN compile，CUDA graphs 关闭；DiT 的 INT8/多卡权重卸载暂不支持。
CFG/SP 常驻 DiT 可保留主卡 T5/VAE CPU 卸载，INT8 常驻 DiT 也可搭配主卡组件卸载。
逐层卸载支持显式 TeaCache，仍拒绝 cache_dit；缓存与 compile 的组合限制仍按缓存配置校验。
使用上述加速路径先关闭卸载：

```bash
--no-dit-layerwise-offload --no-dit-cpu-offload \
--no-text-encoder-cpu-offload --no-vae-cpu-offload
```

<a id="single-gpu"></a>
<a id="cache"></a>
<a id="parallel"></a>
<a id="quantization"></a>
全驻留可使用注意力后端、局部 FFN compile、二选一残差缓存、文本投影缓存及 CFG/SP。
历史测试数据不构成迁移后所有组合的验收。具体参数见 [CLI](cli.md)。

<a id="acceptance"></a>
## 质量与性能验证

当前单卡近似优化按用户指定的整段 RGB SSIM ≥ 0.98 验收，同时报告 MSE、MAE、最低帧和时序误差。
固定输入、权重、提示词、seed、采样和窗口配置，与未优化输出比较。
RGB 范围为 0–255，SSIM 使用 11×11 Gaussian 窗口、sigma=1.5、reflect 边界。
帧数、尺寸和帧率必须一致。缓存、量化和 VAE tiling 需记录并检查画面差异。

性能至少五次同条件重复，加载与推理分开统计，记录中位数、波动和其他进程占用。
allocated/reserved 是本进程 PyTorch 指标，不代表整卡占用。

2026-09-28 的 NCCL DiT 并行任务使用更新后的 **整片 RGB SSIM ≥0.985** 门槛。
历史 0.98 门槛下的编译/缓存结果不能直接作为本轮合格配置。
DP、CFG、Ulysses、流式 Ring/USP、TP、FSDP/HSDP 的实现边界与新测量见
[NCCL 并行实施与验收](distributed_parallel_20260928.md)。
