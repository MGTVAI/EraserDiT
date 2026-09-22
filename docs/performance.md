# 性能配置

[CLI](cli.md) · [服务 API](service_api.md) · [测量步骤](validation.md) · [测试](../tests/README.md)

CLI 默认使用 SDPA、BF16、单 GPU 和 `dynamic_offload`（2 GiB 受管权重预算）；
性能对照应显式指定 `--resource-policy fullgpu`。根据显存与吞吐需求选择下列配置，
每次只改变一个选项，并使用实际素材检查画面和耗时。

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
双卡单任务目前不能叠加残差缓存或卸载；`0.3` 缓存与卸载的组合尚未重跑，
不能套用表内 `0.02` 组合的性能。

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

## 配置选择

| 目的 | 配置 | 组合限制 |
| --- | --- | --- |
| 单卡注意力加速 | `--attention-backend sage_attn` | 先安装可选注意力依赖 |
| 编译加速 | `--enable-torch-compile --warmup` | fullgpu；关闭残差缓存、单任务并行、量化及算子融合 |
| 节省权重显存 | `--resource-policy dynamic_offload` | 关闭 compile；预算只约束受管权重 |
| 双卡单任务 | `--cfg-degree 2` | fullgpu；关闭 compile、缓存和量化 |
| 四卡单任务 | `--cfg-degree 2 --sp-degree 2` | 同上；显存不会均分到四卡 |
| 残差复用 | TeaCache / CacheDiT | 关闭 compile、单任务并行和量化；检查画面变化 |
| INT8 量化 | `--transformer-quantization int8_w8a8_native` | fullgpu、BF16；关闭 compile、融合、缓存和单任务并行 |

以下参数片段添加到[完整 CLI 命令](cli.md#单视频)使用。具体支持范围由启动校验决定。

<a id="single-gpu"></a>
## 注意力与编译

```bash
--attention-backend sage_attn --enable-torch-compile --warmup
```

SageAttention / FlashAttention 安装见[可选依赖](setup.md#可选依赖)。
编译和预热有一次性成本，常驻会话内的重复请求更适合测量稳态收益。
`--warmup` 仅为首个任务预热，输入形状变化可能再次触发编译。
更换 GPU 或 Torch 版本后在目标环境重新生成编译产物，并检查输出质量。

<a id="offload"></a>
## 权重卸载

`fullgpu` 保持组件驻留 GPU；`fullgpu_pin_memory` 额外使用 pinned CPU 内存。
`component_offload` 按文本编码、VAE 编解码和去噪阶段搬运整个组件。
`dynamic_offload` 能放下时整阶段驻留，否则按块异步搬运，默认受管权重预算为 2 GiB，下面示例显式提高到 5 GiB：

```bash
--resource-policy dynamic_offload --max-weight-usage 5368709120
```

预算单位为字节，不包含激活、workspace、未包装小层及 allocator reserved。
动态卸载的受管权重保留 pinned CPU 镜像，需预留主机内存；`--pin-memory` 额外固定小层。
卸载不能消除 VAE 激活峰值，不能保证任意 GPU 都能处理原尺寸视频。
CLI 的 `memory_runtime` 与服务 `effective_acceleration.memory_runtime` 提供驻留和搬运信息，
搬运计数按会话累计。

<a id="cache"></a>
## Transformer 缓存

默认 `--transformer-cache-mode off`，支持 `teacache` 和 `cache_dit`。
缓存按 CFG 分支及窗口隔离，正常和异常退出时清理。
文本投影默认随残差缓存开启；单独复用文本投影可使用：

```bash
--transformer-cache-mode off --cache-text-projections
```

残差缓存示例：

```bash
--transformer-cache-mode teacache --teacache-threshold 0.3
# 或
--transformer-cache-mode cache_dit --cache-dit-residual-diff-threshold 0.3
```

TeaCache 根据调制输入变化判断复用；CacheDiT 保留前段探针与中段残差。
TeaCache 和 CacheDiT 默认阈值均为 `0.3`。默认 warmup=4、末步保护=1、最多连续复用一步；残差以 FP32 保存和恢复。
完整原片的耗时、复用率和复跑命令见[阈值 0.3 验证](cache_threshold_03_validation_20260922.md)。
`--cache-residual-predictor linear` 增加变化率预测及显存占用，默认 `none`。
阈值控制复用频率和画面误差，需结合素材调节；通过 `transformer_cache_history` 检查实际命中。

<a id="parallel"></a>
## CFG / SP / VAE / DP 并行

CFG 将正负分支分配到两卡，SP 分配序列计算，VAE 在编解码阶段使用多卡，
DP 将独立视频任务分给不同 worker。

| 组合 | CLI 参数 | 可见 GPU 数 |
| --- | --- | ---: |
| CFG | `--cfg-degree 2` | 2 |
| SP | `--sp-degree 2` 或 `4` | 2 / 4 |
| CFG × SP | `--cfg-degree 2 --sp-degree 2` | 4 |
| VAE 空间分片 | `--vae-degree 2` 或 `4` | 2 / 4 |
| DP × CFG / SP | dispatcher `--dp-degree 2` 加 `--cfg-degree 2` 或 `--sp-degree 2` | 4 |

单任务 mesh 最多四卡，设备数为 `max(cfg_degree × sp_degree, vae_degree)`。
`--cfg-parallel-device` 与 mesh 参数不能混用。
SP 默认 `--sp-linear-mode reference` 保持完整矩阵形状；`sharded` 改变计算布局，需检查数值差异。
VAE 默认保留完整空间上下文；`--vae-tiling` 启用近似分块，需要单独检查画面。
多卡收益取决于通信开销、素材大小和设备占用，DP 主要提高多任务吞吐。
完整命令见 [CLI](cli.md#加速与多卡)。

<a id="quantization"></a>
## INT8 W8A8

`int8_w8a8_native` 使用按输出通道权重量化、按 token 激活量化及 FP32 scale，
通过 Triton 和 `torch._int_mm` 执行，输出 BF16。
默认 `--quantization-scope blocks`，也可选 `ffn`；文本编码器、VAE 和输入输出投影保持原精度。
量化在加载后执行，不修改 checkpoint。私有整数接口依赖 Torch 版本，升级后需复验。
量化不保证提速；应分别记录转换成本、稳态耗时、显存和画面变化。

<a id="acceptance"></a>
## 质量与性能验证

非缓存、非量化优化需同时满足 **RGB SSIM ≥ 0.985、MSE ≤ 36、MAE ≤ 6**。
固定输入、权重、提示词、seed、采样和窗口配置，与未优化输出比较。
解码 RGB 范围为 0–255；MSE/MAE 在所有帧、像素、通道平均；SSIM 使用
11×11 Gaussian 窗口、sigma=1.5、reflect 边界，并按通道和帧平均。
帧数、尺寸和帧率必须一致，FFmpeg 默认 SSIM 或 Y 通道指标不能替代上述定义。

缓存与量化允许有损，不以上述非缓存、非量化数值门槛判定失败；数值指标用于记录损失。
需要进行感知验收时，完整播放检查擦除区域、结构、颜色及闪烁。
同卡同配置至少测量五次，记录中位数、离散度、GPU 占用及 allocated/reserved 峰值。
加载、转换、编译和预热单列，区分冷启动与稳态。具体操作见[测量与验证](validation.md)。

本机非量化优化测试的命令、覆盖范围与结果见[2026-09-22 优化验证](optimization_validation_20260922.md)。
