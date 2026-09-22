# 性能配置

[CLI](cli.md) · [服务 API](service_api.md) · [测量步骤](validation.md) · [测试](../tests/README.md)

默认使用 SDPA、BF16、单 GPU 和 `fullgpu`。根据显存与吞吐需求选择下列配置，
每次只改变一个选项，并使用实际素材检查画面和耗时。

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
`dynamic_offload` 能放下时整阶段驻留，否则按块异步搬运，默认受管权重预算为 5 GiB：

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
--transformer-cache-mode teacache --teacache-threshold 0.005
# 或
--transformer-cache-mode cache_dit --cache-dit-residual-diff-threshold 0.03
```

TeaCache 根据调制输入变化判断复用；CacheDiT 保留前段探针与中段残差。
默认 warmup=4、末步保护=1、最多连续复用一步；残差以 FP32 保存和恢复。
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

缓存与量化需完整播放检查擦除区域、结构、颜色及闪烁，数值指标作为诊断。
同卡同配置至少测量五次，记录中位数、离散度、GPU 占用及 allocated/reserved 峰值。
加载、转换、编译和预热单列，区分冷启动与稳态。具体操作见[测量与验证](validation.md)。
