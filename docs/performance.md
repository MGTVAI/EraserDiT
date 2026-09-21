# 性能与验收

[CLI](cli.md) · [服务 API](service_api.md) · [测量脚本](../scripts/README.md) · [测试](../tests/README.md)

本文合并单卡加速、卸载、缓存、并行、量化及验收报告。数据来自已有本地实验，
本次文档整理未重跑推理。原始日志、视频和 JSON 仍在 `results/`，通常不随 Git 分发。

<a id="acceptance"></a>
## 统一验收标准

非缓存、非量化优化需同时满足 **RGB SSIM ≥ 0.985、MSE ≤ 36、MAE ≤ 6**。
固定输入、权重、提示词、seed、采样、窗口和随机性口径，对照未优化输出。
解码 RGB 像素范围 0–255；MSE/MAE 在所有帧、像素、通道平均；SSIM 使用
11×11 Gaussian 窗口、sigma=1.5、reflect 边界并按通道和帧平均。
视频帧数、尺寸、帧率必须一致，逐帧指标用于诊断，不另设逐帧门槛。

```bash
python -m scripts.validation.optimization_quality \
  --reference baseline.mp4 --candidate optimized.mp4 --report quality.json
```

通过退出 0，不通过仍写报告并退出 2。缓存与量化以画面大致相似为要求；加 `--lossy`
记录诊断指标并标记 `visual_review_required`、`accepted=null`，还需目视检查擦除区域、结构、颜色及闪烁。
历史 Y 通道 SSIM/PSNR、逐字节一致和旧 PASS 记录不能替代新的 RGB 验收。
通信正确性、状态隔离、异常恢复等功能断言仍保留。

性能比较应在同卡、同配置下交错测量，每配置至少五次，记录中位数、离散度与 GPU 占用。
加载、转换、编译、预热单列，区分冷启动与稳态请求耗时；allocated 与 reserved 分开报告。
以下单次探索结果只说明对应素材和环境，不作为稳态性能保证。

## 配置选择与限制

| 目的 | 配置 | 限制 |
| --- | --- | --- |
| 单卡加速 | `sage_attn` + `--enable-torch-compile --warmup` | fullgpu；关闭缓存、并行、量化、算子融合 |
| 节省显存 | `--resource-policy dynamic_offload` | 关闭 compile；预算只约束受管权重 |
| 双卡单任务 | `--cfg-degree 2` | fullgpu；关闭 compile、缓存、量化 |
| 四卡单任务 | `--cfg-degree 2 --sp-degree 2` | 同上；主卡显存不会均分到四卡 |
| 近似推理 | TeaCache / CacheDiT | 关闭 compile、单任务并行和量化；需视觉验收 |
| INT8 实验 | `--transformer-quantization int8_w8a8_native` | fullgpu、BF16、关闭 compile、融合、缓存、单任务并行；未显示提速 |

<a id="single-gpu"></a>
## 单卡注意力与编译

A100 80GB 共享卡、官方快照 `904fb412da76235085dbbccaefdbde4979fa3d29`，
两组原尺寸素材、50 调度步 / strength=0.8、seed=42、窗口 121 / overlap=9。
M3/M4 快速口径为 `ERASERDIT_DETERMINISTIC=0`，每配置五次。

M3 素材 `10268234`：SDPA 端到端中位数 216.4 s、去噪 178.4 s；
SageAttention + compile + warmup 为 173.4 s、148.0 s，预热另计 51.2 s。
M4 两组素材去噪耗时减少约 18.0% / 17.6%（约 1.22× / 1.21×）。
端到端受共享卡非去噪阶段波动影响更大，不能直接套用最高加速比。

单独更换注意力后端的收益较小；预热是上述稳态编译收益的前提。
连续四任务验证覆盖两种画幅与 seed 切换，历史峰值 reserved 达 59.06 GiB，
高于单任务约 53–54 GiB。M3/M4 使用旧质量标准，尚不能据此宣称已过当前 RGB 门禁。

原始记录：`results/m3/`、`results/m3-clean/`、`results/m3-warmup/`、`results/m4/`、
`results/acceptance_payload.json`。历史复现工具在 `scripts/legacy/`；
基线比较依赖相邻的 `EraserDiT-baseline` 工作树，基线与快速口径不能直接混算。

<a id="offload"></a>
## 权重卸载

`component_offload` 按文本编码、VAE 编解码、去噪阶段整体搬运组件，正常或异常退出后返回 CPU。
`dynamic_offload` 默认预算 5 GiB，能放下时整阶段驻留，否则按块异步搬运。

```bash
# 添加到普通 CLI 命令
--resource-policy dynamic_offload --max-weight-usage 5368709120
```

预算单位为字节，不包含激活、workspace、未包装小层及 allocator reserved。
受管权重始终保留 pinned CPU 镜像（约 14.75 GiB）；`--pin-memory` 仅额外固定小层。
整组件卸载也不能消除 VAE 激活峰值。CLI `memory_runtime` 与服务端
`effective_acceleration.memory_runtime` 记录驻留及搬运情况，搬运计数按会话累计。

原尺寸 `10268234`、SDPA、确定性口径、无 compile/融合、单次请求：

| 配置 | 端到端 | 峰值 allocated | 峰值 reserved |
| --- | ---: | ---: | ---: |
| fullgpu | 210.138 s | 47.115 GiB | 53.314 GiB |
| dynamic_offload，5 GiB | 215.269 s | 33.631 GiB | 43.471 GiB |

输出字节一致；allocated 降约 28.6%，端到端增加约 2.4%。
小尺寸整组件卸载为 3.050 → 30.704 s、allocated 15.055 → 8.923 GiB，传输开销明显。
连续任务、注入异常后恢复和服务检查已有记录，但没有两素材五次稳态统计。
报告：`results/dynamic_offload_smoke/fullsize_comparison.json`、`repeat_5g.json`（同目录）。
复现工具：`python -m scripts.validation.offload_verify`，沿用 CLI 参数，
`--output-path` 为 JSON，添加 `--repeat 2 --inject-failure`。

<a id="cache"></a>
## Transformer 缓存

默认关闭残差缓存、预测器为 `none`；文本投影默认 auto，随残差缓存开启。
仅复用固定文本投影可使用 `--transformer-cache-mode off --cache-text-projections`，不跳去噪计算。
所有缓存按 CFG 分支及窗口隔离，正常与异常退出均清理。

TeaCache 使用第一层调制输入的相对 L1，当前是未拟合恒等系数；
CacheDiT 保留前段探针与中段残差。残差以 FP32 相减、保存与恢复。
`--cache-residual-predictor linear` 增加 FP32 变化率和显存占用，未稳定胜出，默认不开启。
默认 warmup=4、末步保护=1、最多连续复用一步。
默认阈值 TeaCache=0.005、CacheDiT=0.03 是实验起点，不是质量保证。

两段完整素材阈值 0.2 复测，SageAttention、无 compile、40 有效步/窗口，单次测量：

| 素材 | 模式 | 端到端 | 相对本轮 off 加速 | 命中 / CFG 前向 | SSIM-Y（指定参考） |
| --- | --- | ---: | ---: | ---: | ---: |
| 10268234 | TeaCache | 129.6 s | 1.628× | 36/80 | 0.987175 |
| 10268234 | CacheDiT | 129.9 s | 1.624× | 36/80 | 0.986218 |
| 113000356 | TeaCache | 244.6 s | 1.643× | 72/160 | 0.990819 |
| 113000356 | CacheDiT | 247.6 s | 1.623× | 72/160 | 0.990772 |

历史抽查两组画面基本相似，仍有纹理差异，不保证其他素材。第一组指定参考包含 compile，
表中质量与速度的对照对象不同，不能将参考误差全部归因于缓存。
试用需显式选择阈值，并保持默认值不变：

```bash
--transformer-cache-mode teacache --teacache-threshold 0.2
# 或
--transformer-cache-mode cache_dit --cache-dit-residual-diff-threshold 0.2
```

原始视频、任务及逐帧指标：`results/cache_correctness/high_threshold/summary.json` 及同目录文件。
测量工具 `scripts/benchmarks/cache_benchmark.py` 要求显式提示词，默认窗口 121、交错五次重复；
`--reference` 可追加指定参考。`scripts/validation/cache_compare.py` 保留区域和逐帧诊断，
掩码取 RGB 第一通道、threshold=0.039，其 Y 通道指标不等同于统一 RGB 门禁。

<a id="parallel"></a>
## CFG / SP / VAE / DP 并行

CFG 将正负分支分配到两卡，SP 在分支内分配序列计算，VAE 在编解码阶段复用设备，
DP 将独立视频任务分给不同 worker。未实现 TP/PP。

| 组合 | CLI 参数 | 所需可见 GPU |
| --- | --- | ---: |
| CFG | `--cfg-degree 2` | 2 |
| SP | `--sp-degree 2` 或 `4` | 2 / 4 |
| CFG × SP | `--cfg-degree 2 --sp-degree 2` | 4 |
| VAE 空间分片 | `--vae-degree 2` 或 `4` | 2 / 4，可复用去噪设备 |
| DP × CFG / SP | dispatcher `--dp-degree 2` 加 `--cfg-degree 2` 或 `--sp-degree 2` | 4 |

单任务 mesh 最多四卡，设备数为 `max(cfg_degree × sp_degree, vae_degree)`。
旧 `--cfg-parallel-device cuda:1` 仍支持，不与新 mesh 参数混用。
SP 默认 `--sp-linear-mode reference` 保留原矩阵形状控制 BF16 漂移；
`sharded` 已通过一组原尺寸质量验证，但性能受其他负载干扰，尚不调整默认模式。
VAE 默认保留完整空间上下文；`--vae-tiling` 是近似分块，需单独验收。

原尺寸 `10268234`、SageAttention、关闭缓存，均为共享卡单次探索数据：

| 配置 | 端到端 | 相对各自单卡基线 |
| --- | ---: | ---: |
| CFG=2（双卡批次） | 118.259 s | 1.774× |
| SP=2 reference | 150.021 s | 1.400× |
| CFG=2 + VAE=2 | 121.353 s | 1.730× |
| CFG=2 × SP=2（四卡批次） | 87.682 s | 2.398× |
| CFG=2 × SP=2 + VAE=4 | 88.191 s | 2.385× |
| SP=4（补测批次） | 118.665 s | 1.762× |

上述输出各自与单卡基线字节一致。双卡优先 CFG，四卡优先 CFG×SP；
叠加 VAE 未显示额外端到端收益。CFG×SP 主卡 allocated 约 47.146 GiB，
增加 VAE=4 后为 44.312 GiB，其余三卡各约 25 GiB。
四卡批次物理卡 7 有常驻占用，不能视为独占稳态统计。
DP×CFG、DP×SP 连续任务通过小尺寸功能验证，尚未给出稳态吞吐结论。

报告：`results/parallel_matrix/validation_summary.json`、
`results/parallel_four_20260921/validation_summary.json`、同目录 `full_sp4.json`；
sharded 质量记录在 `results/parallel_quality_20260921/sharded_quality.json`。
矩阵工具 `python -m scripts.benchmarks.parallel_benchmark` 沿用 CLI 输入参数，输出路径为 JSON。
四卡包装脚本固定检查物理 2、3、6、7 三次空闲后启动，占用时退出 75；首参数 `--wait` 可继续等待。

<a id="quantization"></a>
## 单卡 INT8 W8A8

`int8_w8a8_native` 使用按输出通道权重量化、按 token 激活量化和 FP32 scale，
以 Triton + `torch._int_mm` 执行，输出 BF16，无浮点 GEMM 回退。
默认 `--quantization-scope blocks` 转换 224 个线性层；`ffn` 仅 56 层，需单独验收。
文本编码器、VAE、条件及输入输出投影保留原精度。checkpoint 不修改。
私有整数接口已测环境为 PyTorch 2.6.0+cu126，升级后需复验。

原尺寸 `10268234`、A100 80GB、SageAttention，两种模式分别预热一步后各测一次：

| 配置 | 端到端 | 去噪 | 峰值 allocated |
| --- | ---: | ---: | ---: |
| BF16 | 208.046 s | 180.183 s | 47.115 GiB |
| INT8 | 210.411 s | 182.236 s | 45.586 GiB |

未显示加速，显存减少 1.529 GiB（约 3.25%）。224 层全部执行，fallback=0，
Profiler 确认 INT8 Tensor Core kernel。RGB SSIM=0.972664、MSE=25.1834、MAE=2.73260，
仅作诊断；历史首中末抽帧结构基本一致，未完成全分辨率连续视频目视验收。
保持显式实验选项，不作为提速推荐。

报告及视频：`results/quantization_20260921/validation_summary.json` 及同目录。
工具 `python -m scripts.benchmarks.quantization_benchmark` 沿用 CLI 参数，输出路径为 JSON；
`--warmup --warmup-steps 1` 独立预热两种模式，`--quant-only` 仅执行量化项。
