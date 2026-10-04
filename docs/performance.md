# 性能配置与适用范围

[CLI](cli.md) · [测量与验证](validation.md) · [验证记录索引](README.md#实验与验证记录)

当前 CLI / 服务默认使用 BF16、SDPA、单卡 DiT 逐层卸载、T5 FSDP CPU offload 和 VAE CPU offload。
编译、融合、残差缓存、INT8/FP8、VAE tiling、紧凑尾窗均默认关闭。先完成默认配置推理，再逐项比较优化。
本页描述当前配置；带日期的文档记录当时的源码、环境和结果，不能将不同轮次的耗时直接排名。

## L40S 已验证配置

L40S推荐配置与完整命令见[README](../README.md#快速开始)。同一145帧1080p输入，五次请求中位数：
单卡组件卸载420.23 s、双卡NCCL CFG2 241.17 s、四卡CFG2×SP2 reference加输出通信重叠195.93 s。
三者最高单卡任务进程合计峰值分别20.682 / 21.523 / 21.746 GiB，输出与BF16参考一致。
四卡采用`MGERASE_ULYSSES_HEAD_CHUNKS=4 MGERASE_ULYSSES_OUTPUT_OVERLAP=1`；通用默认仍关闭重叠。
多视频吞吐优先DP4，单请求延迟优先CFG2×DP2；完整计时、质量范围与候选淘汰见[L40S验收记录](l40s_validation_20261004.md)。

<a id="offload"></a>
## 显存与卸载

| 目标 | 配置 | 边界 |
| --- | --- | --- |
| 默认单卡权重卸载 | `--dit-layerwise-offload --text-encoder-cpu-offload --vae-cpu-offload` | DiT 循环预取，T5 FSDP，VAE 整组件搬运 |
| DiT 常驻，保留 T5/VAE 卸载 | `--no-dit-layerwise-offload --no-dit-cpu-offload` | 适用于 CFG/SP、完整 DiT 编译、INT8 与 NCCL 的相应支持路径 |
| 全驻留对照 | 上一行再加 `--no-text-encoder-cpu-offload --no-vae-cpu-offload` | 需要更多权重显存 |
| DiT 整组件卸载 | `--no-dit-layerwise-offload --dit-cpu-offload` | 不与逐层卸载同时开启 |
| 降低 VAE 激活峰值 | `--vae-tiling` | 近似路径，默认关闭；需独立检查画面与接缝 |
| 24 GiB 显存预算 | `--vae-low-memory` | VAE 归一化与卷积按算子分批，保留完整时空邻域；搭配默认权重卸载，见[运行与验证](memory24_20260930.md) |

`--dit-offload-prefetch-size 0` 表示预取一层；没有旧版字节预算参数。
卸载仅减少权重驻留，不能消除输入、激活或 VAE 峰值。单卡逐层卸载支持局部 FFN 编译、TeaCache
和 INT8/FP8；拒绝 CacheDiT 和多卡逐层卸载；多卡 VAE 要关闭所有组件卸载。
量化在执行 GPU 上逐层转换后放回 CPU，再注册卸载 hooks，避免 CPU/GPU 舍入改变量化权重。

`--cuda-memory-limit-gib 22` 限制每个进程执行设备上的 PyTorch 分配器，并在组件交接时释放闲置缓存。
NCCL 常驻 CFG/Ulysses 在此配置下将共用主卡的 rank 0 权重暂存 CPU，只在去噪窗口内搬入 GPU；
其他 rank 保持常驻。该参数不是整卡硬上限，不包含外部 CUDA/NCCL 分配；需要累加同卡所有任务进程的
NVML 占用判断是否满足 24 GiB。当前 L40S 收敛与验证状态见[本轮记录](l40s_completion_plan.md)。
实现、统计口径与历史实现区别见 [SGLang 内存管理](sglang_memory.md)。

<a id="single-gpu"></a>
## 注意力、融合与编译

| 功能 | 参数 | 使用边界 |
| --- | --- | --- |
| 注意力 | `--attention-backend sdpa / flash_attn / sage_attn / sage_fp8 / auto` | `auto` 按 SageAttention → FlashAttention → SDPA 探测；显式后端不可用时报错 |
| 算子融合 | `--operator-fusion-backend auto` | 默认 QK RoPE、AdaLN；不支持的布局回退。强制 `triton` 会检查契约 |
| FFN 编译 | `--enable-torch-compile --torch-compile-scope ffn` | 默认编译范围；可与单卡逐层卸载、peer CFG/SP、残差缓存组合 |
| 完整 DiT 编译 | `--enable-torch-compile --torch-compile-scope transformer` | 仅 peer CFG1/2、SP1、常驻未量化 DiT、SDPA；关闭手工融合和全部 Transformer 缓存 |
| 辅助组件编译 | `--compile-components vae_decoder` 等 | 与 DiT 编译开关独立；T5 编译需关闭 T5 卸载，VAE 编译仅支持单卡 VAE |

统一依赖清单已包含 FlashAttention 与 SageAttention。`sage_fp8` 另要求指定扩展符号和 sm89 GPU；
默认 SageAttention 1.0.6 安装不能视为该实验后端的完整运行环境，A100 不支持此路径。
完整原始素材的最新验收未通过 Sage/FP8/快速融合组合及 CFG2×SP2 `sharded` 模式的平均质量门槛，
这些组合保留为实验选项，不作为通用推荐。以下 121 帧数据仅代表对应切片；
最新质量、显存与候选决策见 [L40S 验收记录](l40s_validation_20261004.md)。

L40S 的 121 帧对照已测得 `sage_fp8` 请求耗时减少约 12.1%，平均 SSIM 0.99173；
配置、适用范围及隔离扩展启用方式见 [Attention 量化验证](attention_quantization_20261003.md)。
后续 [量化组合与内核调优](quantization_optimization_20261003.md) 使请求再减少约 1%，
但质量误差增加，未作为默认推荐；INT8 升维内核保留等价分块优化。

允许近似计算时，单卡 L40S 可显式选择
`--operator-fusion-backend triton --operator-fusion-ops qk_rmsnorm_rope_fast`，
将完整 Q/K RMSNorm 与 RoPE 合成一个 kernel。与 Sage FP8、静态 FP8 FFN 组合后，
121 帧同卡测试的请求耗时从 SDPA 对照 244.52 s 降至 166.76 s（-31.8%，不含加载/预热），
平均 SSIM 0.98758；新增融合相对仅 Sage 组合再降 21.6%。这是显式近似选项，
不与原 `qk_rmsnorm_rope` 同选；配置、两次候选测量与质量边界见
[Attention 性能优化](attention_optimization_20261004.md)。

在该组合上将算子列表扩展为
`qk_rmsnorm_rope_fast,rmsnorm_adaln_fast,gated_residual`，可进一步融合完整 AdaLN 与门控残差。
后续同卡对照从 167.69 s 降至 152.48 s（再降 9.1%，候选两次中位数），峰值显存不变，
相对 BF16 平均 SSIM 0.98758。完整 AdaLN 允许归约误差，需显式选择；
测量、投影筛选与适用范围见 [AdaLN 与残差优化](adaln_optimization_20261004.md)。

FFN 默认保留原生 Linear 边界；`MGERASE_COMPILE_LINEAR_BACKEND=inductor` 显式允许改变 Linear 数值路径。
完整 DiT 编译直接使用 Inductor Linear，要求 `fullgraph=True`，失败会抛出异常。
编译均关闭 CUDA Graph，首次调用/新形状的编译成本计入请求；一次视频未必能摊薄成本。
详细参数与日志字段见 [CLI 编译说明](cli.md#完整-dit-编译实验性)。

<a id="cache"></a>
## TeaCache、CacheDiT 与文本投影缓存

两种残差缓存互斥，默认 `off`；阈值均为 `0.3`、预热 4 步、末步保护 1 步、最多连续复用 1 步。
CacheDiT 默认前 1 / 后 0 个 block 实算。文本投影缓存独立配置，auto 在残差缓存开启时也会开启。
缓存按窗口和 CFG 分支隔离，不跨窗口沿用残差；完整 DiT 编译要关闭全部 Transformer 缓存。
DP worker 使用独立会话与缓存，仍须满足各自卸载、编译和拓扑限制。
NCCL 常驻 CFG/Ulysses 可显式开启 `--cache-text-projections`，复用文本投影与 cross-attention K/V；
默认 auto 在残差缓存关闭时仍不启用。NCCL 常驻 CFG/Ulysses 支持残差缓存及可选局部探针，已完成单素材逐帧质量筛选（见 [报告](nccl_cache_quality_20261002.md)）；与 TP/Ring/FSDP 的组合不支持。
[五组交替 A/B](nccl_text_cache_profile_20261002.md)输出文件一致，但没有证明稳定提速；
本例每 rank 保留约 29.5 MiB 缓存张量，不作为默认速度推荐。

单卡默认卸载配置可追加：

```bash
--transformer-cache-mode teacache --teacache-threshold 0.3
```

CacheDiT 需关闭 DiT 逐层卸载；隔离残差缓存收益时同时关闭文本投影缓存：

```bash
--no-dit-layerwise-offload --no-dit-cpu-offload \
--transformer-cache-mode cache_dit --cache-dit-residual-diff-threshold 0.3 \
--no-cache-text-projections
```

阈值是探针相对变化量，不是擦除误差百分比。默认 `global` 判定采用全局变化；
显式 `--cache-probe-metric mask_frame_max` 增加逐帧、mask 内与边缘保护，降低小目标被全局平均稀释的风险。
命中率和 SSIM 不能替代完整擦除检查。
[2026-09-29 检查](single_gpu_cache_audit_20260929.md)的样片在默认阈值下均复用 45% 分支步，
已受连续复用上限约束，继续提高阈值不会提高该样片命中率。
单次完整请求为无缓存 246.282 s、TeaCache 146.685 s、CacheDiT 150.558 s；
配置和逐帧视觉检查范围见原记录，不能推断为稳定排名或所有素材的质量保证。

<a id="parallel"></a>
## 多卡执行

| 路径 | 入口 / 参数 | 用途与限制 |
| --- | --- | --- |
| peer（默认） | `--cfg-degree 2` 或 `--sp-degree 2` | 单任务 CFG/SP；常驻 DiT，可使用支持的局部编译与缓存组合 |
| NCCL DiT 进程池 | `--dit-parallel-backend nccl` | CFG、Ulysses、Ring/USP、TP、FSDP/HSDP；BF16、SDPA，关闭编译、量化；常驻 CFG/Ulysses 可用文本/残差缓存及限定算子融合 |
| DP dispatcher | `entrypoints.cli.erase_parallel --dp-degree N` | 将独立视频分配给不同 GPU 组；每组使用独立请求缓存，遵守各自拓扑限制 |
| VAE 并行 | `--vae-degree 2 / 4` | 未启用 tiling 时按高度分片、逐层交换边界；不能与组件卸载或 VAE 编译组合 |

VAE 空间并行保留完整时间上下文，在局部分片上执行卷积、归一化和下采样。
BF16 卷积形状变化可能改变舍入结果；双卡速度、显存和解码质量见
[VAE 空间并行验证](vae_spatial_validation_20260929.md)。开启 `--vae-tiling` 仍使用独立的重叠 tile 路径。
完整视频的[端到端验证](vae_e2e_validation_20260930.md)中，主卡 allocated 峰值降低 24.7%，
但整片 SSIM 为 0.98313，未达到 0.99；独立解码器的一致性不能代表经过 DiT 后的整片结果。

NCCL 的 `sp_degree = ulysses_degree × ring_degree`；TP 与 FSDP 不同时管理同一权重。
使用 FSDP/HSDP 时以 `--dit-fsdp-shard-degree` / `--dit-fsdp-replicate-degree` 配置，
旧的 `ServerArgs.use_fsdp_inference` 不是此路径的入口。
`ring_attention_mode=online` 和 `tp_linear_mode=sharded` 的历史候选未通过整片 0.985；
可复用的 reference / streaming / aligned 组合以[逐策略验收表](distributed_parallel_20260928.md#验收状态)为准。

DP 的 GPU 数必须等于 `dp_degree × max(CFG×SP×TP, FSDP_shard×FSDP_replicate, VAE, legacy_CFG)`，
每个 worker 接收不相交的可见设备；DP 度数不能超过任务数，输出路径必须互不相同。
NCCL 单任务与 DP 的完整命令见 [CLI](cli.md#加速与多卡)。

[2026-09-29 边界优化](nccl_boundary_optimization_20260929.md)由 worker 合并 CFG，只返回一个预测张量，
减少 50% 返回张量逻辑字节；同条件 Ulysses2 输出一致，但请求耗时范围重叠，尚未证明稳定端到端提速。
父进程仍拥有 scheduler、RNG 与 T5/VAE，输入/输出默认经过 CPU IPC。
可用 `MGERASE_DIT_BOUNDARY_TRANSPORT=cuda_ipc` 显式测试 GPU 边界，仍保留接收端拷贝与同步；
适用范围和实测见[边界传输记录](cuda_ipc_boundary_20261003.md)。
输入 Ulysses head 分块的[重叠候选](ulysses_overlap_20261003.md)虽有真实 kernel 重叠，
当前 L40S 整片反而变慢，`MGERASE_ULYSSES_HEAD_CHUNKS` 默认保持 1。

[L40S 单窗口对照](window_optimization_20261002.md)固定 121 帧、各测五次：
双卡 CFG2 比 SP2 纯推理耗时减少 21.5%，四卡 CFG2×SP2 比 SP4 减少 7.3%。
在相同卡数下可优先比较 CFG 与 SP 的组合。
`MGERASE_NCCL_PACKING=packed` 显式启用减少打包拷贝的实验路径，原路径仍为默认；
纯 SP 本轮有小幅收益，CFG2×SP2 范围重叠，不能视为通用加速。

<a id="quantization"></a>
## 实验性量化

`--transformer-quantization int8_w8a8_native --quantization-scope blocks` 将选定 DiT Linear
转换为 INT8 W8A8，也可选择 `ffn` 范围。`fp8_w8a8_native` 使用 E4M3 W8A8，
权重按输出通道、激活按 token 缩放；要求 CUDA capability ≥8.9，INT8 要求 ≥8.0。
单卡支持 DiT 整组件或逐层卸载；T5/VAE 卸载可保留。
SP>1 时关闭手工融合；完整 DiT 编译和 NCCL 路径不支持该量化。
量化节省权重不保证降低完整请求峰值或加速，转换耗时与实际调用计数需单列；
INT8 对 token 数 ≥1024、输出宽度 ≥2×输入宽度的扩展层使用融合 GEMM，减少 INT32 中间张量。
`ffn` / `blocks` 范围自动融合 tanh GELU 与降维层激活量化，保持原 INT8 输出舍入；
每层增加独立 128 KiB 查表 buffer。实测与适用条件见 [GELU 融合](quantization_gelu_fusion_20261003.md)。
`fp8_w8a8_native` 保留原生 GEMM、FP32 输出和单独缩放。
新增实验模式 `fp8_w8a8_tensorwise`：权重整张量缩放一次，激活每次动态整张量缩放，
GEMM 内完成缩放和 bias，直接输出 BF16，避免 FP32 中间张量。
两种动态 FP8 模式均关闭 fast accumulation；默认仍为未量化 BF16。
整张量缩放改变数值误差，需要单独验证素材，优先测试 `--quantization-scope ffn`。

速度优先实验模式 `fp8_w8a8_static` 固定激活 scale=1/8（饱和范围 ±56），省去每次 amax
归约并开启快速累加；FP8 FFN 自动融合 GELU 与激活量化。静态范围不会随素材校准，
速度、质量与使用方式见 [FP8 性能优化](fp8_optimization_20261004.md)。

不能将 Linear 局部加速当作完整视频加速；对照见 [L40S INT8/FP8](quantization_20261003.md)
和 [FP8 执行路径修正](fp8_tensorwise_20261003.md)。

<a id="acceptance"></a>
## 质量与性能验证

固定输入、权重、提示词、seed、采样和窗口配置，与未优化输出比较。帧数、尺寸和帧率必须一致。
RGB 范围为 0–255，SSIM 使用 11×11 Gaussian 窗口、sigma=1.5、reflect 边界；
报告整段均值、最低帧、MSE、MAE 和时序误差，并查看擦除区域、边缘及残影。

各轮验收目标不同：2026-09-27 的单卡近似筛选使用整段 SSIM ≥0.98，
2026-09-28/29 的 NCCL 对齐使用 ≥0.985，2026-09-29 的缓存检查关注完整擦除而未设 SSIM 门槛。
这些门槛是各轮实验条件，不是运行时强制约束；换素材需重新验收。

判断稳定性能至少五次同条件重复，加载、编译/预热与稳态请求分开记录，报告中位数、波动和其他进程占用。
allocated/reserved 是本进程 PyTorch 指标；多卡时分别记录 owner 与各 rank，不能当作整卡占用。
详细操作见[测量与验证](validation.md)。

2026-09-22/23 的旧卸载与字节预算已移除；历史结果保存在
[初始优化验证](optimization_validation_20260922.md)、[缓存阈值验证](cache_threshold_03_validation_20260922.md)
和[旧组合验收](composable_acceleration_validation_20260923.md)。复跑历史脚本需要对应代码快照与环境。
`results/` 中的日志、脚本和视频是本机实验产物，不随 Git 分发。

2026-10-02 的[内存与卸载验证](memory_optimization_20261002.md)采用 121 帧单窗口：
VAE 阶段驻留/CPU 权重复用减少权重传输并降低 owner 峰值 allocated，另修复连续请求的上下文循环引用。
耗时、allocated/reserved、CPU RSS 和分配器限额分别报告。

[帧转换与输出缓存优化](frame_copy_optimization_20261002.md)进一步减少 CPU 整窗临时副本，
明确区分局部转换内存、阶段末 RSS 和纯模型推理耗时；保留 Python 默认张量输出契约。

本轮同条件五组配对：[融合与 direct 打包](fusion_memory_optimization_20261002.md)在四卡 CFG2×SP2 上去噪降低 13.5%，输出文件完全一致。
