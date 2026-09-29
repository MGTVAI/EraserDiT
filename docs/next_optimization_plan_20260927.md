# 四卡之后的优化方案：按实测瓶颈安排

本方案在四卡验证后讨论，尚未实施的项目均不算已取得收益。
遵守项目约束：参考 SGLang 实现，不把 `sglang` 当作依赖导入。
源码基于本地 SGLang 提交 `f694186fd899e032b52388d73404d9d00afd5418`。
这里的 LTX 参考是 SGLang 的 **LTX-2**；本项目是擦除用途的 LTXVideo DiT，
不能照搬其音视频双流结构、权重或声称相同模型的性能差距。

## 先明确时间花在哪里

本轮四卡 CFG2×SP2、SDPA、sharded Linear，通信改动后：

| 阶段 | 秒 |
|---|---:|
| 去噪 | 57.761 |
| 条件 VAE 编码 | 8.129 |
| VAE 解码 | 4.175 |
| 文本编码 | 1.491 |
| 其余请求时间 | 24.594 |
| 请求合计 | 96.151 |

其余时间包含预处理、后处理、窗口管理、输出、部分初始化等，尚无足够细分计时，
**不能把它全部归因于颜色校正或通信**。模型加载单列，不在这 96.151 秒内。
不开残差缓存时，非去噪约 38.4 秒；启用缓存后其占比会继续增加。
四卡 TeaCache 的完整请求实测为 **59.611 / 58.810 秒**，去噪 **20.504 / 20.479 秒**，
非去噪 **39.107 / 38.332 秒**，约占请求的三分之二。
以首轮估算：只把去噪加速两倍，总时间约 **49.36 秒**；
只把非去噪加速两倍，总时间约 **40.06 秒**。这是 Amdahl 情景计算，不是实测或收益承诺。

实际代码已发现可优化的路径：
`models/adapters/eraserdit/postprocess.py` 先把 generated/style/mask 转 CPU，
`utils/colorfix_wmask.py` 又将特征搬到 CUDA，在 Python 双循环中逐帧、逐通道求均值与方差，
标量回写 CPU；最后 `window_postprocess.py` 把结果传回 CUDA。
这是明确存在的搬运和同步，但其独立耗时还需要打点确认。

## 建议顺序

| 优先级 | 工作 | 精度影响 | 推进依据 |
|---|---|---|---|
| P0 | 补齐阶段计时；GPU 颜色校正和更少帧搬运 | 可接近等价，先按 SSIM ≥0.98 | 非去噪已经约 38 秒，缓存不能加速它 |
| P1 | 扫描 TeaCache/CacheDiT 的连续复用上限和步数策略 | 有损，按擦除效果验收 | 已验证缓存有效，但当前复用上限限制收益 |
| P2 | 单进程线程通信迁移为常驻多进程 NCCL，合并 QKV | 主要是数值舍入差异 | 为更大 SP、更大模型、编译及吞吐建立基础 |
| P3 | 按真实 SP 形状选择 FA/Sage；扩大编译范围 | Sage 有损；编译可接近等价 | 单卡 FFN 编译和替换 attention 并未带来大幅端到端收益 |
| P4 | INT8 GEMM 与输出反量化融合，按层选择量化 | 有损，按擦除效果验收 | 当前原生 INT8 节省权重，但未独立提速 |
| P5 | 掩码驱动的局部推理、空掩码窗口跳过、稀疏注意力 | 结构性近似，风险较高 | 擦除任务通常只有局部区域需要重建，可能减少更多计算 |

P0 与 P1 优先做小范围验证。P2 是较大的执行架构改造，不能用“上 NCCL”代替收益证明。

## P0：先减少非去噪阶段

1. 给视频读取、mask/crop/resize、窗口准备、CPU/GPU 拷贝、颜色校正、commit、写视频分别计时。
   分开冷启动、模型加载、首窗口、尾窗口和服务稳态；记录主机内存与逐卡峰值。
2. 颜色校正在 GPU 上批量处理 `[frame, channel, pixel]`，FP32 求均值、方差和归一化。
   保留两次 uint8 截断位置及方差 correction；避免为了提速悄悄改变颜色逻辑。
   对二值 0/255 mask 可验证现有 `1-mask` 转 bool 后的等价全帧统计路径；
   对软 mask 或值为 1 的输入必须保留正确分支，不能无条件丢弃掩码。
3. 分块处理帧，复用输出缓冲；最终编码前再转 CPU uint8，避免整段 FP32 来回传输。
   当前代码可能在 postprocess 同时持有多个全分辨率副本，需同步验证显存峰值。
4. VAE 编码约 8.1 秒、解码约 4.2 秒：先减少传输和重复初始化，再比较原生 tiling、空间并行。
   时间分块必须处理因果状态与重叠，不把独立短片解码视为无损替代。

SGLang 的 LTX-2 VAE 提供空间/时间 tiling，可参考其上下文和拼接方式；它不等于
已经为本项目验证了四卡 VAE。当前本项目的多卡 VAE 与组件 offload 组合仍有限制。
参考：`runtime/models/vaes/ltx_2_vae.py` 的 `tiled_encode/decode`、`_temporal_tiled_encode/decode`。

## P1：提高缓存收益，同时约束人物残留和闪烁

当前 TeaCache 阈值 0.3、最多连续复用 3 步、预热 4 步、末尾保护 1 步；双卡原片
160 个逻辑分支步复用 108 个。命中率已经很高，**只增大阈值不一定再提速**。

建议逐项扫描，不直接叠加最激进组合：

- TeaCache：连续上限 3→4→5；预热先保留 4，末尾保护保留 1；再测试阈值 0.3/0.4。
- CacheDiT：前/后实算 block 从 1/1 比较 1/0，连续上限从 3 比较 4/5。
- 按噪声阶段调整复用频率：早期重建结构、末期恢复细节增加实算，中间阶段增加复用。
- 最后再比较一阶 Taylor 预测残差与简单复用；跨窗口清空状态，并保持 CFG 分支隔离。

SGLang `runtime/cache/cache_dit_integration.py` 的 `CacheDitConfig` 已表达
前/后 block、warmup、连续缓存上限、steps computation mask 和 TaylorSeer 配置，
可参考这些边界和生命周期，不能把其他模型的默认参数直接视为 EraserDiT 最优参数。
[cache-dit 官方实现](https://github.com/vipshop/cache-dit) 可用于核对接口和算法定义。
本项目现有 `cache_dit` 模式也不能据此宣称拥有上游包的全部特性。

验收除了全片 SSIM 诊断，更看擦除区域、边缘环带和相邻帧：人形/物体残留、拖影、
背景断裂、颜色跳动与窗口接缝。小 mask 的全片 SSIM 很容易被背景占比稀释。

## P2：参考 SGLang 改造多卡通信

当前四卡为同一 Python 进程中的四个线程，两组 CFG 各做 SP2；每张卡完整持有 DiT 权重。
每 block 的 self-attention 都要交换输入 QKV 和输出，每次有主机 barrier 与 GPU 等待。
本轮只优化大张量接收缓冲，仍然保留这些同步以保证源张量生命周期。

SGLang `runtime/layers/attention/layer.py:UlyssesAttention.forward` 合并 QKV 后做
head↔sequence all-to-all；`runtime/layers/usp.py` 使用 functional collectives；
`runtime/layers/attention/turbo_layer.py` 提供异步通信与专用 stream 的参考实现。
这些路径不是每个模型都自动启用的同一种实现。

建议实现独立的常驻 rank worker，显式建立 CFG 与 SP 两类 NCCL 组：

- 正/负分支各两卡，QKV 合并通信；保持主卡 scheduler 和按序窗口依赖。
- 缓存残差误差先做全局归约，所有 SP rank 使用相同决策，避免 collective 次序分叉。
- 支持尾窗口不同 token 数、异常广播、rank 失败退出、取消任务与缓存隔离。
- 用 GPU event 测通信与计算，不依赖 CPU launch 时间；评估预分配和双缓冲，
  通信/计算重叠需要实际数据依赖允许，不能简单设置 async 就宣称已重叠。

先以 CFG2×SP2 为基线，另外比较 CFG1×SP4。多任务场景再比较两组独立双卡 CFG：
这是吞吐方案，不能把吞吐提高当成单视频延迟降低。
迁移先以相同输出质量与异常处理为门槛，再做配对重复测量；不预设 NCCL 必然更快。

## P3：attention 与编译需要一起处理形状和执行边界

此前单卡 SDPA/FA/Sage2 的整段差距不大；FFN-only 编译的真实形状配对测试几乎没有收益。
重复开启同一编译开关不太可能出现新的大幅收益。

下一轮可改变的内容：

- 按长/短窗口、SP2 的 16 heads 分别测 SDPA、FA2、Sage2，计入通信、layout 转换和量化开销。
- 当前 SP Sage 路径为兼容旧版本全头均值，会额外汇聚 K；需先核查 Sage2 的平滑语义，
  避免重复平滑和无必要的整 K 通信。此项允许数值差异，但必须单独验收。
- 将编译范围从 FFN 扩到可独立编译的 block 子图，包含 normalization、调制、残差和 Linear。
  通信、缓存决策、offload hooks 留在明确的图边界；预热 32640/10200 tokens 对应分片形状。
- 常驻权重、固定 buffer 后再研究 CUDA Graph；实算步、缓存步分别捕获，
  动态分支和 collective 不能误放进同一固定执行图。

SGLang `runtime/pipelines_core/stages/denoising.py:_maybe_enable_torch_compile`
对 module 编译，默认 `max-autotune-no-cudagraphs`，并设置可用的计算/通信重排选项；
这与本项目当前 FFN-only 编译的覆盖范围不同。
[SageAttention 官方仓库](https://github.com/thu-ml/SageAttention) 明确区分 Ampere 路径
与 Blackwell 的 SageAttention3，A100 上不能照搬新硬件的 FP4/FP8 benchmark。

## P4：量化先解决内核瓶颈，再扩大覆盖

当前 INT8 流程：逐 token 量化 → 原生 INT8 GEMM → 完整 INT32 中间张量 → 独立反量化/bias kernel。
长窗口下，2048→2048 和 2048→8192 比 BF16 慢，8192→2048 才有收益。
因此先给 FFN down projection 做融合 INT8 GEMM+反量化+输出，减少中间显存读写；
再比较只量化该投影、整个 FFN、整个 block，不能把权重减半等同于推理时间减半。

SGLang `runtime/layers/quantization/nunchaku_linear.py` 展示了校准参数、低秩项与专用
低比特 GEMM 的组织方式。W4A4/SVDQuant 需要本模型的校准与转换，不是换个 dtype 即可。
优先保留输入/输出投影、norm、敏感 attention 层 BF16；再逐步扩大范围。
首先比较同缓存率下的去噪时间，避免把缓存或前后处理波动误记为量化收益。

## P5：允许更大精度变化时，减少需要计算的内容

这是高潜力但尚未验证的项目专用方向，不应当宣称 SGLang 已直接解决 EraserDiT 擦除：

- 时间窗口内取 mask 的运动联合包围盒，加上下文边距，对齐 VAE/DiT 网格后局部推理，
  输出贴回原分辨率。比较背景结构、相机运动与边界融合；不改变输出分辨率并不代表计算语义没变。
- 无有效 mask 的窗口比较复制源帧的快速路径。必须处理跨窗 raw tail、已有 overlap 和后续窗口依赖。
- 块稀疏 attention 优先保留 mask 及边缘邻域的密集连接，对远处背景减少连接；
  早期/末期与敏感层保留稠密注意力。

SGLang `runtime/layers/attention/backends/sparse_video_gen_2_attn.py` 提供聚类、选择率、
首层/首步稠密保护及 centroid cache 等结构参考，但其元数据、内核和适用模型需单独适配。
单卡 attention 的高倍加速不能直接换算为四卡加缓存后的端到端倍数。
减少采样步数或蒸馏则单独立项，不混入目前保持 50 配置步数的比较。

## 验收与收益判断

先保留本片作为回归，再补大 mask、快速运动、移动镜头、细纹理、多个窗口等不同素材。
近似缓存/量化沿用基本擦除验收，不强制 SSIM 对齐；工程改动先保持全片 SSIM ≥0.98，
更激进 ROI/稀疏算法单列候选，同时报告 mask 内、边缘和背景的变化与时间一致性。
任何候选都必须完整输出相同帧数、帧率和尺寸，记录逐卡 allocated/reserved 与实际后端。

收益用测量或上限说明，不承诺百分比：若某一配置总时间为 T、去噪为 D，
仅把去噪再加速一倍，最终也只是 `T-D/2`；非去噪不变时，不可能突破 `T-D`。
每个有效候选做至少三次交错顺序的完整请求比较，报告中位数与范围，加载与编译单列。
先测 P0 与 P1 的边际收益，再决定投入较大的 NCCL、全 block 编译或稀疏 attention 改造。

## SGLang 源码索引（固定版本）

- [LTX-2 DiT](https://github.com/sgl-project/sglang/blob/f694186fd899e032b52388d73404d9d00afd5418/python/sglang/multimodal_gen/runtime/models/dits/ltx_2.py)：模型边界、量化 Linear 和 SP 组织。
- [Ulysses attention](https://github.com/sgl-project/sglang/blob/f694186fd899e032b52388d73404d9d00afd5418/python/sglang/multimodal_gen/runtime/layers/attention/layer.py)：QKV 合并和两次 all-to-all。
- [异步通信参考](https://github.com/sgl-project/sglang/blob/f694186fd899e032b52388d73404d9d00afd5418/python/sglang/multimodal_gen/runtime/layers/attention/turbo_layer.py)：async collective、请求等待与专用 stream。
- [编译入口](https://github.com/sgl-project/sglang/blob/f694186fd899e032b52388d73404d9d00afd5418/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py)：module 编译与缓存初始化顺序。
- [CacheDiT 集成](https://github.com/sgl-project/sglang/blob/f694186fd899e032b52388d73404d9d00afd5418/python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py)：连续复用、步数策略、Taylor 校准。
- [LTX-2 VAE](https://github.com/sgl-project/sglang/blob/f694186fd899e032b52388d73404d9d00afd5418/python/sglang/multimodal_gen/runtime/models/vaes/ltx_2_vae.py)：空间和时间 tiling。
- [Nunchaku Linear](https://github.com/sgl-project/sglang/blob/f694186fd899e032b52388d73404d9d00afd5418/python/sglang/multimodal_gen/runtime/layers/quantization/nunchaku_linear.py)：校准、低秩项和专用低比特算子。
