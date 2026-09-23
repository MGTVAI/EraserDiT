# 可组合加速方案（设计与实施记录）

目标是在显存预算和质量要求内组合：组件/DiT 卸载、单卡 attention 与编译、
TeaCache 或 cache_dit、CFG 与 sequence parallel。本文保留开发路线；最新已实现范围与测试结果见末尾的实施更新。
USP 在本文指 Ulysses + Ring 的混合序列并行；FA 指 FlashAttention，Sage 指 SageAttention。

## 开发前的实现与边界

- `dynamic_offload`：CPU 初始化、阶段组件卸载、DiT pinned CPU 权重和受预算约束的异步预取；
  与 TeaCache/cache_dit/文本投影缓存已做小模型 GPU 组合验证。
- compile 在 `nodes/stages/denoising.py` 包裹整个 Transformer，默认
  `max-autotune-no-cudagraphs`。卸载、残差缓存、文本缓存与该路径存在显式组合限制。
- `models/adapters/eraserdit/mesh.py` 是单进程多线程和 peer-copy/barrier 实现的 CFG/Ulysses
  路径，每 rank 完整模型副本；按窗口 deepcopy，要求 fullgpu、无 compile、无缓存。
  `reference` 线性模式通过补回全长行保持 GEMM 形状，不是充分的计算/内存分片。
- 单卡 VAE tiling 也走 mesh 校验，目前与 dynamic_offload 不兼容。
- 强制 Triton 算子融合与 compile 互斥；auto 融合遇 compile 时关闭手工融合。
- 本地 SGLang 也并非全部可叠加：`server_args.py` 明确限制 cache-dit 与 layerwise offload，
  并将 FSDP 与该卸载路径分开。其 SP 配置满足 `sp_degree = ulysses_degree * ring_degree`。
  应参考其分层设计和实现，逐个验证本模型的组合。

## 总体结构

```text
输入/窗口调度
  → 组件阶段调度：Text Encoder / VAE Encoder / DiT / VAE Decoder
  → 缓存决策：off / TeaCache / cache_dit
  → 并行协调：CFG 分支组 × SP 组（Ulysses × Ring）
  → 权重准备：rank-local resident / layerwise manager
  → 张量计算：eager / block compile + attention backend
  → 释放、收集输出、更新缓存和执行统计
```

Python 状态、缓存命中判断、参数搬运、stream/event 和取消逻辑在编译区外。
GPU 权重恢复到正确 shape/stride 后，才进入编译计算区。每组只允许一个权重管理者。

保留 eager/fullgpu/SDPA 参考路径。能力检查应根据实际模块与硬件判断，替代简单的
“所有并行都必须 fullgpu”规则；显式请求不支持的组合应说明原因，不静默撤销用户设置。

## 内存策略

1. 显存充足：去噪期间 DiT 整体驻留，Text/VAE 按阶段卸载；避免每一步反复搬运。
2. 权重空间不足：每 rank 使用独立 layerwise manager、copy stream、event 和字节预算。
3. 混合常驻：后续允许一部分 blocks 常驻，其余预取；固定存储池按测量收益引入。
4. VAE 激活不足：单卡空间 tiling，与上述权重策略独立配置，不强制创建多卡 mesh。

预取/权重预算仅约束受管权重。每 rank 还需统计激活、workspace、缓存、通信缓冲和
allocator reserved；主机预算按所有 rank 的 pinned 副本合计。SP/CFG 本身不分片权重。
保留 CPU 初始化，禁止为创建副本先将整个模型搬到 GPU。

## compile 与单卡算子

编译粒度从整个 Transformer 改为 block 的纯张量计算区域：

```text
ensure block weights → wait ready event → compiled compute → retire weights
```

- 第一版禁用 CUDA Graph 捕获，先处理 shape、stride、分支和权重存储变化造成的重编译。
  CUDA Graph 需要额外验证稳定地址和生命周期，不能当作 compile 的前提或自动附赠能力。
- 按窗口形状、dtype、attention 后端、SP 布局建立有限的编译签名，报告图断点/重编译数。
- 编译和预热显存单列，防止模型推理能运行但 autotune 初始化先 OOM。
- 文本投影/KV cache 查询和写回留在图外；编译函数接收张量，避免每步读取 Python 字典。
- 内层优先让 Inductor 处理融合。手工 Triton 融合作为另一条可选路径；需要组合时再将
  明确有收益的内核纳入稳定调用接口，不同时打开两套未经验证的融合。
- SDPA 保留参考；FA、Sage 按 dtype、head size、GPU 架构和安装内核探测。
  当前 A100 不支持仓库的 sm89 专用 sage_fp8 路径。attention 后端通常择一，不叠加。
- Sage 等近似内核、FA/compile 的浮点顺序变化都应验证完整视频；不能统一承诺 bitwise 一致。

## 缓存与并行决策

同一次 DiT 执行选 TeaCache 或 cache_dit，不同时叠加两套残差缓存。
文本投影缓存可以独立组合；每次命中都应减少实际计算和无用 H2D，不能只改变表面计数。

- CFG 正/负分支分别管理缓存，不共享残差、阈值历史或预测器状态。
- 同一个 SP 组必须统一是否计算、执行区间和 collective 顺序。
  按算法定义在组内聚合误差的 sum/count 等统计，再广播决策；不能简单平均各 rank 的局部比值。
- 缓存键包括请求、窗口、CFG 分支、模型版本、dtype、并行布局和形状。
  SP 缓存残差保持 shard-local；布局改变时清空。
- CFG 不同分支在独立 SP 通信组内可以分别决策，但 CFG 输出合并必须有明确的汇合点。
- layerwise 只预取实际执行区间；探针所需的小权重驻留，命中跳层时不搬运被跳过的 blocks。
- warmup、最后步保护、连续跳步限制和异常后缓存清理继续保留。

## 并行模型生命周期

先用现有线程后端验证 rank-local 卸载与 CFG 的数值语义；后续统一为持久化 rank worker。
目标通信后端是一 GPU 一进程、显式 NCCL process groups；不把 Python barrier/peer-copy
硬塞进 compile，也不在每个窗口 deepcopy 已挂载卸载器或编译器的模型。

每个 rank 从 CPU checkpoint 建立独立模型对象，再注册各自的卸载与编译执行器。
第一版可各自持有 CPU pinned 权重，但必须报告总主机内存；跨进程共享 pinned 存储后续单独设计。
Text/VAE 默认在指定 owner 上执行，编码条件按布局分发；scheduler、采样随机数与输出写入
由指定 owner 管理，避免重复采样改变结果。

并行关系（设计，不含 TP/FSDP）：

```text
SP = Ulysses × Ring
每个请求的 DiT GPU 数 = CFG × SP
总 GPU 数 = DP × CFG × SP
```

VAE 可在阶段切换后复用同一设备池，不在上述公式中另乘 VAE degree。
多请求 DP 独立于单请求延迟优化；增加 DP 主要提高吞吐，不保证单请求更快。

- CFG=2：并行正/负分支，优先验证；每卡权重没有自动减半。
- Ulysses SP：分片序列，采用与 head 布局匹配的 all-to-all。
  保留 reference 数值路径，另测真正 sharded GEMM；两者的速度与输出差异分开报告。
- Ring/USP：在 Ulysses 基础上增加环形 KV 通信和稳定的分块 softmax 合并。
  验证 RoPE 索引、mask、padding、非整除长度及 overlap；不是添加一个配置开关就完成。
- compile 先覆盖通信前后的本地张量计算；collectives 初期放在编译区外。
- 加载失败/OOM/取消需要传播至整个通信组，避免部分 rank 进入下一次 collective 后死锁。
- H2D、NCCL/P2P 和 attention 会竞争带宽，预取越深不一定越快。

## 目标兼容矩阵

| 组合 | 开发前状态 | 目标路径 |
|---|---|---|
| layerwise + TeaCache/cache_dit | 已有小模型组合验证 | 扩展完整视频与重复请求验收 |
| layerwise + FA/Sage | 无卸载专属禁止，完整组合未验收 | 后端探测 + 完整质量/性能验证 |
| layerwise + compile | 拒绝 | 编译 block 计算，调度在图外 |
| cache + compile | 拒绝整模 compile | 缓存决策在图外，编译实际计算块 |
| layerwise + CFG | 拒绝 | 每 rank 独立 manager，持久化副本 |
| layerwise + SP | 拒绝 | 分片执行与权重驻留分离 |
| cache + CFG/SP | 拒绝 | 分支隔离、SP 组内统一决策 |
| compile + CFG/SP | 拒绝 | 各 rank 编译本地计算，通信先留图外 |
| offload + 单卡 VAE tiling | 被 mesh 校验拒绝 | 独立 VAE 策略，串行空间块 |
| USP | 当前 EraserDiT 路径未实现完整 Ring 混合 | 后续接入 Ulysses × Ring |
| layerwise + FSDP | 拒绝 | 先维持互斥，不让两套系统同时管理参数 |

## 场景组合（目标，不是当前可直接使用的命令）

| 场景 | 候选配置 |
|---|---|
| 单卡显存充足、优先数值接近基线 | DiT 去噪驻留 + Text/VAE 阶段卸载 + SDPA/FA + block compile + cache off |
| 单卡显存紧张 | layerwise + block compile + SDPA/FA；必要时 VAE tiling |
| 单卡允许近似 | 上述策略 + TeaCache 或 cache_dit；Sage 单独评估后再叠加 |
| 双卡单任务延迟 | CFG=2 + 各卡适量驻留/卸载 + FA + block compile；缓存第二步接入 |
| 双卡长序列/激活压力 | SP=2 + 各卡卸载 + FA；与 CFG=2 实测比较 |
| 四卡单任务 | CFG=2 × SP=2，或 SP=4；先 Ulysses，Ring/USP 后接 |
| 多任务吞吐 | 优先比较 DP 独立副本与 CFG/SP，按吞吐、尾延迟和主机内存选型 |

开启更多选项不保证更快；缓存缩短计算后可能削弱 H2D 重叠，SP 缩短每卡计算后也可能
暴露传输瓶颈。不得把单项加速比直接相乘。

## 实施与验收顺序

1. **拆分执行边界**：提取 block tensor compute、独立缓存控制、独立 VAE tiling 策略；
   明确权重 residency 策略和能力检查，保留基线。
2. **单卡组合**：先 component-only + compile，再 layerwise + block compile，再
   TeaCache/cache_dit + block compile + layerwise。验证 FA/Sage 与手工融合的可选路径。
3. **CFG 组合**：持久化两卡模型、各卡 manager、各分支缓存、各卡编译；完善组级取消与回滚。
4. **SP 组合**：明确通信接口与进程组，Ulysses + 全局缓存决策 + rank-local 卸载；
   编译本地计算，分别验证 reference/sharded 线性模式。
5. **USP 与调优**：Ring、CFG×SP、拓扑和预取调优，再评估固定 GPU buffer、部分常驻。

每一步先用确定性小模型，再测完整双窗口视频。以同 backend/dtype 的 eager 输出作对应参考，
另外报告相对原始 SDPA 全驻留基线的差异；近似缓存/Sage/tiling 使用单独的质量验收标准。
显存统计覆盖初始化、compile/autotune、预热、文本、VAE 编码、去噪、解码和关闭。
每 rank 记录 allocated/reserved、受管权重、缓存、通信缓冲和 H2D；主机记录 RSS/pinned 总量。
包含重复请求、形状切换、非整除序列、缓存命中/失效、部分 H2D 失败、rank 失败和取消。
性能至少五次同配置测量，分离冷启动与稳态，报告端到端、去噪、吞吐和尾延迟。

第一批建议交付“单卡 block compile + 卸载 + 二选一缓存”，再接 CFG=2。
VAE 分块作为单独的显存需求分支，不要求所有速度优化都先改 VAE。

## 2026-09-23 实施更新

本文件前面的“当前”描述是开发前快照。最新实现与验收见
[可组合加速验收](composable_acceleration_validation_20260923.md)。
本轮限定 GPU 6、7，不启动四卡任务；按 SSIM≥0.985 / MSE≤36 / MAE≤6 筛选，
不要求逐像素一致。缓存与量化允许近似。优先比较真正 sharded SP 与 CFG，
只有未达标时才采用较保守的计算路径。四卡混合 USP 和 NCCL 进程架构不在本轮实测范围。
