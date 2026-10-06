# SP2 / SP4 优化路线

更新：2026-10-06。目标是在 BF16、SDPA、当前 reference 输出一致和单卡任务显存
不超过 24 GiB 的约束下，继续降低多卡 DiT 延迟。下述候选尚未完成，不预设加速收益。

## 当前成果

当前优化组合采用纯 Ulysses SP2 / SP4、CFG1，已完成：

- QKV 合并与 direct 打包，按 head 分块并重叠输入通信、Attention 和输出通信。
- aligned 选择性 GEMM 保护：多数投影按本地 token 计算，敏感投影保留数值保护。
- 原生 RMSNorm 归约与逐元素融合，保留 BF16 舍入边界。
- 自动选择通信分块：SP2 长短序列均为四块；SP4 的 32640 token 为四块，10200 token 为两块。
- SP4 短序列 FFN 下投影保护从 10200 行缩到 3072 行，每卡有效 2550 行。

这些是显式启用的优化组合，不代表默认配置全部开启。自动分块和 aligned 优化受
L40S、PyTorch 2.6.0 / CUDA 12.6、模型及形状条件保护，不能直接外推到其他设备或拓扑。
DiT 在去噪窗口内驻留；与 owner/VAE 共卡的 rank 0 在窗口外仍采用
`shared_rank_idle_cpu`，尚未实现所有 rank 全程驻留。

最新 B1 同输入验收如下，均排除加载和预热，每项一次正式请求：

| 配置 | DiT 秒 | DiT 加速（B1=1） | 纯模型推理秒 | 纯推理加速（B1=1） | 请求秒 | 请求加速（B1=1） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 444.816 | 1.00× | 472.162 | 1.00× | 1809.253 | 1.00× |
| SP2 | 203.323 | 2.19× | 231.746 | 2.04× | 259.069 | 6.98× |
| SP4 | 121.539 | 3.66× | 150.061 | 3.15× | 178.029 | 10.16× |

以上是相对原始 B1 的累计工程收益，包含实现、GPU 数量及请求处理路径变化，
不是最近一轮优化的独立收益。SP2 / SP4 最高单卡任务显存采样峰值为 21.527 / 21.613 GiB。
输出与当前 BF16 reference 逐字节一致；不声称与原始 B1 输出逐字节一致。
完整输入、环境、质量证据和复现命令见[最新验收](sp_finish_20261006.md)。

## 下一步：先更新性能剖析

旧 trace 早于 aligned GEMM、原生归约融合和自动分块，不能据此认定当前瓶颈。
先测当前优化组合的 SP2 / SP4 × 32640 / 10200 token，记录：

- GEMM、RMSNorm/RoPE、打包与接收重排、Attention、NCCL 的 GPU 时间线。
- CPU 发射、IPC、CUDA event 和同步等待，以及窗口边界的权重迁移。
- 各 rank 的关键路径、通信与计算重叠、临时分配和逐卡任务显存峰值。

沿用 `nccl_sequence.py` 的 input/output pack、all-to-all、unpack 和 head_concat 区域。
重叠的 NCCL 与计算耗时不能直接相加；性能结论使用关闭 profiler 的配对测量。
基准当前实现时须启用紧凑 FFN：`benchmark_dit` 中不带 `_compact` 的旧原生归约
variant 特意保留 10200 行保护，不能误当作当前生产路径。

## 候选优先级

| 顺序 | 优化方向 | 具体工作 | 主要验收边界 |
| --- | --- | --- | --- |
| 1 | Q/K 归一化、RoPE 与通信打包融合 | 保留原生平方归约和 inverse 计算，将后续归一化、权重乘法、RoPE 直接写入 destination-major 通信缓冲区，减少 Q/K 中间张量及读写 | 保留逐阶段 BF16 舍入；检查不同分块、各 rank 的布局与输出一致性 |
| 2 | 接收端重排与 head 拼接 | 检查 input/output unpack 的 `cat` 和最终 `stack`，尝试让通信缓冲区布局更接近消费者布局，减少完整张量拷贝 | 保留 rank/head 顺序、stream 依赖及缓冲区生命周期；必须测完整 forward |
| 3 | 进一步缩减受保护 GEMM | 研究固定兼容 GEMM 算法或其他精确路径，降低 SP4 短序列 2550→3072 的补零成本；按 profile 决定是否检查其他受保护投影 | 2688 行已失败，不能直接缩小；覆盖 28 层、四个 rank、真实去噪输入 |
| 4 | 降低调度与同步开销 | 根据 trace 复用缓冲区、减少重复 event/同步；仅在发射开销明显时筛选局部 CUDA Graph | 检查多流通信、动态长度切换、地址稳定性和额外显存；不默认捕获整个多卡图 |
| 5 | 共享 rank 0 全程驻留 | 测量窗口间重新加载权重的实际成本，再评估 VAE 阶段释放临时缓冲区等内存调整能否容纳 DiT | 当前峰值已约 21.6 GiB，不能直接假定再常驻权重仍满足 24 GiB；单独报告窗口边界收益 |

优先开展剖析和候选 1 / 2；候选 3 / 4 由关键路径证据决定投入顺序。
候选 5 主要改善窗口边界，不能把省下的迁移时间当作每个去噪步骤的收益。

源码入口：

- [Ulysses 通信与重排](../models/adapters/eraserdit/nccl_sequence.py)、[QKV 打包](../layers/attention/qkv_packing.py)。
- [原生归约融合](../layers/operator_fusion/triton/rmsnorm_native.py)、[Q/K 融合分派](../layers/operator_fusion/qk_rmsnorm_rope.py)。
- [GEMM 保护](../models/adapters/eraserdit/mesh.py)、[自动分块](../layers/attention/ulysses_policy.py)。
- [窗口驻留](../memory/backends/dit_idle_residency.py)、[执行器](../pipelines/runtime/dit_executor.py)。

本地 SGLang 对照源码位于 `/home/zhouhao6/VideoEdit/sglang`，可继续参考
`python/sglang/multimodal_gen/runtime/layers/usp.py` 和 `layernorm.py` 的通信流水线与融合边界。
迁移思路需重新验证本模型的数值、布局和调度条件，不能直接套用性能结论。

## 验收与停止条件

1. 每次只引入一个候选，先检查算子输出，再做真实权重、正负 CFG 分支及输出汇聚的完整 forward。
   覆盖 SP2 / SP4、长短两种长度，至少五组交替配对，记录原始耗时、实际策略和回退情况。
2. 有稳定收益后，复测 B1 同输入完整 145 帧请求及短序列 seed 42 / 7；检查同一进程池长短切换。
   完整输出须与对应当前 BF16 reference 逐字节一致，逐卡任务显存峰值须不超过 24 GiB。
3. 加速比统一报告原始 B1，另列本轮前后绝对耗时或延迟降幅。短序列没有对应 B1 基线，
   不编造 B1 加速比；145 帧 B1 的两个窗口均为 32640 token，短序列收益不能外推到该任务。
4. 分别报告 DiT、纯模型推理、整请求耗时，以及预热、重复次数、设备与源码版本。
   未通过质量或显存约束、只有微基准收益、完整 forward 回退的候选不接入默认优化组合。

已排除：残差加法与平方融合虽通过精确性检查，但多个主要形状变慢；SP4 短序列
FFN 的 2688 行保护产生数值差异。自动分块在长序列和 SP2 上没有新增稳定收益。
除非出现新的实现或剖析证据，不重复把这些候选作为已确认的加速空间。

历史依据：[选择性 GEMM](sp_aligned_20261005.md)、[原生归约融合](sp_native_rms_20261005.md)、
[分块与补零验收](sp_finish_20261006.md)。
