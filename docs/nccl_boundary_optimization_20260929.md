# NCCL 输出汇聚与进程边界优化

本轮继续使用 SDPA、BF16、常驻 DiT、关闭编译/融合/缓存，以整片 RGB SSIM ≥0.985 验收。
参考前一轮 [NCCL 并行验收](distributed_parallel_20260928.md)，不导入 `sglang`。

## 改动

- 最终 SP 输出由 all-gather 改为只在组内首 rank gather。不改变 attention 或 TP 中间激活的通信。
- TP/FSDP 额外副本仍完成所有模型计算，只由 TP 坐标 0、额外副本坐标 0 参与最终输出汇聚。
- CFG2 只在 SP 坐标 0 的必要 CFG 组汇聚正负分支，并仅在全局 rank 0 返回结果。
- NCCL 去噪在 worker 内执行 `negative + scale * (positive - negative)`，保持 FP32 运算顺序。
  每步通过 CPU IPC 返回一个预测张量，原来是两个，因此返回预测的逻辑字节数减少 50%。
  原有返回正负分支的 `predict` 接口继续可用。
- 新增每窗口边界墙钟计时、逻辑张量字节数和各 worker 的 CUDA event 区间计时。

输入仍经过 CPU tensor IPC，scheduler、随机数和窗口依赖仍由父进程管理；本轮不属于 GPU 零拷贝。
返回字节数减少 50% 不等于 IPC 总流量或端到端耗时减少 50%。

## 计时解释

`parallel_history[].dit_parallel` 新增：

| 字段 | 含义 |
| --- | --- |
| `output_assembly` | `owner_only`，仅汇聚必要的最终输出 |
| `boundary_tensor_bytes.input/output` | 本窗口跨进程输入/返回张量的逻辑字节数，不含元数据和 IPC 实现内部开销 |
| `boundary_wall_seconds.input_to_cpu` | 输入转 CPU 的墙钟时间，可能包含等待此前 GPU 工作 |
| `boundary_wall_seconds.worker_roundtrip` | 发送命令、worker 广播/推理/汇聚、返回 CPU 结果及 IPC 等待的总墙钟时间 |
| `boundary_wall_seconds.output_to_device` | 将 CPU 预测搬回 owner 设备的墙钟时间 |
| `rank_reports[].window_gpu_seconds.input_broadcast` | 输入广播前后 CUDA event 的区间时间，包含区间内的主机调度空隙 |
| `rank_reports[].window_gpu_seconds.forward_and_output_gather` | forward、最终输出汇聚，以及 rank 0 CFG 合并的 CUDA event 区间时间 |

各 rank 时间相互重叠，不能相加为请求总耗时；worker 区间与 `worker_roundtrip` 也有包含关系。
event 在窗口 report 时统一读取，不为每个计时区间增加设备同步。

## 正确性检查

- Gloo 九种拓扑：CFG、Ulysses、Ring、TP、USP、CFG×SP、TP×SP、CFG×TP、额外副本。
  同拓扑比较 all-gather 与 owner-only 路径，要求逐元素一致；覆盖不等长分片和非零目的 rank。
- NCCL：双卡 CFG/Ulysses/Ring/TP、SP+FSDP、纯 FSDP 额外副本；双卡/四卡 aligned TP。
- 双卡 CFG 与四卡 CFG×SP 进程池：guidance 0/1/7.5、两种窗口形状、跨窗口常驻复用、
  返回字节数、父进程 singleton group 保持不变、杀死 worker 后的错误传播和清理。
- 新 CFG 合并必须与同拓扑原正负分支的 FP32 表达式逐元素一致。小模型 dense 与 SP 的对比沿用
  既有 BF16 容差，因为 SP 本身改变 GEMM 行数；这不替代新旧同拓扑的严格对比和整片质量验收。

日志和冻结的修改前源码保存在 `results/nccl_boundary_20260929/`。

## 整片对照

重放入口：`results/nccl_boundary_20260929/run_performance.py`，使用其 Python 环境执行。
固定同一组 GPU，修改前后均使用 Ulysses2、`sp_linear_mode=sharded`，各三次完整请求；
首个单列，后两个报告中位数和范围。每个输出独立检查整片 SSIM；加载不计入请求时间。
`performance_manifest.json` 记录命令和源码 SHA256，`performance_gpu_samples.jsonl` 记录共享 GPU 占用。
冻结源码仅由实验脚本通过标准 import finder 加载，运行时代码不依赖 `results/`。

最初使用 GPU 3、7，期间 GPU 7 新增约 55 GB 的外部任务，因此停止该组对照，
其记录归档到 `interrupted_gpus_3_7/`，不用于计算收益。对照改用 GPU 3、6 后，
进一步审计发现 GPU 6 的既有进程从约 642 MiB 短暂增长至 6066 MiB 后退出，因此第一次
`perf_baseline` 也不用于性能归因。优化版完成后补测 `perf_baseline_clean`，单独保存命令与资源采样。
共享主机、动态频率和有限样本仍是测量边界。

最终对照使用 `perf_baseline_clean.json` 与 `perf_optimized.json`，汇总为 `summary.json`；
资源审计为 `resource_audit.json`。两组都只有 owner 和自有 DiT workers 出现在测量 GPU 上，
没有观察到新增外部进程。采样不能代替独占资源分配，也不能排除共享 CPU/IO 的干扰。

环境为 A100 80GB、PyTorch 2.6.0+cu126；样片 `data/113000356.mp4`，1920×1080、145 帧，
seed 42、50 配置步数、strength 0.8、infer_len 121、overlap 9、紧凑尾窗；参考
`results/performance_20260927/fused50.mp4`。本轮整片只比较 Ulysses2，其他拓扑覆盖上述小模型回归。

| 同条件 Ulysses2 | 修改前（补测） | 修改后 |
| --- | ---: | ---: |
| 首个完整请求（不含加载） | 173.95 s | 170.76 s |
| 后两次完整请求中位数 | 174.14 s | 170.20 s |
| 后两次完整请求范围 | 171.83–176.45 s | 167.89–172.51 s |
| 后两次去噪中位数 | 122.331 s | 122.192 s |
| 整片返回预测张量 | 1673.44 MiB | 836.72 MiB |
| worker 最大 peak allocated | 5.360 GiB | 5.360 GiB |
| worker 最大 peak reserved | 10.049 GiB | 10.068 GiB |
| owner 最大 peak allocated | 30.446 GiB | 30.446 GiB |

返回字节数减少 **50%**。六个有效对照输出的整片 RGB SSIM 均为 **0.992833050**，
视频文件 SHA256 全部为 `4c1bc22b65b94ad4446cd7aa9673f7a300f1ba17da973227c7eddd4b54605617`。

完整请求中位数的观测差为 2.26%，但去噪差仅 **0.139 秒（0.11%）**，完整请求范围重叠，
差值主要来自非去噪。因此**本轮未证明稳定的端到端提速**，不将 2.26% 宣称为通信优化收益。
同样不宣称整体显存降低：非返回 rank 的 allocated 略降，但所有 worker 的最大 allocated 不变，
最大 reserved 反而略升。保留此次减少输出通信和增加可观测性的改动，下一阶段按新计时推进。

CPU 专项 19 项（18 通过、1 条件跳过），Gloo 九拓扑矩阵通过；GPU 双卡专项 2 项通过，
aligned TP 双/四卡检查通过，四卡进程池专项通过。初次四卡进程池测试把 dense-vs-SP 也设为
零容差，暴露了既有 BF16 GEMM 行数差异；随后沿用原小模型 dense-vs-SP 容差，复跑通过。
新旧同拓扑输出和 CFG 合并的对比始终要求零容差。`git diff --check` 通过。

## 新计时给出的下一步方向

优化版三次请求的去噪为 122.15–122.35 秒，非去噪为 45.74–50.27 秒。
输入转 CPU 和预测回 GPU 合计仅 0.217–0.234 秒；rank 0 输入广播区间为 0.130–0.140 秒。
worker 往返约 121.78–121.95 秒，其中 forward/最终汇聚区间约 121.16–121.29 秒。
这些时间有包含关系，不可相加；GPU 区间也不是纯 kernel 时间。

据此，下一步先细分非去噪中的读取、颜色校正、窗口提交和编码，以及 DiT 内部的计算/通信占比。
当前结果不支持仅因为存在 CPU IPC 就优先投入 scheduler 迁移或 CUDA IPC 的较大改造。
