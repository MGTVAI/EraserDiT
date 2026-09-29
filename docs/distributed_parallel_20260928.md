# SGLang 风格 DiT 多进程并行：实施与验收记录

用户要求：先完成并行策略，再做 **整片 RGB SSIM ≥0.985** 对齐，通过后才推进性能优化。
参考本地 SGLang 提交 `dcc1a73ff16b95aeb0a5252d3201573841080b2d` 的
`multimodal_gen/runtime/distributed/{parallel_state,parallel_groups}.py`、
`runtime/layers/{usp,linear}.py` 和 `runtime/loader/fsdp_load.py`。
不导入 `sglang` 包。SGLang 扩散 worker 的 PP 仍固定为 1，不能把它的预留接口当作落地策略。

## 实施范围与执行顺序

范围按 SGLang 已落地的扩散策略：DP、CFG、Ulysses、Ring、混合 USP、TP、FSDP/HSDP，
保留现有 VAE 并行接口。实际模型验收针对本仓库 EraserDiT；不将预留的 PP 接口算作已实现。

1. 正交分组与常驻 NCCL worker，和父进程 T5 FSDP 组隔离。
2. 小模型分布式测试：不等长序列、CFG 分支、TP 参数分片、USP 非连续分组、FSDP。
3. 同一原片逐策略质量验收；不以单次 forward 接近替代整片 SSIM。
4. 包括流式 Ring 的策略矩阵质量合格后，才减少 IPC/collective 等开销；每个优化候选仍需复验 0.985。

## 当前实现

`--dit-parallel-backend nccl` 启用独立 DiT 进程池，每个 rank 一张 GPU。
父进程拥有文本/VAE、scheduler、随机数和输出；DiT rank 0 通过 CPU tensor IPC 与父进程交接，
DiT rank 间使用 NCCL，控制元数据使用 Gloo。CPU IPC 是本实现保留的明确成本，不声称已实现零拷贝。
模型副本跨窗口/请求常驻，子进程错误或超时会由父进程传播并清理自有进程。

rank 顺序为 TP → Ulysses → Ring → CFG → FSDP 额外副本（低位到高位）。
`sp_degree = ulysses_degree × ring_degree`。TP 存在时 SP 组可能不连续。
DP dispatcher 给不同进程池分配不相交的 GPU 集合。

- Ulysses：NCCL 变长 all-to-all，支持不能整除的序列，不将补零 token 放入 softmax。
- Ring reference：P2P 环形传递 K/V，按原序列顺序拼接后计算 attention，以优先对齐。
  此模式保留完整 KV，不具有在线 Ring 的 KV 显存节省。
- Ring streaming：按全局 128-token 块顺序遍历，保留 FP32 输出累加器和四个行和，最后一次归一化；不拼接完整 KV。
- Ring online：逐块 FlashAttention、FP32 LSE 合并；当前未通过原片质量门槛，不推荐启用。
- TP：Linear 输出通道权重分片，再 gather 完整输出。
  `tp_linear_mode=reference` 为每个 Linear 临时补零以保留原 GEMM N 和通道偏移，保存的真实权重仍分片，
  但不节省该 Linear 的 FLOPs；`sharded` 只计算本地通道，其数值路径本轮未通过。
  `aligned` 复制文本首层与窄输出投影；与 SP 组合时，FFN 下投影保留参考 GEMM 宽度但真实权重仍分片。
  其他 Linear 计算本地通道。敏感投影的保护来自实测，不能把 aligned 理解为每一层都减少 FLOPs。
  RMSNorm、RoPE、GEGLU 维持原语义。属于保守的 gathered-column TP，不宣称已完成高效 row/column 交替。
- FSDP/HSDP：使用本仓库已有的 FSDP2 包装，对各 block 与根模块分片；TP 与 FSDP 不同时管理权重。
  FSDP mesh 可覆盖 CFG/SP world；纯 FSDP 时额外 ranks 重复相同输入计算，主要用于节省权重显存。

对齐阶段要求 SDPA、关闭编译/量化/手工融合/残差及文本投影缓存、DiT 不卸载。
T5/VAE CPU offload 可保留。之前约 0.982 的完整编译结果不满足本轮 0.985 阈值，不沿用为合格配置。

## 验收状态

整片测试：`data/113000356.mp4`，1920×1080、145 帧，seed 42，50 步、strength 0.8，
`infer_len=121, overlap=9, compact_tail_padding=True`；参考 `results/performance_20260927/fused50.mp4`。
原片逐帧 RGB SSIM 等权平均，阈值 0.985。结果保存在
`results/distributed_parallel_20260928/alignment.json`，逐帧数据在各 `*.quality.json`。

| 策略 | 整片 SSIM | 状态 |
| --- | ---: | --- |
| CFG2 | 0.992833050 | 通过 |
| Ulysses2 | 0.992833050 | 通过 |
| Ring2 reference | 0.992833050 | 通过 |
| FSDP2 | 0.992833050 | 通过 |
| SP2+FSDP2 | 0.992833050 | 通过 |
| TP2 reference | 0.992833050 | 通过；真实权重每 rank 1,924,227,200 字节 |
| DP2（两条独立任务） | 0.992833050 / 0.992833050 | 通过 |
| USP4（Ulysses2×Ring2 reference） | 0.992833050 | 通过 |
| CFG2×SP2 | 0.992833050 | 通过 |
| TP2 reference×SP2 | 0.992833050 | 通过 |
| CFG2×TP2 reference | 0.992833050 | 通过 |
| HSDP4（shard2×replicate2，CFG2×SP2） | 0.992833050 | 通过 |
| TP2 aligned | 0.992833050 | 通过；敏感投影保护，主干权重分片 |
| TP2 aligned×SP2 sharded（最终保护策略） | 0.992833050 | 通过 |
| CFG2×SP2 sharded（优化后） | 0.992833050 | 通过 |
| USP4 streaming×FSDP4 + SP sharded（优化后） | 0.992944427 | 通过 |
| USP4 streaming + FSDP4 | 0.992833050 | 通过；流式 KV，无完整 KV 拼接 |
| Ring2 streaming | 0.992833050 | 通过；流式 KV，无完整 KV 拼接 |

四卡小模型已覆盖 CFG、Ulysses、Ring、TP、SP+FSDP、USP、CFG+SP、TP+SP，含不等长 token。
进程池测试验证了父进程 singleton NCCL 不变、跨窗口复用和杀死一名 worker 后的失败传播与清理。
CFG2/Ulysses2 的解码 RGB SHA256 均为
`d04a28190a3a57c394d9abf375b0e488dcbf82543dcecec2c6d1add70532165b`，与既有紧凑尾窗 SDPA 输出一致。
未填入整片质量结果的策略不应视为已通过验收。性能对照迁移到 GPU 3、7：原 GPU 3、6 测量期间出现其他计算任务，已作废。
此前质量矩阵中的共享 GPU 耗时不用于计算性能收益。

### 已拒绝的数值候选

| 候选 | SSIM | 决策 |
| --- | ---: | --- |
| Ring2 online，BF16 局部输出 + FP32 合并 | 0.982674718 | 未过 0.985，不启用为默认 |
| Ring2 online，FP16 局部输出 + FP32 合并 | 0.982649604 | 未改善，未采用精度转换 |
| TP2，仅本地列 GEMM（`tp_linear_mode=sharded`） | 0.982665424 | 未过 0.985，转为保留 GEMM 形状的参考路径 |
| Ring2 streaming，单个 FP32 行和状态 | 0.982684831 | 单算子接近仍不足整片对齐；保留为失败候选 |
| TP2 aligned（仅保护文本首层）×SP2 sharded | 0.982677093 | 联合行/列切分改变输出与 FFN 下投影；增加形状保护后通过，见后文 |

FP16 实验输出与元数据在 `ring_precision*`；实验源码保存在
`results/distributed_parallel_20260928/rejected_ring_fp16.py`，不属于运行时依赖。
小模型误差检查通过不能替代完整扩散轨迹对齐。

进一步对真实权重做了形状检查：`caption_projection.linear_1`（4096→2048，128 个文本 token）
切分输出列后改变 BF16 GEMM 的数值路径；已检查的其他 Linear 形状在 32640 / 10200 个视频 token
下均逐元素一致。最初仅复制该小投影的 TP2 已整片通过（SSIM 0.992833050）。
继续组合 SP 后，16320-token 的窄 `proj_out` 和 5100-token 的 FFN 下投影发生偏差，
因此最终 aligned 策略还复制 `proj_out`，并在 TP×SP 下对 FFN 下投影使用 reference GEMM N。
后者不保留完整真实权重，只临时补零；新增保护后的组合整片通过，SSIM 0.992833050（`post_tp_boundary.json`）。
证据：`tp_shapes.json`、`tp_shapes_10200.json`、`tp_shapes_16320.json`、`tp_shapes_5100.json` 与相应日志。

流式 Ring 的第二次实现尝试保留四个 FP32 分片行和及未归一化输出，以接近原内核的逐片累加顺序。
数值顺序参考 [PyTorch 2.6 内置 FlashAttention softmax](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/flash_attn/softmax.h)
和 [归约实现](https://github.com/Dao-AILab/flash-attention/blob/v2.5.7/csrc/flash_attn/src/utils.h)，未导入其包。
最终采用四个 FP32 行和、显式首项 FMA、分离的 softmax 乘减与 `rcp.rn.f32` 正确舍入倒数。
在 A100 / PyTorch 2.6 上，10200 / 32640 token、32 heads、head_dim=64、两个 seed 的测试与原生 SDPA 逐元素一致。
四卡 USP+FSDP 与双卡独立 Ring 整片均已通过。其他硬件和内核 profile 不能直接推断为逐元素一致。

## 参数与边界

新增策略均使用 `--dit-parallel-backend nccl`。公共参数：

```bash
--no-dit-cpu-offload --no-dit-layerwise-offload \
--attention-backend sdpa --operator-fusion-backend disabled \
--transformer-cache-mode off --no-cache-text-projections
```

| 策略 | 附加参数 | 所需 GPU |
| --- | --- | ---: |
| CFG | `--cfg-degree 2` | 2 |
| Ulysses | `--sp-degree 2 --ulysses-degree 2` | 2 |
| Ring | `--sp-degree 2 --ulysses-degree 1 --ring-degree 2` | 2 |
| 流式 Ring | 上一行增加 `--ring-attention-mode streaming` | 2 |
| 实验在线合并（未达标） | 上一行改为 `--ring-attention-mode online` | 2 |
| USP | `--sp-degree 4 --ulysses-degree 2 --ring-degree 2` | 4 |
| TP | `--tp-degree 2 --tp-linear-mode aligned` | 2 |
| FSDP | `--dit-fsdp-shard-degree 2` | 2 |
| SP+FSDP | `--sp-degree 2 --ulysses-degree 2 --dit-fsdp-shard-degree 2` | 2 |
| HSDP+CFG+SP | `--cfg-degree 2 --sp-degree 2 --ulysses-degree 2 --dit-fsdp-shard-degree 2 --dit-fsdp-replicate-degree 2` | 4 |

CFG、SP、TP 的 GPU 度数相乘；FSDP mesh 覆盖计算 ranks，不能与 TP 同时管理同一权重。
DP 沿用 `entrypoints.cli.erase_parallel`，对独立视频任务分配不相交的 GPU 组；不把同一视频的滑窗当作独立 DP 任务。
当前为**单机自启动 worker**，不是多节点 torchrun 启动协议。服务启动参数与 CLI 同名，模型进程常驻。
`parallel_history` 包含实际各 rank 的参数字节、成功 forward、通信计数和显存峰值。

PP/PipeFusion/DistriFusion、TP+FSDP、NCCL DiT 编译/量化/缓存组合尚未提供；不能把配置预留或其他后端支持等同于本路径支持。

## 性能阶段

通过上述完整质量矩阵后，实施以下优化：

- 相同 rank 集合复用 NCCL process group；完整 world 直接使用 WORLD，减少重复通信缓冲区。
- Ulysses Q/K/V 按 batch 打包，一层 self-attention 从 4 次 all-to-all 减为 2 次。
- TP 输出通道和 SP 输出 token 长度由拓扑直接计算，避免每次 gather 前的长度通信与 GPU→CPU 同步。
- 性能候选显式选择 `--sp-linear-mode sharded`，省去参考路径的全序列补零 GEMM。

运行 `python results/distributed_parallel_20260928/run_performance.py` 可重放本轮前后对照。
脚本使用相同物理 GPU、模型、输入、采样参数和 allocator，每组 5 次完整请求，首个单独记录，
后 4 次报告中位数和范围；每次输出独立计算整片质量。前版本源码冻结在
`results/distributed_parallel_20260928/before_optimization/`，只由测试脚本加载，运行时代码不依赖此目录。
源码 SHA256 与完整命令保存在 `performance_manifest.json`；GPU 占用时间序列在 `performance_gpu_samples.jsonl`。

本机是共享环境，GPU 7 有其他任务常驻显存；测试开始时这些任务无计算负载。
不能将共享环境测量解释为独占服务器 SLA；发现新增计算占用的样本不计入对照。

CPU 回归：143 项，107 通过、36 项条件跳过（`optimized_cpu_tests.log`）。
GPU 回归：10 项，9 通过、1 项 CPU 专用测试跳过，399.148 秒（`optimized_gpu_tests.log`）。
初始通过的 14 个策略/组合样本加 2 个 DP 输出文件 SHA256 完全一致。
加上优化后的 3 个四卡组合，`quality_matrix.json` 共收录 19 个通过样本；另有 10 个性能请求全部通过。
优化后的流式 USP 输出略有不同，SSIM 0.992944427；其他 28 个已验收输出的文件 SHA256 一致。
前后各 5 次已完成，10 次输出 SSIM 均为 **0.992833050**，输出文件与验收基线一致。
原始记录为 `perf_baseline.json`、`perf_optimized.json`；汇总为 `performance_summary.json`。

| 同条件双卡 Ulysses2 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 首个完整请求（不含模型加载） | 203.13 s | 168.25 s |
| 后 4 次完整请求中位数 | 203.05 s | 175.34 s |
| 后 4 次完整请求范围 | 202.53–205.74 s | 171.10–176.12 s |
| 后 4 次 DiT 去噪中位数 | 149.81 s | 122.39 s |
| DiT worker 峰值 allocated | 5.734 GiB | 5.360 GiB |
| DiT worker 峰值 reserved | 9.736 GiB | 10.049 GiB |
| 主进程峰值 allocated | 30.446 GiB | 30.446 GiB |

端到端耗时下降 **13.65%**（**1.158×**），DiT 去噪耗时下降 **18.31%**（**1.224×**）。
这是“相同 Ulysses2 的旧实现/参考 GEMM 模式 → 通信优化 + 真正序列分片计算”的组合收益，
不是双卡对单卡的扩展效率，也不能把总收益分别归给每一项修改。
既有 peer 四卡路径曾记录 96.151 s（见 `four_gpu_optimization_20260927.md`），测试日期、设备与配置不同，
本轮没有证明 NCCL 比该路径更快。默认仍保留 peer；新增 NCCL 主要扩展真实分布式通信、TP 和 FSDP/USP 等能力。

NCCL 通信组从 WORLD + 两个重复组减少为仅 WORLD，self-attention 的 all-to-all 次数减半。
GPU 7 上 DiT worker 的 NVIDIA 10 秒采样最大占用从 13284 MiB 降到 12128 MiB；这不是瞬时峰值测量。
PyTorch reserved 略升，allocated 下降，不能声称所有显存口径都下降。
父进程的 T5/VAE 峰值没有下降。GPU 3、7 没有出现新增外部进程，GPU 7 原有两个进程的常驻显存保持不变；
使用默认动态频率而非锁频，采样证据见 `performance_resource_audit.json` 和原始时间序列。

优化后的双卡已完成 5 次整片验收；四卡 CFG×SP 已通过（0.992833050，单次 122.50 s），
USP streaming×FSDP 已通过（0.992944427，单次 204.19 s；与 reference 模式不再逐元素相同，但满足门槛）。
TP aligned×SP 修正敏感投影后通过（0.992833050，单次 291.95 s）。上述四卡时间只作本次验收诊断，不计算跨策略的正式加速率。
新增 TP 保护的双卡/四卡专项 2 项通过，69.776 s（`tp_boundary_tests.log`）。
最后的拓扑、配置与服务 API 契约 11 项通过（`final_contracts.log`），`git diff --check` 通过。
TP 的数值保护不影响 TP=1 的双卡 Ulysses 性能对照；性能数据与最终 TP 组合验收分开保存。

### 已验证的参数分片显存

以下是 worker 实际张量的参数字节统计，不是理论估算，也不包含 T5/VAE、激活或 NCCL 缓冲区。

| 配置 | 每 rank 的 DiT 参数 GiB |
| --- | ---: |
| CFG2（完整 DiT 副本） | 3.583 |
| TP2 aligned | 1.800 |
| FSDP2 / HSDP shard2 | 1.832 |
| USP4 streaming + FSDP4 | 0.956 |

FSDP 的根模块驻留等行为使统计值不必恰好等于完整参数除以分片数。
整卡峰值还包括主进程的文本/VAE 和其他进程，不能用上表替代容量规划。

## 可用配置

在上文公共参数基础上，已完成本轮样片验收的配置包括：

```bash
# 双卡 NCCL Ulysses：本轮五次性能对照配置
--dit-parallel-backend nccl --sp-degree 2 --ulysses-degree 2 --sp-linear-mode sharded

# 四卡 CFG×SP：本轮低延迟组合验收配置
--dit-parallel-backend nccl --cfg-degree 2 --sp-degree 2 --ulysses-degree 2 --sp-linear-mode sharded

# 双卡 TP，或再加 --sp-degree 2 --ulysses-degree 2 使用四卡 TP×SP
--dit-parallel-backend nccl --tp-degree 2 --tp-linear-mode aligned --sp-linear-mode sharded

# 四卡流式 USP + 权重分片
--dit-parallel-backend nccl --sp-degree 4 --ulysses-degree 2 --ring-degree 2 \
--ring-attention-mode streaming --sp-linear-mode sharded --dit-fsdp-shard-degree 4
```

DP 优先用于独立任务吞吐；CFG/SP 用于单任务计算分摊；TP/FSDP/HSDP 提供权重分片选择。
本次 gathered-column TP 与流式 Ring 的精度保护都有额外计算/通信成本，不保证每种策略都降低延迟。
上述数值只覆盖指定模型、样片、参数与 A100/PyTorch 2.6；更换素材/形状/硬件后应按同一门槛复验。
