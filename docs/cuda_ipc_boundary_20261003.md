# NCCL owner/worker 的 CUDA IPC 边界

本轮针对每步 latent 和预测结果的 CPU 中转。默认仍使用 CPU tensor IPC。
`MGERASE_DIT_BOUNDARY_TRANSPORT=cuda_ipc` 在创建 NCCL DiT 进程池时启用 GPU 张量 IPC；
运行中的池不随环境变量变化而切换，需要在空闲状态重建。

## 实现与所有权

- 控制消息仍走 Pipe，DiT ranks 之间仍走 NCCL，不改父进程 T5 的默认 process group。
- owner 保留输入张量直到本次调用完成，发送前同步输入设备，覆盖非默认 stream 的生产者。
- rank 0 将输入复制到自己的 GPU 存储后再广播。跨步保存的静态条件不引用 owner 的 IPC 存储。
- rank 0 保留输出直到 owner 发来下一条指令；owner 将导入的输出 clone 到自己的存储并同步，
  释放导入句柄后才返回，因此持有的结果不依赖 worker 在 reset/关闭后继续存活。
- GPU 输入必须位于 rank 0 使用的设备；不自动回退到 CPU、不绕过传输错误。

这条路径去掉 CPU 中转，但保留接收端 GPU 拷贝和必要同步，不是零拷贝或计算通信重叠。
只在已验证的配置上按实测选择：GPU 专项覆盖常驻 CFG2/SP1 和 CFG2×SP2，
完整视频使用 CFG2×SP2；TP/Ring/FSDP 组合未在本轮验收。

报告中的 `boundary_transport` 区分 `cpu_tensor_ipc` 和 `cuda_tensor_ipc`。
`boundary_tensor_bytes` 仍为逻辑输入/输出大小，不代表实际 PCIe 流量。
CUDA 路径的 `worker_roundtrip` 包含输入设备同步、控制消息、DiT 计算及输出 GPU 拷贝；
不能将它全部归因于通信。`input_to_cpu` 在 CUDA 路径为零，轻量 packet 准备另记为 `input_gpu_prepare`。

## 验证

真实多卡小模型检查覆盖变化输入、非连续张量、非默认 stream、CFG guidance 0/1/7.5、
多窗口、文本缓存与强制实算残差缓存、保留旧输出，以及 peer 失败后清理与正常关闭。
同拓扑 guided 与原 FP32 CFG 运算要求逐元素一致；不同 SP 形状对 dense 模型沿用既有容差。

CPU 回归 202 项：147 通过、55 按硬件/专项开关跳过。
额外 GPU 正常关闭及 rank 0 故障测试在该次 CPU discovery 之后加入并单独通过。
GPU 故障/生命周期回归 2 项通过，正常关闭与 rank 0 故障回归各 1 项通过。

## 真实视频配对

L40S GPU 0–3，CFG2×SP2、BF16/SDPA、sharded Linear，121 帧固定输入、40 实际步、seed42。
两条路径均开启融合/direct 打包、VAE 激活优化和后处理分块，关闭文本/残差缓存及编译量化。
同一 owner 会话，CPU/CUDA、CUDA/CPU、CPU/CUDA 三组交替；切换时重建 workers，
两次同形状 warm forward 单列，首次整管线预热从请求耗时扣除。

原始记录及冻结源码：`outputs/cuda_ipc_boundary_20261003/`。

| 中位耗时 | CPU IPC | CUDA IPC | 减少 |
| --- | ---: | ---: | ---: |
| 去噪 | 64.219 s | 63.567 s | 1.02% |
| 纯推理 | 77.832 s | 77.155 s | 0.87% |
| 端到端请求 | 95.384 s | 94.605 s | 0.82% |

三组端到端配对差值（CUDA−CPU）为 −0.645、−0.157、−1.768 s，中位 −0.645 s。
三组均变快，但 CPU 范围 94.762–96.422 s、CUDA 范围 93.616–95.777 s 有重叠；
单素材/seed、未锁频的小样本不足以推广稳定收益，默认仍为 CPU，CUDA 作为显式实验选项。

六个视频 SHA256 均为
`21203e33ed3b6fc11973897181f30b74a1b27606a0558d7554bdf775e8964354`，与前轮参考一致。
Owner 峰值 allocated 分别 19.210 / 19.209 GiB，worker 最大峰值均 5.159 GiB，基本不变；
这些不是包含多个进程与 CUDA 上下文的整卡占用。逐卡 GPU/进程采样保存在 `gpu_samples.jsonl`。

CUDA 路径确实消除了显式 CPU 输入/输出搬运，但首组 owner 这两项合计仅约 0.337 s。
worker roundtrip 内还包含 worker 回拷和 IPC 成本，不把 0.337 s 当作完整可节省时间；
实测小幅收益表明当前不宜优先扩大 scheduler 下沉改造。后续先在当前融合/direct 基线上
剖析并验证 DiT 内部通信与计算重叠，收益需另做同条件对照。

`summary.json` 保留中位数、范围、逐组差值及显存；`implementation.diff` 仅包含本轮
相对于上一轮冻结源码的实现/测试差异。性能测试源码与最终运行时代码相同，随后只补充
了正常关闭、rank 0 故障测试和说明文档。
