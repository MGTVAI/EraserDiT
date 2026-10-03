# Ulysses head 分块与计算通信重叠

**候选已实现并通过正确性检查，但当前 L40S 整片未提速，默认仍为不分块。**
本轮以常驻 NCCL CFG2×SP2、融合/direct 打包为基线，owner/worker 保持 CPU IPC，
隔离上一轮 CUDA IPC 的影响。环境为 Torch 2.6.0+cu126 / Triton 3.2.0 / L40S。

## 最终整片筛选

121 帧 1080p 固定素材、40 实际步、seed42，三种配置均开启融合/direct、VAE 激活优化和
后处理分块，关闭近似缓存、编译及量化。同一 owner 会话，切换时重建 workers，
加载、重建与预热单列；正式请求不启用 profiler。

| 最终实现 | 去噪 | 端到端请求 | Worker 峰值 allocated / reserved |
| --- | ---: | ---: | ---: |
| 不分块参考 | 64.022 s | 95.628 s | 5.160 / 11.268 GiB |
| 两块重叠 | 66.983 s | 97.498 s | 5.034 / 10.881 GiB |
| 四块重叠 | 68.283 s | 99.847 s | 5.034 / 10.881 GiB |

每配置一次，属于筛选，不是稳定性能排名。两块/四块端到端分别慢约 1.96% / 4.41%，
因此不纳入当前机器的速度推荐。Owner 峰值 allocated 均约 19.21 GiB，不能将 worker
的小幅下降当作整请求/整卡显存收益。Reserved 与多进程整卡占用也不是同一口径。

三个视频 SHA256 均为
`21203e33ed3b6fc11973897181f30b74a1b27606a0558d7554bdf775e8964354`，与既有参考逐字节一致。
记录与冻结源码位于 `outputs/ulysses_overlap_20261003/retirement_pilot/`，
汇总为 `outputs/ulysses_overlap_20261003/final_summary.json`。

## 实现与使用边界

`MGERASE_ULYSSES_HEAD_CHUNKS=1|2|4`，默认 1 保持原路径。开启分块需要
`MGERASE_NCCL_PACKING=direct`、纯 Ulysses SP2/4，不能与 Ring、TP、FSDP 组合，
且模型 heads 必须能被 `SP × chunks` 整除。仅支持无梯度、无 mask、非因果、无 dropout 的
CUDA self-attention；推理配置沿用 BF16/SDPA、关闭编译和量化的限制。

- Triton 直接将每个目的 rank 的指定 head 范围打包，不先复制完整 QKV 再切片。
- 通信 stream 等待当前 QKV 就绪；计算 stream 等待首块 CUDA event 后启动 SDPA，
  再排入下一块通信，使其有机会与当前 attention 重叠。各 rank 保持相同 collective 顺序。
- 各块结果按原 head 次序拼接，再执行原有输出 all-to-all。输出交换尚未分块重叠。
- 最终版本显式保留跨 stream 接收缓冲到读取结束，再安排通信 stream 等待计算完成后复用，
  最后汇合两条 stream。事件顺序避免循环依赖，异常路径同样完成依赖安排。
  不依赖每个 QKV 张量的 `record_stream` 延迟回收，不增加常驻大缓冲池。

分块不减少 token 或 softmax 归约范围，但每 block 的输入 collective 从 1 次增至 2/4 次，
输出仍为 1 次。额外启动、调度和资源竞争均有成本；本轮没有进一步分离它们对整片回退的贡献。
报告新增 `ulysses_head_chunks` 与 `ulysses_head_overlap`，默认路径报告未启用重叠。

## 为什么没有按单 forward 结果推广

在采用 `record_stream` 管理缓冲的中间版本中，真实权重、固定合成激活、五组交替测量的
forward 中位耗时如下。所有输出与同拓扑基线逐元素一致。

| Token 数 | 融合/direct 基线 | 两块串行 | 两块重叠 | 四块重叠 |
| --- | ---: | ---: | ---: | ---: |
| 32640 | 1.4713 s | 1.4610 s | 1.4311 s | 1.4193 s |
| 10200 | 0.3296 s | 0.3317 s | 0.3276 s | 0.3139 s |

长窗口四块在该筛选中快约 3.53%，但随后两组完整视频配对回退：参考 95.417/95.589 s，
四块 104.261/104.113 s；四个文件均精确一致。Worker reserved 还从约 11.3 增至 14.3 GiB。
原定三组对照因此在完成两组后停止，记录位于 `e2e/`，不能记为完成了三组。

最终显式缓冲回收版本消除了该 reserved 上涨，但完整视频仍慢，结果见首表。
不同阶段数据不能跨组直接算提速；保留了中间源码和各自 manifest，便于复查。

## 实际重叠证据

当前基线的一个真实权重 forward 采样中，各 rank 输入 all-to-all 的 inclusive device time
约 199–204 ms，self-attention 约 460–470 ms。NCCL 包含 rank 等待，父子区间也有重叠，
不能相加估算总耗时或将等待全部视为传输。

对 GPU kernel 时间区间先取并集、再求 NCCL 与 Flash attention 的交集：基线为 0；
中间四块版 CFG2×SP2 各 rank 为 48.7–50.5 ms；最终显式回收版另在 CFG1×SP2、两个
串行 CFG 分支中观测到每 rank 92.5–94.3 ms。后二者拓扑/工作量不同，不能直接比较大小。
这些只证明发生了真实重叠；profiler 扰动调度且导出有额外开销，不能用其 wall time 宣称收益。
原始 trace 与区间分析保存在 `profiles/`、`interleaved_profiles/`、`retirement_profiles/`、`kernel_overlap.json`。

## 正确性与回归

- 直接 head 打包覆盖不同 stride、batch、dtype 及 NaN/Inf/正负零位模式。
- 真实 NCCL SP2、CFG2×SP2、SP4 覆盖非等长分片、变化形状、非默认 stream、保留旧输出；
  串行/重叠两块及重叠四块与未分块输出精确一致。
- 注入 attention 失败后，同进程组和 stream 可再次调用；真实进程池覆盖融合、
  文本/残差缓存、窗口重置与 peer 故障清理。
- 最终 CPU 回归 208 项：149 通过、59 按硬件/专项开关跳过，跳过不计作 GPU 通过。
- 最终运行时代码与 `retirement_pilot/source/` 冻结版本一致；日志及哈希见 `final_validation.json`。

本项完成的是输入 head 分块候选的实现与筛选，尚未完成输出通信流水、IO 流水或更大编译子图。
后续应以持续真实请求为筛选入口，先定位进程池调度与多次 collective 的成本；
不再仅凭孤立 forward 的改善扩大分块数或推广默认配置。
