# 内存、卸载与拷贝优化审计

本轮目标：在固定 121 帧单窗口、相同采样和精度下，减少无效权重搬运与重复分配，
并分别验证纯推理耗时、GPU allocated/reserved、CPU RSS 和输出一致性。
VAE 新路径显式启用；请求结束时解除运行上下文循环引用的修复默认生效。

## 当前已有能力

| 路径 | 已有优化 | 当前边界 |
| --- | --- | --- |
| DiT 逐层卸载 | 同层同 dtype 权重合并到 pinned CPU 缓冲区，独立 copy stream 预取；计算后释放 GPU 引用，不回拷不变权重 | 仅单卡；预取增加占用，CPU 权重需长期保存 |
| 文本编码器 | T5 FSDP CPU offload，按 block 搬运并 reshard；请求内复用文本编码 | 需要主机内存；有 FSDP 生命周期与异常恢复成本 |
| VAE 卸载 | 编码/解码阶段分别 acquire/release，阶段外权重在 CPU | 原版两阶段均搬运整个 VAE，结束后重新分配 CPU 存储并回拷 |
| 激活与帧缓存 | 窗口调度、uint8 ArrayFrameCache、清理窗口引用、VAE 前释放全分辨率输入；mask 处理分批 | CPU/GPU 仍需完整窗口活跃张量 |
| 低显存 VAE | RMSNorm 时间分批、Conv3d 按完整时间邻域分批；可选空间 tiling | 分批卷积会改变 BF16 kernel 路径，tiling 属近似路径 |
| 多卡权重 | NCCL FSDP/HSDP、TP | 分片节省单卡权重，占用与通信需要同拓扑衡量 |
| 多卡数据拷贝 | peer 大张量直写目标；NCCL 只在 owner 汇聚，worker 内合成 CFG 后仅返回一个预测 | scheduler 在父进程，CPU IPC 保留；已有计时不支持优先大改 IPC |
| Ulysses 打包 | 显式 packed 路径把逐片拷贝加拼接改为整体打包 | 单窗口 A/B 输出一致；收益按配置变化，见[并行对照](window_optimization_20261002.md) |

实现入口：`memory/backends/layerwise_offload.py`、`memory/adapters/sglang_memory_adapter.py`、
`memory/backends/fsdp_offload.py`、`models/vaes/memory.py`、`pipelines/runtime/windowing/`、
`models/adapters/eraserdit/nccl_sequence.py`、`pipelines/runtime/dit_executor.py`。

## 优先方案：VAE 阶段驻留与 CPU 权重复用

1. 编码阶段只上传 encoder，解码阶段只上传 decoder；两阶段均保留必要的根参数/buffer。
2. 缓存模式保留 CPU 权重存储，遵循 pin_cpu_memory 设置，之后复用；卸载时恢复 CPU 引用。
3. eval 阶段未修改的参数不回拷；buffer 始终回拷，正常原地操作/load_state_dict 导致的参数版本变化也回拷。
4. 部分上传失败后恢复 CPU 驻留并支持重试。阶段内直接 `.data` 写参数不属于支持的更新方式。
5. 保留 `full` / `split` / `cached` 三条路径做归因，默认仍是 `full`。

实验开关：`MGERASE_VAE_OFFLOAD_MODE=full|split|cached`。
只影响启用 VAE CPU offload 的推理。缓存主机存储在 GPU 计算时仍存活，并非消除 CPU 内存需求；
需要同时观察 CPU RSS、pinned CPU 字节和 GPU 峰值，不能只看某个局部张量。

阶段事件新增逻辑 H2D/D2H 权重字节数，快照新增 CPU backing/pinned 字节。
这些字节是张量传输量，不等同于 profiler 或 PCIe 硬件计数。
acquire/release 墙钟时间可包含等待此前 GPU 工作，不能单独当作 memcpy kernel 时间。

## 已实现：请求结束时解除循环引用

原路径存在 `Req.extra["runtime_context"] → context.request_batch → Req` 循环，
多对象转发回调还会捕获 context/task state。上下文持有完整输出和帧缓存，
即使请求已结束，也要等待 Python 循环垃圾回收才释放。

`close_runtime_resources` 现在在 finally 中解除请求双向引用并清理对象转发回调；
返回的输出张量和已发布诊断元数据仍保留。不额外调用 gc.collect，也不改变窗口内计算。

独立诊断使用相同 121 帧窗口、3 次请求、2 配置步数，只验证生命周期，不用于性能结论：
原版结束后 3 个上下文存活，显式 gc 后变为 0，RSS 33.534 → 20.215 GiB；
修复版每次只存活当前上下文，结束后无需 gc 即为 0，RSS 20.371 GiB，
再 gc 几乎不变。修复前后 3 个输出文件分别逐字节一致。
禁用循环 GC 的单元测试覆盖单对象、多对象回调和资源关闭异常，修复前 3 项失败，修复后通过。

## VAE 三路径正式对照

L40S 物理卡 0–3，CFG2×SP2、BF16 SDPA、常驻 NCCL DiT，T5/VAE CPU offload，
vae-low-memory；缓存/编译/融合关闭，Ulysses reference。
同一 1920×1080、121 帧无损 fixture，seed 42，50 配置步数、strength=0.8，40 有效去噪步。
每组预热一次（2 步），连续 5 次正式请求；各请求恰好一个窗口。
纯推理不包含加载、预热和视频 I/O。

| 项目 | full 原版 | split 阶段卸载 | cached 阶段卸载+复用 |
| --- | ---: | ---: | ---: |
| 纯推理中位数 / 秒 | 85.150 | 84.015 | 83.484 |
| 纯推理范围 / 秒 | 84.761–85.542 | 83.672–84.141 | 83.346–84.250 |
| 去噪中位数 / 秒 | 71.525 | 70.291 | 70.126 |
| owner 峰值 allocated / GiB | 20.842 | 19.812 | 19.812 |
| 每窗 VAE H2D / 字节 | 4,987,655,112 | 2,493,828,068 | 2,493,828,068 |
| 每窗 VAE D2H / 字节 | 4,987,655,112 | 2,493,828,068 | 1,024 |
| VAE 驻留切换墙钟中位数 / 秒 | 0.782 | 0.624 | 0.526 |

确定收益：GPU 活跃权重峰值减少约 **1.03 GiB**；VAE 权重总搬运量减少约 **75%**，
D2H 仅剩根 buffer。驻留切换墙钟下降约 33%。
本次纯推理中位数下降 **1.96%**，但去噪也下降 1.40 秒，并非此优化直接改变的计算路径；
各组顺序测量，期间其他空闲 GPU 有独立诊断，不能把全部耗时差归因于 VAE 优化，
也不承诺固定比例的推理加速。

15 个正式输出及先导输出的 SHA256 均为
`21203e33ed3b6fc11973897181f30b74a1b27606a0558d7554bdf775e8964354`。
三组保留修复前的生命周期行为，用于隔离 VAE 变量；CPU RSS 仍有延迟释放，
不能据此宣称 cached 本身修复 CPU 增长。

显存数字是 owner 进程 PyTorch allocated，**不是整卡总占用或所有 worker 总和**。
reserved 包含分配器缓存：首次请求原版 35.504 GiB、新路径 35.494 GiB（峰值计数包含本次预热）；
后续原版 30.209 GiB、split/cached 27.084 GiB。不要把它与 allocated 或 NVML used 混用。
CPU 权重约 2.32 GiB，cached 在 GPU 驻留时也保留该 backing，并遵循 pin_cpu_memory 设置。

## 单卡 21 GiB 分配器限额

GPU 6，默认逐层 DiT 卸载、T5/VAE CPU offload、vae-low-memory，
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，同一 121 帧、40 有效步。

- full：在 VAE 编码阶段 OOM，尚未完成推理。
- cached：完整成功，峰值 allocated **20.018 GiB**，reserved **20.252 GiB**。
- 另跑不限额单卡 full，输出与限额 cached 逐字节一致：
  `f6baceed78a900464db10037ac5c6dd4facbee080cb5c8a30ce43567be91c1ec`。

这是 L40S 上的 PyTorch 分配器限额验证，不等于真实 21/24 GiB 显卡适配认证；
CUDA context 和分配器外内存不在该限额内，也不用于跨拓扑耗时比较。

## 最终组合验证

cached 加请求生命周期修复，同一卡组 5 次完整单窗口全部完成，输出 SHA256 与前述 15 次相同。

| 项目 | 修复前 full（5 次） | 修复前 cached（5 次） | cached + 生命周期修复（5 次） |
| --- | ---: | ---: | ---: |
| owner 阶段末 RSS，第 1 次 / GiB | 35.221 | 34.974 | 33.103 |
| owner 阶段末 RSS，第 5 次 / GiB | 52.981 | 48.294 | 33.130 |
| 第 1–5 次 RSS 增长 / GiB | 17.760 | 13.320 | 0.027 |
| owner 峰值 allocated / GiB | 20.842 | 19.812 | 19.812 |
| 纯推理中位数 / 秒 | 85.150 | 83.484 | 87.678 |
| 去噪中位数 / 秒 | 71.525 | 70.126 | 74.347 |

最终组合 RSS 范围为 **33.103–33.132 GiB**，这是同一个阶段采样点，非整请求主机内存峰值，
也不是全部 worker 的 PSS。确定消除了本例中逐请求的循环引用累积。
最终组合纯推理范围 86.583–87.767 秒；耗时差主要来自去噪，不能把较早缓存组的 1.96% 当作最终稳定收益。
随后在相同当前源码、相同卡组补测 full 3 次：纯推理中位数 **83.891 秒**
（83.460–83.953），去噪中位数 70.230 秒；RSS 三次均约 33.103 GiB。
相对这组 full，最终 cached 中位数慢 **4.5%**，差异集中在去噪阶段。
目前没有证据把差异明确归因于单一原因；纯推理测量未证明稳定加速，
因此 **full 仍为默认；cached 是降低显存和传输量的可选取舍，不作为提速推荐**。
生命周期修复独立默认启用，在 full 下也消除了 CPU 逐请求增长。
23 个四卡正式输出全部逐字节一致，汇总校验见 `verified_results.json`。

## 后续优先级

1. **当前可用**：显存紧张时开启 cached；长期常驻服务使用生命周期修复。
   VAE 模式默认保留 full，已验证范围为本素材和 eager VAE，不外推编译/量化/空间并行组合。
2. **进一步降低主机峰值**：preload 路径保存视频时仍构造整窗 FP32 final_video。
   需要先明确调用方是否索取内存输出，再引入仅文件输出契约；不能直接删除返回结果。
3. **进一步减少搬运**：在显存有预算时研究 encoder/decoder 常驻或重叠预取。
   会与 DiT 激活争显存，需按阶段峰值验证，不能无条件开启。
4. **暂缓**：重写 scheduler/worker IPC、多卡 VAE 重构、全局 pinned 激活池。
   当前证据优先支持消除无效 VAE 传输和确定的请求引用滞留；其他改动需要独立 profile 与质量验收。

最终回归：CPU unittest 全套 177 项，131 通过、46 条件跳过；
GPU VAE/组件卸载及生命周期专项 10 项全部通过。

验证日志、manifest、源码哈希与视频位于 `outputs/memory_optimization_20261002/`
（本机产物，不随 Git 分发）。正式三组为 `formal_full` / `formal_split` / `formal_cached`，
最终组合为 `final_cached_cleanup`，当前源码补测为 `final_full_cleanup`，诊断为 `diag_retention.log` / `diag_fixed.log`，
限额为 `cap21_results.json` 与对应日志。

复现实验（每次指定一个新的输出目录；物理卡须空闲）：

```bash
uv run --no-project python -m entrypoints.cli.benchmark_window \
  --run-dir outputs/my_cached_check --devices 0,1,2,3 \
  --configs cfg2_sp2 --repeats 5 --vae-offload-mode cached
```

把模式改为 `full` 或 `split` 即可对照；CLI 会拒绝覆盖已有目录，校验单窗口和 121 帧，
并保存命令、环境、逐次纯推理耗时与内存统计。
