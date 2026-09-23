# DiT 逐层卸载实现

`dynamic_offload` 现采用阶段组件调度和 DiT 逐层预取。参考本地 SGLang 的
`python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py`，独立实现于
[manager](../memory/backends/layerwise_offload.py) 和
[adapter](../memory/adapters/layerwise_memory_adapter.py)。

## 存储和预算

连续权重按 block/dtype 合并到 pinned CPU 存储；非连续权重保留 stride。
GPU 以 storage 为单位复制，参数恢复为相应 view。用完恢复 CPU view，不写回权重。
CPU view 保留真实形状，避免占位张量影响参数检查。权重只读，运行中不支持 refit。
跨 block 或与常驻部分共享 storage 的权重在注册前拒绝；同一个 Parameter 的层内别名保留。
不同 Tensor 对象的重叠层内视图目前也拒绝；不重叠的打包视图可以重新注册。

预算覆盖在途复制、已预取、正在计算和等待计算完成的 GPU block 存储。
释放后仍持有 buffers 和完成 event，直到 event 完成才扣除预算。
当前层必要加载可以淘汰无用预取并等待预算；额外预取不会强制等待预算。
单层超预算在改变参数 storage 前拒绝。预算不含 allocator reserved 和其他模型/激活。

管理器构造不分配 GPU 权重；进入去噪阶段也只加载非 block 部分。
首次访问 block 时加载当前层，并在 copy stream 预取后续层。计算流只等待当前层。
`record_stream` 和完成 event 共同保护跨流生命周期；部分 H2D 失败会同步复制流并回滚。

## 阶段边界

Text Encoder 整体、VAE encoder/decoder 分别按阶段加载。VAE 共享状态随活动阶段加载。
DiT 非 block 权重及首层 TeaCache 小探针参数在去噪阶段驻留。
去噪结束清空所有在途/预取权重；随后才进入 VAE 解码。无跨组件预取。
CPU 保留原始组件权重；释放 GPU 副本不发生 D2H。
组件进入前和退出后清理空闲 CUDA allocator 缓存，避免预处理/VAE 大型 workspace
造成后续阶段的 reserved 峰值叠加；不在 block 之间清理缓存。

所有常规 block 执行统一经过 `Module.__call__`，内部继续使用原融合实现。
缓存的每个实际执行区间限定预取边界；跳层和重复调用仍保证当前层按需就绪。
TeaCache 直接执行已驻留的小探针，避免整层传输。

正常关闭移除 hooks，将 DiT 权重恢复成普通 CPU 存储；终止关闭直接释放权重引用。
阶段退出的 finally 负责异常清理。组件搬运历史最多保留 64 条，避免会话日志无限增长。

## 初始化峰值

loader 根据 offload 策略选择 CPU，并显式启用 `low_cpu_mem_usage`。
不会先将完整 DiT 放上 GPU，再创建 CPU 副本。当前仍由 `from_pretrained` 完成 checkpoint
加载，然后逐 block 整理 pinned 存储；尚未直接从 checkpoint 写入最终 pinned 存储。

初始化报告记录加载前、加载后、注册后的 CUDA 和 CPU 内存。
CUDA peak 是最近外部 reset 后的累计值，CPU peak RSS 是进程历史峰值。
阶段搬运记录也采集累计峰值，但不声称它们是独立阶段峰值。
观察实际物理占用还需区分 CUDA context、其他库分配以及其他进程；它们不在 PyTorch
allocated/reserved 中。释放 CPU pinned 引用也不保证主机 allocator 立即归还 RSS。

## 当前边界

支持单 GPU、BF16 eager、TeaCache、cache_dit、文本投影缓存。
INT8、FFN compile、CFG/SP 与卸载已接入组合路径，详见
[组合验收](composable_acceleration_validation_20260923.md)。单卡 VAE tiling 可叠加；
多卡 VAE 暂要求 fullgpu，FSDP 明确拒绝。
T5 逐层卸载、VAE 激活优化、checkpoint 直接入 pinned 存储、固定 GPU buffer 池、
部分层常驻和跨组件预取不在本次实现中。旧 extent 后端暂留供原有适配接口和回归使用，
EraserDiT 的 `dynamic_offload` 不再走该后端。

CLI 用法和预算口径见 [性能配置](performance.md#offload)。
