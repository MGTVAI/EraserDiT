# SGLang 源码迁移的内存管理

本项目不依赖 `sglang` Python 包。迁移来源为本地 SGLang commit
`cdd427a588037dc8a8eb860ac17654bb2e55e752`，许可证保存在
`memory/backends/SGLANG_LICENSE`。

## 源码与接入

- `memory/backends/layerwise_offload.py`：迁移上游同名文件，替换平台、日志、配置导入，
  并增加 TeaCache 所需的限定层执行计划和局部 FFN 编译预热驻留上下文。包括 `LayerwiseOffloadManager`、
  `OffloadableDiTMixin` 和权重物化迭代器。
- `memory/backends/fsdp_offload.py`：迁移上游 `runtime/loader/fsdp_load.py` 的 `shard_model`。
- `loader/meta_load.py`：按上游 meta 初始化、safetensors 权重物化流程适配本项目 checkpoint；
  使用本模型的原始名称，保留共享 Parameter，并严格检查缺失、额外名称及形状。
- `memory/adapters/sglang_memory_adapter.py`：阶段生命周期、单进程 NCCL mesh 所有权和观测。
  不实现第二套预取调度，也不提供字节预算。

DiT 权重按层/dtype 合并到 CPU 存储，非连续权重保留 stride。独立复制流进行成组循环预取，
当前层等待自身完成 event，末端回绕到首层。卸载参数使用 GPU 小占位 tensor。
初始化即加载首批层，非 block 权重常驻执行设备。去噪阶段结束调用上游 `release_all`。
默认 pin CPU 权重；上游在 pin 分配失败时允许退回普通 CPU 存储。

T5 使用 `T5Block` 作为 FSDP 分片边界，最后包装根模块以覆盖共享 embedding 等参数；
`CPUOffloadPolicy`、`reshard_after_forward=True`。当前集成仅支持 world_size=1。
Torch 2.6 异常 forward 会遗漏 FSDP post-hook；阶段清理补完该 hook 后 reshard，
此处使用 `_get_fsdp_state` / `_post_forward`，升级 Torch 时需重跑异常恢复测试。
进程组由本项目创建时在最后一个使用者关闭后销毁；外部提供的组不由本项目销毁。

VAE 按上游组件级 `Module.to` 路径整体搬运；单卡空间 tiling 调用模型原生重叠分块方法，
保持其归一化及 posterior 采样语义。tiling 默认关闭，时间分块尚未作为入口配置开放。
卸载不会消除视频预处理或 VAE 激活峰值。
原生 tiling 是近似路径：本轮 256/224 小块配置未通过画质门槛，默认保持关闭。

旧 extent 后端、自定义逐层管理器、CPU 权重镜像组件恢复路径均已移除。
关闭后的 pipeline 不支持再次推理，也不提供旧的模型恢复/重新注册接口。

## 配置

CLI 和服务默认启用 DiT 逐层卸载、T5 FSDP CPU offload、VAE CPU offload、pin CPU memory。
直接构造 `ServerArgs` 时卸载开关默认关闭，调用方需显式启用。

```bash
--dit-layerwise-offload --text-encoder-cpu-offload --vae-cpu-offload \
--pin-cpu-memory --dit-offload-prefetch-size 0
```

预取参数采用上游语义：`0 <= p < 1` 时为 `1 + round(p * (层数 - 1))`；
`p >= 1` 时取整数层数，最多为总层数。**0 表示预取一层，不是关闭预取。**

全驻留对照：

```bash
--no-dit-layerwise-offload --no-dit-cpu-offload \
--no-text-encoder-cpu-offload --no-vae-cpu-offload
```

DiT 整组件卸载：`--no-dit-layerwise-offload --dit-cpu-offload`。
`--resource-policy`、`--max-weight-usage`、`--pin-memory` 已删除，不提供静默映射。

## 当前组合边界

迁移卸载支持单 GPU BF16 eager 和局部 FFN compile，允许单卡原生 VAE 空间 tiling。
FFN 编译预热按层借用权重，等待复制 event 并记录计算流使用，结束或异常后释放。
正式推理的复制与释放 hook 留在 eager，CUDA graphs 关闭；QK RoPE/AdaLN 融合在编译区域之外。
CFG/SP mesh 可以使用常驻 DiT 副本，同时保留主卡 T5/VAE CPU 卸载；
多卡 DiT 权重卸载、多卡 VAE 卸载、旧 `cfg_parallel_device` 卸载路径仍拒绝。
INT8 与 DiT 卸载的组合尚未开放。
逐层卸载允许显式 TeaCache：探针使用仅首层的执行计划，完整计算使用所有 block 的计划，
预取不超出计划且不在末端回绕；复制流分配的权重记录计算流使用，退出计划或异常时释放。
无缓存路径继续使用原有循环预取。`cache_dit` + 逐层卸载仍在执行请求前拒绝。
TeaCache 属于近似计算，支持组合不表示任意阈值均满足质量要求。
关闭所有卸载后，原有全驻留缓存、编译及 peer mesh 路径仍可使用。
单卡卸载集成仍拒绝旧 `use_fsdp_inference` 开关；DiT FSDP/HSDP 已在独立 NCCL 后端接入，
通过 `--dit-fsdp-shard-degree` / `--dit-fsdp-replicate-degree` 选择，见 [NCCL 并行](distributed_parallel_20260928.md)。
T5 的 FSDP 使用独立进程组，不受该旧开关控制。

## 统计口径

`memory_runtime.backend=sglang_source`。`resident_bytes` 和 `live_layers` 是采样时管理器
持有的 DiT block 权重；不含 allocator 等待释放的存储、非 block 权重、T5、VAE 或激活。
`observed_peak_resident_bytes` 只是采样最大值，不代表真实逐层峰值。
`weight_budget_scope=null`，不再报告预算等待或伪造 H2D 次数。
`pinned_cpu_bytes` 只统计 DiT 管理器，T5 FSDP 的 pinned 存储不在其中。

请求总峰值看 timing 的 allocated/reserved；组件观测中的 CUDA peak 是累计峰值，CPU peak
RSS 是进程历史峰值，均不是独立阶段峰值。初始化和模型加载单列。
迁移前的历史验收不能用作新实现的性能或组合保证。

迁移时的基线见 [2026-09-24 迁移验证](sglang_memory_validation_20260924.md)，
后续组合见 [单卡编译与融合](single_gpu_optimization_20260927.md)及[缓存检查](single_gpu_cache_audit_20260929.md)。
