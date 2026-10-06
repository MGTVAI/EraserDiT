# README 失败项修复与复测（2026-10-05）

范围：补齐 V1、V3、V5，并使用相同权重驻留策略更新 V0、V2、V4 对照。
B0 按本次用户要求不处理，保留原始 OOM 记录。所有性能数据沿用 README 的
1080p、145 帧、seed 42、50 步×strength 0.8、完整尾窗、一次 1-step 预热加一次正式请求口径。

## 失败原因

V1、V3、V5 原始日志分别位于
`results/readme_retest_20261004/V{1,3,5}_superseded/run.log`。
三项均在加载模型前的 `validate_memory_config()` 报错：

```text
ValueError: SGLang offload currently requires single-GPU EraserDiT
```

基准入口的公共参数启用了 `--text-encoder-cpu-offload --vae-cpu-offload`。
VAE 并行度为 2 或 4 时，`memory/validation.py` 禁止任意已启用的组件卸载策略；
因此没有进入预热或正式推理，更没有可填写的阶段耗时。这是基准配置组合错误，
不能据此认定 VAE 空间并行实现执行失败，也不能通过删除校验来解决。
只关闭 VAE 卸载仍不够：文本编码器卸载同样会使 resource policy 处于 enabled 状态。

原 V0/V2/V4 的 VAE degree=1 可通过校验，但继续使用组件卸载；
将它们与关闭通用组件卸载的 V1/V3/V5 比较，会同时改变并行度和权重搬运策略。
所以这次六个 V profile 统一关闭 DiT 逐层卸载、DiT/文本编码器/VAE 组件卸载以及
VAE tiling，保留 `vae_low_memory`，仅比较约定的 DiT 拓扑和 VAE 并行度。

这里关闭的是 `dit_cpu_offload` / `dit_layerwise_offload` 等通用资源策略，
不代表所有 DiT rank 全程常驻。M1–M4、V0–V5 的实际 worker 报告均显示：
rank 0 为 `shared_rank_idle_cpu`，其他 rank 为 `resident`。
`pipelines/runtime/dit_executor.py` 在配置显存上限时，为与主进程共用 GPU 的 rank 0
启用独立的 `DiTIdleResidency`：窗口开始时载入完整权重，40 个去噪步内保持 GPU 驻留，
窗口重置时释放 GPU 权重并恢复 CPU 引用。CPU 权重副本始终保留，因此释放时无需 D2H 拷贝，
下一个窗口仍有 H2D 载入。这项策略在六组 VAE 对照中一致，已包含在实测耗时中。

常驻模型、激活和多进程的实际显存需求超过原来的 22 GiB allocator 上限。
入口为 V0–V5 默认选择每进程 44 GiB 上限、46 GiB 采样目标；其他 profile 仍为
22/24 GiB，显式命令行预算仍优先。采样目标不是物理显存扩容，也不代表达到
24 GiB 部署目标；实际逐卡进程树峰值需要单独报告。

## 复测与证据

- V0/V1 复用已完成的修正配置结果：`results/readme_retest_20261004/V0/`、`V1/`。
- V2 的前一次重跑留下 48 字节不完整视频、无完成报告，且本次开始时进程已不存在；
  该目录原样保留，不纳入统计。
- V2–V5 使用新目录 `results/readme_repair_20261005/` 串行运行。
- 六组推理源码均来自同一份 `V1/source` 快照，核对 manifest 的逐文件 SHA-256；
  不将文档或入口预算默认值调整混入推理源码消融。
- 每个结果核验退出码、一个正式请求、两窗各 40 步、实际 VAE degree、源文件未变、
  采样无外部进程及错误、输出哈希、FFprobe 规格和 FFmpeg 全片解码。
- VAE 实际执行以请求报告的 `parallel_history` 中的
  `vae_parallel_encode/decode.effective_degree` 为准；顶层 runtime metadata 中的
  `native_vae_parallel` 字段描述另一条分布式运行路径，不能单独用于判断本次空间并行是否生效。

所有六组均完成一次 1-step 预热及一次完整正式请求。单位：秒；显存为进程树 NVML 峰值（GiB），含加载和预热。

| 配置 | 预热 | 请求 | 纯模型 | 文本 | VAE 编码 | DiT | VAE 解码 | 各卡峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V0 | 62.45 | 243.37 | 215.73 | 0.48 | 16.46 | 188.46 | 10.32 | GPU0: 31.35 / GPU1: 7.09 |
| V1 | 54.98 | 234.58 | 206.31 | 0.54 | 10.41 | 188.59 | 6.77 | GPU0: 30.26 / GPU1: 17.99 |
| V2 | 68.19 | 336.60 | 310.22 | 0.30 | 16.19 | 283.46 | 10.27 | GPU0: 31.36 / GPU1: 11.83 |
| V3 | 57.73 | 329.10 | 300.56 | 0.29 | 10.93 | 283.25 | 6.07 | GPU0: 30.26 / GPU1: 19.92 |
| V4 | 63.13 | 208.55 | 181.34 | 0.49 | 17.39 | 152.71 | 10.75 | GPU0: 31.58 / GPU1: 11.84 / GPU2: 11.98 / GPU3: 11.94 |
| V5 | 54.10 | 196.51 | 168.31 | 0.30 | 8.67 | 153.27 | 6.06 | GPU0: 30.50 / GPU1: 12.78 / GPU2: 13.04 / GPU3: 12.74 |

VAE 并行相对同拓扑 VAE1 的本次单样本变化：

- V0 → V1：VAE 编码 1.58×，解码 1.52×；请求耗时变化 -8.79 秒（-3.61%）。
- V2 → V3：VAE 编码 1.48×，解码 1.69×；请求耗时变化 -7.50 秒（-2.23%）。
- V4 → V5：VAE 编码 2.01×，解码 1.77×；请求耗时变化 -12.04 秒（-5.77%）。

以上不代表多次重复统计或质量等价；VAE 空间切分会改变卷积形状及归一化布局。
输出规格和全片解码验证只证明文件完整有效，不证明与单卡逐像素一致。

完整汇总：`results/readme_repair_20261005/readme_results.json`；逐项审计：
`results/readme_repair_20261005/completeness.json`。除 B0 按用户要求不处理外，27 项通过。
V0–V5 无外部 GPU 进程污染、采样错误或冻结源码变化；并行组两窗都实际执行请求的 VAE degree。
8 项配置与显存预算回归通过，日志：`results/readme_repair_20261005/regression.log`。

## 复现

从仓库根目录执行，选择全新结果目录；V1/V3 用空闲双卡，V5 用空闲四卡。

```bash
uv run --no-project python -m entrypoints.cli.benchmark_l40s \
  --profile v1 --devices 0,1 --repeats 1 --run-dir results/my-vae2-cfg2
uv run --no-project python -m entrypoints.cli.benchmark_l40s \
  --profile v3 --devices 0,1 --repeats 1 --run-dir results/my-vae2-sp2
uv run --no-project python -m entrypoints.cli.benchmark_l40s \
  --profile v5 --devices 0,1,2,3 --repeats 1 --run-dir results/my-vae4
```

对应 VAE1 对照 profile 是 `v0`、`v2`、`v4`。默认预算为本机 L40S 常驻实验设置，
按用户补充要求，超过 24 GiB 不作为失败或剔除条件，只记录实测峰值。
回归检查覆盖六组实际生成的 CLI、配置校验、原始冲突拒绝、
单卡预算和用户预算覆盖：

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-project python -m unittest tests.test_l40s_profiles tests.test_allocator_budget -v
```
