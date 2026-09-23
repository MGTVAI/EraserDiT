# DiT 逐层卸载验证（2026-09-23）

实现说明见 [逐层卸载](layerwise_offload.md)，配置见 [性能说明](performance.md#offload)。

## 环境与范围

使用 `/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python`，A100 80GB，完整视频在物理 GPU 2
运行。Torch 2.6.0+cu126、BF16、SDPA、确定性配置、无 compile、无量化、无缓存。
`data/model`，原始视频与 mask 为 `data/113000356*.mp4`，1920×1080、145 帧、24000/1001 FPS。
seed 42、50 个配置步、strength 0.8，每窗口实际 40 步，两个窗口。
DiT 权重预算 2 GiB、预取 1。原有媒体重构工作区修改保持原状。

## 正确性和回归

- CPU 回归：102 项，82 项通过、20 项按 GPU 条件跳过（新增用例加入前）。
- 单 GPU 全量回归：112 项，105 项通过、7 项按多卡/INT8 opt-in 条件跳过。
- 最终专项回归：10 项全部通过，覆盖预算、非连续布局、跳层/重复调用、实际缓存命中、
  部分 H2D 失败、计算异常、阶段切换、关闭恢复、重新注册和 FSDP 组合拒绝。
- 真实小 Transformer 的无缓存、TeaCache、cache_dit 输出与对应全驻留参考逐元素一致。
- 原始完整视频全部解码成功，145 帧、1920×1080、24000/1001 FPS。
  最终输出 RGB SHA256 与旧 dynamic_offload 输出相同：
  `eb849312b1cab539f73135c2e281af34ac34a2c5278ebcd8e9ce5a49169a5951`。
- `git diff --check`、修改模块的 Python 编译检查通过。

## 峰值与耗时

| 配置 | 请求耗时 s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|
| 2026-09-22 旧 extent 历史对照 | 409.31 | 33.64 | 57.07 |
| 新逐层后端，未做阶段缓存清理 | 423.35 | 33.64 | 65.71 |
| 新逐层后端，阶段边界清理空闲缓存（最终） | 420.29 | 33.64 | 51.19 |

最终 reserved 比旧历史值低约 10.3%，比未做阶段清理的新后端低约 22.1%。
allocated 峰值出现在 VAE 编码阶段，继续调整 DiT 预取不会消除该激活峰值。
这些都是单次请求，主机还有其他 GPU 作业，未进行五次交替测量；不能据此声称提速。
最终请求纯推理计时 393.86 s，加载 30.46 s（不含在请求耗时内）。

最终 DiT 受管权重峰值 1.376 GiB，低于 2 GiB 预算；请求结束 resident 为 0，
live layers 和 pending releases 均为 0。两个窗口累计 4480 次 block 搬运。
CPU pinned DiT 权重为 3.502 GiB；最大 block 为 128.09 MiB。

完整模型初始化的加载前、加载后、注册后 PyTorch allocated/reserved 及峰值均为 0；
初始化搬运数为 0，所有模型参数留在 CPU。该数字不包含 CUDA context 等非 PyTorch 分配。
最终视频进程的初始化 CPU 峰值 RSS 27.36 GiB；独立启动检查为 27.14 GiB。
这只描述加载期，视频缓存与预处理还会增加请求期 CPU 占用；
本次阶段记录已观察到 48.30 GiB 的进程历史 CPU 峰值 RSS，并非只有 27.36 GiB。
直接从 checkpoint 写入最终 pinned 存储尚未实现，CPU 加载峰值仍有优化空间。

## 预取对比

使用真实 28 层 DiT 权重和 256 token 合成输入，单卡 GPU 3，预热一次后各测三次。
每个配置输出与同输入的全驻留模型逐元素一致。

| 预取数 | 单次 forward 中位数 ms | peak allocated GiB | 受管权重峰值 GiB |
|---|---:|---:|---:|
| 0 | 164.19 | 1.257 | 1.126 |
| 1 | 177.33 | 1.382 | 1.251 |
| 2 | 163.99 | 1.507 | 1.376 |
| 4 | 178.23 | 1.757 | 1.626 |

小输入结果不能外推到 1080p 视频，也未证明增大预取有稳定收益，默认保持 1。
预取数不等于实际持有层数：CPU 可以提前提交多个 block，待完成的 buffers 仍计入预算。

## 本地产物

- [最终复现命令](../results/layerwise_offload_20260923/trim_command.sh)
- [最终日志](../results/layerwise_offload_20260923/trim_run.log)
- [结果与内存统计](../results/layerwise_offload_20260923/trim_result.json)
- [最终视频](../results/layerwise_offload_20260923/trim_output.mp4)
- [全量 GPU 回归](../results/layerwise_offload_20260923/final_gpu_tests.log)
- [最终专项回归](../results/layerwise_offload_20260923/layerwise_tests.log)
- [独立启动统计](../results/layerwise_offload_20260923/startup.json)
- [预取基准](../results/layerwise_offload_20260923/prefetch_benchmark.json)

`results/` 为本地忽略目录。多卡卸载、量化卸载、compile 卸载、T5 逐层卸载和 VAE 激活
优化均未在本次启用，仍受原有组合限制约束。
