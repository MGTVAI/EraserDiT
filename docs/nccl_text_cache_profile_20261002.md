# NCCL 文本缓存与 DiT 单步剖析

本轮先补齐 NCCL 计算内部观测，并将已有文本投影/K/V 缓存接入常驻 rank。
默认不启用文本缓存，也不改变 attention、GEMM、采样步数或数值精度。

## 实现边界

- `--cache-text-projections` 支持常驻 CFG、Ulysses 及其组合；TP、Ring、FSDP 仍拒绝该组合。
  NCCL 的残差缓存、编译、量化和算子融合限制保持不变。
- 窗口首步将设置传给 worker；每个 CFG 分支独立缓存 caption projection 与各层 cross-attention K/V。
  窗口结束/切换清空引用和统计，窗口中途不允许切换模式。
- 正常原地修改或替换文本相关参数时检查版本并失效；条件张量变化沿用原文本缓存的版本检查。
  直接 `.data` 写权重不在支持的更新方式内。进程池失败时销毁 worker，避免复用损坏状态。
- rank report 新增 `text_cache`，记录分支命中数和保留张量峰值字节。

## 诊断与复现

设置 `MGERASE_DIT_PROFILE_DIR` 后，每窗采集一个去噪步，默认第 2 步；
`MGERASE_DIT_PROFILE_STEP` 从 1 计数。未设置目录时不创建 profiler，不安装模块 hooks。
仅采集期间安装 hooks，正常退出和异常均移除。各 rank 导出 trace 与算子汇总，路径写入 report。

trace 标记各层投影/norm/attention/FFN，以及 Ulysses 的 QKV 拼接、输入/输出打包、
all-to-all 与收包重排。完整 trace 可分析 kernel、memcpy 和等待；采集只覆盖 rank forward 与最终汇聚，
不覆盖父进程 scheduler、输入广播、视频 IO 或 VAE。
父子模块区间重叠，各 rank 并行，NCCL kernel 可含对端等待，不能将它们相加成端到端延迟。
profiler 影响调度，诊断请求不进入性能 A/B。
算子汇总的 CPU/CUDA annotation 可同名，使用 `device_type` 区分；benchmark 汇总标记 `diagnostic_only`。

```bash
# 正式性能对照：每种模式五次，交替 off/on、on/off，同一常驻 session。
uv run --no-project python -m entrypoints.cli.benchmark_window \
  --run-dir outputs/my_text_cache_ab --devices 0,1,2,3 \
  --configs cfg2_sp2 --repeats 5 --text-cache-ab

# 单独诊断；不要将采集耗时用于速度排名。
uv run --no-project python -m entrypoints.cli.benchmark_window \
  --run-dir outputs/my_dit_profile --devices 0,1,2,3 \
  --configs cfg2_sp2 --repeats 1 --profile-step 2
```

benchmark 记录源码指纹（含未提交 Python 文件）、命令、输入信息、逐次输出 SHA256、
完整请求/纯推理/去噪时间和各 rank 报告。完整请求从 CLI 调用墙钟扣除显式预热；模型加载单列。

## 首次真实权重剖析

L40S 四卡 CFG2×SP2，BF16 SDPA、sharded Linear、reference packing、DiT 常驻，
T5/VAE CPU offload、VAE low-memory；121 帧，未减尾窗填充。
单独使用 5 配置步数、strength 0.8（4 个有效步），采集第 2 步，文本缓存关闭。
这次短去噪仅用于瓶颈定位，不用于正式画质或速度结论。

从 trace 的真实 kernel 事件看，每个 rank 的 self-attention Flash kernel 累计约 472–480 ms，
GEMM 类 kernel 约 247–253 ms，其他 kernel 约 655–656 ms；后者包含逐元素、归约和布局操作。
NCCL kernel 约 264–513 ms，差异包含 rank 等待。分类按 kernel 名称，不能当作互斥阶段的墙钟分解。
本轮优先级因此转向纯计算融合与布局处理，再研究通信重叠；仅凭 NCCL kernel 时长不能决定改通信算法。

文本 K/V 的两次投影在 28 层累计仅约 1.3 ms/rank（不含其他文本处理和 CPU launch），
所以文本缓存主要先补齐组合能力，不能预设会显著缩短完整请求。

本机产物位于 `outputs/nccl_text_cache_20261002/`，含冻结源码、GPU/进程采样、四个 rank 的 trace、
`profile_analysis.json` 和完整视频；这些产物不随 Git 分发。

## 完整单窗口 A/B

物理 GPU 0–3，同一常驻 session，按 off/on、on/off 交替，共五对、十次正式请求。
1920×1080、121 帧，seed 42、guidance 3、50 配置步数、strength 0.8，每次实算 40 个去噪步。
其余配置与诊断相同；正式对照关闭 profiler、残差缓存、编译、融合、量化、tiling 和尾窗减填充。
只在首个请求前预热一次（2 配置步数），预热 38.532 秒从完整请求墙钟中扣除，模型加载另计。

| 指标 / 秒 | 文本缓存关闭，中位数（范围） | 文本缓存开启，中位数（范围） |
| --- | ---: | ---: |
| 纯推理 | 84.619（84.339–85.000） | 84.591（84.212–84.920） |
| 去噪 | 70.972（70.688–71.343） | 70.953（70.566–71.261） |
| 完整请求，不含加载/预热 | 104.887（103.795–106.084） | 105.079（104.618–105.782） |

纯推理中位数仅下降 0.028 秒（约 0.03%）。配对的开启减关闭差值中位数为：
纯推理 −0.131 秒、去噪 −0.174 秒、完整请求 +0.002 秒；五对中纯推理三对变快、两对变慢。
**没有证明稳定的纯推理或端到端提速，保持默认关闭，不作为速度推荐。**

- 十个视频文件 SHA256 全部一致，也与之前单窗口参考一致：
  `21203e33ed3b6fc11973897181f30b74a1b27606a0558d7554bdf775e8964354`。
- 每个缓存 rank/分支：39 次文本投影命中、1092 次 K/V 命中，保留张量峰值 30,932,992 字节（29.5 MiB）。
  关闭模式的所有 rank 均无缓存条目；报告核对覆盖全部 40 份 rank/window 记录。
- 最大 worker peak allocated：关闭约 5.345–5.346 GiB，开启约 5.373 GiB。
  owner peak allocated 均为 20.842 GiB，阶段末 RSS 为 30.2996–30.3003 GiB。
  RSS 是固定阶段采样点，不是主机峰值，也不是进程树 PSS。
- 第一次关闭时 worker 最大 reserved 约 10.455 GiB，启用后为 10.982 GiB，随后关闭也保留该分配器缓存。
  不把 reserved 增长直接当作仍存活的文本缓存，也不将 owner 数字当作整卡占用。
- 每 15 秒采样 GPU/进程；未观察到 0–3 卡新增外部任务。机器未锁频，同主机其他卡有既有进程，
  结果仅代表本素材、硬件和配置，不外推其他任务或拓扑。

正式测量之后，仅给 profiler 汇总补充设备类型、benchmark 补充诊断标记及对应测试；
推理计算未变。差异文件记录在 `post_validation_source_changes.json`，冻结源码保存在 `source/`。

最终验证：CPU 全套 186 项，138 通过、48 条件跳过；真实 GPU 两项专项通过，内部覆盖
CFG2、Ulysses2、CFG2×Ulysses2 的同拓扑精确对照，以及常驻进程池跨窗/故障清理。
另做 CUDA profiler smoke，输出逐元素一致，hooks 正常清理，CPU/CUDA 设备类型正确记录。
`git diff --check` 通过。以上测试与十次正式视频对照分别保存，不把条件跳过记为 GPU 验收。

## 下一项实施依据

文本缓存已打通但不构成当前主要加速来源。优先验证 NCCL 下原生数值边界保持的 RoPE/AdaLN 融合，
其次减少 QKV 拼接、打包和收包重排；通信缓冲池需同时衡量长期驻留字节与实际等待。
norm 原生归约的精度边界仍需保留，不能因其耗时高就直接替换归约算法。
scheduler/CPU IPC 迁移、Ring 改造与更激进近似缓存不因本轮结果提高优先级。
