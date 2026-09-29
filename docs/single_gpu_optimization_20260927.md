# 单卡方案实施与筛选（2026-09-27）

按照 [单卡优化方案](single_gpu_optimization_plan.md) 实施融合后 profiling、文本投影缓存筛选、
gated residual 精确融合，以及 resident FFN 编译筛选。未引入 SGLang 包。

## 实现范围

新增显式算子 `gated_residual`，将 `hidden + update * gate` 合并为一次 Triton launch。
保留原生 BF16 乘法结果的舍入，再执行独立 FP32 加法并写回 BF16；输出独立分配，输入不修改。
支持 batch 维的 gate stride，尾元素有 mask；非连续 hidden/update、非 BF16、非 CUDA、
不支持的 gate 形状或需要梯度时，auto 回退到原表达式，强制模式报错。
两处调用仍位于 block 内，逐层卸载 hooks 的执行边界保持不变。

新增算子不进入默认集合。原有 `--operator-fusion-backend auto` 仍只选择 QK RoPE 和 AdaLN。
需要实验时，在原有命令上指定：

```bash
--operator-fusion-backend auto \
--operator-fusion-ops qk_rmsnorm_rope,rmsnorm_adaln,gated_residual
```

## 融合后的瓶颈

A100 80GB，物理 GPU 3，真实 transformer 权重，固定随机输入 latent `[1,128,16,34,60]`，
32,640 token，BF16、SDPA、确定性配置、预计算 RoPE，QK RoPE/AdaLN 融合开启。
一次预热后采集 resident forward；只汇总 Chrome trace 中 `cat=kernel` 的持续时间，
不叠加 CPU aten 父事件，未包含逐层 H2D。

| 类别 | kernel 时间 ms | 占比 |
|---|---:|---:|
| Self-attention FlashAttention | 1293.20 | 62.0% |
| GEMM | 409.91 | 19.7% |
| 其他 | 382.34 | 18.3% |
| 合计 | 2085.45 | 100% |

这是单次算子采样，不是完整请求耗时分解。它表明剩余可通过逐元素融合节省的比例有限。

## 五组交替 forward 筛选

与上述形状相同，启用逐层卸载、预取一层；每条路径分别预热，按 A/B、B/A、A/B、B/A、A/B
交替运行。各候选独立对照 QK RoPE + AdaLN 基线，不把不同实验的绝对耗时交叉比较。
每次输出均与对应基线逐元素一致。

| 候选 | 基线中位数 ms | 候选中位数 ms | 耗时变化 | 结论 |
|---|---:|---:|---:|---|
| 文本投影/KV 缓存 | 2115.04 | 2116.63 | 增加 0.07% | 未发现提速，不改变默认 |
| Gated residual 融合 | 2118.04 | 2105.86 | 减少 0.57% | 小幅收益，保留显式实验选项 |

文本缓存每个 CFG 分支保存约 29.5 MiB（包含借用的 conditioning storage）；
两分支约 59 MiB。缓存可以命中，但这部分投影计算并非大形状的主要成本。
不能将缓存命中率当作性能改善的证据。

## FFN compile 筛选

物理 GPU 6，真实模型第一层 FFN；native linear、自定义算子舍入边界、
`emulate_precision_casts=True`、CUDA graphs 关闭，compile mode 为 `default`。
每个形状先编译，再做五组交替测量，每个样本连续调用十次并同步，表中为单次中位数。

| Token 数 | Eager ms | Compile ms | 输出逐元素一致 |
|---|---:|---:|---|
| 512 | 0.2642 | 0.3230 | 是 |
| 8160 | 2.4153 | 2.4048 | 是 |
| 32640 | 9.4567 | 9.4155 | 是 |

首形状编译/首次调用约 4.38 s，后两个形状约 0.36 / 0.24 s，均不计入稳态耗时。
大形状约 0.4% 的 FFN 收益不能支持投入卸载兼容改造，小形状反而更慢。
本轮因此停止该分支；这不代表穷尽全部编译模式，只表示所测精确路径没有充分收益。
没有放开 compile + offload 或 compile + fusion 的现有配置限制。

## 完整请求与短片验证

原片条件沿用上一轮：1920×1080、145 帧、窗口 121/重叠 9、seed=42、配置 50 步、
strength=0.8，每窗口实际 40 步；BF16、SDPA、三组件卸载策略不变。

| 配置 | 请求 s | 去噪 s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|---:|
| 上一轮 QK RoPE + AdaLN 基线 | 391.85 | 340.45 | 30.653 | 48.387 |
| 基线 + 文本投影缓存 | 384.62 | 339.67 | 30.653 | 48.422 |
| 基线 + gated residual | 389.55 | 337.31 | 30.653 | 48.387 |

这些是独立单次请求，基线及文本缓存使用物理 GPU 2，残差融合使用 GPU 3；
不包含模型加载，没有显式请求预热，Triton 磁盘缓存可能已存在。
有其他主机作业，VAE 编码等未修改阶段也存在耗时变化，因此不能将请求时差全部归因于候选。
特别是文本缓存的去噪时间基本不变，不能据此宣称缓存带来约 7 s 的提速。

文本缓存在两个窗口均正常关闭；每窗口每 CFG 分支 projection 命中 39 次、K/V 命中 1092 次，
两分支 retained storage 峰值合计 61,865,984 字节，约 59 MiB。
残差融合实际调用 8960 次，没有回退。两条候选结束时受管权重驻留均为零。
全请求显存峰值仍由 VAE 编码决定，残差融合没有改变这一峰值。

基线、文本缓存候选、残差融合候选均完整解码 145 帧，1920×1080、24000/1001 FPS，
RGB SHA256 同为 `62e8723b73fd90ae6aa60b74d426d056a232c9ba538c06152d7e83e1bfe70cce`，
即相对当前基线逐像素一致。两个候选是分别验证，未将其收益相加。
复现时使用 [上一轮完整命令](performance_optimization_20260927.md#原片-50-步验收)，
文本缓存候选将 `--no-cache-text-projections` 改为 `--cache-text-projections`；
残差融合候选保留缓存关闭并增加本文开头的 `--operator-fusion-ops` 选择，分别指定新的输出路径。

短片文本缓存另在 GPU 6 的同一 session 中测试：192×320、33 帧，50 个配置步、
窗口 25/重叠 9，各路径预热一次，再做五组交替请求。

| 配置 | 请求中位数 s | 请求范围 s | 去噪中位数 s |
|---|---:|---:|---:|
| 文本投影缓存关闭 | 26.9241 | 26.8791–26.9846 | 24.2341 |
| 文本投影缓存开启 | 26.9310 | 26.8850–26.9991 | 24.2176 |

十段视频全部完整解码，帧数、尺寸、帧率相同，RGB SHA256 完全一致。
短片同样未显示端到端收益，不据此更改文本缓存默认值。

## 下一阶段

本轮没有测得达到 3% 的大形状 forward 增益，不将实验候选加入默认配置。
后续更大收益需要评估 attention 内核或减少实际 DiT 执行次数。
残差缓存 + 卸载仍需独立实现执行计划、探针/跳步的预取管理和异常清理，
并进行多素材画质验证；当前仍拒绝这个组合，不能直接删除校验。

原始视频质量锚点仍需保留：本轮的精确性对照是上一轮显存优化后的基线，
不是更早的原始实现。完整请求的单次观测不能替代五组交替端到端统计。
本轮先完成真实形状的五组交替筛选；候选收益不足，尚未投入原片完整请求的五组 A/B，
因此不声称稳定的原片端到端提速。主机 RSS 本轮未重新采样。

## 回归

CPU 全量 112 项：92 通过、20 按设备或 opt-in 条件跳过。随后新增默认选择/拓扑契约检查，
最终 CPU 融合专项 6 项：3 通过、3 因无 CUDA 跳过。
GPU 融合、卸载和缓存专项 39 项全部通过；最终融合专项 6 项全部通过，
覆盖新增选择契约和融合 + 文本缓存 + 重复逐层卸载组合。
新增算子验证了不同 batch、奇数宽度、尾 token、不同幅值、gate 非连续 batch stride、
输入不被修改、布局回退和 autograd 回退。GPU 上逐元素比较使用零容差。

## 本地产物

`results/single_gpu_20260927/` 为忽略目录：

- `profile_dit.py`、`dit_trace.json`、`kernel_summary.json`：融合后的采样。
- `benchmark_textcache.py/.json`、`benchmark_residual.py/.json`：真实形状交替 forward。
- `benchmark_ffn.py/.json`：精确 FFN 编译筛选。
- `textcache50.log/.json/.mp4`、`residual50.log/.json/.mp4`：原片验证。
- `short_cache_tasks.json`、`short_cache.log`：短片交替缓存实验。
- `short_cache_summary.json`：五组完整短片请求及解码校验。
- `summarize.py`、`summary.json`：完整解码的 RGB 校验与计时汇总。
- `cpu_tests.log`、`gpu_tests.log`、`cpu_fusion_final.log`、`gpu_fusion_final.log`：回归记录。
