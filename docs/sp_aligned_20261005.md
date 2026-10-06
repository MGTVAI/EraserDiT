# SP2 / SP4 选择性 GEMM 保护

显式 `--sp-linear-mode aligned` 减少 reference 的重复全长计算，默认仍为 reference。
实现位于 `layers/sequence_linear.py`、`SequenceRank.linear_scope()`；CLI、服务启动参数和
`benchmark_l40s` 的 `sp2_aligned` / `sp4_aligned` profile 均已接入。

## 数值边界

- 首尾投影保留全长；其他符合筛查条件的 BF16 投影直接计算本地 token。
- SP4、10200 token 时单独保护 FFN 下投影，避免保护整个 FFN。
- 仅对 L40S、PyTorch 2.6.0 / CUDA 12.6、确定性模式、当前 EraserDiT 2048 宽度、
  batch 1、32640 / 10200 token 启用。未覆盖的环境、模型、形状或激活 dtype 沿用 reference。
- 仅支持常驻 NCCL Ulysses SP2/4；拒绝 peer、TP、Ring、FSDP、编译与量化组合。
- 包装在 forward 退出或异常时恢复，补零工作区不跨 forward 保留。

前置真实权重探针检查了 226 个视频侧 Linear 的实际 forward 输入激活。
SP2 的两个长度均需保护最终输出投影；SP4 短序列还需保护 28 层 FFN 下投影。
`proj_in` 的现有全长计算保持不变。探针记录位于
`outputs/sp_source_probe_20261005/forward_activation_probe.json`，它不替代完整视频验收。

`sp_linear_policy` 报告最近一次分支 forward 的实际模式、回退原因、`local_calls`、
`reference_calls` 和 `protected_ffn_down_calls`。这些是包装调用次数，不是请求 GEMM 总数。
初版微基准曾被最后一个 FP32 输出投影覆盖 effective 标记；当前按是否实际执行本地调用记录，
该修正不改变数值路径。微基准的 `sp_linear_mode` 配置记录正确。

## 交替 forward 测量

真实权重、固定合成输入、CFG1，每次包含正负两个分支及输出汇聚；BF16/SDPA、原生数值
边界融合、direct 打包。基线已包含 reference 补零工作区复用。每次候选输出均与同输入
reference 逐元素一致；各轮正反顺序交替。SP2 各 3 次，SP4 各 5 次，中位数单位为秒。

| 拓扑 | Token | reference，未分块 | aligned，未分块 | aligned + 输入/输出重叠 | 重叠块数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SP2 | 32640 | 3.2281 | 2.7299 | 2.6580 | 4 |
| SP2 | 10200 | 0.6971 | 0.5430 | 0.5187 | 4 |
| SP4 | 32640 | 2.3785 | 1.6589 | 1.5515 | 4 |
| SP4 | 10200 | 0.6263 | 0.4600 | 0.4452 | 2 |

单独 aligned 的耗时下降为 SP2 长/短 15.4% / 22.1%、SP4 长/短 30.3% / 26.6%。
联合重叠相对未分块 reference 为 17.7% / 25.6%、34.8% / 28.9%，不能全归因于 GEMM。
SP4 短序列四块为 0.4744 s，慢于两块；不将四块设为通用默认。

记录：`outputs/sp_aligned_20261005/sp2_screen/`、`sp4_screen/`。
这些是 forward 筛选，不是完整请求加速比；加载、VAE、视频读写不在测量范围内。

最终另做同通信配置的配对，隔离新增 GEMM 优化：两侧均为四块输入/输出重叠，
SP2/SP4 各五组交替、当前运行时源码指纹未变化，全部输出逐元素一致。

| 拓扑 | Token | reference + 四块重叠 | aligned + 四块重叠 | 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| SP2 | 32640 | 3.1570 | 2.6636 | 15.63% |
| SP2 | 10200 | 0.6725 | 0.5194 | 22.77% |
| SP4 | 32640 | 2.1964 | 1.4881 | 32.25% |
| SP4 | 10200 | 0.6063 | 0.4586 | 24.36% |

优先用此表评估新增 GEMM 保护策略的收益。记录：`sp2_final/`、`sp4_final/`。
最终 SP4 短序列报告每分支 28 次 FFN 下投影保护，确认该保护路径实际执行。
短序列选两块的结论来自上一轮同组筛选，不跨轮次直接比较耗时。

## 完整素材验收

完整 145 帧横屏与 120 帧竖屏、seed 42、各窗口 40 实际去噪步，不缩短尾窗，不启用残差缓存。
输入、mask、prompt、seed 与 `results/l40s_completion_20261004/cfg2_sp2_reference_quality/`
中的对应 BF16 参考一致。比较完整视频 SHA256，不用单算子误差替代输出验收。

SP2、SP4 均使用四块输入/输出重叠，四个输出都与上述参考逐字节一致。
请求耗时排除加载和预热，单位为秒：

| 拓扑 | 素材 | 请求 | DiT 去噪 | 最高单卡任务峰值 GiB |
| --- | --- | ---: | ---: | ---: |
| SP2 | 横屏 145 帧 | 278.623 | 224.381 | 21.527 |
| SP2 | 竖屏 120 帧 | 148.847 | 113.308 | 21.527 |
| SP4 | 横屏 145 帧 | 181.933 | 127.267 | 21.613 |
| SP4 | 竖屏 120 帧 | 101.520 | 63.987 | 21.613 |

显存是同卡任务进程合计峰值，覆盖加载、预热和同一进程池全部请求，不是每个请求的独立峰值。
采样无错误、目标卡无外部进程，冻结源码未变化。其他 GPU 有现存服务，整机不是独占。
每个素材仅一次请求，此处不声称稳定端到端加速比，也不外推到其他 seed 或任意视频。

记录：`outputs/sp_aligned_20261005/sp2_video/`、`sp4_video/`，包含冻结源码、命令、输入指纹、
视频、逐卡进程显存采样和逐请求报告。`audit_sp2_video_sp4_video.json` 核对输入指纹、
prompt/seed、输出哈希、媒体元数据、显存与源码。SP2 冻结运行时与当前一致；SP4 的差异
仅为上述统计修正及说明，不改变数值计算。SP2 报告每个分支有 196 次本地调用与 1 次保护调用。

## 复现

```bash
# forward 配对；SP4 将可见卡改为 0,1,2,3，并使用 --sp 4。
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
uv run --no-project python -m entrypoints.cli.benchmark_dit \
  --run-dir outputs/sp2_aligned_forward --cfg 1 --sp 2 --sp-linear-mode reference \
  --variants fusion_direct,aligned,aligned_heads2_output,aligned_heads4_output \
  --latent-frames 16,5 --repeats 5

# 完整请求；目录必须不存在。SP4 改为 sp4_aligned 和四张可见物理卡。
MGERASE_ULYSSES_HEAD_CHUNKS=4 MGERASE_ULYSSES_OUTPUT_OVERLAP=1 \
uv run --no-project python -m entrypoints.cli.benchmark_l40s \
  --run-dir outputs/sp2_aligned_video --profile sp2_aligned --devices 0,1 --repeats 5
```

基准自动设置 direct 打包。直接调用推理 CLI 还需 `MGERASE_NCCL_PACKING=direct`，并显式
设置 `--cfg-degree 1 --sp-degree 2 --sp-linear-mode aligned`；四卡纯 SP 使用 `--sp-degree 4`。
服务启动参数同名。不能直接用 sharded 的历史计时作为 reference 基线。

回归覆盖 `test_sequence_linear` 的六项选择性保护、未知形状回退、嵌套异常恢复、统计和
拓扑检查，以及 `test_sequence_padding`、`test_l40s_profiles`、`test_mesh`、
`test_parallel_compatibility`、`test_service_api` 和 NCCL 配置验证。真实多卡数值与生命周期
由上述配对 forward 及复用进程池的完整视频请求验证；文档链接和差异格式检查通过。
