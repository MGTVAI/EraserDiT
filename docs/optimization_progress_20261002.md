# 整体优化推进记录

范围为本轮讨论的计算、内存、通信、并行与近似策略。每项以实现并验证，或有证据的淘汰收尾；
不将仅有实验开关、历史不同条件的快慢排名或未验收的质量变化记为完成。
保留默认精度与行为，候选通过独立验收后才考虑推荐。

| 项目 | 状态 | 验收/下一步 |
| --- | --- | --- |
| NCCL DiT 内部剖析、配对基准 | 已实现并验证 | [记录](nccl_text_cache_profile_20261002.md)；单卡卸载/短窗口筛选见下文 |
| NCCL 文本 K/V 缓存 | 已验收，未证明稳定提速 | 输出精确一致、默认关闭 |
| NCCL 原生数值边界保持的融合 | 已实现并完成配对验收 | 五组视频 A/B 去噪降低 13.5%、端到端降低 9.2%，十个视频哈希一致 |
| Ulysses 拼接/打包、收发缓冲 | direct 打包已实现并验证 | 一次 Triton 写入通信布局，去掉整块 QKV cat；不额外缓存大通信缓冲 |
| NCCL 计算子图编译/CUDA Graph | 筛选完成，未采用 | FFN 编译收益不足 1%；含 copy/clone 的 FFN graph 更慢；[证据](optimization_screening_20261002.md) |
| NCCL 残差缓存及局部质量保护 | 已实现并完成单素材验收 | CFG2/SP2/CFG2×SP2 测试及七项视频对照通过；逐帧局部指标与质量/速度记录见 [报告](nccl_cache_quality_20261002.md) |
| VAE 激活复用及残差分块 | 已实现并验收 | 145 帧双窗口输出精确一致，owner 峰值 20.844→19.211 GiB；预算筛选保留 16 Mi 元素默认 |
| 后处理 FP32 激活生命周期 | 分块转换已实现并完成单素材验收 | 121×1080p 后处理峰值 7.766→5.057 GiB；145 帧双窗口视频哈希一致 |
| 单卡预取与权重缓冲 | 筛选完成，保留一层 | 1/2/4 层延迟基本相同，4 层额外占 0.75 GiB；不新增驻留缓冲 |
| 长视频流式与输出管线 | uint8 缓存已实现并验收 | 257 帧测试缓存上界；145 帧真实双窗口及 VAE 组合与 preload 视频逐字节一致 |
| DP 吞吐与设备拓扑 | 同四卡双任务筛选完成 | DP2 冷批次吞吐 +26.1%，单条延迟 +30.8%，USS 32.7→60.2 GiB；[完整口径](dp_topology_20261002.md) |
| attention/INT8/FP8 候选 | 本环境筛选完成 | INT8 大部分矩阵更慢；优化 FP8 在真实长窗口仅快 0.36% 且有预测误差；外部 attention 未安装 |
| 空 mask/ROI/稀疏候选 | 语义审查完成，未采用 | 与全局 attention、mask 外输出和跨窗 raw tail 契约不等价 |

已完成的前序优化及实验边界见 [performance](performance.md) 与[记录索引](README.md)。

## 使用本轮精确优化

下列配置对应已验收的四卡常驻 CFG2×SP2 路径；保留默认完整尾窗、BF16/SDPA 与缓存关闭。
按实际分配选择 GPU。默认行为没有自动切换为实验策略。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 \
MGERASE_NCCL_PACKING=direct MGERASE_POSTPROCESS_CHUNKED_FP32=1 \
MGERASE_VAE_INPLACE_ACTIVATIONS=1 \
uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/optimized.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --dit-parallel-backend nccl --cfg-degree 2 --sp-degree 2 --sp-linear-mode sharded \
  --no-dit-layerwise-offload --no-dit-cpu-offload \
  --text-encoder-cpu-offload --vae-cpu-offload --vae-low-memory \
  --attention-backend sdpa --operator-fusion-backend triton \
  --operator-fusion-ops qk_rmsnorm_rope,rmsnorm_adaln \
  --transformer-cache-mode off --no-cache-text-projections \
  --num-inference-steps 50 --strength 0.8 --guidance-scale 3 --seed 42
```

需限制长视频 CPU 帧缓存时附加 `--runtime-mode windowed_streaming --streaming-cache-dtype uint8`。
真实视频精确验收与资源统计边界见[融合/内存记录](fusion_memory_optimization_20261002.md)。
近似缓存应单独选择阈值，不能将单素材速度与视觉检查推广为所有素材的质量保证。

## 回归

最终 CPU 回归：200 项，146 通过、54 按 CUDA/专项开关条件跳过。
GPU 专项已覆盖 VAE 激活输入不变与数值、后处理、直接打包、CFG/SP 融合与缓存共识、
真实进程池窗口重置和故障清理；完整视频验收另记，CPU 跳过项不计为 GPU 通过。

日志归档：`outputs/optimization_final_validation_20261002/`。本轮各项已完成实现验收或候选筛选，
未测外部 attention 扩展和跨素材/跨硬件推广明确保留为范围外，不作为已验证收益。

## 2026-10-03 后续推进

已实现可选 CUDA IPC owner/worker 边界，接收端独立 GPU 存储隔离生命周期。
三组同条件视频配对均精确一致，端到端中位数 95.384→94.605 s（约 0.82%），收益小，默认 CPU。
GPU 生命周期/故障检查及 CPU 回归通过，详见[实施记录](cuda_ipc_boundary_20261003.md)。
上述 CPU IPC 阶段没有完成计算通信重叠、更大 NCCL 编译子图、IO 流水或扩大质量验证。
后续输入通信重叠的实际筛选结论见下节。

## 2026-10-03 输入通信重叠筛选

已实现 Ulysses head 分块、专用 stream、事件依赖及显式缓冲回收，SP2/CFG2×SP2/SP4 的
正确性和生命周期检查通过，GPU trace 证实实际重叠。但最终整片筛选：参考 95.628 s、
两块 97.498 s、四块 99.847 s，输出均精确一致，当前 L40S 不推荐启用，默认保持 1 块。
详见[实验记录](ulysses_overlap_20261003.md)。本项以有证据的未采用结论收尾；输出通信流水、
IO 流水和更大 NCCL 编译子图仍未完成。
