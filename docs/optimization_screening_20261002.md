# 编译、预取与量化候选筛选

本轮在 L40S / Torch 2.6.0+cu126 上使用当前真实 checkpoint 权重。
固定合成激活用于隔离内核成本；以下不是完整视频速度或质量承诺。

| 候选 | 同条件结果 | 本轮决定 |
| --- | --- | --- |
| NCCL rank 内 native Linear FFN compile | CFG1/SP2，长窗口 2.7594→2.7543 s，短窗口 0.5455→0.5417 s；三组交替、逐元素一致；首次编译 forward 10.25 s | 不开放 NCCL 编译入口：热态收益不足 1%，范围与噪声重叠 |
| 单 FFN CUDA Graph（包含输入 copy 与输出 clone） | 16320-token eager 5.955 ms、graph 6.270 ms；5100-token 1.912/1.975 ms，逐元素一致 | 不引入 graph 生命周期和静态输出别名管理，当前没有收益 |
| 单卡卸载预取 1/2/4 层 | 长窗口 2.5877/2.5915/2.5840 s，峰值 2.512/2.762/3.263 GiB；短窗口 0.5661/0.5652/0.5647 s | 保留一层；增加驻留权重没有稳定提速，输出逐元素一致 |
| INT8 Linear | 16320-token Q projection 0.648→1.025 ms；FFN up 2.637→3.778 ms；FFN down 2.519→2.476 ms | 不扩大 NCCL INT8 支持；大部分候选矩阵的量化/后处理抵消 GEMM 收益 |
| FP8 Linear，原生动态 scale | 包含 FP32 转换/amax/cast 后明显慢于 BF16 | 淘汰该实现 |
| FP8 Linear，Triton 行最大值及量化 | FFN up 2.599→1.943 ms；Q 0.656→0.659 ms；FFN down 2.618→2.546 ms | 只对 up projection 继续真实模型筛选 |
| 28 层 FFN up FP8 替换 | 五组交替：长窗口 2.5544→2.5453 s（0.36%），短窗口 0.5502→0.5424 s（1.42%）；预测相对 RMSE 1.48%/1.44% | 不集成：整模型收益很小且引入近似；不宣称已通过视频质量验收 |
| 外部 FlashAttention / SageAttention | 当前环境没有安装对应扩展；当前 SDPA 已使用 Flash CUDA 内核 | 维持已有显式后端探测和 SDPA 默认；本轮不将未测外部扩展列为已优化 |
| 空 mask / ROI / 稀疏 token | 模型有全局 attention；窗口输入包含跨窗 raw tail；原后处理在 mask 外也使用模型输出 | 不以源帧复制、裁剪或丢弃 token 替代原路径，这些会改变现有数值与窗口契约 |

预取数据是一个分支 forward，不含文本/VAE、视频 IO 或完整 CFG。
Graph 数据是一个 FFN，不可与 DiT forward 直接比较。量化误差是模型预测误差，不是视觉擦除评价。
所有候选均保留原默认值；已存在的 peer 编译、INT8 和 attention 接口保持可用。

原始脚本、逐次样本和结果位于：

- `outputs/nccl_compile_screen_20261002/`，`entrypoints.cli.benchmark_dit --variants fusion_direct,compiled`
- `outputs/ffn_graph_screen_20261002/`
- `outputs/offload_prefetch_screen_20261002/`
- `outputs/quant_kernel_screen_20261002/`（包含低效 FP8 与融合 FP8 两轮，以及完整模型 forward）

本轮 VAE、后处理与流式路径另见[融合/内存记录](fusion_memory_optimization_20261002.md)。
