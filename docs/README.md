# 文档索引

从[项目 README](../README.md#快速开始)完成首次安装和推理。以下指南描述当前代码，
带日期的实验记录保留当时的环境、参数、质量目标与测量结果。

本轮 L40S 24 GiB 验收见[最新记录](l40s_validation_20261004.md)和[收敛计划](l40s_completion_plan.md)。
前序整体优化见[推进记录](optimization_progress_20261002.md)。

## 使用与开发

| 文档 | 内容 |
| --- | --- |
| [安装与部署](setup.md) | Python 3.10 / uv、统一依赖安装顺序、模型下载、容器与排错 |
| [CLI](cli.md) | 单视频、任务 JSON、缓存、编译、peer/NCCL 多卡与 DP |
| [服务 API](service_api.md) | 启动、路径输入/上传、队列、进度、取消和结果下载 |
| [性能配置](performance.md) | 当前默认值、优化组合边界、质量与性能口径 |
| [测量与验证](validation.md) | 环境检查、CPU 回归、完整视频与同条件性能对照 |
| [配置说明](../config/README.md) | 模型、请求、服务与运行配置的职责 |
| [代码架构](architecture.md) | 模块、执行流程、依赖约束与 NCCL 进程边界 |
| [SGLang 内存管理](sglang_memory.md) / [逐层卸载](layerwise_offload.md) | 源码来源、预取、T5/VAE 卸载与统计口径 |
| [回归测试](../tests/README.md) | CPU/GPU 测试、专项开关与跳过条件 |
| [开发路线](roadmap.md) | 后续性能、质量、低显存与交互入口工作 |

## 实验与验证记录

先看最近的相关记录，再按需要追溯实现。单次筛选、稳态重复、不同帧数或质量目标的结果不能直接排名。
记录中的 `results/` 路径指本机产物，不随 Git 分发；克隆仓库后不能假设实验脚本、视频或冻结源码存在。
复跑旧记录需恢复对应源码和环境；当前使用命令以 CLI 和安装指南为准。

| 日期 | 记录 | 范围 |
| --- | --- | --- |
| 2026-10-04 | [L40S 24 GiB 验收](l40s_validation_20261004.md) | 1/2/4 卡逐卡峰值、缓存初筛、WebUI 与取消恢复 |
| 2026-10-04 | [Attention 性能优化](attention_optimization_20261004.md) | Sage FP8 与完整 Q/K RMSNorm + RoPE 融合、整请求对照 |
| 2026-10-04 | [AdaLN 与残差优化](adaln_optimization_20261004.md) | 完整 RMSNorm + AdaLN 融合、门控残差组合与投影筛选 |
| 2026-10-04 | [FP8 性能优化](fp8_optimization_20261004.md) | GELU 融合、静态激活缩放与快速累加、完整 FFN/DiT 和视频对照 |
| 2026-10-03 | [INT8 GELU 融合](quantization_gelu_fusion_20261003.md) | 保持量化输出的激活融合、完整 FFN/DiT 测量及卸载与编译验证 |
| 2026-10-03 | [Ulysses 分块重叠](ulysses_overlap_20261003.md) | GPU kernel 重叠证据、head 分块筛选、数值/生命周期和整片配对 |
| 2026-10-03 | [CUDA IPC 边界传输](cuda_ipc_boundary_20261003.md) | 去除每步 CPU 张量中转、GPU 所有权与进程生命周期、同条件视频对照 |
| 2026-10-02 | [同卡数 DP 筛选](dp_topology_20261002.md) | 四卡单 worker 与双 worker 的冷批次吞吐、单条延迟、私有内存及输出差异 |
| 2026-10-02 | [融合与内存验收](fusion_memory_optimization_20261002.md) | NCCL 融合/direct 打包、VAE 激活、后处理及 uint8 流式组合 |
| 2026-10-02 | [残差缓存与局部探针](nccl_cache_quality_20261002.md) | 七项视频筛选、逐帧 mask/边缘指标与视觉检查 |
| 2026-10-02 | [编译、预取与量化筛选](optimization_screening_20261002.md) | 真实权重 FFN compile/Graph、预取、INT8/FP8 及 ROI 语义审查 |
| 2026-10-02 | [NCCL 文本缓存与 DiT 剖析](nccl_text_cache_profile_20261002.md) | 常驻 CFG/Ulysses 文本 K/V 复用、单步 trace 与交替 A/B |
| 2026-10-02 | [帧转换与输出缓存优化](frame_copy_optimization_20261002.md) | 减少整窗临时分配、仅文件输出与单窗口对照 |
| 2026-10-02 | [内存、卸载与拷贝优化](memory_optimization_20261002.md) | VAE 阶段驻留、CPU 权重复用、121 帧对照与显存限额 |
| 2026-10-02 | [单窗口并行与打包优化](window_optimization_20261002.md) | 121 帧、各五次；CFG/SP 配置对照，Ulysses 打包 A/B 与文件一致性 |
| 2026-09-30 | [单卡 24 GiB 显存预算](memory24_20260930.md) | VAE 算子分批、默认权重卸载、原尺寸整片限额验证 |
| 2026-09-30 | [双卡 VAE 端到端验证](vae_e2e_validation_20260930.md) | 完整视频各三次、逐卡峰值降低；整片 SSIM 0.98313，未达到 0.99 |
| 2026-09-29 | [双卡 VAE 空间并行](vae_spatial_validation_20260929.md) | 局部卷积、边界交换、编解码性能与逐卡峰值、数值及接缝验证 |
| 2026-09-29 | [单卡缓存检查](single_gpu_cache_audit_20260929.md) | TeaCache / CacheDiT 实现、完整擦除与单素材视觉检查 |
| 2026-09-29 | [NCCL 边界优化](nccl_boundary_optimization_20260929.md) | owner 输出汇聚、CFG 合并、返回字节与分段计时 |
| 2026-09-28 | [NCCL 并行实施与验收](distributed_parallel_20260928.md) | DP/CFG/SP/USP/TP/FSDP/HSDP，整片 SSIM ≥0.985 |
| 2026-09-28 | [完整 DiT 编译](full_transformer_compile_20260928.md) / [双卡 CFG 编译](cfg_compile_20260928.md) | 整图编译、冷/热启动与双卡执行 |
| 2026-09-28 | [组件编译](component_compile_20260928.md) / [解码器整片验证](decoder_full_validation_20260928.md) | T5/VAE 实验、DiT 与 decoder 组合 |
| 2026-09-27 | [单卡实施](single_gpu_optimization_20260927.md) / [融合性能](performance_optimization_20260927.md) | 文本复用、FFN 编译、精确融合与原片对照 |
| 2026-09-27 | [内存优化](memory_optimization_20260927.md) | CPU/GPU 内存生命周期与峰值 |
| 2026-09-27 | [0.99 目标](quality99_optimization_20260927.md) / [0.985 目标](quality985_optimization_20260927.md) / [0.98 目标](quality98_optimization_20260927.md) | 各轮注意力、编译与尾窗筛选；目标不同 |
| 2026-09-27 | [多卡缓存与量化](multi_gpu_cache_quant_20260927.md) / [四卡优化](four_gpu_optimization_20260927.md) | peer 并行、缓存与 INT8 组合 |
| 2026-09-24 | [内存迁移验证](sglang_memory_validation_20260924.md) | SGLang 源码卸载、重复请求与显存 |
| 2026-09-23 | [旧逐层卸载验证](layerwise_offload_validation_20260923.md) / [旧组合验收](composable_acceleration_validation_20260923.md) | 迁移前实现，部分参数已删除 |
| 2026-09-22 | [初始优化矩阵](optimization_validation_20260922.md) / [缓存阈值 0.3](cache_threshold_03_validation_20260922.md) | 迁移前性能与质量对照 |

## 阶段方案

[单卡方案](single_gpu_optimization_plan.md)、[可组合加速方案](composable_acceleration_plan.md)
和[四卡之后的方案](next_optimization_plan_20260927.md)保留设计过程。
其中 NCCL、完整 DiT 与组件编译已在后续记录中落地；未完成事项以[开发路线](roadmap.md)及最近记录为准。
