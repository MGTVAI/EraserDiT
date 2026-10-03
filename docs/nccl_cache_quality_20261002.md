# NCCL 残差缓存与局部探针验收

实现沿用既有 EraserDiT TeaCache/CacheDiT 控制器，新增常驻 NCCL CFG/Ulysses 接入。
CFG 分支独立持有 FP32 残差，SP 组同步分子/分母及最终决策；每窗口重置，异常时清理。
TP、Ring、FSDP 仍不支持残差缓存。默认关闭，文本缓存仍可独立选择。

`--cache-probe-metric mask_frame_max` 计算每帧全图、latent mask 内、一圈空间边缘的相对 L1，
先跨 SP 求和分子与分母，再取所有区域/帧的最大值。空 mask 仍保留全帧检查。
它能避免小目标或短暂变化被全窗口平均稀释，但不能保证任意素材的擦除质量。
默认 `global` 保留旧策略，不改变 TeaCache 系数或阈值。

## 正确性

- CFG2、SP2、CFG2×SP2 多步和跨窗口测试通过，强制实算与同拓扑无缓存输出逐元素一致。
- 实际进程池验证请求参数传递、guided prediction、SP 决策、窗口清理及故障后进程回收。
- 全局均值低于 0.01 的单帧小目标变化，局部探针测试测得 1.0；不等长 SP shard 汇总与完整输入相同。
- 回归包含空区域、非有限值、形状校验、分支隔离及原有缓存控制器行为。

## 完整视频筛选

L40S×4、CFG2×Ulysses2、BF16/SDPA，融合与 direct 打包开启；VAE low-memory/组件卸载、
分块 FP32 后处理；关闭文本投影缓存、编译、量化和 tiling。
121 帧固定切片、50 配置步、strength=0.8（40 实际步）、seed=42、同一 owner/worker 会话。
每项一次，用于筛选，不作为稳定耗时排名。

| 配置 | 纯推理 | 去噪 | CFG 分支复用步 / 80 | mask PSNR 对无缓存 |
| --- | ---: | ---: | ---: | ---: |
| 无缓存 | 76.666 s | 63.072 s | 0 | — |
| TeaCache global，0.3 | 49.345 s | 35.709 s | 36 | 44.65 dB |
| TeaCache mask_frame_max，0.3 | 50.437 s | 36.818 s | 36 | 同 global，视频 SHA 相同 |
| CacheDiT global，0.3，前1/后0 | 50.805 s | 37.153 s | 36 | 44.55 dB |
| CacheDiT mask_frame_max，0.3，前1/后0 | 50.705 s | 37.051 s | 36 | 同 global，视频 SHA 相同 |
| TeaCache mask_frame_max，0.03 | 63.702 s | 50.048 s | 18 | 48.64 dB |
| CacheDiT mask_frame_max，0.3，前1/后1 | 51.339 s | 37.694 s | 36 | 45.69 dB |

0.3 下，本素材局部约束仍未改变交替实算/复用节奏。0.03 增加实算并更接近无缓存输出；
后部实算一个 block 也提高 mask/边缘 PSNR，但未降低本次非运动补偿的全图时间差误差。
PSNR 和命中率均不作为擦除成功的自动门槛。

已查看覆盖 0–120 所有帧的 16 张逐帧 mask 区域缩放图，以及 0/30/60/90/120 整帧概览。
这些图中，四个不同输出候选都未见可辨认的人体、头部、肢体残留或明显黑块，背景及水花纹理有差异。
第 102 帧以后 mask 为空，后续水花无缓存参考也保留。本次未做原分辨率实时播放或多素材评审，
不据此自动启用缓存或宣称无细微闪烁。

记录：`outputs/nccl_cache_quality_20261002/`，包含每 rank 报告、逐帧区域指标、视频、
`visual_review.json` 和对照图。使用的冻结源码是 `outputs/memory_cache_e2e_20261002/source/`，
驱动为其中 `run_cache_screen.py`；FFmpeg 线程策略显式设为 auto，与既有 121 帧参考一致。
前一个混合内存实验采用 4 个编码线程，编码文件不能直接与 auto 参考比较；后续已统一重跑。
