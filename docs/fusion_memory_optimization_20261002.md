# NCCL 融合、通信打包与内存优化

环境：8×L40S，正式配对使用 GPU 0–3，Torch 2.6.0+cu126、Triton 3.2.0。
输入为示例视频的前 121 帧无损固定切片；CFG2×Ulysses2、sharded Linear、BF16/SDPA，
50 配置步、strength=0.8（40 实际步）、seed=42。关闭文本/残差缓存、编译、量化和空间 tiling，
启用既有 VAE low-memory 与组件卸载。

## 已验收的计算优化

- NCCL rank 内支持 `qk_rmsnorm_rope,rmsnorm_adaln` 融合。保留原 RMSNorm 归约与 dtype
  舍入边界，只融合 RoPE/AdaLN 逐点计算。每窗口报告实际调用及 fallback，重置统计与缓存。
- `MGERASE_NCCL_PACKING=direct` 将不同 stride 的 Q/K/V 直接写成目的 rank 优先的
  all-to-all 布局，避免整块 QKV `cat` 和第二次打包。支持不等长 shard，保留原通信顺序。
- 支持组合限常驻 CFG/Ulysses SP1/2/4；TP、Ring、FSDP 尚未验收该融合组合。

真实权重固定输入筛选中，32640-token 四卡 forward 中位数为 reference 1.790 s、
fusion 1.587 s、direct 1.728 s、组合 1.529 s。四者输出逐元素一致。

完整视频使用同一 owner 会话，按 AB/BA 交替完成五组。每次切换在空闲状态重建 DiT workers，
独立执行两次同形状预热；加载、重建、预热均单独记录，不计入正式请求。

| 中位耗时 | Reference | Fusion + direct | 减少 |
| --- | ---: | ---: | ---: |
| 去噪 | 72.575 s | 62.776 s | 13.50% |
| 纯推理 | 86.147 s | 76.434 s | 11.27% |
| 端到端请求 | 103.733 s | 94.224 s | 9.17% |

十个输出文件 SHA256 均为 `21203e33ed3b6fc11973897181f30b74a1b27606a0558d7554bdf775e8964354`。
每 rank、每窗口有 1120 次 QK/RoPE 和 2240 次 RMSNorm/AdaLN 融合调用，无运行时 fallback。
这是单素材、单 seed、固定硬件的验收，不承诺其他拓扑的同等提速。

复现选项：在既有 NCCL CFG2/SP2 命令上增加
`--operator-fusion-backend triton --operator-fusion-ops qk_rmsnorm_rope,rmsnorm_adaln`，
并设置环境变量 `MGERASE_NCCL_PACKING=direct`。默认保持关闭。

原始记录与冻结源码：`outputs/nccl_fusion_direct_screen_20261002/`、
`outputs/nccl_fusion_direct_e2e_20261002/`（manifest、源码哈希、逐请求指标、GPU 采样、视频）。

## 内存筛选

`MGERASE_POSTPROCESS_CHUNKED_FP32=1` 将 BF16 解码结果按颜色校正块转为 FP32，
避免整窗口 FP32 副本；最终 FP32 patch、两次 uint8 截断及 CPU FP32 raw tail 契约不变。
121×1080p 合成解码输入、三组交替测量，输出 patch 与 raw tail 哈希完全一致，
该阶段峰值 allocated 从 7.766 GiB 降至 5.057 GiB（约 2.709 GiB）。
这是后处理阶段峰值，不能直接当作整请求峰值降低量；完整双窗口输出已验收，见下表。

`--vae-chunk-elements` 暴露既有 low-memory 的目标元素预算（不是硬字节上限，
完整帧/卷积邻域是下限）。默认保持 16777216。真实权重、16×34×60 latent 解码对照中，
4/8/16/32 Mi 元素预算的峰值 allocated 均约 18.389 GiB；改变预算触发不同 BF16 卷积
数值，未见峰值收益，故不推荐更改默认值。

内存筛选脚本及结果：`outputs/memory_screen_20261002/`。
第一次后处理测量脚本曾保留上一轮输出视图，已修正重测；只使用
`postprocess_corrected.log` 和 `postprocess.json` 的结果。

## VAE 激活与流式路径验收

`MGERASE_VAE_INPLACE_ACTIVATIONS=1` 配合 `--vae-low-memory`，在无梯度模式复用
归一化/卷积的私有输出存储，执行 SiLU、调制与残差加法；保留原 BF16 两次舍入，
不改写调用方输入。下采样残差按输出帧分块，保留完整 channel/stride 归约组。
训练/梯度路径回退原实现。真实 checkpoint 三组配对中，编码输出完整 moments、解码
输出均逐元素一致：编码峰值 20.696→19.063 GiB，解码 18.389→14.903 GiB。
编码分块依据实际分配剖析定位；额外尝试的 Python 引用转移没有收益，未保留。

`--runtime-mode windowed_streaming --streaming-cache-dtype uint8` 使用既有分块缓存，
避免旧 BF16 缓存对源帧和提交帧的量化。mask 使用与 preload 相同的 RGB 首通道及
全视频最大值阈值；软/暗 mask 通过有界预扫描确定阈值，遇到 255 可提前结束。
扫描支持取消并关闭 reader。257 帧、多窗口、亮度变化的软 mask 测试验证逐帧次序、
缓存帧数上界及输出文件与 preload 一致。默认仍为 preload，流式 dtype 默认仍为 bf16。

145 帧原始素材、两个完整 121 帧窗口、同一会话与 seed、融合/direct 打包、残差缓存关闭：

| 路径 | Owner 峰值 allocated | 输出 |
| --- | ---: | --- |
| Preload 参考 / 仅后处理分块 | 20.844 GiB | 相同 |
| uint8 streaming + 后处理分块 | 20.844 GiB | 相同 |
| Preload + VAE 激活优化 + 后处理分块 | 19.211 GiB | 相同 |
| uint8 streaming + VAE 激活优化 + 后处理分块 | 19.211 GiB | 相同 |

五个文件 SHA256 均为 `4b48862b1fc4f9f360762d14e048bff5b1dabd2e1ae3169e739baeb6323bc528`。
整片 owner 峰值减少 1.633 GiB（7.84%）。这是单进程 allocated，不能当作包含 DiT workers、
NCCL/CUDA 上下文的整卡占用。流式主要限制随视频长度增长的 CPU 帧缓存；145 帧素材不足以
证明超长视频的实测内存曲线，缓存上界由上述多窗口测试覆盖。
本组视频使用默认 4 个 FFmpeg 编码线程，与前文 auto 线程的 121 帧试验不能跨组比较文件哈希。

记录：`outputs/memory_screen_20261002/` 中的 `vae_residual_encode.json`、`vae_inplace.json`；
`outputs/memory_cache_e2e_20261002/` 保存原 preload 与后处理对照；
`outputs/vae_stream_acceptance_20261002/` 保存三条新路径、冻结源码、逐请求指标和进程树采样。

## 近似缓存

NCCL 常驻 CFG/Ulysses 支持 TeaCache/CacheDiT，沿用分支独立状态、SP 决策同步、窗口重置。
强制实算通过 CFG2、SP2、CFG2×SP2 精确对照及真实进程池测试。
`--cache-probe-metric mask_frame_max` 取逐帧全图、mask 内和一圈边缘的相对 L1 最大值，
先跨 SP 汇总分子分母再计算比值；空 mask 仍检查每帧全图。默认 `global` 保持原行为。
七项完整视频对照及逐帧局部质量审查见[缓存验收](nccl_cache_quality_20261002.md)。
探针是近似复用约束，不是擦除质量保证，缓存默认关闭。
