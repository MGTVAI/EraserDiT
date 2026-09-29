# 显存生命周期与静态条件复用（2026-09-27）

本轮保留迁移的 SGLang 逐层卸载管理器，不引入 `sglang` 包，也不改变采样、VAE tiling、
attention 后端或默认卸载配置。工作区原有的内存迁移修改继续保留。

## 实现

- 预处理单通道 mask 先补帧，再按最多 8 帧扩展 RGB 通道并做空间膨胀及视频遮罩。
  输出写入预分配存储，时间压缩和灰度归一化仍使用原始完整序列规则。
- 后处理参考图像以 uint8 存放在 CPU，参考 mask 的三通道用视图扩展。
  后处理直接在 CPU 转换参考图像，不再先搬回 GPU 再由颜色校正搬回 CPU。
- `eraserdit_model_frames` 保存模型输入帧数。VAE 输入归一化和 dtype 转换后，立即解除
  `padded_video`、`masked_video` 两个引用；去噪和后处理读取帧数元数据。
  编码结束释放 VAE 输入、posterior，mask 下采样后释放 `padded_mask`。
- T5 正负提示编码在同一请求内复用，最多保留一组 CPU 结果；提示词、长度、dtype、
  encoder/tokenizer 身份变化时重建。缓存随任务状态释放，不跨请求共享。
- 单 GPU 的 RoPE 在每个窗口内计算一次，供所有去噪步和两个 CFG 分支使用，
  窗口结束释放。多卡路径仍使用原有设备本地计算。

这些静态结果复用不跳过 DiT block，不属于 TeaCache/cache_dit，也不放宽其卸载限制。

## 原尺寸显存对照

A100 80GB，物理 GPU 2，Torch 2.6.0+cu126，BF16、确定性配置、SDPA；
`data/model`、`data/113000356.mp4` 及对应 mask，1920×1080、145 帧、24000/1001 FPS。
seed=42，strength=0.8，窗口 121/重叠 9，两个窗口；DiT 预取参数 0（一层），
T5 FSDP 与 VAE CPU offload。未启用 compile、量化、残差缓存或文本投影缓存。

先以 5 个配置步（每窗口实际 4 步）比较改动前与仅显存优化的版本，两次独立进程：

| 配置 | 请求时间 s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|
| 本轮改动前 | 100.05 | 34.882 | 56.502 |
| 仅显存生命周期优化 | 82.70 | 30.653 | 49.240 |

allocated 降低 4.228 GiB（12.1%），reserved 降低 7.262 GiB（12.9%）。
时间是单次观测，不能视为稳定加速比；不包含模型加载。allocated/reserved 均为
PyTorch 进程指标，不是整张卡或 CPU 内存。

首窗口阶段边界的 allocated 变化：

| 观测点 | 改动前 GiB | 仅显存优化 GiB |
|---|---:|---:|
| VAE 编码进入 | 6.916 | 5.513 |
| VAE 编码结束 | 4.607 | 0.255 |
| DiT 进入 | 4.622 | 0.270 |
| VAE 解码进入 | 6.837 | 2.485 |

预处理完成前的累计峰值由 20.224 降至 17.586 GiB。全请求峰值仍在首窗口 VAE 编码期间
达到；上述边界观测不应误读为各阶段的独立峰值。没有重置外部持有的峰值计数器。

两个视频全部解码为 RGB24，各为 902,016,000 字节，SHA256 完全相同：
`cf4fd6584fbb7ca0ed674f14acce34a3f5ee3d67af58f53385af1050e70149ae`。

CPU 阶段采样的历史 peak RSS 从 53.95 到 54.62 GiB，本轮并没有降低主机内存；
参考图像转到 CPU 是显存与主机内存之间的取舍，视频 preload 仍然存在。

## 完整 50 步验证与像素差异归因

保持原片及采样配置，每窗口实际去噪 40 步。因为最终版本与历史基线的 RGB 哈希不同，
追加了改动前 stage 快照与仅显存优化两个对照，结果如下：

| 配置 | 请求 s | peak allocated GiB | peak reserved GiB | RGB SHA256 前缀 |
|---|---:|---:|---:|---|
| 改动前 stage 复跑 | 425.59 | 34.882 | 56.502 | `eb849312b1ca` |
| 仅显存优化 | 412.22 | 30.653 | 49.240 | `62e8723b73fd` |
| 显存优化 + 静态条件复用 | 415.33 | 30.653 | 48.387 | `62e8723b73fd` |

改动前复跑完整哈希与 9 月 24 日历史产物一致；仅显存优化与最终版本的完整哈希一致：
`62e8723b73fd90ae6aa60b74d426d056a232c9ba538c06152d7e83e1bfe70cce`。
因此这次 50 步差异已定位到显存优化组合，静态条件复用没有引入额外像素差异。
尚未对显存组合内的各项改动逐一归因，不将差异未经验证地归咎于某个 CUDA 算法。

与原始基线逐帧比较：RGB SSIM **0.996333**、MSE **0.497063**、MAE **0.203259**，
满足项目整片门槛，但**不是逐像素一致**。前 109 帧一致，后 36 帧存在差异；
后 24 帧分段平均 SSIM 约 0.9832，不应将整片均值解释为每帧都达到 0.985。
两侧均完整解码 145 帧、1920×1080、24000/1001 FPS。未做人工播放验收。

这三项为独立进程单次测量，改动前 stage 复跑使用 GPU 3，其余使用 GPU 2；
消融请求有并发运行及共享主机争用。不能据此宣称原片稳定速度提升，或将 412.22 与
415.33 秒的差别归因于静态复用。最终版本的阶段采样 CPU 历史 peak RSS 为 54.75 GiB。

保存的 `before_*.py` 是本轮开始时未修改的预处理与 stage 的 Git HEAD 快照；
`benchmark_before.py` 将这些方法挂回当前 loader/内存迁移实现，以保留工作区原有改动。
结果为 `before50.*`、`memory50.*`、`ablation50.json`、`historical_quality.json`。

## 静态条件复用的性能消融

物理 GPU 3，同一模型会话，192×320（宽×高）、33 帧，50 个配置步，strength=0.8，
窗口 25、重叠 9，每请求两个窗口。两组均包含显存优化，均使用原有默认逐层卸载。
分别完整预热一次，然后按 A/B、B/A、A/B、B/A、A/B 顺序执行，各五次。
A 使用保存的复用前 text/denoising stage，B 使用最终静态条件复用实现。

| 配置 | 请求中位数 s | 范围 s | 最大 allocated GiB |
|---|---:|---:|---:|
| A：不复用静态条件 | 29.496 | 28.908–29.870 | 2.683 |
| B：请求内 T5 + 窗口内 RoPE 复用 | 28.574 | 26.899–28.623 | 2.684 |

这组样本中请求中位数降低 3.1%。每次 B 请求均记录一次 `text_encoding_cache_hit`。
DiT 去噪中位数仅由约 25.857 到 25.762 秒，收益主要来自减少重复文本编码；
本轮尚未解决 DiT 大部分计算成本。

十个正式输出 RGB SHA256 全部相同：
`af6079cb1000471ec444b329bc0815df596d8d5257c08f4b512e38dcdb392e17`。
短片结果不能外推 1080p。测试期间 GPU 2 同时执行原片验证，其他 GPU 也有作业，
共享主机带宽及 CPU 条件并非独占，特别是末尾样本耗时出现下降。

复现脚本为 `results/memory_optimization_20260927/benchmark_static.py`，对应
`memory_only_{denoising,text_encoding}.py` 是消融用 stage 快照，任务列表为 `perf_tasks.json`。
它们只用于本地验证，不提供产品级运行时切换开关。

## DiT 计算诊断

额外在物理 GPU 6 加载真实 DiT 权重，以原片首窗口 latent 形状
`[1,128,16,34,60]`、128 个文本 token 的随机输入，预热一次后采集一个全驻留 forward。
BF16、SDPA、已预计算 RoPE；这不是整请求基准，也不包含逐层搬运。

Chrome trace 的 CUDA kernel 总时长约 2252.7 ms，其中自注意力的 PyTorch FlashAttention
kernel 为 1283.2 ms（约 57%），两项主要 BF16 GEMM 合计约 403.8 ms（约 18%）。
因此当前 SDPA 已经走 Flash kernel，不能把改一个后端名称当作确定的优化收益。
进一步提速需要对注意力后端、融合/编译、并行进行真实形状消融；本轮静态条件复用
没有触及最大的自注意力成本。

统计只累加 trace 中 `cat=kernel` 的事件，没有将 `aten` 父事件与 CUDA kernel 重复计数。
诊断脚本和 trace 为 `profile_dit.py`、`dit_trace.json`、`dit_kernel_summary.json`。

## 回归

- CPU 全量：107 项，90 通过，17 因 GPU 或显式 opt-in 条件跳过。
- GPU 卸载、缓存、生命周期与静态复用专项：40 项全部通过。
- Python 语法检查及 `git diff --check` 通过。
- mask 数值对照覆盖首/后窗口、镜像补帧、单/三通道及两种膨胀分支；
  文本缓存验证请求隔离、提示/长度变化与输入修改；RoPE 复用覆盖不同窗口形状。

## 复现与产物

从仓库根目录运行（GPU 编号按空闲情况选择）：

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
  /mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model --video-input data/113000356.mp4 \
  --mask-input data/113000356_mask.mp4 --output-path /tmp/eraserdit-memory.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 \
  --dit-layerwise-offload --text-encoder-cpu-offload --vae-cpu-offload \
  --dit-offload-prefetch-size 0 --attention-backend sdpa \
  --transformer-cache-mode off --no-cache-text-projections
```

本地产物位于 `results/memory_optimization_20260927/`（Git 忽略）：
`baseline.*` 为改动前的 5 步基线；`memory.*` 为仅显存优化的 5 步对照；
`final50.*` 为最终实现的完整 50 步验证。

汇总为 `summary.json`（包括 `matches_historical50=false`，没有掩盖差异）；
`ablation50.json` 记录追加的 50 步对照；`environment.json` 记录硬件与并发条件。

原生小块 VAE tiling 的历史画质失败仍然有效；本轮没有启用该路径，也不声明已支持
更低显存卡。进一步降低峰值需要优化 VAE 激活。
