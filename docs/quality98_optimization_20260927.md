# 单卡编译与注意力后端：RGB SSIM ≥ 0.98（2026-09-27）

用户将整段 RGB SSIM 门槛调整为 **0.98**，允许编译、FlashAttention、SageAttention。
三者不是必须同时启用：编译作用于 FFN；每次 self-attention 选择一种后端。
原片仍为 1920×1080、145 帧、50 配置步、strength=0.8、seed=42；窗口 121、overlap 9，
开启已实现的尾窗口减填充。分辨率、输出帧数、帧率与采样步数不变。

## 新接入

- 单 GPU 逐层卸载可搭配局部 FFN compile。编译预热前逐层加载并等待复制 event，
  用 `record_stream` 保护计算流中的权重；正常和异常退出均释放。正式推理仍使用原有 eager 卸载 hook。
- 编译边界仅为 FFN，CUDA graphs 关闭；允许与编译区域外的 QK RoPE、AdaLN 融合共存。
  SP compile 的融合限制保留，多卡卸载和 INT8 卸载仍拒绝。
- `MGERASE_COMPILE_LINEAR_BACKEND=inductor` 允许优化 Linear/激活组合，默认仍为 `native`。
  对近似路径必须测试最终组合视频，不能把各单项 SSIM 当作组合质量。
- 每个新窗口形状提前预热 FFN，日志 `preparation_seconds_total` / `preparation_history`
  记录编译模型生命周期内的累计预热时间及各形状耗时。请求时间包含本次发生的预热。

SGLang 仅用作源码参考，不作为 Python 包导入。

## FFN 筛选

真实第一层 FFN 权重、BF16，物理 A100 GPU 3，Inductor `max-autotune-no-cudagraphs`。
每种形状五组交替顺序、每组十次，GPU 同步后用墙钟计时：

| tokens | eager ms | compile ms | 耗时减少 |
|---|---:|---:|---:|
| 10200 | 3.053 | 2.893 | 5.2% |
| 32640 | 9.591 | 9.272 | 3.3% |

此处仅表示 FFN 部分收益；随机输入输出非逐像素一致，不能代替视频验收。
首次形状预热约 24.25 s，第二形状约 1.19 s；磁盘编译缓存状态会影响冷启动时间。

## 完整视频筛选

质量参考统一为 `results/performance_20260927/fused50.mp4`，即未减填充的 SDPA 融合基线。
SSIM 使用 RGB 三通道、11×11 Gaussian、sigma=1.5、reflect border，逐帧等权平均。
整段平均门槛不保证每帧达到 0.98；另记录最低帧和时序误差。

| 配置 | 请求 s | 去噪 s | 整段 SSIM | 达到 0.98 |
|---|---:|---:|---:|---|
| SDPA + 减填充，上轮同卡复跑 | 240.915 | 201.709 | 0.992833 | 是 |
| SageAttention 2 + 减填充，上轮结果重判 | 235.614 | 197.404 | 0.982647 | 是 |
| FlashAttention 2 + 减填充，本轮 | 238.248 | 200.219 | 0.982683 | 是 |
| FA 2 + Inductor FFN + 减填充，GPU 2 | 241.725 | 202.793 | 0.982654 | 是 |
| Sage 2 + Inductor FFN + 减填充，GPU 3 | 250.095 | 205.719 | 0.982664 | 是 |

前三项峰值 allocated 都为 **30.653 GiB**，reserved **48.387 GiB**。
FA 实际使用 `flash_attn`，Sage 使用 `sage_attn`，均无回退；cross-attention 保持 SDPA。
FA 版本 2.8.3；Sage 候选为隔离构建的 v2.2.0（安装环境中原有 1.0.6 未覆盖），
构建方法与源码提交见 [上一轮记录](quality985_optimization_20260927.md)。

完整视频是单次筛选，不是五组稳态统计；模型加载单列，其他 GPU 存在主机作业。
FA 与前两项在物理 GPU 2 运行；编译组合记录各自设备与预热成本，避免将微小波动当作确定收益。

编译组合包含当前进程两个窗口形状的首次预热：FA 为 **4.699 s**，Sage 为 **6.509 s**。
已有磁盘编译缓存，因此不代表从空缓存开始编译的成本。两者峰值 allocated 均为
**30.653 GiB**，但 reserved 分别升至 **51.434 / 50.799 GiB**。

FA 的长/短窗口稳态单步中位数从 **4200.548 / 784.830 ms** 降到
**4171.339 / 779.367 ms**，约减少 0.7%；首个 step 不计入这些中位数。
不能把预热直接从总时间中扣掉就当作另一次实测请求。首次请求包含预热时并未加速。
当前整段筛选中最快的达标项仍是 **Sage 2 + 减填充，不开启编译**。

FA、FA+编译、Sage+编译最低帧 SSIM 分别为 **0.961256 / 0.961112 / 0.961105**，
相邻帧误差变化 MAE 分别为 **1.038984 / 1.040164 / 1.041500**（0–255 RGB）。
这些候选达到整段 0.98；不表示每帧 0.98，也没有进行整段人工播放验收。

补充同卡完整 DiT 对照：GPU 2、真实权重、随机输入、Sage 2、两种图外融合和逐层卸载，
每种形状预热后五次交替测量。长窗口 eager **2048.764 ms**、compile **2048.972 ms**，
短窗口 eager **392.172 ms**、compile **391.171 ms**。整体收益接近零，
因此不将编译纳入当前速度推荐。编译兼容性代码和显式选项保留。

当前推荐 Sage 2 + 减填充另与显存优化前的 `before50.mp4` 比较，整段 SSIM
**0.982655**，也达到 0.98。相对未减填充的 391.849 s 基线，235.614 s 减少
约 **39.9%**，约 **1.66×**；相对最近 SDPA 减填充 240.915 s 额外减少 **2.2%**。

## 使用

普通单卡保持 SDPA；允许本轮近似程度时，可按下面命令复现 Sage 2 候选。
`PYTHONPATH` 指向上一轮隔离构建好的 v2.2.0，不会把当前安装的 1.0.6 误当作 V2：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
PYTHONPATH="$PWD/results/quality985_20260927/SageAttention" \
python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/sage2-tail.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 --infer-len 121 --overlap 9 \
  --dit-layerwise-offload --text-encoder-cpu-offload --vae-cpu-offload \
  --dit-offload-prefetch-size 0 --attention-backend sage_attn \
  --operator-fusion-backend auto --transformer-cache-mode off \
  --no-cache-text-projections --compact-tail-padding
```

测试 FA 时把后端换成 `flash_attn`；测试编译组合时增加 `--enable-torch-compile`，
并设置 `MGERASE_COMPILE_LINEAR_BACKEND=inductor`。每种配置使用不同输出路径。
编译已可用，但本轮结果不足以推荐在单次任务中默认启用；新素材需要重新验证最终视频。

## 验证

GPU 相关专项 51 项通过，包含 native/Inductor 编译下反复卸载、两种形状切换、
与常驻编译输出逐像素一致、推理不额外重编译、预热异常释放及重试恢复。
CPU 全量 120 项，96 通过、24 按设备或 opt-in 条件跳过。
实验产物位于 `results/quality98_20260927/`；此前阈值报告保留原始判定。
