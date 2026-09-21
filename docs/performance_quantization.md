# EraserDiT 单卡 INT8 量化

2026-09-21，用户要求以单卡推进量化，视觉结果大致相同即可。
本轮使用物理 GPU 2（A100-SXM4-80GB）；TeaCache、Cache-DiT、文本投影缓存关闭。
质量依照 [最新验收标准](optimization_acceptance.md)，数值只做诊断，不用无损硬阈值判定量化失败。

## 实现

`--transformer-quantization int8_w8a8_native` 选择对称 INT8 W8A8：
权重按输出通道静态量化，激活按 token 动态量化；scale 使用 FP32。
执行路径是 Triton 激活量化、`torch._int_mm` INT8×INT8→INT32、Triton 反量化与 bias 融合，输出 BF16。
小行数补齐到至少 32 行并对齐 8，最后裁去补齐行。零输入与零权重使用 scale=1。
没有 BF16 权重副本或浮点 GEMM 回退。转换前执行真实 INT8 GEMM 探测；后端不可用直接报错。
`torch._int_mm` 是私有接口，本轮已测 PyTorch 2.6.0+cu126；升级环境后需重新验证。

默认 `--quantization-scope blocks` 显式选择 28 个 block 中各 8 个线性层，共 224 层：
self-attention Q/K/V/output、cross-attention Q/output、FFN 两层。
输入/输出投影、时间条件、caption 投影、cross-attention K/V、归一化保持 BF16。
文本编码器与 VAE 不量化。不直接套用 LTX095 的固定层覆盖契约。
`--quantization-scope ffn` 可仅转换 56 个 FFN 线性层，属于单独待验收配置。

首版要求 fullgpu、BF16 加载、compile 和算子融合关闭，拒绝与单任务 CFG/SP/VAE 并行或缓存组合。
量化发生在组件加载后、阶段和内存管理器注册前。checkpoint 不修改，转换耗时单列；
模型运行报告包含实际执行层数、INT8 调用次数、权重字节数和 fallback_count。
缺少本机外部 ViDiT-Q 扩展，因此未启用该后端；本实现不依赖它。

## 命令

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 ./inference_cli.sh \
  --model-path results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/int8_output.mp4 --prompt "There is a bridge over the lake." \
  --attention-backend sage_attn --transformer-quantization int8_w8a8_native \
  --quantization-scope blocks --transformer-cache-mode off --no-cache-text-projections
```

`python scripts/quantization_benchmark.py` 接受同样输入参数，output-path 指向 JSON。
默认在常驻模型中依次跑 BF16、转换为 INT8、再跑 INT8；`--warmup --warmup-steps 1` 为每种模式独立预热。
`--quant-only` 仅执行量化项；性能必须与等设置基线比较。加载和转换不计入请求耗时，报告中单列。

## 初步验证

- 底层 INT8 与显式整数乘法参考比较通过；覆盖短行数、零输入、有无 bias。
- 小尺寸 192×320、25 帧、双窗口、40 有效去噪步：BF16 13.032 s，INT8 21.155 s。
  此轮未预热，包含 INT8 新 kernel 编译；小尺寸没有显示加速。
- 小尺寸 allocated 峰值 15.055 → 13.526 GiB；选中线性层的权重/bias/scale 存储 3,289,595,904 → 1,647,951,872 字节。
- 224 层全部执行，双窗口累计 35,840 次 INT8 调用，fallback=0。
- RGB SSIM 0.8628、MSE 396.16、MAE 10.23，仅作诊断。抽帧布局相近，生成区域存在局部差异。
  不能把小尺寸结果当作原尺寸质量结论，也不能把抽帧评估当作完整视频时序验收。

## 原尺寸单卡对照

素材 `10268234`：1080×1920、120 帧、SageAttention、50 采样步 / 40 有效步。
同一会话中 BF16 和 INT8 分别预热 1 步后各测一次；加载、转换、预热单列，不计入请求耗时。

| 指标 | BF16 | INT8 |
| --- | ---: | ---: |
| 端到端 | 208.046 s | 210.411 s |
| 去噪阶段 | 180.183 s | 182.236 s |
| 峰值 allocated | 47.115 GiB | 45.586 GiB |

耗时增加约 1.14%，没有显示加速；峰值显存减少 1.529 GiB（约 3.25%）。
转换耗时 0.080 s；224 层全部执行，含预热共 18,368 次调用，fallback=0。
Profiler 确认执行 `cutlass_80_tensorop_i16832gemm_s8` INT8 Tensor Core kernel，
以及 Triton `_quant_rows` / `_epilogue`，见 `kernel_profile.json`。

RGB SSIM=0.972664、MSE=25.1834、MAE=2.73260。量化不套用无损门槛。
首、中、末共 4 个缩小显示的帧对比中，湖面、栈桥、树木结构大致一致，局部水面/纹理有变化，
未见明显整体损坏；未逐帧目视检查全部原分辨率细节与连续运动，不能宣称已完成完整时序验收。
对比图 `full_comparison.png`，视频 `full.bf16_0.mp4` / `full.int8_0.mp4`。
数值脚本保留 `visual_review_required`，抽帧观察另记 `visual_review.json`，不混同两种判断。

最终回归 60 项：55 项通过、5 项双卡测试因本轮单卡而跳过。
包含原生整数参考对照、模型加载时转换、幂等性、层覆盖、真实小模型前向以及既有单卡回归。
结果索引：`validation_summary.json`。

当前保留为显式启用的实验选项，默认仍为 none，不推荐作为提速配置。
后续优先评估融合 INT8 GEMM 的反量化 epilogue，减少 INT32 中间张量读写及 kernel 调度，
再测试层选择/校准；此处是待验证方向，不是已证实的耗时归因。


产物目录：`results/quantization_20260921/`。

实现参考：[Triton 官方归约与融合教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。
