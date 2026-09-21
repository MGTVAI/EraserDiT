# 完整视频缓存正确性复测

参考视频固定为 `results/out_10268234.mp4` 和 `results/out_113000356.mp4`，
哈希与媒体元数据保存于 `results/cache_correctness/references.json`。复测使用 GPU 2、3，
原始输入为 `data/` 中同名视频与掩码；不会覆盖参考视频。

## 修复内容

- TeaCache 原先只比较 `temb`，同一时间步下无法感知视频特征变化。现在通过第一层
  forward 获取 `norm1(hidden_states) * (1 + scale_msa) + shift_msa`，再计算相对 L1。
  探针也经过动态卸载包装器，不直接读取尚在 CPU 的层权重。系数策略名称同步变更，
  不沿用其他模型的多项式；仍是未拟合的恒等映射。报告新增逐分支 `probe_trace`。
- CacheDiT 的前段残差减法也使用 EraserDiT 的 FP32 策略，与已有中段残差一致。
  例如 bf16 输入 1024、输出 1.25，先用 bf16 相减会得到 -1024，FP32 正确值是 -1022.75；
  先舍入再转 FP32 不能恢复这个判据精度。通用 LTX095 控制器保留原有数值策略。
- `cache_benchmark.py` 必须显式传入提示词，默认窗口改为 121，新增 seed、attention backend
  和外部 reference 参数。此前短视频、17 帧窗口、未显式传提示词的报告不能直接作为完整素材结论。
- `cache_compare.py --reference` 可直接比较指定视频；掩码采用与推理一致的 RGB 第一通道和
  `mask_threshold=0.039`，保留逐帧指标以检查窗口衔接。历史报告使用 Y>127，区域统计口径不同。

## 复测设置

| 项目 | 设置 |
|---|---|
| 模型 | `results/cache_prediction_model`，官方已校验权重 |
| 精度 / 注意力 | bf16 / SageAttention，自注意力外的 cross-attention 使用 SDPA |
| 采样 | seed=42，50 调度步，strength=0.8，每窗口实际 40 步，CFG=3 |
| 窗口 | infer_len=121，overlap=9，windowed_preload |
| 10268234 | 120 帧，1080×1920，提示词 `There is a bridge over the lake.` |
| 113000356 | 145 帧，1920×1080，提示词 `There is a rooftop terrace overlooking the city at sunset.` |
| 近似缓存 | predictor=none，warmup=4，end_guard=1，最多连续复用一次 |

指定第一组参考视频的原日志 `results/logs/retry_10268234.log` 包含 torch.compile；缓存复测不启用
whole-transformer compile。因此同时报告相对指定参考与相对本次 off 的误差，不能把全部参考差异
归因于缓存。耗时为单次完整请求，不包含加载；本次重点是正确性，不作为多次重复的性能统计。

## 复现

当前完整验收任务、日志及视频保存在 `results/cache_correctness/high_threshold/`，两种缓存阈值均为 0.2，
以画面基本相似为准。此前低阈值的 before/after 批次已中止，已完成的视频仅作为辅助对照；
TeaCache 0.02 队列及动态卸载恢复推理在启动前取消，不计作完成的验证。

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 ./inference_cli.sh \
  --model-path results/cache_prediction_model \
  --task-file results/cache_correctness/high_threshold/10268234_tasks.json \
  --attention-backend sage_attn

/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python scripts/cache_compare.py \
  --directory results/cache_correctness/high_threshold --mask data/10268234_mask.mp4 \
  --reference results/out_10268234.mp4 --report-name example_reference_quality.json \
  --names 10268234_teacache_02 10268234_cache_dit_02
```

强制全计算验证用于检查接入是否改变模型结果；有缓存命中时仍属近似推理，不能宣称逐像素等价。

## 阈值 0.2 完整结果

两种模式在这两段完整素材上均保持基本相似。抽查包含第一组误差较大的第 104 帧、第二组第 102 帧，
以及第二组窗口边界附近的第 112、121 帧和末帧 144，未见明显新增人物残留或画面结构损坏。
仍有纹理与像素差异；这些观察不等同于逐像素一致或对其他素材的质量保证。

下表 SSIM/PSNR 全部相对用户指定参考视频；加速比相对本次同参数 off（211.0 / 401.9 秒）。

| 素材 | 模式（阈值 0.2） | 秒 | 加速比 | 命中 / CFG 前向 | 整帧 SSIM-Y | 擦除区 SSIM-Y | 擦除区 PSNR dB |
|---|---|---:|---:|---:|---:|---:|---:|
| 10268234 | [TeaCache](../results/cache_correctness/high_threshold/10268234_teacache_02.mp4) | 129.6 | 1.628× | 36/80 | 0.987175 | 0.980612 | 43.34 |
| 10268234 | [CacheDiT](../results/cache_correctness/high_threshold/10268234_cache_dit_02.mp4) | 129.9 | 1.624× | 36/80 | 0.986218 | 0.979314 | 43.19 |
| 113000356 | [TeaCache](../results/cache_correctness/high_threshold/113000356_teacache_02.mp4) | 244.6 | 1.643× | 72/160 | 0.990819 | 0.993056 | 46.35 |
| 113000356 | [CacheDiT](../results/cache_correctness/high_threshold/113000356_cache_dit_02.mp4) | 247.6 | 1.623× | 72/160 | 0.990772 | 0.992915 | 46.40 |

两者命中率均为 45%，达到本次 warmup/末步/最多连续复用一步保护下的上限。
这两组上 TeaCache 0.2 的相似度略高、速度相近，可优先采用。保持默认阈值不变，实际使用时显式传入：

```bash
--transformer-cache-mode teacache --teacache-threshold 0.2
# 或
--transformer-cache-mode cache_dit --cache-dit-residual-diff-threshold 0.2
```

40 项测试通过（18 项缓存测试，22 项 GPU/组件卸载测试）。第一组强制完整计算输出与 off 的 SHA256 一致。
四次高阈值请求均正常关闭缓存；第二组的两个窗口各有独立缓存状态。指定参考视频哈希保持不变。

原始统计与逐帧质量：[summary.json](../results/cache_correctness/high_threshold/summary.json)，
`*_runs.json`、`*_reference_quality.json`、`*_off_quality.json`。
图片：[10268234](../results/cache_correctness/high_threshold/10268234_comparison.jpg)、
[113000356](../results/cache_correctness/high_threshold/113000356_comparison.jpg)。
