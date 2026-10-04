# Attention 性能优化（2026-10-04）

后续完整素材验收：快速 QK/AdaLN 的 SDPA 与 Sage 组合均未通过质量门槛，不作为通用推荐。
下文保留 121 帧切片实验的数据与实现，见 [最新验收记录](l40s_validation_20261004.md)。

目标是降低完整请求耗时，允许数值近似。本轮所有性能组均使用静态 FP8 FFN，分别对比 SDPA、
Sage FP8 和 Sage FP8 + 完整 Q/K 融合。采样步数、视频尺寸、CFG 和缓存设置相同。

## 实现

新增显式算子 `qk_rmsnorm_rope_fast`：一个 Triton kernel 完成 Q/K 全宽 RMSNorm、
权重乘法和相邻特征对 RoPE，避免 PyTorch 归一化、旋转所生成的大量中间张量。
归约使用 FP32，保留归一化后、权重乘法后的 BF16 边界；归约顺序与 RoPE FMA 允许变化。
原 `qk_rmsnorm_rope` 精确路径保留，两者不能同时选择，默认融合集合不变。

支持 BF16、连续 `[B,S,2048]`、Diffusers RMSNorm、eps=1e-5、完整 FP32 cos/sin。
仅验证单卡推理；自动模式在不支持的契约或启用梯度时回退，强制模式报错。
Sage FP8 的量化粒度仍为 `per_thread`，累加仍为 `fp32+fp32`，没有改变注意力窗口或减少步数。

## 算子与完整 DiT

L40S、PyTorch 2.6.0+cu126，BF16，32640 tokens，32 heads × 64。Q/K 算子测量在 GPU3；
完整 DiT 与视频在 GPU2。完整 DiT 使用真实模型权重和随机输入，交替顺序测量三次：

| 测量 | 对照 | 优化 | 结果 |
| --- | ---: | ---: | --- |
| Q/K 归一化 + RoPE | PyTorch 20.348 ms | 完整融合 1.469 ms | 单算子约 13.9× |
| Q/K 归一化 + RoPE | 仅 RoPE 融合 8.404 ms | 完整融合 1.469 ms | 单算子约 5.7× |
| 完整 DiT 前向 | SDPA 2.364 s | Sage FP8 1.971 s | 三次中位数 |
| 完整 DiT 前向 | Sage + 仅 RoPE 融合 1.649 s | Sage + 完整融合 1.473 s | 三次中位数 |

随机输入的 Q/K 相对 RMSE 分别为 1.32e-5、1.21e-5；完整 DiT 相对 SDPA 的 RMSE：
Sage 0.014398，Sage + 完整融合 0.014380。这些值不代替完整视频质量检查。

## 完整视频

121 帧、1920×1080、seed42、CFG3、50 steps × strength0.8 = 40 个实际去噪步。
DiT 常驻，T5/VAE CPU offload，VAE low-memory；关闭残差缓存、文本投影缓存及编译。
同一 GPU2 顺序运行 SDPA 一次、Sage 一次、Sage + 完整融合两次，每组模型加载后预热两步。
请求耗时排除模型初始化和显式预热，包含视频读写；不能直接当成冷启动时间。

| 配置（均为静态 FP8 FFN） | 去噪 | 纯推理 | 完整请求 |
| --- | ---: | ---: | ---: |
| SDPA | 208.535 s | 222.984 s | 244.518 s |
| Sage FP8 | 177.619 s | 192.063 s | 212.705 s |
| Sage FP8 + 完整 Q/K 融合，第 1 次 | 131.165 s | 146.747 s | 167.031 s |
| Sage FP8 + 完整 Q/K 融合，第 2 次 | 130.982 s | 145.353 s | 166.493 s |
| 完整融合中位数 | **131.074 s** | **146.050 s** | **166.762 s** |

相对本轮 SDPA，完整请求减少 **31.80%（77.76 s）**，去噪减少 **37.15%**。
相对仅启用 Sage，新增完整 Q/K 融合进一步减少请求耗时 **21.60%（45.94 s）**。
对照组各一次、最终候选两次，未计算统计置信区间。

三组峰值 allocated 均为 23.569 GiB、reserved 均为 43.234 GiB。
每次正式请求完整 Q/K 融合调用 2240 次，无 Q/K 回退；未选择的 AdaLN 保持参考实现。
Sage 后端无失败或回退。两次候选输出 SHA256 相同：
`a56d13535851b491329ede932d7ba2297d65cfd426752ecbd785e58d0b73f9c7`。

日志记录的模型加载时间分别为 29.98 / 32.78 / 43.08 s；首次请求预热分别为
45.05 / 44.50 / 48.88 s。上表均不包含这些时间，也不包含 Python 进程启动成本。
含显式预热的首次任务时间分别为 289.57 / 257.21 / 215.92 s。


相对历史 BF16 + SDPA 视频的质量（121 帧，其中 103 帧存在掩码）：

| 配置 | 平均 SSIM | 最低整帧 SSIM | 掩码平均 SSIM | 最低掩码帧 SSIM | 时间差分 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sage + 静态 FP8 FFN | 0.987584 | 0.984624 | 0.986697 | 0.982611 | 0.994424 |
| 上述配置 + 完整 Q/K 融合 | 0.987581 | 0.984620 | 0.986707 | 0.981074 | 0.993779 |

时间差分 MAE 比较候选与 BF16 的帧间变化。人工检查首/中/末帧及最差掩码帧（0-based 101）裁剪，
未见明显新增结构伪影；不是全素材质量保证。本轮 SDPA 输出 SHA256 与上一轮静态 FP8 输出一致，
基线未漂移。平均 SSIM 接近不代表逐元素一致。


## 复现

本地隔离扩展为 SageAttention v2.2.0，路径沿用前轮构建，未修改默认 SageAttention 1.0.6 环境。
其他机器需要先安装兼容的 SM89 扩展，见 [Attention 量化环境](attention_quantization_20261003.md)。

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
PYTHONPATH="$PWD:$PWD/results/quant_opt_20261003/SageAttention" \
uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input results/quantization_20261003/full121/video.mp4 \
  --mask-input results/quantization_20261003/full121/mask.mp4 \
  --output-path outputs/attention-fast.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." --seed 42 \
  --transformer-quantization fp8_w8a8_static --quantization-scope ffn \
  --attention-backend sage_fp8 \
  --operator-fusion-backend triton --operator-fusion-ops qk_rmsnorm_rope_fast \
  --transformer-cache-mode off --no-cache-text-projections \
  --no-dit-layerwise-offload --no-dit-cpu-offload \
  --text-encoder-cpu-offload --vae-cpu-offload --vae-low-memory \
  --infer-len 121 --overlap 9 --num-inference-steps 50 --strength 0.8 --guidance-scale 3 \
  --warmup --warmup-steps 2
```

验证：新融合及精确路径专项 9 项通过；全套 225 项测试完成，178 项通过、47 项按环境条件跳过。
全套启用了 Sage FP8 GPU 测试。没有要求优化输出逐元素一致。

原始材料在 `results/attention_opt_20261004/`：`screen.json`、`forward.json`、`manifest.json`、
`tests.log`、`suite.log`，以及 `full121/` 中的命令、任务、日志、视频、汇总和质量报告。
结果只覆盖该 L40S 长序列素材；短序列、多卡、其他 GPU 和其他素材需独立验证。
