# 完整 DiT + VAE decoder 编译：原片验证（2026-09-28）

承接 [辅助组件编译](component_compile_20260928.md)，将此前短片通过的
`--compile-components vae_decoder` 推进到原片完整验证。
没有开启 T5 或 VAE encoder 编译，没有导入 `sglang` 包。

## 条件与对照

物理 GPU 6，A100 80GB，BF16、SDPA、完整 DiT 编译、模式 `default`。
DiT 常驻，T5 FSDP CPU offload、VAE CPU offload。手写融合、残差/文本投影缓存关闭。
原片及 mask 为 `data/113000356.mp4` 和 `data/113000356_mask.mp4`；
1920×1080、145 帧、24000/1001 FPS，seed 42、50 配置步、strength 0.8，
窗口 121/重叠 9、尾窗口减填充，每窗口实际去噪 40 步。

同一会话依次执行：组合首次请求、组合重复请求、仅 DiT 编译的重复请求。
最后一个请求仅临时恢复 decoder 的 eager forward，其他模型/设置不变。
首次请求先做质量检查，达到 0.98 后才继续稳态对照。
模型加载单列；首次调用的编译和计算全部保留在请求耗时内。
测试主机其他 GPU 有作业；顺序配对只用于筛选，不是多轮交错稳态统计。

保存了长/短窗口的真实 decoder 输入（CPU），供后续单组件诊断；首次请求计时包含
这两次诊断拷贝与存盘。重复请求不会再次保存。

## 完整视频质量

| 参考 | RGB SSIM | MSE | MAE |
|---|---:|---:|---:|
| 未减填充 SDPA 融合参考 `performance_20260927/fused50.mp4` | **0.982273** | 2.669022 | 1.095073 |
| 仅完整 DiT 编译 `full_compile_20260928/full50.mp4` | **0.983202** | 2.252962 | 0.983603 |

组合结果相对原始参考达到此前整段 SSIM ≥0.98 门槛。
最低帧 SSIM 为 **0.961045**，相邻帧误差变化 MAE 为 **1.089255**；
整段均值通过不代表每帧通过，也不能将两个单项 SSIM 相加预测组合质量。
两份对照均完整解码 145 帧，尺寸和帧率一致。

抽查 0、24、48、72、96、132 帧的三列对照，未见明显可辨识人物残留或黑帧，
背景/水花纹理有变化。这是缩略图抽查，不是逐帧全分辨率或动态播放验收。
没有多素材泛化验证。

产物在 `results/decoder_full_20260928/`：
`validate.py/.log`、`summary.json`、各请求 MP4、`quality_original.json`、
`quality_dit_only.json`、`review.jpg`、`visual_review.json` 及两个真实 decoder 输入。

## 复现单次请求

```bash
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
TORCHINDUCTOR_COMPILE_THREADS=4 MGERASE_TORCH_COMPILE_MODE=default \
python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/compiled-dit-decoder.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 \
  --infer-len 121 --overlap 9 --compact-tail-padding \
  --no-dit-layerwise-offload --no-dit-cpu-offload \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --attention-backend sdpa --operator-fusion-backend disabled \
  --transformer-cache-mode off --no-cache-text-projections \
  --enable-torch-compile --torch-compile-scope transformer \
  --compile-components vae_decoder
```

同一会话重复执行可通过 CLI `--task-file` 提供多项任务，或使用常驻服务。
每次重新启动 CLI 都会重新建立编译入口；磁盘缓存不等于已预热的驻留进程。
`validate.py` 使用相同参数自动执行本报告的对照顺序，并在首轮质量失败时停止后续性能比较。


## 请求与阶段耗时

模型加载 **54.196 秒**，不计入下面请求时间。计时包括请求的完整视频输出。

| 配置 | 请求 s | 去噪 s | 解码阶段 s | allocated GiB | reserved GiB |
|---|---:|---:|---:|---:|---:|
| 完整 DiT + decoder，首次 | 301.703 | 217.318 | 41.890 | 34.045 | 52.074 |
| 完整 DiT + decoder，重复 | 224.084 | 175.062 | 6.330 | 34.046 | 62.381 |
| 仅完整 DiT，重复 | 224.128 | 175.116 | 7.368 | 34.046 | 62.381 |

VAE decoder 两种形状的首次调用分别为 **20.729 / 17.465 秒**，包含编译与一次计算，
已计入首行请求和解码阶段。首行也包含 DiT 的首次编译。

重复请求中，decoder 编译使解码阶段从 **7.368 降到 6.330 秒**，单次观察约减少 **14.1%**。
但完整请求仅从 **224.128 降到 224.084 秒**，差 **0.044 秒（约 0.02%）**，
其他阶段的差异抵消了解码阶段节省的时间。本轮没有证明有效的端到端增益，不能称作稳定提速。
首轮与重复请求的巨大差距主要是首次编译成本，不是新增 decoder 编译的净收益。

allocated 峰值基本维持 **34.05 GiB**；reserved 从首次 **52.07** 上升到重复请求 **62.38 GiB**，
且最后仅 DiT 编译的对照仍保留相同 reserved 峰值。这是同一进程的缓存分配状态，
不能将该增量未经消融直接归因于 decoder 编译，也不能声称完整请求显存降低。

结论：该组合在本片的整段质量检查通过，但暂无足够端到端收益把额外 decoder 编译加入
默认速度配置。可保留显式选项供常驻服务进一步测试；当前更应优先优化未被编译覆盖的
前后处理和帧搬运。T5/VAE encoder 的此前短片质量失败结论保持不变。

## 输出复现检查

首次与重复组合请求的完整解码 RGB SHA256 相同：
`8d45ddbe3f7f496af1194c166e0f00e1da36cd308f7fdb06e1c6a2a901741096`。
最后一个仅 DiT 编译的请求与此前对应参考也逐像素一致：
`5e7aad5092d4db5a38e6cecf10e02eb2e7e21b5b44fedb527b24093657fd4096`。
因此上述质量报告同样适用于本轮重复组合输出，且本次对照基线复现了此前结果。
四份视频均完整解码 145 帧。详情见 `reproducibility.json` 和 `comparison.log`。

本轮只新增验证脚本、产物与文档，没有修改运行时实现；文档 `git diff --check` 通过。
