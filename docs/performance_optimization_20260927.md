# DiT 融合性能优化（2026-09-27）

在上一轮显存生命周期优化和静态条件复用的基础上，修正 QK/RoPE 融合的数值边界，
与已有 AdaLN 调制融合一起验证。保留 SDPA、逐层卸载、T5 FSDP、VAE CPU offload，
没有引入 `sglang` 包、量化、残差缓存或 VAE tiling。

## 实现与启用

原来的 `qk_rmsnorm_rope` 用 Triton 同时实现 RMSNorm 归约、权重乘法和 RoPE，
改变原生归约及 BF16 中间舍入，历史完整视频质量未达标。
现在先调用模型自己的 Q/K RMSNorm，再用一个 Triton kernel 完成两路 RoPE。
FP32 两次乘法与加法的舍入独立保留，最后写入 BF16，不进行 FMA 收缩。
kernel 直接读取相邻特征，不再构造旋转后的完整张量及多个 FP32 临时张量。

配置名称 `qk_rmsnorm_rope` 继续表示整个优化位置；它不再表示 RMSNorm 归约被融合。
`rmsnorm_adaln` 同样保留原生 RMSNorm，只融合后续调制。这一边界与模型原始数值语义一致。

在原有命令上增加：

```bash
--operator-fusion-backend auto
```

默认选择 `qk_rmsnorm_rope,rmsnorm_adaln`；也可使用 `--operator-fusion-ops` 指定单项。
`auto` 对不满足 dtype、设备、宽度、形状或布局契约的输入回退到原实现。
`triton` 要求契约满足，失败会报错。默认值仍为 `disabled`，便于显式保留参考路径；
本轮没有放宽 compile、INT8、多卡卸载等既有组合限制。

## 真实形状 forward 基准

A100 80GB，物理 GPU 3；Torch 2.6.0+cu126，BF16、确定性配置，真实 `data/model/transformer`
权重。使用原片首窗口 latent 形状 `[1,128,16,34,60]`，即 32,640 个 token，文本长度 128。
输入为固定随机张量，RoPE 已在窗口外预计算。

每种驻留策略分别对两条路径预热一次，然后按 A/B、B/A、A/B、B/A、A/B 交替测量。
每条路径五次，计时前后 CUDA 同步；不含加载、预热和首次 Triton 编译。

| 驻留方式 | 关闭融合中位数 ms | 两项融合中位数 ms | 耗时减少 |
|---|---:|---:|---:|
| DiT 全驻留 | 2273.23 | 2099.67 | 7.63% |
| SGLang 来源逐层卸载，预取一层 | 2282.57 | 2106.91 | 7.70% |

两组各十次 forward 输出均与对应参考逐元素一致。逐层路径仍通过 block 的 Module hooks
执行预取和释放，未将搬运、event 或 hook 捕获进计算 kernel。

这是 DiT 单 forward 的五次测量，不是完整视频请求的五次稳态统计。全驻留与逐层两组
顺序执行，不能将其之间的小幅时差解释为卸载成本的精确归因。

## 原片 50 步验收

物理 GPU 2，1920×1080、145 帧、24000/1001 FPS；seed=42，50 个配置步，strength=0.8，
窗口 121、重叠 9，每窗口实际去噪 40 步。默认单卡逐层卸载，预取参数 0（一层），
文本投影缓存关闭、残差缓存关闭。候选只增加 `--operator-fusion-backend auto`。

基线为上一轮已经完成显存优化和静态条件复用的
`results/memory_optimization_20260927/final50.mp4`，不是迁移前实现。

| 配置 | 请求 s | 去噪阶段 s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|---:|
| 本轮改动前，关闭融合 | 415.33 | 366.80 | 30.653 | 48.387 |
| 修正后的两项融合 | 391.85 | 340.45 | 30.653 | 48.387 |

本次单次观测请求耗时减少约 5.65%，去噪阶段减少约 7.19%；不是完整请求的五次稳态统计。
不包含模型加载，没有显式请求预热；候选可使用前面微基准生成的 Triton 磁盘编译缓存。
两次请求不是同期交替运行，主机还有其他作业，不能将所有时差归因于单一改动。

两段视频均完整解码 145 帧、尺寸和帧率一致；RGB SHA256 完全相同：
`62e8723b73fd90ae6aa60b74d426d056a232c9ba538c06152d7e83e1bfe70cce`。
即相对本轮基线逐像素一致，SSIM=1、MSE=0、MAE=0。
上一轮显存优化相对更早原始基线的像素差异仍存在，本轮没有增加差异。

实际执行 `qk_rmsnorm_rope` 4480 次、`rmsnorm_adaln` 8960 次，运行时没有回退。
请求结束 DiT 受管权重驻留为零；VAE 编码仍决定整请求显存峰值。

复现命令：

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
  /mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model --video-input data/113000356.mp4 \
  --mask-input data/113000356_mask.mp4 --output-path /tmp/eraserdit-fused.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 \
  --dit-layerwise-offload --text-encoder-cpu-offload --vae-cpu-offload \
  --dit-offload-prefetch-size 0 --attention-backend sdpa \
  --transformer-cache-mode off --no-cache-text-projections \
  --operator-fusion-backend auto
```

## 注意力后端筛选

物理 GPU 2，BF16 Q/K/V `[1,32640,32,64]`、无 mask 的随机输入，两次预热后各五次：

| 后端 | 中位数 ms | 相对 SDPA 最大绝对差 |
|---|---:|---:|
| SDPA | 46.830 | 0 |
| FlashAttention | 46.340 | 0.000244 |
| SageAttention | 45.237 | 0.002441 |

各后端按组顺序测量，结果仅用于筛选；当前 SDPA 已使用 PyTorch FlashAttention kernel。
另外两个后端在该输入上的差距较小，并且输出有数值差异，本轮没有将它们改成默认或
声称通过本轮完整视频质量验证。

## 产物

CPU 全量回归 110 项：91 通过、19 按设备或显式 opt-in 条件跳过。
GPU 融合、逐层卸载、组件卸载与缓存专项 37 项全部通过。
新增用例覆盖两种原生 RMSNorm、不同幅值和 batch/token 形状、自动回退/强制失败、
完整 block 数值一致性，以及融合下重复逐层卸载 forward 与释放。
`git diff --check` 通过。

`results/performance_20260927/` 为本地忽略目录：

- `benchmark_dit.py/.json/.log`：真实权重、两种驻留方式的交替 forward 基准。
- `attention_probe.json`：注意力后端筛选。
- `previous_qk_kernel.py`、`previous_qk_dispatch.py`：改动前融合实现快照。
- `fused50.json/.log/.mp4`：原片 50 步最终验证。
- `cpu_tests.log`、`gpu_tests.log`：回归记录。

性能数据基于此机器和模型形状，其他架构、素材、分辨率和多卡拓扑需要独立验证。
