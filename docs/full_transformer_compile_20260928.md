# 完整 DiT torch.compile（2026-09-28）

新增 `--enable-torch-compile --torch-compile-scope transformer`。
`ffn` 仍为默认编译范围。完整模式编译模型 forward，包括输入、文本和时间投影、
全部 28 层的 self/cross attention、norm、调制、残差、FFN 和输出投影。
窗口内 RoPE 仍提前计算一次；VAE、scheduler、CFG 合成、视频 I/O 不在图内。

使用 `fullgraph=True, dynamic=False`，不允许静默断图或回退。
[PyTorch 官方说明](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)
定义了单张 FX 图或报错的契约。
Inductor 使用原生 Linear 图，不沿用 FFN 模式的 opaque native Linear 包装。
保留精度转换模拟，关闭 CUDA Graph。禁用手写融合时，编译使用 block 的原生等价表达式，
避开基于 ContextVar 的算子调用统计；attention 后端在图外预解析。

首版支持 SDPA、单卡常驻、未量化 DiT；T5/VAE CPU offload 可以保留。
首版拒绝 DiT 卸载、线程 CFG/SP 多卡、残差/文本投影缓存以及自定义算子融合。
后续已扩展到 CFG2/SP1，每卡持久化完整编译入口；SP>1 仍不支持。
详见 [双卡验证](cfg_compile_20260928.md)。
本页下方数据仍为首版单卡实测。
CLI、服务请求、运行时均验证边界。未引入 `sglang` 包。
具体用法见 [CLI](cli.md#完整-dit-编译实验性)。

## 真实权重 forward 基准

物理 GPU 6，A100 80GB，Torch 2.6.0+cu126，BF16、SDPA、确定性设置。
使用 `data/model/transformer` 的 28 层权重，batch 1、文本长度 128、固定随机 latent。
长窗口 latent `[1,128,16,34,60]`，短窗口 `[1,128,5,34,60]`。
两边都保持 DiT 常驻、关闭手写融合与缓存，RoPE 预计算。
模式为 `default`；每个形状先分别预热，再交替顺序各测五次。

| tokens | eager 中位数 ms | 完整 compile 中位数 ms | 耗时减少 |
|---|---:|---:|---:|
| 32640 | 2271.519 | 1880.247 | 17.2% |
| 10200 | 440.713 | 310.206 | 29.6% |

首次完整调用分别为 45.762 / 38.993 秒，包含编译与一次计算，不计入稳态表格。
长窗口峰值 allocated 约 5.787 / 5.791 GiB，短窗口约 4.298 / 4.295 GiB；
这是纯 DiT forward 的进程峰值，不能与完整请求 VAE 峰值混淆。
两种形状相对 eager 输出最大绝对差均为 0.0546875，L2 相对误差约 1.41%。
随机输入的误差不是视频 SSIM，也不是视频质量验收。

GPU 同机有其他作业；短窗口计时阶段与小模型 GPU 编译专项有时间重叠，
其 eager 样本范围为 439.703–560.438 ms，compile 为 309.198–314.197 ms。
长窗口无该专项重叠。这是单次会话五组 forward，不是多轮完整请求统计。
上述对照为未手写融合的 eager，不是此前融合基线；不能把 17.2% 全部视作相对最快既有配置的增益。

## 观测与验证

全量 CPU 回归 127 项：101 通过、26 条件跳过；新增编译专项在 CUDA 上 4 项全部通过。
Python 语法检查与 `git diff --check` 通过。

- CPU 完整图捕获：两个形状各生成一张 FX 图，改变 timestep 数值和返回旧形状不额外编译。
- GPU 小模型：Inductor 完整图实际运行，覆盖不同窗口形状及复用；数值容差检查通过。
- Stage 集成：选择完整编译入口，保留 eager 模型引用，报告成功 forward 数，支持显式退出编译模式。
- 配置测试：拒绝未支持的组合，保留 FFN 模式行为。

`torch_compile.first_call_history` 记录每种输入签名的首次调用时间（编译和计算合计），
`successful_forwards` 记录完成次数。`applied` 是入口注册状态；首次调用失败直接抛错。
首次成本保留在请求耗时中，不通过扣减伪造另一条稳态结果。

本地实验目录 `results/full_compile_20260928/` 保存 benchmark 脚本、日志、JSON 及回归日志。

## 原片复现命令

```bash
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
TORCHINDUCTOR_COMPILE_THREADS=4 MGERASE_TORCH_COMPILE_MODE=default \
python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/full-compile.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 \
  --infer-len 121 --overlap 9 --compact-tail-padding \
  --no-dit-layerwise-offload --no-dit-cpu-offload \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --attention-backend sdpa --operator-fusion-backend disabled \
  --transformer-cache-mode off --no-cache-text-projections \
  --enable-torch-compile --torch-compile-scope transformer
```


## 原片完整请求

相同原片 1920×1080、145 帧、24000/1001 FPS、seed 42、50 配置步、strength 0.8，
窗口 121/重叠 9，尾窗减填充开启，每窗口实际去噪 40 步。
物理 GPU 6，模式 `default`，没有显式请求预热。

- 模型加载 **41.664 秒**，单列于请求之外。
- 完整请求 **310.927 秒**，去噪 **253.962 秒**。
- 两种形状的首次完整 forward **43.723 / 38.518 秒**，含编译与一次计算，已计入上述时间。
- 实际成功执行 **160 次**完整图 forward，无 attention 回退，两个窗口均完成。
- 峰值 allocated **34.045 GiB**、reserved **52.074 GiB**。

首次请求没有胜过此前 240.915 秒的单卡 SDPA 减填充结果，编译成本是重要因素；
两次也不是同期配对测试，非去噪时间存在差异。不能从 310.927 秒中直接扣掉首次
forward 耗时后宣称得到一次新的稳态请求实测。常驻服务的完整请求收益尚需复跑。


相对未减填充 SDPA 融合参考 `results/performance_20260927/fused50.mp4`：

- 整段 RGB SSIM **0.982325**，达到此前整段 0.98 门槛。
- MSE **2.648438**，MAE **1.087710**。
- 最低帧 SSIM **0.961303**，相邻帧误差变化 MAE **1.085832**。
- 两侧均完整解码 **145 帧**，分辨率与帧率一致。

这是完整编译与尾窗减填充的组合结果，平均达标不表示每帧达标。
没有进行逐帧人工播放或多素材泛化验收。质量脚本沿用此前 RGB Gaussian SSIM 实现，
本次明确设定阈值为 0.98，原始逐帧数据保存在 `full50_quality.json`。
