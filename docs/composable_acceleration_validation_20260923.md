# 可组合加速验收（2026-09-23）

仅使用物理 GPU 6、7（A100 80GB），`CUDA_VISIBLE_DEVICES=6,7`；未启动四卡任务。
GPU 7 原有约 12 GiB 占用未触碰。该占用不计入本进程 PyTorch allocator 指标。

质量门槛为 RGB SSIM ≥ 0.985、MSE ≤ 36、MAE ≤ 6。无需逐像素一致；在达标候选中选速度更快的配置。
缓存和量化记录指标并检查视觉差异，不用上述硬门槛淘汰。指标使用 0..255 RGB 通道、
11×11 Gaussian SSIM（sigma=1.5、reflect 边界），与前次结果口径相同。

## 实现

- DiT 权重管理器在 block 外，编译边界为 FFN；缓存判断、Python 通信、stream/event 留在图外。
  CUDA graph 关闭；默认保留原生 GEMM 的 BF16 bias 舍入，激活交给 Inductor。
- 每 rank 的模型从 CPU 创建并跨窗口复用。动态卸载使用独立的 pinned CPU 权重、copy stream、
  预算和生命周期；副本不通过主卡完整复制。去噪阶段结束即卸载，各窗口缓存独立清理。
- CFG 的正/负分支拥有独立缓存。SP 对误差分子/分母做全局归约，并核对跳层决定，不能各自跳层。
- `sharded` 真正分片 FFN/投影；`reference` 保留全长 GEMM 作为质量回退。
- Ulysses 使用线程 peer-copy/barrier，**不是 NCCL 多进程实现**。新增 Ring 为 SP2 + SDPA
  实验路径，采用 native flash 的 LSE 和 FP32 online softmax 合并；不宣称完成四卡混合 USP。
- INT8 支持 CPU 逐层量化，避免先把整个 BF16 DiT 放入 GPU。量化缓存与普通缓存生命周期一致。
  编译区内部不累加 Python 调用次数；量化 runtime 计数明确不包含编译执行。
- 单卡 VAE tiling 可叠加卸载；多卡 VAE 暂仍要求 fullgpu。FSDP 与此卸载/peer mesh 互斥。
  旧 `--cfg-parallel-device` 保留旧限制，组合使用 `--cfg-degree 2`。

## 测量范围

完整视频为 `data/113000356.mp4` / 对应 mask：1920×1080、145 帧、24000/1001 FPS。
固定 seed 42、50 个配置步、strength=0.8，两个窗口，各实际执行 40 步。
参考为 `results/layerwise_offload_20260923/trim_output.mp4`，已核对其解码 RGB 与旧基线一致。
历史单卡动态卸载耗时 420.29 s，主卡 allocated 峰值 33.64 GiB。

正式结果、逐帧质量、命令和日志保存在 `results/composable_acceleration_20260923/`。
任务时间不含模型加载；无显式预热的首轮任务包含惰性编译。统计为单次观测，不能当作多次稳态统计。
CFG native 验证会话先跑短片再跑原片，其 CUDA 库已暖；其余表中配置为新进程的首个原片任务。
小幅时差不能据此归因于某个单独开关。

## 已发现的问题

初始全 Inductor FFN + dynamic_offload + CFG2：277.62 s，主卡峰值 33.64 GiB，
SSIM 0.983124、MSE 2.282402、MAE 0.991076，**未通过 SSIM 门槛**。
生成代码把首个 Linear 的 bias 移入 GELU kernel，改变了 BF16 中间舍入。
该候选保留在结果目录供复核，不作为达标推荐。


## 编译边界定位

32,640 tokens × 2,048 hidden 的 BF16 FFN 探针：普通 Inductor、ATEN-only 和关闭
GEMM epilogue fusion 三者均出现 MAE 0.000598；保留 native Linear 边界后探针误差为零。
探针单次 FFN 均值从约 8.70 ms 变为 8.86 ms，差约 1.8%，不代表整段视频差值。
512×288 / 33 帧短片的 dynamic_offload + CFG2 + native-boundary compile 已通过三项指标。
完整视频结果另列。详细探针数据见结果目录 `compile_probe.json`。

Torch 2.6 的 FX tracing 会临时修改进程级 Module 调用路径，首次并行编译 INT8 时触发过
线程间干扰。现在在启动 rank forward 前串行准备各 rank、各 FFN 的形状，准备过程同样遵守
逐层权重预算且不消耗 RNG；实际去噪仍双卡并发。相同形状跨窗口不重复准备。


## 已完成的完整视频结果

| 配置 | 任务秒 | 主卡 allocated / reserved GiB | SSIM / MSE / MAE | 验收 |
|---|---:|---:|---|---|
| 动态卸载 + CFG2 + native-boundary compile + SDPA | 241.03 | 33.71 / 51.25 | 1.000000 / 0 / 0 | 三指标通过，同会话先跑短片 |
| 动态卸载 + SP2 sharded + compile + SDPA | 263.25 | 33.64 / 50.85 | 1.000000 / 0 / 0 | 三指标通过 |
| 动态卸载 + CFG2 + compile + Sage + TeaCache | 164.62 | 33.64 / 51.22 | 0.982008 / 3.133369 / 1.263465 | 允许缓存近似，抽查相近 |
| 动态卸载 + SP2 sharded + compile + FA + cache_dit | 185.69 | 33.64 / 50.85 | 0.981938 / 3.139208 / 1.257284 | 允许缓存近似，抽查相近 |
| 组件阶段卸载 + CFG2 + compile + SDPA | 256.23 | 34.67 / 63.13 | 1.000000 / 0 / 0 | 三指标通过 |
| 组件阶段卸载 + CFG2 + compile + Sage + TeaCache | 170.37 | 34.67 / 63.12 | 0.982008 / 3.133369 / 1.263465 | 允许缓存近似，抽查相近 |
| 动态卸载 + CFG2 + compile + INT8 + SDPA + cache_dit | 180.83 | 33.64 / 50.85 | 0.980880 / 3.758604 / 1.385532 | 允许缓存/量化近似，抽查相近 |

上述有效编译配置均采用默认 native Linear 边界，FFN 局部编译。
无缓存 CFG2 的第二卡 allocated 峰值约 3.70 GiB；不包含该卡原有的其他进程占用。
其两次窗口编译准备分别为 4.12 s、0.0004 s，验证相同形状跨窗口复用；同一会话先执行了短片，
也验证了短片到原片的形状切换。初始化含副本创建期间，两卡的 PyTorch weight allocation 均为 0，
CPU 峰值 RSS 约 27.36 GiB。CUDA context/驱动占用不在 allocator 指标中。

上述 TeaCache 与 cache_dit 组合实际复用 72 / 160 个逻辑 CFG 分支步（45%）；cache_dit 命中时仍执行 front block。
缓存统计按 SP 组 leader 聚合逻辑步，各 rank 独立计数保留在 `ranks`；缓存显存汇总是各 rank
峰值之和的上界，不冒充同一时刻采样到的多卡峰值。
已检查完整解码和逐帧指标，并抽查首、中、末、最低 SSIM 帧；未做全片人工播放审查。
对比图和审查范围保存在每组目录的 `comparison.png`、`visual_review.json`。


## 推荐组合与边界

本素材优先推荐动态卸载 + CFG2 + FFN compile + Sage + TeaCache，单次观测 164.62 s。
相比历史单卡动态卸载 420.29 s 约为 2.55×，这是整套双卡组合的观测收益，不是 compile 的单项收益。
不使用近似缓存时选 CFG2 + SDPA + compile；SP2 sharded 已通过本原片三指标，但耗时更长，
第二卡峰值约 2.89 GiB，适用于更在意去噪激活占用的场景。其他素材/分辨率仍需按指标选型。

组件阶段卸载没有在本次样本中胜出，且 reserved 峰值更高；INT8 组合也未超过 Sage + TeaCache，
所以两者都作为可选策略，不叠加为默认。量化不会解决本片约 33.64 GiB 的 VAE 激活峰值。
本轮没有声称完成 VAE tiling 的完整视频质量验收，也没有做五次原片稳态重复统计。

复跑最快已测组合（在 EraserDiT Python 环境、仓库根目录执行）：

```bash
CUDA_VISIBLE_DEVICES=6,7 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
MGERASE_COMPILE_LINEAR_BACKEND=native \
python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path results/composable_repro.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 \
  --resource-policy dynamic_offload --max-weight-usage 2147483648 \
  --dit-offload-prefetch-size 1 --cfg-degree 2 --enable-torch-compile \
  --attention-backend sage_attn --transformer-cache-mode teacache \
  --teacache-threshold 0.3 --cache-text-projections
```

无近似缓存时替换为 `--attention-backend sdpa --transformer-cache-mode off --no-cache-text-projections`。
`--cfg-degree 2` 与 `--sp-degree 2` 本轮二选一；两者同时开启需要四卡，不在本轮任务范围。
Ring 只完成双卡小模型组合/故障验证；完整 Ring 视频、混合 USP、NCCL 多进程路径留待后续。
强制手工 Triton 算子融合仍不与 compile 叠加；INT8 也要求手工融合关闭。

## 回归与异常覆盖

最终选择 80 项回归：79 通过、1 跳过（需要显式真实 VAE checkpoint 的可选检查）。
涵盖 CLI/HTTP 缓存参数、服务入口、局部编译、权重预算、预取跳层、非整除 SP 序列、
全局加权缓存误差与决策不一致、CFG 分支隔离、Ring2、CPU INT8 转换、量化内核、
部分 H2D 失败、peer 失败、取消后两卡权重归还 CPU、重复窗口副本复用和关闭。
未执行已有的四逻辑 rank CPU 模拟，也未启动任何四 GPU 任务。
命令与日志为结果目录的 `run_final_tests.py`、`final_tests.log`；`git diff --check` 通过。


## 补测：逐层卸载 + 单卡计算优化

在物理 GPU 6 上补跑同一完整视频，使用逐层卸载 + FFN 局部 compile（native Linear 边界）+ SDPA，
CFG / SP degree 均为 1，关闭残差缓存、文本投影缓存及量化。未先跑短片，也未显式预热；
任务内发生的编译计入耗时。其余输入、seed、步数和 strength 与上表相同。

| 完整任务耗时 | 相对原算法基线加速 | 峰值 allocated | SSIM / MSE / MAE |
| ---: | ---: | ---: | --- |
| 433.56 s | 0.92× | 33.64 GiB | 1.000000 / 0 / 0 |

模型加载耗时 31.76 s，不计入上表；三项质量指标通过，完整解码 145 帧。
本次比历史仅逐层卸载的 420.29 s 慢约 3.2%，显存峰值相同，未观察到单卡编译收益。
这是一次完整任务观测，不能据此认定所有输入或稳态重复推理都不会从编译获益。

命令、日志、输出与质量数据位于 `results/composable_acceleration_20260923/dynamic_single_native_sdpa/`，
复跑脚本为 `results/composable_acceleration_20260923/run_single.py`：

```bash
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
MGERASE_COMPILE_LINEAR_BACKEND=native \
python results/composable_acceleration_20260923/run_single.py
```
