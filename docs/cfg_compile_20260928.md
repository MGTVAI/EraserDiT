# 双卡 CFG 与完整 DiT 编译（2026-09-28）

本轮将完整 DiT 编译扩展到 CFG2/SP1：正、负提示词分支分别在一张 GPU 上运行，
每张卡持有完整权重和自己的编译入口。没有引入 `sglang` 包，也没有改变 SP 通信实现。
仍为单进程双线程、peer copy；不能据此声称已支持 NCCL 或 CFG2×SP2 完整编译。

## 实现与边界

- `EraserDiTReplicaPool` 为每张卡保存完整编译入口，主卡复用 denoising stage 的入口，
  副卡不再错误地套用 FFN 编译。入口在多个窗口与请求间复用。
- mesh 在图外处理设备搬运、静态条件和 RoPE，按 rank 调用对应编译入口。
  静态输入搬运支持嵌套元组，避免预计算 RoPE 留在主卡。
- 每个窗口第一个 CFG 步骤依次执行两卡，避免同时首次追踪与初始化；后续步骤并发。
  首步是真实推理，不额外更新缓存或 scheduler。
- 显式回退清理两卡编译入口；副卡抛错后释放窗口资源，下一窗口可重新使用副本池。
- `torch_compile.rank_compile` 保存逐卡首次调用与成功 forward 数；顶层成功数为两卡总和。
  顶层 `first_call_history` 仍仅表示主卡。首次调用记录不是 Dynamo 编译图数量计数器。

需要 SP1、CFG1/CFG2、SDPA、未量化且常驻的 DiT、关闭手工融合与残差/文本投影缓存。
旧式 `cfg_parallel_device` 不支持完整编译，请使用 `--cfg-degree 2`。
T5 与 VAE CPU offload 可以保留。使用 `fullgraph=True, dynamic=False`、关闭 CUDA Graph，
编译失败直接报错。没有自动开启实验性功能或修改默认编译范围。

## 实验条件

物理 GPU 3、6，两张 A100 80GB；实验开始时均空闲，其他卡仍有外部任务。
Torch 2.6.0+cu126，BF16，编译模式 `default`，CPU/Inductor 编译线程均为 4。
原片 `data/113000356.mp4` 及对应 mask，145 帧、1920×1080、24000/1001 fps；
seed 42、50 配置步数、strength 0.8、infer_len 121、overlap 9、compact tail。
两个窗口各 40 个实际去噪步骤，每请求两卡各执行 80 次 forward。

同一个常驻会话按顺序运行：编译首次请求 → eager 1 → 编译 1 → eager 2 → 编译 2。
eager 对照只切换到相同副本的原始 forward，保持 SDPA、无手工融合、无残差缓存。
eager 1 是该会话首次完整 eager 请求，没有单独丢弃一个 eager 预热请求，因此同时报告第二轮。
首次编译允许复用磁盘中的 Inductor 缓存，不代表清空磁盘缓存后的最坏冷启动。
请求耗时在所有参与 GPU 同步后记录，包含视频 I/O 和保存，不含模型加载。
质量检查在请求计时结束后执行。

## 同会话编译对照结果

模型加载 **43.007 秒**，单列于请求之外。

| 配置 | 请求秒 | 去噪秒 | 主/副卡 allocated 峰值 GiB |
|---|---:|---:|---:|
| 编译首次请求 | 253.567 | 208.461 | 34.045 / 5.829 |
| eager 第一次对照 | 162.242 | 112.910 | 34.108 / 5.826 |
| 编译预热后第一次 | 142.994 | 91.902 | 34.108 / 5.829 |
| eager 第二次对照 | 163.319 | 112.378 | 34.108 / 5.827 |
| 编译预热后第二次 | 143.726 | 92.328 | 34.108 / 5.829 |

两次对照中位数：请求 **162.781 → 143.360 秒，减少 11.93%**；
去噪 **112.644 → 92.115 秒，减少 18.22%**。
这是相对无手工融合 eager 的结果，不等于相对现有最快手工融合方案的收益。
仅同一素材、两次测量，不构成多素材或大样本稳态统计。

主卡长/短窗口首次调用 **24.478 / 18.655 秒**；副卡 **39.139 / 37.990 秒**。
这些耗时包含编译与一次计算，首次请求不能作为预热后性能。
每次编译请求的成功次数增量均为 `[80, 80]`，eager 对照为 `[0, 0]`。
完整编译已在两张卡上执行，没有只编译主卡却报告双卡编译。

编译前后 allocated 峰值基本相同。主卡 reserved 从首次 52.074 增至后续 62.459 GiB，
同一进程保留了分配器历史，不能将其直接解释为编译增加了约 10 GiB 活跃显存。
副卡 reserved 峰值：编译预热后约 7.043 GiB，eager 约 7.697 / 7.939 GiB。
不同卡的峰值发生时刻不同，不能直接相加当作同时总峰值。

## 与现有手工融合方案比较

另启一个进程，使用相同物理 GPU、视频和推理参数，关闭编译并开启
`--operator-fusion-backend auto`，连续执行首次与预热后请求：

| 双卡配置 | 请求秒 | 去噪秒 |
|---|---:|---:|
| 手工融合，首次请求 | 152.054 | 104.323 |
| 手工融合，预热后请求 | 150.910 | 103.583 |
| 完整编译，预热后两次中位数 | 143.360 | 92.115 |

相对这次手工融合预热结果，编译请求减少 **5.00%**、约 **7.55 秒**；去噪减少 **11.07%**。
手工融合加载 44.385 秒，未计入表内。融合首次请求期间另有 CPU 视频哈希/抽帧检查，
它们在融合预热后请求前已经完成；比较采用预热后结果。
这是同机顺序运行的独立进程对照，融合仅一次预热后样本，证据弱于前面的同会话交错对照。
此前约 141 秒的双卡数据来自另一轮，不能拿本轮 143 秒直接判断性能退化。

编译首次请求仍需 253.567 秒，不适合用首次延迟作为优势。
按这组数据粗算，相对手工融合约需 **15 次相同形状请求**才抵消首次额外成本；
这是由首次/预热时间推算的回本点，没有实跑 15 次，也不适用于频繁切换分辨率或窗口形状。
本轮保留默认不开启，适合显式选择用于能复用编译图的连续推理。

未编译的独立融合进程，主卡 reserved 也从首次 **52.074 → 62.203 GiB**，
预热后 allocated 为 **34.108 GiB**；说明此次 reserved 增长并非完整编译独有。
这不等于已定位分配器增长的具体来源，也不等于排除其他形状下的编译显存开销。

## 画质与重复性

全部输出为 145 帧、1920×1080、24000/1001 fps。
相对原始 `fused50.mp4`：整段 RGB SSIM **0.9823088061 ≥0.98**，
MSE 2.647895、MAE 1.088271、最差帧 SSIM 0.961152、temporal error MAE 1.087299。
0.98 是整段阈值，不代表每帧均达到 0.98。
相对此前单卡完整编译输出 SSIM **0.9831866389**，并非像素完全一致；
本轮没有将差异归因于某个未经单独验证的 kernel。

三次双卡编译输出的逐帧 RGB SHA256 完全相同：

`5ed426abf54a0ee9841257b3fa31d38fbbe01db70287aaa6edd7cbb956378e7a`

两次 eager 对照也完全相同，且与此前单/双/四卡普通并行输出一致：

`d04a28190a3a57c394d9abf375b0e488dcbf82543dcecec2c6d1add70532165b`

补测的两次手工融合输出也具有此哈希。

抽查帧 0、24、48、72、96、132 的三列对照图 `review.jpg`，未见明显人形残留或黑帧；
背景与水花纹理存在差异。这是缩略图抽帧检查，不代表全分辨率逐帧或动态播放验收。

## 代码验证

- 全量 CPU 回归 `python -m unittest discover -s tests`：133 项，105 通过、28 条件跳过。
- GPU 可见且启用 `ERASERDIT_TEST_FULL_COMPILE=1 ERASERDIT_TEST_CFG_COMPILE=1` 的
  `tests.test_transformer_compile`：6 项全部通过，包含真实双卡 Inductor 执行。
- 真实双卡 `tests.test_mesh_gpu.GPUParallelTests.test_cfg_sp_and_available_hybrid` 通过，
  检查既有 CFG2 与 SP2 路径。
- CPU 捕获测试确认两卡模型、两个窗口形状共四张完整图，改变 timestep 或复用窗口不新增图；
  检查副卡异常后的资源释放、下一窗口恢复、报告快照隔离及显式回退。
- `git diff --check` 通过。结果目录保存 `regression.log`、`compile_gpu.log` 和 `mesh_gpu.log`。

## 复现

```bash
CUDA_VISIBLE_DEVICES=3,6 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
TORCHINDUCTOR_COMPILE_THREADS=4 MGERASE_TORCH_COMPILE_MODE=default \
python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/cfg2-compile.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 \
  --infer-len 121 --overlap 9 --compact-tail-padding \
  --cfg-degree 2 --sp-degree 1 --parallel-devices 0,1 \
  --no-dit-layerwise-offload --no-dit-cpu-offload \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --attention-backend sdpa --operator-fusion-backend disabled \
  --transformer-cache-mode off --no-cache-text-projections \
  --enable-torch-compile --torch-compile-scope transformer
```

本地脚本 `results/cfg_compile_20260928/validate.py` 接收相同参数，在同一会话内执行上述对照；
`summarize.py` 汇总结果并逐帧计算 RGB SHA256，`review.py` 生成抽帧对照图。
这些实验的 Python 为 `/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python`。
