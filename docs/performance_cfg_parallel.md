# EraserDiT 双卡 CFG 并行

2026-09-20，按用户要求仅使用物理 GPU 2、3，TeaCache、Cache-DiT 及文本投影缓存全部关闭。

## 实现

`CUDA_VISIBLE_DEVICES=2,3` 将物理 2、3 映射为本地 `cuda:0`、`cuda:1`。
主卡负责文本编码、VAE、正 CFG 分支、scheduler 和视频写出；辅卡只持有 Transformer 副本并计算负分支。
每步在两个线程上提交独立模型前向，将负分支结果搬回主卡，以原顺序在 float32 合并。
不跳步、不复用模型计算结果、不改变 seed 或窗口衔接。

副本和单个工作线程属于当前窗口；正常结束与异常退出均清理，不在连续请求间保留条件张量。
副本创建耗时包含在去噪与端到端耗时内，同时单独报告 `replica_setup_seconds`。
当前复制过程会短暂在主卡增加一份 Transformer 权重；辅卡显存与主卡显存分别记录。
此版明确拒绝与 CPU 卸载、torch.compile、TeaCache / Cache-DiT、文本投影缓存组合。
默认仍是单卡；此入口是单进程双卡，不使用 torchrun，也未接入序列并行或 VAE 并行。

## 运行

```bash
CUDA_VISIBLE_DEVICES=2,3 ./inference_cli.sh \
  --model-path results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/cfg_parallel/output.mp4 \
  --prompt "There is a bridge over the lake." \
  --attention-backend sage_attn --cfg-parallel-device cuda:1 \
  --transformer-cache-mode off --no-cache-text-projections
```

同进程 A/B（输出 JSON、两份视频；重复次数可设为 5，轮换执行顺序）：

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python scripts/cfg_parallel_benchmark.py \
  --model-path results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/cfg_parallel/full_10268234.json \
  --prompt "There is a bridge over the lake." \
  --attention-backend sage_attn --cfg-parallel-device cuda:1 \
  --transformer-cache-mode off --no-cache-text-projections --repeat 1
```

## 验证

- 全量回归 42 项：41 项通过，双卡 opt-in 测试默认跳过；另以 `ERASERDIT_TEST_TWO_GPU=1` 单独运行双卡测试，2 项全部通过。
- 双卡测试：小型真实 Transformer 的多步、多窗口预测与单卡逐位相同；工作线程异常传播和清理通过，配置冲突拒绝通过。
- 192×320、25 帧、双窗口、6 个采样步、SDPA 真实完整模型单卡 / 双卡输出 SHA256 相同：`8ec23485438d3b0ae9defe3897b47a5eb8b78a34db1759e92d14f809c21db81b`。
- 原尺寸第一组、完整 40 个有效去噪步，单卡 / 双卡视频逐字节相同：`0b50b4c9176930c1adbb289298b5ea67c70525c352b58138c6b0ca786b876e0b`。
- 320×192、33 帧、三窗口（0:17 / 8:25 / 16:33），每窗口 5 个有效步；同一会话轮换单卡 / 双卡各两次，四份输出逐字节相同。两模式独立预热已验证，每次任务的缓存历史均为三条 off，双卡每窗口均报告执行 5 步。数据：`results/cfg_parallel/three_windows.json`。小尺寸中位耗时单卡 3.320 s、双卡 3.413 s，副本和调度开销超过并行收益，因此不推荐用双卡加速此类短小任务。

## 原尺寸首轮结果

素材 `10268234`：1080×1920、120 帧、单窗口；seed=42、50 个采样步、strength=0.8、CFG=3。
两侧使用相同本地权重、BF16、SageAttention、确定性口径，compile / 算子融合 / 所有推理缓存关闭。
同一常驻会话，先单卡后双卡，各一次；模型加载另计，窗口副本创建计入以下数据。

| 指标 | 单卡（物理 2） | 双卡（物理 2、3） |
| --- | ---: | ---: |
| 端到端 | 209.788 s | 118.259 s |
| 去噪 | 181.240 s | 91.399 s |
| 主卡峰值 allocated | 47.115 GiB | 47.115 GiB |
| 辅卡峰值 allocated | 0 | 5.631 GiB |
| 窗口副本创建 | — | 0.335 s |

首轮端到端 **1.774×**，去噪 **1.983×**，分别减少约 43.6%、49.6% 时间。
双卡报告确认辅卡执行 40 步，缓存历史为 `off`，文本缓存为 false、缓存张量保留 0 字节。
完整数据：`results/cfg_parallel/full_10268234.json`。

这是单次探索性结果，没有独立预热；单卡测量期间有短暂的小型回归测试进程并发。
不能代替独占窗口下 ≥5 次预热后稳态统计。第二组原尺寸双窗口的完整质量 / 性能验收待做。
使用 `--warmup --repeat 5` 可分别为两种模式预热、轮换顺序复测；预热单列并排除在测量外。

测试日志与测量产物位于 `results/cfg_parallel/`。
