# 四卡 CFG×SP 实测（2026-09-27）

使用用户指定的物理 GPU **2、3、6、7**，四张 A100 80GB，均有 NVLink。
7 号卡开始时另有约 13 GiB 分配，未干预其他进程；表内显存为本进程 PyTorch allocated。
沿用原片 `data/113000356.mp4`、mask、prompt、seed=42、50 配置步数、strength=0.8、
121 帧窗口、9 帧 overlap，尾窗口减填充为 33 帧；输出仍为 145 帧 1920×1080、24000/1001 fps。
参考为 `results/performance_20260927/fused50.mp4`，模型加载单列于请求时间之外。

## 配置与结果

四卡采用 **CFG2×SP2 / Ulysses / sharded Linear / SDPA**。
两组卡分别处理正、负分支，各组再分摊序列；DiT 副本常驻，T5/VAE 保留主卡 CPU offload。
sharded Linear 只计算本地 token，不用全长补零维持 GEMM 行数。
本次没有导入 SGLang，也未声称用上 NCCL、多卡 VAE、编译或四卡 INT8。

| 配置 | 请求 s | 去噪 s | RGB SSIM |
|---|---:|---:|---:|
| 双卡 CFG SDPA（前轮） | 141.052 | 102.288 | 0.992833 |
| 四卡 CFG2×SP2，原通信 | 98.365 | 59.176 | 0.992833 |
| 四卡 CFG2×SP2，大张量直写 | **96.151** | **57.761** | **0.992833** |
| 双卡 CFG TeaCache（前轮） | 79.206 | 36.682 | 0.979842 |
| 四卡 CFG2×SP2 TeaCache，首轮 | **59.611** | **20.504** | **0.979842** |
| 四卡 CFG2×SP2 TeaCache，复跑 | **58.810** | **20.479** | **0.979842** |

四卡无缓存相比双卡请求减少 **31.8%**、约 **1.47×**；四卡 TeaCache 首轮相比
双卡 TeaCache 请求减少 **24.7%**、约 **1.33×**，去噪减少约 **44.1%**。
这些是同一素材的完整请求结果，双卡数字来自前轮，未构成随机交错多次稳态统计。
额外两张卡仍有收益，但没有线性加速；约 39 秒的非去噪开销并未因 SP 减半。
四卡 TeaCache 两次均约 **59 秒**，输出 RGB 哈希一致；两次结果范围不能代替多素材稳态统计。

无缓存峰值 allocated：**34.045 / 5.170 / 5.170 / 5.170 GiB**。
TeaCache 峰值 allocated：**34.045 / 5.421 / 5.421 / 5.421 GiB**。
主卡 reserved 均为 **52.074 GiB**。逐卡峰值发生时间不同，不能相加当作同时总峰值。
主卡整体峰值没有下降：它仍承担全分辨率视频/VAE/前后处理；序列分片只减小部分 DiT 激活。

## 本轮通信改动

`models/adapters/eraserdit/mesh.py:PeerExchange.exchange` 对输出 ≥32 MiB 的 CUDA 张量，
直接分配最终拼接缓冲，各 peer 使用非阻塞 copy 写入相应 slice；避免先复制临时张量再 cat。
保留生产者等待、消费者完成等待和 barrier，防止源存储在远端复制完成前被复用。
小张量保留旧路径。报表新增 `direct_copy_tensors_per_rank`，可验证路径实际被执行。
首窗口每 rank 使用 **4480** 次大张量直写，尾窗口 **0** 次。

四 GPU、两组 SP2、五组交错顺序微基准（每组 30 次 QKV+输出交换，中位数 ms）：

| 总 token 数 | 原复制+cat | 总是直接写入 | event 等待试验 |
|---|---:|---:|---:|
| 32640 | 2.777 | **2.159** | 2.478 |
| 10200 | **1.211** | 1.681 | 1.648 |

因此只选择大张量直写，未采用 event 版本。32 MiB 是针对本机长/短窗口筛选的启发式阈值，
不宣称覆盖所有 GPU、拓扑和形状。微基准保存在 `bench_exchange.py`，保留旧实现可复现对照。
完整请求 98.365→96.151 s、去噪 59.176→57.761 s；单次配对不足以宣称稳定 2.2 秒收益。

## 质量与验证

四卡普通并行整段 SSIM **0.9928330497720625 ≥0.98**。
改通信前后输出逐帧 RGB SHA256 相同，并与此前单/双卡 SDPA 减填充输出一致：

`d04a28190a3a57c394d9abf375b0e488dcbf82543dcecec2c6d1add70532165b`

TeaCache 参数：threshold=0.3、max consecutive skip=3、warmup=4、end guard=1。
两个窗口共 160 个逻辑分支步复用 108 个（67.5%）；SP rank 的重复执行统计没有翻倍计入逻辑命中率。
整段 SSIM **0.9798420760619393** 仅诊断，不适用普通并行的 0.98 拒绝门槛。
逐帧解码完整；抽查 **0、24、48、72、96、132** 帧，站立和下落人物已基本擦除，
未见明显人形残留或黑帧；背景/水花纹理有变化。此为抽样检查，不代表多素材或逐帧动态播放验收。
对照图与检查记录：`results/four_gpu_20260927/review.jpg`、`visual_review.json`。

CPU mesh/组合配置专项 **12 项通过**。新增物理四卡测试 `tests/test_mesh_gpu.py`，
检查两个独立 SP2 组、SP4、不等长序列、非连续源/目标 slice、反复原地复用源存储、大/小张量分支，已通过。
现有与新增 GPU 测试一起运行：5 项，4 通过、真实 VAE checkpoint 项未启用而跳过。
全量 `python -m unittest discover -s tests`：123 项，**115 通过、8 条件跳过**；
`git diff --check` 通过。

## 复现

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 \
python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/four-gpu-tea.mp4 \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 50 --strength 0.8 --infer-len 121 --overlap 9 \
  --cfg-degree 2 --sp-degree 2 --sp-linear-mode sharded --parallel-devices 0,1,2,3 \
  --no-dit-layerwise-offload --no-dit-cpu-offload \
  --text-encoder-cpu-offload --vae-cpu-offload \
  --attention-backend sdpa --operator-fusion-backend auto \
  --compact-tail-padding --no-cache-text-projections \
  --transformer-cache-mode teacache --teacache-threshold 0.3 \
  --max-teacache-consecutive-skip 3 --teacache-warmup-steps 4 --cache-end-guard-steps 1
```

无缓存改为 `--transformer-cache-mode off`，去掉 TeaCache 参数。
本轮实际 Python：`/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python`。
物理 GPU 测试：

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 ERASERDIT_TEST_MESH=1 OMP_NUM_THREADS=2 \
python -m unittest tests.test_mesh_gpu
```

原始日志、提取 JSON、整段质量报告和汇总均位于 `results/four_gpu_20260927/`。
后续讨论见 [按瓶颈安排的优化方案](next_optimization_plan_20260927.md)。
