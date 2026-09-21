# 四卡组合并行测试（2026-09-21）

最新验收规则见 [优化验收标准](optimization_acceptance.md)：非 cache/量化优化采用 SSIM ≥ 0.985、MSE ≤ 36、MAE ≤ 6；cache/量化按视觉大致一致验收。下文历史字节一致结果保留，不再作为必须条件。

使用物理 GPU 2、3、6、7，用户本轮直接指定该设备池。7 号卡启动时有 12,865 MiB 常驻占用、利用率 0%；测试未修改或终止其他进程。性能属于存在其他常驻任务的单次探索测量。

TeaCache、Cache-DiT、文本投影缓存关闭；TP、PP 未启用。SP 使用 reference 模式；VAE 使用保留完整空间上下文的空间分片，不启用近似 tiling。

## 分组

- CFG=2×SP=2：2/3 号卡负责正分支，6/7 号卡负责负分支，每个分支内 SP=2。
- 再加 VAE=4：编码和解码阶段复用全部四卡。
- SP=4：四卡协作计算一个分支，正负分支依次执行。

## 验证

四卡环境下 `test_eraserdit_mesh_gpu.py` 四项测试通过，包含实际四卡 CFG×SP 和 SP=4。真实 VAE 单元测试仍是双卡；四卡 VAE 由下述完整视频矩阵覆盖。

192×320、25 帧、两个窗口、6 采样步 / 5 有效去噪步，SDPA：

| 配置 | 端到端秒数 | 与单卡输出 |
| --- | ---: | --- |
| serial | 3.507 | 字节一致 |
| cfg2_sp2 | 4.443 | 字节一致 |
| sp4 | 6.673 | 字节一致 |
| cfg2_sp2_spatial_vae4 | 4.801 | 字节一致 |
| sp4_spatial_vae4 | 7.226 | 字节一致 |

小尺寸下通信及副本创建成本超过收益，不能从此表推断原尺寸性能。


## 原尺寸结果

素材 `10268234`，1080×1920、120 帧、50 采样步 / 40 有效去噪步，SageAttention。
同一常驻会话顺序运行，每配置一次，副本建立计入请求耗时；模型初次加载另计。

| 配置 | 端到端秒数 | 对单卡加速 | 各卡峰值 allocated GiB（2/3/6/7） |
| --- | ---: | ---: | --- |
| serial | 210.302 | 1.000× | 47.115 / 0.000 / 0.000 / 0.000 |
| cfg2 | 119.000 | 1.767× | 47.115 / 5.631 / 0.000 / 0.000 |
| cfg2_sp2 | 87.682 | 2.398× | 47.146 / 5.539 / 5.539 / 5.539 |
| cfg2_sp2_spatial_vae4 | 88.191 | 2.385× | 44.312 / 25.059 / 25.059 / 25.044 |

四份 MP4 与单卡基线逐字节一致，SHA256：
`0b50b4c9176930c1adbb289298b5ea67c70525c352b58138c6b0ca786b876e0b`。

CFG×SP 相对本轮 CFG 双卡耗时减少 26.3%。
叠加 VAE=4 没有显示额外端到端加速；优先选 CFG=2×SP=2。
CFG×SP 的主卡峰值没有下降，不能将其视作四卡均分显存方案。
叠加 VAE=4 后主卡峰值从 47.146 降至 44.312 GiB（约 6.0%），但其余卡各需约 25 GiB，且未显示提速。
7 号卡有其他常驻进程；监控记录为 `gpu_occupancy.jsonl`。此轮不代表独占四卡的稳态统计。

产物目录：`results/parallel_four_20260921/`。

## 复现 CFG×SP

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 HF_HUB_OFFLINE=1 ./inference_cli.sh \
  --model-path results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/cfg_sp_four.mp4 --prompt "There is a bridge over the lake." \
  --attention-backend sage_attn --cfg-degree 2 --sp-degree 2 \
  --transformer-cache-mode off --no-cache-text-projections
```

本次按最新用户指定的四卡池直接测试，未使用要求所有卡占用 ≤64 MiB 的自动等待脚本。

## DP 四卡组合

DP=2×CFG=2 和 DP=2×SP=2 均通过真实模型连续任务测试：
每种组合分成物理 2/3 与 6/7 两个任务组，各处理两个任务，共八份输出均与小尺寸单卡基线字节一致。
这是功能验证，未对 DP 吞吐做稳态性能结论。worker 退出码、分卡及任务分配见
`dp_cfg/report.json` 和 `dp_sp/report.json`。

全部结果索引为 `results/parallel_four_20260921/validation_summary.json`。

## 补测：原尺寸纯 SP=4

同一原始视频 `data/10268234.mp4`，1080×1920、120 帧、40 个有效去噪步，SageAttention；
本轮在同一常驻会话重跑单卡后执行 SP=4，CFG=1、VAE=1、缓存关闭。

| 配置 | 端到端耗时 | 对本轮单卡加速 |
| --- | ---: | ---: |
| 单卡 | 209.114 s | 1× |
| 纯 SP=4 | 118.665 s | 1.762× |

耗时减少 43.3%，输出与单卡逐字节一致。
物理 2/3/6/7 各卡峰值 allocated GiB：47.115 / 5.346 / 5.346 / 5.346。
纯 SP=4 仍慢于上一轮 CFG=2×SP=2 的 87.682 s；当前四卡优先 CFG×SP。
结果属于单次探索测量，7 号卡有其他常驻占用，未作为独占稳态统计。
报告：`results/parallel_four_20260921/full_sp4.json`；校验：`full_sp4_validation.json`。
复现时将上方命令的 `--cfg-degree 2 --sp-degree 2` 改为 `--sp-degree 4`。
