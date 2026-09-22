# TeaCache / CacheDiT 阈值 0.3 完整推理验证

2026-09-22：TeaCache 和 CacheDiT 的默认阈值统一为 `0.3`，覆盖 CLI、采样参数、HTTP 请求模型和底层缓存参数。缓存与量化允许有损；本次只重跑两种缓存，量化保持 `none`。

## 实测结果

沿用 `/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python`，GPU 2，输入 `data/113000356.mp4` 与对应 mask；1920×1080、145 帧、24000/1001 FPS、两个窗口。SDPA、BF16、fullgpu、seed=42、配置步数=50、strength=0.8，每窗口实际去噪 40 步。启用文本投影缓存，warmup=4、末步保护=1、最多连续复用一步，预测器为 `none`。

| 配置 | 完整任务耗时（不含模型加载） | 纯推理 | 峰值 allocated | 缓存复用率 | 相对历史基线加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TeaCache 0.3 | 248.20 s | 215.83 s | 47.14 GiB | 45% | 1.61× |
| CacheDiT 0.3 | 252.65 s | 219.00 s | 47.14 GiB | 45% | 1.58× |

两项各执行一次，共用一个常驻模型会话；包含加载的总墙钟时间为 545.77 s，峰值 reserved 均为 69.45 GiB。每项两个窗口的 CFG 分支合计 160 步，计算 88 步、复用 72 步；CacheDiT 的复用指中段残差复用，仍计算探针。两个窗口均正常关闭，没有中止。

速度比使用历史相同素材的 SDPA/fullgpu、缓存关闭基线 400.37 s；该基线没有启用文本投影缓存。本次不是同日重复统计或只改变阈值的隔离基准，速度比仅代表单次观察。旧性能结果（包括 SP=2）保留在[优化验证报告](optimization_validation_20260922.md)。

## 完整性与质量

两项均退出码 0，FFprobe 确认 145 帧、尺寸和帧率一致；FFmpeg 使用 `-xerror` 完整解码成功。缓存、CLI、采样参数与 HTTP 默认值核对通过，mask 阈值仍为 `0.039`。缓存单测 17 项通过、1 项跳过；服务 API 测试 9 项通过。

相对历史无缓存视频的逐帧 RGB 指标如下，SSIM 使用 11×11 Gaussian、sigma=1.5、reflect 边界并按通道及帧平均：

| 配置 | SSIM | MSE | MAE |
| --- | ---: | ---: | ---: |
| TeaCache 0.3 | 0.982012 | 3.138675 | 1.265181 |
| CacheDiT 0.3 | 0.981961 | 3.131396 | 1.255964 |

缓存允许有损，上述指标只用于记录差异，不按非缓存、非量化门槛判定失败。本次完成数值和完整解码检查，没有进行整段人工播放验收。

## 复跑命令与记录

在仓库根目录执行以下命令，使用原环境顺序运行两种缓存的完整视频：

```bash
bash results/cache_threshold_03_20260922/command.sh
```

脚本显式传入 `--teacache-threshold 0.3 --cache-dit-residual-diff-threshold 0.3`，由 `tasks.json` 分别选择 `teacache` 与 `cache_dit`。完整命令与环境见[command.sh](../results/cache_threshold_03_20260922/command.sh)；数值汇总见[summary.json](../results/cache_threshold_03_20260922/summary.json)，完整运行参数与缓存历史见[payload.json](../results/cache_threshold_03_20260922/payload.json)。

输出：[TeaCache 视频](../results/cache_threshold_03_20260922/teacache.mp4)、[CacheDiT 视频](../results/cache_threshold_03_20260922/cache_dit.mp4)。`run.log`、`verification.log`、各模式的 `*_quality.json`、测试日志和完整解码 RGB 哈希均保留在同目录。`initial_setup_*` 是已中止的准备阶段记录，不计入本报告结果。
