# 辅助脚本

从仓库根目录运行，使用已安装项目依赖的 Python；日常推理见 [CLI](../docs/cli.md)，
指标和限制见 [性能说明](../docs/performance.md)。Python 脚本建议用模块方式运行，保证项目包可导入：

```bash
python -m scripts.validation.optimization_quality --help
python -m scripts.benchmarks.parallel_benchmark --help
```

若直接执行 `.py` 文件，需设置 `PYTHONPATH=.`。GPU 工具另需指定可用设备和本地权重，
建议设置 `HF_HUB_OFFLINE=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
产物写入 `results/` 或显式指定目录，不写入脚本目录。

## benchmarks：性能测量

| 脚本 | 用途 |
| --- | --- |
| `cache_benchmark.py` | 缓存交错 A/B；使用 `--video`、`--mask`、`--prompt`、`--directory` |
| `cfg_parallel_benchmark.py` | 旧双卡 CFG A/B |
| `parallel_benchmark.py` | CFG/SP/VAE 组合矩阵，`--matrix-configs` 选项见帮助 |
| `quantization_benchmark.py` | BF16 / INT8 A/B，独立记录转换耗时 |
| `parallel_four_gpu.sh` | 固定物理 2、3、6、7 空闲检查及四卡矩阵；占用退出 75，`--wait` 等待 |

除 cache 工具外，Python 基准脚本沿用 CLI 输入参数，`--output-path` 指 JSON 报告，视频保存在旁边。

## validation：功能与质量验证

| 脚本 | 用途 |
| --- | --- |
| `optimization_quality.py` | 当前 RGB 质量门禁；缓存/量化加 `--lossy` 留待目视验收 |
| `cache_compare.py` | 缓存区域/逐帧 Y 通道指标，可指定外部 `--reference` |
| `quantization_contact_sheet.py` | BF16 / INT8 抽帧对照图 |
| `offload_verify.py` | 常驻会话重复请求与注入异常恢复，输出路径为 JSON |
| `acceptance.py` | 批量输出完整性、seed 隔离、相对源视频的非擦除区检查；依赖相邻 baseline 工作树 |
| `service_smoke.py` | 对已运行服务验收接口、任务生命周期和下载 |
| `service_verify.sh` | 启动临时服务并验收，退出时清理该服务；可指定 GPU |

服务验收示例见 [服务 API](../docs/service_api.md)。`acceptance.py` 的旧非擦除区门槛
不等同于优化相对未优化输出的 RGB 门禁。

## legacy：历史实验复现

保留 M1b/M3/M4 工具用于追溯已有数据，不作为当前验收标准：

- `m1b_model_ab.py`：迁移前后模型 A/B。
- `m3_attn_ab.py`：注意力 A/B；`m3_measure.sh`：单配置测量。
- `m3_sweep.sh`、`m3_probe.sh`、`m3_cleanup.sh`：历史扫描与候选复测。
- `m3_report.py`：M3 汇总。
- `m4_measure.sh`、`m4_report.py`、`m4_side_by_side.sh`：双素材测量、汇总与四联视频。

这些工具包含本机权重、素材及相邻 `EraserDiT-baseline` 工作树假设，迁移机器时需检查配置。
Shell 脚本用 `bash scripts/legacy/<名称>.sh ...`，Python 报告用 `python -m scripts.legacy.<名称> ...`。
旧报告如需完整内容可从 Git 历史查阅，关键结论与原始产物索引已合并到性能文档。
