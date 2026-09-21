# 优化验收标准

2026-09-21 用户更新：不再要求优化结果与基线逐字节一致。
本标准覆盖此前文档中的逐字节验收要求及 SSIM 0.95 / PSNR 28 dB 等旧质量门槛；历史测量保留。

## 非 cache、非量化优化

并行、卸载、注意力执行方式等优化，必须同时满足：

- SSIM ≥ 0.985。
- MSE ≤ 36。
- MAE ≤ 6。

默认计算口径：同一输入、权重、提示词、seed、采样和窗口配置，相对未优化输出；
解码 RGB、像素范围 0–255，MSE/MAE 在所有帧、像素和通道上平均；
SSIM 使用每通道 11×11 Gaussian 窗口、sigma=1.5、reflect 边界，再平均。
三项均为整段全帧指标；另记录每帧结果供诊断，不额外附加未获要求的逐帧门槛。
现有 Y 通道报告不直接等同于新 RGB 验收结果。

入口：

```bash
python scripts/optimization_quality.py \
  --reference baseline.mp4 --candidate optimized.mp4 --report quality.json
```

通过退出 0；质量不通过仍写报告并退出 2。视频帧数、尺寸、帧率必须一致。
字节相同可作为更强的诊断证据，但不再作为性能优化通过的必要条件。
功能测试中的通信正确性、设备分组和异常清理等断言仍保留，不统一放宽。

## Cache 与量化

用户要求图像观感大致相同，不套用上述硬指标。使用 `--lossy` 时继续记录数值，
但标记 `visual_review_required`、`accepted=null`，不能用“关闭指标门槛”冒充已经完成视觉验收。
视觉对比关注擦除结果、结构与颜色，以及视频中的明显闪烁或损坏。

本次变更仅改变验收标准，不自动开启 TeaCache、Cache-DiT 或量化；当前并行实验继续关闭缓存。
SP reference 兼容模式保留；sharded 模式通过原尺寸质量测试后再调整推荐配置。

## 旧报告入口

`scripts/m3_report.py`、`scripts/m4_report.py` 的 SSIM/PSNR 门槛属于历史里程碑报告，
不能作为当前优化的通过依据；新实验统一追加 `optimization_quality.py` 报告。
`scripts/acceptance.py` 对原始输入非擦除区的检查属于另一种运行时验收，
并非优化输出相对未优化输出的质量对照，保留其功能检查。

## 本轮 SP 局部计算验证

`sp_linear_mode=sharded`、SP=2、原尺寸 `10268234`、40 有效步、SageAttention，
缓存关闭，输出 SHA256 与未优化基线相同。见
`results/parallel_quality_20260921/sharded.json` 与 `sharded_quality.json`。
GPU 3 在测试期间有其他计算负载；295.561 s 的耗时不用于性能结论。
这是单素材质量验证，不自动修改默认 reference 模式；四卡组合及更多输入仍需按新标准验证。
