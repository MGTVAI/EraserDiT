# 非量化优化验证（2026-09-22）

> 历史记录：下表使用当时的 TeaCache 0.005 / CacheDiT 0.03 默认值。当前两者默认均为 0.3；
> 复现旧记录需显式传入旧阈值。缓存与量化允许有损，表内“数值门槛”仅作诊断，不代表有损方案验收失败。

两种缓存阈值改为 `0.3` 后的完整原片重跑见[追加验证](cache_threshold_03_validation_20260922.md)。

本次覆盖当前可用双卡范围内的全部优化类型及代表组合，不是所有参数的笛卡尔组合。
复用 `/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python`，未新建环境或安装依赖。
初始矩阵使用物理 GPU 2、6（A100 80GB）；追加的 SP=2 原片测试使用 GPU 6、7。
权重为 `data/model/`，BF16，无权重量化。

## 结果与边界

- 44 个端到端配置：37 个短片配置、7 个原片配置，共生成并完整解码 58 个输出视频。
- GPU 回归 99 项：97 项通过，2 项量化测试按要求跳过。
- 7 项不支持组合的启动约束检查均正确拒绝，包括双卡 + TeaCache、双卡 + 卸载、双卡 + TeaCache + 卸载。
- **双卡单任务不能同时开启 TeaCache 或卸载**。支持的 TeaCache + 动态卸载已单独完成原片验证。
- 当前只有两张空闲卡；SP4、CFG2×SP2、VAE4、DP2×CFG/SP 等四卡组合未实测。
- `sage_fp8` 是 sm89 专用后端，当前 sm80 A100 不支持，未运行；INT8/FP8 权重量化未运行。
- 所有配置均执行成功，但执行成功不等于质量通过：21/44 项达到数值门槛，其余 23 项未达到。注意力、编译、QK/RoPE 融合、分块及部分缓存配置有数值差异。

## 测试条件

短片从原素材提取前 33 帧并缩放为 512×288，mask 使用 nearest 缩放；原片为
1920×1080、145 帧、24000/1001 fps。prompt 为
`There is a rooftop terrace overlooking the city at sunset.`，seed=42，50 步，strength=0.8，
每窗口实际去噪 40 步。原片执行两个窗口；短片执行一个窗口。

短片以本次 `short_baseline/output_0.mp4` 对照；原片以此前同环境、同权重、同采样配置的
`results/remove_ltx095_validation/113000356.mp4` 对照（正式推理 400.37 秒、主卡 allocated 47.14 GiB）。
短片基线、SageAttention、SageAttention+compile 各重复 5 次；SDPA+compile 重复 2 次；
其余为单次功能和质量测试。表内耗时不含加载及显式预热；单次原片数值不是五次稳态性能结论。
显存列仅是主卡 PyTorch peak allocated，不是所有 GPU 的合计，也不是 nvidia-smi reserved。

质量按整段 RGB 0–255 计算：SSIM 使用 11×11 Gaussian、sigma=1.5、reflect 边界；
门槛为 SSIM≥0.985、MSE≤36、MAE≤6。检查全部帧的完整解码、尺寸、帧率及帧数；
重复任务另比较整段 RGB 哈希。缓存/分块的数值通过也不等价于感知验收：本次只额外抽帧审阅，
未进行整段人工播放，因此不作无闪烁或生产质量保证。

## 原片结果

| 配置 | 正式推理 s | 主卡 allocated GiB | SSIM | MSE | MAE | 数值门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 双卡 CFG | 225.45 | 47.14 | 1.000000 | 0.000 | 0.000 | 通过 |
| 双卡 SP=2 reference（追加） | 285.62 | 47.14 | 1.000000 | 0.000 | 0.000 | 通过 |
| 动态卸载 2 GiB | 409.31 | 33.64 | 1.000000 | 0.000 | 0.000 | 通过 |
| SageAttention + 两种融合 | 357.38 | 47.14 | 0.983132 | 2.286 | 0.992 | 未通过 |
| TeaCache 0.02 + 动态卸载 | 348.14 | 33.64 | 0.983139 | 2.280 | 0.990 | 未通过 |
| 默认 TeaCache + 动态卸载 | 413.50 | 33.64 | 1.000000 | 0.000 | 0.000 | 通过 |
| TeaCache 默认 0.005 | 408.64 | 47.14 | 1.000000 | 0.000 | 0.000 | 通过 |

追加 SP=2 原片测试：正式推理 **285.62 秒**，纯推理 **254.99 秒**，
加载 **35.29 秒**，进程总耗时 **330.75 秒**；主卡 peak allocated **47.14 GiB**、
peak reserved **70.17 GiB**。两个窗口均执行 40 个去噪步，各 SP rank 每窗口记录 4480 次通信。
全部 145 帧的尺寸和帧率正确，整段 RGB SHA256 与单卡基线相同。

本次 GPU 2 已有活跃任务，因此使用计算空闲的 GPU 6、7；GPU 7 测试前已有 12253 MiB 显存占用。
本次耗时为此前单卡基线 400.37 秒的约 71.3%（时间比约 1.40×），但设备和占用条件不同，
不能将这一次结果视为严格同条件的重复性能结论。原始 43 项性能记录保留不变，
新增结果保存在 [SP=2 性能记录](../results/optimization_validation_20260922/full_sp2_reference/performance.json)，
合并视图为 [44 项汇总](../results/optimization_validation_20260922/summary_with_sp2.json)。

默认 TeaCache 0.005 在短片和原片均为 0 次残差复用：能跑通但没有残差加速收益。
提高到 0.02 后，原片每窗口正负分支合计复用 14/80 次，命中率 17.5%；两个窗口共复用 28 次。
该阈值用于覆盖实际复用路径，不是质量推荐值。线性预测、强制计算和 CacheDiT 实际复用也已覆盖。

双卡 CFG、SP reference、非分块 VAE 双卡、文本投影缓存、单独 RMSNorm+AdaLN 融合及卸载类
配置在对应测试素材上与基线像素一致。SP sharded、QK/RoPE 融合、注意力替换、编译和近似分块
存在数值差异，应按下表和具体素材验证，不能仅凭速度替换基线。

## 可直接运行的原片命令

以下命令在仓库根目录执行。GPU 编号采用本次实际设备，`cuda:0/1` 对应物理卡 2/6。
显式使用现有环境和 fullgpu 基线，避免依赖 CLI 的动态卸载默认值。

```bash
export CUDA_VISIBLE_DEVICES=2,6 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=2
ERASE_PY=/mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python
mkdir -p outputs/optimization_validation
ERASE_CMD=("$ERASE_PY" -m entrypoints.cli.erase_eraserdit
  --model-path data/model
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4
  --prompt "There is a rooftop terrace overlooking the city at sunset."
  --seed 42 --num-inference-steps 50 --strength 0.8
  --resource-policy fullgpu --attention-backend sdpa
  --transformer-cache-mode off --no-cache-text-projections)

# 双卡 CFG：本次原片像素一致
"${ERASE_CMD[@]}" --cfg-degree 2 --output-path outputs/optimization_validation/cfg2.mp4

# 双卡 SP=2：追加测试使用物理 GPU 6、7，原片像素一致
CUDA_VISIBLE_DEVICES=6,7 "${ERASE_CMD[@]}" --sp-degree 2 --sp-linear-mode reference \
  --output-path outputs/optimization_validation/sp2.mp4

# 动态卸载：预算约束受管权重，不代表总显存限制为 2 GiB
"${ERASE_CMD[@]}" --resource-policy dynamic_offload --max-weight-usage 2147483648   --output-path outputs/optimization_validation/offload.mp4

# 单卡精确融合：短片像素一致
"${ERASE_CMD[@]}" --operator-fusion-backend triton --operator-fusion-ops rmsnorm_adaln   --output-path outputs/optimization_validation/adaln.mp4

# TeaCache 默认值：本次素材残差命中为 0
"${ERASE_CMD[@]}" --transformer-cache-mode teacache --cache-text-projections   --teacache-threshold 0.005 --output-path outputs/optimization_validation/teacache.mp4

# 实际命中的 TeaCache + 卸载：有损测试配置，先查看质量结果
"${ERASE_CMD[@]}" --resource-policy dynamic_offload --max-weight-usage 2147483648   --transformer-cache-mode teacache --cache-text-projections --teacache-threshold 0.02   --output-path outputs/optimization_validation/teacache_offload.mp4

# 单卡编译：本次完成短片重复验证；原片该组合未单独实测
"${ERASE_CMD[@]}" --attention-backend sage_attn --enable-torch-compile --warmup   --output-path outputs/optimization_validation/compile.mp4
```

带有注意力替换、编译或实际残差复用的选项应查看矩阵质量结果后再用于实际任务。
每项**实际执行过的完整命令**保存在下表链接的 `command.sh`；包含输入、采样、环境、
GPU、输出和优化参数。可以直接 `bash results/optimization_validation_20260922/<配置>/command.sh` 复现。
DP 复跑需换用尚不存在的 `--parallel-run-dir`，避免覆盖先前的分配记录。

## 完整配置矩阵

短片耗时为正式推理中位数；次数=1 的行即单次结果。DP 的主卡峰值和单任务中位数不由 dispatcher 汇总，
完整 worker 报告保存在其 `workers/` 子目录。

| 配置 / 实际命令 | 优化参数（其余使用显式基线） | 次数 | 推理 s | 主卡 GiB | SSIM | 数值门槛 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [full_cfg2](../results/optimization_validation_20260922/full_cfg2/command.sh) | `--cfg-degree 2` | 1 | 225.449 | 47.140 | 1.000000 | 通过 |
| [full_dynamic_2g](../results/optimization_validation_20260922/full_dynamic_2g/command.sh) | `--resource-policy dynamic_offload --max-weight-usage 2147483648` | 1 | 409.310 | 33.644 | 1.000000 | 通过 |
| [full_sage_fusion](../results/optimization_validation_20260922/full_sage_fusion/command.sh) | `--attention-backend sage_attn --operator-fusion-backend triton` | 1 | 357.377 | 47.140 | 0.983132 | 未通过 |
| [full_sp2_reference](../results/optimization_validation_20260922/full_sp2_reference/command.sh) | `--sp-degree 2 --sp-linear-mode reference` | 1 | 285.615 | 47.140 | 1.000000 | 通过 |
| [full_tea_active_dynamic](../results/optimization_validation_20260922/full_tea_active_dynamic/command.sh) | `--resource-policy dynamic_offload --transformer-cache-mode teacache --cache-text-projections --teacache-threshold 0.02` | 1 | 348.138 | 33.644 | 0.983139 | 未通过 |
| [full_tea_dynamic](../results/optimization_validation_20260922/full_tea_dynamic/command.sh) | `--resource-policy dynamic_offload --transformer-cache-mode teacache --cache-text-projections` | 1 | 413.501 | 33.644 | 1.000000 | 通过 |
| [full_teacache](../results/optimization_validation_20260922/full_teacache/command.sh) | `--transformer-cache-mode teacache --cache-text-projections` | 1 | 408.640 | 47.140 | 1.000000 | 通过 |
| [short_auto_attention](../results/optimization_validation_20260922/short_auto_attention/command.sh) | `--attention-backend auto` | 1 | 10.665 | 17.033 | 0.983904 | 未通过 |
| [short_baseline](../results/optimization_validation_20260922/short_baseline/command.sh) | `基线` | 5 | 8.727 | 17.033 | 1.000000 | 通过 |
| [short_cache_dit](../results/optimization_validation_20260922/short_cache_dit/command.sh) | `--transformer-cache-mode cache_dit --cache-text-projections` | 1 | 8.364 | 17.033 | 0.983879 | 未通过 |
| [short_cache_dit_active](../results/optimization_validation_20260922/short_cache_dit_active/command.sh) | `--transformer-cache-mode cache_dit --cache-text-projections --cache-dit-residual-diff-threshold 0.1` | 1 | 6.520 | 17.033 | 0.981098 | 未通过 |
| [short_cache_dit_active_linear](../results/optimization_validation_20260922/short_cache_dit_active_linear/command.sh) | `--transformer-cache-mode cache_dit --cache-text-projections --cache-dit-residual-diff-threshold 0.1 --cache-residual-predictor linear` | 1 | 6.355 | 17.033 | 0.983400 | 未通过 |
| [short_cache_dit_linear](../results/optimization_validation_20260922/short_cache_dit_linear/command.sh) | `--transformer-cache-mode cache_dit --cache-text-projections --cache-residual-predictor linear` | 1 | 8.250 | 17.033 | 0.983698 | 未通过 |
| [short_cfg2](../results/optimization_validation_20260922/short_cfg2/command.sh) | `--cfg-degree 2` | 1 | 6.614 | 18.644 | 1.000000 | 通过 |
| [short_cfg2_vae2_sage](../results/optimization_validation_20260922/short_cfg2_vae2_sage/command.sh) | `--cfg-degree 2 --vae-degree 2 --attention-backend sage_attn` | 1 | 8.129 | 18.644 | 0.983904 | 未通过 |
| [short_cfg_legacy](../results/optimization_validation_20260922/short_cfg_legacy/command.sh) | `--cfg-parallel-device cuda:1` | 1 | 6.578 | 18.644 | 1.000000 | 通过 |
| [short_compile_sage](../results/optimization_validation_20260922/short_compile_sage/command.sh) | `--attention-backend sage_attn --enable-torch-compile --warmup` | 5 | 6.541 | 17.033 | 0.983674 | 未通过 |
| [short_compile_sdpa](../results/optimization_validation_20260922/short_compile_sdpa/command.sh) | `--enable-torch-compile --warmup` | 2 | 6.505 | 17.033 | 0.983813 | 未通过 |
| [short_component_offload](../results/optimization_validation_20260922/short_component_offload/command.sh) | `--resource-policy component_offload` | 1 | 19.773 | 8.925 | 1.000000 | 通过 |
| [short_dp2](../results/optimization_validation_20260922/short_dp2/command.sh) | `--dp-degree 2` | 2 | — | — | 1.000000 | 通过 |
| [short_dynamic_2g](../results/optimization_validation_20260922/short_dynamic_2g/command.sh) | `--resource-policy dynamic_offload --max-weight-usage 2147483648` | 1 | 15.632 | 3.536 | 1.000000 | 通过 |
| [short_dynamic_5g_pin](../results/optimization_validation_20260922/short_dynamic_5g_pin/command.sh) | `--resource-policy dynamic_offload --max-weight-usage 5368709120 --pin-memory` | 1 | 10.616 | 4.726 | 1.000000 | 通过 |
| [short_flash](../results/optimization_validation_20260922/short_flash/command.sh) | `--attention-backend flash_attn` | 1 | 9.759 | 17.033 | 0.983834 | 未通过 |
| [short_fullgpu_pin](../results/optimization_validation_20260922/short_fullgpu_pin/command.sh) | `--resource-policy fullgpu_pin_memory` | 1 | 9.664 | 17.033 | 1.000000 | 通过 |
| [short_fusion_adaln](../results/optimization_validation_20260922/short_fusion_adaln/command.sh) | `--operator-fusion-backend triton --operator-fusion-ops rmsnorm_adaln` | 1 | 11.435 | 17.033 | 1.000000 | 通过 |
| [short_fusion_auto](../results/optimization_validation_20260922/short_fusion_auto/command.sh) | `--operator-fusion-backend auto` | 1 | 8.696 | 17.033 | 0.983877 | 未通过 |
| [short_fusion_both](../results/optimization_validation_20260922/short_fusion_both/command.sh) | `--operator-fusion-backend triton` | 1 | 9.006 | 17.033 | 0.983877 | 未通过 |
| [short_fusion_qk](../results/optimization_validation_20260922/short_fusion_qk/command.sh) | `--operator-fusion-backend triton --operator-fusion-ops qk_rmsnorm_rope` | 1 | 9.523 | 17.033 | 0.983877 | 未通过 |
| [short_sage](../results/optimization_validation_20260922/short_sage/command.sh) | `--attention-backend sage_attn` | 5 | 8.925 | 17.033 | 0.983904 | 未通过 |
| [short_sage_fusion_tea_dynamic](../results/optimization_validation_20260922/short_sage_fusion_tea_dynamic/command.sh) | `--attention-backend sage_attn --operator-fusion-backend triton --resource-policy dynamic_offload --transformer-cache-mode teacache --cache-text-projections` | 1 | 16.917 | 3.536 | 0.983827 | 未通过 |
| [short_sp2_reference](../results/optimization_validation_20260922/short_sp2_reference/command.sh) | `--sp-degree 2` | 1 | 13.190 | 18.644 | 1.000000 | 通过 |
| [short_sp2_sharded](../results/optimization_validation_20260922/short_sp2_sharded/command.sh) | `--sp-degree 2 --sp-linear-mode sharded` | 1 | 11.328 | 18.644 | 0.983788 | 未通过 |
| [short_streaming](../results/optimization_validation_20260922/short_streaming/command.sh) | `--runtime-mode windowed_streaming` | 1 | 9.607 | 17.033 | 0.983329 | 未通过 |
| [short_tea_active](../results/optimization_validation_20260922/short_tea_active/command.sh) | `--transformer-cache-mode teacache --cache-text-projections --teacache-threshold 0.02` | 1 | 8.537 | 17.033 | 0.983851 | 未通过 |
| [short_tea_active_dynamic](../results/optimization_validation_20260922/short_tea_active_dynamic/command.sh) | `--resource-policy dynamic_offload --transformer-cache-mode teacache --cache-text-projections --teacache-threshold 0.02` | 1 | 13.900 | 3.536 | 0.983851 | 未通过 |
| [short_tea_active_linear](../results/optimization_validation_20260922/short_tea_active_linear/command.sh) | `--transformer-cache-mode teacache --cache-text-projections --teacache-threshold 0.02 --cache-residual-predictor linear` | 1 | 8.549 | 17.033 | 0.983836 | 未通过 |
| [short_tea_dynamic](../results/optimization_validation_20260922/short_tea_dynamic/command.sh) | `--resource-policy dynamic_offload --transformer-cache-mode teacache --cache-text-projections` | 1 | 15.968 | 3.536 | 1.000000 | 通过 |
| [short_teacache](../results/optimization_validation_20260922/short_teacache/command.sh) | `--transformer-cache-mode teacache --cache-text-projections` | 1 | 9.785 | 17.033 | 1.000000 | 通过 |
| [short_teacache_force](../results/optimization_validation_20260922/short_teacache_force/command.sh) | `--transformer-cache-mode teacache --cache-text-projections --transformer-cache-force-compute` | 1 | 9.698 | 17.033 | 1.000000 | 通过 |
| [short_teacache_linear](../results/optimization_validation_20260922/short_teacache_linear/command.sh) | `--transformer-cache-mode teacache --cache-text-projections --cache-residual-predictor linear` | 1 | 9.729 | 17.033 | 1.000000 | 通过 |
| [short_text_cache](../results/optimization_validation_20260922/short_text_cache/command.sh) | `--cache-text-projections` | 1 | 9.380 | 17.033 | 1.000000 | 通过 |
| [short_vae2](../results/optimization_validation_20260922/short_vae2/command.sh) | `--vae-degree 2` | 1 | 10.283 | 17.084 | 1.000000 | 通过 |
| [short_vae2_tiling](../results/optimization_validation_20260922/short_vae2_tiling/command.sh) | `--vae-degree 2 --vae-tiling --vae-tile-size 256 --vae-tile-stride 192` | 1 | 10.016 | 16.459 | 0.979798 | 未通过 |
| [short_vae_tiling](../results/optimization_validation_20260922/short_vae_tiling/command.sh) | `--vae-tiling --vae-tile-size 256 --vae-tile-stride 192` | 1 | 10.303 | 16.065 | 0.979798 | 未通过 |

## 原始证据

- [初始 43 项汇总 JSON](../results/optimization_validation_20260922/summary.json)：全部计时、实际后端、缓存命中、搬运计数、编译与融合生效报告、输出哈希。
- [实际生效与命令检查](../results/optimization_validation_20260922/effective_checks.json)。
- [环境与范围](../results/optimization_validation_20260922/environment.json)。
- [GPU 回归日志](../results/optimization_validation_20260922/gpu_tests.log)。
- [不支持组合的拒绝原因](../results/optimization_validation_20260922/constraints.json)。
- [初始 57 个产物校验结果](../results/optimization_validation_20260922/verification.log)。
- [追加 SP=2 校验结果](../results/optimization_validation_20260922/sp2_verification.log)。
- [原片抽帧对比](../results/optimization_validation_20260922/full_comparison.jpg)。
- 每个配置目录还包含 `run.log`、`result.json`、`quality.json` 和输出视频；非 DP 配置另有 `payload.json`，DP 的逐任务 JSON 报告位于 worker 日志末尾。

原始产物位于本机 `results/optimization_validation_20260922/`，不随 Git 分发。
