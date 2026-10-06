# SP2 / SP4 原生归约 RMSNorm 融合

新增显式算子 `qk_rmsnorm_rope_native`、`rmsnorm_adaln_native`，在 aligned GEMM、
direct 打包和通信重叠之上减少 RMSNorm 中间张量及逐元素 kernel。默认算子集合不变。

## 实现与边界

Triton 计算 FP32 平方，输出与 eager 相同形状、连续布局和 dtype 的张量；均值、epsilon
加法和 rsqrt 继续调用 PyTorch。后续归一化、权重乘法、RoPE 或 AdaLN 合并，显式保留
每处 BF16 舍入与 FP32 非 FMA 运算。因此不采用此前 `*_fast` 的近似归约。

公开入口检查 CUDA BF16、宽度 2048、布局、epsilon、仿射参数和原始 diffusers RMSNorm
实现；仅用于无梯度推理。`auto` 不满足契约时执行原路径，强制 `triton` 报错。
Q/K 需要 FP32 同形 RoPE，AdaLN 限 batch 1、非仿射 norm。输入不原地修改，工作区随调用释放。
NCCL 仍限制常驻 CFG/Ulysses SP1/2/4，不允许 TP、Ring、FSDP 或近似融合算子。
完整视频验收范围为本机 L40S、PyTorch 2.6.0 / CUDA 12.6、BF16 SDPA、纯 SP2/4。

## 复现

```bash
# 在已配置的常驻 NCCL SP2/SP4 CLI 或服务参数中选择：
--operator-fusion-backend triton \
--operator-fusion-ops qk_rmsnorm_rope_native,rmsnorm_adaln_native

# 五组交替 forward；四卡改为 0,1,2,3 和 --sp 4。
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
uv run --no-project python -m entrypoints.cli.benchmark_dit \
  --run-dir outputs/sp2_native_forward --cfg 1 --sp 2 \
  --variants aligned_heads4_output,aligned_heads4_output_native_rms \
  --latent-frames 16,5 --repeats 5

# 完整请求；四卡使用 sp4_native_rms 和 --devices 0,1,2,3。
MGERASE_ULYSSES_HEAD_CHUNKS=4 MGERASE_ULYSSES_OUTPUT_OVERLAP=1 \
uv run --no-project python -m entrypoints.cli.benchmark_l40s \
  --run-dir outputs/sp2_native_video --profile sp2_native_rms --devices 0,1 --repeats 5
```

新 profile 显式选择 aligned 和原生归约融合；直接调用 CLI 的 GEMM 与通信参数见
[aligned 记录](sp_aligned_20261005.md)。不把四块通信设为所有序列的默认值。

## 验证记录

真实权重、固定合成输入，CFG1 的正负两个分支及输出汇聚；两侧均为 aligned、direct、
四块输入/输出重叠，只改变 RMSNorm 融合。每项五组正反顺序交替，中位数单位为秒。
全部候选输出与基线逐元素一致，运行时源码指纹与当前一致。

| 拓扑 | Token | 原融合 | 原生归约融合 | 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| SP2 | 32640 | 2.64447 | 2.39501 | 9.43% |
| SP2 | 10200 | 0.51632 | 0.48855 | 5.38% |
| SP4 | 32640 | 1.48988 | 1.41413 | 5.08% |
| SP4 | 10200 | 0.47256 | 0.46995 | 0.55% |

SP4 短序列差距接近波动，不能据此声称稳定收益。这是相对上一轮 aligned 的增量优化，
不含加载、VAE 或视频读写，不是完整请求加速比。两轮均确认新算子实际执行且无回退。
与记录的源码指纹相比仅调整 profile 和 NCCL 配置测试，推理源码未变。

记录位于 `outputs/sp_native_rms_20261005/`。`sp2_screen` 分别启用 Q/K、AdaLN 及两者，
所有 forward 逐元素一致，两者组合收益最大。单算子微基准 `kernel_screen.json` 只用于筛选，
不代表完整模型加速比。

`tests.test_native_rms_fusion` 检查原生数值逐元素一致、空序列与尾部、小/大/零输入、
实际 SP 本地长度、非默认 CUDA stream、输入不变、梯度回退、自定义 forward 回退和算子互斥。
`tests.test_l40s_profiles` 检查新 profile 生成的配置及近似算子仍被 NCCL 拒绝。
融合相关测试首轮 21 项通过，新增 profile 检查后该模块 5 项通过；集成回归 39 项通过、
22 项显式启用的进程/GPU 测试跳过。本轮真实多卡执行另由上述 forward 和完整视频验证。
记录：`tests.log`、`profile_tests.log`、`integration_tests_final.log`。旧配置断言匹配的
报错文案已随新增合法算子更新，拒绝 gated residual 的行为保持。

## 完整素材验收

使用与 BF16 参考相同的两份完整素材、mask、prompt、seed 42 和采样参数：横屏 145 帧、
竖屏 120 帧，每窗 40 实际去噪步，不缩短尾窗，不启用残差或文本投影缓存。
参考目录为 `results/l40s_completion_20261004/cfg2_sp2_reference_quality/`。

SP2、SP4 共四个输出均与参考逐字节一致。每种拓扑各三个窗口，分别共 6 / 12 份 rank
报告均执行新融合且无回退。
请求耗时排除加载和预热，单位为秒：

| 拓扑 | 素材 | 请求 | DiT 去噪 | 最高单卡任务峰值 GiB |
| --- | --- | ---: | ---: | ---: |
| SP2 | 横屏 145 帧 | 259.225 | 202.952 | 21.527 |
| SP2 | 竖屏 120 帧 | 138.545 | 101.569 | 21.527 |
| SP4 | 横屏 145 帧 | 177.770 | 121.280 | 21.613 |
| SP4 | 竖屏 120 帧 | 95.477 | 60.576 | 21.613 |

显存覆盖进程树加载、预热及全部请求，是同卡任务进程合计峰值，非每请求独立峰值。
每素材仅一次，不能据此声称稳定端到端加速比；其他 GPU 有现存服务，整机非独占。
目标卡无外部进程、显存采样无错误，冻结推理源码与当前一致。

`audit.py` 核对实际生效的 CLI 参数（重复参数取末次值）、输入指纹、prompt/seed、
输出 SHA256、媒体元数据、逐窗口 rank 融合调用、显存与源码指纹。
视频和详细报告位于同记录目录的 `sp2_video/`、`sp4_video/`，联合审计为
`audit_sp2_video_sp4_video.json`。两种拓扑均通过 24 GiB 任务显存验收。

## 后续候选

本地 SGLang 的 `runtime/layers/layernorm.py` 将 residual、norm 和调制连续融合，
`runtime/layers/usp.py::ulysses_attention_pipelined` 则按 head 分块重叠收发与计算。
本项目已具备后一类流水线；本轮采用保留原生均值的方式减少 norm 开销。
下一步可筛选 cross-attention 残差加法与 FP32 平方输出合并，以及按序列长度选择两块/四块通信。
前者需要保持残差 BF16 舍入，后者必须确保所有 rank 选择一致。二者尚未实现或证明收益。
