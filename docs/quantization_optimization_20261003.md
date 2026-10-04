# L40S 量化组合与内核调优（2026-10-03）

本轮以上一轮 `sage_fp8` Attention、BF16 Linear 为起点，分别验证累加方式、QK 量化粒度、
Attention 布局、Linear 范围以及 INT8 GEMM 分块。视频参数保持不变。

## 已落地的实现

- 增加 `--sage-fp8-accum-dtype fp32+fp32 / fp32+fp16`，默认仍为前者。
- 增加 `--sage-fp8-qk-quant-gran per_thread / per_warp`，默认仍为前者。
  非默认设置要求显式选择 `sage_fp8`，preflight 和运行报告记录实际设置。
- 增加 `--quantization-scope ffn_up`，只量化 28 个 FFN 升维 Linear；`ffn` 仍是全部 56 层。
- SM89、M≥8192、K=2048、N=8192 的 INT8 融合 GEMM 使用 BM/BN/BK=128/128/64、8 warps、3 stages。
  其他形状保留原分块。降维层保留原生 GEMM + epilogue。
- Attention benchmark 支持新的精度选项；增加参数传递、配置拒绝、非整齐长度数值及大尺寸 INT8 精确对照测试。

## 筛选结果

固定 SageAttention v2.2.0 commit `eb615cf6cf4d221338033340ee2de1c37fbdba4a`，
CUDA 12.8.1 在 `results/quant_opt_20261003/` 隔离构建，源文件未改。
当前 Torch 仍为 2.6.0+cu126，未替换运行环境。GPU 数值及完整视频验证检查了这一组合。

32,640 token、32 heads、head_dim=64，BF16 合成输入的完整 Attention，5 组×20 次 eager 中位数：

| 累加 / QK 粒度 | ms |
|---|---:|
| FP32+FP32 / per_thread | 25.694 |
| FP32+FP32 / per_warp | 25.134 |
| FP32+FP16 / per_thread | 26.657 |
| FP32+FP16 / per_warp | 25.805 |

在这组形状上，FP16 累加没有胜出。另一次同卡配对布局试验中，
NHD 为 24.672 ms，HND（计入转换）为 25.082 ms，因此保留 NHD。
不同试验的绝对值有波动，按各自配对结果比较，不跨表挑选最低值。

真实权重、合成完整输入的 DiT forward（3 组交替顺序，中位数，非视频质量验收）：

| 配置 | 秒 |
|---|---:|
| per_thread Attention + BF16 Linear | 2.0890 |
| per_thread Attention + INT8 FFN | 2.0303 |
| per_warp Attention + BF16 Linear | 2.0774 |
| per_warp Attention + INT8 FFN 升维 | 2.0365 |
| per_warp Attention + INT8 整个 FFN | 2.0155 |
| per_warp Attention + FP8 整个 FFN | 2.0382 |

上表使用调优前的 INT8 分块。随后对 INT8 升维层做精确等价调优，
7 组×30 次完整 Linear（包含激活量化）测得：BF16 5.700 ms、旧 INT8 3.739 ms、
新 INT8 3.369 ms，新 INT8 比旧实现减少约 9.9%。不同分块输出逐元素完全一致。
视频候选使用调优后的 INT8。

## 视频协议

同一 L40S GPU 2 串行运行：重测原 Attention 方案一次，候选两次。
输入 121 帧、1920×1080、seed=42、CFG=3，50 步配置、strength=0.8（实际 40 去噪步）。
缓存关闭、手工融合关闭、DiT 常驻、T5/VAE 卸载；进程首任务预热 2 步。
请求耗时排除模型加载和预热。所有结果保存在 `results/quant_opt_20261003/full121/`。
本轮重测原 Attention 输出与上一轮 SHA256 完全一致。

质量参照始终为原始 BF16 SDPA 视频，而非量化 Attention 视频；对整帧、mask、edge 和时序差分做检查。

## 视频结果与结论

| 配置 | 去噪 s | 请求 s | 平均 SSIM（对 BF16 SDPA） |
|---|---:|---:|---:|
| 原 Attention 方案，本轮重测 | 183.647 | 220.296 | 0.991730 |
| per_warp + 调优 INT8 FFN，第一次 | 181.291 | 217.869 | 0.987154 |
| 同配置，第二次 | 181.305 | 218.261 | 0.987154 |
| 组合中位数 | 181.298 | 218.065 | 0.987154 |

组合比原 Attention 方案请求耗时再减少 **1.01%**（2.231 秒），去噪再减少 **1.28%**。
对照上一轮未量化 BF16 SDPA 的 251.397 秒，组合约为 **1.153×**（耗时减少 13.26%）；
该未量化值是历史同配置数据，本轮新增收益应以上表同卡重测为准。
没有获得接近 2× 的整体加速，也不把算子约 10% 的改善描述为请求约 10% 的改善。

候选峰值 allocated 为 23.566 GiB，原方案 24.441 GiB；reserved 从 43.105 增至 43.230 GiB，
因此不能声称请求保留显存下降。56 个量化 Linear 均执行，融合 GEMM 累计调用 2352/4592，
Attention 报告 per_warp、FP32+FP32、无回退和失败。

两次组合视频 SHA256 完全一致：`de3ebd16ba5f3a54501262bad9413c6799f2e259dadf5735904af9ce9d283224`。
质量统计共 121 帧，其中 103 帧含非空 mask：最低整帧 SSIM 0.982260，
平均 mask SSIM 0.987440、最低 mask 帧 0.981050，平均 edge SSIM 0.988141，
MAE 0.900382（0–255），时序差分 MAE 0.980100。
已查看首、中、末帧和 mask SSIM 最低的第 102 帧裁剪，未见明显新增整体失真，
但相对仅量化 Attention 的数值误差明显增加。

**保留 INT8 内核调优和显式实验选项，默认推荐仍是上一轮仅量化 Attention 的方案。**
组合的额外收益只有约 1%，不宜为此默认扩大有损量化范围。
`ffn_up` 完成完整 DiT 合成输入筛选和 GPU 模型转换验证，本轮未运行其完整视频质量对照。

验证：精度选项及量化 GPU 测试通过；新增大尺寸非整齐 INT8 分块测试与原生结果逐元素一致；
`blocks`/`ffn_up` 三种 Linear 模式的模型转换与执行验证通过；回归套件通过（215 项，跳过 63 项，
另新增的大尺寸 GPU 用例在专项量化套件中通过）。CLI、server 和 benchmark 入口检查通过。

## 使用

保留原推理参数，实验配置追加：

```sh
--attention-backend sage_fp8 \
--sage-fp8-accum-dtype fp32+fp32 \
--sage-fp8-qk-quant-gran per_warp \
--transformer-quantization int8_w8a8_native --quantization-scope ffn
```

当前隔离扩展通过 `PYTHONPATH="$PWD/results/quant_opt_20261003/SageAttention"` 启用。
模型加载、量化范围、数值选项都通过 CLI 设置；默认配置未改变。

实验脚本、全部数据与依赖指纹见 `results/quant_opt_20261003/`：
`attention_screen.json`、`layout_screen.json`、`forward_screen.json`、`forward_fast_screen.json`、
`int8_tuning.json`、`tile_verify.json`、`manifest.json` 以及 `full121/*_command.json`。
