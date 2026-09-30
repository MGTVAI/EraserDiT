# 双卡 VAE 空间并行优化与验证

当前 EraserDiT LTX VAE 使用 `spatial_halo`：沿高度分片、逐层交换相邻边界，保留完整时间上下文。
本记录验证真实权重的独立 VAE 阶段，不包含 DiT、T5、视频 IO，也不是 SGLang Wan 的结果。

## 当前结果

双 A100 80GB / NVLink，BF16，每个模式预热一次，三轮按单/双、双/单、单/双交替测量。
编码输入为真实首窗口 `[1,3,121,1088,1920]`，解码输入为此前真实 DiT 输出 `[1,128,16,34,60]`。

| 阶段 | 单卡中位数（范围） | 双卡中位数（范围） | 加速比 | 单卡峰值 allocated | 双卡主卡 / 副卡峰值 allocated |
| --- | --- | --- | --- | --- | --- |
| Encode | 5.296 s（5.275–5.363） | 3.180 s（3.084–3.289） | 1.67× | 30.300 GiB | 19.699 / 17.251 GiB |
| Decode | 3.068 s（3.061–3.077） | 1.785 s（1.667–1.789） | 1.72× | 26.141 GiB | 16.245 / 14.906 GiB |

主卡峰值分别降低 **35.0% / 37.9%**，耗时分别降低 **40.0% / 41.8%**。
峰值是 PyTorch 本进程的 allocated，不是 nvidia-smi 总占用或 reserved；两卡各自峰值不相加。
输入和完整 VAE 权重仍驻留主卡，输出汇聚回主卡，因此每卡峰值不会严格减半。

测量包含模型副本创建、线程创建、边界交换、输出汇聚；副本 setup 中位数约 0.083 / 0.075 s。
输入加载、上传、输出比较不计时。每次调用前同步、回收空闲缓存并重置峰值；参考输出放在 CPU。
这是预热内核后清空空闲缓存的测量口径，未将模型副本跨调用常驻。

最终测量使用物理 GPU 6、7；共享机器上仍有其他进程驻留，JSON 保留每次调用前后的进程快照。
此前 GPU 3、7 的 v2 测量中途有外部任务占用主卡约 46 GiB，导致显著变慢和 OOM，未用于性能结论。

## 实现

- 卷积仅计算局部分片及必要边界，移除补零恢复完整高度的参考计算。
- 边界和本地特征直接写入同一缓冲；全局边缘显式补零，卷积直接生成局部有效高度，避免裁剪复制。
- RMSNorm 和下采样残差在本卡分片上计算，不再恢复全局张量或修改归一化方法。
- CUDA event 处理生产、跨卡复制和源存储复用的依赖；保留两次短线程屏障，移除每层主机等待 GPU 完成。
- 高度核为 1 的卷积不交换边界。输出预分配后逐片写入，避免先搬完整分片再 `cat`。
- 正常结束或异常时恢复卷积方法和 padding，释放副本、线程与交换引用。目录结构不变。

单卡路径及显式 `--vae-tiling` 路径保持原有算法。启用局部空间并行使用 `--vae-degree 2 --no-vae-tiling`；
按现有组合约束关闭组件卸载和 VAE 编译，例如同时传入
`--no-dit-layerwise-offload --no-dit-cpu-offload --no-text-encoder-cpu-offload --no-vae-cpu-offload`。
模型副本不跨阶段缓存，以免长期占用后续 DiT 阶段需要的显存。

## 数值与质量

空间感受野完整，但局部卷积形状可能改变 cuDNN 内核及 BF16 舍入，**不承诺与单卡逐元素一致**。
固定输入的三轮统计一致，单卡重复执行为零误差。

| 指标 | Encode（后验参数） | Decode |
| --- | --- | --- |
| 最大绝对误差 | 0.125 | 0.0283203 |
| 平均绝对误差 | 0.00176725 | 0.000338663 |
| 相对 RMSE | 0.09415% | 0.17899% |
| 接缝两行 RMSE / 全局参考 RMS | 0.10265% | 0.16777% |

另行拆开编码后验检查，均值的最大/平均绝对误差为 0.0078125 / 0.000625882，相对 RMSE 为 **0.56463%**；
logvar 的相对 RMSE 为 0.09397%。后验参数整体的相对误差受 logvar 尺度影响，不能替代均值误差。

解码 RGB SSIM：平均 **0.999774**、最低帧 **0.999687**、接缝上下各 16 行 **0.999798**。
使用全部 121 帧、原始解码值裁剪到 [-1,1] 后映射至 [0,255]，不经过视频压缩；Gaussian 11×11，sigma 1.5，reflect 边界。
基准要求整体相对 RMSE ≤1%、接缝相对误差 ≤2%、逐元素 `atol=0.15, rtol=0.02`，三项 SSIM 均 ≥0.99。
这些是独立 VAE 的检查。后续[端到端验证](vae_e2e_validation_20260930.md)中，
优化编码后经过 DiT 的整片 SSIM 为 0.98313，未达到 0.99，不能沿用此处独立解码器的通过结论。

小尺寸随机图像编码出的均值是较敏感的解码输入：单/双卡 BF16 相互有约 4%–5% 的相对 RMSE。
FP32 对照探测显示单卡 BF16 本身约 4.8%–5.0%，双卡相近；不能将单卡 BF16 当作精确真值。
真实权重回归对此要求双卡相对 FP32 的 RMSE 不超过单卡的 1.1 倍，最大误差及接缝 RMSE 不超过 1.25 倍。
普通编码保持 1% 相对 RMSE 检查，原生 tiled 编解码仍要求逐元素一致。

## 回归与候选比较

- `VAEHaloTests`：FP32 局部/完整卷积对齐，覆盖因果与非因果时间 padding、高度核 1/3、不等长及单行分片、非默认 stream、源存储复用。
  两张 GPU 上另用四个逻辑 rank 检查同时接收上下两个邻居；不代表实测四张 GPU 的性能。
- 真实权重：原生空间 tiling、局部编解码、FP32 精度对照、失败后的方法/padding 恢复及重试通过。
- CPU mesh、模块兼容、内存生命周期及架构回归共 23 项，22 通过、1 条件跳过。
- 完整尺寸基准通过，所有实际参与卡的卷积调用次数分别为编码 46、解码 45，未回退单卡。
- 中间局部卷积版本仍做输出裁剪复制，双卡编码/解码中位数为 3.189 / 1.974 s；最终版为 3.180 / 1.785 s。
  编码差异落在波动范围内，解码有进一步收益；峰值增加约 0.015 GiB，误差统计相同。

优化前的 `spatial_halo_reference` 为保持参考形状，补回完整高度再计算。
历史三轮结果：编码单/双卡 5.246 / 5.932 s、主卡峰值 30.300 / 31.976 GiB；
解码 2.969 / 3.283 s、26.141 / 27.581 GiB。它能逐元素对齐，但无法降低主要计算和临时激活。
历史记录位于 `results/vae_spatial_full_20260929.json`；不可将不同时间的测量直接作为受控配对。

逐层 CUDA event 采样中，每卡卷积累计约为编码 1.77–1.78 s、解码 0.79–0.85 s；
RMSNorm 约为编码 0.59–0.61 s、解码 0.40–0.45 s。
这是额外插桩的单次归因采样，不用于替换上面的三轮基准，也不能将两卡时间直接相加。
当前仍有归一化和卷积内核优化空间；本轮选择的是已验证候选，未宣称所有硬件与形状下全局最优。
采样与后验分项记录为 `results/vae_local_profile.json`，脚本为 `results/profile_vae_local_hooks.py`。

## 复现

测试默认跳过，需显式开启；默认输入形状 `17,320,192`，随机输入固定 seed 42。
指定完整形状、真实输入并重复三次：

```bash
CUDA_VISIBLE_DEVICES=6,7 ERASERDIT_TEST_VAE_BENCHMARK=1 \
ERASERDIT_TEST_MODEL=data/model ERASERDIT_VAE_REPEATS=3 \
ERASERDIT_VAE_SHAPE=121,1088,1920 \
ERASERDIT_VAE_ENCODE_INPUT=results/vae_spatial_encoder_input_20260929.pt \
ERASERDIT_VAE_DECODE_INPUT=results/decoder_full_20260928/decoder_input_16.pt \
ERASERDIT_VAE_REPORT=/tmp/vae_spatial.json \
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
uv run --no-project python -m unittest tests.test_mesh_gpu.VAESpatialBenchmarkTests -v
```

fixture 是本机忽略产物；删除两个 `*_INPUT` 变量可用固定随机输入独立复现性能，但不是本轮真实素材质量复现。
编码 fixture 是 CPU tensor，解码 fixture 是 `(latent, timestep)`；准备脚本为 `results/prepare_vae_spatial_20260929.py`。
GPU 回归开启 `ERASERDIT_TEST_TWO_GPU=1`、`ERASERDIT_TEST_MODEL=data/model`，运行
`tests.test_mesh_gpu.VAEHaloTests` 和 `tests.test_mesh_gpu.GPUParallelTests.test_native_vae_tiling_matches_two_devices`。

本机 uv 默认解释器缺少 diffusers，实际命令增加
`--python /mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python`，使用现有 Python 3.10 / Torch 2.6 环境。
最终基准 JSON / 日志为 `results/vae_local_full_v4.*`；中间版为 `vae_local_full_v3.*`；
GPU 回归为 `vae_no_crop_regression.log`，CPU 回归为 `vae_local_cpu.log`。
JSON 记录权重路径、输入形状、数值门槛、设备、峰值、耗时和源码 SHA256。
