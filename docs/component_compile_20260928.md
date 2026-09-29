# T5 与 VAE 编译（2026-09-28）

在已有 DiT 编译上增加辅助组件选择：
`--compile-components text_encoder,vae_encoder,vae_decoder`，可选择任意子集，默认关闭。
该开关独立于 DiT 的 `--enable-torch-compile`；两者可以组合。

## 实现边界

- T5 编译整个 forward，tokenizer 与请求内提示词缓存保持原逻辑。首版要求 T5 常驻，
  不把 FSDP CPU offload hooks 放进完整计算图。
- VAE 编译 encoder/decoder forward；后验分布、随机采样、分块调度、阶段间搬运留在图外。
  保留原 Module 与权重，不改 state_dict 路径，单卡 VAE CPU offload 可继续使用。
  多卡 VAE 的线程复制/空间上下文路径暂不支持。
- 使用 `fullgraph=True, dynamic=False`，关闭 CUDA Graph，不静默回退。
  生命周期管理器在关闭/注册失败时恢复原 forward，诊断信息在图外统计。
- Torch 2.6 的精度转换模拟在 T5 相对位置计算上触发
  `ValueRangeAnalysis.to_dtype() got an unexpected keyword argument 'use_compute_types'`。
  仅 T5 的编译 options 关闭 `emulate_precision_casts`，使用标准 Inductor 语义；
  VAE 保留该选项。不修改安装环境或全局精度设置。

[PyTorch 官方示例](https://pytorch.org/blog/accelerating-generative-ai-3/)展示了 VAE decode 编译；
本项目选择内部 encoder/decoder 的边界以保留采样和卸载的生命周期。
详细 CLI 用法见 [CLI](cli.md#编译-t5-与-vae-组件实验性)。没有导入 `sglang` 包。

## 真实权重、小输入微基准

物理 GPU 6，A100 80GB，Torch 2.6.0+cu126，BF16、确定性配置、模式 `default`。
T5 与 VAE 均使用 `data/model` 的真实完整权重；T5 token 为固定编号序列，
VAE 输入为随机张量。两边权重均常驻 GPU。每种形状先分别执行一次，再按
A/B、B/A、A/B 各测三次，计时前后同步，统计中位数。

| 组件与输入 | eager ms | compile ms | 耗时减少 | 输出 L2 相对误差 |
|---|---:|---:|---:|---:|
| T5，128 tokens | 29.216 | 13.404 | 54.1% | 0.01562 |
| T5，64 tokens | 30.006 | 13.256 | 55.8% | 0.01681 |
| VAE 编码，9×64×64 | 14.315 | 6.866 | 52.0% | 0.00142 |
| VAE 编码，17×64×64 | 14.354 | 7.429 | 48.2% | 0.00198 |
| VAE 解码，latent 2×2×2 | 18.625 | 6.072 | 67.4% | 0.00488 |
| VAE 解码，latent 3×2×2 | 19.139 | 6.685 | 65.1% | 0.00598 |

首次调用包含编译：T5 两种形状约 20.29 / 13.80 秒；VAE encoder 约 15.47 / 15.76 秒；
VAE decoder 约 17.30 / 16.66 秒。磁盘已有其他实验的编译缓存，不代表空缓存冷启动。
每组编译后首个 eager 样本明显较慢，全部原始样本保留于 JSON，三次测量只作为筛选。
同机其他 GPU 有任务，不能将小输入微基准外推为 1080p 端到端加速比。
输出有数值差异，编译多个组件后的组合质量必须另外验证。

本地产物：`results/component_compile_20260928/benchmark.py/.json/.log`；
原始 T5 失败日志单独保留为 `t5_precision_cast_failure.log`。

## 验证

- 全量 CPU 回归 131 项：104 通过、27 条件跳过。
- GPU 专项 4 项通过：真实 T5 的完整图捕获与形状复用、配置边界、失败回滚和关闭恢复，
  以及 CUDA Inductor 下 VAE 因果卷积在 CPU/GPU 权重往返后的输出验证。
- 上述真实完整 checkpoint 微基准覆盖 T5、VAE encoder/decoder 的两个输入形状。
- Python 语法检查和 `git diff --check` 通过。

## 组合请求复现

本地 `integration.py` 在同一会话依次运行辅助组件 eager 基线、辅助组件编译首次请求、
辅助组件编译重复请求。三者 DiT 均保持完整编译；基线临时恢复辅助组件原始 forward，
完成后恢复编译入口。T5 常驻，VAE 在各阶段之间 CPU offload。
输入是原片前 17 帧缩放到 320×192 的 smoke 素材，配置 5 步、strength 0.8，
实际去噪 4 步、窗口 17/重叠 9、seed 42，不能用于声明原片 50 步质量或加速比。

```bash
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
TORCHINDUCTOR_COMPILE_THREADS=4 MGERASE_TORCH_COMPILE_MODE=default \
python results/component_compile_20260928/integration.py \
  --model-path data/model --task-file results/component_compile_20260928/tasks.json \
  --prompt 'There is a rooftop terrace overlooking the city at sunset.' \
  --seed 42 --num-inference-steps 5 --strength 0.8 --infer-len 17 --overlap 9 \
  --no-dit-layerwise-offload --no-dit-cpu-offload --no-text-encoder-cpu-offload \
  --vae-cpu-offload --attention-backend sdpa --operator-fusion-backend disabled \
  --transformer-cache-mode off --no-cache-text-projections \
  --enable-torch-compile --torch-compile-scope transformer \
  --compile-components text_encoder,vae_encoder,vae_decoder
```

正常使用时直接执行 `python -m entrypoints.cli.erase_eraserdit`，替换自己的输入输出路径；
无需实验脚本或临时恢复 forward。普通 CLI 和服务启动均支持同一组件参数。

首轮组合 smoke：辅助 eager 基线 44.430 秒（含 DiT 首次编译），辅助编译首次请求
41.137 秒（DiT 已预热），重复请求 1.084 秒。两次首次请求的编译对象不同，
不能据此算出端到端加速比。辅助组件实际执行累计 T5 4 次、VAE encoder/decoder 各 2 次。
同形状重复请求没有增加首次调用记录，VAE CPU offload 保持开启。

首轮相对辅助 eager 基线，17 帧 RGB SSIM **0.975616**、MSE **10.504066**、
MAE **1.737744**，**未达到此前 0.98 门槛**。完整输出和编译运行成功不等于画质验收通过。
三组件一起启用暂不纳入推荐配置。下面追加独立拆分。


## T5/VAE 分组消融

另一个同配置会话先执行辅助 eager 基线，再复跑 eager、仅编译 T5、仅编译 VAE 两部分、
三者同时编译。DiT 始终为完整编译。参考为该会话的辅助 eager 复跑输出。

| 辅助编译目标 | 请求 s | 整段 SSIM | 达到 0.98 |
|---|---:|---:|---|
| 无，DiT 首次编译 | 24.747 | 1.000000 | 是 |
| 无，DiT 已预热 | 1.117 | 1.000000 | 是 |
| 仅 T5，含其首次编译 | 9.583 | 0.976205 | 否 |
| VAE encoder+decoder，含其首次编译 | 10.854 | 0.975315 | 否 |
| T5+VAE，全已预热 | 1.064 | 0.975616 | 否 |

两个 eager 输出逐像素一致；全编译与前轮得到相同 SSIM。
T5 与 VAE 编解码组合各自都足以使此样本低于门槛，不能把差异只归因于 T5。
最后一行与第二行的单次时间差不足以证明稳定端到端收益；这个短片仅用于组合兼容和数值检查。
产物为 `ablation.py/.json/.log`、`quality_{target}.json`。


## 单独 VAE decoder

使用正式 CLI，在相同 smoke 输入、seed、步数与完整 DiT 编译下，只增加
`--compile-components vae_decoder`，保持 VAE CPU offload。相对辅助 eager 基线：

- 17 帧整段 RGB SSIM **0.983054 ≥ 0.98**。
- MSE **2.682435**，MAE **1.130091**，最低帧 SSIM **0.980188**。
- 首次请求 **31.961 秒**，包含本进程 DiT 与 decoder 的首次编译，未作为稳态时间。
- 所有帧完整解码，尺寸和帧率相同；没有做人工动态播放验收。

因此可优先试验只编译 decoder；T5 和 VAE encoder+decoder 组合暂不推荐。
这里相对的是**已编译 DiT、辅助组件 eager** 的短片基线，不是未编译原片。
本节只记录短片结果；后续 [1080p、50 步原片验证](decoder_full_validation_20260928.md)
确认仅 decoder 与完整 DiT 编译的组合达到整段 0.98，但重复请求相对仅 DiT 编译
没有体现有效端到端收益。所有组件仍默认关闭。
