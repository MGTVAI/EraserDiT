# SGLang 内存管理迁移验证（2026-09-24）

实现和行为变化见 [内存管理](sglang_memory.md)。不依赖 SGLang 包；已验证加载 CLI 与
pipeline 后 `sys.modules` 中没有 `sglang`。迁移的逐层管理器与固定版本源码除三个依赖导入
及来源头部外逐行相同。T5 使用迁移的 `shard_model`，模型 block 条件由本项目提供。

环境：A100 80GB，物理 GPU 2，Python 3.10，Torch 2.6.0+cu126，BF16、SDPA、确定性配置。
模型为 `data/model`，无 compile、量化或残差缓存。

## 回归

- CPU 全量：101 项，86 通过、15 按 GPU/多卡/INT8 条件跳过。
- GPU 专项及缓存：36 项全部通过。
- `git diff --check`、Python 编译检查通过。
- 覆盖循环预取比例/层数、非连续布局、重复 forward/窗口、T5 共享 embedding、两个会话
  共享进程组、阶段异常、部分 H2D 失败、T5 forward 失败后 reshard 及下一请求。
- meta 加载检查分片、共享参数、dtype 和缺失/多余/错误形状的权重；旧参数和不支持组合拒绝。

异常测试发现 Torch 2.6 的单模块 FSDP forward 异常会遗漏 post-hook，根参数仍保持 GPU
驻留；阶段清理补完该 hook 后再 reshard。该适配依赖 Torch 内部接口，升级需复验。

## 真实模型重复请求

视频为 `results/dynamic_offload_smoke/video_33.mp4` 和对应 mask：320×192、33 帧。
seed 42，5 个配置步，strength=0.8，窗口长度 25、重叠 9，每请求两个窗口。
全驻留与逐层卸载分别新建一个会话，各连续请求五次，无显式预热。
下表耗时不含加载，显存为五次中的最大值。

| 配置 | 请求中位数 s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|
| 全驻留 | 2.02 | 15.083 | 16.160 |
| SGLang 逐层 + T5 FSDP + VAE CPU offload | 6.44 | 2.828 | 4.717 |

两组不是交替测量，主机和其他 GPU 有作业，不能据此宣称稳定速度比或外推原片。
这份短片上卸载明显节省显存，同时增加请求耗时。首请求 allocated 2.828 GiB，
后续请求约 2.702 GiB；上表包含首请求。

10 个输出完整解码后的 RGB SHA256 全部相同：
`95e1d84d5cbb15347f5e6163cd210c117f6b5097334285850dcc6b95b755c25a`。
五次卸载请求结束均没有受管理的 GPU block；组件搬运记录有界。

另测 DiT 整组件卸载 + T5 FSDP + VAE CPU offload，单次 10.84 s，allocated 3.661 GiB，
reserved 5.277 GiB，RGB SHA256 同上。

该卸载会话初始化 CPU 历史峰值约 24.22 GiB；注册后 allocated 约 0.206 GiB，初始化
allocated 峰值约 0.698 GiB。首次预取及 FSDP 注册有 GPU 分配，不能沿用旧实现的零分配声明。

## 完整原片

首次原尺寸请求在视频预处理期间 OOM，当时另一进程占用约 54.64 GiB。
这次失败不能代表独占 A100 的能力，原始日志保留为 `full_oom.log`。
空闲后重跑成功，使用原始 1920×1080、145 帧、24000/1001 FPS 视频及 mask；
seed 42、50 个配置步、strength=0.8，两个窗口各实际 40 步，预取参数 0（一层）。

| 配置 | 请求 s | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|
| 迁移前自定义逐层后端，9 月 23 日历史单次 | 420.29 | 33.64 | 51.19 |
| SGLang 源码迁移，本轮单次 | 437.46 | 34.88 | 56.50 |

本轮纯推理 411.85 s，模型加载 44.52 s（不计入请求时间）。
与历史值相比，请求约增加 4.1%、allocated 约增加 3.7%、reserved 约增加 10.4%。
不是同条件五次交替测试，不能确定小幅时差的归因，也没有观察到整段视频的显存收益。
原生方案整体搬运 VAE、DiT 非 block 权重常驻且初始化预取；不再采用旧方案的阶段缓存清理。
这些行为解释了内存策略的差异，但尚未做逐项消融。

完整解码 RGB SHA256 与迁移前原片一致：
`eb849312b1cab539f73135c2e281af34ac34a2c5278ebcd8e9ce5a49169a5951`。
帧数、尺寸、帧率通过；请求结束 `resident_bytes=0`、`live_layers=0`，
这是 DiT 受管 block 的释放结果，不代表模型其余权重全部离开 GPU。

本轮初始化 CPU 历史峰值约 24.19 GiB；组件阶段观测到请求期历史峰值 53.81 GiB，
比旧记录 48.30 GiB 更高。CPU allocator/pinned 缓存和视频处理均可能贡献，未做归因分析。
未做全片人工播放验收。

## 原生 VAE tiling 冒烟与质量

同一 320×192、33 帧双窗口短片，开启 `--vae-tiling --vae-tile-size 256
--vae-tile-stride 224`，其余与重复请求一致。请求 9.38 s，allocated 2.825 GiB、
reserved 4.664 GiB，完整输出可解码。与全驻留、不分块参考比较：

- RGB SSIM 0.845385、MSE 550.890784、MAE 9.662048。
- **未通过非缓存、非量化画质门槛**，不能将这组小块配置作为达标推荐。
- tiling 默认关闭；默认路径的原片逐像素一致不包含这项近似。
- 默认 tile=512/stride=448 的完整原片质量未在本轮验证。

## 本地产物

`results/sglang_memory_20260924/` 为本机忽略目录：

- `final_cpu_suite.log`、`gpu_tests.log`：最终回归。
- `run_repeated.py`、`resident_command.json`、`offload_command.json`：重复请求复现。
- `resident_result.json`、`offload_result.json`、`summary.json`：结果和 RGB 哈希。
- `component_result.json`：整组件模式结果。
- `full_command.sh`、`full.log`：完整原片复跑命令和日志。

原生 tiling 的失败指标保留在 `native_tiling_quality.json`，质量脚本返回 2 表示未通过门槛。
