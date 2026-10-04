# 测量与验证

验证使用正式 [CLI](cli.md)、[服务 API](service_api.md) 和 [回归测试](../tests/README.md)。

## 换机验收

1. 按 [部署说明](setup.md) 检查驱动、依赖、FFmpeg 和入口 `--help`。
2. 运行 CPU 回归，确认没有失败；GPU 测试跳过不等于 GPU 验证通过。
3. 使用 `data/model/` 模型和默认素材 `data/113000356.mp4`、`data/113000356_mask.mp4` 完成一次 SDPA 推理，播放结果并检查擦除区域。
4. 按实际需要运行服务、上传素材、查询进度、下载产物；多卡和 INT8 单独启用对应测试。

CPU 回归从仓库根目录执行（隐藏 GPU，避免自动运行 CUDA 用例）：

```bash
CUDA_VISIBLE_DEVICES='' ERASERDIT_TEST_TWO_GPU=0 ERASERDIT_TEST_INT8=0 \
  OMP_NUM_THREADS=1 uv run --no-project python -m unittest discover -s tests -v
```

`httpx` 已包含在统一依赖清单中，无需单独安装测试依赖。

用 FFprobe 检查输入与输出元数据，对每段视频分别执行：

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames \
  -of json outputs/result.mp4
```

尺寸、帧数、帧率需符合预期；再完整解码排查损坏：

```bash
ffmpeg -v error -xerror -i outputs/result.mp4 -f null -
```

解码成功不代表目标已擦除，仍需查看 mask 内、边缘和窗口衔接处的画面。

## 性能对照

L40S 收敛工具直接调用正式 CLI，记录源码、依赖、输入指纹、逐请求计时与从加载开始的 NVML
进程树采样。输出目录必须不存在；不自动抢占已被其他计算进程使用的 GPU。

```bash
uv run --no-project python -m entrypoints.cli.benchmark_l40s \
  --run-dir results/l40s_bf16_new --profile bf16 --devices 0 --repeats 5
# 双卡 / 四卡分别选择 cfg2 / cfg2_sp2，并提供对应数量的设备。
# 快速单卡选择 fast_offload；Sage FP8 扩展需事先安装或用 --sage-source 指定。
uv run --no-project python -m entrypoints.cli.compare_videos \
  --reference results/l40s_bf16_new/output_0.mp4 \
  --candidate outputs/candidate.mp4 --mask data/113000356_mask.mp4 \
  --output results/candidate_quality.json --review-dir results/candidate_review
```

`sampled_memory_pass` 只表示已观测任务进程峰值通过，不代表质量通过，也不能排除采样间隔内更短的峰值。
保留 `memory.jsonl`、`memory.json`、`report.json`、`manifest.json` 和质量逐帧报告。
多进程 RSS 求和包含共享页，不能当作系统独占内存；报告同时保存其他 GPU 的启动占用情况。
性能统计之外还需检查输出像素、实际执行配置以及 `source_changes_during_run.json`。

固定代码提交、完整权重版本、输入、prompt、seed、采样步数、窗口及确定性设置，
保存 GPU 型号、驱动、依赖版本和其他进程占用情况。每次仅改变一个待测配置。
配置及组合限制见 [性能说明](performance.md)。

在 CLI 的 `tasks.json` 中重复相同任务至少五次，每项使用独立 `id` 和 `output`，
以同一常驻 session 测量；需要预热时为所有配置使用相同 `--warmup --warmup-steps`。
将 CLI 标准输出与日志保存到各自输出目录：

```bash
MODEL_DIR="$PWD/data/model"
mkdir -p outputs/baseline
CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path "$MODEL_DIR" --task-file tasks.json \
  --attention-backend sdpa --warmup --warmup-steps 1 \
  > outputs/baseline/run.txt 2> outputs/baseline/run.log
```

末尾 JSON 包含逐任务计时、预热、显存及实际执行配置；标准输出可能包含其他日志，
不要把整个文件直接当作 JSON。分别保存基线及候选输出；比较中位数、范围和离散程度。
加载、量化转换、编译及预热单列，不混入稳态耗时；allocated 和 reserved 分开报告。
按 A/B、B/A 顺序交替测量以减少时间漂移，注明这是分批请求还是两个常驻会话间的交错测量。
比较不同版本时使用明确的代码快照及各自环境，从对应仓库根目录执行。

## 质量与恢复

逐帧解码 RGB，按 [质量指标与各轮验收目标](performance.md#acceptance) 比较基线与候选；
采用外部工具时核实像素范围、SSIM 窗口、边界处理及聚合方式，记录工具版本和参数。
FFmpeg 默认 SSIM 或 Y 通道 PSNR 不等价于该 RGB 标准。
缓存／量化还需完整播放检查纹理、颜色和时间闪烁，不能仅靠平均分数判定。
抽帧使用系统 FFmpeg：

```bash
mkdir -p outputs/frames
ffmpeg -i outputs/result.mp4 -vf fps=1 outputs/frames/frame-%04d.png
```

批量任务加入不同 seed 后再恢复原 seed，检查同一会话无状态串扰。
卸载配置检查报告中的 `memory_runtime`，缓存检查 `transformer_cache_history`，
并行检查 `parallel_history` / `cfg_parallel`，量化检查 `quantization`，确认实际执行了请求的配置。
异常回滚与清理由现有卸载、缓存、执行控制测试覆盖；真实服务取消后再提交一次请求检查恢复。
这些步骤不等同于所有真实模型故障注入场景已通过。


## 历史验证：移除旧模型（2026-09-22）

以下是当时的单次验证，不代表当前所有配置的验收结果。最新实验见[记录索引](README.md#实验与验证记录)。

移除 MGErase LTX 0.9.5 支持后，使用已有 EraserDiT 环境、GPU 2（A100 80GB）、
`data/model/` 和原始示例视频完成两窗口推理：1920×1080、145 帧、24000/1001 fps，
seed 42、50 步、strength 0.8（每窗口实际去噪 40 步）、SDPA、fullgpu、缓存关闭。
请求耗时 400.37 秒，纯推理 379.86 秒，峰值 allocated 显存 47.14 GiB。

全部输出帧解码成功，尺寸、帧数和帧率与输入一致；整段 RGB 解码 SHA256 与
此前同配置验证产物完全相同。CPU 回归 98 项通过（其中 20 项按 GPU 条件跳过），
补充装配与窗口规划回归 10 项通过。此次未额外验证真实多卡或 INT8 推理。
本地日志、配置、视频及验证报告保存在 `results/remove_ltx095_validation/`。

本机非量化优化测试的命令、覆盖范围与结果见[2026-09-22 优化验证](optimization_validation_20260922.md)。

### 静态 FP8 激活范围审计

`entrypoints.cli.audit_static_fp8` 接受 `--audit-report PATH` 和正常 CLI 参数，
逐层统计真正送入量化器的激活（包含融合 GELU 后的值）：有限值最大绝对值、
超过静态范围 ±56 的数量与非有限值数量。例如在正常 FP8 命令的模块名处改为
`entrypoints.cli.audit_static_fp8 --audit-report results/audit.json`。
使用 `fp8_w8a8_static`、关闭 compile，并选择已构建的 Sage FP8 路径。
此诊断会同步每个 Linear，不能用其耗时作性能结果；它不保存大激活、不修改输入。
审计报告必须同时检查 `observed`、`failed` 与非有限值，不能把未执行的报告计为通过。

### DP 进程树测量

`benchmark_l40s` 支持 `--dp-degree 2` 或 `4`；`--devices` 须包含
`profile 卡数 × dp-degree` 张互不重复的物理卡，`--repeats` 是总请求数。
工具读取 dispatcher 下每个 worker 的正式 CLI 报告，分别记录加载时间、
冷批次吞吐和按最长 worker 请求时间之和估计的热态服务率；后者不包含调度开销，
不能当作实际持续到达负载测试。NVML 仍覆盖整个进程树和全部物理卡。

### 精度组合筛选

`benchmark_l40s --profile` 另外提供 `sage_bf16`（Sage FP8 attention + BF16 FFN）、
`sage_fp8_dynamic`（动态 FP8 FFN）及 `sdpa_fast_fusion`（SDPA + BF16 FFN）。
三者均启用快速 QK/AdaLN 与 gated-residual 融合、组件卸载；这些是实验配置名，非质量通过标记。
使用 `--cases` 固定原始素材、prompt 与 seed，并与 `bf16` 参考逐帧比较。
原 `fast*` 配置仍代表静态 FP8 组合；该组合未通过最新完整素材验收，不作为通用推荐。
