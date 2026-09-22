# 测量与验证

验证使用正式 [CLI](cli.md)、[服务 API](service_api.md) 和 [回归测试](../tests/README.md)。

## 换机验收

1. 按 [部署说明](setup.md) 检查驱动、依赖、FFmpeg 和入口 `--help`。
2. 运行 CPU 回归，确认没有失败；GPU 测试跳过不等于 GPU 验证通过。
3. 使用 `data/model/` 模型和默认素材 `data/113000356.mp4`、`data/113000356_mask.mp4` 完成一次 SDPA 推理，播放结果并检查擦除区域。
4. 按实际需要运行服务、上传素材、查询进度、下载产物；多卡和 INT8 单独启用对应测试。

用 FFprobe 检查输入与输出元数据，对每段视频分别执行：

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames \
  -of json outputs/result.mp4
```

尺寸、帧数、帧率需符合预期；不能仅以命令退出成功判定画面正确。

## 性能对照

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

逐帧解码 RGB，按 [统一质量标准](performance.md#acceptance) 比较基线与候选；
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

