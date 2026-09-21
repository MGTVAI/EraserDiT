# P1b 动态卸载接入与验证

2026-09-20，接续 [P1a 整组件卸载](performance_offload.md)。后续 GPU 操作仅使用用户指定的物理 2、3 号卡。

## 已接入的行为

EraserDiT 支持 `--resource-policy dynamic_offload`，默认管理权重预算为 5 GiB：

- T5、Transformer、VAE encoder / decoder 分别注册，共 69 个权重块。
- 当前组件的受管权重能放进预算时，在整个阶段保持驻留。Transformer 的正/负 CFG 及全部去噪步共用这次驻留。
- 放不进预算时，前向按权重块在独立 HtoD stream 上加载，计算 stream 等待依赖；计算完成后后台释放 GPU 权重视图，回到 pinned CPU 镜像。已经提交的传输可与上一块 GPU 计算重叠；没有额外的预测式多层预取器。
- `--max-weight-usage` 的单位是字节，约束 **受管权重**，不约束激活、CUDA workspace、未包装的小层权重和 allocator reserved。过小、不能容纳不可再拆分层的预算，在注册前报错。
- 受管权重始终保留 pinned CPU 镜像（本模型约 14.75 GiB）。`--pin-memory` 仅额外固定未包装的小层权重，不控制受管权重的镜像。
- CLI `memory_runtime` 和 `/server_info` 的 `effective_acceleration.memory_runtime` 给出预算、块数、实际搬运次数/字节数、阶段驻留次数、预算不足回退次数与队列状态。搬运计数是会话累计值。
- 保持默认 `fullgpu`；`component_offload` 继续可选。动态/整组件卸载与 Transformer compile 的组合仍显式拒绝。

实现修复了两处模型接入问题：T5 的两个 Embedding 对象共享同一参数，注册期间通过官方 setter 合并为单个模块，关闭时恢复原结构；EraserDiT 的融合 block helper 原本绕过 `block.forward`，现在注册的 block 经卸载包装器调用同一个融合 helper，保持数学运算不变。

另补齐异步后端的异常计算结束事件、部分传输回滚、已退休事件的重复清理、唯一事件 ID、共享参数别名恢复和关闭后包装器清理。

## 小尺寸 GPU 对照

192×320、25 帧、两个窗口，`infer_len=17, overlap=9, num_inference_steps=4, seed=42`，确定性模式、SDPA、bf16。以下各配置单次运行，没有预热，不是正式性能基准。

| 配置 | 端到端 / 秒 | peak allocated / GiB | peak reserved / GiB |
| --- | ---: | ---: | ---: |
| fullgpu（P1a 对照） | 3.050 | 15.055 | 15.096 |
| component_offload（P1a） | 30.704 | 8.923 | 8.951 |
| dynamic_offload，2 GiB 预算 | 8.822 | 1.850 | 2.377 |

三份 MP4 逐字节一致，SHA256：
`56347c538e7ad47bb9b753521970d99f85c427761ad0883d8c611ba00f03b4b7`。

2 GiB 下 T5 与 Transformer 走逐层搬运，VAE 编码/解码各自完整驻留；两个窗口结束后，事件队列、受管权重和驻留权重计数均为零。小尺寸测试在用户指定卡限制之前完成，原始 GPU 为 7；后续仅使用 2、3 号卡。

## 连续任务和异常恢复

192×320、33 帧、三个窗口，默认 5 GiB 预算；在一个会话中按“成功任务 → 注入第一个 Transformer block 计算异常 → 再次成功任务”执行。

- 两次成功输出逐字节一致，SHA256：`27b4aa7a3ab9cd5414b9ff61093134e7875607e1eecdac7166302d2e7b7612d9`。
- 端到端分别 7.803、6.413 秒，allocated 峰值分别 4.725、4.749 GiB。
- 正常与异常结束后，每个组件的参数和 buffer 均在 CPU，T5 权重共享关系保留，无活动组件租约，事件队列和驻留字节归零。
- 该项也在用户指定卡限制前完成，原始 GPU 为 7。

原始报告：`results/dynamic_offload_smoke/repeat_5g.json`。新增验证脚本可重复此流程：

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python scripts/offload_verify.py \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model \
  --video-input /path/to/video.mp4 --mask-input /path/to/mask.mp4 \
  --output-path results/offload_verify.json \
  --prompt 'There is a bridge over the lake.' \
  --infer-len 17 --overlap 9 --num-inference-steps 4 \
  --resource-policy dynamic_offload --repeat 2 --inject-failure
```

`--output-path` 在此验证脚本中指 JSON 报告，成功请求的视频保存在旁边；普通 CLI 中仍是视频路径。

## 原尺寸、回归和服务验证

物理 GPU 2，同一卡串行运行，`10268234` 原尺寸 1080×1920、120 帧，模型内部对齐为 1088×1920；窗口 121，默认 50 个调度步 / strength 0.8，实际 40 个去噪步。相同权重、提示词、seed=42、确定性口径、SDPA、bf16，关闭 compile 与融合，各配置一次，无预热。

| 配置 | 端到端 / 秒 | 去噪阶段 / 秒 | peak allocated / GiB | peak reserved / GiB |
| --- | ---: | ---: | ---: | ---: |
| fullgpu | 210.138 | 182.495 | 47.115 | 53.314 |
| dynamic_offload，5 GiB 预算 | 215.269 | 183.143 | 33.631 | 43.471 |

动态卸载 allocated 峰值减少 **13.48 GiB / 28.6%**，端到端增加 **5.13 秒 / 2.4%**。
两份 MP4 **逐字节一致**，SHA256：
`0b2c25b40f34d938c106379c5c47ffc67053be7341140f6d43c777d7e0d6ad43`。
因此该对照不存在新增画面差异；这不替代对模型原有擦除质量的评价。

该次受管权重峰值 5,279,797,248 字节，低于预算 5,368,709,120 字节；Transformer 阶段驻留权重约 3.58 GiB。任务结束后受管权重、驻留权重、事件队列归零。加载分别 13.948 / 20.780 秒，独立于端到端列出。

这是共享卡上的单次功能与显存验证，不是 5 次重复性能验收。原始记录：
`results/dynamic_offload_smoke/fullsize_comparison.json`、同目录 `fullsize_*.log` 和 `.mp4`。

最终回归在物理 GPU 2 上全部通过：**22 项**，包含整组件卸载回归、两种预算路径、连续请求、层计算异常、部分 HtoD 失败、注册失败回滚、非默认 CUDA stream、T5 共享 embedding、真实 EraserDiT block 调度，以及小模型关闭后设备状态清理。记录：`results/dynamic_offload_smoke/tests_gpu2.log`。

HTTP 服务也在物理 GPU 2 上通过全部端点、严格请求契约、任务生命周期、指标、结果下载与删除检查。`/server_info` 确认注册 69 个块、动态策略生效；任务结束后队列、驻留字节归零，实际搬运计数非零。已关闭验证用临时服务。记录：`service_smoke_gpu2.log`、`server_info_after_task_gpu2.json`（同一结果目录）。

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python \
  -m unittest discover -s tests -v
```

本地 HTTP 验证需设置 `NO_PROXY=127.0.0.1,localhost`，避免环境代理将回环地址请求转发出去。

## 使用示例与剩余工作

```bash
CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model \
  --video-input data/10268234.mp4 --mask-input data/10268234_mask.mp4 \
  --output-path results/dynamic.mp4 --prompt 'There is a bridge over the lake.' \
  --resource-policy dynamic_offload --max-weight-usage 5368709120
```

本轮不宣称速度验收通过：正式推荐还需要两组原尺寸素材、每配置 ≥5 次、同卡占用记录和离散度；尚未量化主机 RSS、加载峰值及 GPU 传输耗时细分。当前验证使用 SDPA、关闭算子融合和 compile，其他组合没有纳入本轮验收。

P2 将先接入 TeaCache / cache-dit 的 EraserDiT 前向位置与独立 CFG 状态，再做阈值扫描和质量验证；P3 随后接入量化。
