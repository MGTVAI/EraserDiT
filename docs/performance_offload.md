# P1a 整组件卸载验证

本文保留 P1a 实施当时的记录；动态卸载后续进展见 [P1b 报告](performance_dynamic_offload.md)。

2026-09-20。开发安排见 [performance_plan.md](../vibe/performance_plan.md)。

## 实现边界

`--resource-policy component_offload` 已接入 EraserDiT CLI 与通用服务启动入口。
加载器把 T5、VAE、Transformer 权重放在 CPU；四个计算阶段通过统一装饰器在进入时
执行阻塞的 `Module.to(device)`，在 `finally` 中返回 CPU。VAE 整体搬运，根级 buffer
一起移动；Transformer 在整个去噪循环内驻留。未启用时没有搬运。

预加载组件同样归位 CPU。与 `torch.compile` 的组合暂未验证，显式拒绝；EraserDiT
`dynamic_offload` 尚未接入，在加载模型前报错。新策略暂不允许其他模型管线使用。
CLI 结果与 `/server_info` 的 `effective_acceleration.resource_policy` 显示解析后的策略。
服务入口与配置代码已接入，但本轮没有重跑 HTTP 服务端到端测试。

## 实测

GPU 7（共享卡），PyTorch 2.6.0+cu126，确定性模式、SDPA、bf16、seed=42；权重快照
`904fb412da76235085dbbccaefdbde4979fa3d29`。将第一组视频与 mask 缩到 192×320，截取 25 帧；
`infer_len=17, overlap=9, num_inference_steps=4, strength=0.8`，覆盖两窗口及衔接路径。
两个配置各一次，未预热；以下是功能冒烟结果，不是正式性能基准，也不能外推至 1080p。

| 配置 | 端到端 / 秒 | peak allocated / GiB | peak reserved / GiB |
| --- | ---: | ---: | ---: |
| fullgpu | 3.050 | 15.055 | 15.096 |
| component_offload | 30.704 | 8.923 | 8.951 |

两份输出 MP4 **逐字节一致**，SHA256：
`56347c538e7ad47bb9b753521970d99f85c427761ad0883d8c611ba00f03b4b7`。

峰值 allocated 减少约 40.7%，但小尺寸、少步数下传输占主导，耗时增加约 10 倍。
当前适用目标是显存容量；不得将其称为加速配置。尚未测量加载峰值、主机 RSS、传输时间
细分、完整分辨率或 5 次重复。VAE 激活峰值不会被整组件卸载消除。

本地原始日志、输出、汇总：`results/component_offload_smoke/{fullgpu,offload}.log`、
同名 `.mp4`、`comparison.json`。这些是生成产物，不随源码提交。

## 复现

```bash
mkdir -p results/component_offload_smoke
ffmpeg -hide_banner -loglevel error -y -i data/10268234.mp4 \
  -vf scale=192:320 -frames:v 25 -an results/component_offload_smoke/video.mp4
ffmpeg -hide_banner -loglevel error -y -i data/10268234_mask.mp4 \
  -vf scale=192:320:flags=neighbor -frames:v 25 -an results/component_offload_smoke/mask.mp4

for policy in fullgpu component_offload; do
  CUDA_VISIBLE_DEVICES=2 ./inference_cli.sh \
    --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model \
    --video-input results/component_offload_smoke/video.mp4 \
    --mask-input results/component_offload_smoke/mask.mp4 \
    --output-path "results/component_offload_smoke/${policy}.mp4" \
    --prompt 'There is a bridge over the lake.' \
    --infer-len 17 --overlap 9 --num-inference-steps 4 --resource-policy "$policy"
done
```

## 回归

9 项测试全部通过，包含 CUDA 实测：默认无搬运、正常阶段搬运、执行异常、部分加载失败、
CPU 加载策略、未支持组合在模型加载前拒绝、预加载组件归位、其他管线拒绝新策略、重复
GPU 搬运后输出/共享参数/buffer 保持正确。

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/shanhai-ai/envs/conda/envs/EraserDiT/bin/python \
  -m unittest discover -s tests -p test_component_offload.py -v
```

下一步 P1b：权重预算与按层卸载注册、异步预取及阶段搬运观测；随后推进 TeaCache、cache-dit，
最后量化。全分辨率回归、连续真实任务及 HTTP 验收仍需补齐。
