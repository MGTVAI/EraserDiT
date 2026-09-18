# 服务端接口

启动：

```bash
./inference_server.sh --pipeline-name EraserDiTErasePipeline \
  --model-path /root/.cache/huggingface/hub/models--jieeliu--EraserDiT/snapshots/904fb412da76235085dbbccaefdbde4979fa3d29 \
  --task-root /tmp/mgerase_tasks --input-allowed-root "$PWD/data"
```

`--pipeline-name` 决定使用哪个模型；请求 schema、采样参数构造与 capability 标识都由该管线
声明的 `service_contract` 提供（`service/contracts/`），服务骨架不感知模型。

## 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/videos` | 创建任务，202；body 为 JSON（本地路径）或 multipart |
| GET | `/v1/videos` | 列表，`after` / `limit` / `order` |
| GET | `/v1/videos/{id}` | 单任务快照 |
| DELETE | `/v1/videos/{id}` | 运行中→取消，终态→删除 |
| GET | `/v1/videos/{id}/content` | 结果 mp4，仅 `completed` 且本地存储 |
| GET | `/v1/models`、`/v1/models/{id}` | 模型卡（含 `capability`） |
| GET | `/health` | 就绪状态 + 工作组心跳 |
| GET | `/server_info` | 启动配置 + `effective_acceleration` |
| GET | `/model_info` | 模型摘要 |
| GET | `/stats` | 队列与任务计数 |

## 任务状态与阶段

状态 `queued` → `running` → `completed` / `failed` / `cancelled`；
阶段 `queued` → `preparing` → `processing` → `finalizing` → `terminal`。
响应携带 `progress`（0–100）、`queue_position`、`object_index/object_count`、
`window_index/window_count`；`metrics` 只在终态填充。

## 请求契约

`extra="forbid"`：未声明字段返回 422。错误体统一为
`{"error": {"code", "message"}}`，失败任务额外带 `phase`；FastAPI 路由级 404/405 仍是
框架默认的 `{"detail": ...}`。

第一阶段只接受本地路径（受 `--input-allowed-root` 白名单约束）与本地结果存储。
EraserDiT 的参数面见 `service/contracts/eraserdit.py`，默认值即冻结基线（50 步 / strength 0.8 /
guidance 3.0 / infer_len 121 / overlap 9）。

## 验收

```bash
python scripts/service_smoke.py --base-url http://127.0.0.1:30000 \
  --video data/10268234.mp4 --mask data/10268234_mask.mp4 \
  --prompt "There is a bridge over the lake."
```

覆盖全部端点、严格契约、任务生命周期、终态快照、结果下载与删除。
