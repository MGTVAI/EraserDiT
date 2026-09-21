# 服务端接口

启动：

```bash
./inference_server.sh --pipeline-name EraserDiTErasePipeline \
  --model-path /mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT/results/cache_prediction_model \
  --task-root /tmp/mgerase_tasks --input-allowed-root "$PWD/data"
```

`--pipeline-name` 决定使用哪个模型；请求 schema、采样参数构造与 capability 标识都由该管线
声明的 `service_contract` 提供（`config/service_contracts/`），服务骨架不感知模型。

## 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/videos` | 创建任务，202；body 为 JSON（本地路径）或 multipart |
| POST | `/v1/videos/eraser` | 擦除任务入口，与 `/v1/videos` 使用相同契约 |
| GET | `/v1/videos` | 列表，`after` / `limit` / `order`，返回 `first_id` / `last_id` / `has_more` |
| GET | `/v1/videos/{id}` | 单任务快照 |
| GET | `/v1/videos/{id}/progress` | 状态、阶段、进度、排队位置与窗口/对象计数 |
| DELETE | `/v1/videos/{id}` | 运行中→取消，终态→删除 |
| GET | `/v1/videos/{id}/content` | 结果 mp4，仅 `completed` 且本地存储 |
| GET | `/v1/models`、`/v1/models/{id}` | 模型卡（含 `capability`） |
| GET | `/health`、`/ready` | 就绪状态 + 工作组心跳；未就绪返回 503 |
| GET | `/server_info`、`/get_server_info` | 启动配置 + `effective_acceleration` |
| GET | `/model_info`、`/get_model_info` | 模型摘要 |
| GET | `/stats` | 队列与任务计数 |

## 任务状态与阶段

状态 `queued` → `running` → `completed` / `failed` / `cancelled`；
阶段 `queued` → `preparing` → `processing` → `finalizing` → `terminal`。
响应携带 `progress`（0–100）、`queue_position`、`object_index/object_count`、
`window_index/window_count`；`metrics` 只在终态填充。

## 请求契约

`extra="forbid"`：未声明字段返回 422。错误体统一为
`{"error": {"code", "message"}}`，失败任务额外带 `phase`；路由级 404/405 也使用该错误体。

输入接受本地路径（受 `--input-allowed-root` 白名单约束）或 multipart 文件上传。
结果存储由部署配置决定，可使用本地存储或已有的 S3 发布后端。
EraserDiT 的参数面见 `config/service_contracts/eraserdit.py`，默认值即冻结基线（50 步 / strength 0.8 /
guidance 3.0 / infer_len 121 / overlap 9）。

## 调用示例与兼容范围

参考本地 sglang 的 `multimodal_gen/runtime/entrypoints/openai/video_api.py` 中
repair、progress 和异步任务组织方式，保留 EraserDiT 的采样契约。
这不是 sglang VideoEdit 请求体的完整兼容层：输入字段仍为 `video_path` / `mask_path`，
任务 ID 由服务生成；不接受客户端 `task_id`、远程输入 URL、回调或请求级输出路径。
任务记录在内存中，重启后不能继续查询旧任务。

`model` 可省略；指定时必须等于 `GET /v1/models` 返回的模型 ID（启动模型目录的名称），
不匹配返回 404。该字段只用于服务模型校验，不传入算法采样参数。
`/docs` 与 `/openapi.json` 展示启动管线的 JSON 参数和 multipart 定义。

```bash
# 创建后保存响应中的 id
curl -sS http://127.0.0.1:30000/v1/videos/eraser \
  -H 'Content-Type: application/json' \
  -d "{\"video_path\":\"$PWD/data/10268234.mp4\",\"mask_path\":\"$PWD/data/10268234_mask.mp4\",\"seed\":42}"

# 文件上传：parameters 是 JSON 字符串，不是多个独立表单字段
curl -sS http://127.0.0.1:30000/v1/videos/eraser \
  -F 'video=@data/10268234.mp4' \
  -F 'mask=@data/10268234_mask.mp4' \
  -F 'parameters={"seed":42,"num_inference_steps":50}'

TASK_ID='替换为创建响应中的id'
curl -sS "http://127.0.0.1:30000/v1/videos/$TASK_ID/progress"
# completed 后下载本地产物；远程结果使用任务响应中的 url
curl -f "http://127.0.0.1:30000/v1/videos/$TASK_ID/content" -o result.mp4
curl -sS -X DELETE "http://127.0.0.1:30000/v1/videos/$TASK_ID"
```

分页 `limit` 为 1–100，默认 20；`order` 为 `asc` / `desc`，默认 `desc`。
下一页以当前页 `last_id` 作为 `after`；只有确实存在后续记录时 `has_more=true`。
排队位置从 0 开始。对运行中任务的取消是协作式取消，需要继续查询到终态。
工作组未就绪、心跳过期或停止接单时，提交返回 503；队列满返回 429。
无效路径、未知参数、重复 multipart 字段返回 422；上传超限沿用 429。
上传大小限制针对复制到任务目录的文件，网关仍应设置请求体限制以约束表单解析阶段的临时文件。

无需 GPU 的接口回归测试（需要服务依赖和 `httpx`）：

```bash
python -m unittest discover -s tests -p test_service_api.py -v
```

## 验收

```bash
python scripts/service_smoke.py --base-url http://127.0.0.1:30000 \
  --video data/10268234.mp4 --mask data/10268234_mask.mp4 \
  --prompt "There is a bridge over the lake."
```

覆盖全部端点、严格契约、任务生命周期、终态快照、结果下载与删除。

一键起服务 + 验收（等卡、起服务、跑验收、无论成败都拆干净）：

```bash
scripts/service_verify.sh          # 自动等一张 ≥60 GiB 空闲的卡
scripts/service_verify.sh 2        # 指定卡
```

**模型路径有改动后必须重跑。** M2 的首次验收（`6c714f2`）跑在 cross-attention 修复
（`26c5a6b`）之前：端点检查全过，但它下载到的 mp4 是损坏版本产出的——验收查的是接口
行为，不查画面。修复后重验通过，且下载产物与 M1b 验收产物 md5 一致
（`ea93b4f2…`），即服务端内容也回到了已验证状态。
