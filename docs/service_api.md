# 服务端接口

[CLI](cli.md) · [性能与验收](performance.md)

从仓库根目录运行，先按 [安装与部署](setup.md) 安装依赖并创建 `.venv` 环境。
环境变量显式设置，见 [CLI](cli.md)。

启动（模型统一使用 `data/model`，默认输入目录为 `data/`）：

```bash
MODEL_DIR="$PWD/data/model"
INPUT_DIR="$PWD/data"
CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m entrypoints.server.serve \
  --pipeline-name EraserDiTErasePipeline --model-path "$MODEL_DIR" \
  --task-root "$PWD/outputs/service" --input-allowed-root "$INPUT_DIR" \
  --host 127.0.0.1 --port 30000
```

服务在前台运行，使用 Ctrl+C 停止。`--task-root` 是服务端可写的产物目录；
`--input-allowed-root` 是服务端输入白名单，JSON 请求中的输入文件必须位于该目录内。
服务不会自动选择或等待 GPU。请求采样参数可以逐任务设置，设备、卸载、编译和并行拓扑需在启动时确定。

| 部署参数 | 默认值 / 含义 |
| --- | --- |
| `--max-queued-tasks` | `8`，等待队列上限 |
| `--max-upload-bytes` | `4294967296`，上传大小限制 |
| `--terminal-task-ttl-seconds` | `86400`，终态任务保留时间 |
| `--max-terminal-tasks` | `128`，终态任务保留上限 |
| `--result-storage-mode` | `local`；可选 `s3`，需配置 bucket |

S3 参数与其他服务选项用 `uv run --no-project python -m entrypoints.server.serve --help` 查看。
任务记录仅保存在内存中，服务重启后不会恢复；及时下载需要保留的结果。

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
EraserDiT 的参数面见 `config/service_contracts/eraserdit.py`，默认采样配置为 50 步 / strength 0.8 /
guidance 3.0 / infer_len 121 / overlap 9。

## 调用示例与兼容范围

服务接口设计参考 [SGLang](https://github.com/sgl-project/sglang)，使用 EraserDiT 的采样契约。
这不是 sglang VideoEdit 请求体的完整兼容层：输入字段仍为 `video_path` / `mask_path`，
任务 ID 由服务生成；不接受客户端 `task_id`、远程输入 URL、回调或请求级输出路径。
任务记录在内存中，重启后不能继续查询旧任务。

`model` 可省略；指定时必须等于 `GET /v1/models` 返回的模型 ID（启动模型目录的名称），
不匹配返回 404。该字段只用于服务模型校验，不传入算法采样参数。
`/docs` 与 `/openapi.json` 展示启动管线的 JSON 参数和 multipart 定义。

```bash
# 在同机客户端终端从仓库根目录执行；创建后保存响应中的 id
INPUT_DIR="$PWD/data"
curl -sS http://127.0.0.1:30000/v1/videos/eraser \
  -H 'Content-Type: application/json' \
  -d "{\"video_path\":\"$INPUT_DIR/113000356.mp4\",\"mask_path\":\"$INPUT_DIR/113000356_mask.mp4\",\"seed\":42,\"prompt\":\"There is a rooftop terrace overlooking the city at sunset.\"}"

# 文件上传：parameters 是 JSON 字符串，不是多个独立表单字段
curl -sS http://127.0.0.1:30000/v1/videos/eraser \
  -F "video=@$INPUT_DIR/113000356.mp4" \
  -F "mask=@$INPUT_DIR/113000356_mask.mp4" \
  -F 'parameters={"seed":42,"num_inference_steps":50,"prompt":"There is a rooftop terrace overlooking the city at sunset."}'

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
uv run --no-project python -m unittest discover -s tests -p test_service_api.py -v
```

## 验收

服务就绪后，在另一终端执行：

```bash
curl -fsS http://127.0.0.1:30000/ready
curl -fsS http://127.0.0.1:30000/v1/models
```

按上面的调用示例提交任务、查询进度；确认状态为 `completed` 后下载并播放视频，再删除任务。
远程客户端使用 multipart 上传，或提供服务端可读且在白名单中的路径。
另提交一个任务，在运行中取消，查询到 `cancelled` 后再提交新任务，确认服务仍能正常完成请求。
严格参数校验、错误码和任务生命周期由上面的无模型接口测试覆盖。
如本机配置了 HTTP 代理，为本地调用设置 `NO_PROXY=127.0.0.1,localhost`。

模型或权重变更后应重新验收。接口验收不判断画面质量，需另按
[质量标准](performance.md#acceptance) 对照输出视频。

服务默认开启单卡 SGLang 源码卸载；开关与 CLI 一致，见 [内存管理](sglang_memory.md)。

NCCL DiT 多进程服务通过启动参数 `--dit-parallel-backend nccl` 选择；
CFG、SP/Ulysses/Ring、TP 与 FSDP/HSDP 的度数在启动时固定，不能按请求改变通信拓扑。
当前要求 BF16 权重、关闭编译；常驻 CFG/Ulysses SP1/2/4 可使用 SDPA 或 Sage FP8、静态 FP8 FFN、已支持的融合与缓存。TP/Ring/FSDP 仍要求 SDPA、未量化。
`--cuda-memory-limit-gib` 与 CLI 共用；预算模式在去噪窗口外释放共用主卡的 rank 0 权重，
仍需用 NVML 汇总同卡进程占用。TP/Ring/FSDP 的组合边界见性能说明；不支持的请求会拒绝。
响应指标中的 `parallel_history` 记录各窗口的实际后端、rank 参数字节、通信次数与显存峰值。
策略参数、整片质量和性能证据见 [NCCL 并行验收](distributed_parallel_20260928.md)。

## 浏览器工作台

服务启动后访问 `http://127.0.0.1:30000/ui`。页面提供视频与 mask 上传、
静态区域画笔、提示词、采样参数、任务进度、取消、预览和下载。
画笔生成的区域固定应用于全部帧；运动目标请上传逐帧 mask 视频。

Multipart `mask` 支持原有 mask 视频或与视频分辨率一致的单帧 PNG。
PNG 透明区域视为黑色，后端按视频实际帧数与帧率生成无损 mask；尺寸不匹配返回 422。
JSON 路径请求仍使用 mask 视频。

页面的质量策略关闭去噪缓存；低显存策略使用 65 帧窗口；速度候选使用
TeaCache 0.1、`mask_frame_max` 局部保护。后两者会改变结果，应检查窗口衔接和擦除区域。
GPU 数量、FP8、卸载和显存预算由服务启动配置决定，页面不会动态切换常驻模型。
