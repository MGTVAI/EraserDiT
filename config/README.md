# 配置说明

| 文件 | 用途 |
| --- | --- |
| `server_args.py` | 模型加载、设备、精度、注意力、编译和资源策略 |
| `sampling_params.py` | 通用采样参数 |
| `eraserdit.py` | 模型配置与采样默认值 |
| `service_args.py` | HTTP 服务、队列、输入路径和结果存储 |
| `service_contract.py` / `service_contracts/` | 模型请求 schema、采样参数构造与 capability |
| `parallel.py` | 并行配置和计划类型 |
| `resource_policy.py` | 权重驻留和卸载策略 |
| `eraserdit_cache.py` / `transformer_cache.py` | Transformer 缓存参数与校验 |

进程级配置通过 CLI 设置；请求级采样参数通过 CLI、任务 JSON 或 HTTP 请求设置。
字段和默认值以各入口的 `--help` 及服务 `/docs` 为准。

```bash
uv run --no-project python -m entrypoints.cli.erase_eraserdit --help
uv run --no-project python -m entrypoints.server.serve --help
```

使用示例见[CLI](../docs/cli.md)和[服务 API](../docs/service_api.md)。
