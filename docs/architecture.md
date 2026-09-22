# 代码架构

EraserDiT 将模型计算、流水线装配、窗口调度与服务入口分开管理。
CLI 和 HTTP worker 通过常驻 session 复用模型，各请求独立维护采样和执行状态。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `config/` | 请求参数、模型配置、服务契约、资源与并行配置 |
| `entrypoints/cli/` | 单视频、批量任务和多 GPU 任务入口 |
| `entrypoints/server/` | HTTP API、任务队列、worker、取消与结果存储 |
| `pipelines/` | 模型注册、组件装配、常驻会话与执行流程 |
| `pipelines/stages/` | 模型专用的预处理、编码、去噪、解码及窗口提交 |
| `pipelines/runtime/` | 窗口规划、对象调度、帧缓存管理、提交与输出 |
| `nodes/` | 共享请求对象、stage 基类、执行器和取消检查点 |
| `models/` | Transformer、VAE、文本编码器及模型专用适配 |
| `layers/` | 注意力、线性层、位置编码、量化与融合算子 |
| `loader/` | 组件构造和权重加载 |
| `cache/` | Transformer 残差缓存与复用策略 |
| `memory/` | 组件驻留、权重卸载、张量搬运及阶段内存管理 |
| `parallel/` | 并行策略、布局、分片与运行时初始化 |
| `distributed/` | 通信组、通信操作与分布式状态 |
| `media/` | 视频读写、编码契约与帧缓存 |
| `utils/` | 裁剪框、窗口、日志、计时等基础工具 |

## 执行流程

```text
CLI / HTTP worker → EraseSession → pipeline 装配
                                   ├─ loader → 模型组件
                                   ├─ 模型 stages → models / layers / cache / memory
                                   └─ 窗口 runtime → nodes / parallel / media
```

视频按窗口处理，重叠部分为相邻窗口提供连续性。运行时负责装载输入、执行 stages、
提交稳定帧、释放窗口资源并写出结果。模型专用的预处理、后处理和并行适配位于
`models/adapters/`，流程 stages 位于 `pipelines/stages/`。

`PipelineRegistry` 选择模型流水线，模型声明的 `service_contract` 提供请求 schema、
采样参数构造和 capability。服务进度适配位于 `entrypoints/server/control.py`，
通用取消令牌和同步检查点位于 `nodes/control.py`。

## 依赖约束

- 非入口模块不导入 `entrypoints`。
- `config` 定义配置，不导入模型、流水线或执行实现。
- `nodes` 不依赖模型和流水线装配；窗口 runtime 不反向导入模型 stages 或装配模块。
- `distributed` 不依赖上层并行策略；通用 `parallel` 不导入模型实现。
- `layers` 可以使用通用并行能力，`models` 不依赖 loader 或流水线。
- `media` 独立于其他项目包；`utils` 只依赖本包和 `media`。
- `memory` 仅使用配置和基础工具，不依赖模型装配。

`tests/test_architecture.py` 检查包依赖方向、实现包静态依赖图无环及独立导入行为。
检查覆盖普通导入、函数内导入、类型检查分支和字面量动态导入。

## 开发验证

新增模型需注册 pipeline、声明配置及服务契约、装配模型 stages。
配置约束、执行控制、窗口状态和模型计算分别在对应层验证。
测试命令和 GPU 条件见[回归测试](../tests/README.md)，完整视频验证见[测量与验证](validation.md)。

## 参考

- [EraserDiT](https://github.com/JieLiu95/EraserDiT)
- [SGLang](https://github.com/sgl-project/sglang)
