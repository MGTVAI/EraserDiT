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
| `utils/` | 视频读写、编码契约、帧缓存、裁剪框、窗口、日志、计时等基础工具 |

## 执行流程

```text
CLI / HTTP worker → EraseSession → pipeline 装配
                                   ├─ loader → 模型组件
                                   ├─ 模型 stages → models / layers / cache / memory
                                   └─ 窗口 runtime → nodes / parallel / utils
```

视频按窗口处理，重叠部分为相邻窗口提供连续性。运行时负责装载输入、执行 stages、
提交稳定帧、释放窗口资源并写出结果。模型专用的预处理、后处理和并行适配位于
`models/adapters/`，流程 stages 位于 `pipelines/stages/`。

同一对象、场景和正负提示词的 T5 embedding 只在首个实际处理窗口计算一次；后续窗口
在进入文本编码器的驻留阶段前直接读取请求内 CPU 缓存。显式选择 NCCL 且未指定
`cfg_degree` 时，只要设备数覆盖 `2 × sp_degree`，默认使用 CFG2 并行计算正负分支；
显式 `--cfg-degree` 始终覆盖该默认值。

`PipelineRegistry` 选择模型流水线，模型声明的 `service_contract` 提供请求 schema、
采样参数构造和 capability。服务进度适配位于 `entrypoints/server/control.py`，
通用取消令牌和同步检查点位于 `nodes/control.py`。

流水线负责模型组件、stage 装配和请求资源生命周期；窗口驱动直接调用运行时的事件、
帧加载、对象调度与提交函数，不再由流水线构造回调集合并逐层传递。
`windowing/handlers.py` 只负责输入张量格式转换。模型专用的 RGB 掩码读取阈值仍由
EraserDiT 流水线传入，帧缓存释放和对象间转发 hooks 则由运行时维护。

窗口命令和共用错误同步由 `windowing/sp_dispatch.py` 管理，提交模块
`windowing/commit_sync.py` 单向依赖它，并保留原有错误与同步函数导出。
模型 stages 的 latent 归一化、反归一化和帧数换算统一使用 `utils/latent.py`。

## 依赖约束

- 非入口模块不导入 `entrypoints`。
- `config` 定义配置，不导入模型、流水线或执行实现。
- `nodes` 不依赖模型和流水线装配；窗口 runtime 不反向导入模型 stages 或装配模块。
- `distributed` 不依赖上层并行策略；通用 `parallel` 不导入模型实现。
- `layers` 可以使用通用并行能力，`models` 不依赖 loader 或流水线。
- `utils` 的基础工具实现只依赖本包，不依赖其他项目包。
- `memory` 仅使用配置和基础工具，不依赖模型装配。

`tests/test_architecture.py` 检查包依赖方向、实现包及运行时模块静态依赖图无环，
并验证独立导入行为。
检查覆盖普通导入、函数内导入、类型检查分支和字面量动态导入。

## 开发验证

新增模型需注册 pipeline、声明配置及服务契约、装配模型 stages。
配置约束、执行控制、窗口状态和模型计算分别在对应层验证。
测试命令和 GPU 条件见[回归测试](../tests/README.md)，完整视频验证见[测量与验证](validation.md)。

## 参考

- [EraserDiT](https://github.com/JieLiu95/EraserDiT)
- [SGLang](https://github.com/sgl-project/sglang)

当前仅注册 `EraserDiTErasePipeline`，CLI 与 HTTP 服务默认使用 EraserDiT。
共享窗口运行时、会话、FlowMatch 调度器和缓存控制器独立于旧模型路径；
TeaCache 的模型标识和校准策略由 EraserDiT 适配层显式提供。

## 内存管理

`memory/backends` 保存迁移的 SGLang 逐层和 FSDP 包装源码；
`memory/adapters/sglang_memory_adapter.py` 管理阶段与进程组生命周期；
`loader/meta_load.py` 负责本模型的 meta 初始化和严格权重物化。
配置与当前组合边界见 [SGLang 内存管理](sglang_memory.md)。

NCCL DiT 路径由 `pipelines/runtime/dit_executor.py` 管理常驻子进程；
父进程保留 T5/VAE、scheduler 与 RNG，DiT worker 使用独立 NCCL world，避免替换父进程 T5 的 FSDP 组。
`distributed/dit_groups.py` 只处理通用分组与通信，模型计算适配位于
`models/adapters/eraserdit/nccl_runner.py` 和 `nccl_sequence.py`；TP Linear 位于 `layers/dit_tensor_parallel.py`。
worker 在返回前以 FP32 合并 CFG，并仅由 owner 汇聚必要的 SP 输出；输入和预测默认经 CPU IPC 传递。
实验 CUDA IPC 路径由 `MGERASE_DIT_BOUNDARY_TRANSPORT=cuda_ipc` 启用，接收端复制到自身 GPU 存储，
以隔离调用方与 worker 的张量生命周期；父进程仍负责 scheduler 和 RNG。
策略范围见 [NCCL 并行记录](distributed_parallel_20260928.md)，边界计时与输出通信优化见
[2026-09-29 验证](nccl_boundary_optimization_20260929.md)。
