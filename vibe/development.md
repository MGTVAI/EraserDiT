# 加速框架迁移开发方案

## 架构

采用 MGErase 的组件加载、阶段执行和管线组装架构，EraserDiT 作为第一个接入模型。目录目标如下，后续阶段模块按需迁入：

```text
EraserDiT/
├── inference_cli.sh
├── inference_server.sh
├── config/
├── entrypoints/{cli,server}/
├── layers/{attention,operator_fusion,quantization,rotary_embedding}/
├── memory/{adapters,backends,policies}/
├── loader/component_loaders/
├── models/{dits,vaes,text_encoders,schedulers}/
├── nodes/{stages,executors}/
├── pipelines/
├── utils/
├── cache/
├── distributed/
├── parallel/
├── profiling/
├── service/
├── videoerase/
├── scripts/
├── data/
├── vibe/
├── docs/
└── docker/
```

映射规则：`python.<module>` → `<module>`；`python.runtime.<module>` → `<module>`。同步更新静态 import、动态注册路径、脚本和文档。迁入代码不依赖 MGErase 的绝对路径，保留来源及许可证说明。

内存管理为上述映射的例外：`python.layers.memory` 与 `python.runtime.resource` 统一重组到根目录 `memory/`，不保留原目录或兼容 import。按职责分配到 `adapters/`（模型内存适配）、`backends/`（具体搬运与卸载实现）、`policies/`（模型驻留策略和推理阶段协调）。策略通过适配器调用底层实现，底层实现不依赖任务管线。

## 职责边界

| 模块 | 职责 |
| --- | --- |
| `config/` | 通用运行配置、模型配置和采样参数 |
| `models/`、`loader/` | 模型注册、组件实现、权重加载及模型适配 |
| `layers/` | 可复用注意力、融合、量化和位置编码计算模块 |
| `memory/` | 统一管理模型内存适配、搬运卸载实现、驻留策略和推理阶段协调 |
| `nodes/`、`pipelines/` | 阶段接口、执行器、模型专属阶段及任务管线组装 |
| `videoerase/` | 擦除任务分段、衔接、输入输出及任务状态 |
| `service/` | 模型常驻和顺序任务服务 |
| `cache/`、`parallel/`、`distributed/` | 缓存策略和并行执行能力 |
| `profiling/` | 性能分析和计时记录 |
| `vibe/`、`docs/`、`docker/` | 需求与开发方案、使用与技术文档、容器构建及环境配置 |

将 MGErase 中硬编码的 LTX 行为收敛到模型专属配置、阶段或适配器；通用层不假定模型维度、VAE 压缩比例、采样规则和分段方式。新模型通过注册、组件加载、专属阶段和管线接入，无需改写公共执行器。第一阶段不要求实现第二个模型。

## 第一阶段实现

### 算法与生命周期

- 冻结原版代码版本、工作区差异、权重标识和运行配置，保存两组基准视频。
- 将原版预处理、mask 处理、scheduler、CFG、采样、分段衔接和后处理迁入新架构；不套用 MGErase 的默认算法参数。
- 原版当前入口未固定 seed；基线生成需固定并记录全部相关随机源，不改变算法参数。
- 模型实例归常驻会话所有；每个任务独立持有随机数、采样状态、读写资源、衔接帧及缓存。正常结束和异常退出均释放任务资源。
- Shell 仅负责启动和参数传递。CLI 与服务共用执行链路，第一阶段串行处理任务。
- 不维护旧 API 或启动方式；迁移验证后删除被替代实现。

### 注意力与融合

- 通过 Attention Processor 接入 self-attention 后端：SDPA、FlashAttention、SageAttention。cross-attention 保留原 SDPA 及文本 mask 语义。
- 迁入 MGErase 的 Q/K RMSNorm + RoPE、RMSNorm + 调制融合点。
- 检查内核的维度、dtype、布局、归一化参数及设备约束，不直接假定模型兼容。
- 显式指定不支持的后端时明确报错；`auto` 可回退，但必须记录实际后端及原因。

### 编译与预热

- 权重加载和注意力配置完成后编译 Transformer，保留原始模型引用。
- 预热覆盖实际尺寸、正常分段及尾段形状；使用独立任务状态，不能污染正式任务的随机数或输出。
- 未覆盖形状产生的编译开销计入该次任务，并标记首次与稳定运行差异。
- 验证实际前向执行，不能仅以编译包装函数返回作为成功依据。
- 沿用 MGErase 当前组合限制：手工 Triton 融合与整模型编译分开验证；缓存与整模型编译不能同时启用。后续量化按模式检查兼容性。

## 开发里程碑

| 顺序 | 工作 | 通过条件 |
| --- | --- | --- |
| 1 | 环境、权重检查与基线冻结 | 原版完整跑通两组视频并保存配置 |
| 2 | 目录重组、注册、加载器和执行骨架 | 新入口能加载 EraserDiT 组件 |
| 3 | 原版算法接入 | 新架构关闭加速后与原版视频对齐 |
| 4 | 常驻 CLI 与基础服务 | 两组素材交替、重复运行，无串扰或持续显存增长 |
| 5 | 注意力、融合、编译逐项接入 | 各候选配置通过视频检查并完成性能测量 |
| 6 | 组合筛选与交付 | 比较注意力＋融合、注意力＋编译，给出实测推荐配置 |

按同一 GPU、可比负载交替运行基线与候选配置。必要的算子和 Transformer 数值检查用于定位错误，最终验收使用完整输出视频。计时处理 GPU 异步执行，正式性能测量关闭详细诊断开销。

## 后续扩展

- 缓存：接入任务、分段、采样步骤生命周期，隔离 CFG 两个分支和不同任务的缓存。
- 量化：在加载后、编译前接入模型转换，按硬件能力及模式验收。
- 多卡：扩展启动器、组件设备分配、序列/CFG/VAE 并行，单独验证通信及输出对齐。

当前已观察到 A100 80GB GPU，存在其他任务显存占用。开发前仍需核查实际依赖、可用 GPU 资源和本地权重；部分 MGErase FP8 路径不适用于当前硬件，不能作为本机已验证能力交付。
