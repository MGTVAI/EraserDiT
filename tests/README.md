# 回归测试

使用 `unittest`，保持单层目录，按功能命名。以下命令从仓库根目录执行。
先按[安装说明](../docs/setup.md)创建 uv 环境并安装项目依赖；`httpx` 等测试依赖已包含在统一的 `requirements.txt` 中，视频 IO 测试需 FFmpeg。
测试中的固定路径字符串和模拟 GPU 编号是契约测试数据，不会读取这些文件或占用对应物理卡。
实际文件测试使用临时目录并自动清理；无需下载模型或准备仓库示例视频即可运行默认 CPU 回归。GPU 测试有跳过条件，跳过不表示已通过验证。

## 默认 CPU 回归

显式隐藏 GPU，避免卸载和缓存测试自动使用可见 CUDA 设备：

```bash
CUDA_VISIBLE_DEVICES='' ERASERDIT_TEST_TWO_GPU=0 ERASERDIT_TEST_INT8=0 \
  OMP_NUM_THREADS=1 uv run --no-project python -m unittest discover -s tests -v
```

单文件：`uv run --no-project python -m unittest discover -s tests -p test_service_api.py -v`。

| 文件 | 覆盖范围 |
| --- | --- |
| `test_block_compile_offload.py` | FFN 编译与图外融合；GPU 上权重反复卸载、形状切换、预热异常释放和重试 |
| `test_architecture.py` | 单向依赖边界、取消机制独立导入 |
| `test_execution_control.py` | 本地取消、对端取消传播、服务端兼容符号 |
| `test_assembly_contracts.py` | EraserDiT 默认契约与旧模型拒绝、并行计划类型兼容及默认策略 |
| `test_stage_compatibility.py` | stage 导入顺序、模块身份、patch 与装配契约 |
| `test_parallel_compatibility.py` | 通用并行/算子独立导入、模型适配模块身份及 patch 契约 |
| `test_runtime_boundaries.py` | 分布式状态与日志判断、资源策略回退、视频 IO；预加载/流式窗口顺序、重叠、跳过、多对象传递和提交异常释放 |
| `test_prepost_boundaries.py` | EraserDiT 预/后处理导入契约、patch 行为、通用裁剪独立导入 |
| `test_memory_lifetimes.py` | mask 分批与完整处理的 CPU/GPU 数值一致性、进入 VAE 前视频引用释放 |
| `test_vae_memory.py` | VAE 分块预算、归一化/卷积邻域、原位激活及残差分块的精确性、输入不变、梯度回退和 CLI/服务参数 |
| `test_static_condition_reuse.py` | 请求内文本编码缓存失效和隔离、预计算 RoPE 的 CPU/GPU 数值一致性 |
| `test_operator_fusion_precision.py` | QK RoPE、gated residual 舍入一致性，布局/梯度回退，完整 block 与文本缓存、逐层卸载组合 |
| `test_service_api.py` | HTTP 契约、任务和产物；使用 scheduler stub，无权重 |
| `test_component_offload.py` | 组件租约、异常清理；CUDA 可用时附加设备验证 |
| `test_layerwise_offload.py` | 迁移管理器的循环预取、布局、重复推理、T5 FSDP、异常清理；CUDA 用例需 GPU |
| `test_cache_regions.py` | 逐帧全图/mask/边缘探针，小目标保护、非等长分片汇总、空区域和非有限值 |
| `test_cache.py` | CFG/窗口隔离、探针、FP32 残差、请求契约；部分测试需要 CUDA |
| `test_cfg_parallel.py` | CFG 配置约束与显式双卡小模型检查 |
| `test_mesh.py` | CPU 分组、非整除分片、通信、失败传播及 DP 分配 |
| `test_nccl_dit.py` | 正交拓扑、真实 NCCL Ulysses/Ring/TP/FSDP/混合组、父进程组隔离及 worker 故障清理 |
| `test_ulysses_overlap.py` | 真实 NCCL head 分块流水，SP2/CFG2×SP2/SP4、非等长分片、stream 生命周期与异常后复用 |
| `test_nccl_packing.py` | Ulysses 新旧打包的收发数据精确一致性，覆盖等长/不等长分片及非连续布局 |
| `test_nccl_text_cache.py` | rank 文本缓存条件/权重失效、分支/窗口隔离、抽样 profiler 与异常 hooks 清理；真实多卡检查位于 `test_nccl_dit.py` |
| `test_window_benchmark.py` | 单窗口计时汇总，拒绝多窗口和非 121 帧报告 |
| `test_mesh_gpu.py` | 显式两/四卡 Transformer、异常恢复和可选真实 VAE |
| `test_quantization.py` | 层覆盖及组合约束，INT8/FP8 GPU 数值验证和融合 GEMM 对照 |

## GPU 回归

在已分配的设备上执行。单卡会自动运行 CUDA 可用性控制的卸载/缓存测试；
INT8 需额外开启开关：

```bash
CUDA_VISIBLE_DEVICES=0 ERASERDIT_TEST_INT8=1 OMP_NUM_THREADS=1 \
  uv run --no-project python -m unittest discover -s tests -v
```

两卡及真实 VAE（使用 `data/model/` 中的完整模型，需包含 `vae/`；GPU 编号按机器选择）：

```bash
CUDA_VISIBLE_DEVICES=0,1 ERASERDIT_TEST_TWO_GPU=1 \
  ERASERDIT_TEST_MODEL="$PWD/data/model" \
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  uv run --no-project python -m unittest discover -s tests -v
```

不设置 `ERASERDIT_TEST_MODEL` 会跳过真实 VAE checkpoint 测试。
单/双卡 VAE 性能与显存对照单独设置 `ERASERDIT_TEST_VAE_BENCHMARK=1`，运行
`uv run --no-project python -m unittest tests.test_mesh_gpu.VAESpatialBenchmarkTests -v`。
测试分别预热、交替测量编码/解码，记录 BF16 误差并检查解码及接缝 SSIM；性能不设强制提升门槛。
输入形状、真实输入 fixture 与结果说明见 [双卡 VAE 验证](../docs/vae_spatial_validation_20260929.md)。
`VAEHaloTests` 覆盖 FP32 边界、非默认 stream 和源存储复用；真实权重回归包含 FP32 精度对照及异常恢复。
四卡检查沿用 `ERASERDIT_TEST_TWO_GPU=1`，暴露四张设备，运行 `test_mesh_gpu.py`；
该文件根据可见设备数执行四卡组合。
本轮新增大张量直写回归，使用 `ERASERDIT_TEST_MESH=1` 并暴露四卡启用，
覆盖两组 CFG/SP 通信、非连续与不等长张量、源存储复用及大小张量分支。

这些回归不替代完整视频质量与性能验证；端到端操作见 [测量与验证](../docs/validation.md)，
验收口径见 [performance](../docs/performance.md#acceptance)。

迁移后的专项回归：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 uv run --no-project python -m unittest \
  tests.test_layerwise_offload tests.test_component_offload tests.test_meta_load -v
```

`test_composable_gpu.py` 现在检查不支持的卸载组合在分配副本前拒绝。
旧 extent 测试随已删除后端移除。真实模型结果见 `results/sglang_memory_20260924/`。

完整 DiT 编译专项：`uv run --no-project python -m unittest tests.test_transformer_compile`。
双卡完整 DiT 编译：选择两张空闲卡，设置 `ERASERDIT_TEST_CFG_COMPILE=1`，
运行 `tests.test_transformer_compile.TransformerCompileTests.test_two_gpu_cfg_inductor`。
覆盖两卡 CFG 分支、跨窗口形状复用、编译副卡故障恢复与显式回退。
加 `ERASERDIT_TEST_FULL_COMPILE=1` 并提供 CUDA 设备运行真实 Inductor 形状切换测试。

辅助模型编译专项：`uv run --no-project python -m unittest tests.test_component_compile`；
设置 `ERASERDIT_TEST_COMPONENT_COMPILE=1` 验证 CUDA Inductor 与 VAE 权重 CPU/GPU 往返。

NCCL DiT 专项（自启动独立进程池，至少两卡，四卡增加混合并行覆盖）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ERASERDIT_TEST_DIT_NCCL=1 OMP_NUM_THREADS=1 \
  uv run --no-project python -m unittest tests.test_nccl_dit -v
```

包含不等长 Ulysses/Ring、TP 权重分片、FSDP、混合 USP、父进程组隔离和 worker 故障清理。
输出汇聚检查同时运行全 rank 返回与仅 owner 返回，要求同拓扑结果逐元素一致；
进程池检查 CFG 合并前移后与原 FP32 运算逐元素一致，覆盖 guidance 0/1/7.5、窗口形状切换和返回字节数。
四卡额外运行 `test_process_pool_cfg_sp_guided`，覆盖 CFG2×SP2 的真实进程池路径。
严格 Flash 数值 profile 检查限定 A100 / PyTorch 2.6；跳过不代表其他设备已完成效果验收。
无 GPU 时可设置 `ERASERDIT_TEST_DIT_PROCESSES=1` 运行 Gloo 小模型矩阵。
整片 SSIM 与性能记录见 [NCCL 并行验收](../docs/distributed_parallel_20260928.md)。

VAE 阶段卸载回归：`CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m unittest tests.test_vae_residency -v`。
覆盖编码/解码驻留、精确输出、CPU backing 复用、可变参数/buffer、共享参数、替换注册张量及上传失败恢复。
真实模型单窗口对照和显存限额验证见[内存优化验证](../docs/memory_optimization_20261002.md)。
`tests.test_runtime_cleanup` 在禁用循环 GC 时验证请求和对象链回调能立即释放上下文，包含关闭资源异常路径。

帧转换与输出契约：`uv run --no-project python -m unittest tests.test_frame_conversion tests.test_runtime_boundaries -v`。
覆盖精确数值/布局、只读及反向 NumPy 视图、输入输出存储独立、预热结果释放，以及仅文件输出与原路径视频逐字节一致。

本轮补充：`test_runtime_boundaries` 包含 257 帧多窗口 uint8 流式缓存上界、软/彩色 mask
阈值与 preload 等价性、扫描取消后 reader 关闭；`test_colorfix` 检查分块 FP32 后处理
的 patch/raw tail 精确性。`test_nccl_dit` 增加融合、残差缓存的 CFG/SP 共识及进程池窗口重置。
单卡 VAE/打包专项可运行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 uv run --no-project python -m unittest \
  tests.test_vae_memory tests.test_nccl_packing tests.test_colorfix -v
```

CUDA IPC 边界专项（显式分配两卡或四卡）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ERASERDIT_TEST_DIT_NCCL=1 OMP_NUM_THREADS=1 \
  uv run --no-project python -m unittest \
  tests.test_nccl_dit.DistributedDiTTests.test_cuda_ipc_process_pool \
  tests.test_nccl_dit.DistributedDiTTests.test_cuda_ipc_normal_close \
  tests.test_nccl_dit.DistributedDiTTests.test_cuda_ipc_owner_rank_failure -v
```

覆盖非默认 stream、非连续/变化输入、跨窗缓存及故障/正常关闭后已返回结果的所有权。
对照与限制见[CUDA IPC 边界记录](../docs/cuda_ipc_boundary_20261003.md)。

Ulysses 分块重叠专项：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ERASERDIT_TEST_DIT_NCCL=1 OMP_NUM_THREADS=1 \
  uv run --no-project python -m unittest tests.test_ulysses_overlap tests.test_nccl_packing \
  tests.test_nccl_dit.DistributedDiTTests.test_head_overlap_process_pool -v
```

真实权重筛选：`entrypoints.cli.benchmark_dit --variants fusion_direct,heads2_serial,heads2,heads4`。
`heads2_serial` 仅为基准中的分块串行对照；实际并行与整片收益见[验收记录](../docs/ulysses_overlap_20261003.md)。

Sage FP8 配置与数值检查：`CUDA_VISIBLE_DEVICES=0 ERASERDIT_TEST_SAGE_FP8=1 PYTHONPATH=results/quant_opt_20261003/SageAttention uv run --no-project python -m unittest tests.test_sage_fp8_options`。GPU 部分需要 SM89 和兼容的已构建扩展。

INT8 GELU 融合：`CUDA_VISIBLE_DEVICES=0 ERASERDIT_TEST_INT8=1 OMP_NUM_THREADS=1 uv run --no-project python -m unittest tests.test_quantization`。
包括有限 BF16 位模式、非连续/不整齐输入、模型转换与激活匹配、独立 LUT buffer，以及 CPU 转换后逐层卸载和局部编译的精确输出。

同一量化专项也覆盖 FP8 动态/静态模式：GELU 融合、固定 scale 与饱和、快速累加参考、
空输入/非连续张量、CPU 转换和逐层卸载加局部编译。不同量化模式以误差和视频质量比较。

近似完整 Q/K 融合：`CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m unittest tests.test_qk_fast_fusion tests.test_operator_fusion_precision`。
覆盖误差上限、零输入/多 batch/不同幅度、输入不变、统计归属、契约回退与梯度拒绝；同时保留原精确路径回归。

完整 AdaLN 融合：`CUDA_VISIBLE_DEVICES=0 uv run --no-project python -m unittest tests.test_adaln_fast_fusion tests.test_operator_fusion_precision`。
覆盖零输入与不同幅度、调制张量切片、数值误差、原始输入不变、自动回退/梯度、完整 block 集成和融合计数。
