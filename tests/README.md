# 回归测试

使用 `unittest`，保持单层目录，按功能命名。以下命令从仓库根目录执行。
先按[安装说明](../docs/setup.md)创建 uv 环境并安装项目依赖；测试附加依赖安装 `uv pip install --python .venv/bin/python -r requirements-test.txt`，视频 IO 测试需 FFmpeg。
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
| `test_runtime_boundaries.py` | 分布式状态与日志判断、资源策略回退、异步视频 IO 与编码契约 |
| `test_prepost_boundaries.py` | EraserDiT 预/后处理导入契约、patch 行为、通用裁剪独立导入 |
| `test_memory_lifetimes.py` | mask 分批与完整处理的 CPU/GPU 数值一致性、进入 VAE 前视频引用释放 |
| `test_static_condition_reuse.py` | 请求内文本编码缓存失效和隔离、预计算 RoPE 的 CPU/GPU 数值一致性 |
| `test_operator_fusion_precision.py` | QK RoPE、gated residual 舍入一致性，布局/梯度回退，完整 block 与文本缓存、逐层卸载组合 |
| `test_service_api.py` | HTTP 契约、任务和产物；使用 scheduler stub，无权重 |
| `test_component_offload.py` | 组件租约、异常清理；CUDA 可用时附加设备验证 |
| `test_layerwise_offload.py` | 迁移管理器的循环预取、布局、重复推理、T5 FSDP、异常清理；CUDA 用例需 GPU |
| `test_cache.py` | CFG/窗口隔离、探针、FP32 残差、请求契约；部分测试需要 CUDA |
| `test_cfg_parallel.py` | CFG 配置约束与显式双卡小模型检查 |
| `test_mesh.py` | CPU 分组、非整除分片、通信、失败传播及 DP 分配 |
| `test_nccl_dit.py` | 正交拓扑、真实 NCCL Ulysses/Ring/TP/FSDP/混合组、父进程组隔离及 worker 故障清理 |
| `test_mesh_gpu.py` | 显式两/四卡 Transformer、异常恢复和可选真实 VAE |
| `test_quantization.py` | 层覆盖及组合约束，显式 INT8 GPU 验证 |

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
四卡检查沿用 `ERASERDIT_TEST_TWO_GPU=1`，暴露四张设备，运行 `test_mesh_gpu.py`；
该文件根据可见设备数执行四卡组合。
本轮新增大张量直写回归，使用 `ERASERDIT_TEST_MESH=1` 并暴露四卡启用，
覆盖两组 CFG/SP 通信、非连续与不等长张量、源存储复用及大小张量分支。

这些回归不替代完整视频质量与性能验证；端到端操作见 [测量与验证](../docs/validation.md)，
验收口径见 [performance](../docs/performance.md#acceptance)。

迁移后的专项回归：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m unittest \
  tests.test_layerwise_offload tests.test_component_offload tests.test_meta_load -v
```

`test_composable_gpu.py` 现在检查不支持的卸载组合在分配副本前拒绝。
旧 extent 测试随已删除后端移除。真实模型结果见 `results/sglang_memory_20260924/`。

完整 DiT 编译专项：`python -m unittest tests.test_transformer_compile`。
双卡完整 DiT 编译：选择两张空闲卡，设置 `ERASERDIT_TEST_CFG_COMPILE=1`，
运行 `tests.test_transformer_compile.TransformerCompileTests.test_two_gpu_cfg_inductor`。
覆盖两卡 CFG 分支、跨窗口形状复用、编译副卡故障恢复与显式回退。
加 `ERASERDIT_TEST_FULL_COMPILE=1` 并提供 CUDA 设备运行真实 Inductor 形状切换测试。

辅助模型编译专项：`python -m unittest tests.test_component_compile`；
设置 `ERASERDIT_TEST_COMPONENT_COMPILE=1` 验证 CUDA Inductor 与 VAE 权重 CPU/GPU 往返。

NCCL DiT 专项（自启动独立进程池，至少两卡，四卡增加混合并行覆盖）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ERASERDIT_TEST_DIT_NCCL=1 OMP_NUM_THREADS=1 \
  python -m unittest tests.test_nccl_dit -v
```

包含不等长 Ulysses/Ring、TP 权重分片、FSDP、混合 USP、父进程组隔离和 worker 故障清理。
严格 Flash 数值 profile 检查限定 A100 / PyTorch 2.6；跳过不代表其他设备已完成效果验收。
无 GPU 时可设置 `ERASERDIT_TEST_DIT_PROCESSES=1` 运行 Gloo 小模型矩阵。
整片 SSIM 与性能记录见 [NCCL 并行验收](../docs/distributed_parallel_20260928.md)。
