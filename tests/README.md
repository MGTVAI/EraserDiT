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
| `test_architecture.py` | 单向依赖边界、取消机制独立导入 |
| `test_execution_control.py` | 本地取消、对端取消传播、服务端兼容符号 |
| `test_assembly_contracts.py` | 两种模型契约选择、并行计划类型兼容及默认策略 |
| `test_stage_compatibility.py` | stage 导入顺序、模块身份、patch 与装配契约 |
| `test_parallel_compatibility.py` | 通用并行/算子独立导入、模型适配模块身份及 patch 契约 |
| `test_runtime_boundaries.py` | 分布式状态与日志判断、资源策略回退、异步视频 IO 与编码契约 |
| `test_prepost_boundaries.py` | 两模型预/后处理导入契约、patch 行为、通用裁剪独立导入及局部输出契约 |
| `test_service_api.py` | HTTP 契约、任务和产物；使用 scheduler stub，无权重 |
| `test_component_offload.py` | 组件租约、异常清理；CUDA 可用时附加设备验证 |
| `test_dynamic_offload.py` | 事件、预算、搬运回滚与恢复；部分测试需要 CUDA |
| `test_cache.py` | CFG/窗口隔离、探针、FP32 残差、请求契约；部分测试需要 CUDA |
| `test_cfg_parallel.py` | CFG 配置约束与显式双卡小模型检查 |
| `test_mesh.py` | CPU 分组、非整除分片、通信、失败传播及 DP 分配 |
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

这些回归不替代完整视频质量与性能验证；端到端操作见 [测量与验证](../docs/validation.md)，
验收口径见 [performance](../docs/performance.md#acceptance)。
