<h1 align="center">
  <span style="color:#2196f3;"><b>EraserDiT</b></span>: Fast Video Inpainting with Diffusion Transformer Model
</h1>

<p align="center">
  <a href="https://huggingface.co/jieeliu/EraserDiT"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/JieLiu95/EraserDiT"><img alt="Github" src="https://img.shields.io/badge/EraserDiT-github-black"></a>
  <a href="https://arxiv.org/abs/2506.12853"><img alt="arXiv" src="https://img.shields.io/badge/EraserDiT-arXiv-b31b1b"></a>
  <a href="https://jieliu95.github.io/EraserDiT_demo/"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</p>

## 原始算法：EraserDiT

**EraserDiT: Fast Video Inpainting with Diffusion Transformer Model** 是 Jie Liu 和 Zheng Hui 在芒果tv期间的视频擦除工作，可根据指定区域擦除视频中的物体，并恢复背景内容与时序一致性。
模型、论文与演示来自 [原始算法仓库](https://github.com/JieLiu95/EraserDiT)。

本仓库的推理入口接收视频、掩码和背景提示词，权重沿用原作者发布的 [Hugging Face 模型](https://huggingface.co/jieeliu/EraserDiT)。

## 本仓库：面向算法的推理 infra

本仓库在原始 EraserDiT 算法基础上构建推理加速与服务基础设施，围绕模型加载、流水线、内存管理、算子、缓存和多 GPU 执行进行优化。
实现思路参考 SGLang 的 `python/sglang/multimodal_gen`，可以理解为面向 EraserDiT 的 **mini SGLang 多模态推理运行时**。

所需能力在仓库内实现，无需安装完整 SGLang。这样可以在算法使用的环境中维护兼容的 Torch、Diffusers、Transformers 依赖，减少框架升级对算法迭代的牵制，也便于直接**阅读和修改**。
当前验证的依赖组合见 [requirements.txt](requirements.txt)；更换库版本仍需重新验证。

### 已实现的优化

| 方向 | 实现 | 用途与边界 |
| --- | --- | --- |
| **内存与显存** | 整组件分阶段卸载；DiT 逐层卸载、pinned CPU 权重、独立 CUDA stream 预取与权重预算；阶段边界回收空闲显存缓存 | 降低权重驻留及初始化显存；预算不包含激活、VAE 峰值或 allocator reserved |
| **视频内存** | 按窗口执行与帧缓存释放；可选流式读取、VAE 分块 | 控制长视频缓存或 VAE 激活开销；流式模式和近似 VAE 分块需检查画面差异 |
| **单卡计算** | SDPA / FlashAttention / SageAttention 后端；`torch.compile`；Triton RMSNorm + AdaLN、QK RMSNorm + RoPE 融合；文本投影复用 | 降低注意力、算子和重复计算开销；收益依赖输入、硬件及组合，部分路径有数值差异 |
| **多卡单任务** | CFG 正负分支并行、SP 序列并行、VAE 编解码并行及 CFG × SP 组合 | 降低单视频延迟；单任务 mesh 最多四卡，显存不会按卡数均分 |
| **多卡多任务** | DP dispatcher 将独立视频分配给不同 worker | 提高批量吞吐，worker 内复用已加载模型 |
| **TeaCache / CacheDiT** | 根据步间变化复用 Transformer 残差；按窗口和 CFG 分支隔离缓存；预热、末步保护与连续跳步限制 | 有损加速；默认关闭，两种缓存启用后的默认阈值均为 `0.3` |
| **实验性量化** | DiT Linear 的 INT8 W8A8，支持 blocks / FFN 范围 | 已验证与卸载、编译、CFG2 和 cache_dit 组合；本样例未快于推荐方案 |

详细参数与组合限制见 [性能配置](docs/performance.md)，逐层预取和预算管理见 [内存实现](docs/layerwise_offload.md)。

### 实测结果

> **推荐组合：逐层卸载 + CFG2 + compile + Sage + TeaCache**
>
> 单窗口耗时 **89.87 s**，相对单卡基线加速 **2.37×**，总显存上界 **37.90 GiB**。

| 测试项 | 设置 |
| --- | --- |
| 硬件 | A100 80GB |
| 输入 | 1920 × 1080，121 帧 |
| 推理参数 | BF16，seed = 42，50 步，strength = 0.8；一个窗口执行 40 个去噪步 |
| 基线 | 本仓库原算法单卡复现，关闭缓存、编译和卸载 |
| 计时范围 | 单窗口完整任务，不含模型加载；加速配置排除请求级 warmup |

**性能对比**

双卡配置均开启 FFN 局部编译；SP2 使用 sharded 模式。Sage / FA 分别指 SageAttention / FlashAttention。
INT8 行对 DiT Linear 量化，其余配置使用 BF16。

| 配置 | GPU 数 | 耗时 ↓ | 加速比 ↑ | 总峰值显存 ↓ |
| --- | ---: | ---: | ---: | ---: |
| 原算法复现基线 | 1 | 213.00 s | 1.00× | 69.33 GiB |
| 逐层卸载 | 1 | 215.04 s | 0.99× | 33.64 GiB |
| 逐层卸载 · CFG2 · SDPA | 2 | 131.73 s | 1.62× | 37.37 GiB |
| 逐层卸载 · SP2 · SDPA | 2 | 133.15 s | 1.60× | 36.59 GiB |
| **逐层卸载 · CFG2 · Sage · TeaCache（推荐）** | **2** | **89.87 s** | **2.37×** | **37.90 GiB** |
| 逐层卸载 · SP2 · FA · cache_dit | 2 | 97.28 s | 2.19× | 36.56 GiB |
| 组件阶段卸载 · CFG2 · SDPA | 2 | 127.01 s | 1.68× | 40.50 GiB |
| 组件阶段卸载 · CFG2 · Sage · TeaCache | 2 | 83.45 s | 2.55× | 41.02 GiB |
| 逐层卸载 · CFG2 · INT8 · SDPA · cache_dit | 2 | 85.66 s | 2.49× | 38.07 GiB |

**画面质量**

| 配置 | SSIM ↑ | MSE ↓ | MAE ↓ | 验收结果 |
| --- | ---: | ---: | ---: | --- |
| 表内 SDPA 无缓存、无量化组合 | 1.000000 | 0 | 0 | 三指标通过 |
| **逐层卸载 · CFG2 · Sage · TeaCache** | **0.981816** | **3.917522** | **1.398953** | **允许近似** |
| 逐层卸载 · SP2 · FA · cache_dit | 0.981808 | 3.911658 | 1.389924 | 允许近似 |
| 组件阶段卸载 · CFG2 · Sage · TeaCache | 0.981816 | 3.917522 | 1.398953 | 允许近似 |
| 逐层卸载 · CFG2 · INT8 · SDPA · cache_dit | 0.981175 | 4.414913 | 1.492013 | 允许近似 |

<details>
<summary>测量说明与组合参数</summary>

- **编译**：FFN 局部编译，保留原生 Linear 边界。
- **缓存**：TeaCache / cache_dit 阈值均为 0.3，开启文本投影缓存，逻辑分支步复用率为 45%；cache_dit 命中时仍执行探针块。

</details>


## 快速开始

需要 Linux x86_64、NVIDIA GPU 和支持 CUDA 12.6 的驱动。先确认 `nvidia-smi` 正常。
Python 3.10 和依赖由 [uv](https://docs.astral.sh/uv/) 管理，默认使用无需编译扩展的 SDPA。

### 1. 安装

```bash
sudo apt-get install -y git curl ffmpeg libgl1 libglib2.0-0

git clone https://github.com/MGTVAI/EraserDiT.git
cd EraserDiT
uv venv --python 3.10
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
```

### 2. 下载模型

从 [Hugging Face](https://huggingface.co/jieeliu/EraserDiT) 下载完整模型到 `data/model/`：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
uv run --no-project hf download jieeliu/EraserDiT \
  --revision 904fb412da76235085dbbccaefdbde4979fa3d29 \
  --local-dir data/model \
  --exclude ".DS_Store"
```

### 3. 运行

以下命令使用仓库中的示例视频和掩码，结果写入 `outputs/result.mp4`：

```bash
mkdir -p outputs
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/result.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --attention-backend sdpa \
  --resource-policy dynamic_offload
```

## 常用优化配置

完成上面的单视频运行后，按目标**替换**命令中的资源策略或注意力选项，并追加对应参数。
CLI 默认使用 `dynamic_offload`，可叠加 FFN 局部编译、CFG / SP、缓存及 INT8。

```bash
# 内存优化：DiT 逐层卸载，2 GiB 受管权重预算，向前预取 1 个 block
--resource-policy dynamic_offload --max-weight-usage 2147483648 --dit-offload-prefetch-size 1

# 单卡：SageAttention + 编译（先安装可选注意力依赖）
--resource-policy dynamic_offload --attention-backend sage_attn --enable-torch-compile --warmup

# 单卡：TeaCache，允许有损的残差复用
--resource-policy dynamic_offload --transformer-cache-mode teacache --teacache-threshold 0.3
```

以上是参数片段；[可选依赖安装](docs/setup.md#可选依赖) 和 [缓存调参](docs/performance.md#cache) 见详细文档。
TeaCache 根据调制输入的变化估计是否需要重新计算 Transformer；命中时复用残差。
缓存按窗口与 CFG 分支维护，默认预热 4 步、保护末步、最多连续复用 1 步；阈值越宽松通常越容易复用，也需检查擦除区域和时序稳定性。
可通过 `transformer_cache_history` 查看实际命中，不能仅凭开启开关判断加速是否生效。

双卡推荐组合（需安装 SageAttention，允许缓存近似）：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.cli.erase_eraserdit \
  --model-path data/model \
  --video-input data/113000356.mp4 --mask-input data/113000356_mask.mp4 \
  --output-path outputs/result_cfg2.mp4 \
  --prompt "There is a rooftop terrace overlooking the city at sunset." \
  --attention-backend sage_attn --resource-policy dynamic_offload \
  --cfg-degree 2 --enable-torch-compile \
  --transformer-cache-mode teacache --teacache-threshold 0.3 --cache-text-projections
```

双卡 SP 可将 `--cfg-degree 2` 替换为 `--sp-degree 2`；批量 DP 使用独立的 `erase_parallel` 入口，见 [命令行推理](docs/cli.md#加速与多卡)。
多卡 VAE 仍要求 `fullgpu`；强制手工 Triton 融合不能与 compile 或 INT8 叠加。
双卡 Ring 仅完成小模型验证，CFG × SP 四卡组合未验收。完整约束见 [性能配置](docs/performance.md)。

## 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run --no-project python -m entrypoints.server.serve \
  --pipeline-name EraserDiTErasePipeline --model-path data/model \
  --task-root outputs/service --input-allowed-root "$PWD/data" \
  --host 127.0.0.1 --port 30000
```

启动后访问 `http://127.0.0.1:30000/docs` 查看交互式 API。
上传、查询进度、下载和取消任务见[服务 API](docs/service_api.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [安装与部署](docs/setup.md) | 新机器安装、模型下载、可选依赖、容器和排错 |
| [命令行推理](docs/cli.md) | 单视频、批量任务、参数与多卡入口 |
| [服务 API](docs/service_api.md) | 服务配置、请求、任务和结果 |
| [性能配置](docs/performance.md) | 注意力、编译、卸载、缓存、并行与量化 |
| [配置说明](config/README.md) | 模型、服务与运行配置 |
| [代码架构](docs/architecture.md) | 模块职责、执行流程与依赖边界 |
| [逐层卸载](docs/layerwise_offload.md) | CPU/GPU 权重存储、异步预取与预算管理 |


## 下一步工作

| 方向 | 计划 | 阶段目标 |
| --- | --- | --- |
| **消费级显卡适配** | 优化 VAE 编解码峰值、激活生命周期、窗口缓存和 CPU/GPU 搬运；完善 VAE tiling、流式处理及低显存预设 | 优先支持 24 GiB 显卡稳定完成 1080p 视频，再评估 16 GiB 配置的分辨率、帧数和速度边界 |
| **WebUI** | 提供视频与 mask 上传、画笔编辑、提示词和性能配置、任务进度、预览、取消及结果下载 | 无需编写命令即可完成单视频擦除，并提供低显存、质量优先和速度优先预设 |
| **ComfyUI** | 提供模型加载、视频输入、mask、EraserDiT 推理和视频输出节点，复用常驻模型与现有缓存/并行配置 | 发布可安装的自定义节点和示例工作流，覆盖基础推理与常用加速组合 |
| **单卡性能** | 分阶段分析文本编码、去噪、VAE 和数据搬运；继续优化 attention、局部 compile、算子融合及预取调度 | 在固定质量门槛下改善稳态延迟，并分别报告冷启动、warmup 和正式推理耗时 |
| **多卡性能** | 优化 CFG/SP 的通信与计算重叠、持久化副本和拓扑选择；评估 NCCL 多进程实现及 CFG × SP 四卡组合 | 降低双卡通信开销，完成 2/4 卡吞吐、延迟、显存和故障恢复验证 |
| **TeaCache / CacheDiT 调优** | 按素材、窗口和去噪阶段调整阈值、保护步与连续复用限制；增加擦除区域和时序稳定性指标 | 给出质量—速度 Pareto 曲线、推荐默认值和典型场景预设，减少闪烁与局部结构损失 |
| **FP8 量化** | 实现权重/激活量化、scale 策略、校准与 BF16 回退；针对支持 FP8 Tensor Core 的硬件优化内核 | 完成质量、速度、显存和硬件兼容性验证，并与 BF16、INT8 组合进行同口径对比 |

## 参考仓库

- [EraserDiT](https://github.com/JieLiu95/EraserDiT)：模型与视频擦除算法。
- [SGLang multimodal_gen](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)：多模态推理运行时、服务与逐层卸载设计参考。

## 本仓库开发人员（按贡献度排名）

- [ZhiHeng66](https://github.com/ZhiHeng66)
- [balbalabal](https://github.com/balbalabal)
- [Alwaysssssss](https://github.com/Alwaysssssss)

## 📜 Citation

If you find our work helpful, please consider giving a star 🌟 and citation 📝

```
@article{liu2025eraserdit,
  title={EraserDiT: Fast Video Inpainting with Diffusion Transformer Model},
  author={Liu, Jie and Hui, Zheng},
  journal={arXiv preprint arXiv:2506.12853},
  year={2025}
}
```
