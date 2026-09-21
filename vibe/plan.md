# 开发方案

配套 `requirements.md`（做什么）与 `environment.md`（环境实测）。本文只写怎么做。

## 1. 已核查的关键事实

以下六条决定了方案形态，均已对照源码或实测确认。

**分段算术同源。** MGErase `utils/windowing.py:131 build_window_specs` 在
`infer_len=121, overlap=9` 下的窗口规划与 EraserDiT 的 `TEMP_INFER_LEN=121 / shift_alpha=9`
（`utils/pre.py:13,44`）对两组素材逐窗一致：120 帧 → 1 窗、提交 120 帧；145 帧 → 首窗装载并
提交 121 帧，次窗装载 `[112,145)`、提交 24 帧。两边都是首窗提交 `min(121, 素材帧数)`、其后
每窗 112 帧，`load_start` 步长 `112`。
（`time_sample=8, time_shift=1` 不参与该规划，只影响已关闭的 `use_dynamic_num_frames`。）

重叠语义一致：两边都用**上一窗口生成的尾帧**作为下一窗口前缀（原版 `inference.py:102` 的
`pre_video_shift` ↔ MGErase 的 `overlap_cache`），且重叠区像素**保留上一窗口的生成结果**——
对应 MGErase `overlap_fuse_mode="before"`，也正是其配置默认值；用 `"after"` 会改用本窗口
重生成的帧，与原版不符。该等价依赖「非末窗无补齐」：窗口一旦被补齐，原版的 `pre_video_shift`
会包含补齐帧的生成结果，而 MGErase 从提交区取缓存。两组素材（单窗 / 首窗满 121 帧 + 末窗）
都满足，≥3 窗素材需另行处理。

末窗补齐规则两边不同：原版取 `flip(0)[1:-1]`（从倒数第 2 帧开始镜像、两端点不重复），
MGErase 为末帧重复。按原版实现，替换 `pad_single_window`；注意该函数同时补视频与掩码。

**掩码时间压缩同源、空域预处理不同。** MGErase `utils/mask.py:156 encode_mask` 打开
`return_one_channel` 后是「latent0 = 第 0 帧，latent j = 帧 8j-7..8j」，即 LTX 因果压缩
（分组大小 8 由 VAE 时间压缩比决定，`return_one_channel=8` 只作真值判断），与 EraserDiT
非首段 `shift_n_frame=0` 压缩 112 帧再前置 2 个零 latent 等价：2 个零 latent 覆盖模型输入的
第 0..8 帧，与 MGErase 把 9 帧前缀掩码置零等价，其后 latent 一一对齐。

但空域预处理必须按原版复刻（`utils/pre.py:243-310`）：`0.299/255`、`0.587/255`、`0.0004`
三通道灰度核（蓝通道系数**未除 255**），`ksize=(9,9)` 十字核近似膨胀 9 次，阈值 `0.039`。
**掩码开值不是 1.0**：wrap/left 帧是 0/255 二值，经灰度核后约 `0.988`；首 latent 帧直接取
膨胀后的 uint8 0/1，仅约 `0.0039`——两者必须照抄。这些是模型适配器专属参数，不得沿用
MGErase 的 `0.3×max` 二值化。

**形状唯一。** 两组素材按 32 对齐后都是 `1088×1920` → latent `16×34×60` →
序列长度都是 **32640**。单一静态形状，`torch.compile` 可一次覆盖；且正好命中 MGErase
TeaCache 系数表已有的 32640 条目。

**速度瓶颈不在注意力。** 每步是 CFG 拆开的两次 batch=1 前向，每次约 336 TFLOP
（自注意力 244 T、GEMM 92 T）。实测 4.5 s/步 ⇒ 约 150 TFLOPS ≈ A100 bf16 峰值 48%。
剩余成本主要是 RoPE/转置/逐元素与 kernel 调度。**收益预期：去噪 1.3–1.8×，端到端
1.15–1.4×**（VAE 编解码与文本编码占比不低；第二段 24 帧真内容补到 121 帧是固有浪费）。
需求不预设倍数，此区间仅作判断基准与退出条件。

**非擦除区不回贴源像素。** `utils/post_pkg.py:94 torch_nhwc_to_video_stream` 全程只调用
`adaptive_instance_normalization_mask`（整帧逐通道 AdaIN 调色），**没有任何
`orig*(1-mask) + gen*mask` 的合成**。非擦除区的像素同样是模型重生成的，只是被调色对齐。
实测（`10268234`）：基线输出 vs 源视频在非擦除区 SSIM 0.9259–0.9434 / PSNR 27.3–27.8 dB，
整帧 SSIM 0.8754 / PSNR 22.8 dB——非擦除区虽然没被回贴，但已被整条生成链改写。
因此**非擦除区 SSIM 是对整条生成链的极敏感判据**，任何 kernel 差异都会显形。

**原版不可逐帧复现，且这个噪声下限贴着 0.99。** 同 seed、同进程、同常驻 pipeline、同代码的
四轮基线运行（`10268234`：10:42 / 10:48 / 10:55 / 11:00）实测。口径固定为
`compare_outputs.py`（ffmpeg `ssim`/`psnr` 滤镜，Y 平面，含编码噪声），非擦除区 = mask 并集
包围盒之外的左右条带（本素材包围盒 `x∈[116,906)`，左条带仅 116 px）：

| 对比 | 非擦除区（左 / 右条带） | 整帧 |
| --- | --- | --- |
| 基线 A vs B（10:42 / 10:48） | SSIM 0.9901 / 0.9889，PSNR 38.90 / 38.67 dB | SSIM 0.9881 / PSNR 38.76 dB |
| 基线 A vs C（10:42 / 10:55） | SSIM 0.9901 / 0.9891，PSNR 39.02 / 38.98 dB | SSIM 0.9879 / PSNR 39.08 dB |
| 基线 C vs D（10:55 / 11:00） | 逐字节相同 | 逐字节相同 |
| 基线 A vs 源视频 | SSIM 0.9434 / 0.9259，PSNR 27.28 / 27.80 dB | SSIM 0.8754 / PSNR 22.80 dB |

四轮落成三个不同结果（{A}、{B}、{C=D}），任意两轮的整帧差异都在同一个量级；换「逐像素
非擦除」口径结果同量级，故不是区域定义造成的。**旧记录里的非擦除区 0.974 / 37.3 dB 与源视频
一行用当前工具无法复现，已按实测替换。**

但**同 seed 的可复现性并不总是失败**：C 与 D 逐字节一致，第二组四份输出也全部逐字节一致
（见 `environment.md` 运行记录）。差异与种子、参数无关（`run_batch` 每次重新播种），只能来自
环境相关分支（最可能为 bf16 矩阵乘 / SDPA 的归约顺序随占用状态改变），因此**它是可以被口径
固定掉的**——这正是 M0 已经做完的事（见下）。

**结论：非擦除区 SSIM ≥ 0.99 只比基线自洽度 0.9889–0.9901 高一点点，PSNR ≥ 40 dB 也高于实测
38.7–39.0 dB，在旧口径下任何实现（含原版对原版）都不通过。** 这正是 M0 必须先把随机性口径
统一掉的原因；统一后基线自身逐字节可复现（见 M0）。

## 2. 方案选择（已定）

| 项 | 选择 | 连带约束 |
| --- | --- | --- |
| 窗口执行层 | 复用 MGErase windowed streaming | 吃下 `commit_ops` 契约；末窗补齐按原版替入；重叠融合保持默认 `"before"`；以整帧 bbox 提交 |
| 等价门禁 | 非擦除区 SSIM ≥ 0.99 + 逐帧目视 | 随机性口径已由 M0 统一（基线逐字节可复现），门禁成立；掩码预处理与末窗补齐仍按原版复刻 |
| VAE 治理 | 纳入第一阶段 | 做**独立可选路径、默认关闭**，需通过差异验证才可用于正式测量 |
| 验证节奏 | 严格按阶段顺序 | 阶段一架构中预留三项加速接入点，阶段二不返工 |

## 3. 目录

`MGErase/python/` 内容平铺到仓库根，不保留 `python/`、`runtime/` 容器（映射规则
`python.<module>` → `<module>`、`python.runtime.<module>` → `<module>`；同步改静态 import、
动态注册路径、脚本与文档，并保留来源与许可证说明）；`runtime/resource/` 与 `layers/memory/`
合并为根 `memory/`。

```
config/ entrypoints/{cli,server}/ layers/{attention,operator_fusion,quantization,rotary_embedding}/
loader/component_loaders/ models/{dits,schedulers,text_encoders,vaes}/ nodes/{preprocess,executors,stages}/
pipelines/ memory/{adapters,backends,policies}/ cache/ distributed/ parallel/
utils/ data/ docs/ docker/ vibe/ scripts/
inference_cli.sh  inference_server.sh
```
`layers/`、`models/`、`nodes/` 下另有平铺模块（`linear.py`、`layernorm.py`、`mlp.py`、
`registry.py`、`schedule_batch.py` 等），`entrypoints/` 顶层还有 `http_server.py` 与
`erase_runner.py`。第一阶段 `memory/` 只落三职责边界与策略接口，具体驻留策略待实测可用
显存后确定并记录；常驻/卸载实现可复用原版已随仓库带入的 group offloading 与 layerwise
casting（`models/hooks/`、`models/modeling_utils.py`）。

## 4. 模型适配器边界

服务骨架与管线不写死模型；适配器提供：

1. **组件声明**——显式声明 `transformer/vae/text_encoder/tokenizer/scheduler` 五个子目录与类名，
   不要求快照含 `model_index.json`（`EraserDiT` 快照正缺该文件，见 `environment.md`）。
2. **请求参数 schema 与采样参数 builder**——默认值与字段集按 EraserDiT 原版算法定义。
3. **capability 标识**——供 `/v1/models`、`/model_info` 使用。
4. **窗口算术与补齐规则**——`infer_len=121 / overlap=9 / stride=112`、末窗镜像补齐。
5. **掩码预处理参数**——灰度核、十字膨胀、阈值、时间压缩，以及两个容易漏掉的语义：
   ① 送 VAE 编码的视频是**掩码区置零**后的 `video*(1-mask)`，掩码另作一路通道；
   ② 掩码降到 latent 分辨率用 `interpolate(..., mode='nearest')`（原版默认），
   不是 MGErase 的 avg-pool + 二值化。
6. **阶段工厂**——按需装配 `validate → text_encode → preprocess → latent_prep → timestep_prep
   → denoise → decode → window_postprocess → commit`。
7. **提交契约**——MGErase 的模型只生成 bbox 内 patch 再回贴源帧，EraserDiT 生成整帧且非擦除区
   不做任何合成（`utils/post_pkg.py:94` 只做 AdaIN 调色），因此适配器须声明**整帧 bbox**
   （`x=0, y=0, w=W, h=H`），让裁剪/回贴退化为恒等，`overlap_cache` 因而等于本窗生成帧。
   `crop_flag=True` 的高分辨率裁剪路径在原版即已损坏（`inference.py:52` 少传 `inference_idx`），
   第一阶段不支持，显式报错。
8. **随机性契约**——每任务一个 `torch.Generator(device='cuda')`、播种一次、窗口之间不重播；
   逐窗取数顺序固定为 (a) VAE encode 的 `latent_dist.sample(generator)`（encode 是**随机**的）、
   (b) 初始 latent 噪声 `randn_tensor`、(c) 解码噪声 `randn_tensor`（`decode_noise_scale=0`
   数值上无效，但**仍消耗随机数**；解码走 `timestep_conditioning=true` 分支，`temb` 由
   `decode_timestep=0.0` 决定）。顺序或次数不一致即等价性失败。
9. **加载与 dtype 契约**——`text_encoder` 直接以 bf16 加载，`vae`/`transformer` 先按 fp32 加载，
   再由 `pipeline.to(device, bf16)` 统一转 bf16 并常驻 GPU；原加载标志
   （`use_safetensor=True, low_cpu_mem_usage=False`）一并保留。
10. **输出写出契约**——libx264 / yuv420p / 码率取输入视频流 `bit_rate//1000000` M / 帧率取视频流
    而非 mask 流；首窗写出 `min(121, 本窗真实帧数)` 帧，其后每窗写出 `output[9:]` 并截断到本窗
    真实帧数。输入解码保持 decord 口径（`decord.cpu(0)` + `dlpack`），不换 ffmpeg/OpenCV。

## 5. 阶段与里程碑

**M0 基线冻结补齐（已完成，2026-09-18）。** 两组素材各 4 轮基线已跑通；确定性口径已作为第 3 类
最小外挂改动并入冻结副本（`inference.py` 注入 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、
`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`，
由 `ERASERDIT_DETERMINISTIC` 控制、默认开，`=0` 走快速口径；不触碰算法与采样参数）。

实测：确定性口径下两组素材各自四轮输出**逐字节相同**，且与未施加开关时偶然自洽的那批输出也
逐字节相同——开关是把计算钉在既有的数值路径上，不改变结果。**基线自洽下限 = 逐字节相同，
「非擦除区 SSIM ≥ 0.99」门禁成立，无需下调。** 冻结记录（提交号、工作区差异、快照标识、提示词、
全部采样参数、环境版本、两组基准视频、两套口径耗时差）见 `environment.md` 的
「基线运行记录」与「M0 确定性口径」两节。

**M1 新架构骨架 + 算法接入（未加速）。** 目录迁移与旧入口删除；模型适配器；
EraserDiT 算法按 windowed 执行落地；`inference_cli.sh`（单任务参数 / JSON 任务文件，
顺序执行，结果按任务 id 落盘，任务结构与服务端一致）。模型实例归常驻会话所有，
但随机数状态、采样状态、读写资源、衔接帧与缓存**按任务隔离**，正常结束与异常退出都要释放；
预热使用独立任务状态，不得污染正式任务的随机数或输出。

**M1b 等价性验证。** 新架构未加速版 vs 冻结基线，同权重/输入/提示词/种子/采样参数/尺寸/分段，
**两侧使用同一套确定性口径**。非擦除区取 mask 并集包围盒之外的最大矩形（左/右条带；
第一组包围盒 `x∈[116,906)`、第二组 `x∈[648,1521)`），逐帧计算 SSIM 与 PSNR。
门禁：**非擦除区 SSIM ≥ 0.99**、整帧 PSNR 作辅助记录，外加 mask 内与边缘逐帧目视，
无擦除失败 / 目标残留 / 画面损坏 / 新增时序退化。确定性口径下基线自身是逐字节可复现的，
因此新架构任何一处 kernel 差异都会直接暴露在非擦除区指标上。

**M1c VAE 低显存路径（可选开关，默认关）。** 只开时间方向分块
（`use_framewise_decoding=True`：解码走 `_temporal_tiled_decode`、**编码也会改走
`_temporal_tiled_encode`**，都取 `tile_sample_min_num_frames=16` 配 stride 8、用 `blend_t`
重叠混合）；空间 tiling（`enable_tiling` → `tiled_encode/tiled_decode`）接缝风险高，第一阶段不动。
注意 `use_slicing` 只按 batch 切分，本流程 batch=1，对显存没有帮助。
单独做「分块 vs 一次性」差异验证：mask 内 / mask 边缘 / 非擦除区分区统计 + 逐帧目视，
无接缝方可进入正式测量。

**M2 服务端。** 沿用 MGErase 协议骨架、任务生命周期与服务组件划分：11 个端点、
`queued/running/completed/failed/cancelled` × `queued/preparing/processing/finalizing/terminal`、
统一错误体 `{"error":{"code","message"}}`（失败任务额外带 `phase`）、拒绝未声明字段、
常驻工作组持模型实例并提供健康心跳。第一阶段仅本地路径提交（受输入白名单约束）与本地结果存储。
`server_info` 必须含实际生效的加速后端、融合与编译开关，以及 `auto` 回退后的实际选择及原因。
**这是对 MGErase 的增量**：它现在只回显启动时的原始 CLI 参数，实际生效值
（`AttentionBackendSelection`、`OperatorFusionDecision`）在每次请求的管线 preflight 里才算出来，
需要显式接到启动配置或首个任务后的状态上报。

**M3 单卡加速三项 + 组合限制矩阵。** 按需求顺序实现注意力后端、算子融合、Transformer 编译，
逐项与组合测量。已知可用性与约束：
- 注意力：`auto | sdpa | flash_attn | sage_attn | sage_fp8`。自注意力可走 flash/sage；
  **交叉注意力无条件走 PyTorch SDPA**（实现里直接指定，不参与探针与回退；flash/sage 探针也确实
  拒绝 mask）。`auto` 的顺序是 **sage → flash → sdpa**，A100 上 head_size 64 / bf16 / sm80 全部
  满足，会实际选中 `sage_attn`，矩阵里必须单列 `auto` 的实际选择。显式指定不可用后端要报错，
  不得静默回退。
- 融合：`qk_rmsnorm_rope` 要求 Q/K 隐宽 2048、q/k eps 同为 1e-5、bf16、无 bias、
  **norm 带 weight**、`cos/sin` 为 `[B,S,2048]` fp32；`rmsnorm_adaln` 要求隐宽 2048、eps 1e-6、
  bf16、**norm 无 weight 无 bias**、batch=1 且 scale/shift 恰为 `(1,1,2048)`（CFG 拆成两次
  batch=1 前向正好满足）。EraserDiT 全部满足：`Attention` 默认 `eps=1e-5`、`rms_norm_across_heads`
  建出的 Q/K norm 带 `(2048,)` weight；block 的 `norm1/norm2` 为 `elementwise_affine=False`、
  eps 1e-6。`LTXVideoTransformerBlock` 的 `norm1/attn1/norm2/attn2/ff/scale_shift_table` 属性名
  与 6 路 unbind 顺序与 MGErase 目标逐字相同；只有 `rmsnorm_adaln` 需要重写 block forward，
  `qk_rmsnorm_rope` 挂在注意力处理器内。
- 编译：`torch.compile(transformer, mode="max-autotune-no-cudagraphs", fullgraph=False)` + 预热。
  与 `operator_fusion_backend=triton` **互斥**：显式组合直接报错，`auto` 则降为空融合并把
  `torch_compile_active` 作为回退原因——两种都要进矩阵。与缓存同样互斥。`fullgraph=False` 会
  吞掉图断裂，验收必须确认实际执行的是编译后的前向，不能只看包装函数有返回。
- **硬件不可用路径**：FP8 全系要求 sm89——注意力侧 `sage_fp8`、量化侧 `fp8_w8a8` 与
  `fp8_w8a8_triton_selective`，A100 均不可用，不得宣称通过验收。INT8 只有量化侧
  `int8_w8a8_viditq` 一项（要求 sm ≥ 8.0，A100 满足；依赖 `/root/viditq` 内核与
  `2048 ≤ in_features ≤ 8192` 形状约束，且本机该目录不存在），属第三阶段，补齐后再逐项实测。

**M4 验收与交付。** 两组素材的原版 / 加速版 / 并排对比视频；性能与差异报告；
连续任务验收脚本与任务文件样例；推荐加速配置及组合限制；重写 `docker/base.dockerfile`
（现有文件是模板残留，引用 `Acceptance/`、`.venv/bin/torchrun` 等本仓库不存在的东西）
并单独验证构建与启动。

## 6. 验收数据

### 6.1 报告矩阵

每个配置 × 每组素材报告同一组字段，缺一不可。配置：`B` 冻结基线 / `N` 新架构未加速 /
`A` 加速版（含单项与组合）。

| 类别 | 字段 |
| --- | --- |
| 时间 | `t_load`（单独列，不计入端到端）、`t_e2e`、`t_denoise`、`t_step_med`（每步中位数）、`t_vae`、`t_text`、`t_io` |
| 显存 | `peak_allocated_gib`、`peak_reserved_gib` |
| 首次开销 | `t_compile`、`t_warmup`、未覆盖形状的额外耗时（计入该次任务并单列） |
| 质量 | 非擦除区 SSIM / PSNR、整帧 SSIM / PSNR、mask 内与边缘目视结论 |
| 条件 | 实际 GPU 编号、同卡占用状态、环境版本、权重快照标识、提示词、种子、全部采样参数、随机性口径 |

质量指标口径固定为 `EraserDiT-baseline/compare_outputs.py`（ffmpeg `ssim`/`psnr` 滤镜，Y 平面，
含编码噪声），区域定义见 M1b；换口径必须同时给出新旧两套数。加载一律用 `HF_HUB_OFFLINE=1`。

重复：每个配置 ≥ 5 次，报中位数与离散度。预热覆盖两组素材尺寸及正段/尾段形状，且使用独立任务
状态，不污染正式任务的随机数与输出。计时必须处理 GPU 异步执行（`torch.cuda.synchronize`），
正式测量关闭详细诊断开销；两套随机性口径的耗时差单独列出。

### 6.2 锚点

**实测（共享卡，同卡有其他租户，仅作上界；行程见 `environment.md` 运行记录）**

| 素材 | 分段/步数 | t_e2e | 峰值 |
| --- | --- | --- | --- |
| `10268234` | 1 段 / 40 步 | 确定性口径 337.0 s（4 轮中位）；快速口径 318.0 s（3 轮中位） | 42.78 GiB alloc / 60.0 GiB resv |
| `113000356` | 2 段 / 80 步 | 确定性口径 638.1 s（4 轮中位）；快速口径 675.6 s（此前 2 轮中位，未与确定性口径同批测） | 44.19 GiB alloc / 60.01 GiB resv |

加载 7.5 s（页缓存热，需 `HF_HUB_OFFLINE=1`）。每步约 4.5 s（确定性口径 4.6 s），为共享卡观测值，
独占窗口待测。加速收益一律以同批同口径的 `N` 为基准。
基线自洽（非确定性口径，四轮三个结果）：非擦除区 SSIM 0.9889–0.9901 / PSNR 38.67–39.02 dB，
整帧 0.9879–0.9881 / 38.76–39.08 dB。基线 vs 源：非擦除区 0.9259–0.9434 / 27.28–27.80 dB，
整帧 0.8754 / 22.80 dB。

**推导（独占窗口，按每步 672 TFLOP、60% MFU 估；M0 未取得独占窗口，仍待替换）**

| 素材 | 每步 | t_denoise（B） | 非去噪 | t_e2e（B） | t_e2e（A，去噪 1.5×） |
| --- | --- | --- | --- | --- | --- |
| `10268234` | 3.0–3.8 s | 120–152 s | ≈ 150 s | 270–300 s | 220–250 s（≈ 1.20×） |
| `113000356` | 3.0–3.8 s | 240–304 s | ≈ 220 s | 460–525 s | 370–425 s（≈ 1.23×） |

实测（共享卡）比推导高约 1.3–1.5×，差距来自同卡占用而非算法；对照基准只能取独占窗口的实测值。

### 6.3 合格线

| 类别 | 指标 | 合格线 | 依据 |
| --- | --- | --- | --- |
| 基线自洽 M0 | 确定性口径下同素材重复 ≥ 4 次 | **已达成**：两组各四轮输出逐字节相同（见 `environment.md`） | 决定 M1b 门禁是否成立 |
| 等价性 M1b | 非擦除区 SSIM | ≥ 0.99 | 需求指定 |
| | 非擦除区 PSNR | ≥ 40 dB | 确定性口径下应显著优于基线自洽 38.9 dB |
| | 整帧 SSIM / PSNR | ≥ 0.995 / ≥ 40 dB | 辅助项，但必须高于基线自洽 0.9881 / 38.76 dB，否则无区分度 |
| | 目视 | 无擦除失败 / 目标残留 / 画面损坏 / 时序退化 | 需求 |
| 稳定性 M1 | `t_e2e(N)` / `t_e2e(B)` | ≤ 1.15 | 架构重构不得引入系统性变慢 |
| | `peak_reserved(N)` | ≤ 基线 × 1.10 | |
| 收益 M3 | `t_denoise` 中位数 | 较 N 改善 ≥ 15%，且 5/5 次一致 | "可重复的实际收益"的操作化 |
| | `t_e2e` 中位数 | 较 N 改善 ≥ 10% | |
| | `peak_reserved(A)` | ≤ `peak_reserved(N)` × 1.05 | 加速不得以显存换速度 |
| 加速版质量 | 整帧 SSIM / PSNR vs N | ≥ 0.95 / ≥ 28 dB | 需求允许颜色、纹理、生成细节差异 |
| | 目视 | mask 内、mask 边缘、完整视频均无擦除失败 / 残留 / 损坏 / 时序退化 | 需求 |
| 组合矩阵 | FP8 全系（`sage_fp8`、`fp8_w8a8*`） | 不得出现在任何推荐配置 | A100 sm_80 探针拒绝 |
| | 重复性 | 每配置 5 次重复全部通过质量线 | |

不达标处理：单项未过收益线则不进推荐组合；组合未过则如实报告并给出降级配置。
加速倍数一律以 N 为基准，不以 B 为基准（B 与 N 的差异属于架构重构，不属于加速）。

### 6.4 前置条件

- **独占（或同卡无大任务）窗口。** 原版峰值 reserved 60.0 GiB、allocated 42.8–44.2 GiB
  （两组素材一致）；当前 8 卡空闲上限 62673 MiB（GPU 2、GPU 3），GPU 7 为 33677 MiB，
  其余卡不足 25 GiB。正式测量前必须协调窗口，否则只能出标注占用状态的相对对比。
- **两套随机性口径。** 等价性验证在确定性口径下进行，性能测量在快速口径下进行；两者耗时差单独
  量化并列出，不得混入加速倍数。

## 7. 风险

| 风险 | 处置 |
| --- | --- |
| 基线可复现性不稳定（同 seed、同代码下曾出现不一致） | **已解决**：M0 注入确定性口径后两组各四轮逐字节相同；正式测量仍用快速口径，两口径耗时差单列（约 +6%） |
| ≥3 窗素材的前缀语义缺口（原版前缀含补齐帧的生成结果） | 两组素材不触发；接入更长素材前先对齐 `pre_video_shift` 与 `overlap_cache` 的取值来源 |
| `docker/base.dockerfile` 为模板残留（引用 `Acceptance/`、`.venv/torchrun`，与本仓库入口无关） | M4 按新入口重写后再验证构建与启动 |
| VAE 时间分块接缝（`use_framewise_decoding`，`blend_t` 混合） | 默认关闭；先做分区差异验证再启用；空间方向不动 |
| 加速收益不足（估计去噪 1.3–1.8×） | M3 后设检查点；若组合收益不可重复，按需求"不预设倍数"如实报告并给出降级配置 |
| 独占窗口不可得 | 提前协调；M4 前若仍不可得，先交付标注占用状态的相对对比并说明 |
| 末窗补齐/掩码预处理复刻偏差 | 按原版实现，成本低；由 M1b 门禁兜住 |
| `113000356` 帧率元数据不一致（视频 24000/1001、mask 1199/50） | 保留原版帧对应与输出帧率处理，不擅自重采样 |
