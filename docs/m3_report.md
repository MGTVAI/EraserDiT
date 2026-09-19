# M3 加速测量结果

素材 `data/10268234.mp4`（1080×1920，120 帧，单窗口），快速口径
（`ERASERDIT_DETERMINISTIC=0`），每配置 5 次重复取中位数。N = `sdpa`
（新架构未加速）。复现：`python3 scripts/m3_report.py --results-dir
results/m3 results/m3-clean results/m3-warmup`。

| 配置 | 有效后端 | t_e2e | Δe2e | t_denoise | Δdenoise | t_step_med | t_warmup | peak_resv | SSIM | PSNR | 门槛 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **N** `sdpa` | torch_sdpa | 216.4 | — | 178.4 | — | 4483 | 0 | 53.37 | 1 | ∞ | — |
| `flash_attn` | flash_attn | 214.3 | +1.0% | 176.9 | +0.9% | 4443 | 0 | 53.37 | .9905 | 41.01 | fail |
| `sage_attn` | sage_attn | 208.8 | +3.5% | 175.9 | +1.4% | 4390 | 0 | 53.37 | .9909 | 41.38 | fail |
| `auto` | sage_attn | 198.3 | +8.4% | 175.8 | +1.5% | 4388 | 0 | 53.37 | .9909 | 41.38 | fail |
| `compile_default` | torch_sdpa | 207.7 | +4.0% | 167.9 | +5.9% | 3910 | 0 | 53.37 | .9909 | 41.32 | fail |
| `compile` | torch_sdpa | 190.1 | +12.2% | 166.8 | +6.5% | 3888 | 0 | 53.37 | .9909 | 41.28 | fail |
| `sage_compile_default` | sage_attn | 194.0 | +10.4% | 161.0 | +9.8% | 3723 | 0 | 53.37 | .9908 | 41.33 | fail |
| `fusion_qk` | torch_sdpa | 197.0 | +9.0% | 160.6 | +10.0% | 4016 | 0 | 53.37 | .9910 | 41.47 | fail |
| `sage_compile` | sage_attn | 198.8 | +8.1% | 159.8 | +10.5% | 3707 | 0 | 53.37 | .9909 | 41.30 | fail |
| `fusion_all` | torch_sdpa | 196.8 | +9.1% | 158.9 | +11.0% | 3974 | 0 | 53.37 | .9910 | 41.47 | fail |
| `fusion_qk_sage` | sage_attn | 193.5 | +10.6% | 157.1 | +11.9% | 3926 | 0 | 53.37 | .9910 | 41.47 | fail |
| `fusion_all_sage` | sage_attn | 194.2 | +10.3% | 155.1 | +13.1% | 3879 | 0 | 53.37 | .9909 | 41.42 | fail |
| **`sage_compile_warmup`** | sage_attn | **173.4** | **+19.9%** | **148.0** | **+17.0%** | 3731 | 51.2 | 53.37 | .9909 | 41.30 | **PASS** |

门槛：Δdenoise ≥ 15%、Δe2e ≥ 10%、peak_reserved ≤ N×1.05、SSIM ≥ 0.95、
PSNR ≥ 28 dB、5/5 次重复全过。全部 5 次重复 0 失败。

## 推荐配置

```
--attention-backend sage_attn --enable-torch-compile --warmup
```

`auto` 在 A100 (sm80) 上实际选中 `sage_attn`，两者等价。

## 结论

- **注意力后端本身几乎无收益**（sage +1.4%、flash +0.9%），瓶颈在 MLP/线性层，
  所以 `torch.compile` 是主力杠杆、融合次之。
- **`rmsnorm_adaln` 与 `qk_rmsnorm_rope` 两个融合都赚**：sage 后端下
  `fusion_all_sage` (+13.1%) > `fusion_qk_sage` (+11.9%)。但单项融合都不足以
  过 15% 线，且与编译互斥，不进推荐配置。
- **`reduce-overhead` 编译模式否决**：`fullgraph=False` 下 CUDA graph 反复
  捕获，比不编译更慢（denoise +11.3% 耗时，即 −11.3% 收益）。
- **编译项达标依赖预热**：Inductor 自动调优 51.2 s 落在预热请求里。plan 的
  编译项写的就是 `torch.compile + 预热`，此前 EraserDiT 的 CLI 没接出这个
  开关（`--warmup`），调优因此落进正式任务，`sage_compile` 的 `t_denoise`
  159.8 s 里有 11.5 s 是这笔开销，显示成 +10.5% 而不过线。

### 口径说明

`t_e2e` 不含预热（`e2e_seconds_excluding_warmup`），预热单列在 `t_warmup`。
若把预热计入首次任务，该次 e2e 为 224.7 s，比 N 差 3.8%——**代价真实存在，
只是每进程只付一次**。因此推荐配置的前提是常驻 pipeline（多任务），单任务
冷启动场景不适用。

## 组合限制

- `torch.compile` 与 `operator_fusion_backend=triton` **互斥**，显式组合报错
  （`ValueError`，已验证）；`operator_fusion_backend=auto` 降为空融合并以
  `torch_compile_active` 作为回退原因。
- `sage_fp8` 及量化侧 FP8 全系要求 sm89，A100 (sm80) 探针拒绝，不得进入
  任何推荐配置；`int8_w8a8_viditq` 依赖 `/root/viditq`（本机不存在）。
- 显存零代价：全部配置 `peak_reserved` = 53.37 GiB = N。
