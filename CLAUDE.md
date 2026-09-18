
# 输出的文档保持简洁

- 参考原始代码：worktree `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT-ref`（detached 在 9944867），只读。
- 运行基线：worktree `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/EraserDiT-baseline`（分支 `baseline-run`），承载基线冻结的三处外挂改动（固定随机源、常驻 pipeline、确定性口径；后者用 `ERASERDIT_DETERMINISTIC=0` 关闭，对照工具为同目录 `compare_outputs.py`）。
- 跑基线前设 `HF_HUB_OFFLINE=1`，否则加载会卡在直连 huggingface.co 上。


## 运行
- 当代码开始运行时，定时检查代码是否运行完成，不要一直消耗token