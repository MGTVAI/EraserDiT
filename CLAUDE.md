# 项目约定

- 文档保持简洁，描述当前功能和使用方式。
- Python 3.10 与依赖使用 uv 管理；安装和模型下载见 `docs/setup.md`。
- 从仓库根目录使用 `uv run --no-project python` 运行入口和测试。
- 模型放在 `data/model/`；示例视频为 `data/113000356.mp4`，掩码为 `data/113000356_mask.mp4`。
- 完整权重下载后可设 `HF_HUB_OFFLINE=1`。
- 模块职责见 `docs/architecture.md`，验证方式见 `tests/README.md` 和 `docs/validation.md`。
- 运行长任务时定期检查完成状态。
