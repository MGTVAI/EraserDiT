# DiT 逐层卸载

当前使用源码迁移的 SGLang 原生管理器，不依赖 SGLang 包。
实现来源、配置、T5/VAE 管理和统计口径见 [SGLang 内存管理](sglang_memory.md)。

2026-09-23 的字节预算、受限区间预取和 CPU view 恢复属于已移除的实现；
[旧验证记录](layerwise_offload_validation_20260923.md)仅供历史对照。
