# Patch 交付

`vllm-kev-qwen3-v0.26.0.patch` 是从固定上游基线
`vllm-project/vllm@568afb3a1`（tag `v0.26.0`）到本仓库固定实施 checkout 的
完整 binary-safe diff。应用前应检出该基线并运行：

```bash
git apply --check vllm-kev-qwen3-v0.26.0.patch
git apply vllm-kev-qwen3-v0.26.0.patch
```

补丁 SHA256 和验证结果记录在
[`../task_1_5_8_1_delivery.md`](../task_1_5_8_1_delivery.md)。
