# 任务 4.1–4.2：输入编排与索引证据

实现位置：`vllm-upstream/vllm/entrypoints/pooling/systemone/`。

验证覆盖：

- `noul`、`choice`、`score` 的严格请求结构和未知字段拒绝。
- 嵌套对象、数组、中文、布尔值、`null` 和非有限数值。
- 用户输入中的伪造 `<|...|>` 控制标记转义。
- 每题独立的 `state + question` 完整因果序列。
- position IDs、选项末尾索引和决策索引。
- question ID 仅参与映射，不进入模型 token 序列。

执行结果：

```text
pytest --confcutdir=tests/entrypoints/pooling/systemone \
  tests/entrypoints/pooling/systemone/test_protocol.py -q
7 passed

ruff check vllm/entrypoints/pooling/systemone \
  tests/entrypoints/pooling/systemone
All checks passed!

ruff format --check vllm/entrypoints/pooling/systemone \
  tests/entrypoints/pooling/systemone
4 files already formatted
```

此外，使用固定模型 revision 下载的真实 Qwen3 tokenizer，将三类请求同时送入固定 KEV `encode`/`rows_of` 和新实现，逐项比较完整 token IDs、position IDs、选项末尾索引与决策索引，结果为 `3/3 question branches exact`。
