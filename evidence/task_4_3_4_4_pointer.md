# 任务 4.3–4.4：FP32 pointer readout 证据

实现位置：`vllm-upstream/vllm/model_executor/layers/pooler/kev.py`。

实现从 vLLM runtime cursor 取得每条序列在扁平 batch 中的起点，将序列内的选项末尾和决策索引转换为 batch 全局索引。只 gather 这些位置的 hidden states，再转为 FP32 执行带 bias 的 q/k 投影、缩放、温度校准和逐题 softmax。不同题目的选项不会参与同一个 softmax。

固定 hidden states 测试覆盖：

- head bias 与非默认温度 2.3。
- K=1、2、8、255。
- 不同序列长度组成的同一 batch。
- FP32 参考公式 `atol=1e-5, rtol=1e-5`。
- 概率有限、逐题和为 1。
- 缺失、版本错误、空选项和越界 metadata 的拒绝行为。

执行结果：

```text
pytest --confcutdir=tests/model_executor/layers \
  tests/model_executor/layers/test_kev_pooler.py -q
9 passed

ruff check vllm/model_executor/layers/pooler/kev.py \
  tests/model_executor/layers/test_kev_pooler.py
All checks passed!
```

## 数据传输与同步审查

- 候选项索引先在 CPU 侧按请求汇总，再一次性传到模型设备。
- q/k 投影和 softmax 均按 batch 向量化执行，没有候选项级 `.item()`。
- backbone 输出只 gather 选项末尾和决策位置；pooler 返回每题概率张量，不返回完整 hidden states。
- FP32 转换发生在 gather 之后，避免复制整个 token hidden-state batch。
- 当前层不主动传回 CPU；最终结果传输由 vLLM 既有 pooling 输出路径统一处理。
