# 任务 4.5、5.2、5.4–5.6：批处理与生命周期证据

实现位置：

- `vllm-upstream/vllm/entrypoints/pooling/systemone/renderer.py`
- `vllm-upstream/vllm/entrypoints/pooling/systemone/serving.py`
- `vllm-upstream/vllm/model_executor/layers/pooler/kev.py`

## 输入与 batch

请求在任何子任务进入调度器前完成全部问题的 token 化和限额检查。state、单分支、问题数、选项数和逻辑 token 总量均按实际 token IDs 校验；不执行截断。测试覆盖最后一题超限时 backend 调用数为零。

vLLM 运行时将不同长度序列表示为扁平 token batch 和 cursor offset，不向 pooler 传入右侧 padding。ragged batch 测试覆盖 K=1/2/8/255 和不同序列长度；重排后每题结果保持一致，每题分别归一化，不执行跨题全局 softmax。

## 调度与清理

- 已接纳父请求、全局子请求和单父在途子请求均有硬上限。
- 中央调度器按父请求轮转取任务；一个父请求不能超过自己的 fan-out。
- 内部父子 request ID 与用户 question ID 分离。
- Future 按输入位置聚合，因此子任务乱序完成仍保持原问题顺序。
- 超时包含队列等待；超时、断连或任一分支失败都会移除未提交任务并调用 backend abort。
- 失败后父请求配额和调度状态被释放，后续请求可以成功执行。

执行结果：

```text
SystemOne protocol/service tests: 26 passed
KEV pointer/ragged batch tests: 10 passed
ruff check: passed
ruff format --check: 10 files already formatted
```
