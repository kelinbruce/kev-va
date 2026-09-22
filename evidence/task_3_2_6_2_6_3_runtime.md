# 任务 3.2、6.2、6.3：进程边界、就绪门禁与指标

## 3.2 plugin pooling 进程边界

`test_plugin_metadata_survives_engine_process_boundary_for_ragged_outputs` 使用
vLLM 多进程 EngineCore 实际采用的 `MsgpackEncoder` 和
`MsgpackDecoder(EngineCoreRequest)`，而不是自定义序列化替身。测试确认：

- `PoolingParams(task="plugin")` 的 `extra_kwargs` 在编码/解码后保持不变；
- `parent-a/0`、`parent-b/3` request ID 保持对应；
- 解码后的 metadata 可直接驱动 `KevPooler`；
- 同一 ragged batch 中 K=2、K=3 分别返回正确长度的归一化概率。

```text
$ pytest -q tests/model_executor/layers/test_kev_pooler.py::test_plugin_metadata_survives_engine_process_boundary_for_ragged_outputs
.                                                                        [100%]
1 passed, 1 warning in 0.09s
```

本地 macOS Torch 2.14 对既有 `PIN_MEMORY=True` CPU cursor 路径会产生原生段
错误，因此该定向测试仅在测试内将 metadata 模块的 `PIN_MEMORY` 改为 false；
目标 Linux/Ascend 路径未作代码修改。

## 6.2 就绪门禁

- `validate_runtime_artifact` 在启用 endpoint 前要求本地制品目录，验证格式、
  Qwen3 KEV 架构、option-isolation、正温度、不可变来源 revision、原始来源
  SHA、全部导出文件 SHA、权重 index、pointer head 条目、已加载 config 和结构
  token ID。
- Engine client 已完成模型加载后才初始化服务；制品验证通过后再执行一条有效
  `noul` 决策 warm-up。只有完整概率后处理成功才设置 `ready=True`。
- 校验或 warm-up 失败时 HTTP 服务对象仍存在，但 `/v1/systemone` 返回 503；
  readiness error 被保留用于诊断，warm-up 请求会调用 engine abort。

## 6.3 Prometheus metrics

新增以下指标，均带稳定的 model/kind 或 code 标签：

- `vllm:systemone_input_tokens_total`：logical/actual；
- `vllm:systemone_success_total`：request/question；
- `vllm:systemone_queue_duration_seconds`：分支排队时间；
- `vllm:systemone_request_duration_seconds`：包含排队的端到端时间；
- `vllm:systemone_in_flight`：request/branch；
- `vllm:systemone_errors_total`：API 错误、超时和取消 code。

成功计数和 token 计数只在全请求成功聚合后增加。测试确认 deadline timeout 和
client disconnect 均执行 abort、在途量回零，并且不增加成功请求或成功问题数。

## 检查结果

项目环境缺少可选测试依赖 `llguidance`、`xgrammar`；服务层定向测试仅提供不参与
被测路径的 import stub，没有下载依赖或模型。

```text
$ ruff check <changed SystemOne files>
All checks passed!

$ ruff format --check <changed SystemOne files>
17 files already formatted

$ pytest -q tests/entrypoints/pooling/systemone/test_api_router.py \
    tests/entrypoints/pooling/systemone/test_backend.py \
    tests/entrypoints/pooling/systemone/test_factory.py \
    tests/entrypoints/pooling/systemone/test_serving.py
.....................                                                    [100%]
21 passed, 1 warning in 1.54s
```

任务 3.4 和所有 Ascend/真实权重验收保持未完成；上述结果不代表 NPU 路径通过。
