# 任务 8.5：最终验证与 spec/evidence 审计

## 回归结果

```text
$ ./.venv/bin/python -m pytest -q \
    tests/test_kev_benchmark.py tests/test_kev_artifact.py
............................                                             [100%]
28 passed in 1.36s

$ ./.venv/bin/ruff check tools tests
All checks passed!

$ ./.venv/bin/ruff format --check tools tests
5 files already formatted

$ OPENSPEC_TELEMETRY=0 openspec validate \
    add-kev-qwen3-decision-serving --strict --no-interactive
Change 'add-kev-qwen3-decision-serving' is valid

$ git diff --check
# exit 0
```

上游 vLLM 在交付 commit `dc7f73910b48c0d7ba2446bfcc3a1168fe716005`
执行的变更相关回归：SystemOne API/backend/factory/service 21 passed；进程边界 ragged
pooling 定向测试 1 passed；17 个变更文件 ruff check/format 通过。完整命令和 macOS
Torch pin-memory 说明见 `task_3_2_6_2_6_3_runtime.md`。

vLLM-Ascend 文档在交付 commit
`0553039e12b671e37de5fbce414dd6a4503a3405` 完成 codespell、typos、文件名、
suggestion 和手动 markdownlint。完整默认 pre-commit 中，gitleaks 因本机缺少 `wget`、
check-symbolic-meta 因 hook 环境缺少 `python` 而阻塞；文档内容检查均通过，详情见
`task_8_3_tutorial.md`。

## Spec 到证据逐项核对

表中“待真实环境”场景已经有固定配置/入口，但没有被错误标成验收通过。

| Spec requirement | 场景与证据 | 状态/剩余工作 |
|---|---|---|
| artifact / Immutable provenance | Resolve checkpoint：`task_1_3_source_report.md`；missing base revision：immutable revision/tooling 拒绝测试 | 来源固定通过；实际制品仍由 2.2 验证 |
| artifact / Preserve trained parameters | Export trained checkpoint、non-default calibration：synthetic FP32 merge、head bias/dtype/temperature 测试 | 工具通过；真实 checkpoint 导出和合并前后误差待 2.2/2.4 |
| artifact / Reject corrupt artifacts | Corrupt/incomplete weights、wrong architecture：checksum、inventory、missing head、Qwen3.5 拒绝测试 | 通过 |
| artifact / Offline reproducibility | Deploy offline：工具无网络/remote code，synthetic 完整制品重复校验 | synthetic 通过；真实完整制品离线加载待 2.5 |
| inference / Reference input | Structured/control-like text、rename question：renderer golden 与 question-ID mapping-only 测试 | 通过 |
| inference / Isolation/interaction | Sibling isolation、mixed batch、option ordering：独立完整序列、ragged batch/重排测试 | CPU 语义通过；真实权重排列敏感性待 7.4 |
| inference / Calibrated distribution | Different K、multiple readout positions：K=1/2/8/255、温度、bias、逐题 softmax | 通过 |
| inference / Real-weight parity | NPU/reference compare、business thresholds：配置固定误差、argmax、clear-winner、0.5/0.8/0.9、Brier/ECE | 待真实环境 7.3/7.5 |
| inference / Batch invariance | Rebatch unchanged question：synthetic 重排/混排测试和真实阈值配置 | CPU 通过；真实权重 <=0.005 待 7.4 |
| API / Typed request | Mixed types、invalid/generation parameters：protocol/router 与教程请求 | 通过 |
| API / Deterministic mapping | Noul/score、rounded tie：固定 KEV golden、raw-before-rounding、stable tie 测试 | 通过 |
| API / Explicit limits | Last question exceeds、unknown model：全请求预检、无提交、结构化错误测试 | 通过 |
| API / Bounded atomic aggregation | Out-of-order、saturated service：父子 ID、轮转、容量、全成功响应测试 | 通过 |
| API / Cleanup | Branch failure、disconnect：siblings abort、timeout/disconnect、恢复与 metrics 测试 | CPU 通过；NPU 执行中恢复待 7.6 |
| API / Identity and usage | Shared repeated state：规范 model ID、logical/actual usage 和序列化 output usage 测试 | 通过 |
| API / Decision-only scope | Generation request：专用架构注册和 generation 路由隔离测试 | 通过 |
| runtime / Environment report | Matrix mismatch、baseline failure：`task_8_4_status_matrix.md` 保持 CANN 9.0.1 待验证 | 实际容器诊断和普通 Qwen3 基线待 1.1/1.2/6.4 |
| runtime / Initial profile | Unsupported flag、whole-sequence budget：TP/dtype/eager/APC/chunk/graph/quant/LoRA 和 budget 测试 | 通过 |
| runtime / Executable readiness | False-ready：artifact/config/tokenizer 校验和有效 warm-up 失败测试 | 通过 |
| runtime / Logical vs executed work | Successful repeated state、abort before all branches：logical/actual metrics 与取消不计成功吞吐测试 | CPU 通过；真实计量由 benchmark `/metrics` 入口复核 |
| runtime / Mandatory real weights | Dummy-only success、completed baseline report：状态矩阵明确 dummy 不替代真实验收，配置冻结报告字段 | 待 7.1--7.7，不因本任务勾选而跳过 |

## 环境阻塞与边界

- 当前 macOS 开发环境没有 Ascend NPU、CANN、torch-npu 或已安装的 vLLM-Ascend，
  因此 1.1、1.2、1.5、3.4、6.4 和 7.1--7.7 保持未完成。
- 按用户要求未下载任何模型或权重，因此 2.2、2.4、2.5 和真实权重验收保持未完成。
- 上游完整 pytest 收集受可选 `llguidance`/`xgrammar` 缺失影响；变更相关服务测试用
  不参与被测路径的 import stub 执行，并单独记录该限制。
