# KEV Qwen3 决策服务规范

当前变更：[add-kev-qwen3-decision-serving](changes/add-kev-qwen3-decision-serving/proposal.md)。

这是一份待评审的首版实现提案。用户已确认 Qwen3、vLLM-Ascend 0.26、Ascend 910 系列和 CANN 9.0.1；具体模型大小、容量默认值与数值阈值是本提案的设计建议。CANN 组合尚需真实环境验证。

阅读顺序：

1. [proposal.md](changes/add-kev-qwen3-decision-serving/proposal.md)：目标、范围及能力划分。
2. [design.md](changes/add-kev-qwen3-decision-serving/design.md)：架构、API 样例、限额、版本及验收标准。
3. [模型制品](changes/add-kev-qwen3-decision-serving/specs/kev-model-artifact/spec.md)、[决策推理](changes/add-kev-qwen3-decision-serving/specs/kev-decision-inference/spec.md)、[SystemOne API](changes/add-kev-qwen3-decision-serving/specs/systemone-decision-api/spec.md)、[Ascend 运行](changes/add-kev-qwen3-decision-serving/specs/kev-ascend-runtime/spec.md)：可验证的行为约定。
4. [tasks.md](changes/add-kev-qwen3-decision-serving/tasks.md)：按依赖排序的实施清单。

在仓库根目录检查：

```bash
OPENSPEC_TELEMETRY=0 openspec status --change add-kev-qwen3-decision-serving
OPENSPEC_TELEMETRY=0 openspec validate add-kev-qwen3-decision-serving --strict --no-interactive
```

主规范目录 openspec/specs 当前为空；变更实施并完成验收后再归档合入，避免把计划中的能力描述成已存在。
