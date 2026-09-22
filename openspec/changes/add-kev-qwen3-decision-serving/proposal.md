## Why

在 Ascend 910 系列上提供基于 KEV 的专用决策推理服务，使业务通过一次请求获得多个问题的结构化概率结果。第一版以 Qwen3 权重的计算语义和真实权重数值一致性为目标，为后续共享 state 与吞吐优化建立基线。

## What Changes

- 增加 SystemOne 风格的决策接口，支持同一请求中的 `noul`、`choice` 和 `score`。
- 接入 Qwen3 backbone、已训练的 KEV pointer head 和校准温度，直接读取概率，不执行文本解码。
- 按问题拆分独立因果序列，支持不同问题数、选项数和序列长度的批处理，并按原问题 ID 聚合结果。
- 定义可复现的模型导出制品，包含基座、adapter、head、tokenizer 和源代码的固定版本与校验信息。
- 增加输入限额、请求取消与超时、失败清理、逻辑/实际 token 计量和真实权重验收要求。
- 建立 CANN 9.0.1 与 vLLM-Ascend 0.26 的环境验证门槛，未经验证不宣称该组合受支持。

### 已确认约束与建议默认值

| 类别 | 内容 |
| --- | --- |
| 用户已确认 | vLLM-Ascend 0.26 系列；Ascend 910 系列；CANN 9.0.1；第一版使用 Qwen3；服务只需支持 JEV 类决策推理 |
| 本提案建议 | 首个验收模型为 `jaredpalmer/kev-4b` 的 `qwen3` revision，底座为 `Qwen/Qwen3-4B-Base` |
| 本提案建议 | 单卡 TP=1，优先 BF16 backbone、FP32 pointer head；eager；关闭 APC、chunked prefill 和量化 |
| 待实施时记录 | 具体 910 型号、显存、镜像 digest、vLLM/Ascend commit、驱动、CANN、torch-npu、模型本地路径 |

“最新 KEV”指采用核对过的当前代码并固定 commit；Qwen3 权重必须显式指定 revision。默认 `main` 权重已转为 Qwen3.5，不能直接用于本提案。

### 第一版边界

第一版允许每条问题序列重复计算 state；不承诺一次请求内 state 只计算一次。共享前缀、分支 Attention kernel、chunked prefill、ACLGraph 性能优化、量化、多卡 TP 和 Qwen3.5 均作为后续变更；不在本次任务中实现。

## Capabilities

### New Capabilities

- `kev-model-artifact`: 固定来源、导出和校验 Qwen3 KEV 推理制品，保留训练后的 adapter、head 与校准信息。
- `kev-decision-inference`: 保持输入编排、问题隔离、选项交互和概率计算语义，并通过真实权重一致性验证。
- `systemone-decision-api`: 定义三类问题的请求/响应、限额、错误、并发聚合和取消行为。
- `kev-ascend-runtime`: 约束环境诊断、支持配置、就绪状态、计量和验收证据。

### Modified Capabilities

无。仓库此前没有 OpenSpec 主规范；本变更引入四项新能力。现有非 KEV 模型的接口和行为不变。

## Impact

- 通用 API、模型、Pooler 和加载逻辑需要在匹配版本的上游 vLLM 中实现；当前 Ascend 仓库承载必要后端补丁、验证脚本及文档。
- KEV 专用实例新增 `/v1/systemone`，保留模型查询、健康检查和 metrics；不提供文本生成服务。
- 导出流程可能需要隔离的 PEFT/Transformers 工具环境；运行容器保持版本配套，不为导出升级运行时 Transformers。
- 官方 0.26 配套 CANN 为 9.1.0；用户的 9.0.1 保持为当前环境约束。先进行基线验证，失败时记录阻塞原因；本提案不授权自动升级 CANN 或降级 vLLM-Ascend。
- 本次交付为可评审的 OpenSpec 文档，不代表推理实现、环境兼容性或性能已经验证。
