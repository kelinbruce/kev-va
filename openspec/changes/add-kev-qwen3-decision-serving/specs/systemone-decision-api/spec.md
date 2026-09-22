## Purpose

提供面向软件调用的 SystemOne 决策接口，使调用方能够以共享 state 和命名问题获得三类结构化答案，并在无效输入、资源不足、超时或取消时得到确定行为和完整的请求清理。

## ADDED Requirements

### Requirement: Typed SystemOne request

专用服务 SHALL 提供 `POST /v1/systemone`，接受 state、可选 model 和非空 questions 映射。state、instructions 与描述值 SHALL 支持固定 KEV 的 JSONContent（字符串、对象、数组、有限数值、布尔值、null）。每个问题 SHALL 包含 type 和 instructions；type 仅限 noul、choice、score。

choice SHALL 提供 1..255 个有序命名候选项；score SHALL 提供 2..255 个有序等级描述；noul SHALL 接受缺省 criteria 或只含 true/false 描述的对象。model 缺省值为 `kev-latest`，映射到当前已加载的唯一模型。

#### Scenario: Mixed question types

- **WHEN** 一次请求包含合法的 noul、choice 和 score
- **THEN** 返回 HTTP 200，answers 按原 question ID 和原问题顺序包含三项对应类型结果

#### Scenario: Invalid type or generation parameters

- **WHEN** 请求含未知字段、stream/max_tokens 等生成参数、非有限数值、非法 criteria 或空 questions
- **THEN** 返回 HTTP 422 和结构化错误，且该请求没有任何子问题进入推理

### Requirement: Deterministic answer mapping

服务 SHALL 使用原始概率完成决策，再按照固定 KEV 规则保留两位小数。noul SHALL 返回 yes 概率；choice SHALL 返回胜出选项、所有命名概率及 confidence；score SHALL 返回等级索引的概率加权平均、legend、分布与 confidence。并列最大概率 SHALL 按输入选项顺序选择。

choice confidence SHALL 在 K>1 时为 `(max(p)-1/K)/(1-1/K)`，K=1 时为 1；score confidence SHALL 为 `1-sum(p_i*abs(i-mode))/(K-1)`。对外舍入后的概率不要求精确求和为 1。该 confidence SHALL 不被描述为经验证的正确率。

#### Scenario: Noul and ordinal score

- **WHEN** noul 的 yes 概率为 0.95，score 的原始分布为 [0.1,0.7,0.2]
- **THEN** noul 返回 0.95；score 返回 1.1、等级 0/1/2 的 legend 和概率，以及 confidence 0.85

#### Scenario: Rounded probabilities tie

- **WHEN** 原始最高概率对应选项 B，但 A 和 B 在两位小数展示上相同
- **THEN** choice 仍返回 B，不根据舍入后的概率重新选取答案

### Requirement: Explicit input limits without truncation

服务 SHALL 在提交任何子请求前，按启动配置校验问题数、选项数、state token 数、单分支总长度和请求逻辑 token 总量。默认问题数上限 32、state 上限 8192、分支总长上限 8192、逻辑总量上限 32768。所有长度 SHALL 按实际 renderer/tokenizer 结果计算，结构标记计入长度。

#### Scenario: Last question exceeds the limit

- **WHEN** 一个多问题请求只有最后一个问题超过分支长度或选项数限制
- **THEN** 整个请求返回 HTTP 422，所有问题均不进入推理，输入不被静默截断

#### Scenario: Unknown model

- **WHEN** model 不是已加载模型 ID 或其显式别名
- **THEN** 返回 HTTP 404 和结构化错误，不回退到任意默认模型

### Requirement: Bounded aggregation and atomic response

服务 SHALL 对已接纳父请求、全局子请求和每个父请求的在途子请求设置并发上限，默认分别为 32、16、4。所有问题完成前 SHALL 不返回部分成功响应。到达父请求容量时 SHALL 返回 HTTP 429；尚未就绪时 SHALL 返回 HTTP 503。

#### Scenario: Out-of-order completion

- **WHEN** 一个请求的子问题以不同于输入的顺序完成
- **THEN** 系统按原问题 ID 聚合，输出顺序保持输入顺序，不混入其他请求的答案

#### Scenario: Saturated service

- **WHEN** 已接纳父请求达到配置上限且又收到一个请求
- **THEN** 新请求返回 HTTP 429，且不向引擎提交其子问题

### Requirement: Timeout cancellation and failure cleanup

服务 SHALL 使用包含排队时间的请求 deadline，默认 30 秒。超时 SHALL 返回 HTTP 504；子问题执行失败 SHALL 返回 HTTP 500；所有错误 SHALL 使用包含 message、type、code、param 的 error 对象。断连、超时或子问题失败 SHALL 取消剩余工作并释放请求关联资源。

#### Scenario: One branch fails

- **WHEN** 多问题请求中的一个子问题推理失败
- **THEN** 整体返回结构化错误，不返回部分 answers，并取消其他未完成子问题

#### Scenario: Client disconnects

- **WHEN** 客户端在部分问题排队或执行时断开
- **THEN** 不再提交剩余问题，终止已提交的未完成请求，并释放映射与并发配额

### Requirement: Explicit model identity and usage

响应 SHALL 包含实际加载的规范 model ID、answers、usage.input_tokens、usage.output_tokens 和 latency_ms。输入 usage SHALL 按 state 一次加全部问题后缀计算；output_tokens SHALL 按固定 KEV 的 answers 序列化/tokenizer 规则计算；latency_ms SHALL 包含服务端排队和执行时间。服务 SHALL 提供模型查询与现有健康检查能力。

#### Scenario: Shared state with repeated computation

- **WHEN** 两个问题分别执行同一个 S-token state，后缀长度为 L1 和 L2
- **THEN** API input_tokens 为 S+L1+L2，且不会把 output_tokens 描述为神经生成 token 数

### Requirement: Decision-only instance scope

KEV 专用实例 SHALL 不提供文本生成、聊天或流式输出能力；该限制 SHALL 仅作用于 KEV 实例，不改变其他模型实例已有路由行为。

#### Scenario: Generation request to a decision instance

- **WHEN** 客户端向 KEV 实例调用 chat/completions 或 completions
- **THEN** 请求以 HTTP 404 或明确的 HTTP 400 unsupported-operation 失败，且不执行推理
